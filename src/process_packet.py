"""
process_packet.py -- turn one acquired packet into training data.

Input: a packet manifest written by ``take_packet.py`` (``<work_dir>/packets/<id>.json``)
and the raw files it points to.  Output, in ``<output dir>/<packet_id>/``:

    traces/<kind><n>_acm<board>_ch<ch>.npz  every decoded sample of every trace (lossless):
                                          ``values`` (ADU), ``flags`` (12 flag bits), the inline
                                          L1 records, and the per-pixel CDS matrix; <kind> is
                                          "ref" or "theta", <n> the trace set (1, 2, 3)
    features.json                         per trace: quality check, timing, and the trace tools
                                          (median and spread over the usable pixels); per
                                          board and channel: the tools averaged over the
                                          repeated trace sets, their scatter between sets, and
                                          trace minus reference-trace differences
    fits/avg_<image>.fits                 the image, decompressed from .fits.bz2
    waders/acm<board>/                    panaSKImg output, its log, the exact command and JSON
    labels.json                           per board and channel: gain, sigma_e, lambda with
                                          errors, fit-quality flags and the WADERs version
    result.json                           summary of everything above (used for the dataset)

and, for all processed packets together, ``<output dir>/dataset.csv`` and ``dataset.jsonl``:
one row per (packet, board, channel) with that board's theta, its change from the
reference, the trace features (mean over the repeated sets and their scatter), the
reference-trace features, their difference and the image labels.

Steps (each is a function below; the walkthrough notebook runs them one by one)
-----
    1. read the manifest and find the files (DAQ paths are mapped to local copies)
    2. traces:  decode -> quality check -> regions -> tools -> .npz + features; then the
                repeated trace sets are combined per (kind, board, channel)
    3. images:  decompress .fits.bz2 -> run panaSKImg -> read MEGAIN/MESIG/MEDC from the header
    3b. drift:  the along-row baseline drift of each image (image_drift.py): if present, the first 200 columns are
                cut and the slow remainder subtracted, and WADERs runs on that copy (labels record it)
    4. labels:  sigma_e = MESIG/MEGAIN, lambda = MEDC/MEGAIN, flags for fits stuck at a bound;
                lambda row by row and its split into dark current and CIC (image_rows.py)
    5. group:   rebuild dataset.csv / dataset.jsonl from every result.json
    6. model inputs: one ready-to-train file per model family in model_inputs/ (ml/real_inputs.py)

Packets taken by hand (daq_scripts/manual_packet.sh) get their manifest from make_manual_manifest.py.

Usage
-----
    python process_packet.py <manifest.json> [<manifest.json> ...] [--config ../configs/processing.toml]
    python process_packet.py --packets-dir <work_dir>/packets          # every packet with status ok
    options: --skip-waders (traces only), --out DIR, --path-map OLD=NEW
"""

from __future__ import annotations

import argparse
import bz2
import csv
import json
import math
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from trace_features import trace_features
from decode_trace import decode_trace
from trace_regions import split_regions

CONFIG_DEFAULT = Path(__file__).resolve().parent.parent / "configs" / "processing.toml"

#: Fit bounds used by WADERs' FitDarkCurrentProcess when the JSON does not set them
#: (pysimdamicm/processes/skipper_analysis.py, FitDarkCurrentProcess.__init__).
WADERS_DEFAULT_BOUNDS = {"mu": (-1.0, 1.0), "sigma": (0.001, 100.0), "dc": (0.001, 10.0),
                         "gain": (1.0, 100.0)}


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class ProcessingConfig:
    """Settings of processing (paths, WADERs command and timeout, what to save); read from a TOML file by ``load``."""
    out_dir: Path
    path_map: dict[str, str]
    save_raw: bool = True
    waders_command: list[str] = field(default_factory=lambda: ["panaSKImg"])
    waders_json: Path | None = None
    waders_extra_args: list[str] = field(default_factory=lambda: ["--acm"])
    waders_timeout_s: float = 1800
    pysimdamicm_dir: Path | None = None
    image_overrides: dict[str, Any] = field(default_factory=dict)
    waders_runner: str = "panaSKImg"       # "panaSKImg" (one JSON for every image) or "run_WADERs" (comm_scripts)
    run_waders_command: list[str] = field(default_factory=lambda: ["run_WADERs"])
    run_waders_args: list[str] = field(default_factory=lambda: ["--no-clusters"])
    drift_enabled: bool = True             # along-row baseline drift: detect, cut, correct before WADERs (image_drift.py)
    drift_cut: str = "200"                 # columns cut where a drift is found ("auto": per image, diagnostic only)
    drift_channels: list[int] = field(default_factory=lambda: [0, 1, 3])   # channels that decide (the good ones of board 101)
    drift_width_tol: float = 0.05
    rows_enabled: bool = True              # per-row lambda and the dark-current / CIC split (image_rows.py)
    rows_reverse: bool = False             # True if FITS row 0 is the LAST row read out
    model_inputs: bool = True              # also write the per-model input files (ml/real_inputs.py)

    @classmethod
    def load(cls, path: str | Path) -> "ProcessingConfig":
        """Read the TOML configuration; relative paths are relative to the configuration file."""
        path = Path(path).resolve()
        with open(path, "rb") as handle:
            data = tomllib.load(handle)

        def resolve(value):                     # relative paths: relative to the config file
            p = Path(value).expanduser()
            return p if p.is_absolute() else (path.parent / p).resolve()

        waders = data.get("waders", {})
        return cls(
            out_dir=resolve(data.get("output", {}).get("dir", "processed")),
            path_map={k: str(resolve(v)) for k, v in data.get("paths", {}).items()},
            save_raw=bool(data.get("traces", {}).get("save_raw", True)),
            waders_command=list(waders.get("command", ["panaSKImg"])),
            waders_json=resolve(waders["json"]) if waders.get("json") else None,
            waders_extra_args=list(waders.get("extra_args", ["--acm"])),
            waders_timeout_s=float(waders.get("timeout_s", 1800)),
            pysimdamicm_dir=resolve(waders["pysimdamicm_dir"]) if waders.get("pysimdamicm_dir") else None,
            image_overrides=dict(waders.get("image_overrides", {})),
            waders_runner=str(waders.get("runner", "panaSKImg")),
            run_waders_command=[str(resolve(c)) if "/" in str(c) else str(c) for c in waders.get("run_waders_command", ["run_WADERs"])],
            run_waders_args=list(waders.get("run_waders_args", ["--no-clusters"])),
            drift_enabled=bool(data.get("drift", {}).get("enabled", True)),
            drift_cut=str(data.get("drift", {}).get("cut", "200")),
            drift_channels=[int(c) for c in data.get("drift", {}).get("channels", [0, 1, 3])],
            drift_width_tol=float(data.get("drift", {}).get("width_tol", 0.05)),
            rows_enabled=bool(data.get("rows", {}).get("enabled", True)),
            rows_reverse=bool(data.get("rows", {}).get("reverse", False)),
            model_inputs=bool(data.get("model_inputs", {}).get("enabled", True)))


def localize(path: str | Path, path_map: dict[str, str]) -> Path:
    """Map a path recorded on the DAQ machine to this computer (unchanged if it already exists)."""
    path = Path(path)
    if path.exists():
        return path
    text = str(path)
    for daq_prefix, local_prefix in path_map.items():
        if text.startswith(daq_prefix):
            return Path(local_prefix + text[len(daq_prefix):])
    return path


def _json_safe(value):
    """numpy scalars/arrays and NaN -> plain JSON values."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


# --------------------------------------------------------------------------- #
# Step 1: the files of a packet                                                #
# --------------------------------------------------------------------------- #

def packet_files(manifest: dict, cfg: ProcessingConfig) -> dict[str, list[dict]]:
    """
    List the trace and image files of a packet, with paths mapped to this computer.

    Traces: ``{kind, board, channel, repeat, data, meta}``; the data file of a trace of channel c is
    ``<stem>.RJ45ADCDATA<c>`` (the other three channels of that acquisition are empty frames).
    Images: ``{board, meta, fits}`` where ``fits`` is the image builder's ``avg_<name>.fits*``.
    """
    traces = []
    for kind in ("reference_traces", "traces"):
        for t in manifest.get(kind, []):
            stem = localize(t["stem"], cfg.path_map)
            traces.append({"kind": "ref" if kind == "reference_traces" else "theta",
                           "board": int(t["board"]), "channel": int(t["channel"]),
                           "repeat": int(t.get("repeat", 1)),
                           "data": stem.parent / f"{stem.name}.RJ45ADCDATA{t['channel']}",
                           "meta": stem.with_suffix(".meta")})
    images = []
    for im in manifest.get("images", []):
        stem = localize(im["stem"], cfg.path_map)
        fits = [localize(f, cfg.path_map) for f in im.get("fits", [])]
        fits = [f for f in fits if Path(f).name.startswith(("avg_", "cds_avg_"))]
        if not fits:                                       # not recorded: look next to the image
            fits = sorted(stem.parent.glob(f"avg_{stem.name}.fits*")) + \
                   sorted(stem.parent.glob(f"cds_avg_{stem.name}*"))
        images.append({"board": int(im["board"]), "meta": stem.with_suffix(".meta"),
                       "fits": Path(fits[0]) if fits else None})
    return {"traces": traces, "images": images}


# --------------------------------------------------------------------------- #
# Step 2: traces                                                               #
# --------------------------------------------------------------------------- #

def _ragged_to_matrix(rows: list[np.ndarray]) -> np.ndarray:
    """Stack arrays of possibly different lengths into a NaN-padded float matrix."""
    width = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), width), np.nan)
    for i, row in enumerate(rows):
        out[i, :len(row)] = row
    return out


def summarize_tools(tools: dict[str, np.ndarray]) -> tuple[dict[str, float], dict[str, float], dict[str, np.ndarray]]:
    """
    Reduce the per-pixel tool values of one trace to one number per tool.

    Returns (median over the replicates, robust spread = 1.4826 x MAD, curves).  Curves (tools
    with one value per point, e.g. the mean skip shape) are kept as their median curve.
    """
    median, spread, curves = {}, {}, {}
    for name, values in tools.items():
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            med = float(np.nanmedian(values))
            median[name] = med
            spread[name] = float(1.4826 * np.nanmedian(np.abs(values - med))) if len(values) > 1 else float("nan")
        else:
            curves[name] = np.nanmedian(values, axis=0)
    return median, spread, curves


def process_trace(entry: dict, out_dir: Path, cfg: ProcessingConfig) -> dict:
    """
    Decode one trace, check it, compute every feature, and save the arrays.

    The features come from ``trace_features.trace_features``: the 39 comparison tools of
    ``compare_trace.py``, the advanced features and the 2026-10-21 additions: 141 numbers and 18 curves, each documented in
    ``trace_features.FEATURES`` and ``docs/overleaf_trace_features/trace_features.tex``.  They use the usable pixels (no
    row-start or charged pixels).  Returns a summary dict (also stored in features.json).  A trace
    with quality problems, or whose features cannot be computed, is decoded and saved but marked ``usable = False``;
    no features.
    """
    name = f"{entry['kind']}{entry['repeat']}_acm{entry['board']}_ch{entry['channel']}"
    summary = {"name": name, "kind": entry["kind"], "board": entry["board"],
               "channel": entry["channel"], "repeat": entry["repeat"], "file": str(entry["data"])}
    if not entry["data"].exists():
        summary.update(usable=False, problems=["file not found"])
        return summary

    trace = decode_trace(entry["data"])
    problems = trace.quality_problems()
    summary.update(n_samples=int(len(trace.values)), problems=problems, usable=not problems)
    meta = json.loads(entry["meta"].read_text()) if entry["meta"].exists() else {}
    summary["amplifier"] = meta.get(f"ch{entry['channel']}")
    summary["fake"] = bool(meta.get("FAKE", False))

    arrays: dict[str, np.ndarray] = {}
    if cfg.save_raw:
        arrays.update(values=trace.values, flags=trace.flags, l1_pixel=trace.l1_pixel,
                      l1_is_signal=trace.l1_is_signal, l1_n=trace.l1_n, l1_sum=trace.l1_sum)

    if not problems:
        try:
            regions = split_regions(trace, meta)
            features = trace_features(trace, meta=meta, regions=regions)
        except Exception as err:            # noqa: BLE001 - a damaged trace that passed the checks: unusable, not fatal
            problems = [f"features failed: {type(err).__name__}: {err}"]
            summary.update(problems=problems, usable=False)
    if not problems:
        summary.update(pixels_used=features.info["pixels_used"],
                       row_start_pixels=features.info["row_start_pixels"],
                       charged_pixels=features.info["charged_pixels"],
                       timing=features.info["timing"], tools=features.scalars, tools_spread=features.spread)
        arrays["cds"] = _ragged_to_matrix([np.asarray(regions.cds[p], float) for p in range(regions.n_pixels)])
        for curve_name, curve in features.curves.items():
            arrays["curve: " + curve_name] = curve

    path = out_dir / "traces" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    summary["npz"] = str(path)
    return summary


#: With at least this many usable repeats, values far from the others are rejected per feature.
MIN_REPEATS_FOR_REJECTION = 5
#: A value is rejected if it is more than this many robust standard deviations from the median.
REJECT_ROBUST_SIGMA = 5.0


def _robust_keep(v: np.ndarray) -> np.ndarray:
    """
    Which repeats of one feature to keep (boolean mask).

    A damaged trace is already refused by the quality check (lost data).  A trace can still be
    unusual without lost data (a cosmic ray, a glitch, a pixel with charge), so with
    >= MIN_REPEATS_FOR_REJECTION finite values a value is dropped if
    |v - median| > REJECT_ROBUST_SIGMA * 1.4826 * MAD.  With fewer repeats nothing is dropped:
    the median and MAD of 3 values are too uncertain.
    """
    finite = np.isfinite(v)
    if finite.sum() < MIN_REPEATS_FOR_REJECTION:
        return finite
    med = np.median(v[finite])
    robust_sd = 1.4826 * np.median(np.abs(v[finite] - med))
    if robust_sd == 0:
        return finite
    return finite & (np.abs(v - med) <= REJECT_ROBUST_SIGMA * robust_sd)


def combine_repeats(traces: list[dict]) -> list[dict]:
    """
    Combine the repeated trace sets of every (kind, board, channel).

    For each tool: ``mean`` = average of the per-trace values over the usable repeats, and
    ``sd`` = their standard deviation (ddof = 1): the scatter between trace sets taken a minute
    apart at the same settings, i.e. the run-to-run noise of that tool.  With 5 or more repeats
    (the recommended packet has 10), values more than 5 robust standard deviations from the
    median are left out first (``_robust_keep``); ``n_rejected`` counts them per tool.
    """
    groups: dict[tuple, list[dict]] = {}
    for t in traces:
        groups.setdefault((t["kind"], t["board"], t["channel"]), []).append(t)
    combined = []
    for (kind, board, channel), members in sorted(groups.items()):
        usable = [t for t in members if t.get("usable")]
        entry = {"kind": kind, "board": board, "channel": channel,
                 "n_repeats": len(members), "n_usable": len(usable),
                 "amplifier": next((t.get("amplifier") for t in members if t.get("amplifier")), None)}
        if usable:
            names = [k for k in usable[0]["tools"] if all(k in t["tools"] for t in usable)]
            values = {k: np.array([t["tools"][k] for t in usable], dtype=float) for k in names}
            # NaN where a feature does not exist in a trace (e.g. charge loss without a charged pixel)
            keep = {k: _robust_keep(v) for k, v in values.items()}
            entry["mean"] = {k: float(np.mean(v[keep[k]])) if keep[k].any() else float("nan")
                             for k, v in values.items()}
            entry["sd"] = {k: float(np.std(v[keep[k]], ddof=1)) if keep[k].sum() > 1 else float("nan")
                           for k, v in values.items()}
            entry["n_rejected"] = {k: int(np.isfinite(v).sum() - keep[k].sum()) for k, v in values.items()
                                   if np.isfinite(v).sum() > keep[k].sum()}
        combined.append(entry)
    return combined


def trace_differences(combined: list[dict]) -> list[dict]:
    """For every (board, channel) traced at theta and at the reference: mean(theta) - mean(reference)."""
    by_key = {(c["kind"], c["board"], c["channel"]): c for c in combined}
    out = []
    for (kind, board, channel), c in by_key.items():
        ref = by_key.get(("ref", board, channel))
        if kind != "theta" or ref is None or "mean" not in c or "mean" not in ref:
            continue
        out.append({"board": board, "channel": channel,
                    "difference": {k: c["mean"][k] - ref["mean"][k] for k in c["mean"] if k in ref["mean"]}})
    return out


# --------------------------------------------------------------------------- #
# Step 3: images and WADERs                                                    #
# --------------------------------------------------------------------------- #

def decompress_fits(src: Path, dst_dir: Path) -> Path:
    """Write the decompressed copy of ``*.fits.bz2`` into ``dst_dir`` (the original is untouched)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.suffix != ".bz2":
        dst = dst_dir / src.name
        shutil.copyfile(src, dst)
        return dst
    dst = dst_dir / src.name[:-len(".bz2")]
    with bz2.open(src, "rb") as compressed, open(dst, "wb") as plain:
        shutil.copyfileobj(compressed, plain)
    return dst


def read_fits_primary(path: Path) -> dict[str, Any]:
    """
    Read the primary header of a FITS file (plain or .bz2) without astropy.

    A FITS header is a sequence of 80-character "cards" in 2880-byte blocks ending with END.
    Numbers, quoted strings and logicals (T/F) are converted; comments are dropped.  WADERs
    writes every MEGAIN/MESIG/MEDC key into the primary header, the ACM image builder writes
    the acquisition metadata (NROW, NDCM, ...) there.
    """
    opener = bz2.open if str(path).endswith(".bz2") else open
    header: dict[str, Any] = {}
    with opener(path, "rb") as handle:
        while True:
            block = handle.read(2880)
            if len(block) < 2880:
                raise ValueError(f"{path}: no END card in the primary header")
            for i in range(0, 2880, 80):
                card = block[i:i + 80].decode("ascii", "replace")
                key = card[:8].strip()
                if key == "END":
                    return header
                if card[8:10] == "= ":
                    header[key] = _card_value(card[10:])


def _card_value(text: str):
    """Value part of a header card: 'string', T/F, integer, float (comment after / removed)."""
    text = text.strip()
    if text.startswith("'"):
        end = text.find("'", 1)
        return text[1:end].rstrip()
    value = text.split("/")[0].strip()
    if value in ("T", "F"):
        return value == "T"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def waders_version(cfg: ProcessingConfig) -> dict[str, Any]:
    """Git commit (and whether the checkout has local changes) of the pysimdamicm used."""
    info: dict[str, Any] = {"pysimdamicm_dir": str(cfg.pysimdamicm_dir) if cfg.pysimdamicm_dir else None}
    if cfg.pysimdamicm_dir and cfg.pysimdamicm_dir.exists():
        try:
            info["commit"] = subprocess.run(["git", "-C", str(cfg.pysimdamicm_dir), "rev-parse", "--short", "HEAD"],
                                            capture_output=True, text=True, timeout=30).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(cfg.pysimdamicm_dir), "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, timeout=30).stdout.strip()
            info["local_changes"] = bool(dirty)
        except (OSError, subprocess.SubprocessError) as err:
            info["error"] = str(err)
    return info


def waders_config_for(cfg: ProcessingConfig, out_dir: Path) -> Path:
    """The WADERs JSON to use: the configured file, or a copy with ``image_overrides`` applied."""
    if cfg.waders_json is None:
        raise ValueError("no WADERs JSON configured ([waders] json)")
    if not cfg.image_overrides:
        return cfg.waders_json
    config = json.loads(cfg.waders_json.read_text())
    config["input"]["image"].update(cfg.image_overrides)
    path = out_dir / f"waders_config_{cfg.waders_json.stem}.json"
    path.write_text(json.dumps(config, indent=4))
    return path


def run_waders(fits_path: Path, out_dir: Path, cfg: ProcessingConfig) -> dict[str, Any]:
    """
    Run panaSKImg on one decompressed image.  Everything needed to repeat it is saved:
    the command line, the JSON (copied), the full log, the return code and the duration.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = waders_config_for(cfg, out_dir)
    shutil.copyfile(json_path, out_dir / "waders_config_used.json")
    command = [*cfg.waders_command, str(fits_path), *cfg.waders_extra_args,
               "-j", str(json_path), "-o", str(out_dir)]
    (out_dir / "command.txt").write_text(" ".join(command) + "\n")

    started = time.monotonic()
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=cfg.waders_timeout_s)
        returncode, log = done.returncode, done.stdout + "\n--- stderr ---\n" + done.stderr
    except subprocess.TimeoutExpired as err:
        returncode, log = -1, f"timed out after {cfg.waders_timeout_s} s\n{err.stdout or ''}"
    except OSError as err:
        returncode, log = -2, f"could not start {command[0]}: {err}"
    (out_dir / "log.txt").write_text(log)

    outputs = sorted(out_dir.glob("*_waders.fits*"))
    return {"command": command, "returncode": returncode, "seconds": round(time.monotonic() - started, 1),
            "log": str(out_dir / "log.txt"), "output": str(outputs[0]) if outputs else None,
            "json": str(json_path), "json_sha1": _sha1(json_path)}


def run_run_waders(fits_path: Path, out_dir: Path, cfg: ProcessingConfig) -> dict[str, Any]:
    """
    Run comm_scripts' ``run_WADERs`` on one decompressed image (runner = "run_WADERs").  It builds the panaSKImg
    configuration per CCD from the image itself (geometry from the header, gain, noise, pedestal method, fitting only
    the Poisson-like 0e-/1e- peaks) and writes fit_quality.csv with the fit results and their errors (scaled by
    sqrt(chi2/ndf)).  The command, log, return code and duration are saved, as for panaSKImg.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [*cfg.run_waders_command, str(fits_path), "-o", str(out_dir), *cfg.run_waders_args]
    (out_dir / "command.txt").write_text(" ".join(command) + "\n")
    started = time.monotonic()
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=cfg.waders_timeout_s)
        returncode, log = done.returncode, done.stdout + "\n--- stderr ---\n" + done.stderr
    except subprocess.TimeoutExpired as err:
        returncode, log = -1, f"timed out after {cfg.waders_timeout_s} s\n{err.stdout or ''}"
    except OSError as err:
        returncode, log = -2, f"could not start {command[0]}: {err}"
    (out_dir / "log.txt").write_text(log)
    csvs = sorted(out_dir.glob("**/fit_quality.csv"))
    return {"runner": "run_WADERs", "command": command, "returncode": returncode, "seconds": round(time.monotonic() - started, 1),
            "log": str(out_dir / "log.txt"), "output": str(csvs[0]) if csvs else None}


def labels_from_run_waders(csv_path: Path, image_meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """
    The labels of every channel from run_WADERs' fit_quality.csv (one row per CCD = ACM channel; EXT<n> = channel
    n-1), in the same form as ``extract_labels``.  The errors are the scaled ones (x sqrt(chi2/ndf)).  Flags: a channel
    run_WADERs did not fit (flooded or unresolved electron peaks), an error larger than half the value, or a fit with
    chi2/ndf > 3.
    """
    import csv
    rows = {r["ext"]: r for r in csv.DictReader(open(csv_path))}
    labels = []
    for ext in range(1, 5):
        row: dict[str, Any] = {"ext": ext, "channel": ext - 1, "flags": [], "labels_from": "run_WADERs"}
        if image_meta:
            row["amplifier"] = image_meta.get(f"ch{ext - 1}")
        r = rows.get(f"EXT{ext}")
        if r is None:
            row["flags"].append("not fitted by run_WADERs (flooded or unresolved peaks)")
            labels.append(row)
            continue
        f = {k: (float(v) if v not in ("", "None", None) else float("nan")) for k, v in r.items() if k not in ("image", "ext", "gain_held")}
        g, eg = f["gain"], f["gain_err_s"]
        row.update(gain_adu_per_e=g, gain_err=eg, sigma_e=f["noise"], sigma_e_err=f["noise_err_s"],
                   lambda_e_per_bin=f["dc"], lambda_err=f["dc_err_s"], sigma_adu=f["noise"] * g,
                   sigma_err=f["noise_err_s"] * g, dc_adu=f["dc"] * g, dc_err=f["dc_err_s"] * g, mu_adu=f.get("mu"),
                   chi2ndf=f.get("chi2ndf"), dc_bias=f.get("dc_bias"), gain_held=str(r.get("gain_held")) == "True")
        for name, value, err in (("gain", g, eg), ("sigma_e", f["noise"], f["noise_err_s"]), ("lambda", f["dc"], f["dc_err_s"])):
            if not (value > 0) or not (err == err) or err > 0.5 * abs(value):
                row["flags"].append(f"{name} = {value:.3g} +/- {err:.2g} is not constrained")
        if f.get("chi2ndf", 0) > 3:
            row["flags"].append(f"chi2/ndf = {f['chi2ndf']:.1f} > 3: the fit model does not describe the image")
        labels.append(row)
    return labels


def _sha1(path: Path) -> str:
    import hashlib
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Step 4: labels                                                               #
# --------------------------------------------------------------------------- #

def fit_bounds(waders_json: Path | None) -> dict[str, tuple[float, float]]:
    """The parameter limits WADERs used (JSON values, else the pysimdamicm defaults)."""
    bounds = dict(WADERS_DEFAULT_BOUNDS)
    if waders_json and Path(waders_json).exists():
        fit = json.loads(Path(waders_json).read_text()).get("process", {}).get("FitDarkCurrentProcess", {})
        for name in bounds:
            low, high = bounds[name]
            bounds[name] = (float(fit.get(f"{name}_min", low)), float(fit.get(f"{name}_max", high)))
    return bounds


def _at_bound(value: float, low: float, high: float) -> bool:
    """True if ``value`` is within 0.1 % of the allowed range from one of its limits."""
    return any(abs(value - b) <= 1e-3 * (high - low) for b in (low, high))


def extract_labels(header: dict[str, Any], bounds: dict[str, tuple[float, float]],
                   image_meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """
    Per amplifier extension EXT<n> (= ACM channel n-1): the WADERs dark-current fit results.

    WADERs fits the pixel-charge distribution (ADU, pedestal subtracted) with
    Norm x sum_k Gauss(x; mu0 + k*gain, sigma) x Poisson(k; lambda): so
        MEGAIN = gain (ADU per electron), MESIG = sigma (ADU), MEDC = lambda x gain (ADU),
        sigma_e = MESIG / MEGAIN  (read noise, electrons)
        lambda  = MEDC / MEGAIN   (electrons per pixel (bin) per image: dark current +
                                   spurious charge accumulated during exposure and readout)
    Flags (empty = trustworthy):
      * a parameter at (within 0.1 % of the range from) its fit limit: the fit was stopped by
        the limit, the value is not a measurement;
      * an error that is zero or implausibly small (< 1e-4 of the value): typical of a parameter
        stuck at a limit;
      * an error larger than half the value: the image does not constrain it (e.g. too few
        1-electron pixels to measure the gain or the dark current);
      * missing keys.
    """
    labels = []
    for ext in range(1, 5):
        keys = {"gain": f"MEGAIN{ext}", "sigma": f"MESIG{ext}", "dc": f"MEDC{ext}", "mu": f"MEMU0{ext}"}
        row: dict[str, Any] = {"ext": ext, "channel": ext - 1, "flags": []}
        if image_meta:
            row["amplifier"] = image_meta.get(f"ch{ext - 1}")
        if not all(k in header for k in keys.values()):
            row["flags"].append("missing in WADERs output")
            labels.append(row)
            continue
        value = {name: float(header[k]) for name, k in keys.items()}
        error = {name: float(header.get("E" + k, float("nan"))) for name, k in keys.items()}
        row.update({f"{name}_adu": value[name] for name in ("sigma", "dc", "mu")},
                   gain_adu_per_e=value["gain"],
                   **{f"{name}_err": error[name] for name in value},
                   included_fraction=header.get(f"MEFRAC{ext}"))
        g, eg = value["gain"], error["gain"]
        row["sigma_e"] = value["sigma"] / g
        row["sigma_e_err"] = math.hypot(error["sigma"] / g, value["sigma"] * eg / g ** 2)
        row["lambda_e_per_bin"] = value["dc"] / g
        row["lambda_err"] = math.hypot(error["dc"] / g, value["dc"] * eg / g ** 2)
        for name in ("gain", "sigma", "dc", "mu"):
            low, high = bounds[name]
            if _at_bound(value[name], low, high):
                row["flags"].append(f"{name} at fit limit ({value[name]:g}; limits {low:g}..{high:g})")
            if name != "gain" and not (error[name] > 1e-4 * abs(value[name])):
                row["flags"].append(f"{name} error {error[name]:.2g} is zero or implausibly small")
            if name != "mu" and error[name] > 0.5 * abs(value[name]):
                row["flags"].append(f"{name} = {value[name]:.3g} +/- {error[name]:.2g} is not constrained")
        labels.append(row)
    return labels


# --------------------------------------------------------------------------- #
# The whole packet                                                             #
# --------------------------------------------------------------------------- #

def process_packet(manifest_path: str | Path, cfg: ProcessingConfig, skip_waders: bool = False) -> dict:
    """Run steps 1-4 for one packet and write its results; returns the result summary."""
    manifest = json.loads(Path(manifest_path).read_text())
    out = cfg.out_dir / manifest["packet_id"]
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"packet_id": manifest["packet_id"], "manifest": str(manifest_path),
                              "manifest_status": manifest.get("status"), "warnings": []}
    if manifest.get("status") != "ok":
        result["warnings"].append(f"manifest status is {manifest.get('status')!r}; processed anyway")
    files = packet_files(manifest, cfg)

    # -- traces ----------------------------------------------------------------
    traces = [process_trace(entry, out, cfg) for entry in files["traces"]]
    combined = combine_repeats(traces)
    differences = trace_differences(combined)
    (out / "features.json").write_text(json.dumps(_json_safe(
        {"traces": traces, "combined": combined, "differences": differences}), indent=1))

    # -- images and labels -------------------------------------------------------
    images = []
    bounds = fit_bounds(cfg.waders_json)
    for image in files["images"]:
        entry: dict[str, Any] = {"board": image["board"], "fits_original": str(image["fits"]) if image["fits"] else None}
        meta = json.loads(image["meta"].read_text()) if image["meta"].exists() else {}
        entry["fake"] = bool(meta.get("FAKE", False))
        if image["fits"] is None or not image["fits"].exists():
            entry["error"] = "FITS file not found (image builder output missing?)"
            images.append(entry)
            continue
        fits = decompress_fits(image["fits"], out / "fits")
        entry["fits"] = str(fits)
        if cfg.drift_enabled:
            # the along-row baseline drift (image_drift.py): if present, WADERs and the per-row fit use the corrected copy
            # (first ``drift_cut`` columns removed, the slow remainder subtracted); without a drift the copy is unchanged
            try:
                from image_drift import correct_image
                (out / "fits_drift").mkdir(parents=True, exist_ok=True)
                drift = correct_image(fits, out / "fits_drift", cfg.drift_cut, cfg.drift_width_tol, 400, cfg.drift_channels)
                entry["drift"] = {k: drift.get(k) for k in ("drift", "cut", "warning", "warning_channels", "output")}
                entry["drift"]["channels"] = drift.get("channels")
                if drift.get("drift"):
                    entry["fits_original"] = str(fits)
                    fits = Path(drift["output"])
                    entry["fits"] = str(fits)
                    if drift.get("warning"):
                        result["warnings"].append(f"drift on board {image['board']}: {drift['warning']}")
            except Exception as exc:                              # never lose the image over the drift step
                result["warnings"].append(f"drift check failed on board {image['board']} (image used as is): {exc}")
        header = read_fits_primary(fits)
        entry["image_header"] = {k: header.get(k) for k in ("NROW", "NCOL", "NDCM", "NPBIN", "NSBIN", "EXPOSURE")}
        if not skip_waders and cfg.waders_runner == "run_WADERs":
            run = run_run_waders(fits, out / "waders" / f"acm{image['board']}", cfg)
            entry["waders"] = run
            if run["output"]:
                entry["labels"] = labels_from_run_waders(Path(run["output"]), meta)
                if run["returncode"] != 0:
                    result["warnings"].append(f"run_WADERs returned {run['returncode']} on board {image['board']}; see {run['log']}")
            else:
                entry["error"] = f"run_WADERs wrote no fit_quality.csv (return code {run['returncode']}); see {run['log']}"
        elif not skip_waders:
            run = run_waders(fits, out / "waders" / f"acm{image['board']}", cfg)
            entry["waders"] = run
            if run["returncode"] == 0 and run["output"]:
                entry["labels"] = extract_labels(read_fits_primary(Path(run["output"])), bounds, meta)
            else:
                entry["error"] = f"panaSKImg failed (return code {run['returncode']}); see {run['log']}"
        if entry.get("labels") and cfg.rows_enabled:
            try:
                from image_rows import image_rows
                # geometry for the active columns: the FITS header, overridden by the .meta where both have it
                # geometry: the image's own header wins (a drift-corrected copy has NCOLPRE / NCOL adjusted for the cut)
                rows = image_rows(fits, entry["labels"], {**meta, **header}, reverse=cfg.rows_reverse)
                for lab in entry["labels"]:
                    lab.update(rows.get(lab["channel"], {}))
            except Exception as exc:                              # never lose the labels over the row fit
                result["warnings"].append(f"per-row lambda failed on board {image['board']}: {exc}")
        if entry.get("labels") and entry.get("drift"):
            d = entry["drift"]
            for lab in entry["labels"]:
                lab.update(drift_corrected=bool(d.get("drift")), drift_cut=d.get("cut") if d.get("drift") else 0,
                           drift_warning=lab["channel"] in (d.get("warning_channels") or []))
        images.append(entry)

    labels = {"waders_version": waders_version(cfg) if not skip_waders else None,
              "waders_json": str(cfg.waders_json), "fit_bounds": bounds,
              "images": [{k: v for k, v in im.items() if k in ("board", "labels", "error", "fake")} for im in images]}
    (out / "labels.json").write_text(json.dumps(_json_safe(labels), indent=1))

    result.update(traces=traces, combined=combined, differences=differences, images=images,
                  waders_version=labels["waders_version"],
                  fake=any(t.get("fake") for t in traces) or any(im.get("fake") for im in images))
    if result["fake"]:
        result["warnings"].append("FAKE packet: files are copies of older data, not measurements at theta")
    (out / "result.json").write_text(json.dumps(_json_safe(result), indent=1))
    return result


# --------------------------------------------------------------------------- #
# Step 5: grouping into a dataset                                              #
# --------------------------------------------------------------------------- #

def dataset_rows(result: dict, manifest: dict) -> list[dict[str, Any]]:
    """
    One row per (board, channel) of a packet: that board's settings and change, the trace
    features and the labels.

    Columns: ``theta.*`` and ``delta.*`` (this board's parameters and change from the
    reference), ``label.*`` (WADERs), ``trace.*`` / ``ref.*`` (tools averaged over the repeated
    trace sets at theta / at the reference), ``trace_sd.*`` / ``ref_sd.*`` (their scatter between
    sets), ``diff.*`` (trace - ref).
    """
    def per_board(table, board):                    # manifest tables are keyed by board (as strings in JSON)
        return (table or {}).get(str(board), (table or {}).get(board, {})) or {}

    combined = {(c["kind"], c["board"], c["channel"]): c for c in result.get("combined", [])}
    diffs = {(d["board"], d["channel"]): d["difference"] for d in result["differences"]}
    rows = []
    for image in result["images"]:
        board = image["board"]
        base = {"packet_id": manifest["packet_id"], "scan_name": manifest.get("scan_name"),
                "board": board, "theta_id": per_board(manifest.get("theta_id"), board) or None,
                "reference_id": per_board(manifest.get("reference_id"), board) or None,
                "date_start": manifest.get("date_start"),
                "seconds_since_previous": manifest.get("seconds_since_previous"),
                "fake": result.get("fake", False),
                **{f"theta.{k}": v for k, v in per_board(manifest.get("theta"), board).items()},
                **{f"delta.{k}": v for k, v in per_board(manifest.get("delta_theta"), board).items()}}
        for label in image.get("labels") or [{"channel": c, "flags": ["no labels"]} for c in range(4)]:
            ch = label["channel"]
            row = {**base, "channel": ch, "amplifier": label.get("amplifier")}
            for key in ("gain_adu_per_e", "gain_err", "sigma_e", "sigma_e_err", "lambda_e_per_bin",
                        "lambda_err", "sigma_adu", "sigma_err", "dc_adu", "dc_err", "mu_adu", "included_fraction",
                        "lambda_dark", "lambda_cic", "lambda_dark_err", "lambda_cic_err", "lambda_dark_cic_cov",
                        "rows_chi2_dof", "n_rows_fit", "drift_corrected", "drift_cut", "drift_warning"):
                row[f"label.{key}"] = label.get(key)
            for key, value in (image.get("image_header") or {}).items():
                row[f"image.{key}"] = value
            row["label.flags"] = "; ".join(label.get("flags", []))
            row["label.ok"] = not label.get("flags")
            for kind, prefix in (("theta", "trace"), ("ref", "ref")):
                c = combined.get((kind, board, ch), {})
                row[f"{prefix}.n_usable"] = c.get("n_usable", 0)
                # repeat values left out as outliers (summed over the features; 0 = none)
                row[f"{prefix}.n_rejected_values"] = sum(c.get("n_rejected", {}).values())
                for tool, value in c.get("mean", {}).items():
                    row[f"{prefix}.{tool}"] = value
                for tool, value in c.get("sd", {}).items():
                    row[f"{prefix}_sd.{tool}"] = value
            for tool, value in diffs.get((board, ch), {}).items():
                row[f"diff.{tool}"] = value
            rows.append(row)
    return rows


def build_dataset(out_dir: Path) -> Path:
    """Rebuild dataset.jsonl and dataset.csv from every processed packet in ``out_dir``."""
    rows = []
    for result_path in sorted(out_dir.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        manifest = json.loads(Path(result["manifest"]).read_text())
        rows += dataset_rows(result, manifest)
    with open(out_dir / "dataset.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row)) + "\n")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(out_dir / "dataset.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(_json_safe(rows))
    return out_dir / "dataset.csv"


def build_model_inputs(out_dir: Path) -> None:
    """The per-model input files (``<out_dir>/model_inputs/``; ml/real_inputs.py), from dataset.csv and the .npz curves."""
    try:
        from ml import real_inputs
    except ImportError as exc:                                  # the ML environment (requirements-ml.txt) is missing
        print(f"model inputs not written ({exc}); run: python -m ml.real_inputs {out_dir}")
        return
    for path in real_inputs.build_all(out_dir):
        print("wrote", path)


# --------------------------------------------------------------------------- #
# Command line                                                                 #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("manifests", nargs="*", help="packet manifest(s)")
    parser.add_argument("--packets-dir", help="process every manifest with status ok in this directory")
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--out", help="override [output] dir")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                        help="map a DAQ path prefix to a local one (repeatable)")
    parser.add_argument("--skip-waders", action="store_true", help="traces only")
    args = parser.parse_args()

    cfg = ProcessingConfig.load(args.config)
    if args.out:
        cfg.out_dir = Path(args.out).resolve()
    for item in args.path_map:
        old, new = item.split("=", 1)
        cfg.path_map = {old: new, **cfg.path_map}

    manifests = [Path(m) for m in args.manifests]
    if args.packets_dir:
        for path in sorted(Path(args.packets_dir).glob("*.json")):
            status = json.loads(path.read_text()).get("status")
            if status == "ok":
                manifests.append(path)
            else:
                print(f"skipped {path.name}: status {status!r} (check its warnings; give it explicitly to process it)")
    if not manifests:
        parser.error("give manifest files or --packets-dir")

    for path in manifests:
        result = process_packet(path, cfg, args.skip_waders)
        n_ok = sum(1 for t in result["traces"] if t.get("usable"))
        labels = [l for im in result["images"] for l in im.get("labels", [])]
        print(f"{result['packet_id']}: {n_ok}/{len(result['traces'])} traces usable, "
              f"{sum(1 for l in labels if not l['flags'])}/{len(labels)} labels without flags"
              + ("".join(f"\n   WARNING {w}" for w in result["warnings"])))
    print(f"dataset: {build_dataset(cfg.out_dir)}")
    if cfg.model_inputs:
        build_model_inputs(cfg.out_dir)


if __name__ == "__main__":
    main()
