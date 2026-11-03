"""
real_inputs.py -- ready-to-train input files, one per model family, from processed packets.

``process_packet.py`` writes ``dataset.csv`` (one row per packet x board x channel: settings, trace features,
labels) and, per trace, an ``.npz`` with the curves.  Different models need different things (numbers, curves,
every trace set separately, unlabeled traces), so this step writes one file per model FAMILY into
``<processed>/model_inputs/``; a model then needs one ``load`` call:

=============  ====================================================  ==========================================
file           models                                                content (besides the common part)
=============  ====================================================  ==========================================
features.npz   Bayesian ridge, boosted trees, GP, NN ensemble        own trace features, own reference features
curves.npz     CNN on curves; D (pre-trained, probe or fine-tuned);  the curves at theta and of the reference
               C (deep kernel)                                       traces (C x 256 each), log scales applied
physics.npz    A, physics-structured (+ joint dark current / CIC)    own features, the Allan curve (common
                                                                     pattern removed), skips, per-row lambda fit
sets.npz       E, set model (inside A)                               every trace set's Allan curve, separately
pretrain.npz   D's pre-training                                      curves of EVERY usable trace (reference and
                                                                     theta, every set), no labels needed
=============  ====================================================  ==========================================

Common part of every labeled file (row i = one amplifier in one packet; ONE TRACE -> ONE TRAINING EXAMPLE):

    packet_id, board, channel, amplifier   identification; ``packet_id`` is the group for cross-validation (the
                                           rows of a packet share the CCD's state at that moment and must stay
                                           together); board/channel/amplifier are for evaluation only and are
                                           NEVER model inputs (the level-1 models use no amplifier labels)
    Y, Y_err, targets                      log sigma_e, log lambda, log gain and their errors in the log (relative
                                           errors from WADERs); NaN where missing
    label_ok, fake                         WADERs fit without flags; packet made of copied files (rehearsal)
    trace_ok                               at least one usable trace at the packet's own settings
    theta, theta_names                     the settings (as the boards report them)

Nothing here is fitted to the data (no standardization, no imputation), so nothing leaks between training and test
rows: every model's preparation happens inside its training folds (lvl1/train_lvl1.py).

    python -m ml.real_inputs <processed dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .structured import REAL_ALLAN

TARGETS = ("sigma_e", "lambda", "gain")
LABEL_COLS = {"sigma_e": ("label.sigma_e", "label.sigma_e_err"), "lambda": ("label.lambda_e_per_bin", "label.lambda_err"),
              "gain": ("label.gain_adu_per_e", "label.gain_err")}
CURVE_LENGTH = 256
FAMILIES = {
    "features": "ridge, boosted trees, GP, NN ensemble: settings + own trace + own reference features",
    "curves": "CNN on curves; D (pre-trained); C (deep kernel): trace and reference curves + settings",
    "physics": "A, physics-structured (+ joint dark current / CIC): own features, Allan curve, skips, per-row lambda",
    "sets": "E, set model: every trace set's Allan curve",
    "pretrain": "D's pre-training: the curves of every usable trace, no labels",
}


def _log_names(names) -> tuple[str, ...]:
    """Curves spanning decades are taken in log: every Allan, Hadamard and modified Allan deviation, the spectra and the
    within-cycle structure function."""
    return tuple(n for n in names if any(k in n for k in ("Allan", "Hadamard", "spectrum", "structure function")))


def common(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """The part every labeled file has (see the module docstring)."""
    Y = np.full((len(df), 3), np.nan)
    E = np.full((len(df), 3), np.nan)
    for t, name in enumerate(TARGETS):
        v, e = LABEL_COLS[name]
        val = pd.to_numeric(df.get(v), errors="coerce").to_numpy(float) if v in df else np.full(len(df), np.nan)
        err = pd.to_numeric(df.get(e), errors="coerce").to_numpy(float) if e in df else np.full(len(df), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            Y[:, t] = np.where(val > 0, np.log(np.where(val > 0, val, 1.0)), np.nan)
            E[:, t] = np.where(val > 0, err / np.where(val > 0, val, 1.0), np.nan)
    theta_names = data.columns(df, "theta")
    ok = df["label.ok"].astype(str).str.lower().eq("true").to_numpy() if "label.ok" in df else np.zeros(len(df), bool)
    # trace_ok: the amplifier has at least one usable trace at the packet's own settings (2026-11-03: in L1_37, L1_40 and
    # L1_41 every trace at the new settings was damaged; a model then predicts from filled-in values, not from a trace)
    n_usable = pd.to_numeric(df.get("trace.n_usable"), errors="coerce") if "trace.n_usable" in df else None
    trace_ok = (n_usable.fillna(0).to_numpy() > 0) if n_usable is not None else np.ones(len(df), bool)
    return {"packet_id": df["packet_id"].astype(str).to_numpy(), "board": df["board"].to_numpy(int),
            "channel": df["channel"].to_numpy(int), "amplifier": df.get("amplifier", pd.Series([""] * len(df))).astype(str).to_numpy(),
            "Y": Y, "Y_err": E, "targets": np.array(TARGETS), "label_ok": ok, "trace_ok": trace_ok,
            "fake": df.get("fake", pd.Series([False] * len(df))).astype(str).str.lower().eq("true").to_numpy(),
            "theta": df[theta_names].apply(pd.to_numeric, errors="coerce").to_numpy(float), "theta_names": np.array(theta_names)}


def _own_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    trace, ref = data.columns(df, "trace"), data.columns(df, "ref")
    names = sorted(set(c.split(".", 1)[1] for c in trace) & set(c.split(".", 1)[1] for c in ref))
    num = lambda cols: df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return {"X_trace": num([f"trace.{n}" for n in names]), "X_ref": num([f"ref.{n}" for n in names]),
            "feature_names": np.array(names)}


def _l1_own_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """The level-1 strip features as the "own" inputs: X_trace = l1.* (at theta), X_ref = l1ref.* (at the reference)."""
    new = [c for c in df.columns if c.startswith("l1.") and c != "l1.usable"]
    ref = [c for c in df.columns if c.startswith("l1ref.") and c != "l1ref.usable"]
    names = sorted(set(c.split(".", 1)[1] for c in new) & set(c.split(".", 1)[1] for c in ref))
    num = lambda cols: df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return {"X_trace": num([f"l1.{n}" for n in names]), "X_ref": num([f"l1ref.{n}" for n in names]),
            "feature_names": np.array(names)}


def _l1_allan(out_dir: Path, df: pd.DataFrame, ns: np.ndarray) -> np.ndarray:
    """Per row, the strip's within-pixel Allan curve (at the packet's settings), interpolated (in log) onto ``ns``."""
    curves = {}
    for result_path in sorted(out_dir.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        for st in result.get("strips") or []:
            if st.get("kind") == "new" and st.get("usable") and st.get("npz") and Path(st["npz"]).exists():
                with np.load(st["npz"]) as z:
                    a, n = z["curve: l1: Allan deviation"], z["curve: l1: Allan n"]
                ok = np.isfinite(a) & (a > 0)
                if ok.sum() >= 2:
                    curves[(result["packet_id"], st["board"], st["channel"])] = np.exp(
                        np.interp(np.log(ns), np.log(n[ok]), np.log(a[ok]), left=np.nan, right=np.nan))
    out = np.full((len(df), len(ns)), np.nan)
    for i, (p, b, c) in enumerate(zip(df["packet_id"].astype(str), df["board"].astype(int), df["channel"].astype(int))):
        if (p, b, c) in curves:
            out[i] = curves[(p, b, c)]
    return out


def _traces(out_dir: Path) -> list[dict]:
    """Every trace of every processed packet: packet_id, kind, board, channel, repeat, usable, npz, theta."""
    rows = []
    for result_path in sorted(out_dir.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        manifest = {}
        if result.get("manifest") and Path(result["manifest"]).exists():
            manifest = json.loads(Path(result["manifest"]).read_text())
        for t in result.get("traces", []):
            key = "theta" if t.get("kind") == "theta" else "theta_ref"
            th = (manifest.get(key) or {}).get(str(t["board"]), (manifest.get(key) or {}).get(t["board"], {})) or {}
            rows.append({**{k: t.get(k) for k in ("kind", "board", "channel", "repeat", "usable", "npz")},
                         "packet_id": result["packet_id"], "theta": th})
    return rows


def _curves_of(npz_path: str) -> dict[str, np.ndarray]:
    with np.load(npz_path) as arrays:
        return {k[7:]: arrays[k] for k in arrays.files if k.startswith("curve: ")}


def per_row_curves(df: pd.DataFrame, traces: list[dict], kind: str) -> tuple[dict[str, np.ndarray], list[list[dict]]]:
    """
    For every dataset row: the median of its usable traces' curves (kind 'theta' or 'ref'), and the list of
    each trace set's curves (for the set model).  Curves are resampled to the length of the first one found.
    """
    by_row: dict[tuple, list[dict]] = {}
    for t in traces:
        if t["kind"] == kind and t.get("usable") and t.get("npz") and Path(t["npz"]).exists():
            by_row.setdefault((t["packet_id"], int(t["board"]), int(t["channel"])), []).append(_curves_of(t["npz"]))
    names = sorted({n for sets in by_row.values() for c in sets for n in c})
    lengths = {n: next(len(c[n]) for sets in by_row.values() for c in sets if n in c) for n in names}
    med = {n: np.full((len(df), lengths[n]), np.nan) for n in names}
    sets_per_row = []
    for i, row in enumerate(df.itertuples(index=False)):
        sets = by_row.get((str(row.packet_id), int(row.board), int(row.channel)), [])
        sets_per_row.append(sets)
        for n in names:
            vals = [data.resample(c[n], lengths[n]) for c in sets if n in c]
            if vals:
                med[n][i] = np.nanmedian(vals, axis=0)
    return med, sets_per_row


def _tensor(curves: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return data.curve_tensor(curves, names, length=CURVE_LENGTH, log_names=_log_names(names), log_frequency_spectra=True)


def build_all(out_dir: str | Path) -> list[Path]:
    """Write every model-input file for the processed packets in ``out_dir``; returns the paths."""
    out_dir = Path(out_dir)
    df = pd.read_csv(out_dir / "dataset.csv")
    dest = out_dir / "model_inputs"
    dest.mkdir(exist_ok=True)
    base = common(df)
    has_l1 = "l1.usable" in df
    if has_l1:
        # level-1 strips present: the three sources (traces, strips, both) must be scored on the SAME rows, so a row needs
        # a usable trace AND a usable strip in all three (docs/context/2026-11-03, section 13)
        base["trace_ok"] = base["trace_ok"] & df["l1.usable"].astype(str).str.lower().eq("true").to_numpy()
    own = _own_features(df)
    traces = _traces(out_dir)
    written = []

    def save(name, dest=dest, **arrays):
        path = dest / f"{name}.npz"
        arrays = {k: (np.asarray(v).astype(str) if np.asarray(v).dtype == object else v) for k, v in arrays.items()}
        np.savez_compressed(path, family=np.array(name), description=np.array(FAMILIES[name]), **arrays)
        written.append(path)

    save("features", **base, **own)

    cur_t, sets_t = per_row_curves(df, traces, "theta")
    cur_r, _ = per_row_curves(df, traces, "ref")
    names = sorted(set(cur_t) & set(cur_r))
    if names:
        save("curves", **base, curves_theta=_tensor(cur_t, names), curves_ref=_tensor(cur_r, names), curve_names=np.array(names))

    from trace_features import ALLAN_N
    n_skips = pd.to_numeric(df.get("image.NDCM"), errors="coerce").to_numpy(float) if "image.NDCM" in df else np.full(len(df), np.nan)
    allan_name = REAL_ALLAN if REAL_ALLAN in cur_t else "noise: Allan deviation"
    physics = {**base, **own, "n_skips": n_skips, "allan_ns": np.asarray(ALLAN_N, float), "allan_curve_name": np.array(allan_name),
               "allan": cur_t.get(allan_name, np.full((len(df), len(ALLAN_N)), np.nan))}
    if "label.lambda_dark" in df:
        parts = df[["label.lambda_dark", "label.lambda_cic"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        cov = np.full((len(df), 2, 2), np.nan)
        for j, (a, b) in enumerate((("label.lambda_dark_err", "label.lambda_dark_err"), ("label.lambda_cic_err", "label.lambda_cic_err"))):
            cov[:, j, j] = pd.to_numeric(df[a], errors="coerce").to_numpy(float) ** 2
        cov[:, 0, 1] = cov[:, 1, 0] = pd.to_numeric(df.get("label.lambda_dark_cic_cov"), errors="coerce").to_numpy(float)
        physics.update(lambda_parts=parts, lambda_cov=cov)
    save("physics", **physics)

    n_sets = np.array([len(s) for s in sets_t])
    if n_sets.max(initial=0) > 0:
        S = int(n_sets.max())
        sets = np.full((len(df), S, len(ALLAN_N)), np.nan)
        for i, row_sets in enumerate(sets_t):
            for k, c in enumerate(row_sets):
                if allan_name in c:
                    sets[i, k] = data.resample(c[allan_name], len(ALLAN_N))
        save("sets", **base, sets=sets, n_sets=n_sets, allan_ns=np.asarray(ALLAN_N, float), allan_curve_name=np.array(allan_name))

    usable = [t for t in traces if t.get("usable") and t.get("npz") and Path(t["npz"]).exists()]
    if usable and names:
        cs = [_curves_of(t["npz"]) for t in usable]
        lengths = {n: len(next(c[n] for c in cs if n in c)) for n in names}
        stacked = {n: np.array([data.resample(c[n], lengths[n]) if n in c else np.full(lengths[n], np.nan) for c in cs]) for n in names}
        tn = sorted({k for t in usable for k in t["theta"]})
        save("pretrain", curves=_tensor(stacked, names), curve_names=np.array(names),
             packet_id=np.array([t["packet_id"] for t in usable]), kind=np.array([t["kind"] for t in usable]),
             board=np.array([int(t["board"]) for t in usable]), channel=np.array([int(t["channel"]) for t in usable]),
             theta=np.array([[float(t["theta"].get(k, np.nan)) for k in tn] for t in usable]), theta_names=np.array(tn))
    if has_l1:
        # source "l1": the strip features instead of the trace features (features and physics families; the curve and
        # set families are trace-only); source "both": trace + strip features (and the trace families as they are)
        l1_own = _l1_own_features(df)
        for name, d in (("l1", out_dir / "model_inputs_l1"), ("both", out_dir / "model_inputs_both")):
            d.mkdir(exist_ok=True)
            if name == "l1":
                o = l1_own
                phys = {**physics, **o, "allan": _l1_allan(out_dir, df, np.asarray(ALLAN_N, float)),
                        "allan_curve_name": np.array("l1: Allan deviation")}
            else:
                o = {"X_trace": np.hstack([own["X_trace"], l1_own["X_trace"]]), "X_ref": np.hstack([own["X_ref"], l1_own["X_ref"]]),
                     "feature_names": np.concatenate([own["feature_names"], l1_own["feature_names"]])}
                phys = {**physics, **o}
            save("features", dest=d, **base, **o)
            save("physics", dest=d, **phys)
            if name == "both":
                for fam in ("curves", "sets", "pretrain"):
                    if (dest / f"{fam}.npz").exists():
                        z = dict(np.load(dest / f"{fam}.npz", allow_pickle=False))
                        if "trace_ok" in z:
                            z["trace_ok"] = base["trace_ok"]
                        np.savez_compressed(d / f"{fam}.npz", **z)
                        written.append(d / f"{fam}.npz")
    return written


def load(path: str | Path) -> dict[str, np.ndarray]:
    """One model-input file as a dict of arrays (0-d string arrays become str)."""
    with np.load(path, allow_pickle=False) as f:
        return {k: (f[k].item() if f[k].ndim == 0 else f[k]) for k in f.files}


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    for p in build_all(sys.argv[1]):
        print("wrote", p)
