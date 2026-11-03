"""
make_manual_manifest.py -- a packet manifest (the input of process_packet.py) for packets taken by hand.

``take_packet.py`` writes its manifest itself.  A packet queued with ``daq_scripts/manual_packet.sh`` only
leaves a small RECORD (``<PACKET_ID>.json``: the packet's name, the requested changes, the trace and image
settings) plus the raw files that cdaq writes.  This script finds the packet's files in the run directory
and writes a manifest that ``process_packet.py`` reads like any other:

* traces: every ``<PACKET_ID>_ref<set>_*.meta`` (reference, at the nominal configuration) and
  ``<PACKET_ID>_new<set>_*.meta`` (at the new settings); board from the .meta (ACMID), channel from the name;
* images: per board, the first image (a .meta with an ``avg_<name>.fits*`` next to it) that started after
  the packet's last trace of that board, and before any trace of another packet;
* theta: the parameters of the traces at the new settings, READ FROM THEIR .meta FILES (what the board
  reports, not what was asked); theta_ref the same for the reference traces; delta_theta = theta - theta_ref.
  Every requested change is compared with the .meta values: a change that did not arrive is a warning and
  the manifest status becomes "check".

A second mode handles older run directories without records (e.g. the 2026-10-12 packet tests): consecutive
files of one board with identical settings form a packet; each packet with an image becomes one manifest; the
reference traces are those of the group given by ``--reference-group`` (default: the first).

    python make_manual_manifest.py <record.json> [<record.json> ...] --run-dir <dir> [--out manifests]
    python make_manual_manifest.py --legacy-run <dir> --board 101 [--out manifests]
    then: python process_packet.py manifests/<packet_id>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

#: Parameter names that make up theta (operating parameters; geometry such as NDCM or NROW is not theta).
THETA_PATTERNS = [r"^(V[123]|H[123]|TG|SW|OG|RG|DG)_[A-C12]_[HL]$", r"^(VSUB|VDD[12]|VDRAIN[12]|VR[12])$",
                  r"^delay", r"^(NVWAIT|NPSD)$", r"^OFS_"]
#: Image / trace geometry recorded with the manifest (not theta).
GEOMETRY_KEYS = ("NROW", "NCOL", "NDCM", "NPBIN", "NSBIN", "EXPOSURE", "NCOLPRE", "NROWPRE", "PRESCAN", "PHYSROW", "PHYSCOl")
#: Tolerance when comparing a requested change with the reported value: a clock rail reads back what was set; a
#: bias reads back the nearest potentiometer step (45 -> 45.095, 40 -> 40.186, 50 -> 50.173 on 2026-11-02), so it
#: gets the readback tolerance of daq_scripts/manual_packet.sh.
VOLT_TOL = 0.15
BIAS_TOL = 0.3
BIAS_RE = re.compile(r"^(VSUB|VDD[12]|VDRAIN[12]|VR[12])$")
TRACE_RE = re.compile(r"_ch(\d)_(\d+)_\d{8}_\d{6}_\d+$")


def is_theta_key(key: str) -> bool:
    """True if a .meta key is an operating parameter (a voltage or a timing of THETA_PATTERNS)."""
    return any(re.match(p, key) for p in THETA_PATTERNS)


def theta_of(meta: dict) -> dict[str, float]:
    """The operating parameters of a .meta (numeric values of the theta keys)."""
    return {k: v for k, v in meta.items() if is_theta_key(k) and isinstance(v, (int, float)) and not isinstance(v, bool)}


def config_id(theta: dict) -> str:
    """A short stable hash of a setting (sorted theta): packets at identical settings share it."""
    return hashlib.sha1(json.dumps(theta, sort_keys=True).encode()).hexdigest()[:10]


def read_metas(run_dir: Path) -> list[dict]:
    """Every .meta of the run directory with its stem, start time and whether an averaged image exists."""
    out = []
    for p in sorted(run_dir.glob("*.meta")):
        try:
            m = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        stem = p.with_suffix("")
        out.append({"meta": m, "stem": stem, "name": stem.name, "board": int(m.get("ACMID", -1)),
                    "start": datetime.fromisoformat(m["DATEini"]) if "DATEini" in m else None,
                    "end": datetime.fromisoformat(m["DATEend"]) if "DATEend" in m else None,
                    "is_image": any(run_dir.glob(f"avg_{stem.name}.fits*"))})
    return [e for e in out if e["start"] is not None]


def trace_entry(e: dict, repeat: int) -> dict:
    """The manifest entry of one trace: board, channel, repeat (trace set) and file stem."""
    m = TRACE_RE.search(e["name"])
    return {"board": e["board"], "channel": int(m.group(1)), "repeat": repeat, "stem": str(e["stem"])}


def parse_changes(changes: list[str], boards: list[int]) -> dict[int, dict[str, float]]:
    """'NAME=VALUE' (every board) or 'acm101:NAME=VALUE' (one board) -> {board: {name: value}}."""
    out: dict[int, dict[str, float]] = {b: {} for b in boards}
    for c in changes:
        target, spec = (c.split(":", 1) if ":" in c else (None, c))
        name, value = spec.split("=", 1)
        for b in boards:
            if target is None or target == f"acm{b}":
                out[b][name.strip()] = float(value)
    return out


def _consistent(entries: list[dict], label: str, warnings: list[str]) -> dict[str, float]:
    """theta of the first entry; a warning for any parameter that differs between the entries."""
    first = theta_of(entries[0]["meta"])
    for e in entries[1:]:
        other = theta_of(e["meta"])
        diff = [k for k in first if k in other and abs(first[k] - other[k]) > 1e-6]
        if diff:
            warnings.append(f"{label}: {e['name']} differs from {entries[0]['name']} in {diff[:5]}")
    return first


def manifest_from_record(record: dict, run_dir: Path, metas: list[dict]) -> dict:
    """The packet manifest from a manual_packet.sh record and the run's .meta files (see the module docstring)."""
    pid = record["packet_id"]
    warnings: list[str] = []
    ref = [e for e in metas if e["name"].startswith(record["trace_prefix_ref"]) and TRACE_RE.search(e["name"])]
    new = [e for e in metas if e["name"].startswith(record["trace_prefix_new"]) and TRACE_RE.search(e["name"])]
    if not new:
        raise SystemExit(f"{pid}: no traces named {record['trace_prefix_new']}* in {run_dir}")
    boards = sorted({e["board"] for e in ref + new})

    def repeat_of(e, prefix):
        m = re.match(re.escape(prefix) + r"(\d+)_", e["name"])
        return int(m.group(1)) if m else 1

    manifest = {"manifest_version": 3, "packet_id": pid, "scan_name": record.get("scan_name"), "status": "ok",
                "source": "manual_packet.sh", "record": record, "run_directories": {b: str(run_dir) for b in boards},
                "reference_traces": [trace_entry(e, repeat_of(e, record["trace_prefix_ref"])) for e in ref],
                "traces": [trace_entry(e, repeat_of(e, record["trace_prefix_new"])) for e in new],
                "images": [], "theta": {}, "theta_ref": {}, "delta_theta": {}, "theta_id": {}, "reference_id": {},
                "geometry": {}, "warnings": warnings}
    # images: per board, the first image after this packet's last trace and before any other packet's trace
    others = [e for e in metas if not e["is_image"] and TRACE_RE.search(e["name"]) and not e["name"].startswith(pid + "_")]
    for b in boards:
        last = max(e["end"] or e["start"] for e in new if e["board"] == b)
        next_other = min((e["start"] for e in others if e["board"] == b and e["start"] > last), default=None)
        cands = sorted((e for e in metas if e["is_image"] and e["board"] == b and e["start"] > last
                        and (next_other is None or e["start"] < next_other)), key=lambda e: e["start"])
        if not cands:
            warnings.append(f"board {b}: no image found after the packet's last trace")
            continue
        img = cands[0]
        manifest["images"].append({"board": b, "stem": str(img["stem"]),
                                   "fits": [str(p) for p in sorted(run_dir.glob(f"avg_{img['name']}.fits*"))]})
        gap = (img["start"] - last).total_seconds()
        if gap > 1800:
            warnings.append(f"board {b}: the image started {gap / 60:.0f} min after the last trace")
        manifest["geometry"][b] = {"image": {k: img["meta"].get(k) for k in GEOMETRY_KEYS},
                                   "trace": {k: new[0]["meta"].get(k) for k in GEOMETRY_KEYS}}
        tr = new[0]["meta"]
        manifest.setdefault("trace_matches_image", {})[b] = all(
            tr.get(k) == img["meta"].get(k) for k in ("NDCM", "NPBIN", "NSBIN"))
    # theta from the .meta files
    requested = parse_changes(record.get("changes", []), boards)
    for b in boards:
        nb = [e for e in new if e["board"] == b]
        rb = [e for e in ref if e["board"] == b]
        th = _consistent(nb, f"board {b} traces at theta", warnings)
        manifest["theta"][b] = th
        manifest["theta_id"][b] = config_id(th)
        if rb:
            tr_ = _consistent(rb, f"board {b} reference traces", warnings)
            manifest["theta_ref"][b] = tr_
            manifest["reference_id"][b] = config_id(tr_)
            manifest["delta_theta"][b] = {k: th[k] - tr_[k] for k in th if k in tr_}
        else:
            warnings.append(f"board {b}: no reference traces")
        for k, v in requested.get(b, {}).items():
            if k not in th:
                warnings.append(f"board {b}: requested {k}={v:g} but {k} is not in the .meta files")
            elif abs(th[k] - v) > (0.5 if k.startswith("delay") else BIAS_TOL if BIAS_RE.match(k) else VOLT_TOL):
                warnings.append(f"board {b}: requested {k}={v:g} but the traces report {th[k]:g}")
                manifest["status"] = "check"
        img = next((im for im in manifest["images"] if im["board"] == b), None)
        if img is not None:
            it = theta_of(json.loads(Path(img["stem"] + ".meta").read_text()))
            diff = [k for k in th if k in it and abs(th[k] - it[k]) > 1e-6]
            if diff:
                warnings.append(f"board {b}: the image's settings differ from the traces' in {diff[:5]}")
                manifest["status"] = "check"
    starts = [e["start"] for e in ref + new]
    manifest["date_start"] = min(starts).isoformat()
    manifest["date_end"] = max(e["end"] or e["start"] for e in new).isoformat()
    manifest["seconds_since_previous"] = None
    if not manifest["images"]:
        manifest["status"] = "no image"
    return manifest


def manifests_from_legacy_run(run_dir: Path, board: int, reference_group: int = 1) -> list[dict]:
    """Older run directories: consecutive files of one board with identical theta form one packet."""
    metas = sorted((e for e in read_metas(run_dir) if e["board"] == board), key=lambda e: e["start"])
    groups, sig = [], None
    for e in metas:
        s = json.dumps(theta_of(e["meta"]), sort_keys=True)
        if s != sig:
            groups.append([])
            sig = s
        groups[-1].append(e)
    ref_group = groups[reference_group - 1]
    ref_traces = [e for e in ref_group if not e["is_image"] and TRACE_RE.search(e["name"])]
    out = []
    for g, files in enumerate(groups, start=1):
        traces = [e for e in files if not e["is_image"] and TRACE_RE.search(e["name"])]
        images = [e for e in files if e["is_image"]]
        if not traces or not images:
            continue
        pid = f"legacy_{run_dir.name}_b{board}_g{g:02d}"
        counts: dict[int, int] = {}

        def entries(es):
            counts.clear()
            res = []
            for e in es:
                ch = int(TRACE_RE.search(e["name"]).group(1))
                counts[ch] = counts.get(ch, 0) + 1
                res.append(trace_entry(e, counts[ch]))
            return res
        th, tr_ = theta_of(traces[0]["meta"]), theta_of(ref_traces[0]["meta"])
        out.append({"manifest_version": 3, "packet_id": pid, "scan_name": f"legacy {run_dir.name}", "status": "ok",
                    "source": "legacy run directory", "run_directories": {board: str(run_dir)},
                    "traces": entries(traces), "reference_traces": entries(ref_traces),
                    "images": [{"board": board, "stem": str(images[0]["stem"]),
                                "fits": [str(p) for p in sorted(run_dir.glob(f"avg_{images[0]['name']}.fits*"))]}],
                    "theta": {board: th}, "theta_ref": {board: tr_}, "theta_id": {board: config_id(th)},
                    "reference_id": {board: config_id(tr_)}, "delta_theta": {board: {k: th[k] - tr_[k] for k in th if k in tr_}},
                    "geometry": {board: {"image": {k: images[0]["meta"].get(k) for k in GEOMETRY_KEYS},
                                         "trace": {k: traces[0]["meta"].get(k) for k in GEOMETRY_KEYS}}},
                    "trace_matches_image": {board: all(traces[0]["meta"].get(k) == images[0]["meta"].get(k) for k in ("NDCM", "NPBIN", "NSBIN"))},
                    "date_start": traces[0]["start"].isoformat(), "date_end": images[0]["start"].isoformat(),
                    "seconds_since_previous": None,
                    "warnings": [f"reference traces are those of group {reference_group}, not taken in this packet"] if g != reference_group else []})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("records", nargs="*", help="packet record(s) written by manual_packet.sh")
    parser.add_argument("--run-dir", help="directory with the raw files (.meta, .RJ45ADCDATA*, avg_*.fits.bz2)")
    parser.add_argument("--legacy-run", help="an older run directory without records")
    parser.add_argument("--board", type=int, default=101, help="board for --legacy-run")
    parser.add_argument("--reference-group", type=int, default=1, help="settings group used as reference (--legacy-run)")
    parser.add_argument("--out", default="manifests", help="directory for the manifests")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.legacy_run:
        manifests = manifests_from_legacy_run(Path(args.legacy_run).resolve(), args.board, args.reference_group)
    else:
        if not args.run_dir:
            parser.error("--run-dir is needed with records")
        run_dir = Path(args.run_dir).resolve()
        metas = read_metas(run_dir)
        manifests = [manifest_from_record(json.loads(Path(r).read_text()), run_dir, metas) for r in args.records]
    for m in manifests:
        path = out / f"{m['packet_id']}.json"
        path.write_text(json.dumps(m, indent=1, default=str))
        n_t, n_r = len(m["traces"]), len(m["reference_traces"])
        print(f"{path}: status {m['status']}, {n_r} reference + {n_t} traces, images on boards "
              f"{[im['board'] for im in m['images']]}")
        for w in m["warnings"]:
            print("   warning:", w)


if __name__ == "__main__":
    main()
