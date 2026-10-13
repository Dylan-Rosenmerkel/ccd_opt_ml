"""
check_trace_analysis.py -- test the trace analysis on synthetic traces of many geometries and timings.

For every configuration in ``CONFIGS`` a trace is written with ``simulate_trace.py`` and analysed
with ``trace_regions.split_regions`` (with and without its ``.meta``) and
``trace_features.trace_features``.  The found layout is compared with the truth:

    pixels, skips per pixel, window length, signal-window offset, exact cycle length,
    horizontal transfers per pixel, vertical transfers, row starts

and the features are checked: no feature group may fail, and the number of NaN features is
reported (NaN is expected where a feature cannot exist, e.g. 50-skip averages in a 20-skip
trace).  The real traces in ``data/`` are checked too (layout from their ``.meta``).

    python check_trace_analysis.py [--out DIR] [--only NAME ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from decode_trace import decode_trace
from simulate_trace import DEFAULTS, simulate_trace
from trace_features import trace_features
from trace_geometry import SAMPLES_PER_TICK, TraceGeometry
from trace_regions import split_regions

#: name -> parameters that differ from the reference (2x5 pixels, 500 skips, 20x20, reference timing)
CONFIGS = {
    "reference": {},
    "1 skip": dict(NDCM=1),
    "2 skips": dict(NDCM=2),
    "5 skips": dict(NDCM=5),
    "20 skips": dict(NDCM=20),
    "100 skips": dict(NDCM=100),
    "2000 skips, 1x2": dict(NDCM=2000, NROW=1, NCOL=2),
    "no binning (1x1)": dict(NSBIN=1, NPBIN=1),
    "matched to a 1x1 image (prescan 100)": dict(NSBIN=1, NPBIN=1, NCOLPRE=100, NROW=2, NCOL=5),
    "20x20 with prescan 185 (as 2026-10-12)": dict(NCOLPRE=185, NDCM=100),
    "1x10 binning": dict(NPBIN=1, NSBIN=10),
    "2x2 binning": dict(NPBIN=2, NSBIN=2),
    "1 row": dict(NROW=1, NCOL=5),
    "1 column": dict(NROW=4, NCOL=1),
    "1 pixel": dict(NROW=1, NCOL=1),
    "3x7 = 21 pixels": dict(NROW=3, NCOL=7, NDCM=100),
    "1x20 pixels": dict(NROW=1, NCOL=20, NDCM=50),
    "short windows (delayInt 200)": dict(delayInt=200),
    "long windows (delayInt 3000)": dict(delayInt=3000, NDCM=200),
    "fast H clocks (600)": dict(delay_Ha=600, delay_Hb=600, delay_Hc=600),
    "slow H clocks (6000)": dict(delay_Ha=6000, delay_Hb=6000, delay_Hc=6000, NDCM=100),
    "fast V clocks (2000)": dict(delay_Va=2000, delay_Vb=2000),
    "slow V clocks (20000)": dict(delay_Va=20000, delay_Vb=20000, NDCM=100),
    "long SW (delaySWL 1500)": dict(delaySWL=1500),
    "late reset (delayOGL 500, RGH 2000)": dict(delayOGL=500, delayRGH=2000, NDCM=200),
    "charged pixel": dict(),                      # charged_pixels set below
    "mixed": dict(NROW=3, NCOL=7, NDCM=37, NSBIN=3, NPBIN=2, delayInt=700, delay_Ha=900,
                  delay_Hb=900, delay_Hc=900, delay_Va=3000, delay_Vb=3000, delaySWL=500),
}


def truth(params: dict) -> dict:
    p = {**DEFAULTS, **params}
    g = TraceGeometry.from_meta({**p, "SEQNAME": "reference_v5_pumping.seq"})
    return {"pixels": p["NROW"] * p["NCOL"], "skips": p["NDCM"], "n_ped": round(g.n_window),
            "sig": g.sig_offset, "cycle": g.cycle, "nsbin": p["NSBIN"], "npbin": p["NPBIN"],
            "ncolpre": p.get("NCOLPRE", 0),
            "rows": [i * p["NCOL"] for i in range(p["NROW"])]}


def check_layout(r, t: dict, use_meta: bool) -> list[str]:
    """Differences between the found layout and the truth (empty list = all correct)."""
    bad = []
    if r.n_pixels != t["pixels"]:
        bad.append(f"pixels {r.n_pixels} != {t['pixels']}")
    if any(len(c) != t["skips"] for c in r.skip) or any(len(c) != t["skips"] for c in r.cds):
        bad.append(f"skips {sorted({len(c) for c in r.skip})} != {t['skips']}")
    if abs(r.n_ped - t["n_ped"]) > 1:
        bad.append(f"window {r.n_ped} != {t['n_ped']}")
    if abs(r.sig_start - t["sig"]) > 1.5:
        bad.append(f"signal offset {r.sig_start} != {t['sig']:.1f}")
    if abs(r.period_exact - t["cycle"]) > 0.05 * max(1, 500 / max(t["skips"], 1)):
        bad.append(f"cycle {r.period_exact:.3f} != {t['cycle']:.3f}")
    if r.row_starts != t["rows"]:
        bad.append(f"row starts {r.row_starts} != {t['rows']}")
    n_h = sorted({len(h) for h in r.h})
    expected_h = {t["nsbin"], t["nsbin"] - 1} if (t["nsbin"] > 1 or use_meta) else {0, 1}
    if t["ncolpre"]:                                   # row-start pixels also carry the prescan transfers
        expected_h |= {t["nsbin"] * (t["ncolpre"] + 1), t["nsbin"] * (t["ncolpre"] + 1) - 1}
    if not set(n_h) <= expected_h:
        bad.append(f"horizontal transfers per pixel {n_h}, expected {sorted(expected_h)}")
    if r.v is None and (t["npbin"] > 1 or use_meta):
        bad.append("no vertical transfers found")
    if use_meta and r.warnings:
        bad.append("warnings: " + "; ".join(r.warnings))
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", help="directory for the synthetic traces (default: a temporary one)")
    parser.add_argument("--only", nargs="*", help="run only these configurations")
    args = parser.parse_args()
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="trace_check_"))

    print(f"{'configuration':38s} {'layout (meta)':14s} {'layout (no meta)':16s} {'groups failed':14s} {'NaN':>4s} {'s':>5s}")
    all_ok = True
    for name, params in CONFIGS.items():
        if args.only and name not in args.only:
            continue
        charged = (3,) if name == "charged pixel" else ()
        path = simulate_trace(out / name.replace(" ", "_").replace("/", "-"), seed=1, charged_pixels=charged, **params)
        meta = json.loads(path.with_suffix(".meta").read_text())
        started = time.time()
        trace = decode_trace(path)
        problems = trace.quality_problems()
        t = truth(params)
        with_meta = check_layout(split_regions(trace, meta), t, True) if not problems else problems
        without = check_layout(split_regions(trace), t, False) if not problems else problems
        features = trace_features(trace, meta=meta)
        errors = features.info["errors"]
        n_nan = sum(not np.isfinite(v) for v in features.scalars.values())
        ok = not with_meta and not without and not errors
        all_ok &= ok
        print(f"{name:38s} {'ok' if not with_meta else 'WRONG':14s} {'ok' if not without else 'WRONG':16s} "
              f"{len(errors):14d} {n_nan:4d} {time.time() - started:5.1f}")
        for line in with_meta:
            print(f"      with meta: {line}")
        for line in without:
            print(f"      no meta:   {line}")
        for group, err in errors.items():
            print(f"      group {group} failed: {err}")

    print("\nreal traces (layout from their .meta):")
    data = Path(__file__).resolve().parent.parent / "data"
    for path in sorted(glob.glob(str(data / "2026-10-0[68]*" / "trace_*.RJ45ADCDATA*"))):
        if not path.endswith(path.split("_ch")[1][0]):
            continue                                       # only the channel's own data file
        trace = decode_trace(path)
        if trace.quality_problems():
            print(f"  {Path(path).name}: damaged (refused, as intended)")
            continue
        meta = json.loads(Path(path).with_suffix(".meta").read_text())
        r = split_regions(trace, meta)
        features = trace_features(trace, meta=meta)
        status = "ok" if not r.warnings and not features.info["errors"] else f"{r.warnings} {features.info['errors']}"
        print(f"  {Path(path).name}: {status}")
    print("\nALL CONFIGURATIONS OK" if all_ok else "\nSOME CONFIGURATIONS FAILED")


if __name__ == "__main__":
    main()
