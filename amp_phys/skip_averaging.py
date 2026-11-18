"""
skip_averaging.py -- what a skipper pixel sees: the noise of the AVERAGE of N skips (N = 1 ... 512), from the per-skip
CDS series of a raw trace, within pixels (consecutive N-skip blocks of the same pixel, difference / sqrt 2: pixel
charge cancels), after removing the mean within-pixel profile (a sequence-locked drift, identical in
every pixel).  White noise falls as 1/sqrt N; slow noise (1/f, telegraph, setup wander) makes it flatten.  Also the
per-trace excess kurtosis of the consecutive-skip differences and the share of traces with telegraph episodes.

    .venv/bin/python amp_phys/skip_averaging.py --traces <npz|csv> ... --label <name> --drop 20 --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_anatomy import load, rising  # noqa: E402

NS = (1, 4, 16, 64, 128, 256, 512)


def cds_by_pixel(v, fl, drop):
    e = rising(fl, 0)
    P = int(np.median(np.diff(e)))
    s0 = int(np.median([s for k in e[:300] for s in rising(fl[k:k + P], 1)[:1]]))
    has = np.array([k + s0 + 2 < len(fl) and ((fl[k + s0 + 2] >> 1) & 1) == 1 for k in e], bool)
    c = e[has]
    c = c[c + s0 + 150 <= len(v)]
    newpix = np.r_[True, np.abs(np.diff(c) - P) > 3]
    pid = np.cumsum(newpix)
    pos = np.arange(len(c)) - np.flatnonzero(newpix)[pid - 1]
    cs = np.cumsum(np.r_[0, v])
    cds = (cs[c + s0 + 145] - cs[c + s0 + 5]) / 140 - (cs[c + 145] - cs[c + 5]) / 140
    return [cds[(pid == p) & (pos >= drop)] for p in np.unique(pid)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--drop", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for t in a.traces:
        try:
            pix = cds_by_pixel(*load(t), a.drop)
        except Exception as ex:
            print("skip", t, ex)
            continue
        # remove the deterministic within-pixel profile (same in every pixel: sequence-locked drift), per trace
        L = int(np.median([len(q) for q in pix]))
        full = [q[:L] for q in pix if len(q) >= L]
        if len(full) >= 3:
            prof = np.mean([q - q.mean() for q in full], 0)
            pix = [q[:L] - q[:L].mean() - prof for q in full]
        r = {"label": a.label, "file": str(Path(t).parent.parent.name) + "/" + Path(t).name,
             "skips/pixel": int(np.median([len(p) for p in pix]))}
        d1 = np.concatenate([np.diff(p) for p in pix if len(p) > 1])
        r["kurtosis"] = float(pd.Series(d1).kurt())
        for N in NS:
            ds = [np.diff(p[:len(p) // N * N].reshape(-1, N).mean(1)) for p in pix if len(p) >= 2 * N]
            ds = np.concatenate(ds) if ds else np.array([])
            r[f"sigma_{N} (ADU)"] = np.std(ds) / np.sqrt(2) if len(ds) >= 4 else np.nan
        rows.append(r)
    R = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "skip_averaging.csv"
    R.to_csv(p, mode="a", header=not p.exists(), index=False)
    print(a.label, len(R), "traces")


if __name__ == "__main__":
    main()
