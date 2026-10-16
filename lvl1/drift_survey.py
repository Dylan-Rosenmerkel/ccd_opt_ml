"""
drift_survey.py -- the along-row baseline drift in every averaged image of a set of run directories, next to its settings.

The drift (lvl1/real_image_checks/2026-10-16_PA03_PQ04_16rows.md): the zero-electron level of the active region sits up
to ~0.5 ADU below the row's overscan pedestal at the start of the readout and recovers over a few hundred columns.  To
find when and with which settings it appears, this measures, per image and channel,

    offset(c) = median over rows of the zero-electron pixels (|x - row pedestal| < max(0.6, 2.5 sigma_ovs)) at column c
                minus the row's overscan median, smoothed over 15 columns, at c = 10, 100, 400 and the last active column

and reports it only where the zero-electron peak is resolved (overscan noise < 0.6 ADU), with the header settings that
could matter (board, skips, binning, NCOLPRE, BOOLCLR, NDCMPRE, EXPOSURE, FILTER, H3_A_L, VDD, date).  Images without an
overscan region (or binned so that none remains) are listed as such.

    python lvl1/drift_survey.py <dir> [<dir> ...] [--max-mb 120] [--rows 200] [--out survey.csv]
"""

from __future__ import annotations

import argparse
import bz2
import csv
import io
import sys
from pathlib import Path

import numpy as np

COLS = (10, 100, 400)
KEYS = ("ACMID", "NROW", "NCOL", "NDCM", "NPBIN", "NSBIN", "NCOLPRE", "BOOLCLR", "NDCMPRE", "EXPOSURE", "FILTER", "H3_A_L",
        "VDD1", "DATEINI")
rob = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))


def open_any(path: Path):
    from astropy.io import fits
    raw = path.read_bytes()
    return fits.open(io.BytesIO(bz2.decompress(raw) if path.suffix == ".bz2" else raw))


def image_hdus(h):
    return [x for x in h[1:] if getattr(x, "data", None) is not None and x.data.ndim == 2]


def measure(d: np.ndarray, na: int) -> dict:
    from scipy.ndimage import median_filter
    ovs = d[:, na + 8:]
    ped = np.median(ovs, 1)
    s = float(np.median([rob(r) for r in ovs]))
    act = d[:, :na] - ped[:, None]
    w = max(0.6, 2.5 * s)
    z = np.where(np.abs(act) < w, act, np.nan)
    prof = median_filter(np.nan_to_num(np.nanmedian(z, 0)), 15, mode="nearest")
    out = {"sigma_ovs": s}
    for c in COLS:
        out[f"off@{c}"] = float(prof[c]) if c < na else np.nan
    out["off@end"] = float(prof[na - 20])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--max-mb", type=float, default=120)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--out")
    args = parser.parse_args()
    files = []
    for d in args.dirs:
        d = Path(d).expanduser()
        files += [p for p in sorted(d.glob("avg_*")) if p.suffix in (".bz2", ".fits", ".fz") and p.stat().st_size < args.max_mb * 1e6]
    # .fits and .fits.bz2 copies of the same image: keep one
    seen, uniq = set(), []
    for p in files:
        key = (p.parent, p.name.split(".fits")[0].split(".fz")[0])
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    rows = []
    for p in uniq:
        try:
            h = open_any(p)
        except Exception as exc:
            print(f"{p.name}: cannot read ({exc})", file=sys.stderr)
            continue
        H = h[0].header
        hdus = image_hdus(h)
        if not hdus:
            continue
        H = H if "NCOL" in H else hdus[0].header
        settings = {k: H.get(k) for k in KEYS}
        ncol, nsbin = hdus[0].data.shape[1], int(H.get("NSBIN", 1) or 1)
        na = int((H.get("PHYSCOL", 6144) + H.get("PRESCAN", 8) - H.get("NCOLPRE", 0)) / nsbin) if "PHYSCOL" in H else None
        for i, x in enumerate(hdus):
            row = {"run": p.parent.name, "image": p.name.split(".fits")[0].split(".fz")[0], "ch": i, **settings}
            if na is None or ncol - na < 30 or na < 60:
                row["note"] = "no overscan region"
            else:
                d = x.data.astype(float)
                d = d[np.linspace(0, len(d) - 1, min(args.rows, len(d))).astype(int)]
                m = measure(d, na)
                row.update(m)
                row["note"] = "" if m["sigma_ovs"] < 0.6 else "zero peak not resolved"
            rows.append(row)
            if row["note"] == "":
                print(f"{row['run'][:28]:28s} {row['image'][-34:]:34s} ch{i} skips {settings['NDCM']} NCOLPRE {settings['NCOLPRE']} "
                      f"BOOLCLR {settings['BOOLCLR']} F {settings['FILTER']} EXP {settings['EXPOSURE']} | σ_ovs {row['sigma_ovs']:.2f} | "
                      f"offset @10/100/400/end {row['off@10']:+.2f} {row['off@100']:+.2f} {row['off@400']:+.2f} {row['off@end']:+.2f}", flush=True)
            else:
                print(f"{row['run'][:28]:28s} {row['image'][-34:]:34s} ch{i}: {row['note']}", flush=True)
    if args.out:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
