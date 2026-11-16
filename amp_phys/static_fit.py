"""
static_fit.py -- the output's noise spectrum fitted directly from a STATIC record (all clocks frozen; LSM psd_*.csv):
    S(f) = (S_w + K/f) * L(f) + S_0,   L(f) = 1 / (1 + (f/f_c)^(2n))   (frontend anti-alias roll-off)
over 2 kHz - 4 MHz (below 2 kHz the setup's slow pickup takes over).  No CDS model: an independent check of
trap_from_trace.py (white part, 1/f corner K/S_w) on the same channel.  Fit on log ASD (equal weight per decade).

    .venv/bin/python amp_phys/static_fit.py --files <psd_*.csv> --out <dir> [--compare <trap_from_trace.csv>]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import welch

FS = 15e6


def model(p, f):
    lw, lk, lfc, n, l0 = p
    return (10 ** lw + 10 ** lk / f) / (1 + (f / 10 ** lfc) ** (2 * n)) + 10 ** l0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--gain-adu-per-uv", type=float, default=0.875)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    a = ap.parse_args()
    rows = []
    for f in sorted(a.files):
        v = pd.read_csv(f, usecols=["val"])["val"].to_numpy(float)
        fr, P = welch(v - v.mean(), fs=FS, nperseg=2 ** 16)
        S = P / (a.gain_adu_per_uv * 1e-3) ** 2 / 1e0            # nV^2/Hz at the input
        m = (fr > 2e3) & (fr < 4e6)
        # log-spaced bins (median) so every decade counts the same
        edges = np.logspace(np.log10(2e3), np.log10(4e6), 60)
        idx = np.digitize(fr[m], edges)
        fb = np.array([np.median(fr[m][idx == i]) for i in np.unique(idx)])
        sb = np.array([np.median(S[m][idx == i]) for i in np.unique(idx)])
        p0 = [np.log10(80), np.log10(80 * 7e4), np.log10(3e5), 1.0, np.log10(15)]
        r = least_squares(lambda p: np.log(model(p, fb)) - np.log(sb), p0,
                          bounds=([0, 3, 4, 0.5, -1], [4, 9, 6.5, 4, 3]))
        lw, lk, lfc, n, l0 = r.x
        mm = re.search(r"psd_ch(\d)_(\d+)_", Path(f).name)
        rows.append({"label": f"ACM{mm.group(2)} ch{mm.group(1)}", "white (nV/rtHz)": np.sqrt(10 ** lw),
                     "1/f corner (kHz)": 10 ** lk / 10 ** lw / 1e3, "1/f ASD at 10 kHz (nV/rtHz)": np.sqrt(10 ** lk / 1e4),
                     "filter corner (kHz)": 10 ** lfc / 1e3, "filter order": n, "floor (nV/rtHz)": np.sqrt(10 ** l0),
                     "fit rms (log)": np.sqrt(np.mean(r.fun ** 2))})
    R = pd.DataFrame(rows).set_index("label")
    if a.compare:
        T = pd.read_csv(a.compare).groupby("label").median(numeric_only=True)
        R["window method: white"] = T["white ASD (nV/rtHz)"]
        R["window method: corner (kHz)"] = T["1/f corner (kHz)"]
        R["window method: residual"] = T["fit rms rel. residual"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "static_fit.csv")
    pd.set_option("display.width", 250)
    print(R.round(3).to_string())


if __name__ == "__main__":
    main()
