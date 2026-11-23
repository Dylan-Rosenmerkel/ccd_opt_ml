"""
static_psd.py -- spectrum of a STATIC noise record (LSM ccd-cdaq psd_ch*_ACM_*.csv: every clock frozen, no waveform to
remove): band ASDs at the video input (G ADU/uV), the white floor 1-7.5 MHz, the 1/f slope below 10 kHz, and lines.

    .venv/bin/python amp_phys/static_psd.py --files <psd_*.csv> ... --out amp_phys/results/lsm_hightemp/static_psd
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import welch  # noqa: E402

FS = 15e6
BANDS = ((200, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 7.5e6))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--gain-adu-per-uv", type=float, default=0.875)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for f in sorted(a.files):
        d = pd.read_csv(f)
        clk = d.drop(columns=["clk", "val"]).to_numpy()
        assert (np.diff(clk, axis=0) == 0).all(), f"{f}: clocks not static"
        v = d["val"].to_numpy(float)
        fr, P = welch(v - v.mean(), fs=FS, nperseg=2 ** 16, window="hann")
        asd = np.sqrt(P) / a.gain_adu_per_uv * 1e3                         # nV/rtHz at the video input
        m = re.search(r"psd_ch(\d)_(\d+)_", Path(f).name)
        r = {"board": int(m.group(2)), "channel": int(m.group(1)), "duration (ms)": len(v) / FS * 1e3,
             "rms (ADU)": v.std()}
        for lo, hi in BANDS:
            r[f"ASD {lo:g}-{hi:g} Hz (nV/rtHz)"] = np.median(asd[(fr >= lo) & (fr < hi)])
        k = (fr > 300) & (fr < 1e4)
        r["slope 0.3-10 kHz (ASD ~ f^x)"] = np.polyfit(np.log(fr[k]), np.log(asd[k]), 1)[0]
        med = pd.Series(asd).rolling(41, center=True, min_periods=1).median().to_numpy()
        ex = asd / med
        pk = [i for i in np.argsort(-ex) if fr[i] > 500][:6]
        r["lines (kHz)"] = ", ".join(f"{fr[i] / 1e3:.1f}" for i in sorted(pk))
        rows.append(r)
        ax.loglog(fr[1:], asd[1:], lw=0.7, label=f"ACM{r['board']} ch{r['channel']}")
    ax.axhline(17.1, color="k", ls=":", lw=1)
    ax.text(3e2, 19, "frontend Johnson floor 17.1 nV/rtHz", fontsize=7)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("ASD at the video input (nV/rtHz)")
    ax.set_title("Static noise records (all clocks frozen)", loc="left", fontsize=9)
    ax.legend(fontsize=6, ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "static_psd.png", dpi=150)
    R = pd.DataFrame(rows).sort_values(["board", "channel"])
    R.to_csv(out / "static_psd.csv", index=False)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
