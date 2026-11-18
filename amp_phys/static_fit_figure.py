"""
static_fit_figure.py -- figure for the 1/f cross-check: (left) one static-record spectrum with its fit
S = (S_w + K/f) L(f) + S_0 and the white / 1/f parts; (right) the 1/f level K from the readout-trace window method
against K from the static spectrum, per LSM channel (y = x line; flagged channels marked).

    .venv/bin/python amp_phys/static_fit_figure.py --psd <one psd_*.csv> --out <png>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.signal import welch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_fit import model  # noqa: E402

R = Path(__file__).resolve().parent / "results/lsm_hightemp"
BLUE, GREY, ORANGE, INK = "#2e6fb7", "#6b6b6b", "#d9822b", "#333333"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--psd", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v = pd.read_csv(a.psd, usecols=["val"])["val"].to_numpy(float)
    fr, P = welch(v - v.mean(), fs=15e6, nperseg=2 ** 16)
    S = P / (0.875e-3) ** 2
    m = (fr > 2e3) & (fr < 4e6)
    edges = np.logspace(np.log10(2e3), np.log10(4e6), 60)
    idx = np.digitize(fr[m], edges)
    fb = np.array([np.median(fr[m][idx == i]) for i in np.unique(idx)])
    sb = np.array([np.median(S[m][idx == i]) for i in np.unique(idx)])
    r = least_squares(lambda p: np.log(model(p, fb)) - np.log(sb),
                      [np.log10(80), np.log10(80 * 7e4), np.log10(3e5), 1.0, np.log10(15)],
                      bounds=([0, 3, 4, 0.5, -1], [4, 9, 6.5, 4, 3]))
    lw, lk, lfc, n, l0 = r.x
    L = 1 / (1 + (fb / 10 ** lfc) ** (2 * n))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax.loglog(fr[(fr > 1e3) & (fr < 6e6)], np.sqrt(S[(fr > 1e3) & (fr < 6e6)]), color="#b9cde6", lw=0.6,
              label="static record")
    ax.loglog(fb, np.sqrt(model(r.x, fb)), color=BLUE, lw=2, label="fit")
    ax.loglog(fb, np.sqrt(10 ** lk / fb * L), color=ORANGE, lw=1.5, ls="--", label="1/f part, $K/f$")
    ax.loglog(fb, np.sqrt(10 ** lw * L), color=GREY, lw=1.5, ls=":", label="white part")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("noise at the video input (nV/$\\sqrt{\\mathrm{Hz}}$)")
    ax.set_title(f"{Path(a.psd).name.split('_2025')[0]}: clocks frozen, no CDS model", loc="left", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, which="major", lw=0.3, alpha=0.4)
    s = pd.read_csv(R / "static_fit/static_fit.csv").set_index("label")
    t = pd.read_csv(R / "traps/trap_from_trace.csv").groupby("label").median(numeric_only=True)
    Ks = s["white (nV/rtHz)"] ** 2 * s["1/f corner (kHz)"] * 1e3 / 1e6
    Kw = t["white ASD (nV/rtHz)"] ** 2 * t["1/f corner (kHz)"] * 1e3 / 1e6
    flagged = {"ACM109 ch1": "noise bursts", "ACM110 ch3": "interference", "ACM109 ch3": "fit failed"}
    lim = [3, 12]
    bx.plot(lim, lim, color=GREY, lw=1, ls="--")
    bx.text(11.3, 10.6, "equal", color=GREY, fontsize=8)
    for lab in s.index:
        if lab == "ACM109 ch3":
            continue
        fl = lab in ("ACM109 ch1", "ACM110 ch3")
        bx.plot(Ks[lab], Kw[lab], "s" if fl else "o", ms=8, color=ORANGE if fl else BLUE, mec="white", mew=1.5)
        if fl:
            left = lab == "ACM109 ch1"
            bx.annotate(f"{lab} ({flagged[lab]})", (Ks[lab], Kw[lab]), xytext=(0, 9) if left else (8, -3),
                        textcoords="offset points", fontsize=7.5, color=INK, ha="center" if left else "left")
    ok = [x for x in s.index if x not in flagged]
    rat = (Kw[ok] / Ks[ok])
    bx.set_xlim(lim)
    bx.set_ylim(lim)
    bx.set_xlabel("1/f level $K$, static spectrum ($10^6$ nV$^2$)")
    bx.set_ylabel("1/f level $K$, readout-trace windows ($10^6$ nV$^2$)")
    bx.set_title(f"same channels, two methods: ratio {rat.median():.2f} $\\pm$ {rat.std():.2f} (9 clean channels)",
                 loc="left", fontsize=9)
    bx.grid(True, lw=0.3, alpha=0.4)
    for z in (ax, bx):
        z.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=170)


if __name__ == "__main__":
    main()
