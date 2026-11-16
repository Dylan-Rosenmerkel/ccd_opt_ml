"""
trace_electrons.py -- count single electrons INSIDE the 0.2 s traces: each trace holds 10 pixels x 500 skips, enough to
resolve one electron (pixel-mean noise ~0.2 e-).  A Poisson x Gaussian mixture over the pixel means gives, from traces
alone (no image):
  * the GAIN (ADU per electron) = the spacing of the electron peaks,
  * the charge OCCUPANCY of the trace pixels (lambda_trace, e-/pixel): the charge collected in the few ms the trace
    pixels spend in the register, i.e. mostly clock-induced and serial-register charge,
  * the pixel-mean NOISE, in electrons.
Per amplifier the fit pools all packets; per packet lambda_trace is refitted with the amplifier's gain and noise fixed
and compared with the image's lambda.

Pixel means: skips >= --skip0 (transient removed), a per-skip common mode (median of the 10 pixels) removed, then the
per-pixel-position pattern (median over traces) removed.  The mixture: n ~ Poisson(lam), x = off + g n + N(0, s^2),
plus a broad component (fraction b, width 3 s) for pixels hit by drift or transients.

    .venv/bin/python amp_phys/trace_electrons.py --data processed_v3n --out amp_phys/results/v3n
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import norm, poisson, spearmanr  # noqa: E402

NMAX = 12


def pixel_means(data: Path, amp: str, skip0: int):
    rows = []
    for f in sorted(glob.glob(str(data / "N3_*" / "traces" / f"*_{amp}.npz"))):
        z = np.load(f, allow_pickle=True)
        if "cds" not in z.files:
            continue
        c = z["cds"].astype(float)[:, skip0:]
        c = c - np.median(c, 0, keepdims=True)
        kind = re.sub(r"\d+$", "", Path(f).name.split("_")[0])
        rows.append((Path(f).parts[-3], kind, c.mean(1)))
    M = np.array([r[2] for r in rows])
    M = M - np.median(M, 0, keepdims=True)
    return [r[0] for r in rows], [r[1] for r in rows], M


def nll(p, x, fixed=None):
    off, lg, llam, ls, lb = p if fixed is None else (p[0], fixed[0], p[1], fixed[1], p[2])
    g, lam, s, b = np.exp(lg), np.exp(llam), np.exp(ls), 1 / (1 + np.exp(-lb))
    n = np.arange(NMAX)
    w = poisson.pmf(n, lam)
    core = (w[None, :] * norm.pdf(x[:, None], off + g * n[None, :], s)).sum(1)
    broad = (w[None, :] * norm.pdf(x[:, None], off + g * n[None, :], 3 * s)).sum(1)
    return -np.log((1 - b) * core + b * broad + 1e-300).sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--skip0", type=int, default=20)
    ap.add_argument("--amps", nargs="+", default=["acm101_ch0", "acm101_ch1", "acm101_ch2", "acm101_ch3",
                                                   "acm102_ch2", "acm102_ch3"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    data = Path(args.data)
    d = pd.read_csv(data / "dataset.csv", low_memory=False)
    res, per_packet = [], []
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, amp in zip(axs.ravel(), args.amps):
        pk, kind, M = pixel_means(data, amp, args.skip0)
        x = M.ravel()
        x = x[np.abs(x) < 15]
        best = None
        for g0 in (0.9, 1.2, 1.5):
            for l0 in (0.1, 0.5):
                r = minimize(nll, [0.0, np.log(g0), np.log(l0), np.log(0.25), -2.0], args=(x,), method="Nelder-Mead",
                             options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-3})
                if best is None or r.fun < best.fun:
                    best = r
        off, g, lam, s, b = best.x[0], *np.exp(best.x[1:4]), 1 / (1 + np.exp(-best.x[4]))
        # bootstrap over traces for the gain error
        gs = []
        rng = np.random.default_rng(0)
        for _ in range(30):
            Mi = M[rng.integers(0, len(M), len(M))].ravel()
            Mi = Mi[np.abs(Mi) < 15]
            ri = minimize(nll, best.x, args=(Mi,), method="Nelder-Mead", options={"maxiter": 1500})
            gs.append(np.exp(ri.x[1]))
        board, ch = amp[3:6], int(amp[-1])
        sel = (d.board == int(board)) & (d.channel == ch)
        img_gain = pd.to_numeric(d.loc[sel, "label.gain_adu_per_e"], errors="coerce").median()
        res.append({"amp": f"{board} ch{ch}", "pixels": len(x), "trace gain (ADU/e)": g, "trace gain err": np.std(gs),
                    "image gain (ADU/e)": img_gain, "trace/image": g / img_gain, "pixel noise (e)": s / g,
                    "trace occupancy (e/pix)": lam, "broad fraction": b, "offset (e)": off / g})
        # per packet: lambda only (gain, noise fixed)
        for p in sorted(set(pk)):
            xi = M[[i for i, q in enumerate(pk) if q == p]].ravel()
            ri = minimize(lambda q: nll(q, xi, fixed=(best.x[1], best.x[3])), [best.x[0], best.x[2], best.x[4]],
                          method="Nelder-Mead", options={"maxiter": 800})
            row = d[sel & (d.packet_id == p)]
            per_packet.append({"amp": f"{board} ch{ch}", "packet": p, "trace occupancy": np.exp(ri.x[1]),
                               "image lambda": pd.to_numeric(row["label.lambda_e_per_bin"], errors="coerce").median(),
                               "image CIC": pd.to_numeric(row["label.lambda_cic"], errors="coerce").median(),
                               "image overscan": pd.to_numeric(row["label.lambda_overscan"], errors="coerce").median(),
                               "VSUB": pd.to_numeric(row["theta.VSUB"], errors="coerce").median()})
        h, e = np.histogram(x, bins=np.arange(-2, 7, 0.05))
        ax.bar(e[:-1], h, width=0.05, color="#b7b6b0", align="edge")
        xx = np.linspace(-2, 7, 600)
        n = np.arange(NMAX)
        w = poisson.pmf(n, lam)
        dens = ((1 - b) * (w[None] * norm.pdf(xx[:, None], off + g * n, s)).sum(1)
                + b * (w[None] * norm.pdf(xx[:, None], off + g * n, 3 * s)).sum(1))
        ax.plot(xx, dens * len(x) * 0.05, color="#2a78d6", lw=1.5)
        for k in range(4):
            ax.axvline(off + g * k, color="#eb6834", lw=0.8, ls=":")
        ax.set_title(f"{board} ch{ch}: gain {g:.3f} ADU/e (image {img_gain:.3f}), occupancy {lam:.2f} e/pix",
                     loc="left", fontsize=8.5)
        ax.set_xlabel("trace pixel mean (ADU)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Single electrons inside the 0.2 s traces (10 pixels x 500 skips; dotted = 0, 1, 2, 3 e-)", x=0.01,
                 ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "trace_electrons.png", dpi=150)
    R = pd.DataFrame(res).set_index("amp")
    PP = pd.DataFrame(per_packet)
    R.round(4).to_csv(out / "trace_electrons.csv")
    PP.round(4).to_csv(out / "trace_occupancy_per_packet.csv", index=False)
    L = ["# Single electrons in the traces", "", R.round(3).to_markdown(), "",
         "## Trace occupancy vs the image, per packet (Spearman over packets)", "",
         "| amp | vs image lambda | vs image CIC | vs image overscan | vs VSUB |", "|---|---|---|---|---|"]
    for a, g_ in PP.groupby("amp"):
        def sp(c):
            ok = g_[[c, "trace occupancy"]].dropna()
            return spearmanr(ok[c], ok["trace occupancy"])[0] if len(ok) > 10 else np.nan
        L.append(f"| {a} | {sp('image lambda'):+.2f} | {sp('image CIC'):+.2f} | {sp('image overscan'):+.2f} | {sp('VSUB'):+.2f} |")
    (out / "trace_electrons.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
