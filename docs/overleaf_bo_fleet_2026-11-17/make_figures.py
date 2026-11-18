"""Figures for bo_fleet.tex (run from the repository root):
    .venv/bin/python docs/overleaf_bo_fleet_2026-11-17/make_figures.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE, GREY, INK, GREEN, RED = "#2e6fb7", "#d9822b", "#8a8a8a", "#333333", "#3a9a5b", "#c0392b"


def matern(a, b, ls):
    r = np.abs(a[:, None] - b[None, :]) / ls
    return (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)


def gp(xtr, ytr, xg, ls=1.0, s2=1.0, n2=0.01, mu=0.0):
    K = s2 * matern(xtr, xtr, ls) + n2 * np.eye(len(xtr))
    ks = s2 * matern(xg, xtr, ls)
    m = mu + ks @ np.linalg.solve(K, ytr - mu)
    v = s2 - np.einsum("ij,ji->i", ks, np.linalg.solve(K, ks.T))
    return m, np.sqrt(np.clip(v, 1e-12, None))


def truth(x):
    return 0.6 * np.sin(1.3 * x) + 0.25 * x - 0.1 * (x - 1) ** 2


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, lw=0.3, alpha=0.4)


def fig_gp_basics():
    """Prior samples, then the posterior after 2 and 5 measurements."""
    xg = np.linspace(-3, 3, 300)
    rng = np.random.default_rng(1)
    K = matern(xg, xg, 1.0) + 1e-8 * np.eye(len(xg))
    L = np.linalg.cholesky(K)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for _ in range(5):
        axs[0].plot(xg, L @ rng.standard_normal(len(xg)), lw=1, color=BLUE, alpha=0.6)
    axs[0].fill_between(xg, -2, 2, color=BLUE, alpha=0.08, lw=0)
    axs[0].set_title("before any data: smooth functions, ±2 sd band", loc="left", fontsize=9)
    for ax, xs in zip(axs[1:], (np.array([-1.5, 1.0]), np.array([-2.5, -1.5, 0.0, 1.0, 2.2]))):
        ys = truth(xs) + rng.normal(0, 0.05, len(xs))
        m, s = gp(xs, ys, xg)
        ax.plot(xg, truth(xg), color=GREY, ls="--", lw=1, label="unknown true response")
        ax.plot(xg, m, color=BLUE, lw=2, label="GP mean")
        ax.fill_between(xg, m - 2 * s, m + 2 * s, color=BLUE, alpha=0.15, lw=0, label="± 2 sd")
        ax.plot(xs, ys, "o", color=ORANGE, mec=INK, ms=7, label="measurements")
        ax.set_title(f"after {len(xs)} measurements", loc="left", fontsize=9)
    axs[2].legend(fontsize=7, frameon=False, loc="lower right")
    for ax in axs:
        style(ax)
        ax.set_xlabel("setting x (one voltage, standardised)")
    axs[0].set_ylabel("objective J")
    fig.tight_layout()
    fig.savefig(OUT / "gp_basics.png", dpi=150)


def fig_lengthscale():
    xg = np.linspace(-3, 3, 300)
    xs = np.array([-2.5, -1.5, 0.0, 1.0, 2.2])
    ys = truth(xs)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for ax, ls in zip(axs, (0.3, 1.0, 4.0)):
        m, s = gp(xs, ys, xg, ls=ls)
        ax.plot(xg, m, color=BLUE, lw=2)
        ax.fill_between(xg, m - 2 * s, m + 2 * s, color=BLUE, alpha=0.15, lw=0)
        ax.plot(xs, ys, "o", color=ORANGE, mec=INK, ms=7)
        ax.set_title(f"length scale ℓ = {ls}", loc="left", fontsize=9)
        style(ax)
        ax.set_xlabel("setting x")
    axs[0].set_ylabel("objective J")
    fig.tight_layout()
    fig.savefig(OUT / "lengthscale.png", dpi=150)


def fig_ei():
    xg = np.linspace(-3, 3, 400)
    xs = np.array([-2.5, -0.8, 1.8])
    ys = truth(xs) - 1.5
    m, s = gp(xs, ys - 0, xg, mu=ys.mean())
    best = ys.min()
    z = (best - m) / s
    ei = (best - m) * norm.cdf(z) + s * norm.pdf(z)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a.plot(xg, m, color=BLUE, lw=2, label="predicted mean m(x)")
    a.fill_between(xg, m - 2 * s, m + 2 * s, color=BLUE, alpha=0.15, lw=0, label="± 2 sd")
    a.plot(xs, ys, "o", color=ORANGE, mec=INK, ms=7, label="images so far")
    a.axhline(best, color=GREEN, lw=1, ls="--", label="best so far b")
    i = int(np.argmax(ei))
    a.axvline(xg[i], color=RED, lw=1, ls=":")
    a.set_ylabel("objective J (lower is better)")
    a.legend(fontsize=7, frameon=False, loc="upper left")
    b.plot(xg, ei, color=RED, lw=2)
    b.axvline(xg[i], color=RED, lw=1, ls=":")
    b.text(xg[i] + 0.05, ei.max() * 0.85, "next image", color=RED, fontsize=8)
    b.set_ylabel("expected improvement")
    b.set_xlabel("setting x")
    for ax in (a, b):
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "expected_improvement.png", dpi=150)


def fig_search_curves():
    """Gap to the best usable setting against the number of images, nothing known (no nominal image)."""
    names = {"random order": ("random order", GREY, ":"),
             "expert: hill-climb from a random start": ("person: hill-climb from a random start", GREY, "--"),
             "TRACE-DRIVEN: rank by trace noise + fleet dark charge, gate, image to confirm": ("trace-driven", GREEN, "-"),
             "BO, fleet as prior (multi-task GP) + validity GATE (P<0.5 never imaged)": ("fleet BO + hard gate", BLUE, "-"),
             "HYBRID: trace-driven first image, then fleet BO + gate": ("hybrid", ORANGE, "-")}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (w, title) in zip(axs, (("1.0", "noise only (w = 1)"), ("0.5", "noise + dark charge (w = 0.5)"),
                                     ("0.0", "dark charge only (w = 0)"))):
        r = json.loads((ROOT / f"processed_v3n/lvl1_results/fleet_tests_v3_hybrid_none_w{w}/bo.json").read_text())
        for k, (lab, col, ls) in names.items():
            c = np.array(r["curves"][w][k])[:15]
            ax.plot(np.arange(1, len(c) + 1), np.clip(c, 0.05, None), color=col, ls=ls, lw=2 if ls == "-" else 1.3, label=lab)
        ax.axhline(2, color=INK, lw=0.8, ls=":")
        ax.text(14.8, 2.2, "2 %", ha="right", fontsize=7, color=INK)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("number of images")
        style(ax)
    axs[0].set_ylabel("best found so far: % above the best usable setting")
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "search_curves.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_gp_basics()
    fig_lengthscale()
    fig_ei()
    fig_search_curves()
    print("figures written")
