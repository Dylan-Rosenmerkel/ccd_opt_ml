"""
make_deck_2026_11_04.py -- equation images and new result figures for the 2026-11-04 overview deck
(https://claude.ai/artifact/KrzAfKm1qzchrDqXSdwCwG).  The deck format has no LaTeX, so every equation is a PNG
rendered with matplotlib's mathtext; the result figures read the result files of 2026-11-03/04.

    .venv/bin/python docs/slides/make_deck_2026_11_04.py [--only eq|fig]

Writes docs/slides/deck_2026-11-04/eq_*.png and fig_*.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "slides" / "deck_2026-11-04"
INK, INK2, GRID, SURFACE = "#1b2430", "#4a5462", "#e2e0d8", "#f7f6f2"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"

EQUATIONS = {
    "skipper": [r"$\sigma_{\rm ADU}^2(N)=\dfrac{\sigma_1^2}{N}+\sigma_{\rm floor}^2\qquad\sigma_e=\dfrac{\sigma_{\rm ADU}}{g}$"],
    "waders": [r"$p(x)=\sum_{k=0}^{\infty}\ \mathrm{Pois}(k;\lambda)\ \mathcal{N}\!\left(x;\ \mu+k\,g,\ \sigma^2\right)$"],
    "target": [r"$y_i=\left(\log\sigma_{e,i},\ \log\lambda_i,\ \log g_i\right)\qquad"
               r"\mathrm{error}=100\,\sqrt{\frac{1}{n}\sum_i(\hat y_i-y_i)^2}\ \%$"],
    "allan": [r"$\sigma_A^2(n)=\frac{a^2}{n}+b^2\qquad a=\sigma_1,\ \ b=\sigma_{\rm floor}$"],
    "gp": [r"$y=f(x)+\varepsilon,\quad f\sim\mathcal{GP}(0,k),\quad"
           r"k(x,x')=c\,\left(1+\sqrt{5}r+\frac{5}{3}r^2\right)e^{-\sqrt{5}r},\quad r^2=\sum_j\frac{(x_j-x'_j)^2}{\ell_j^2}$",
           r"$\hat y_*=\mathbf{k}_*^{\top}\left(K+\sigma_n^2 I\right)^{-1}\mathbf{y}\qquad"
           r"\max_{\phi}\ \log p(\mathbf{y}|\phi)=-\frac{1}{2}\mathbf{y}^{\top}K_y^{-1}\mathbf{y}-\frac{1}{2}\log|K_y|-\frac{n}{2}\log 2\pi$"],
    "priors": [r"$\mathrm{gp\_dim}:\ \ \ell_j\sim\mathrm{LogNormal}\left(\sqrt{2}+\frac{1}{2}\log d,\ \sqrt{3}\right)$",
               r"$\mathrm{SAAS}:\ \ \tau\sim\mathrm{HalfCauchy}(0.1),\quad 1/\ell_j^2\sim\mathrm{HalfCauchy}(\tau)$"],
    "trees": [r"$F_0=\bar y,\qquad F_m(x)=F_{m-1}(x)+\nu\,h_m(x),\qquad h_m\ \mathrm{fit\ to}\ \ r_i=y_i-F_{m-1}(x_i)$"],
    "nn": [r"$\mathcal{L}=\frac{1}{2}\left[\log\sigma^2(x)+\frac{(y-\mu(x))^2}{\sigma^2(x)}\right]\qquad"
           r"\hat\mu=\frac{1}{K}\sum_k\mu_k,\quad\hat\sigma^2=\frac{1}{K}\sum_k\sigma_k^2+\mathrm{Var}_k(\mu_k)$"],
    "physics": [r"$\log\hat\sigma_e\ =\ \frac{1}{2}\log\frac{w^2}{N}\ \ -\ \ \log\hat g(x)\ \ +\ \ \hat r(x)$"],
    "shuffle": [r"$S_G=100\sqrt{\frac{1}{n}\sum_i\left(\hat y_i^{(\pi_G)}-\hat y_i\right)^2}\ \%\qquad"
                r"L_G=\mathrm{err}_C^{(\pi_G)}-\mathrm{err}_C$"],
    "eta2": [r"$\eta^2=1-\frac{\sum_a\sum_{i\in a}(x_i-\bar x_a)^2}{\sum_i(x_i-\bar x)^2}$"],
    "level": [r"$\log y(\theta,a)=c_a+f(\theta,\mathrm{trace})\qquad c_a=\log y_{\rm measured}(\theta_0)-f(\theta_0)$"],
    "rows": [r"$\lambda_r=\mathrm{CIC}+\mathrm{DC}\cdot t_r\qquad t_r\ =\ \mathrm{time\ row}\ r\ \mathrm{waits\ before\ readout}$"],
}


def equations():
    plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})
    for name, lines in EQUATIONS.items():
        fig = plt.figure(figsize=(12, 0.9 * len(lines) + 0.1))
        for k, line in enumerate(lines):
            fig.text(0.5, 1 - (k + 0.5) / len(lines), line, ha="center", va="center", fontsize=26, color=INK)
        fig.savefig(OUT / f"eq_{name}.png", dpi=170, transparent=True, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)


def style(ax, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=12)


TITLES = {"sigma_e": "σe (read noise)", "lambda": "λ (dark charge)", "gain": "gain"}


def fig_fingerprint():
    t = pd.read_csv(ROOT / "lvl1" / "feature_study" / "real_AB" / "amp_fingerprint.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})
    top = t.head(14).iloc[::-1]
    col = [ORANGE if c == "ref" else BLUE for c in top.block]
    a.barh(range(len(top)), top.accuracy, color=col, height=0.7)
    a.set_yticks(range(len(top)), [f"{bl}: {f[:42]}" for bl, f in zip(top.block, top.feature)], fontsize=11, color=INK)
    a.axvline(0.25, color=INK2, linestyle="--", linewidth=1)
    a.text(0.26, -0.9, "chance (4 amplifiers)", fontsize=10, color=INK2)
    a.set_xlim(0, 1.05)
    a.set_xlabel("one feature alone names the amplifier of a held-out packet (fraction)", color=INK2, fontsize=12)
    style(a)
    a.grid(axis="x", color=GRID)
    a.set_title("The strongest amplifier fingerprints (η² ≥ 0.99)", color=INK, fontsize=14, fontweight="bold")
    bins = np.linspace(0, 1, 21)
    for blk, c, lab in (("own", BLUE, "own trace (at the packet's settings)"), ("ref", ORANGE, "reference trace (at nominal)")):
        b.hist(t.loc[t.block == blk, "eta2"], bins=bins, color=c, alpha=0.85, label=lab, histtype="stepfilled",
               edgecolor=SURFACE)
    b.axvline(0.9, color=INK, linestyle="--", linewidth=1)
    b.text(0.89, b.get_ylim()[1] * 0.55, "η² > 0.9:\n52 removed ", ha="right", fontsize=11, color=INK)
    b.set_xlabel("η²: share of the feature's variance between amplifiers", color=INK2, fontsize=12)
    style(b, "number of features")
    b.legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(0.12, 1.0))
    b.set_title("Reference features are fingerprints", color=INK, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_fingerprint.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_loao():
    s = json.loads((ROOT / "processed_AB/lvl1_results/2026-11-04_loao/summary.json").read_text())
    models = ["mean", "settings_amp_trees", "gp_dim", "trees", "nn", "pretrained", "physics_white"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, t in zip(axes, ("sigma_e", "lambda", "gain")):
        x = np.arange(len(models))
        vals = [[s.get(f"{m}|{t}", {}).get(k, np.nan) for m in models] for k in ("cv", "loao", "shape")]
        for k, (v, c, lab) in enumerate(zip(vals, (BLUE, ORANGE, GREEN),
                                            ("known amplifiers (CV)", "new amplifier", "new amplifier, after one image"))):
            ax.bar(x + (k - 1) * 0.27, v, 0.25, color=c, label=lab if t == "sigma_e" else None)
        ax.axhline(s[f"mean|{t}"]["loao"], color=INK, linestyle="--", linewidth=1)
        ax.text(len(models) - 0.5, s[f"mean|{t}"]["loao"], "average,\nnew amp", fontsize=10, color=INK, va="bottom", ha="right")
        ax.set_xticks(x, [m.replace("settings_amp_trees", "settings+amp").replace("physics_white", "physics") for m in models],
                      rotation=35, ha="right", fontsize=11)
        ax.set_title(TITLES[t], color=INK, fontsize=14, fontweight="bold")
        style(ax, "typical error (%)" if t == "sigma_e" else None)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_loao.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_no_fingerprint():
    c = json.loads((ROOT / "processed/lvl1_results/2026-11-04_no_fingerprint_eta90/comparison.json").read_text())
    ref = {"sigma_e": 10.2, "lambda": 23.2, "gain": 1.4}             # settings + amplifier (trees), session C
    models = ["physics_white", "cnn", "nn", "deep_kernel", "pretrained", "trees"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, t in zip(axes, ("sigma_e", "lambda", "gain")):
        x = np.arange(len(models))
        for k, (idx, col, lab) in enumerate(((3, BLUE, "all features"), (4, ORANGE, "without 52 fingerprints"),
                                              (5, "#a34a1f", "without 93"))):
            ax.bar(x + (k - 1) * 0.27, [c[f"{m}|{t}"]["values"][idx] for m in models], 0.25, color=col,
                   label=lab if t == "sigma_e" else None)
        ax.axhline(ref[t], color=INK, linestyle="--", linewidth=1)
        ax.text(len(models) - 0.5, ref[t], "settings + amplifier", fontsize=10, color=INK, va="bottom", ha="right")
        ax.set_xticks(x, [m.replace("physics_white", "physics") for m in models], rotation=35, ha="right", fontsize=11)
        ax.set_title(TITLES[t] + ", session C", color=INK, fontsize=14, fontweight="bold")
        style(ax, "typical error (%)" if t == "sigma_e" else None)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_no_fingerprint.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_board102():
    r = json.loads((ROOT / "processed/lvl1_results/2026-11-04_board102/results.json").read_text())
    amp = np.array(r["_test"]["amplifier"])
    Y = np.array(r["_test"]["Y"])
    pid = np.array(r["_test"]["packet_id"])
    m2 = amp == "ch2"
    nom = ("L1_01", "L1_07", "L1_14", "L1_32", "L1_43")
    models = ["mean", "settings_amp_trees", "gp_dim", "deep_kernel", "nn", "physics_white"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for i, (ax, t) in enumerate(zip(axes, ("sigma_e", "lambda", "gain"))):
        raw, cal = [], []
        for m in models:
            res = np.array(r[m]["pred"])[m2, i] - Y[m2, i]
            first = next(j for j, p in enumerate(pid[m2]) if p in nom)
            raw.append(100 * np.sqrt(np.mean(res ** 2)))
            cal.append(100 * np.sqrt(np.mean(np.delete(res - res[first], first) ** 2)))
        x = np.arange(len(models))
        ax.bar(x - 0.2, raw, 0.38, color=ORANGE, label="as predicted" if i == 0 else None)
        ax.bar(x + 0.2, cal, 0.38, color=GREEN, label="after one calibration image" if i == 0 else None)
        if t == "lambda":
            ax.set_yscale("log")
        ax.set_xticks(x, [m.replace("settings_amp_trees", "settings+amp").replace("physics_white", "physics") for m in models],
                      rotation=35, ha="right", fontsize=11)
        ax.set_title(TITLES[t], color=INK, fontsize=14, fontweight="bold")
        style(ax, "typical error (%)" if i == 0 else None)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_board102.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_no_ch2():
    p = ROOT / "processed/lvl1_results/2026-11-04_no_ch2/compare.json"
    if not p.exists():
        print("no_ch2 comparison not there yet")
        return
    c = json.loads(p.read_text())
    models = ["mean", "settings_amp_trees", "gp_dim", "nn", "pretrained", "physics_white"]
    runs = [("all_amps", BLUE, "trained with ch2"), ("no_ch2", ORANGE, "trained without ch2"),
            ("no_ch2_no_fingerprint", GREEN, "without ch2 and fingerprints")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, (key, lab) in enumerate((("c", "session C"), ("new_amp", "new amplifier"))):
        for ax, t in zip(axes[row], ("sigma_e", "lambda", "gain")):
            x = np.arange(len(models))
            for k, (run, col, rl) in enumerate(runs):
                v = [(c.get(f"{m}|{t}", {}).get(run) or {}).get(key, np.nan) for m in models]
                ax.bar(x + (k - 1) * 0.27, v, 0.25, color=col, label=rl if (row == 0 and t == "sigma_e") else None)
            ax.set_xticks(x, [m.replace("settings_amp_trees", "settings+amp").replace("physics_white", "physics")
                              for m in models], rotation=35, ha="right", fontsize=10)
            ax.set_title(f"{TITLES[t]}, {lab}", color=INK, fontsize=13, fontweight="bold")
            style(ax, "typical error (%), ch0/1/3" if t == "sigma_e" else None)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig_no_ch2.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_learning_curve():
    import collections
    agg = collections.defaultdict(list)
    for f in sorted((ROOT / "processed/lvl1_results/2026-11-04_learning_curve").glob("n*.json")):
        for k, v in json.loads(f.read_text()).items():
            if not k.startswith("_"):
                for t, e in v["errors"].items():
                    agg[(v["model"], t, v["n_packets"])].append(e)
    if not agg:
        return
    sizes = [8, 16, 24, 32]
    lines = [("mean", GREY, "average (no trace)"), ("settings_amp_trees", "#86b6ef", "settings + amplifier"),
             ("nn", ORANGE, "nn"), ("cnn", GREEN, "cnn"), ("physics_white", BLUE, "physics")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, t in zip(axes, ("sigma_e", "lambda", "gain")):
        for m, c, lab in lines:
            mu = [np.mean(agg[(m, t, n)]) for n in sizes]
            sd = [np.std(agg[(m, t, n)]) for n in sizes]
            ax.errorbar(sizes, mu, yerr=sd, color=c, marker="o", linewidth=2, capsize=3, label=lab if t == "sigma_e" else None)
        ax.set_xticks(sizes)
        ax.set_xlabel("training packets (A+B)", color=INK2, fontsize=12)
        ax.set_title(TITLES[t] + ", session C", color=INK, fontsize=14, fontweight="bold")
        style(ax, "typical error (%)" if t == "sigma_e" else None)
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "fig_learning_curve.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=("eq", "fig"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only != "fig":
        equations()
    if args.only != "eq":
        plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURFACE})
        fig_fingerprint()
        fig_loao()
        fig_no_fingerprint()
        fig_board102()
        fig_no_ch2()
        fig_learning_curve()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
