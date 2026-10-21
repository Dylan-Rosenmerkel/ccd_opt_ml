"""
make_figures.py -- the figures of feature_tiers_2026-10-21.tex.

* fig_top_new_features, fig_start_amplitude: from lvl1/feature_study/feature_table.csv and
  docs/slides/data/oct12_trace_features.csv (the 59 real traces of 2026-10-12);
* fig_tolerance: from lvl1/feature_study/tolerance_results.csv (synthetic tolerance test);
* fig_learned_tiers: the counts of the five lvl1/feature_importance.py runs (synthetic, 20 noise columns added), typed in
  below because the run outputs are not kept in the repository;
* nb_*.png: copied out of the executed notebook lvl1/new_trace_features.ipynb.

    .venv/bin/python docs/overleaf_feature_tiers_2026-10-21/make_figures.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "figures"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

#: notebook cell -> figure name
NOTEBOOK = {3: "nb_differentiator", 7: "nb_null_cds", 9: "nb_jumps", 11: "nb_allan", 15: "nb_noise_budget"}

#: learned tiers per synthetic dataset: (noise columns in L1, noise columns in L2, real features in L1), of 20 noise columns
LEARNED = {"td41_s0": (0, 6, 5), "td41_s1": (0, 6, 4), "td41_s2": (0, 5, 8),          # the margin was chosen on these
           "td100_s0": (0, 4, 3), "td13_s0": (0, 8, 5)}                               # unseen confirmation sets


def top_new_features():
    ft = pd.read_csv(ROOT / "lvl1" / "feature_study" / "feature_table.csv", index_col=0)
    new = ft[ft["implementation tier"] > 0].sort_values("informativeness", ascending=False).head(16)[::-1]
    fig, ax = plt.subplots(figsize=(13, 8))
    y = np.arange(len(new))
    ax.barh(y + 0.2, np.log10(1 + new["settings F min"].clip(lower=0)), 0.4, color="#c53030",
            label="separates SETTINGS (settings F)")
    ax.barh(y - 0.2, np.log10(1 + new["amplifier F median"].clip(lower=0)), 0.4, color="#2b6cb0",
            label="separates AMPLIFIERS (amplifier F)")
    ax.set_yticks(y, [f"{n}  [{t}]" for n, t in zip(new.index, new["evidence subtier"])], fontsize=10)
    ax.axvline(np.log10(11), ls=":", c="k")
    ax.text(np.log10(11) + 0.03, len(new) - 0.6, "F = 10", fontsize=10)
    ax.set_xlabel("log10(1 + F)   (59 real traces of 2026-10-12; [evidence tier])")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "fig_top_new_features.png", dpi=130)
    plt.close()


def start_amplitude():
    d = pd.read_csv(ROOT / "docs" / "slides" / "data" / "oct12_trace_features.csv")
    order = ["reference", "VDD −20.5 V", "V low rails + TG", "all H low rails", "H3_A_L only"]
    marker = {0: "o", 1: "s", 2: "^", 3: "D"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, g in enumerate(order):
        for ch in range(4):
            v = d[(d.group == g) & (d.channel == ch)]["start: transient amplitude (ADU)"]
            ax.plot(np.full(len(v), i) + (ch - 1.5) * 0.12, v, marker[ch], ms=8,
                    c="#c53030" if "H low" in g else "#4a5568", label=f"ch{ch}" if i == 0 else None)
    ax.set_xticks(range(5), ["reference", "VDD −20.5 V", "V rails + TG", "all H low rails", "H3 low only"], rotation=15)
    ax.set_ylabel("start: transient amplitude [ADU]")
    ax.legend(title="amplifier", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig_start_amplitude.png", dpi=130)
    plt.close()


def tolerance():
    tol = pd.read_csv(ROOT / "lvl1" / "feature_study" / "tolerance_results.csv")
    base = tol[tol.kind == "base"].groupby(["model", "target"])["error"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    models = ["gp", "nn", "ridge", "cnn", "physics", "physics_joint", "sets", "trees"]
    colors = ["#c53030", "#dd6b20", "#d69e2e", "#6b46c1", "#2f855a", "#68d391", "#319795", "#2b6cb0"]
    for ax, t, lab in ((axes[0], "sigma_e", "σe"), (axes[1], "lambda", "λ")):
        extra = tol[tol.kind.isin(["noise", "real"]) & (tol.target == t)].groupby(["model", "k"])["error"].mean().unstack()
        for model, c in zip(models, colors):
            ys = np.array([base[(model, t)]] + [extra.loc[model, k] for k in (20, 50, 100)])
            ax.plot([0, 20, 50, 100], ys - ys[0], "o-", c=c, label=model, lw=2)
        ax.set_xlabel("uninformative feature columns added")
        ax.set_ylabel(f"change in {lab} error [percentage points]")
        ax.set_title(lab)
    axes[0].legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig_tolerance.png", dpi=130)
    plt.close()


def learned_tiers():
    names = list(LEARNED)
    v = np.array([LEARNED[n] for n in names])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.27, v[:, 2], 0.27, color="#2f855a", label="real features in L1")
    ax.bar(x, v[:, 0], 0.27, color="#c53030", label="noise columns in L1 (of 20)")
    ax.bar(x + 0.27, v[:, 1], 0.27, color="#d69e2e", label="noise columns in L2 (of 20)")
    for xi, n in zip(x, v[:, 0]):                                    # a zero bar is invisible: label it
        ax.text(xi, n + 0.15, str(n), ha="center", color="#c53030", fontsize=11, fontweight="bold")
    ax.axvline(2.5, ls=":", c="k")
    ax.text(1, 8.6, "margin chosen here", ha="center", fontsize=10)
    ax.text(3.5, 8.6, "unseen", ha="center", fontsize=10)
    ax.set_xticks(x, names)
    ax.set_ylabel("count")
    ax.set_ylim(0, 9.5)
    ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(OUT / "fig_learned_tiers.png", dpi=130)
    plt.close()


def notebook_figures():
    nb = json.loads((ROOT / "lvl1" / "new_trace_features.ipynb").read_text())
    for i, name in NOTEBOOK.items():
        png = next(o["data"]["image/png"] for o in nb["cells"][i]["outputs"] if "image/png" in o.get("data", {}))
        (OUT / f"{name}.png").write_bytes(base64.b64decode(png))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    top_new_features()
    start_amplitude()
    tolerance()
    learned_tiers()
    notebook_figures()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
