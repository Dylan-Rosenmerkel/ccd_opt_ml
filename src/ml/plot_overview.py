"""
plot_overview.py -- the schematic figures of docs/ml_models.tex (no data): the two model roles and the loop.

    python -m ml.plot_overview OUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, text, color, size=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", fc=color, ec="0.3", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, wrap=True)


def arrow(ax, x0, y0, x1, y1, text="", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.3, color="0.25",
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.02, text, ha="center", va="bottom", fontsize=8.5, color="0.2")


def roles(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    inp, mod, outc = "#e8f1f2", "#f6d6c8", "#e9f5db"
    # predictor
    ax.text(0.02, 0.96, "A. Predictor: \"what would the image say?\" (after the traces are taken)", fontsize=12, weight="bold")
    box(ax, 0.02, 0.73, 0.19, 0.14, "settings $\\theta$\n(what we set)", inp)
    box(ax, 0.27, 0.73, 0.19, 0.14, "reference traces\n(amplifier and\nCCD state now)", inp)
    box(ax, 0.52, 0.73, 0.19, 0.14, "traces at $\\theta$\n85 features\nand/or curves", inp)
    box(ax, 0.82, 0.71, 0.24, 0.18, "model P\n(trained on packets\nwith images)", mod, 11, "bold")
    box(ax, 1.17, 0.71, 0.3, 0.18, "$\\sigma_e$, $\\lambda$, gain\neach with an uncertainty\n(\"virtual image\", ~1 min)", outc, 10)
    arrow(ax, 0.115, 0.73, 0.82, 0.74, rad=0.12)
    arrow(ax, 0.365, 0.73, 0.82, 0.76, rad=0.10)
    arrow(ax, 0.73, 0.80, 0.82, 0.80)
    arrow(ax, 1.06, 0.80, 1.17, 0.80)
    # surrogate + BO
    ax.text(0.02, 0.56, "B. Surrogate + Bayesian optimization: \"which settings next?\" (before anything is taken)", fontsize=12, weight="bold")
    box(ax, 0.02, 0.30, 0.22, 0.16, "all packets so far:\n$\\theta_i$ and their measured\n$J_i = \\log\\sigma_e + w\\log\\lambda$", inp)
    box(ax, 0.32, 0.30, 0.22, 0.16, "model S: $\\theta \\rightarrow J$\nmean $\\mu(\\theta)$ and\nuncertainty $s(\\theta)$", mod, 10.5, "bold")
    box(ax, 0.62, 0.30, 0.24, 0.16, "expected improvement\nover many allowed $\\theta$\n(limits, relations,\nsmall steps)", mod, 10)
    box(ax, 0.94, 0.30, 0.22, 0.16, "next $\\theta$\n(checked again by\ntake_packet limits)", outc, 10.5)
    box(ax, 1.24, 0.30, 0.24, 0.16, "take the packet:\ntraces + image\n(~16 min)", outc, 10.5)
    arrow(ax, 0.24, 0.38, 0.32, 0.38)
    arrow(ax, 0.54, 0.38, 0.62, 0.38)
    arrow(ax, 0.86, 0.38, 0.94, 0.38)
    arrow(ax, 1.16, 0.38, 1.24, 0.38)
    ax.add_patch(FancyArrowPatch((1.36, 0.30), (0.13, 0.30), arrowstyle="-|>", mutation_scale=14, lw=1.3, color="0.25",
                                 connectionstyle="arc3,rad=-0.12"))
    ax.text(0.75, 0.13, "new labels are added; repeat until the objective stops improving", ha="center", fontsize=9.5, color="0.2")
    ax.text(0.75, 0.62, "P can replace some images: traces at many settings, images only where P is unsure (multi-fidelity)",
            ha="center", fontsize=9.5, style="italic", color="#7a3b1e")
    fig.savefig(out / "ml_roles.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    roles(out)
