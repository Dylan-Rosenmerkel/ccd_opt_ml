"""
plot_ml_diagrams.py -- the schematic diagrams of the ML plan (docs/ml_training_plan.tex).

Writes four PNG files into OUT_DIR:

    ml_example.png     one labeled training example: settings + trace -> tool values, image -> physics values
    ml_training.png    the training loop and the held-out test
    ml_optimize.png    using the trained model to propose better settings, with calibrated images when unsure
    ml_tradeoff.png    the read-noise / dark-current trade-off the optimizer searches

These are sketches of the method, not results: no model has been trained yet.

Usage:  python plot_ml_diagrams.py OUT_DIR
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

COLORS = {"data": ("#E1F5EE", "#0F6E56"), "model": ("#EEEDFE", "#534AB7"), "plain": ("#F1EFE8", "#5F5E5A")}


def new_canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width / 80, height / 80))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)                       # y grows downward, like a page
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, subtitle="", kind="plain", dashed=False):
    face, edge = COLORS[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4", facecolor=face,
                                edgecolor=edge, linewidth=1, linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, y + h * (0.36 if subtitle else 0.5), title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=edge)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.72, subtitle, ha="center", va="center", fontsize=9, color=edge)


def arrow(ax, x0, y0, x1, y1, path=None):
    """Straight arrow, or a polyline through ``path`` (list of (x, y)) ending with an arrowhead."""
    if path is None:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color="#444", lw=1))
        return
    xs, ys = zip(*path)
    ax.plot(xs[:-1], ys[:-1], color="#444", lw=1)
    ax.add_patch(FancyArrowPatch((xs[-2], ys[-2]), (xs[-1], ys[-1]), arrowstyle="-|>", mutation_scale=11, color="#444", lw=1))


def example(out: Path) -> None:
    fig, ax = new_canvas(680, 380)
    box(ax, 40, 30, 170, 56, "Settings", "known: we set them", "model")
    box(ax, 255, 30, 170, 56, "Trace", "about one second")
    box(ax, 470, 30, 170, 56, "Calibrated image", "minutes, WADERs fit")
    arrow(ax, 340, 86, 340, 138)
    arrow(ax, 555, 86, 555, 138)
    box(ax, 255, 140, 170, 56, "Tool values", "35 numbers", "data")
    box(ax, 470, 140, 170, 56, "Physics values", "sigma_e, gain, dark rate", "data")
    arrow(ax, 125, 86, 125, 256)
    arrow(ax, 340, 196, 300, 256)
    arrow(ax, 555, 196, 515, 256)
    ax.add_patch(Rectangle((70, 236), 570, 96, fill=False, edgecolor="#888", linestyle="--", linewidth=0.8))
    box(ax, 100, 258, 290, 56, "Input: settings + tool values", "what the model sees", "data")
    box(ax, 430, 258, 170, 56, "Answer", "sigma_e, gain, dark rate", "model")
    ax.text(355, 354, "Trace and image come from the same settings, taken back to back", ha="center", fontsize=9, color="#444")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def training(out: Path) -> None:
    fig, ax = new_canvas(680, 310)
    box(ax, 50, 50, 130, 56, "Packets", "input and answer", "data")
    box(ax, 200, 50, 130, 56, "Model", "adjustable knobs", "model")
    box(ax, 350, 50, 130, 56, "Prediction", "sigma_e, gain, dark", "plain")
    box(ax, 500, 50, 130, 56, "Error", "guess vs image", "plain")
    arrow(ax, 180, 78, 198, 78)
    arrow(ax, 330, 78, 348, 78)
    arrow(ax, 480, 78, 498, 78)
    arrow(ax, 0, 0, 0, 0, path=[(565, 106), (565, 150), (285, 150), (285, 108)])
    ax.text(425, 172, "adjust the knobs to shrink the error, then repeat", ha="center", fontsize=9, color="#444")
    arrow(ax, 230, 106, 230, 226)
    box(ax, 50, 228, 580, 56, "Test on held-out settings and amplifiers", "never used to adjust the knobs: the honest score", "plain", dashed=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def optimize(out: Path) -> None:
    fig, ax = new_canvas(680, 360)
    box(ax, 40, 30, 170, 56, "New trace", "candidate settings")
    box(ax, 255, 30, 170, 56, "Tool values", "same 35 tools", "data")
    box(ax, 470, 30, 170, 56, "Trained model", "knobs now fixed", "model")
    arrow(ax, 210, 58, 253, 58)
    arrow(ax, 425, 58, 468, 58)
    arrow(ax, 555, 86, 555, 148)
    box(ax, 440, 150, 200, 56, "Predicted values", "noise, gain, dark: with error")
    arrow(ax, 440, 178, 393, 178)
    box(ax, 170, 150, 220, 56, "Optimizer", "picks better settings", "model")
    arrow(ax, 0, 0, 0, 0, path=[(170, 178), (125, 178), (125, 88)])
    ax.text(136, 122, "new settings, repeat", fontsize=9, color="#444", va="center")
    arrow(ax, 540, 206, 540, 268)
    box(ax, 300, 270, 340, 56, "Unsure? Take a calibrated image", "becomes a new training example", "plain", dashed=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def tradeoff(out: Path) -> None:
    import numpy as np
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    n = np.linspace(1, 100, 300)
    noise = 1.0 / np.sqrt(n)
    dark = 0.05 + 0.011 * n
    ax.plot(n, noise / noise.max(), color="#1D9E75", lw=2, label="read noise (falls with skips)")
    ax.plot(n, dark / dark.max(), color="#D85A30", lw=2, label="dark current (rises with readout time)")
    ax.axvspan(18, 45, color="0.85", alpha=0.7)
    ax.text(31, 0.93, "compromise", ha="center", fontsize=9)
    ax.set(xlabel="one setting (for example number of skips)", ylabel="relative value (lower is better)",
           title="The optimizer searches for settings where both are low")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    example(out_dir / "ml_example.png")
    training(out_dir / "ml_training.png")
    optimize(out_dir / "ml_optimize.png")
    tradeoff(out_dir / "ml_tradeoff.png")
    print(f"wrote 4 diagrams to {out_dir}")


if __name__ == "__main__":
    main()
