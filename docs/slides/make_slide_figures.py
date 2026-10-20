"""
make_slide_figures.py -- the figures for the project-overview slide deck (docs/slides/OUTLINE.md).

    .venv/bin/python docs/slides/make_slide_figures.py [--only s2_drift_profile ...]

Every figure goes to docs/slides/figures/.  Inputs:
* diagrams (pipeline, run_WADERs flow, model ladder, session timeline): drawn here;
* s2_drift_profile: data/2026-10-16-PA03_PQ04_16rows image 130 as is, and after src/image_drift.py (processing's step);
* s2_chi2: docs/slides/data/drift_fit_quality.csv (run_WADERs on images 127-131, as is and drift-corrected);
* s3_label_error: lvl1/rehearsal_report/4_check_{json,run_WADERs}.md (labels of simulated packets vs the known truth);
* s3_fit_quality_table: docs/slides/data/fit_quality_130_corrected.csv (run_WADERs output for corrected image 130);
* s4_model_errors: docs/slides/data/model_errors_41packets.csv (train_lvl1.py, 3 synthetic 41-packet datasets);
* s4_saas: the λ errors of docs/context/2026-10-19_session_context.md section 4b;
* gc_*: crops of comm_scripts/docs/gain_correction_report.pdf and gain_correction_Am241.pdf (read only).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
DATA = HERE / "data"
COMM = Path.home() / "Documents" / "Software" / "comm_scripts" / "docs"
IMAGE_130 = ROOT / "data" / "2026-10-16-PA03_PQ04_16rows" / "avg_img_16x1000x500_1x1_L2_101_20261016_202914_130.fits.bz2"
sys.path.insert(0, str(ROOT / "src"))

# one colour per idea, on every slide
TRACE, IMAGE, MODEL, GREY, RED = "#2b6cb0", "#dd6b20", "#2f855a", "#718096", "#c53030"
LIGHT = {TRACE: "#ebf4ff", IMAGE: "#fffaf0", MODEL: "#f0fff4", GREY: "#f7fafc", RED: "#fff5f5"}
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, text, color, size=14, weight="normal", sub=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=LIGHT.get(color, "white"), ec=color, lw=2.5))
    if sub:
        ax.text(x, y + h * 0.14, text, ha="center", va="center", fontsize=size, weight="bold", color=color)
        ax.text(x, y - h * 0.2, sub, ha="center", va="center", fontsize=size - 3, color="#2d3748")
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, weight=weight, color=color)


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22, lw=2.2, color=color))


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------- slide 1
def s1_pipeline():
    fig, ax = canvas(15, 5.2, (0, 15), (0, 5.2))
    steps = [(1.3, "Settings", GREY, "clock timings,\nvoltages"),
             (4.1, "Trace", TRACE, "2 rows × 5 cols\n× 500 skips"),
             (6.9, "Image", IMAGE, "16 × 1000 pixels\n× 500 skips"),
             (9.7, "Labels", IMAGE, "σe, λ, gain\n(run_WADERs)"),
             (12.9, "Models", MODEL, "predict the labels\nfrom the trace")]
    for x, t, c, s in steps:
        box(ax, x, 3.5, 2.3, 1.7, t, c, size=17, sub=s)
    for (x0, *_), (x1, *_) in zip(steps[:-1], steps[1:]):
        arrow(ax, x0 + 1.2, 3.5, x1 - 1.2, 3.5)
    ax.annotate("", xy=(12.9, 2.55), xytext=(4.1, 2.55),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=TRACE, connectionstyle="arc3,rad=0.18", mutation_scale=22))
    ax.text(8.5, 1.55, "the question: does the short trace already tell us what the image will show?", ha="center",
            fontsize=15, color=TRACE, style="italic")
    ax.text(8.5, 0.6, "1 packet  =  4 amplifiers  =  4 training examples  ·  no amplifier labels given to the models",
            ha="center", fontsize=14, color="#2d3748")
    fig.savefig(OUT / "s1_pipeline.png")
    plt.close(fig)


def s1_levels():
    fig, ax = canvas(9, 4.2, (0, 9), (0, 4.2))
    levels = [("Level 1", "predict the noise", "packets taken by hand", MODEL, True),
              ("Level 2", "suggest settings", "you decide", GREY, False),
              ("Level 3", "run scans itself", "within safety limits", GREY, False)]
    for i, (name, what, how, c, now) in enumerate(levels):
        x, y = 1.6 + i * 2.9, 0.9 + i * 1.05
        box(ax, x, y, 2.75, 1.25, name, c, size=15, sub=f"{what}\n{how}")
        if now:
            ax.text(x, y - 0.95, "◀ we are here", ha="center", fontsize=14, color=MODEL, weight="bold")
    fig.savefig(OUT / "s1_levels.png")
    plt.close(fig)


# --------------------------------------------------------------------------- slide 2
def s2_drift_profile():
    import image_drift as idr
    from image_drift import analyse_channel, open_any
    corr_dir = DATA / "_tmp_drift"
    corr_dir.mkdir(parents=True, exist_ok=True)
    rep = idr.correct_image(IMAGE_130, corr_dir, "200", 0.05, 400, [0, 1, 3])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, ch in zip(axes, (0, 1)):
        for path, shift, color, label in ((IMAGE_130, 0, RED, "as read out"), (Path(rep["output"]), rep["cut"], MODEL, "after the drift step")):
            h = open_any(Path(path))
            hd = [x for x in h[1:] if getattr(x, "data", None) is not None and x.data.ndim == 2]
            na = idr.n_active(h[0].header, hd[0].data.shape[1])
            pr = analyse_channel(hd[ch].data.astype(float), na, 0.05)["profile"]
            ok = np.isfinite(pr["off"])
            ax.errorbar(pr["c"][ok] + shift, pr["off"][ok], pr["off_err"][ok], fmt="o-", ms=4, lw=1.5, color=color,
                        label=label, capsize=0)
        ax.axvspan(0, rep["cut"], color=GREY, alpha=0.12)
        ax.text(rep["cut"] / 2, -0.95, f"first {rep['cut']}\ncolumns\ncut", ha="center", fontsize=12, color=GREY)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(f"image 130, channel {ch}")
        ax.set_xlabel("column along the row (readout order)")
    axes[0].set_ylabel("0-electron level − row pedestal [ADU]")
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("The zero-electron level sags at the start of every row, then recovers", fontsize=18, y=1.02)
    fig.savefig(OUT / "s2_drift_profile.png")
    plt.close(fig)
    for p in corr_dir.glob("*"):
        p.unlink()
    corr_dir.rmdir()


def s2_chi2():
    df = pd.read_csv(DATA / "drift_fit_quality.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    chans = sorted(df["channel"].unique())
    x = np.arange(len(chans))
    for k, (version, color, dx) in enumerate((("as is", RED, -0.2), ("drift-corrected", MODEL, 0.2))):
        sub = df[df["version"] == version]
        med = [sub[sub.channel == c]["chi2ndf"].median() for c in chans]
        ax.bar(x + dx, med, 0.38, color=color, label=version)
        for i, c in enumerate(chans):
            v = sub[sub.channel == c]["chi2ndf"]
            ax.plot(np.full(len(v), x[i] + dx), v, "o", color="k", ms=4, alpha=0.6)
    ax.axhline(1, color="k", ls="--", lw=1.2)
    ax.text(-0.55, 1.05, "a good fit", ha="left", va="bottom", fontsize=12)
    ax.set_xticks(x, [f"ch{c}" for c in chans])
    ax.set_ylabel("WADERs fit χ²/ndf")
    ax.set_title(f"Fit quality, images {df['image'].min()}–{df['image'].max()} (bar: median, dots: images)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "s2_chi2.png")
    plt.close(fig)


# --------------------------------------------------------------------------- slide 3
def s3_runwaders_flow():
    fig, ax = canvas(16, 4.4, (0, 16), (0, 4.4))
    steps = [(1.4, "Image(s)", IMAGE, ".fits / list"),
             (4.6, "autoconfig", GREY, "geometry from header\ngain · noise · dark current\ndead-channel check"),
             (8.0, "1 JSON per CCD", GREY, "only the Poisson-like\n0e⁻ / 1e⁻ peaks"),
             (11.1, "panaSKImg", GREY, "all CCDs\nin parallel"),
             (14.4, "fit_quality.csv", MODEL, "gain · noise · DC\nχ²/ndf · scaled errors")]
    for x, t, c, s in steps:
        box(ax, x, 2.6, 2.7, 2.0, t, c, size=15, sub=s)
    for (x0, *_), (x1, *_) in zip(steps[:-1], steps[1:]):
        arrow(ax, x0 + 1.4, 2.6, x1 - 1.4, 2.6)
    ax.text(8.0, 0.55, "process_packet.py reads fit_quality.csv  →  labels, flagged if not fitted, error > 50 %, or χ²/ndf > 3",
            ha="center", fontsize=13.5, color=MODEL)
    ax.text(8.0, 4.15, "run_WADERs images.txt --no-clusters --autoconfig-args=--keep-dead", ha="center", fontsize=13.5,
            family="monospace", color="#2d3748")
    fig.savefig(OUT / "s3_runwaders_flow.png")
    plt.close(fig)


def _rehearsal_table(path: Path) -> pd.DataFrame:
    rows = [l for l in path.read_text().splitlines() if l.startswith("| m0")]
    head = next(l for l in path.read_text().splitlines() if l.startswith("| packet"))
    cols = [c.strip() for c in head.strip("|").split("|")]
    return pd.DataFrame([[c.strip() for c in r.strip("|").split("|")] for r in rows], columns=cols)


def s3_label_error():
    rep = ROOT / "lvl1" / "rehearsal_report"
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [("sigma_e", "σe"), ("lambda", "λ"), ("gain", "gain")]
    for k, (fname, color, dx, label) in enumerate((("4_check_json.md", RED, -0.17, "hand-written JSON"),
                                                    ("4_check_run_WADERs.md", MODEL, 0.17, "run_WADERs"))):
        df = _rehearsal_table(rep / fname)
        df = df[df["flags"] == ""]
        for i, (q, _) in enumerate(names):
            e = np.abs(df[f"{q} rel. error %"].astype(float).to_numpy())
            jitter = np.random.default_rng(i).uniform(-0.06, 0.06, len(e))
            ax.plot(i + dx + jitter, e, "o", color=color, alpha=0.55, ms=6)
            ax.plot([i + dx - 0.12, i + dx + 0.12], [np.median(e)] * 2, color=color, lw=4, label=label if i == 0 else None)
    ax.set_xticks(range(3), [n for _, n in names])
    ax.set_ylabel("|label − truth| [%]")
    ax.set_ylim(0, 45)
    ax.set_title("Labels of simulated packets vs the known truth (bar: median)")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "s3_label_error.png")
    plt.close(fig)


def s3_fit_quality_table():
    df = pd.read_csv(DATA / "fit_quality_130_corrected.csv")
    show = pd.DataFrame({"CCD": df["ext"], "gain\n[ADU/e⁻]": df["gain"].map("{:.3f}".format),
                         "noise\n[e⁻]": df["noise"].map("{:.3f}".format), "DC\n[e⁻/pix]": df["dc"].map("{:.3f}".format),
                         "χ²/ndf": df["chi2ndf"].map("{:.2f}".format)})
    fig, ax = plt.subplots(figsize=(9, 0.55 * (len(show) + 1) + 0.5))
    ax.axis("off")
    t = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(14)
    t.scale(1, 2.0)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#cbd5e0")
        if r == 0:
            cell.set_height(cell.get_height() * 1.7)
            cell.set_facecolor(LIGHT[MODEL])
            cell.set_text_props(weight="bold", color=MODEL)
    ax.set_title("fit_quality.csv  (image 130, after the drift step)", fontsize=15, family="monospace")
    fig.savefig(OUT / "s3_fit_quality_table.png")
    plt.close(fig)


# --------------------------------------------------------------------------- slide 4
def s4_model_ladder():
    fig, ax = canvas(17, 3.6, (0, 18.2), (0, 3.6))
    rungs = [("mean", "baseline", GREY), ("ridge · trees", "hand-made\nfeatures", GREY), ("GP", "features +\nerror bars", MODEL),
             ("physics (A)", "trace physics +\nlearned correction", MODEL), ("sets (E)", "each trace set\nseparately", MODEL),
             ("CNN", "learns from\nthe curves", TRACE), ("pretrained (D)", "pre-trained on\nunlabelled traces", TRACE)]
    for i, (t, s, c) in enumerate(rungs):
        x = 1.3 + i * 2.6
        box(ax, x, 1.65 + i * 0.12, 2.4, 1.55, t, c, size=14, sub=s)
    ax.annotate("", xy=(17.9, 0.25), xytext=(0.3, 0.25), arrowprops=dict(arrowstyle="-|>", lw=2, color=GREY))
    ax.text(0.3, 0.45, "simple", fontsize=12, color=GREY)
    ax.text(17.9, 0.45, "complex", fontsize=12, color=GREY, ha="right")
    fig.savefig(OUT / "s4_model_ladder.png")
    plt.close(fig)


def s4_model_errors():
    df = pd.read_csv(DATA / "model_errors_41packets.csv")
    order = ["settings", "ridge", "trees", "gp", "nn", "cnn", "pretrained", "deep_kernel", "physics", "physics_joint", "sets"]
    df = df[df["model"].isin(order + ["mean"])]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=False)
    for ax, t, name in zip(axes, ("sigma_e", "lambda"), ("read noise σe", "dark charge λ")):
        base = df[(df.model == "mean")][t].mean()
        m = df[df.model != "mean"].groupby("model")[t].agg(["mean", "min", "max"]).reindex(order).dropna()
        colors = [MODEL if k.startswith(("physics", "sets")) else (TRACE if k in ("cnn", "pretrained", "deep_kernel") else GREY) for k in m.index]
        y = np.arange(len(m))
        ax.barh(y, m["mean"], color=colors, xerr=[m["mean"] - m["min"], m["max"] - m["mean"]], ecolor="#4a5568", capsize=3)
        ax.axvline(base, color=RED, ls="--", lw=2)
        ax.text(base, len(m) - 0.4, f"  predict the average: {base:.0f} %", color=RED, fontsize=12, va="center")
        ax.set_yticks(y, m.index)
        ax.invert_yaxis()
        ax.set_xlabel("typical error [%]")
        ax.set_title(name)
        ax.set_xlim(0, base * 1.35)
    fig.suptitle("Synthetic test, 41 packets (the planned dataset): bar = mean of 3 datasets, line = range", fontsize=16, y=1.02)
    fig.savefig(OUT / "s4_model_errors.png")
    plt.close(fig)


def s4_saas():
    sizes = ["13", "41", "100"]
    before, after = [19.1, 10.9, 4.9], [15.1, 7.4, 3.9]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(3)
    ax.bar(x - 0.19, before, 0.36, color=GREY, label="before")
    ax.bar(x + 0.19, after, 0.36, color=MODEL, label="SAAS GP")
    for i in range(3):
        ax.text(x[i] - 0.19, before[i] + 0.3, f"{before[i]:.1f}", ha="center", fontsize=12)
        ax.text(x[i] + 0.19, after[i] + 0.3, f"{after[i]:.1f}", ha="center", fontsize=12, color=MODEL, weight="bold")
    ax.set_xticks(x, [f"{s} packets" for s in sizes])
    ax.set_ylabel("λ error [%]")
    ax.set_title("Physics model: λ with a SAAS GP")
    ax.legend(frameon=False)
    fig.savefig(OUT / "s4_saas.png")
    plt.close(fig)


# --------------------------------------------------------------------------- slide 5
def s5_timeline():
    fig, ax = canvas(15, 2.8, (0, 15), (0, 2.8))
    sessions = [("A", "timing only", 13, 3.6, TRACE), ("B", "one voltage group at a time", 17, 4.5, IMAGE),
                ("C", "all settings together", 11, 3.1, MODEL)]
    x = 0.3
    scale = 14.4 / 41
    for name, what, n, hours, c in sessions:
        w = n * scale
        ax.add_patch(FancyBboxPatch((x, 1.0), w - 0.12, 1.1, boxstyle="round,pad=0.02,rounding_size=0.1", fc=LIGHT[c], ec=c, lw=2.5))
        ax.text(x + w / 2, 1.75, f"Session {name}: {n} packets", ha="center", fontsize=15, weight="bold", color=c)
        ax.text(x + w / 2, 1.3, f"{what} · ~{hours} h", ha="center", fontsize=12.5, color="#2d3748")
        x += w
    ax.text(0.3, 0.45, "first lvl-1 report possible after session A", fontsize=13, color=TRACE, style="italic")
    ax.text(14.7, 0.45, "41 packets · ~11 h · ~120 usable examples", fontsize=13, ha="right", color="#2d3748")
    fig.savefig(OUT / "s5_timeline.png")
    plt.close(fig)


# --------------------------------------------------------------------------- gain-correction slide
def _crop(pdf: str, page: int, rect_frac: tuple[float, float, float, float], name: str, dpi: int = 220):
    """Render ``rect_frac`` (x0, y0, x1, y1 as fractions of the page) of a comm_scripts report page to a PNG."""
    import pymupdf
    doc = pymupdf.open(COMM / pdf)
    p = doc[page - 1]
    r = p.rect
    clip = pymupdf.Rect(r.x0 + rect_frac[0] * r.width, r.y0 + rect_frac[1] * r.height,
                        r.x0 + rect_frac[2] * r.width, r.y0 + rect_frac[3] * r.height)
    p.get_pixmap(dpi=dpi, clip=clip).save(OUT / f"{name}.png")


def gc_figures():
    _crop("gain_correction_report.pdf", 2, (0.0, 0.06, 0.5, 1.0), "gc_gain_curves")
    _crop("gain_correction_report.pdf", 7, (0.0, 0.1, 1.0, 0.98), "gc_module_alignment")
    _crop("gain_correction_report.pdf", 10, (0.0, 0.058, 1.0, 1.0), "gc_all_datasets")
    _crop("gain_correction_Am241.pdf", 2, (0.0, 0.551, 1.0, 1.0), "gc_am241")


FIGURES = {f.__name__: f for f in (s1_pipeline, s1_levels, s2_drift_profile, s2_chi2, s3_runwaders_flow, s3_label_error,
                                   s3_fit_quality_table, s4_model_ladder, s4_model_errors, s4_saas, s5_timeline, gc_figures)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", choices=list(FIGURES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.only or FIGURES:
        try:
            FIGURES[name]()
            print("made", name)
        except FileNotFoundError as exc:
            print(f"skipped {name}: {exc}")


if __name__ == "__main__":
    main()
