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

# black and white (the deck's style): black = the result / after, light grey = before / baseline, dark grey = other
TRACE, IMAGE, MODEL, GREY, RED = "#4a4a4a", "#4a4a4a", "#111111", "#9a9a9a", "#9a9a9a"
INK = "#111111"
# data plots keep colour (diagrams stay black and white): bad/before, good/after, a third series, neutral
P_BAD, P_GOOD, P_BLUE, P_GREY = "#c53030", "#2f855a", "#2b6cb0", "#a0aec0"
LIGHT = {TRACE: "#ffffff", MODEL: "#ececec", GREY: "#ffffff"}
plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, text, color, size=14, weight="normal", sub=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=LIGHT.get(color, "white"), ec=color, lw=2.2))
    if sub:
        ax.text(x, y + h * 0.14, text, ha="center", va="center", fontsize=size, weight="bold", color=INK)
        ax.text(x, y - h * 0.2, sub, ha="center", va="center", fontsize=size - 3, color="#333333")
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, weight=weight, color=INK)


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
            ax.text(x, y - 0.95, "◀ we are here", ha="center", fontsize=14, color=INK, weight="bold")
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
        for path, shift, color, label in ((IMAGE_130, 0, P_BAD, "as read out"), (Path(rep["output"]), rep["cut"], P_GOOD, "after the drift step")):
            h = open_any(Path(path))
            hd = [x for x in h[1:] if getattr(x, "data", None) is not None and x.data.ndim == 2]
            na = idr.n_active(h[0].header, hd[0].data.shape[1])
            pr = analyse_channel(hd[ch].data.astype(float), na, 0.05)["profile"]
            ok = np.isfinite(pr["off"])
            ax.errorbar(pr["c"][ok] + shift, pr["off"][ok], pr["off_err"][ok], fmt="o-", ms=5, lw=1.5, color=color,
                        mfc=color, label=label, capsize=0)
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
    for k, (version, color, dx) in enumerate((("as is", P_BAD, -0.2), ("drift-corrected", P_GOOD, 0.2))):
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
    for k, (fname, color, dx, label) in enumerate((("4_check_json.md", P_BAD, -0.17, "hand-written JSON"),
                                                    ("4_check_run_WADERs.md", P_GOOD, 0.17, "run_WADERs"))):
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
            cell.set_facecolor("#ececec")
            cell.set_text_props(weight="bold", color=INK)
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
        colors = [P_GOOD if k.startswith(("physics", "sets")) else (P_BLUE if k in ("cnn", "pretrained", "deep_kernel") else P_GREY) for k in m.index]
        y = np.arange(len(m))
        ax.barh(y, m["mean"], color=colors, xerr=[m["mean"] - m["min"], m["max"] - m["mean"]], ecolor="#4a5568", capsize=3)
        ax.axvline(base, color=P_BAD, ls="--", lw=2)
        ax.text(base, len(m) - 0.4, f"  predict the average: {base:.0f} %", color=P_BAD, fontsize=12, va="center")
        ax.set_yticks(y, m.index)
        ax.invert_yaxis()
        ax.set_xlabel("typical error [%]")
        ax.set_title(name)
        ax.set_xlim(0, base * 1.35)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=P_GREY, label="settings / hand-made features"), Patch(color=P_BLUE, label="neural networks"),
                            Patch(color=P_GOOD, label="physics-based")], frameon=False, loc="lower right", fontsize=12)
    fig.suptitle("Synthetic test, 41 packets (the planned dataset): bar = mean of 3 datasets, line = range", fontsize=16, y=1.02)
    fig.savefig(OUT / "s4_model_errors.png")
    plt.close(fig)


def s4_saas():
    sizes = ["13", "41", "100"]
    before, after = [19.1, 10.9, 4.9], [15.1, 7.4, 3.9]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(3)
    ax.bar(x - 0.19, before, 0.36, color=P_GREY, label="before")
    ax.bar(x + 0.19, after, 0.36, color=P_GOOD, label="SAAS GP")
    for i in range(3):
        ax.text(x[i] - 0.19, before[i] + 0.3, f"{before[i]:.1f}", ha="center", fontsize=12)
        ax.text(x[i] + 0.19, after[i] + 0.3, f"{after[i]:.1f}", ha="center", fontsize=12, color=P_GOOD, weight="bold")
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
        ax.add_patch(FancyBboxPatch((x, 1.0), w - 0.12, 1.1, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec=INK, lw=2.2))
        ax.text(x + w / 2, 1.75, f"Session {name}: {n} packets", ha="center", fontsize=15, weight="bold", color=INK)
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


# --------------------------------------------------------------------------- the trace
TRACE_FILE = ROOT / "data" / "2026-10-12-PA03_PQ04_Packet_Tests" / "trace_2x5x500_20x20_ch0_101_20261012_185604_32.RJ45ADCDATA0"
PIXEL = 3          # a pixel that does not start a row


def _trace():
    import json
    from decode_trace import SAMPLE_RATE_HZ, decode_trace
    from trace_regions import split_regions
    t = decode_trace(TRACE_FILE)
    meta = json.loads(TRACE_FILE.with_suffix(".meta").read_text())
    return t, split_regions(t, meta), SAMPLE_RATE_HZ


def t_trace_anatomy():
    t, r, fs = _trace()
    v = t.values.astype(float)
    fig = plt.figure(figsize=(16, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.45, wspace=0.22)
    # (a) the whole trace: min/max envelope
    ax = fig.add_subplot(gs[0, :])
    nb = 4000
    edges = np.linspace(0, len(v), nb + 1).astype(int)
    lo = np.array([v[a:b].min() for a, b in zip(edges[:-1], edges[1:])])
    hi = np.array([v[a:b].max() for a, b in zip(edges[:-1], edges[1:])])
    tm = edges[:-1] / fs * 1e3
    for k, (a, b) in enumerate(r.skip_bounds):
        ax.axvspan(a / fs * 1e3, b / fs * 1e3, color="#e4e4e4", lw=0)
        ax.text((a + b) / 2 / fs * 1e3, 1.02, f"pixel {k + 1}", transform=ax.get_xaxis_transform(), ha="center", fontsize=11)
    if r.v_bounds:
        ax.axvspan(r.v_bounds[0] / fs * 1e3, r.v_bounds[1] / fs * 1e3, color="#b0b0b0", lw=0)
        ax.text(np.mean(r.v_bounds) / fs * 1e3, 0.05, "row\nchange", transform=ax.get_xaxis_transform(), ha="center", fontsize=11)
    ax.fill_between(tm, lo, hi, color=P_BLUE, lw=0)
    ax.set_xlim(0, len(v) / fs * 1e3)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("ADC [ADU]")
    ax.set_title("A whole trace: 2 rows × 5 pixels, each pixel read 500 times (grey bands)", pad=26)
    # (b) three skip cycles of one pixel
    ax = fig.add_subplot(gs[1, 0])
    ps, pe = t.window_slices()
    ss, se = t.signal_window_slices()
    a0, b0 = r.skip_bounds[PIXEL]
    k = np.flatnonzero(ps >= a0)[20]
    a, b = ps[k] - 40, ps[k] + 3 * r.period
    x = (np.arange(a, b) - a) / fs * 1e6
    for s_, e_ in zip(ps, pe):
        if a <= s_ < b:
            ax.axvspan((s_ - a) / fs * 1e6, (e_ - a) / fs * 1e6, color="#d0d0d0", lw=0)
    for s_, e_ in zip(ss, se):
        if a <= s_ < b:
            ax.axvspan((s_ - a) / fs * 1e6, (e_ - a) / fs * 1e6, facecolor="white", edgecolor="#555555", hatch="///", lw=0)
    ax.plot(x, v[a:b] - np.median(v[a:b]), color=P_BLUE, lw=1.4)
    ax.set_ylim(-30000, 60000)
    ax.set_xlabel("time [µs]")
    ax.set_ylabel("ADC [ADU]")
    ax.set_title("Zoom: three reads (skips) of one pixel")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#d0d0d0", label="pedestal window"), Patch(facecolor="white", edgecolor="#555555", hatch="///", label="signal window")],
              frameon=False, loc="upper left", fontsize=12)
    # (c) CDS per skip
    ax = fig.add_subplot(gs[1, 1])
    c = np.asarray(r.cds[PIXEL], float)
    ax.plot(np.arange(len(c)), c, "o", ms=3, color=P_BLUE)
    ax.axhline(np.median(c[10:]), color=GREY, ls="--", lw=1.5)
    ax.set_ylim(np.median(c) - 120, np.median(c) + 330)
    ax.annotate("first reads:\nsettling", xy=(1, c[0]), xytext=(60, c[0] - 40), fontsize=12, arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xlabel("skip number")
    ax.set_ylabel("pedestal − signal [ADU]")
    ax.set_title("One number per skip (CDS): 500 per pixel")
    fig.savefig(OUT / "t_trace_anatomy.png")
    plt.close(fig)


def t_trace_noise():
    """The trace's own noise curves, exactly as the models get them (trace_features.py)."""
    from trace_features import ALLAN_N, trace_features
    t, r, fs = _trace()
    import json
    f = trace_features(t, meta=json.loads(TRACE_FILE.with_suffix(".meta").read_text()), regions=r)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    ax = axes[0]
    ns = np.asarray(ALLAN_N, float)
    m = np.asarray(f.curves["noise: Allan deviation, common pattern removed"], float)
    ax.loglog(ns, m, "o-", color=P_BLUE, ms=7, lw=2, label="this trace")
    ax.loglog(ns, m[0] / np.sqrt(ns), "--", color=GREY, lw=2, label="pure white noise: 1/√n")
    ax.set_xlabel("skips averaged, n")
    ax.set_ylabel("Allan deviation [ADU]")
    ax.set_title("Averaging skips: how fast the noise falls")
    ax.legend(frameon=False, fontsize=12)
    ax = axes[1]
    spec = np.asarray(f.curves["noise: CDS spectrum"], float)
    fr = np.linspace(0, 15.0, len(spec))
    ax.semilogy(fr[2:], spec[2:], color=P_BLUE, lw=1.5)
    ax.set_xlabel("frequency [kHz]")
    ax.set_ylabel("power [ADU²/Hz]")
    ax.set_title("Which frequencies the noise lives at")
    fig.savefig(OUT / "t_trace_noise.png")
    plt.close(fig)


# --------------------------------------------------------------------------- processing and testing
def p_processing():
    fig, ax = canvas(19, 6.6, (0, 19), (0, 6.6))
    box(ax, 1.55, 3.3, 2.9, 1.7, "manual_packet.sh", INK, size=14, sub="on the DAQ:\nqueue the packet,\nwrite its record")
    box(ax, 4.6, 3.3, 2.6, 1.7, "manifest", INK, size=14, sub="settings read back,\nchanges checked")
    arrow(ax, 3.0, 3.3, 3.25, 3.3, INK)
    ax.add_patch(FancyBboxPatch((6.0, 0.55), 7.1, 5.4, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f6f6f6", ec=GREY, lw=1.5, ls="--"))
    ax.text(9.55, 5.65, "process_packet.py", ha="center", fontsize=15, weight="bold", family="monospace")
    ax.text(6.25, 4.95, "traces", fontsize=13, style="italic")
    ax.text(6.25, 2.15, "image", fontsize=13, style="italic")
    lanes = [(4.0, [("decode", "quality check"), ("features", "85 numbers,\n14 curves"), ("combine", "repeated sets")]),
             (1.35, [("drift step", "detect, cut,\ncorrect"), ("run_WADERs", "σe, λ, gain\n+ flags"), ("λ per row", "dark current\n+ CIC")])]
    for y, steps in lanes:
        for i, (t, s) in enumerate(steps):
            box(ax, 7.2 + i * 2.35, y, 2.05, 1.45, t, TRACE, size=13, sub=s)
            if i:
                arrow(ax, 7.2 + (i - 1) * 2.35 + 1.05, y, 7.2 + i * 2.35 - 1.05, y, INK)
    arrow(ax, 5.95, 3.6, 6.15, 4.0, INK)
    arrow(ax, 5.95, 3.0, 6.15, 1.35, INK)
    box(ax, 14.8, 4.3, 2.6, 1.5, "dataset.csv", INK, size=14, sub="one row per\namplifier")
    box(ax, 14.8, 2.1, 2.6, 1.5, "model_inputs", MODEL, size=14, sub="5 files, one per\nmodel family")
    arrow(ax, 13.15, 3.3, 13.45, 4.3, INK)
    arrow(ax, 14.8, 3.5, 14.8, 2.9, INK)
    box(ax, 17.6, 2.1, 2.3, 1.5, "train_lvl1.py", MODEL, size=14, sub="train, test,\nreport")
    arrow(ax, 16.15, 2.1, 16.4, 2.1, INK)
    fig.savefig(OUT / "p_processing.png")
    plt.close(fig)


def m_cv():
    n_pk, n_amp = 12, 4
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    for ax, title, grouped in ((axes[0], "Random split (not used): the same packet's\nother amplifiers end up in training", False),
                               (axes[1], "Packet-grouped split (used):\nwhole packets are held out", True)):
        test = np.zeros((n_amp, n_pk), bool)
        if grouped:
            test[:, 3:6] = True
        else:
            test.flat[rng.choice(n_amp * n_pk, 12, replace=False)] = True
        for i in range(n_amp):
            for j in range(n_pk):
                ax.add_patch(plt.Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, fc=P_BLUE if test[i, j] else "#dcdcdc", ec="white"))
        ax.set_xlim(0, n_pk)
        ax.set_ylim(n_amp, 0)
        ax.set_xticks(np.arange(n_pk) + 0.5, [str(k + 1) for k in range(n_pk)], fontsize=12)
        ax.set_yticks(np.arange(n_amp) + 0.5, [f"ch{k}" for k in range(n_amp)], fontsize=12)
        ax.set_xlabel("packet")
        ax.set_title(title, fontsize=15)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=P_BLUE, label="predicted (test)"), Patch(color="#dcdcdc", label="used for training")],
               loc="lower center", ncol=2, frameon=False, fontsize=13, bbox_to_anchor=(0.5, -0.08))
    fig.savefig(OUT / "m_cv.png")
    plt.close(fig)


FIGURES = {f.__name__: f for f in (s1_pipeline, s1_levels, s2_drift_profile, s2_chi2, s3_runwaders_flow, s3_label_error,
                                   s3_fit_quality_table, s4_model_ladder, s4_model_errors, s4_saas, s5_timeline, gc_figures,
                                   t_trace_anatomy, t_trace_noise, p_processing, m_cv)}


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
