"""
make_explained_2026_11_05.py -- figures of the plain-language walkthrough
(docs/overleaf_walkthrough_explained_2026-11-05/explained.tex).  One message per figure, few elements, the point
written on the plot.  Numbers come from the result files of 2026-11-03/04/05 (session C summary, scorecard,
relative_extra, rank reports, the funnel replay) or from the context document's tables where that is the only record.

    .venv/bin/python docs/slides/make_explained_2026_11_05.py [name ...]

Writes docs/overleaf_walkthrough_explained_2026-11-05/figures/e_*.png and prints the numbers the slides quote.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lvl1")]
OUT = ROOT / "docs" / "overleaf_walkthrough_explained_2026-11-05" / "figures"
REPLAY = ROOT / "processed" / "lvl1_results" / "2026-11-05_funnel_replay" / "replay.npz"

INK, INK2, GRID, SURFACE = "#1b2430", "#4a5462", "#e2e0d8", "#f7f6f2"
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#2a78d6", "#c9551f", "#13865d", "#b3372e", "#a3a19a", "#7a4fb5"
AMP_COLORS = {0: BLUE, 1: ORANGE, 2: PURPLE, 3: GREEN}
TARGETS = ("read noise $\\sigma_e$", "dark charge $\\lambda$", "gain")

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold", "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", name)


def note(ax, xy, xytext, text, color=INK, **kw):
    ax.annotate(text, xy=xy, xytext=xytext, color=color, fontsize=kw.pop("fontsize", 12), ha=kw.pop("ha", "left"),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4), **kw)


def data(channels=(0, 1, 2, 3)):
    import scorecard as sc
    files, rows, d = sc.load(channels)
    return d.reset_index(drop=True), files["physics"]["Y"][rows]


# ------------------------------------------------------------------------------------------- what we predict
def histogram():
    """An illustration: the pixel-value histogram of an image and the three labels read off it."""
    g, s, lam = 1.3, 0.2, 0.35                       # ADU per electron, noise (ADU), electrons per pixel
    x = np.linspace(-0.8, 3.6, 900)
    pk = [np.exp(-lam) * lam ** k / __import__("math").factorial(k) for k in range(4)]
    y = sum(p * np.exp(-(x - k * g) ** 2 / (2 * s ** 2)) for k, p in enumerate(pk))
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.fill_between(x, y, color=BLUE, alpha=0.25)
    ax.plot(x, y, color=BLUE, lw=2)
    for k, lab in enumerate(("0 electrons", "1 electron", "2 electrons")):
        ax.text(k * g, pk[k] + 0.04, lab, ha="center", fontsize=12, color=INK2)
    ax.annotate("", xy=(-s, 0.5), xytext=(s, 0.5), arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
    ax.text(0, 0.42, "width = read noise $\\sigma_e$", color=RED, ha="center", fontsize=12)
    ax.annotate("", xy=(0, -0.08), xytext=(g, -0.08), arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2), annotation_clip=False)
    ax.text(g / 2, -0.16, "spacing = gain $g$ (ADU per electron)", color=GREEN, ha="center", fontsize=12)
    ax.annotate("", xy=(g + 0.4, pk[1]), xytext=(g + 0.4, 0), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2))
    ax.text(g + 0.5, 0.45, "share of pixels that caught\na stray electron = dark charge $\\lambda$", color=ORANGE, fontsize=12, va="center")
    ax.set_xlabel("pixel value (ADU), after averaging 500 skips")
    ax.set_yticks([])
    ax.set_ylabel("number of pixels")
    ax.set_ylim(-0.22, 0.85)
    ax.spines["left"].set_visible(False)
    save(fig, "e_histogram")


def trace_schematic():
    """An illustration: one skip of the raw waveform, and many skips of one pixel."""
    rng = np.random.default_rng(3)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1.3, 1]})
    t = np.linspace(0, 3, 900)
    base = np.where(t < 0.25, 6 * np.exp(-t / 0.06), 0)               # reset pulse
    sig = np.where(t > 1.5, -2.0, 0)                                    # charge dumped onto the sense node
    w = base + sig + 0.25 * rng.standard_normal(len(t))
    ax = axs[0]
    ax.plot(t, w, color=INK2, lw=1)
    ax.axvspan(0.6, 1.3, color=BLUE, alpha=0.15)
    ax.axvspan(1.9, 2.6, color=ORANGE, alpha=0.15)
    ax.text(0.95, 3.5, "pedestal\nwindow", ha="center", color=BLUE, fontsize=12)
    ax.text(2.25, 3.5, "signal\nwindow", ha="center", color=ORANGE, fontsize=12)
    ax.text(1.5, -1.2, "charge\nmoved in", ha="center", fontsize=11, color=INK2)
    ax.text(0.12, 5.3, "reset", fontsize=11, color=INK2)
    ax.set_title("one skip of the raw trace")
    ax.set_xlabel("time within the skip")
    ax.set_yticks([])
    ax.set_xticks([])
    ax = axs[1]
    v = 0.3 + 5.6 * rng.standard_normal(500)
    ax.plot(np.arange(500), v, ".", ms=3, color=BLUE, alpha=0.6)
    ax.axhline(v.mean(), color=RED, lw=2)
    ax.text(495, 13, "red line: average of 500\n= the pixel value", color=RED, fontsize=11, va="bottom", ha="right",
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.9))
    ax.annotate("", xy=(30, -5.6), xytext=(30, 5.6), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    ax.text(40, -9, "spread of one skip:\nwhite noise $w\\approx5.6$ ADU", fontsize=11)
    ax.set_title("500 skips of one pixel (signal $-$ pedestal)")
    ax.set_xlabel("skip number")
    ax.set_ylabel("ADU")
    save(fig, "e_trace_schematic")


# ------------------------------------------------------------------------------------------- labels
def label_noise():
    """context sections 23 / scorecard header: label noise of a change vs the typical change made by one knob."""
    noise, effect = (1.9, 11.9, 0.8), (3.9, 8.5, 0.4)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, t, n, e in zip(axs, TARGETS, noise, effect):
        b = ax.bar([0, 1], [n, e], color=[GREY, BLUE], width=0.6)
        ax.bar_label(b, labels=[f"{n:g} %", f"{e:g} %"], fontsize=12, padding=2)
        ax.set_xticks([0, 1], ["same setting\nmeasured twice", "typical effect\nof one knob"], fontsize=11)
        ax.set_title(t)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, max(n, e) * 1.35)
        verdict = "effect > noise" if e > 1.5 * n else "effect $\\leq$ noise"
        ax.text(0.5, max(n, e) * 1.22, verdict, ha="center", fontsize=13, fontweight="bold",
                color=GREEN if e > 1.5 * n else RED)
    save(fig, "e_label_noise")


def lambda_drift():
    d, Y = data((0, 1, 3))
    import change_test as ct
    nom = d.packet_id.isin(ct.NOMINAL).to_numpy()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for c in (0, 1, 3):
        m = nom & (d.channel.to_numpy() == c)
        h, l = d.hours.to_numpy()[m], np.exp(Y[m, 1])
        o = np.argsort(h)
        ax.plot(h[o], l[o], "-o", color=AMP_COLORS[c], lw=2, ms=8, label=f"amplifier ch{c}")
        print(f"lambda nominal ch{c}: first {l[o][0]:.3f} last {l[o][-1]:.3f} ({100 * (l[o][-1] / l[o][0] - 1):+.0f} %)")
    ax.set_xlabel("hours into the night")
    ax.set_ylabel("dark charge $\\lambda$ (e$^-$/pixel)")
    ax.set_title("the same (nominal) setting, measured 5 times during the night")
    ax.legend(fontsize=11, loc="upper right")
    ax.text(6.3, 0.2, "nothing changed, yet ch0 fell 22 %, ch1 15 %", fontsize=12, color=RED)
    ax.set_ylim(0.15, 0.62)
    ax.grid(axis="y", color=GRID)
    save(fig, "e_lambda_drift")


# ------------------------------------------------------------------------------------------- absolute models
def session_c():
    """Session C (sealed test), context section 14 / first-data report."""
    no, tr = (10.2, 23.2, 1.4), (7.3, 5.3, 0.9)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, t, a, b, sig in zip(axs, TARGETS, no, tr, ("not significant\n(8 packets)", "significant", "not significant")):
        bb = ax.bar([0, 1], [a, b], color=[GREY, BLUE], width=0.6)
        ax.bar_label(bb, labels=[f"{a:g} %", f"{b:g} %"], fontsize=12, padding=2)
        ax.set_xticks([0, 1], ["no trace\n(settings +\namplifier)", "with the\ntrace"], fontsize=11)
        ax.set_title(t)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, a * 1.45)
        ax.text(0.5, a * 1.25, sig, ha="center", fontsize=11, color=GREEN if sig == "significant" else INK2)
    fig.suptitle("typical error on the sealed test (session C)  -  lower is better", fontsize=13, y=1.04)
    save(fig, "e_session_c")


def physics_white():
    """Image sigma_e vs the trace's own estimate w / sqrt(N) / gain, by kind of packet."""
    import relative_extra as rx
    d, Y = data((0, 1, 3))
    plan = {p["packet_id"]: p for p in json.loads((ROOT / "lvl1/packet_plan/plan.json").read_text())["packets"]}
    kind = np.array([rx.kind_of(plan[p]["changes"]) for p in d.packet_id])
    w = d["trace.fit: white noise per skip (ADU)"].to_numpy(float)
    gain = np.exp(Y[:, 2])
    est = w / np.sqrt(500) / gain
    img = np.exp(Y[:, 0])
    ok = np.isfinite(est) & np.isfinite(img)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    groups = (("nominal", "nominal", INK), ("voltage changes", None, BLUE), ("timing changes", "timing", ORANGE))
    volt = np.isin(kind, ["output stage (VDD, VR, RG, SW)", "clock rails (H, V, TG)", "VSUB"])
    for lab, k, col in groups:
        m = ok & (volt if k is None else kind == k)
        ax.scatter(est[m], img[m], s=45, color=col, edgecolor=SURFACE, lw=0.6, label=lab, zorder=3)
    xx = np.linspace(0.15, 0.27, 10)
    r = np.median(img[ok & (kind == "nominal")] / est[ok & (kind == "nominal")])
    ax.plot(xx, r * xx, color=INK2, lw=1.5, ls="--")
    ax.text(0.27, r * 0.27 - 0.018, f"image = {r:.2f} $\\times$ trace", fontsize=11, color=INK2, ha="left")
    ax.set_xlabel("from the trace: $w/\\sqrt{500}\\,/\\,g$  (e$^-$)")
    ax.set_ylabel("from the image: $\\sigma_e$ (e$^-$)")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(color=GRID)
    print("physics_white: image/trace ratio at nominal", r)
    save(fig, "e_physics_white")


def cv_vs_c():
    """sigma_e error of 6 models: on the training cross-validation and on the sealed session C (summary.csv)."""
    m = {"gp_dim (flexible)": (10.5, 13.5), "trees (flexible)": (12.1, 14.3), "physics_white": (12.3, 7.3),
         "pretrained cnn": (12.7, 12.2), "no trace": (15.7, 10.2)}
    ly = {"trees (flexible)": 11.6, "physics_white": 12.3, "pretrained cnn": 13.0}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (a, b) in m.items():
        col = GREEN if name == "physics_white" else (RED if name.startswith("gp_dim") else (GREY if name == "no trace" else "#8fb6e8"))
        lw = 3 if col in (GREEN, RED) else 1.6
        ax.plot([0, 1], [a, b], "-o", color=col, lw=lw, ms=8)
        ax.text(-0.04, ly.get(name, a), name, ha="right", va="center", fontsize=11, color=col if col != "#8fb6e8" else INK2)
        ax.text(1.04, b, f"{b:g} %", ha="left", va="center", fontsize=11, color=col if col != "#8fb6e8" else INK2)
    ax.set_xticks([0, 1], ["training check\n(cross-validation on A+B)", "the real test\n(sealed session C)"])
    ax.set_xlim(-0.75, 1.25)
    ax.set_ylabel("$\\sigma_e$ error (%)  -  lower is better")
    ax.invert_yaxis()
    ax.grid(axis="y", color=GRID)
    save(fig, "e_cv_vs_c")


# ------------------------------------------------------------------------------------------- amplifiers
def amp_levels():
    d, Y = data()
    import change_test as ct
    nom = d.packet_id.isin(ct.NOMINAL).to_numpy()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
    for t, ax in enumerate(axs):
        for c in range(4):
            v = np.exp(Y[nom & (d.channel.to_numpy() == c), t])
            ax.scatter(np.full(len(v), c) + np.linspace(-0.12, 0.12, len(v)), v, s=60, color=AMP_COLORS[c], edgecolor=SURFACE)
        ax.set_xticks(range(4), [f"ch{c}" for c in range(4)])
        ax.set_title(TARGETS[t])
        ax.grid(axis="y", color=GRID)
    axs[0].set_ylabel("value at the nominal setting")
    fig.suptitle("same setting, four amplifiers: each has its own level", fontsize=13, y=1.03)
    save(fig, "e_amp_levels")


def fingerprint():
    d, _ = data()
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for ax, col, title in ((axs[0], "ref.edge: ringing rms (ADU)", "a fingerprint: edge ringing (reference trace)"),
                           (axs[1], "trace.fit: white noise per skip (ADU)", "not a fingerprint: white noise per skip")):
        for c in range(4):
            m = d.channel.to_numpy() == c
            ax.scatter(d.hours.to_numpy()[m], d[col].to_numpy(float)[m], s=22, color=AMP_COLORS[c], label=f"ch{c}")
        ax.set_title(title, fontsize=12)
        ax.set_yscale("log")
        ax.set_xlabel("hours into the night")
        ax.grid(axis="y", color=GRID)
    axs[0].set_ylabel("feature value (ADU)")
    axs[0].legend(fontsize=10, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "e_fingerprint")


def new_amp():
    """scorecard (1) and (2): best trace model on a known amplifier, on a new one, and the average guess."""
    known, new, avg = (10.1, 28.0, 0.9), (14.7, 41.2, 3.2), (20.1, 50.1, 3.5)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, t, a, b, c in zip(axs, TARGETS, known, new, avg):
        bb = ax.bar([0, 1, 2], [a, b, c], color=[BLUE, ORANGE, GREY], width=0.65)
        ax.bar_label(bb, labels=[f"{v:g} %" for v in (a, b, c)], fontsize=11, padding=2)
        ax.set_xticks([0, 1, 2], ["known\namplifier", "new\namplifier", "new amp:\nguess the\naverage"], fontsize=10)
        ax.set_title(t)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, c * 1.25)
    fig.suptitle("typical error of the best trace model  -  lower is better", fontsize=13, y=1.04)
    save(fig, "e_new_amp")


def offset_toy():
    """An illustration: two amplifiers, different levels, the same effect of a knob."""
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for lvl, col, lab in ((0.30, BLUE, "amplifier A (trained on)"), (0.60, ORANGE, "amplifier B (new)")):
        ax.plot([0, 1], [lvl, 0.8 * lvl], "-o", color=col, lw=2.5, ms=10, label=lab)
    ax.plot([1], [0.24], "x", color=RED, ms=14, mew=3)
    note(ax, (1, 0.24), (1.08, 0.13), "absolute model, asked about B:\n\"0.24\" (it learned A's level)  wrong", color=RED, fontsize=11)
    note(ax, (1, 0.48), (1.08, 0.50), "change model:\n\"20 % lower than before\"  right for both", color=GREEN, fontsize=11)
    ax.set_xticks([0, 1], ["setting 1", "setting 2"])
    ax.set_xlim(-0.2, 2.1)
    ax.set_ylim(0.05, 0.7)
    ax.set_ylabel("dark charge $\\lambda$")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", color=GRID)
    save(fig, "e_offset_toy")


def ch2():
    """ch2_comparison.md section 1: mean lambda error on ch2 (left-out amplifier)."""
    absol = {"settings GP": -54.3, "physics": -99.7, "trees": -75.5, "pretrained cnn": -83.6}
    change = {"settings change GP": -2.3, "trace change trees": -7.1, "bands + GP": 0.3, "physics, differenced": -0.6,
              "pretrained cnn, differenced": -0.7}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = list(absol) + list(change)
    vals = list(absol.values()) + list(change.values())
    cols = [RED] * len(absol) + [GREEN] * len(change)
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, height=0.65)
    for yi, v in zip(y, vals):
        ax.text(v - 2 if v < 0 else v + 2, yi, f"{v:+.0f} %", va="center", ha="right" if v < 0 else "left", fontsize=11)
    ax.set_yticks(y, names, fontsize=11)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlim(-125, 70)
    ax.set_xlabel("average miss on ch2's $\\lambda$ (%)")
    ax.text(22, y[1] - 0.5, "models that predict\nthe LEVEL", color=RED, fontsize=13, fontweight="bold", va="center")
    ax.text(22, y[6], "models that predict\nthe CHANGE", color=GREEN, fontsize=13, fontweight="bold", va="center")
    save(fig, "e_ch2")


def always_up():
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    b = ax.bar([0, 1], [96, 52], color=[RED, BLUE], width=0.55)
    ax.bar_label(b, labels=["96 %", "52 %"], fontsize=13, padding=2)
    ax.axhline(50, color=INK2, ls=":", lw=1.2)
    ax.text(1.42, 51.5, "coin flip", fontsize=10, color=INK2, ha="right")
    ax.set_xticks([0, 1], ["nominal $\\to$ new setting\n(within a packet)", "packet $\\to$ next packet\n(neighbours)"], fontsize=11)
    ax.set_ylabel("$\\sigma_e$ changes that go UP (%)")
    ax.set_ylim(0, 112)
    ax.set_title("score of a model that always answers \"worse\"", fontsize=13)
    save(fig, "e_always_up")


def direction():
    """relative_extra.md, new amplifier, all between-packet pairs."""
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    x = np.arange(3)
    for k, (lab, v, col) in enumerate((("settings only (no trace)", (68, 75, 73), GREY),
                                       ("best change model (traces)", (75, 89, 88), BLUE))):
        b = ax.bar(x + (k - 0.5) * 0.36, v, 0.34, color=col, label=lab)
        ax.bar_label(b, labels=[f"{i} %" for i in v], fontsize=11, padding=2)
    ax.axhline(50, color=INK2, ls=":", lw=1.2)
    ax.text(2.6, 51, "coin flip", fontsize=10, color=INK2, ha="right")
    ax.set_xticks(x, TARGETS)
    ax.set_ylim(0, 105)
    ax.set_ylabel("right about \"better or worse\" (%)")
    ax.legend(fontsize=11, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.32))
    save(fig, "e_direction")


def pairs():
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    b = ax.bar([0, 1], [0.76, 0.85], color=[GREY, BLUE], width=0.55)
    ax.bar_label(b, labels=["0.76", "0.85"], fontsize=13, padding=2)
    ax.set_xticks([0, 1], ["neighbouring\npackets only", "any two packets\n(drift removed)"], fontsize=11)
    ax.set_ylabel("ranking quality (1 = perfect)")
    ax.set_ylim(0, 1.05)
    ax.set_title("trained on which pairs?", fontsize=13)
    save(fig, "e_pairs")


def direction_weight():
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    d = np.linspace(0, 0.6, 600)
    s = 0.12
    ax.plot(100 * d, 1 - np.exp(-d ** 2 / (2 * s ** 2)), color=ORANGE, lw=3)
    ax.axvline(12, color=INK2, ls=":", lw=1.2)
    ax.text(12.5, 0.08, "label noise of $\\lambda$ (12 %)", fontsize=11, color=INK2)
    ax.text(30, 0.75, "real changes:\nfull weight", fontsize=12, color=GREEN)
    ax.text(0.5, 0.55, "changes the size\nof the noise:\nalmost ignored", fontsize=12, color=RED)
    ax.set_xlabel("true change of $\\lambda$ between the two traces (%)")
    ax.set_ylabel("weight in training")
    ax.set_xlim(0, 60)
    ax.grid(color=GRID)
    save(fig, "e_direction_weight")


# ------------------------------------------------------------------------------------------- the funnel
def replay_loaded():
    z = np.load(REPLAY, allow_pickle=True)
    truth = z["truth_amp"].mean(0)[:, 1]
    return z, truth, z["bt_new_w0"]


def sort6():
    """Worked example: 6 first-night settings, the model's pairwise verdicts and the resulting order."""
    import shortlist as sl
    z, truth, score = replay_loaded()
    pk = z["packets"]
    M = np.nanmean([sl.antisymmetric(x) for x in z["D_new"][..., 1]], 0)     # M[i, j] > 0: i lower lambda than j
    order_t = np.argsort(truth)
    pick = order_t[np.linspace(0, len(truth) - 1, 6).round().astype(int)]
    rng = np.random.default_rng(1)
    pick = pick[rng.permutation(6)]                                          # shown in a shuffled order
    names = "ABCDEF"
    P = 0.5 + M[np.ix_(pick, pick)]                                          # P(row has lower lambda than column)
    wins = np.nansum(P, 1)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    show = np.where(np.eye(6, dtype=bool), np.nan, P)
    ax.imshow(show, cmap="RdBu", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            if i != j:
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=12,
                        color="white" if abs(P[i, j] - 0.5) > 0.3 else INK)
        ax.text(6.0, i, f"{wins[i]:.1f}", ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(7.2, i, f"{100 * truth[pick[i]]:+.0f} %", ha="center", va="center", fontsize=12, color=INK2)
    ax.text(6.0, -0.85, "wins", ha="center", fontsize=12, fontweight="bold")
    ax.text(7.2, -0.85, "true $\\lambda$\nchange", ha="center", fontsize=11, color=INK2)
    ax.set_xticks(range(6), [f"vs {n}" for n in names])
    ax.set_yticks(range(6), [f"setting {n}" for n in names])
    ax.set_xlim(-0.5, 7.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "e_sort6")
    rank_model = "".join(names[i] for i in np.argsort(-wins))
    rank_true = "".join(names[i] for i in np.argsort(truth[pick]))
    print("sort6: settings", dict(zip(names, pk[pick])), "order by wins", rank_model, "true order", rank_true,
          "wins", np.round(wins, 1))


def replay_bars():
    z, truth, score = replay_loaded()
    o = np.argsort(score)
    best = np.argmin(truth)
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    cols = [BLUE if k < 5 else GREY for k in range(len(o))]
    v = 100 * truth[o]
    ax.bar(np.arange(1, len(o) + 1), v, color=cols, width=0.75)
    kb = int(np.flatnonzero(o == best)[0]) + 1
    note(ax, (kb, v[kb - 1]), (kb + 3, v[kb - 1] - 8), f"the truly best setting: placed {kb}th", color=GREEN, fontsize=12)
    ax.axhline(0, color=INK, lw=1)
    ax.axvline(5.5, color=BLUE, ls="--", lw=1.2)
    ax.text(3, 115, "shortlist\n(top 5)", ha="center", color=BLUE, fontsize=12, fontweight="bold")
    ax.set_xlabel("the model's order (1 = predicted lowest $\\lambda$)  $\\rightarrow$")
    ax.set_ylabel("TRUE change of $\\lambda$ vs nominal (%)")
    ax.set_xlim(0.3, len(o) + 0.7)
    ax.set_ylim(-45, 150)
    ax.grid(axis="y", color=GRID)
    save(fig, "e_replay_bars")
    print("replay: Spearman", spearmanr(truth, score)[0], "best placed", kb)


def regret():
    z, truth, score = replay_loaded()
    n = len(truth)
    tj = 100 * (truth - truth.min())
    k = np.arange(1, 16)
    model = [tj[np.argsort(score)[:kk]].min() for kk in k]
    rng = np.random.default_rng(0)
    rand = [np.mean([tj[rng.choice(n, kk, replace=False)].min() for _ in range(4000)]) for kk in k]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(k, rand, "-o", color=GREY, lw=2.5, ms=7, label="pick settings at random")
    ax.plot(k, model, "-o", color=BLUE, lw=2.5, ms=7, label="follow the model's order")
    ax.set_xticks(k)
    ax.set_xlabel("number of settings we take an image of")
    ax.set_ylabel("how far the best one found is\nfrom the true best (% of $\\lambda$)")
    ax.legend(fontsize=11)
    ax.grid(color=GRID)
    save(fig, "e_regret")
    print("regret model", np.round(model[:8], 1), "random", np.round(rand[:8], 1))


def noise():
    z, *_ = replay_loaded()
    st, sp = np.exp(z["sigma_true"]), np.exp(z["sigma_cv"])
    err = 100 * np.sqrt(np.nanmean((z["sigma_cv"] - z["sigma_true"]) ** 2))
    fig, ax = plt.subplots(figsize=(6.4, 5))
    xx = np.linspace(0.13, 0.32, 10)
    ax.fill_between(xx, xx * np.exp(-err / 100), xx * np.exp(err / 100), color=GRID)
    ax.plot(xx, xx, color=INK2, lw=1)
    ax.scatter(st, sp, s=30, color=BLUE, edgecolor=SURFACE)
    ax.text(0.25, 0.17, f"grey band: $\\pm${err:.0f} %", fontsize=12, color=INK2)
    ax.set_xlabel("$\\sigma_e$ from the image (e$^-$)")
    ax.set_ylabel("$\\sigma_e$ predicted from one trace (e$^-$)")
    ax.set_xlim(0.13, 0.32)
    ax.set_ylim(0.13, 0.32)
    ax.grid(color=GRID)
    save(fig, "e_noise")


# ------------------------------------------------------------------------------------------- v2
def design():
    d, _ = data((0,))
    nom = d[d.packet_id == "L1_01"].iloc[0]
    base = (float(nom["theta.delayInt"]), float(nom["theta.VR1"]))
    v1 = json.loads((ROOT / "lvl1/packet_plan/plan.json").read_text())["packets"]
    v2 = json.loads((ROOT / "lvl1/packet_plan_v2/plan.json").read_text())["packets"]
    fig, ax = plt.subplots(figsize=(8.6, 5))
    for plan, mk, col, lab, s in ((v2, "o", BLUE, "v2 (planned)", 40), (v1, "s", ORANGE, "first night", 80)):
        pts = np.array([(p["changes"].get("delayInt", base[0]), p["changes"].get("VR1", base[1])) for p in plan
                        if p.get("session") in ("A", "B", "C", "D", "E")], float)
        ax.scatter(pts[:, 0], pts[:, 1], marker=mk, s=s, color=col, edgecolor=SURFACE, label=lab, zorder=3 if plan is v1 else 2)
    ax.scatter(*base, marker="*", s=350, color=INK, zorder=4, label="nominal")
    ax.set_xlabel("delayInt (integration window)")
    ax.set_ylabel("VR1 (reset drain voltage)")
    ax.legend(fontsize=11, loc="upper right", bbox_to_anchor=(1.0, 1.15), ncol=3)
    ax.grid(color=GRID)
    save(fig, "e_design")


def worked_pair():
    """Numbers for the worked example slide: L1_04 -> L1_05, ch0."""
    d, Y = data((0, 1, 3))
    r = pd.read_csv(ROOT / "processed/lvl1_results/2026-11-04_scorecard/relative_predictions.csv")
    for a, b in (("L1_04", "L1_05"), ("L1_02", "L1_03")):
        x = r[(r["packet A"] == a) & (r["packet B"] == b) & (r.channel == 0)].iloc[0]
        ya = Y[(d.packet_id == a).to_numpy() & (d.channel == 0).to_numpy()][0]
        yb = Y[(d.packet_id == b).to_numpy() & (d.channel == 0).to_numpy()][0]
        print(f"pair {a}->{b} ch0: sigma {np.exp(ya[0]):.3f}->{np.exp(yb[0]):.3f} lambda {np.exp(ya[1]):.3f}->{np.exp(yb[1]):.3f}; "
              f"true change sigma {100 * x['true change sigma_e']:+.0f} % lambda {100 * x['true change lambda']:+.0f} %; "
              f"predicted (new amp, change trees) sigma {100 * x['new | trace change: trees | sigma_e']:+.0f} % "
              f"lambda {100 * x['new | trace change: trees | lambda']:+.0f} %")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fs = (histogram, trace_schematic, label_noise, lambda_drift, session_c, physics_white, cv_vs_c, amp_levels,
          fingerprint, new_amp, offset_toy, ch2, always_up, direction, pairs, direction_weight, sort6, replay_bars,
          regret, noise, design, worked_pair)
    for f in fs:
        if len(sys.argv) == 1 or f.__name__ in sys.argv[1:]:
            f()


if __name__ == "__main__":
    main()
