"""
make_walkthrough_2026_11_05.py -- the new figures of the level-1 walkthrough deck
(docs/overleaf_walkthrough_2026-11-05/walkthrough.tex).  Numbers come from the result files of 2026-11-03/04/05
(scorecard, relative_extra, rank_* reports, the funnel replay) or, where the context document is the only record,
from its tables (marked "context section N").

    .venv/bin/python docs/slides/make_walkthrough_2026_11_05.py

Writes docs/overleaf_walkthrough_2026-11-05/figures/w_*.png.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "lvl1")]
OUT = ROOT / "docs" / "overleaf_walkthrough_2026-11-05" / "figures"
SC = ROOT / "docs" / "scorecard_2026-11-04"
REPLAY = ROOT / "processed" / "lvl1_results" / "2026-11-05_funnel_replay" / "replay.npz"

INK, INK2, GRID, SURFACE = "#1b2430", "#4a5462", "#e2e0d8", "#f7f6f2"
BLUE, ORANGE, GREEN, RED, GREY, PURPLE, GOLD = "#2a78d6", "#c9551f", "#13865d", "#b3372e", "#9a9890", "#7a4fb5", "#b8860b"
TARGETS = (r"$\sigma_e$", r"$\lambda$", "gain")
KIND_COLOR = {"nominal": INK, "timing": BLUE, "output stage": ORANGE, "clock rails": GREEN, "VSUB": PURPLE,
              "all at once": GOLD}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", name)


def md_table(path: Path, heading: str) -> dict[str, list[str]]:
    """Rows of the first markdown table after the heading line that contains `heading` (exact substring)."""
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("#") and heading in l)
    rows = {}
    for l in lines[start + 1:]:
        if l.startswith("#"):
            break
        if l.startswith("|") and not l.startswith("|---") and not l.startswith("| model"):
            cells = [c.strip() for c in l.strip("|").split("|")]
            rows[cells[0]] = cells[1:]
    return rows


def pct(cell: str) -> float:
    m = re.match(r"\s*([-+\d.]+|nan|-)", cell)
    return float(m.group(1)) if m and m.group(1) not in ("-",) else np.nan


def kind_short(k: str) -> str:
    return {"output stage (VDD, VR, RG, SW)": "output stage", "clock rails (H, V, TG)": "clock rails"}.get(k, k)


# ------------------------------------------------------------------------------------------------ the first night
def night():
    import relative_extra as rx
    import scorecard as sc
    files, rows, d = sc.load((0, 1, 3))
    Y = files["physics"]["Y"][rows]
    plan = {p["packet_id"]: p for p in json.loads((ROOT / "lvl1/packet_plan/plan.json").read_text())["packets"]}
    kind = np.array([kind_short(rx.kind_of(plan[p]["changes"])) for p in d.packet_id])
    h, ch = d.hours.to_numpy(), d.channel.to_numpy()
    fig, axs = plt.subplots(2, 1, figsize=(14, 5.2), sharex=True)
    edges = [h[d.block.to_numpy() == b].min() for b in range(5)] + [h.max() + 0.3]
    for ax, t, lab in zip(axs, (1, 0), (r"$\lambda$ (e$^-$/pixel)", r"$\sigma_e$ (e$^-$)")):
        for b in range(5):
            ax.axvspan(edges[b] - 0.15, edges[b + 1] - 0.15, color=(INK if b % 2 else SURFACE), alpha=0.05, lw=0)
            if t == 1:
                ax.text((edges[b] + edges[b + 1]) / 2 - 0.15, 1.02, f"time block {b + 1}", transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=8, color=INK2)
        for c, mk in zip((0, 1, 3), ("o", "s", "^")):
            m = ch == c
            nom = m & (kind == "nominal")
            ax.plot(h[nom], np.exp(Y[nom, t]), "-", color=INK, lw=1, alpha=0.5)
            for k, col in KIND_COLOR.items():
                mm = m & (kind == k)
                ax.scatter(h[mm], np.exp(Y[mm, t]), s=(60 if k == "nominal" else 26), marker=mk, color=col,
                           edgecolor=SURFACE, lw=0.6, zorder=3, label=(k if c == 0 else None))
        ax.set_ylabel(lab)
        grid(ax)
    axs[0].set_yscale("log")
    axs[0].legend(ncol=6, loc="upper right", fontsize=8, bbox_to_anchor=(1, 1.22))
    axs[1].set_xlabel("hours since the first packet (2026-11-02 17:20)")
    sess = np.array([plan[p]["session"] for p in d.packet_id])
    for s_, lab, col in (("A", "session A: timing, one knob at a time", BLUE), ("B", "B: voltages, one knob", ORANGE),
                         ("C", "C: joint (test)", GOLD)):
        lo, hi = h[sess == s_].min(), h[sess == s_].max()
        axs[1].annotate("", xy=(lo - 0.1, -0.30), xytext=(hi + 0.1, -0.30), xycoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="|-|", color=col, lw=1.5))
        axs[1].text((lo + hi) / 2, -0.40, lab, transform=axs[1].get_xaxis_transform(), ha="center", va="top", fontsize=8, color=col)
    from matplotlib.lines import Line2D
    axs[1].legend(handles=[Line2D([], [], marker=m, ls="", color=INK2, label=l) for m, l in (("o", "ch0"), ("s", "ch1"), ("^", "ch3"))],
                  ncol=3, fontsize=8, loc="upper right")
    save(fig, "w_night_timeline")


def label_noise():
    """context sections 2, 7, 23 and the scorecard header."""
    groups = [("nominal repeats\n(same setting, hours apart)", (1.7, 12.0, 0.6), GREY),
              ("label noise of a change\n(nominal vs nominal)", (1.9, 11.9, 0.8), RED),
              ("median true change\n(one knob)", (3.9, 8.5, 0.4), BLUE),
              ("RMS true change\n(within a packet)", (17.4, 34.2, 1.0), GREEN)]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    x = np.arange(3)
    for k, (lab, v, col) in enumerate(groups):
        b = ax.bar(x + (k - 1.5) * 0.2, v, 0.19, color=col, label=lab)
        ax.bar_label(b, fmt="%g", fontsize=8, padding=1)
    ax.set_xticks(x, TARGETS)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 300)
    ax.set_ylabel("percent (log scale)")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    grid(ax)
    save(fig, "w_label_noise")


def drift_bug():
    """context section 6: run_WADERs on the raw and on the drift-corrected images."""
    raw = {"L1_14 nominal": (.148, .155, .168, .179), "L1_06 delayInt 3000": (.145, .159, .186, .195),
           "L1_23 delayInt 3000 (replicate)": (.144, .144, .180, .181), "L1_03 delay_Ha 900": (.238, .221, .374, .247)}
    cor = {"L1_14 nominal": (.148, .155, .168, .179), "L1_06 delayInt 3000": (.348, .227, .184, .258),
           "L1_23 delayInt 3000 (replicate)": (np.nan,) * 4, "L1_03 delay_Ha 900": (.694, .342, .404, .348)}
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for i, k in enumerate(raw):
        for c in range(4):
            x = i * 5 + c
            ax.plot([x, x], [raw[k][c], cor[k][c]], color=GREY, lw=1)
            ax.scatter(x, raw[k][c], color=BLUE, s=40, zorder=3, label="raw image" if i == c == 0 else None)
            ax.scatter(x, cor[k][c], color=RED, marker="X", s=50, zorder=3, label="drift-corrected image" if i == c == 0 else None)
        ax.text(i * 5 + 1.5, -0.13, k, transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
    ax.text(12, 0.30, "not rerun with the correction;\nits raw values match L1_06's", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks([i * 5 + c for i in range(4) for c in range(4)], ["ch0", "ch1", "ch2", "ch3"] * 4, fontsize=7)
    ax.set_ylabel(r"image $\sigma_e$ (e$^-$)")
    ax.legend(loc="upper left", fontsize=8)
    grid(ax)
    save(fig, "w_drift_bug")


def cv_models():
    """context section 7: A+B setting-grouped CV, clean labels."""
    m = {"mean": (17.4, 47.8, 2.7), "settings": (17.9, 46.4, 2.7), "physics (Allan floor)": (30.6, 37.0, 1.2),
         "physics_white": (12.6, 37.0, 1.2), "sets": (15.7, 37.0, 1.2), "trees": (13.2, 28.8, 0.8), "gp": (17.0, 40.2, 1.7),
         "nn": (13.0, 33.0, 0.9), "cnn": (16.9, 32.4, 0.9), "pretrained cnn": (11.8, 35.6, 1.2), "deep kernel": (12.7, 43.9, 1.2),
         "trace_physics_white": (20.8, np.nan, np.nan)}
    names = list(m)
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    for t, ax in enumerate(axs):
        v = np.array([m[n][t] for n in names])
        cols = [GREY if n in ("mean", "settings") else (GREEN if n.startswith("physics") or n.startswith("trace_phys") else BLUE) for n in names]
        ax.barh(np.arange(len(names)), v, color=cols, height=0.7)
        ax.axvline(m["settings"][t], color=INK2, ls="--", lw=1)
        for i, x in enumerate(v):
            if np.isfinite(x):
                ax.text(x, i, f" {x:g}", va="center", fontsize=7)
        best = np.nanargmin(v)
        ax.get_children()[best].set_edgecolor(INK)
        ax.set_title(TARGETS[t] + " error (%)")
        grid(ax, "x")
    axs[0].set_yticks(np.arange(len(names)), names)
    axs[0].invert_yaxis()
    save(fig, "w_cv_models")


# ------------------------------------------------------------------------------------------------ scorecard
def scorecard_grid():
    """The four cells: best trace model vs the cell's settings-only reference (scorecard.md)."""
    f = SC / "scorecard.md"
    a1, a2 = md_table(f, "(1) Absolute, trained"), md_table(f, "(2) Absolute, new")
    r3, r4 = md_table(f, "(3) Relative, trained amplifiers, n -> n+1"), md_table(f, "(4) Relative, new amplifier, n -> n+1")
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.4))
    panels = [(axs[0, 0], a1, "settings + amplifier: gp", "(1) absolute, trained amplifiers", "error (%)  - lower is better", False),
              (axs[0, 1], a2, "settings: gp", "(2) absolute, NEW amplifier", "error (%)  - lower is better", False),
              (axs[1, 0], r3, "settings + amplifier: gp", "(3) relative n$\\to$n+1, trained amplifiers", "direction right (%)  - higher is better", True),
              (axs[1, 1], r4, "settings change: gp", "(4) relative n$\\to$n+1, NEW amplifier", "direction right (%)  - higher is better", True)]
    for ax, tab, ref, title, ylab, rel in panels:
        x = np.arange(3)
        cols = [0, 2, 4] if rel else [0, 1, 2]
        refv = [pct(tab[ref][c]) for c in cols]
        tr = {k: v for k, v in tab.items() if k.startswith("trace")}
        vals = np.array([[pct(v[c]) for c in cols] for v in tr.values()])
        best = (np.nanargmax if rel else np.nanargmin)(vals, axis=0)
        bv = vals[best, range(3)]
        ax.bar(x - 0.18, refv, 0.34, color=ORANGE, label="settings only (reference)")
        ax.bar(x + 0.18, bv, 0.34, color=BLUE, label="best trace model")
        names = list(tr)
        for i in range(3):
            verdict = re.search(r"\((\w+)\)", list(tr.values())[best[i]][cols[i]]).group(1)
            ax.text(x[i] + 0.18, bv[i], f"{bv[i]:g}\n{verdict}", ha="center", va="bottom", fontsize=7.5,
                    color=(GREEN if verdict == "better" else (RED if verdict == "worse" else INK2)), fontweight="bold")
            ax.text(x[i] - 0.18, refv[i], f"{refv[i]:g}", ha="center", va="bottom", fontsize=7.5, color=INK2)
            ax.text(x[i] + 0.18, 0.02, names[best[i]].replace("trace: ", "").replace("trace change: ", ""), rotation=90,
                    ha="center", va="bottom", fontsize=6.5, color="white", transform=ax.get_xaxis_transform())
        if rel:
            ax.axhline(50, color=INK2, ls=":", lw=1)
            ax.text(2.45, 51, "coin flip", fontsize=7, color=INK2, ha="right")
            ax.set_ylim(0, 110)
        else:
            ax.set_yscale("log")
            ax.set_ylim(0.5, 120)
        ax.set_xticks(x, TARGETS)
        ax.set_title(title)
        ax.set_ylabel(ylab, fontsize=8)
        grid(ax)
    axs[0, 0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "w_scorecard_grid")


def base_rate():
    f = SC / "scorecard.md"
    w3 = md_table(f, "(3) Relative, trained amplifiers, within packet")
    n3 = md_table(f, "(3) Relative, trained amplifiers, n -> n+1")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, tab, up, title in ((axs[0], w3, (96, 76, 45), "within a packet: nominal $\\to$ new setting"),
                               (axs[1], n3, (52, 51, 50), "neighbouring packets: n $\\to$ n+1")):
        x = np.arange(3)
        ax.bar(x, up, 0.6, color=SURFACE, edgecolor=INK2, hatch="///", label='"always say it goes up"')
        for k, (name, col, dx) in enumerate((("settings + amplifier: gp", ORANGE, -0.15), ("trace change: trees", BLUE, 0.15))):
            v = [pct(tab[name][c]) for c in (0, 2, 4)]
            ax.scatter(x + dx, v, s=70, color=col, zorder=3, label=name.replace(": gp", " GP"))
        for i, u in enumerate(up):
            ax.text(i, u + 2, f"{u} % up", ha="center", fontsize=8, color=INK2)
        ax.axhline(50, color=INK2, ls=":", lw=1)
        ax.set_xticks(x, TARGETS)
        ax.set_title(title)
        ax.set_ylim(0, 110)
        grid(ax)
    axs[0].set_ylabel("clear changes, direction right (%)")
    axs[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(1.05, -0.12), ncol=3)
    save(fig, "w_base_rate")


def ch2_offset():
    tab = md_table(SC / "ch2_comparison.md", "1. The amplifier offset")
    names = [k for k in tab if not k.startswith("change: no change")]
    lab = [k.replace("change: trace change", "change model: trace").replace("change: settings change", "change model: settings").replace("change: absolute ", "difference of two abs. ").replace("absolute ", "ABSOLUTE ").replace(", differenced", "") for k in names]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 6.4), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
    for ax, c, t in ((axs[0], 1, 1), (axs[1], 0, 0)):
        v = np.array([pct(tab[k][c]) for k in names])
        cols = [RED if not k.startswith("change") else BLUE for k in names]
        ax.barh(np.arange(len(names)), v, color=cols)
        for i, x in enumerate(v):
            ax.text(x, i, f" {x:+.1f}" if x >= 0 else f"{x:+.1f} ", va="center", ha=("left" if x >= 0 else "right"), fontsize=8.5)
        ax.axvline(0, color=INK, lw=1)
        ax.set_title(f"mean error on ch2, {TARGETS[t]} (%)")
        grid(ax, "x")
    axs[0].set_xlim(-125, 15)
    axs[1].set_xlim(-22, 8)
    axs[0].set_yticks(np.arange(len(names)), lab, fontsize=10)
    axs[0].invert_yaxis()
    save(fig, "w_ch2_offset")


KINDS = [("nominal vs timing", "nominal vs timing"), ("nominal vs output stage (VDD, VR, RG, SW)", "nominal vs output stage"),
         ("nominal vs clock rails (H, V, TG)", "nominal vs clock rails"), ("nominal vs VSUB", "nominal vs VSUB"),
         ("nominal vs all at once", "nominal vs all at once"), ("nominal vs another packet's change", "nominal vs another packet's change"),
         ("same kind: timing", "timing vs timing"), ("same kind: output stage", "output stage vs output stage"),
         ("same kind: all at once", "all at once vs all at once"), ("two different kinds of change", "two different kinds")]


def kind_heatmap():
    path = SC / "relative_extra.md"
    text = path.read_text()
    new = text[text.index("## new amplifier"):]
    tmp = path.parent / "_new_amp.md"
    tmp.write_text(new)
    M, N = np.full((len(KINDS), 3), np.nan), np.zeros((len(KINDS), 3), int)
    for i, (h, _) in enumerate(KINDS):
        line = next(l for l in new.splitlines() if l.startswith("###") and h in l)
        n = re.findall(r"(sigma_e|lambda|gain) (\d+) \(", line)
        N[i] = [int(c) for _, c in n]
        row = md_table(tmp, h)["combined: A, B, B - A"]
        M[i] = [pct(row[c]) for c in (0, 2, 4)]
    tmp.unlink()
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    show = np.where(N >= 5, M, np.nan)
    im = ax.imshow(show, cmap="RdYlBu", vmin=30, vmax=100, aspect="auto")
    for i in range(len(KINDS)):
        for j in range(3):
            txt = f"{M[i, j]:.0f} %\n({N[i, j]} clear)" if N[i, j] else "no clear\nchanges"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=(INK if N[i, j] >= 5 else GREY))
            if 0 < N[i, j] < 5:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="..", edgecolor=GRID, lw=0))
    ax.set_xticks(range(3), TARGETS)
    ax.set_yticks(range(len(KINDS)), [k for _, k in KINDS], fontsize=8.5)
    ax.axhline(5.5, color=INK, lw=1.2)
    ax.text(2.55, 2.5, "a nominal trace\nvs one kind", rotation=-90, va="center", fontsize=8, color=INK2)
    ax.text(2.55, 7.5, "two changed\nsettings", rotation=-90, va="center", fontsize=8, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.08)
    cb.set_label("direction right on clear changes (%)")
    ax.set_title("combined change model [A, B, B$-$A], NEW amplifier")
    save(fig, "w_kind_heatmap")


def combined_models():
    path = SC / "relative_extra.md"
    text = path.read_text()
    tmp = path.parent / "_new_amp.md"
    tmp.write_text(text[text.index("## new amplifier"):])
    tab = md_table(tmp, "all between-packet pairs")
    tmp.unlink()
    order = [("settings only (settings change: gp)", "settings change GP", ORANGE), ("change trees (B - A)", "trees on B$-$A", "#8fb6e8"),
             ("combined: A, B - A", "trees on [A, B$-$A]", "#5f97df"),
             ("combined: A, B - A, hand-picked bands", "[A, B$-$A] + bands", "#9aa6b8"),
             ("combined: A, B, B - A", "trees on [A, B, B$-$A]", BLUE),
             ("combined: A, B, B - A, hand-picked bands", "[A, B, B$-$A] + bands", "#6c7a90")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    x = np.arange(3)
    for k, (key, lab, col) in enumerate(order):
        r = tab[key]
        d = [pct(r[c]) for c in (0, 2, 4)]
        s = [pct(r[c]) for c in (1, 3, 5)]
        w = 0.13
        axs[0].bar(x + (k - 2.5) * w, d, w, color=col, label=lab, edgecolor=(INK if "B, B" in lab and "bands" not in lab else "none"))
        axs[1].bar(x + (k - 2.5) * w, s, w, color=col, edgecolor=(INK if "B, B" in lab and "bands" not in lab else "none"))
    axs[0].axhline(50, color=INK2, ls=":", lw=1)
    axs[0].set_ylim(40, 100)
    axs[0].set_title("direction right on clear changes (%) - higher is better")
    axs[1].set_title("size error of the change (RMS %) - lower is better")
    axs[1].set_yscale("log")
    for ax in axs:
        ax.set_xticks(x, TARGETS)
        grid(ax)
    axs[0].legend(fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(1.1, -0.1))
    save(fig, "w_combined_models")


def direction_weight():
    s = {r"$\sigma_e$": 0.019, r"$\lambda$": 0.12, "gain": 0.008}
    fig, ax = plt.subplots(figsize=(8, 3.3))
    d = np.linspace(0, 0.5, 600)
    for (lab, sd), col in zip(s.items(), (BLUE, ORANGE, GREEN)):
        ax.plot(100 * d, 1 - np.exp(-d ** 2 / (2 * sd ** 2)), color=col, lw=2.2, label=f"{lab}: label noise of a change {100 * sd:g} %")
        ax.axvline(100 * sd, color=col, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 50)
    ax.set_xlabel("size of the true change |log y(B) $-$ log y(A)|  (%)")
    ax.set_ylabel("weight of the pair in training")
    ax.legend(fontsize=8, loc="upper left")
    grid(ax, "both")
    save(fig, "w_direction_weight")


def rank_report(name):
    """{cell: {method: (regret, rho) per weight}} from a rank_methods report."""
    out, cell = {}, None
    for l in (SC / f"{name}.txt").read_text().splitlines():
        if l.startswith("==="):
            cell = "trained" if "trained" in l else "new"
            out[cell] = {}
        elif cell and "/" in l and not l.strip().startswith("1:0"):
            m = re.match(r"(.+?)\s{2,}(.*)", l)
            vals = re.findall(r"([\d.]+) /\s+([\d.]+|-)", m.group(2))
            out[cell][m.group(1).strip()] = [(float(a), float(b) if b != "-" else np.nan) for a, b in vals]
    return out


def pairs_target():
    combos = [("size", "random"), ("size", "neighbours"), ("direction", "random"), ("direction", "neighbours")]
    bt = "without reference: screen vs screen, bradley-terry"
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, cell, title in ((axs[0], "trained", "trained amplifiers"), (axs[1], "new", "NEW amplifier")):
        for k, (tg, pr) in enumerate(combos):
            r = rank_report(f"rank_{tg}_{pr}")[cell][bt]
            col = BLUE if tg == "direction" else GREY
            b = ax.bar(k, r[2][1], 0.6, color=col, hatch=("" if pr == "random" else "//"), edgecolor=SURFACE)
            ax.bar_label(b, fmt="%.2f", fontsize=9)
        ax.set_xticks(range(4), [f"{tg}\n{pr} pairs" for tg, pr in combos], fontsize=8)
        ax.set_ylim(0.5, 0.92)
        ax.set_title(f"{title}: $\\lambda$ ranking of 35 settings")
        grid(ax)
    axs[0].set_ylabel("Spearman with the true order")
    save(fig, "w_pairs_target")


def sorting_methods():
    r = rank_report("rank_direction_random")
    names = [("with reference: change model", "with reference: change model"), ("with reference: one trace", "with reference: one-trace model"),
             ("without reference: one trace", "no reference: one-trace model"),
             ("without reference: screen vs screen, mean change", "screen vs screen: mean change"),
             ("without reference: screen vs screen, wins", "screen vs screen: wins"),
             ("without reference: screen vs screen, bradley-terry", "screen vs screen: Bradley-Terry")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, cell, title in ((axs[0], "trained", "trained amplifiers"), (axs[1], "new", "NEW amplifier")):
        for k, (key, lab) in enumerate(names):
            v = [x[1] for x in r[cell][key]]
            col = BLUE if "screen vs screen" in lab else (GREY if "one-trace" in lab else ORANGE)
            ax.plot([0, 1, 2], v, "-o", color=col, lw=(2.4 if "Bradley" in lab else 1.2), ms=(7 if "Bradley" in lab else 4),
                    alpha=(1 if "Bradley" in lab or "change model" in lab else 0.6), label=lab)
        ax.set_xticks([0, 1, 2], ["1:0\n$\\sigma_e$ only", "0.5:0.5", "0:1\n$\\lambda$ only"])
        ax.set_title(title)
        grid(ax)
    axs[0].set_ylabel("Spearman with the true order")
    axs[1].legend(fontsize=7.5, loc="lower left")
    save(fig, "w_sorting_methods")


# ------------------------------------------------------------------------------------------------ funnel replay
def replay():
    if not REPLAY.exists():
        print("no replay yet:", REPLAY)
        return
    z = np.load(REPLAY, allow_pickle=True)
    truth = z["truth_amp"].mean(0)[:, 1]                    # board mean lambda effect (log)
    s = z["bt_new_w0"]
    n = len(truth)
    rho = spearmanr(truth, s)[0]
    top = np.argsort(s)[:5]
    best5 = np.argsort(truth)[:5]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(100 * truth, s, s=40, color=GREY, edgecolor=SURFACE, zorder=2, label="a setting (screen)")
    ax.scatter(100 * truth[top], s[top], s=110, color=BLUE, edgecolor=INK, zorder=3, label="shortlisted (top 5 by the model)")
    ax.scatter(100 * truth[best5], s[best5], s=220, facecolor="none", edgecolor=GREEN, lw=2, zorder=4, label="truly best 5")
    ax.axhline(np.sort(s)[4] + 1e-9, color=BLUE, ls="--", lw=1)
    ax.set_xlabel(r"true effect on $\lambda$: log $\lambda$ $-$ log $\lambda_{\rm nominal}$ (%), mean of 3 amplifiers")
    ax.set_ylabel("Bradley-Terry score (lower = predicted better)")
    ax.set_title(f"{n} first-night settings ranked from traces only (new amplifier): Spearman {rho:.2f}")
    ax.legend(fontsize=8, loc="upper left")
    grid(ax, "both")
    save(fig, "w_replay_scatter")

    # regret of the best of the top k
    k = np.arange(1, n + 1)
    tj = truth - truth.min()
    reg_model = [100 * tj[np.argsort(s)[:kk]].min() for kk in k]
    rng = np.random.default_rng(0)
    reg_rand = [100 * np.mean([tj[rng.choice(n, kk, replace=False)].min() for _ in range(3000)]) for kk in k]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.step(k, reg_rand, where="mid", color=GREY, lw=2, label="random picks (average)")
    ax.step(k, reg_model, where="mid", color=BLUE, lw=2.4, label="model's top k")
    ax.axvline(5, color=INK2, ls=":", lw=1)
    ax.text(5.3, max(reg_rand) * 0.9, "5 images", fontsize=8, color=INK2)
    ax.set_xlabel("number of settings imaged (k)")
    ax.set_ylabel(r"$\lambda$ regret of the best imaged (%)")
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks(range(1, 21))
    ax.legend(fontsize=8)
    grid(ax, "both")
    save(fig, "w_replay_regret")

    # the comparison matrix (margins of P(j worse than i)), board mean, ordered by the truth and by the model
    D = z["D_new"][..., 1]                               # amp x i x j: P(j worse than i) - 0.5
    import shortlist as sl
    M = np.nanmean([sl.antisymmetric(x) for x in D], 0)   # M[i, j] > 0: i beats j
    for tag, o, lab in (("truth", np.argsort(truth), "ordered by the TRUE effect"), ("model", np.argsort(s), "ordered by the model's score")):
        fig, ax = plt.subplots(figsize=(4.6, 4.3))
        im = ax.imshow(M[np.ix_(o, o)], cmap="RdBu", vmin=-0.5, vmax=0.5)
        ax.set_xlabel("setting j")
        ax.set_ylabel("setting i")
        ax.set_title(lab, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("P(i lower $\\lambda$ than j) $-$ 0.5", fontsize=8)
        save(fig, f"w_replay_matrix_{tag}")

    # Bradley-Terry iterations (the animation frames)
    H = z["bthist_new_w0"]
    true_rank = np.argsort(np.argsort(truth))
    for f, it in enumerate((0, 1, 3, 10, 59)):
        fig, ax = plt.subplots(figsize=(6.6, 3.9))
        sc_ = H[it] if it < len(H) else H[-1]
        r = spearmanr(truth, sc_)[0]
        ax.scatter(true_rank + 1, sc_, c=100 * truth, cmap="viridis", s=50, edgecolor=INK2, lw=0.4)
        ax.set_xlabel("true rank of the setting (1 = lowest $\\lambda$)")
        ax.set_ylabel("score $-$log $p_i$")
        ax.set_ylim(H.min() - 0.2, H.max() + 0.2)
        ax.set_title(f"MM iteration {it + 1}: Spearman {r:.2f}")
        grid(ax, "both")
        save(fig, f"w_bt_iter{f}")

    # the sigma_e number attached to each pick
    st, sp, amp, nom = z["sigma_true"], z["sigma_cv"], z["sigma_amp"], z["sigma_nominal"]
    err = 100 * np.sqrt(np.nanmean((sp - st) ** 2))
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    lo, hi = np.exp(np.nanmin(st)) * 0.9, np.exp(np.nanmax(st)) * 1.1
    xx = np.linspace(lo, hi, 10)
    ax.fill_between(xx, xx * np.exp(-err / 100), xx * np.exp(err / 100), color=GRID, label=f"$\\pm${err:.0f} % (cross-validated)")
    ax.plot(xx, xx, color=INK2, lw=1)
    for c, col in zip((0, 1, 3), (BLUE, ORANGE, GREEN)):
        m = amp == c
        ax.scatter(np.exp(st[m]), np.exp(sp[m]), s=26, color=col, edgecolor=SURFACE, lw=0.5, label=f"ch{c}")
    ax.set_xlabel(r"image $\sigma_e$ (e$^-$)")
    ax.set_ylabel(r"predicted from one trace + amplifier (e$^-$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper left")
    grid(ax, "both")
    save(fig, "w_noise_attached")


# ------------------------------------------------------------------------------------------------ v2 design
def design_coverage():
    import scorecard as sc
    _, _, d = sc.load((0, 1, 3))
    nom = d[d.packet_id == "L1_01"].iloc[0]
    v1 = json.loads((ROOT / "lvl1/packet_plan/plan.json").read_text())["packets"]
    v2 = json.loads((ROOT / "lvl1/packet_plan_v2/plan.json").read_text())["packets"]
    pairs = [("delayInt", "VR1", "delayInt (ns)", "VR1 (V)"), ("delay_Ha", "VDD1", "delay_H (ns)", "VDD1 (V)")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.0))
    for ax, (a, b, la, lb) in zip(axs, pairs):
        base = (float(nom[f"theta.{a}"]), float(nom[f"theta.{b}"]))
        for plan, mk, col, lab in ((v2, "o", BLUE, "v2 sessions D+E (joint Sobol design)"), (v1, "s", ORANGE, "v1 (first night)")):
            pts = [(p["changes"].get(a, base[0]), p["changes"].get(b, base[1])) for p in plan
                   if p.get("session") in ("A", "B", "C", "D", "E")]
            pts = np.array(pts, float)
            ax.scatter(pts[:, 0], pts[:, 1], marker=mk, s=(34 if plan is v2 else 70), color=col, alpha=(0.75 if plan is v2 else 0.95),
                       edgecolor=SURFACE, lw=0.6, label=lab, zorder=(2 if plan is v2 else 3))
        ax.scatter(*base, marker="*", s=260, color=INK, zorder=4, label="nominal")
        ax.set_xlabel(la)
        ax.set_ylabel(lb)
        grid(ax, "both")
    axs[0].legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.18), ncol=3)
    save(fig, "w_design_coverage")


def v2_budget():
    plan = json.loads((ROOT / "lvl1/packet_plan_v2/plan.json").read_text())["packets"]
    cats = [("D", "nominal", lambda p: p["purpose"].startswith("nominal"), INK),
            ("D", "replicate", lambda p: p["purpose"].startswith("REPLICATE"), GREY),
            ("D", "joint (Sobol)", lambda p: p["purpose"].startswith("joint"), BLUE),
            ("E", "nominal", lambda p: p["purpose"].startswith("nominal"), INK),
            ("E", "joint, held out", lambda p: p["purpose"].startswith("joint"), GREEN),
            ("F", "nominal", lambda p: p["purpose"].startswith("nominal"), INK),
            ("F", "screens (traces only)", lambda p: p["purpose"].startswith("funnel screen"), "#8fb6e8"),
            ("F", "shortlist (model's picks)", lambda p: "SHORTLIST" in p["purpose"], ORANGE),
            ("F", "controls (random picks)", lambda p: "CONTROL" in p["purpose"], PURPLE)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.0), gridspec_kw={"wspace": 0.4})
    for ax, what in ((axs[0], "packets"), (axs[1], "hours")):
        for row, s in enumerate(("D", "E", "F")):
            left = 0
            for ss, lab, f, col in cats:
                if ss != s:
                    continue
                sel = [p for p in plan if p["session"] == s and f(p)]
                v = len(sel) if what == "packets" else sum(p["minutes"] for p in sel) / 60
                ax.barh(row, v, left=left, color=col, edgecolor=SURFACE, height=0.6)
                if v > (11 if what == "packets" else 1.2):
                    ax.text(left + v / 2, row, f"{lab}\n{v:.0f}" if what == "packets" else f"{v:.1f} h", ha="center",
                            va="center", fontsize=9, color="white")
                left += v
            ax.text(left, row, f"  {left:.0f}" if what == "packets" else f"  {left:.1f} h", va="center", fontsize=11, fontweight="bold")
        ax.set_yticks(range(3), ["D: training", "E: held-out test", "F: funnel test"])
        ax.invert_yaxis()
        ax.set_title(f"v2 plan: {what}")
        grid(ax, "x")
    save(fig, "w_v2_budget")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:]
    for f in (night, label_noise, drift_bug, cv_models, scorecard_grid, base_rate, ch2_offset, kind_heatmap,
              combined_models, direction_weight, pairs_target, sorting_methods, replay, design_coverage, v2_budget):
        if not only or f.__name__ in only:
            f()


if __name__ == "__main__":
    main()
