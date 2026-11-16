"""
report_v3.py -- the figures and tables of the v3 (voltage-only) report: what every voltage does to every output, and how
well every model predicts the outputs and their changes, with the frozen top-3 models highlighted.

Inputs: the session-G scorecard (scorecard.json, predictions.pkl), the session-H scorecard (optional), the frozen choice
(lvl1/freeze_choice_v3.py), parameter_influence.py --noise-components (influence.json).

    .venv/bin/python lvl1/report_v3.py --scorecard processed_v3n/lvl1_results/scorecard_G \\
        --holdout processed_v3n/lvl1_results/scorecard_H --frozen lvl1/frozen_choice_v3.json \\
        --influence processed_v3n/lvl1_results/parameter_influence_noise/influence.json \\
        --out docs/overleaf_v3_report/figs
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correlation_v3 import CMAP, INK, MUTED, nice_knob  # noqa: E402

T = ("sigma_e", "lambda", "gain")
TN = {"sigma_e": "noise σe", "lambda": "dark charge λ", "gain": "gain"}
TNX = {"sigma_e": "noise $\\sigma_e$", "lambda": "dark charge $\\lambda$", "gain": "gain"}
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b7b6b0"
AMPC = {"0": "#2a78d6", "1": "#eb6834", "3": "#1baf7a"}
OUTS = ["sigma_e", "lambda", "gain", "white noise", "1/f corner", "Allan dev n=50", "power 0.1-1 MHz",
        "power 1-3 MHz", "power 3-7.5 MHz", "slow/fast CDS power"]
OUTN = {"sigma_e": "σe (image noise)", "lambda": "λ (dark charge)", "gain": "gain", "white noise": "trace white noise",
        "1/f corner": "trace 1/f corner", "Allan dev n=50": "trace Allan dev (n=50)",
        "power 0.1-1 MHz": "trace power 0.1–1 MHz", "power 1-3 MHz": "trace power 1–3 MHz",
        "power 3-7.5 MHz": "trace power 3–7.5 MHz", "slow/fast CDS power": "trace slow/fast CDS power"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def tex(s: str) -> str:
    return (s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("σe", "$\\sigma_e$")
            .replace("λ", "$\\lambda$").replace("→", "$\\to$").replace("−", "$-$").replace("–", "--"))


# --------------------------------------------------------------------------- parameter effects
def effects(inf: dict, out: Path, md: list, tx: list):
    tg = inf["targets"]
    knobs = list(tg["sigma_e"]["knobs"])
    outs = [o for o in OUTS if o in tg]
    E = np.array([[tg[o]["knobs"][k]["pct_per_sd"] for o in outs] for k in knobs])
    lo = np.array([[tg[o]["knobs"][k]["pct_per_sd_68"][0] for o in outs] for k in knobs])
    hi = np.array([[tg[o]["knobs"][k]["pct_per_sd_68"][1] for o in outs] for k in knobs])
    sig = (lo > 0) | (hi < 0)
    S = np.array([[tg[o]["knobs"][k].get("sobol_total", np.nan) for o in outs] for k in knobs])
    names = [nice_knob(k) for k in knobs]

    # F1 effect map: % per SD, colour scaled per column (each output has its own scale), text = value
    fig, ax = plt.subplots(figsize=(12, 8.5))
    scale = np.maximum(np.nanmax(np.abs(E), 0), 1.0)          # an output that barely moves (< 1 %/SD) stays pale
    ax.imshow(E / scale, cmap=CMAP, vmin=-1, vmax=1, aspect="auto")
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            strong = abs(E[i, j]) / scale[j] > 0.6
            ax.text(j, i, f"{E[i, j]:+.1f}".replace("-", "−"), ha="center", va="center", fontsize=7.5,
                    color=("white" if strong else INK) if sig[i, j] else MUTED, fontweight="bold" if sig[i, j] else "normal")
    ax.set_xticks(range(len(outs)), [OUTN[o] + f"\n(R² {tg[o]['r2_cv_trees']:.2f})" for o in outs], rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.axvline(2.5, color=INK, lw=1.2)
    ax.set_title("What each voltage does to each output: % change per 1 SD of the voltage (all voltages fitted together).\n"
                 "Colour scaled per column (at least ±1 %/SD); bold = 68 % interval excludes 0, faded = not distinguishable from 0. "
                 "R² = how well the voltages predict that output (trees, cross-validated)", loc="left", fontsize=9)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out / "effect_map.png", dpi=160)
    plt.close(fig)

    # F2 share of variance (Sobol total) per output
    fig, ax = plt.subplots(figsize=(12, 8.5))
    S2 = np.where(np.isfinite(S), S, 0)
    ax.imshow(S2, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(S2.shape[0]):
        for j in range(S2.shape[1]):
            if S2[i, j] >= 0.02:
                ax.text(j, i, f"{S2[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if S2[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(outs)), [OUTN[o] for o in outs], rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Share of each output's variation over the voltage window due to each voltage (Sobol total index, "
                 "from the tree model;\nblank = below 0.02). Read only the columns with a good R² in the effect map.",
                 loc="left", fontsize=9)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out / "sobol_map.png", dpi=160)
    plt.close(fig)

    # F3 per-amplifier effects of the top knobs for the three labels
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, t in zip(axs, T):
        kn = sorted(knobs, key=lambda k: -abs(tg[t]["knobs"][k]["pct_per_sd"]))[:6][::-1]
        y = np.arange(len(kn))
        for k_i, k in enumerate(kn):
            v = tg[t]["knobs"][k]
            ax.plot(v["pct_per_sd_68"], [k_i, k_i], color=INK, lw=2, solid_capstyle="round")
            ax.plot(v["pct_per_sd"], k_i, "o", color=INK, ms=6, zorder=3)
            for a, c in AMPC.items():
                if a in v["per_amp_pct_per_sd"]:
                    ax.plot(v["per_amp_pct_per_sd"][a], k_i + 0.22, "D", color=c, ms=5, mec="white", mew=1.5,
                            label=f"ch{a}" if k_i == 0 else None)
        ax.axvline(0, color=GRAY, lw=1)
        ax.set_yticks(y, [nice_knob(k) for k in kn])
        ax.set_xlabel("% change per 1 SD")
        ax.set_title(TN[t], loc="left")
    axs[0].legend(frameon=False, loc="lower right", title="per amplifier", fontsize=8)
    fig.suptitle("The six largest voltage effects per label: all amplifiers (black, 68 % interval) and each amplifier "
                 "(diamonds)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "effects_per_amp.png", dpi=160)
    plt.close(fig)

    # tables: top 5 per output, % per SD and per V
    md += ["## What each voltage does (top 5 per output, % per SD [68 %]; per V)", "",
           "| output | R² | 1st | 2nd | 3rd | 4th | 5th |", "|---|---|---|---|---|---|---|"]
    tx += ["\\begin{table}[H]\\centering\\scriptsize",
           "\\caption{The five voltages that move each output most: \\% change per 1~SD of the voltage (all voltages "
           "fitted together), and per volt in brackets. * = 68\\% interval excludes zero. $R^2$: how well the voltages "
           "predict the output (cross-validated trees).}\\label{tab:effects}",
           "\\begin{tabular}{p{2.6cm}cp{2.1cm}p{2.1cm}p{2.1cm}p{2.1cm}p{2.1cm}}\\toprule",
           "output & $R^2$ & 1st & 2nd & 3rd & 4th & 5th\\\\\\midrule"]
    for j, o in enumerate(outs):
        order = np.argsort(-np.abs(E[:, j]))[:5]
        cells_md, cells_tx = [], []
        for i in order:
            v = tg[o]["knobs"][knobs[i]]
            star = "*" if sig[i, j] else ""
            pu = v["pct_per_unit"]
            cells_md.append(f"{names[i]} {E[i, j]:+.1f}{star} ({pu:+.1f}/V)")
            cells_tx.append(f"{names[i]} {E[i, j]:+.1f}{star} ({pu:+.1f}/V)".replace("-", "$-$"))
        md.append(f"| {OUTN[o]} | {tg[o]['r2_cv_trees']:.2f} | " + " | ".join(cells_md) + " |")
        tx.append(f"{tex(OUTN[o])} & {tg[o]['r2_cv_trees']:.2f} & " + " & ".join(cells_tx) + "\\\\")
    tx += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    md.append("")
    return knobs


# --------------------------------------------------------------------------- model performance
def model_bars(sc: dict, sh: dict | None, frozen: dict, out: Path, md: list, tx: list):
    # absolute
    fig, axs = plt.subplots(2, 3, figsize=(14, 9))
    for r, cell in enumerate(("trained", "new")):
        for c, t in enumerate(T):
            ax = axs[r, c]
            items = sorted(((v[t]["error"], k) for k, v in sc["absolute"][cell].items() if t in v and finite(v[t]["error"])),
                           reverse=True)
            top = [x["model"] for x in frozen["absolute"][cell][t]]
            y = np.arange(len(items))
            cols = [BLUE if k in top else (GRAY if "mean" in k else "#9fc0ea") for _, k in items]
            ax.barh(y, [e for e, _ in items], color=cols, height=0.7)
            if sh:
                hc = cell + " E"
                for i, (_, k) in enumerate(items):
                    v = sh["absolute"].get(hc, {}).get(k, {}).get(t)
                    if v and finite(v["error"]):
                        ax.plot(v["error"], i, "D", color=ORANGE, ms=5, mec="white", mew=1.2,
                                label="session H (sealed)" if i == len(items) - 1 or (i == 0 and r == c == 0) else None)
            for i, (e, k) in enumerate(items):
                ax.text(e, i, f" {e:.1f}", va="center", fontsize=7, color=INK)
            ax.set_yticks(y, [k + (" ★" if k in top else "") for _, k in items], fontsize=7.5)
            ax.set_xlabel("typical error %")
            ax.set_title(f"{TN[t]} — {'amplifiers seen in training' if cell == 'trained' else 'new amplifier'}",
                         loc="left", fontsize=9)
    h, l = axs[0, 0].get_legend_handles_labels()
    if h:
        axs[0, 0].legend(h[:1], l[:1], frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("ABSOLUTE models: predict σe, λ, gain from one setting. Bars = session G (cross-validated); "
                 "★ dark blue = frozen top 3; orange diamonds = sealed session H; gray = baselines",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "models_absolute.png", dpi=160)
    plt.close(fig)

    # relative: size error and spearman, random pairs
    fig, axs = plt.subplots(2, 3, figsize=(14, 10))
    for r, cell in enumerate(("trained", "new")):
        for c, t in enumerate(T):
            ax = axs[r, c]
            items = sorted(((v["random pairs"][t]["size_error"], k) for k, v in sc["relative"][cell].items()
                            if t in v["random pairs"] and finite(v["random pairs"][t].get("size_error"))), reverse=True)
            top = [x["model"] for x in frozen["relative"][cell][t]]
            y = np.arange(len(items))
            ax.barh(y, [e for e, _ in items], color=[BLUE if k in top else (GRAY if k == "no change" else "#9fc0ea")
                                                       for _, k in items], height=0.7)
            for i, (e, k) in enumerate(items):
                rho = sc["relative"][cell][k]["random pairs"][t].get("spearman")
                ax.text(e, i, f" {e:.1f}" + (f"  ρ {rho:.2f}" if finite(rho) else ""), va="center", fontsize=6.5)
                if sh:
                    v = sh["relative"].get(cell + " E", {}).get(k, {}).get("random pairs", {}).get(t)
                    if v and finite(v.get("size_error")):
                        ax.plot(v["size_error"], i, "D", color=ORANGE, ms=4.5, mec="white", mew=1.2)
            ax.set_yticks(y, [k.replace(" (sigma_e)", "") + (" ★" if k in top else "") for _, k in items], fontsize=6.5)
            ax.set_xlabel("size error of the predicted change, %")
            ax.set_title(f"{TN[t]} — {'trained amplifiers' if cell == 'trained' else 'new amplifier'}", loc="left", fontsize=9)
            ax.set_xlim(0, max(e for e, _ in items) * 1.35)
    fig.suptitle("CHANGE models: predict how much σe, λ, gain change between two settings (random pairs). "
                 "ρ = rank correlation predicted vs true; ★ = frozen top 3; orange = session H",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "models_change.png", dpi=160)
    plt.close(fig)

    # tables: frozen top 3 (G and H)
    for kind, key, unit in (("absolute", "error", "typical error %"), ("relative", "size_error", "size error % (random pairs)")):
        md += [f"## Frozen top 3, {kind} ({unit}; session G → sealed session H)", "",
               "| cell | target | 1st | 2nd | 3rd |", "|---|---|---|---|---|"]
        tx += ["\\begin{table}[H]\\centering\\scriptsize",
               f"\\caption{{The frozen top three {'absolute' if kind == 'absolute' else 'change'} models per target "
               f"({tex(unit)}): session G (cross-validated, used to choose) $\\to$ sealed session H (scored once). "
               "Lower is better.}",
               "\\begin{tabular}{llp{4cm}p{4cm}p{4cm}}\\toprule", "cell & target & 1st & 2nd & 3rd\\\\\\midrule"]
        for cell in ("trained", "new"):
            for t in T:
                cells = []
                for x in frozen[kind][cell][t]:
                    m = x["model"]
                    g = x.get("error_pct", x.get("size_error_pct"))
                    h = None
                    if sh:
                        v = sh[kind].get(cell + " E", {}).get(m, {})
                        v = v.get("random pairs", {}) if kind == "relative" else v
                        h = v.get(t, {}).get(key)
                    cells.append(f"{m}: {g:.1f}" + (f" → {h:.1f}" if finite(h) else ""))
                md.append(f"| {cell} | {TN[t]} | " + " | ".join(cells) + " |")
                tx.append(f"{'trained amps' if cell == 'trained' else 'new amp'} & {tex(TN[t])} & "
                          + " & ".join(tex(c) for c in cells) + "\\\\")
        md.append("")
        tx += ["\\bottomrule\\end{tabular}\\end{table}", ""]


def scatter(sc_dir: Path, frozen: dict, out: Path):
    Y, _, A, P, R = pickle.loads((sc_dir / "predictions.pkl").read_bytes())
    # absolute: frozen #1 per target, new amplifier and trained
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.5))
    for r, cell in enumerate(("trained", "new")):
        for c, t in enumerate(T):
            ax = axs[r, c]
            m = frozen["absolute"][cell][t][0]["model"]
            p = A[cell][m][:, c]
            ok = np.isfinite(p)
            ax.scatter(np.exp(Y[ok, c]), np.exp(p[ok]), s=14, color=BLUE, alpha=0.75, edgecolor="white", lw=0.5)
            lim = [np.exp(min(Y[ok, c].min(), p[ok].min())), np.exp(max(Y[ok, c].max(), p[ok].max()))]
            ax.plot(lim, lim, color=GRAY, lw=1)
            if t == "lambda":
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_xlabel(f"measured {TN[t]}")
            ax.set_ylabel("predicted")
            ax.set_title(f"{m} — {'trained amps' if cell == 'trained' else 'new amp'}", loc="left", fontsize=8.5)
    fig.suptitle("Absolute: predicted vs measured for the frozen #1 model (session G, cross-validated; line = perfect)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "scatter_absolute.png", dpi=160)
    plt.close(fig)

    # change: frozen #1 per target, new amplifier; random pairs (gray) + one-knob within-packet pairs (blue)
    from scorecard_v2 import PAIR_TYPES
    rnd, wp = PAIR_TYPES.index("random pairs"), PAIR_TYPES.index("within packet (image)")
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.5))
    for r, cell in enumerate(("trained", "new")):
        for c, t in enumerate(T):
            ax = axs[r, c]
            m = frozen["relative"][cell][t][0]["model"]
            if m not in R[cell]:
                ax.set_visible(False)
                continue
            p = R[cell][m][:, c]
            for typ, col, lab, s in ((rnd, GRAY, "random pairs", 5), (wp, BLUE, "same packet: nominal → new setting", 14)):
                ok = np.isfinite(p) & (P["type"] == typ)
                ax.scatter(100 * P["Y"][ok, c], 100 * p[ok], s=s, color=col, alpha=0.6 if typ == rnd else 0.85,
                           edgecolor="none" if typ == rnd else "white", lw=0.5, label=lab)
            ok = np.isfinite(p) & np.isin(P["type"], [rnd, wp])
            lim = np.percentile(np.abs(100 * P["Y"][ok, c]), 99.5)
            ax.plot([-lim, lim], [-lim, lim], color=INK, lw=0.8)
            ax.axhline(0, color=GRAY, lw=0.6)
            ax.axvline(0, color=GRAY, lw=0.6)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel(f"true change of log {TN[t]} (×100 ≈ %)")
            ax.set_ylabel("predicted change")
            ax.set_title(f"{m} — {'trained amps' if cell == 'trained' else 'new amp'}", loc="left", fontsize=8.5)
    axs[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Change: predicted vs true change for the frozen #1 model (session G, cross-validated)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "scatter_change.png", dpi=160)
    plt.close(fig)

    # physics models
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, cell in zip(axs[:2], ("trained", "new")):
        p = A[cell]["trace: physics"][:, 0]
        ok = np.isfinite(p)
        ax.scatter(np.exp(Y[ok, 0]), np.exp(p[ok]), s=14, color=AQUA, edgecolor="white", lw=0.5)
        lim = [np.exp(min(Y[ok, 0].min(), p[ok].min())), np.exp(max(Y[ok, 0].max(), p[ok].max()))]
        ax.plot(lim, lim, color=GRAY, lw=1)
        ax.set_xlabel("measured σe (e⁻)")
        ax.set_ylabel("physics prediction")
        ax.set_title(f"trace: physics, absolute — {'trained amps' if cell == 'trained' else 'new amp'}", loc="left", fontsize=9)
    ax = axs[2]
    for m, col in (("trace: physics difference (sigma_e)", GRAY), ("trace: physics difference + correction (sigma_e)", AQUA)):
        p = R["new"][m][:, 0]
        ok = np.isfinite(p) & (P["type"] == rnd)
        ax.scatter(100 * P["Y"][ok, 0], 100 * p[ok], s=5, color=col, alpha=0.6, label=m.replace(" (sigma_e)", ""))
    ax.plot([-40, 40], [-40, 40], color=INK, lw=0.8)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_xlabel("true change of log σe (×100)")
    ax.set_ylabel("predicted")
    ax.set_title("physics change models — new amp", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle("The physics model: σe from the measured trace noise (white + 1/f, through the CDS filter) and the gain",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "physics.png", dpi=160)
    plt.close(fig)


def corr_table(cdir: Path, out: Path, md: list):
    import pandas as pd
    P = pd.read_csv(cdir / "corr_partial.csv", index_col=0)
    C = pd.read_csv(cdir / "corr_full.csv", index_col=0)
    thr = float((cdir / "correlations.md").read_text().split("|r| >= ")[1].split(".")[0] + "." +
                (cdir / "correlations.md").read_text().split("|r| >= ")[1].split(".")[1][:2])
    tx = ["\\begin{table}[H]\\centering\\scriptsize",
          f"\\caption{{The five voltages most correlated with each output \\emph{{on their own}} (partial correlation: the other "
          f"voltages and the amplifier held fixed), with the plain within-amplifier correlation in brackets. "
          f"* = significant ($|r|\\ge{thr:.2f}$).}}\\label{{tab:corr}}",
          "\\begin{tabular}{p{3cm}p{2.2cm}p{2.2cm}p{2.2cm}p{2.2cm}p{2.2cm}}\\toprule",
          "output & 1st & 2nd & 3rd & 4th & 5th\\\\\\midrule"]
    md += ["## Partial correlation, top 5 per output (plain correlation in brackets)", "",
           "| output | 1st | 2nd | 3rd | 4th | 5th |", "|---|---|---|---|---|---|"]
    for o in P.columns:
        top = P[o].abs().sort_values(ascending=False).index[:5]
        cells = [f"{k} {P.loc[k, o]:+.2f}{'*' if abs(P.loc[k, o]) >= thr else ''} ({C.loc[k, o]:+.2f})" for k in top]
        md.append(f"| {o} | " + " | ".join(cells) + " |")
        tx.append(tex(o) + " & " + " & ".join(c.replace("-", "$-$") for c in cells) + "\\\\")
    tx += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    md.append("")
    (out / "tables_corr.tex").write_text("\n".join(tx) + "\n")


def subtle_table(sdir: Path, out: Path, md: list):
    import pandas as pd
    df = pd.read_csv(sdir / "subtle.csv")
    models = [("settings change: trees", "settings"), ("trace change: trees", "traces"),
              ("ensemble: trace models", "traces (ens.)"), ("settings + trace: combined trees", "settings + traces"),
              ("ensemble: settings + trace", "settings + traces (ens.)")]
    tx = ["\\begin{table}[H]\\centering\\scriptsize",
          "\\caption{Small effects: rank correlation $\\rho$ of predicted vs true change (size error \\% in brackets), all "
          "scored pairs against pairs where the dominant knob moved $<0.25$~SD. True RMS = how big the true changes are.}"
          "\\label{tab:subtle}",
          "\\begin{tabular}{lllr r" + "l" * len(models) + "}\\toprule",
          "cell & label & pairs & n & true RMS & " + " & ".join(m[1] for m in models) + "\\\\\\midrule"]
    for _, r in df.iterrows():
        tx.append(f"{'trained' if r.cell == 'trained' else 'new amp'} & {tex(TN[r.target])} & {r.subset} & {r.n} & "
                  f"{r.true_rms:.1f}\\% & " + " & ".join(f"{r[m + ' rho']:.2f} ({r[m + ' size']:.1f})" for m, _ in models)
                  + "\\\\")
    tx += ["\\bottomrule\\end{tabular}\\end{table}", ""]
    (out / "tables_subtle.tex").write_text("\n".join(tx) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scorecard", required=True, help="session-G scorecard folder")
    ap.add_argument("--holdout", help="session-H scorecard folder")
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--influence", required=True)
    ap.add_argument("--corr", help="correlation_v3.py output folder (corr_partial.csv)")
    ap.add_argument("--subtle", help="subtle_v3.py output folder (subtle.csv)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = json.loads((Path(args.scorecard) / "scorecard.json").read_text())
    sh = json.loads((Path(args.holdout) / "scorecard.json").read_text()) if args.holdout and (Path(args.holdout) / "scorecard.json").exists() else None
    frozen = json.loads(Path(args.frozen).read_text())
    inf = json.loads(Path(args.influence).read_text())
    md, tx_e, tx_m = ["# v3 report tables", ""], [], []
    effects(inf, out, md, tx_e)
    model_bars(sc, sh, frozen, out, md, tx_m)
    scatter(Path(args.scorecard), frozen, out)
    (out / "tables_effects.tex").write_text("\n".join(tx_e) + "\n")
    (out / "tables_frozen.tex").write_text("\n".join(tx_m) + "\n")
    if args.corr:
        corr_table(Path(args.corr), out, md)
    if args.subtle:
        subtle_table(Path(args.subtle), out, md)
    (out / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
