"""
report_models.py -- the generated part of the first-data report's section "The models one by one"
(docs/overleaf_first_data_2026-11-03): every model's performance at the feature set chosen on A+B (cross-validation
and session C, paired against settings + amplifier), and what every model relies on (lvl1/explain_models.py).

Writes into the report folder:
    models_performance.tex          every model and target: set, A+B CV error, session C error, paired difference
    models/<model>.tex              per model: performance, input groups it relies on, top inputs, fitted parameters
    figures/fig7_all_models.png     every model's error, CV and session C
    figures/fig8_reliance.png       what every model's predictions move with (input-group shuffles)
    figures/fig9_physics_terms.png  the physics-structured sigma_e, term by term

    .venv/bin/python lvl1/report_models.py [--explain processed/lvl1_results/2026-11-03_explain/explain.json]
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lvl1"))
sys.path.insert(0, str(ROOT / "src"))

import plot_first_data as pf  # noqa: E402
import train_lvl1 as tl  # noqa: E402
from ml.evaluate import bootstrap_weights, percentile_interval  # noqa: E402

LADDER = ROOT / "processed_AB" / "lvl1_results" / "2026-11-03_ladder_raw" / "report.json"
AMP_BASELINE = ROOT / "processed_AB" / "lvl1_results" / "2026-11-03_amp_baseline" / "report.json"
HOLDOUT = ROOT / "processed" / "lvl1_results" / "2026-11-03_holdoutC"
REPORT = ROOT / "docs" / "overleaf_first_data_2026-11-03"
TARGETS = tl.TARGETS
TNAME = {"sigma_e": r"$\sigma_e$", "lambda": r"$\lambda$", "gain": "gain"}
REF = "settings_amp_trees"
#: display order: the baselines, then the models on scalar features, the curve models, the physics family
ORDER = ["mean", "settings", "settings_amp", "settings_amp_trees", "ridge", "trees", "gp", "gp_dim", "nn", "cnn",
         "pretrained", "deep_kernel", "physics", "physics_white", "physics_joint", "sets", "physics_hier"]
FROZEN = {"sigma_e": "gp_dim", "lambda": "pretrained", "gain": "gp_dim"}
COARSE = ["settings", "amplifier identity", "own trace", "reference trace", "packet state", "trace noise term",
          "own curves", "reference curves", "trace sets"]


def tex(s: str) -> str:
    """Plain text -> LaTeX text."""
    rep = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "{": r"\{",
           "}": r"\}", "~": r"\~{}", "^": r"\^{}"}
    return "".join(rep.get(c, c) for c in str(s))


def mname(m: str) -> str:
    return r"\texttt{" + tex(m) + "}"


# --------------------------------------------------------------------------- #
# performance                                                                  #
# --------------------------------------------------------------------------- #

class Performance:
    """CV and session C errors and the paired session C comparison with settings + amplifier (trees)."""

    def __init__(self, sets: dict):
        self.sets = sets
        self.ladder = json.loads(LADDER.read_text())["results"]
        self.ampb = json.loads(AMP_BASELINE.read_text())["results"]
        self.hold = {p.name: json.loads((p / "report.json").read_text()) for p in HOLDOUT.iterdir()
                     if (p / "report.json").exists()}
        self._pred = {}
        base = self.pred("baselines")
        self.c_rows = base[base["held out"].astype(str).str.lower().eq("true")].reset_index(drop=True)
        self.keys = list(zip(self.c_rows.packet_id, self.c_rows.channel))
        self.W = bootstrap_weights(self.c_rows.packet_id.to_numpy(), 1000, seed=1)
        self.Y = np.column_stack([self.c_rows[f"true log {t}"] for t in TARGETS])

    def pred(self, folder: str) -> pd.DataFrame:
        if folder not in self._pred:
            self._pred[folder] = pd.read_csv(HOLDOUT / folder / "predictions.csv")
        return self._pred[folder]

    def c_mean(self, model: str, t: str) -> np.ndarray:
        s = self.sets[model][t]
        p = self.pred("baselines" if model in pf_baselines() else s)
        p = p[p["held out"].astype(str).str.lower().eq("true")].set_index(["packet_id", "channel"])
        return p.loc[self.keys, f"{model}: log {t}"].to_numpy(float)

    def cv(self, model: str, t: str) -> float:
        s = self.sets[model][t]
        if model in pf_baselines():
            return 100 * self.ampb[model]["scores"][t]["error"]
        r = self.ladder.get(f"{model} [{s}]") or self.hold[s]["results"][model]
        return 100 * r["scores"][t]["error"]

    def row(self, model: str, t: str) -> dict:
        i = TARGETS.index(t)
        m, ref = self.c_mean(model, t), self.c_mean(REF, t)
        y = self.Y[:, i]
        err = lambda a: np.sqrt(np.mean((a - y) ** 2))
        boot = np.sqrt(self.W @ ((m - y) ** 2) / self.W.sum(1))
        out = {"set": self.sets[model][t], "cv": self.cv(model, t), "c": 100 * err(m),
               "c_interval": [100 * v for v in percentile_interval(boot)["68"]]}
        if model != REF:
            d = np.sqrt(self.W @ ((m - y) ** 2) / self.W.sum(1)) - np.sqrt(self.W @ ((ref - y) ** 2) / self.W.sum(1))
            iv = percentile_interval(d)["95"]
            out.update(delta=100 * (err(m) - err(ref)), delta_95=[100 * v for v in iv], p_better=float(np.mean(d < 0)))
        return out


def pf_baselines():
    return ("mean", "settings", "settings_amp", "settings_amp_trees")


def performance_table(perf: Performance) -> str:
    lines = [r"\begin{tabular}{llrrrl}", r"\toprule",
             r"model & set (chosen on A+B) & A+B CV & session C & C $-$ settings + amp. [95\,\%] & P(better) \\",
             r"\midrule"]
    for t in TARGETS:
        lines += [r"\multicolumn{6}{l}{\textbf{" + TNAME[t] + r"} \quad error in \%; reference: settings + amplifier (trees)} \\",
                  r"\midrule"]
        for m in ORDER:
            r = perf.row(m, t)
            frozen = r"$^{\dagger}$" if FROZEN[t] == m else ""
            if m == REF:
                d, p = "reference", ""
            else:
                lo, hi = r["delta_95"]
                sig = lo > 0 or hi < 0
                d = f"{r['delta']:+.1f} [{lo:+.1f}, {hi:+.1f}]"
                d = r"\textbf{" + d + "}" if sig else d
                p = f"{r['p_better']:.2f}"
            lines.append(f"{mname(m)}{frozen} & {tex(r['set'])} & {r['cv']:.1f} & {r['c']:.1f} & {d} & {p} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# per-model fragments                                                          #
# --------------------------------------------------------------------------- #

def fit_of(ex: dict, model: str, t: str) -> dict:
    return ex[f"{model} [{ex['_chosen_sets'][model][t]}]"]


def reliance_table(ex: dict, model: str) -> str:
    """Coarse input groups: prediction change when shuffled across all rows / within each amplifier, and the session C
    error increase when shuffled among the C rows."""
    groups = [g for g in COARSE if any(g in fit_of(ex, model, t).get("coarse", {}) for t in TARGETS)]
    if not groups:
        return ""
    lines = [r"\begin{tabular}{l" + "rrr" * 3 + "}", r"\toprule",
             " & " + " & ".join(r"\multicolumn{3}{c}{" + TNAME[t] + "}" for t in TARGETS) + r" \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"input group & all & in amp. & C loss & all & in amp. & C loss & all & in amp. & C loss \\", r"\midrule"]
    for g in groups:
        cells = []
        for t in TARGETS:
            c = fit_of(ex, model, t).get("coarse", {}).get(g)
            if c is None:
                cells += ["--"] * 3
                continue
            cells += [f"{c['sensitivity'][t]:.1f}", f"{c['sensitivity_within_amp'][t]:.1f}", f"{c['loss_C'][t]:+.1f}"]
        n = max(fit_of(ex, model, t).get("coarse", {}).get(g, {}).get("n", 0) for t in TARGETS)
        lines.append(f"{tex(g)} ({n}) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def top_table(columns: dict[str, list[str]], k: int = 6) -> str:
    """Up to k ranked entries per target, side by side."""
    lines = [r"\begin{tabular}{rp{0.29\textwidth}p{0.29\textwidth}p{0.29\textwidth}}", r"\toprule",
             " & " + " & ".join(TNAME[t] for t in TARGETS) + r" \\", r"\midrule"]
    for i in range(k):
        cells = [columns[t][i] if i < len(columns[t]) else "" for t in TARGETS]
        if not any(cells):
            break
        lines.append(f"{i + 1} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def short(name: str) -> str:
    """A shorter display name of an input."""
    return (name.replace("own: ", "own ").replace("ref: ", "ref ").replace("state: ", "state ")
            .replace("own curve: ", "own curve ").replace("ref curve: ", "ref curve "))


def native_columns(ex: dict, model: str, k: int = 6) -> tuple[dict, list[str], str]:
    """(ranked entries per target, extra lines, what the numbers are) from the fitted parameters."""
    cols, extra, what = {t: [] for t in TARGETS}, [], ""
    if model == "mean":
        nat = fit_of(ex, model, "sigma_e")["native"]
        units = {"sigma_e": r"e$^-$", "lambda": r"e$^-$/pixel", "gain": r"ADU/e$^-$"}
        for t in TARGETS:
            cols[t] = [f"{nat[t]['value']:.3g} {units[t]} (sd {100 * nat[t]['sd_log']:.0f}\\,\\%)"]
        return cols, extra, "the prediction: the geometric mean of the A+B labels (and their spread)"
    for t in TARGETS:
        nat = fit_of(ex, model, t).get("native", {})
        d = nat.get(t, {})
        if "relevance" in d:
            what = r"inverse length scale $1/\ell_j$ of each input (ARD relevance; larger = the function varies faster with it)"
            cols[t] = [f"{tex(short(n))} ({v:.2g})" for n, v in d["relevance"][:k]]
            hp = [f"{h} {d[h]:.2g}" for h in ("amplitude", "noise") if h in d]
            if hp:
                extra.append(f"{TNAME[t]}: " + ", ".join(hp))
        elif "coefficient" in d:
            what = r"coefficient $w_j$, in \% change of the target per standard deviation of the input"
            cols[t] = [f"{tex(short(n))} ({100 * v:+.1f})" for n, v in d["coefficient"][:k]]
            extra.append(f"{TNAME[t]}: noise sd {100 / np.sqrt(d['noise_precision']):.1f}\\,\\%, prior weight sd "
                         f"{100 / np.sqrt(d['weight_precision']):.2f}\\,\\%")
        elif "split_gain_share" in d:
            what = r"share of the total squared-error reduction of all splits made on the input"
            cols[t] = [f"{tex(short(n))} ({v:.2f})" for n, v in d["split_gain_share"][:k] if v > 0]
            extra.append(f"{TNAME[t]}: {d['n_inputs_used']} inputs used in any split")
        elif "terms" in nat:                                           # the physics family
            what = (r"inverse length scales $1/\ell_j$ of the GP that carries the target: $\sigma_e$, the correction "
                    r"$r(x)$; $\lambda$, the $\lambda$ GP; gain, the gain GP $\hat g(x)$")
            part = {"sigma_e": nat["sigma_e"]["correction GP"], "gain": nat["gain"]["gain GP"],
                    "lambda": nat["lambda"]["lambda GP"]}[t]
            rel = part.get("relevance") or part.get("relevance (dark current)") or []
            cols[t] = [f"{tex(short(n))} ({v:.2g})" for n, v in rel[:k]]
            hp = {kk: v for kk, v in part.items() if isinstance(v, float)}
            if hp:
                extra.append(f"{TNAME[t]} GP: " + ", ".join(f"{tex(kk)} {v:.2g}" for kk, v in hp.items()))
            if "relevance (CIC)" in part:
                extra.append(r"$\lambda$, CIC part, top inputs: " + ", ".join(f"{tex(short(n))} ({v:.2g})" for n, v in part["relevance (CIC)"][:4]))
    return cols, extra, what


def physics_terms_line(ex: dict, model: str) -> str:
    nat = fit_of(ex, model, "sigma_e").get("native", {})
    if "terms" not in nat:
        return ""
    rows = ex["_rows"]
    y = np.array(rows["Y"])[:, 0]
    held = np.array(rows["held_out"])
    T, G, R = (np.array(nat["terms"][k]) for k in ("trace", "minus_log_gain", "correction"))
    sd = lambda a: 100 * np.std(a)
    pred = T + G + R
    rmsC = lambda a: 100 * np.sqrt(np.mean((a[held] - y[held]) ** 2))
    gm = np.mean(-G[~held])
    return (r"Over all rows the three terms of $\log\sigma_e$ vary by (sd): trace term " + f"{sd(T):.1f}\\,\\%, "
            r"$-\log\hat g$ " + f"{sd(G):.1f}\\,\\%, correction $r$ {sd(R):.1f}\\,\\% (mean {100 * np.mean(R):+.0f}\\,\\%: "
            r"the image's noise relative to the trace's); the labels vary by " + f"{sd(y):.1f}\\,\\%. "
            r"Session C error with the trace term and the average gain only: " + f"{rmsC(T - gm + np.mean(R[~held])):.1f}\\,\\%; "
            r"with the learned gain: " + f"{rmsC(T + G + np.mean(R[~held])):.1f}\\,\\%; with the correction (the model): "
            f"{rmsC(pred):.1f}\\,\\%. Rows down-weighted as outliers in the correction fit: "
            f"{nat['sigma_e'].get('n_outliers', 0)} of {int((~held).sum())}.")


def model_fragment(ex: dict, perf: Performance, model: str) -> str:
    out = [r"\paragraph{Performance.}"]
    cells = []
    for t in TARGETS:
        r = perf.row(model, t)
        s = (f"{TNAME[t]}: set {tex(r['set'])}, CV {r['cv']:.1f}\\,\\%, session C {r['c']:.1f}\\,\\%")
        if "delta" in r:
            lo, hi = r["delta_95"]
            s += f" ({r['delta']:+.1f} vs settings + amplifier, 95\\,\\% [{lo:+.1f}, {hi:+.1f}]" + (", significant)" if lo > 0 or hi < 0 else ")")
        cells.append(s)
    out.append("; ".join(cells) + ".")
    if model == "mean":
        cols, extra, what = native_columns(ex, model)
        out += [r"\paragraph{Fitted parameters} (" + what + "): " + "; ".join(f"{TNAME[t]} {cols[t][0]}" for t in TARGETS) + "."]
        return "\n".join(out) + "\n"
    rel = reliance_table(ex, model)
    if rel:
        out += [r"\paragraph{What the predictions rely on.} Change of the prediction (RMS, \% of the target) when an input group is "
                r"shuffled across all rows (\emph{all}) or only among the rows of the same amplifier (\emph{in amp.}), and "
                r"the increase of the session C error (percentage points) when it is shuffled among the C rows (\emph{C loss}). "
                r"Number of inputs in brackets.", r"\begin{center}\footnotesize", rel, r"\end{center}"]
    singles = {t: [f"{tex(short(n))} ({v:.1f})" for n, _, v in fit_of(ex, model, t).get("single", {}).get(t, [])[:6]
                   if v >= 0.05] for t in TARGETS}
    if any(singles.values()):
        out += [r"\paragraph{Single inputs that move the prediction most} (shuffled one at a time across all rows; RMS change, \%):",
                r"\begin{center}\footnotesize", top_table(singles), r"\end{center}"]
    cols, extra, what = native_columns(ex, model)
    if any(cols.values()):
        out += [r"\paragraph{Fitted parameters:} " + what + ".", r"\begin{center}\footnotesize", top_table(cols), r"\end{center}"]
        if extra:
            out.append(r"{\footnotesize " + "; ".join(extra) + ".}")
    elif model in ("nn", "cnn", "pretrained"):
        out.append(r"\paragraph{Fitted parameters:} network weights, with no per-input meaning; the shuffles above are "
                   r"the summary of what it uses.")
    line = physics_terms_line(ex, model)
    if line:
        out += [r"\paragraph{The three terms.}", line]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# figures                                                                      #
# --------------------------------------------------------------------------- #

def fig_all_models(perf: Performance, noise: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.4), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for ax, t in zip(axes, TARGETS):
        rows = [perf.row(m, t) for m in ORDER]
        cv = [r["cv"] for r in rows]
        c = [r["c"] for r in rows]
        lo = [r["c"] - r["c_interval"][0] for r in rows]
        hi = [r["c_interval"][1] - r["c"] for r in rows]
        ax.axvline(noise[t], color=pf.INK, linestyle="--", linewidth=1)
        ref = perf.row(REF, t)["c"]
        ax.axvline(ref, color="#86b6ef", linewidth=1.5, zorder=0)
        ax.errorbar(c, y, xerr=[lo, hi], fmt="o", color="#eb6834", markersize=7, elinewidth=1.2, capsize=0,
                    markeredgecolor=pf.SURFACE, label="session C (bar: 68 % packet bootstrap)", zorder=3)
        ax.plot(cv, y, "o", markerfacecolor="none", markeredgecolor="#2a78d6", markersize=7, markeredgewidth=1.5,
                label="A+B cross-validation", zorder=2)
        ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
        ticks = {"sigma_e": [2, 5, 10, 20, 40], "lambda": [5, 10, 20, 40, 80], "gain": [0.5, 1, 2, 4]}[t]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(pf.TARGETS[t], color=pf.INK)
        pf.style(ax)
        ax.grid(axis="x", color=pf.GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("typical error (%), log scale", color=pf.INK2)
        ax.text(noise[t], len(ORDER) - 0.3, " label noise", fontsize=7.5, color=pf.INK, va="bottom")
        ax.text(ref, -1.0, " settings +\n amplifier (C)", fontsize=7.5, color="#2a78d6", va="top")
    axes[0].set_yticks(y, [m + (" †" if m in FROZEN.values() else "") for m in ORDER])
    axes[0].set_ylim(-1.9, len(ORDER) - 0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_reliance(ex: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.2), sharey=True)
    models = [m for m in ORDER if m != "mean"]
    for ax, t in zip(axes, TARGETS):
        M = np.full((len(models), len(COARSE)), np.nan)
        for i, m in enumerate(models):
            for j, g in enumerate(COARSE):
                c = fit_of(ex, m, t).get("coarse", {}).get(g)
                if c is not None:
                    M[i, j] = c["sensitivity"][t]
        vmax = np.nanpercentile(M, 95)
        ax.imshow(np.where(np.isnan(M), 0, M), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isnan(M[i, j]):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=pf.SURFACE, zorder=2))
                    continue
                ax.text(j, i, f"{M[i, j]:.0f}" if M[i, j] >= 1 else f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > 0.6 * vmax else pf.INK, zorder=3)
        ax.set_xticks(range(len(COARSE)), COARSE, rotation=55, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(models)), models, fontsize=8)
        ax.set_title(pf.TARGETS[t] + "\nprediction change when shuffled (%)", color=pf.INK, fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0, colors=pf.INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_physics_terms(ex: dict, path: Path):
    nat = ex["physics_white [P+A]"]["native"]
    rows = ex["_rows"]
    y = np.array(rows["Y"])[:, 0]
    held = np.array(rows["held_out"])
    amp = np.array(rows["amplifier"])
    T, G, R = (np.array(nat["terms"][k]) for k in ("trace", "minus_log_gain", "correction"))
    gm, rm = np.mean(-G[~held]), np.mean(R[~held])
    steps = [("trace term, average gain", T - gm + rm), ("+ learned gain", T + G + rm), ("+ learned correction (the model)", T + G + R)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    lim = np.exp([min(y.min(), min(s.min() for _, s in steps)) - 0.05, max(y.max(), max(s.max() for _, s in steps)) + 0.05])
    for ax, (title, p) in zip(axes, steps):
        ax.plot(lim, lim, color=pf.INK2, linewidth=1)
        ax.scatter(np.exp(y[~held]), np.exp(p[~held]), s=10, color="#d6d5d0", label="A+B (training rows)")
        for k, ch in enumerate(sorted(set(amp))):
            m = held & (amp == ch)
            ax.scatter(np.exp(y[m]), np.exp(p[m]), s=30, marker=pf.AMP_MARKERS[k], color=pf.AMP_COLORS[k],
                       edgecolor=pf.SURFACE, linewidth=0.8, label=f"session C, {ch}", zorder=3)
        err = 100 * np.sqrt(np.mean((p[held] - y[held]) ** 2))
        ax.set_title(f"{title}\nsession C error {err:.1f} %", color=pf.INK, fontsize=9)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(r"true $\sigma_e$ (e$^-$)", color=pf.INK2)
        pf.style(ax, r"predicted $\sigma_e$ (e$^-$)" if ax is axes[0] else None)
        ax.grid(axis="x", color=pf.GRID, linewidth=0.8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--explain", default=str(ROOT / "processed" / "lvl1_results" / "2026-11-03_explain" / "explain.json"))
    parser.add_argument("--out", default=str(REPORT))
    args = parser.parse_args()
    ex = json.loads(Path(args.explain).read_text())
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    perf = Performance(ex["_chosen_sets"])
    d = pd.read_csv(ROOT / "processed_AB" / "dataset.csv")
    d = d[d.board == 101].copy()
    d["ok"] = d["label.ok"].astype(str).str.lower().eq("true")
    noise = pf.label_noise(d)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": pf.SURFACE})
    (out / "models_performance.tex").write_text(performance_table(perf))
    for m in ORDER:
        if all(f"{m} [{ex['_chosen_sets'][m][t]}]" in ex for t in TARGETS):
            (out / "models" / f"{m}.tex").write_text(model_fragment(ex, perf, m))
        else:
            print(f"{m}: not in {args.explain}")
    fig_all_models(perf, noise, out / "figures" / "fig7_all_models.png")
    fig_reliance(ex, out / "figures" / "fig8_reliance.png")
    fig_physics_terms(ex, out / "figures" / "fig9_physics_terms.png")
    for t in TARGETS:
        print(t, {m: round(perf.row(m, t)["c"], 1) for m in ORDER})
    print("wrote", out)


if __name__ == "__main__":
    main()
