"""
plot_first_data.py -- the summary figures of the first real packets (2026-11-02/03, L1_01..L1_43) for
docs/overleaf_first_data_2026-11-03/.  Every number is read from the result files, nothing is typed in:

    processed_AB/dataset.csv                                   labels and trace features (sessions A + B)
    processed_AB/lvl1_results/2026-11-03_ladder_raw/           cross-validation of every model at every feature set
    <scratchpad or results>/amp_baseline/                      settings + amplifier references (cross-validation)
    processed/lvl1_results/2026-11-03_holdoutC/<set>/          session C, scored once (the frozen choice)

    .venv/bin/python lvl1/plot_first_data.py --amp-baseline DIR [--out docs/overleaf_first_data_2026-11-03/figures]

Figures (light theme; amplifiers in the reference palette's first four slots, each with its own marker as well):
    fig1_traces_help.png      error of the baselines and the frozen trace models, cross-validation and session C
    fig2_session_c.png        frozen models on session C: predicted vs true
    fig3_sigma_error_by_packet.png   where the sigma_e error comes from (packets, by session)
    fig4_trace_vs_image.png   the trace's noise vs the image's sigma_e, by kind of change
    fig5_lambda_night.png     lambda of the nominal packets over the night
    fig6_ladder.png           error vs number of trace features for four models
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
#: reference palette (dataviz skill, references/palette.md), light mode, slots 1-4, in fixed order
AMP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
AMP_MARKERS = ["o", "s", "^", "D"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TARGETS = {"sigma_e": "σe (read noise)", "lambda": "λ (dark charge)", "gain": "gain"}
#: the label noise: repeat scatter of the labels over the 4 nominal packets (sd of log), median over the amplifiers
FROZEN = {"sigma_e": ("gp_dim", "P+A1"), "lambda": ("pretrained", "all"), "gain": ("gp_dim", "P")}
#: the physics-structured model (physics_white: white-noise trace term; its lambda and gain are physics') at the feature
#: set the A+B cross-validation chose for each target (before 2026-11-04 lambda and gain were shown at P+A and P+A1,
#: sets picked after looking at session C)
PHYS = {"sigma_e": ("physics_white", "P+A"), "lambda": ("physics", "all"), "gain": ("physics", "P")}


def style(ax, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def label_noise(d: pd.DataFrame) -> dict:
    nom = d[d.packet_id.isin(["L1_01", "L1_07", "L1_14", "L1_32"]) & d.ok]
    cols = {"sigma_e": "label.sigma_e", "lambda": "label.lambda_e_per_bin", "gain": "label.gain_adu_per_e"}
    return {t: 100 * np.median([np.log(g[c]).std(ddof=1) for _, g in nom.groupby("channel")]) for t, c in cols.items()}


def cv_error(report: dict, key: str, target: str) -> float:
    return 100 * report["results"][key]["scores"][target]["error"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ab", default=str(ROOT / "processed_AB"))
    parser.add_argument("--full", default=str(ROOT / "processed"))
    parser.add_argument("--amp-baseline", required=True, help="results of the settings_amp references (A+B CV)")
    parser.add_argument("--out", default=str(ROOT / "docs" / "overleaf_first_data_2026-11-03" / "figures"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ab, full = Path(args.ab), Path(args.full)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": SURFACE})

    d = pd.read_csv(ab / "dataset.csv")
    d = d[d.board == 101].copy()
    d["ok"] = d["label.ok"].astype(str).str.lower().eq("true")
    noise = label_noise(d)
    ladder = json.loads((ab / "lvl1_results" / "2026-11-03_ladder_raw" / "report.json").read_text())
    ampref = json.loads((Path(args.amp_baseline) / "report.json").read_text())
    hold = {p.name: json.loads((p / "report.json").read_text())
            for p in (full / "lvl1_results" / "2026-11-03_holdoutC").iterdir() if (p / "report.json").exists()}
    plan = {p["packet_id"]: p for p in json.loads((ROOT / "lvl1" / "packet_plan" / "plan.json").read_text())["packets"]}

    def held(model, fset, target):
        r = hold["baselines" if fset is None else fset]["holdout"]["results"][model]
        return 100 * r["scores"][target]["error"]

    # ---- fig 1: do the traces help? -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    bars = [("always the average\n(no trace)", "#b9b8b2"), ("settings + which amplifier\n(no trace, told the amplifier)", "#86b6ef"),
            ("frozen trace model\n(chosen on A+B)", "#2a78d6"), ("physics-structured\ntrace model", "#eb6834")]
    for ax, (t, title) in zip(axes, TARGETS.items()):
        model, fset = FROZEN[t]
        ref_best = min(("settings_amp", "settings_amp_trees"), key=lambda k: cv_error(ampref, k, t))
        pm, pset = PHYS[t]
        cv = [cv_error(ampref, "mean", t), cv_error(ampref, ref_best, t),
              cv_error(ladder, f"{model} [{fset}]", t), cv_error(ladder, f"{pm} [{pset}]", t)]
        c_ref_best = min(("settings_amp", "settings_amp_trees"), key=lambda k: held(k, None, t))
        hc = [held("mean", None, t), held(c_ref_best, None, t), held(model, fset, t), held(pm, pset, t)]
        x = np.arange(2)
        w = 0.2
        for i, (lab, col) in enumerate(bars):
            vals = [cv[i], hc[i]]
            rects = ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=col, label=lab if t == "sigma_e" else None)
            for r, v in zip(rects, vals):
                ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.axhline(noise[t], color=INK, linestyle="--", linewidth=1)
        ax.text(1.42, noise[t], "label\nnoise", va="center", ha="left", fontsize=7.5, color=INK)
        ax.set_xticks(x, ["A+B\ncross-validation", "session C\nheld-out test"])
        ax.set_title(title, color=INK)
        style(ax, "typical error (%)" if t == "sigma_e" else None)
        ax.set_xlim(-0.5, 1.75)
    fig.legend(loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig1_traces_help.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 2: frozen models on session C, predicted vs true ------------------------------------------------
    fig, axes2 = plt.subplots(2, 3, figsize=(10, 6.6))
    for ax, (t, title), (model, fset), row in [(axes2[r, j], tt, choice[tt[0]], r) for r, choice in enumerate((FROZEN, PHYS))
                                               for j, tt in enumerate(TARGETS.items())]:
        p = pd.read_csv(full / "lvl1_results" / "2026-11-03_holdoutC" / fset / "predictions.csv")
        col = next(c for c in p.columns if c.startswith(model) and c.endswith(f": log {t}"))
        ho = p["held out"].astype(str).str.lower().eq("true")
        lo, hi = np.exp(np.nanmin(p[f"true log {t}"])), np.exp(np.nanmax(p[f"true log {t}"]))
        ax.scatter(np.exp(p.loc[~ho, f"true log {t}"]), np.exp(p.loc[~ho, col]), s=10, color="#d6d5d0",
                   label="A+B (cross-validation)" if t == "sigma_e" and row == 0 else None, zorder=1)
        for ch in range(4):
            m = ho & (p.channel == ch)
            ax.scatter(np.exp(p.loc[m, f"true log {t}"]), np.exp(p.loc[m, col]), s=34, marker=AMP_MARKERS[ch],
                       color=AMP_COLORS[ch], edgecolor=SURFACE, linewidth=0.8, zorder=3,
                       label=f"session C, amplifier ch{ch}" if t == "sigma_e" and row == 0 else None)
        g = np.linspace(lo * 0.9, hi * 1.1, 2)
        ax.plot(g, g, color=INK2, linewidth=1, zorder=2)
        ax.fill_between(g, g * np.exp(-noise[t] / 100), g * np.exp(noise[t] / 100), color="#e4e3df", alpha=0.6, zorder=0,
                        label="± label noise" if t == "sigma_e" and row == 0 else None)
        if t == "lambda":                                     # lambda spans a decade; the others are narrow
            from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter
            ax.set_xscale("log"); ax.set_yscale("log")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 1, 2, 3]))
                axis.set_major_formatter(ScalarFormatter())
                axis.set_minor_locator(NullLocator()); axis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{title}\n{model} [{fset}]" + (", frozen choice" if row == 0 else ", physics-structured"), color=INK, fontsize=9)
        ax.set_xlabel("true (image, WADERs)" if row == 1 else "", color=INK2)
        style(ax, "predicted from the traces" if t == "sigma_e" else None)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.legend(loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "fig2_session_c.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 3: where the sigma_e error comes from ------------------------------------------------------------
    model, fset = FROZEN["sigma_e"]
    p = pd.read_csv(ab / "lvl1_results" / "2026-11-03_ladder_raw" / "predictions.csv")
    r = (p[f"{model} [{fset}]: log sigma_e"] - p["true log sigma_e"]) ** 2
    per = r.groupby(p.packet_id).mean().pipe(lambda s: 100 * np.sqrt(s))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    sess_col = {"A": "#eb6834", "B": "#2a78d6"}
    for i, (pid, v) in enumerate(per.items()):
        s = plan[pid]["session"]
        ax.bar(i, v, 0.8, color=sess_col[s])
        ch = plan[pid]["changes"]
        name = "nominal" if not ch else ", ".join(f"{k}={v}" for k, v in list(ch.items())[:1])
        ax.text(i, -1.0, f"{pid[3:]} {name}", rotation=90, ha="center", va="top", fontsize=6.5, color=INK2)
    ax.axhline(noise["sigma_e"], color=INK, linestyle="--", linewidth=1)
    ax.text(len(per) - 0.5, noise["sigma_e"] + 0.5, "label noise", ha="right", fontsize=7.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=sess_col["A"], label="session A: timing changed"),
                       Patch(color=sess_col["B"], label="session B: one voltage changed")],
              frameon=False, fontsize=8, loc="upper right")
    ax.set_xticks([])
    ax.set_title(f"σe error of {model} [{fset}] per packet (cross-validation, 4 amplifiers each)", color=INK)
    style(ax, "typical error (%)")
    fig.tight_layout()
    fig.savefig(out / "fig3_sigma_error_by_packet.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 4: the trace's noise vs the image's sigma_e ------------------------------------------------------
    ok = d[d.ok].copy()
    ok["trace_e"] = ok["trace.fit: white noise per skip (ADU)"] / np.sqrt(ok["image.NDCM"]) / ok["label.gain_adu_per_e"]
    kind = ok.packet_id.map(lambda q: "nominal" if not plan[q]["changes"] else ("timing" if plan[q]["session"] == "A" else "voltage"))
    kcol = {"nominal": INK, "voltage": "#2a78d6", "timing": "#eb6834"}
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), sharey=False)
    for ch, ax in enumerate(axes):
        for k, col in kcol.items():
            m = (ok.channel == ch) & (kind == k)
            ax.scatter(ok.loc[m, "trace_e"], ok.loc[m, "label.sigma_e"], s=30, color=col, marker=AMP_MARKERS[ch],
                       edgecolor=SURFACE, linewidth=0.8, label=k if ch == 0 else None)
        nom = ok[(ok.channel == ch) & (kind == "nominal")]
        k0 = np.median(nom["label.sigma_e"] / nom["trace_e"])
        g = np.array([ok.loc[ok.channel == ch, "trace_e"].min(), ok.loc[ok.channel == ch, "trace_e"].max()])
        ax.plot(g, k0 * g, color=INK2, linewidth=1, linestyle="--")
        ax.set_title(f"amplifier ch{ch}", color=INK)
        ax.set_xlabel("trace: white noise / √N / gain (e⁻)", color=INK2, fontsize=8)
        style(ax, "image σe (e⁻)" if ch == 0 else None)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08), title=None)
    fig.suptitle("Does the trace's noise follow the image's noise?  (dashed: proportional, through the nominal packets)",
                 color=INK, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out / "fig4_trace_vs_image.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 5: lambda over the night ---------------------------------------------------------------------------
    times = {}
    for pid in d.packet_id.unique():
        m = json.loads((ROOT / "manifests" / f"{pid}.json").read_text())
        img = [x for x in m["images"] if x["board"] == 101][0]
        meta = str(img.get("meta") or img.get("stem")).replace("/home/damicm/run/PA_PQ_testing", str(ROOT / "data"))
        meta = meta if meta.endswith(".meta") else meta + ".meta"
        times[pid] = datetime.fromisoformat(json.loads(Path(meta).read_text())["DATEini"])
    d["time"] = d.packet_id.map(times)
    fig, ax = plt.subplots(figsize=(8, 3.3))
    for ch in range(4):
        g = d[d.ok & (d.channel == ch)].sort_values("time")
        ax.scatter(g["time"], g["label.lambda_e_per_bin"], s=12, color=AMP_COLORS[ch], alpha=0.3, marker=AMP_MARKERS[ch])
        n = g[g.packet_id.isin(["L1_01", "L1_07", "L1_14", "L1_32"])]
        ax.plot(n["time"], n["label.lambda_e_per_bin"], color=AMP_COLORS[ch], marker=AMP_MARKERS[ch], linewidth=2,
                markersize=7, label=f"ch{ch}, nominal packets")
    import matplotlib.dates as mdates
    ax.set_ylim(0, None)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("time of the image (2026-11-02 to 11-03)", color=INK2)
    ax.set_title("λ over the night: nominal packets (lines) and all packets (faint)", color=INK, pad=28)
    style(ax, "λ (e⁻ / pixel)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(out / "fig5_lambda_night.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 6: how many features? (tiers order) ----------------------------------------------------------------
    sets = [("none", 0), ("P", 8), ("P+A1", 44), ("P+A", 81), ("P+A+B", 96), ("all", 141)]
    models = ["gp_dim", "trees", "nn", "physics_white"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (t, title) in zip(axes, TARGETS.items()):
        for i, mname in enumerate(models):
            xs = np.arange(len(sets))
            ys = [cv_error(ladder, f"{mname} [{s}]", t) for s, _ in sets]
            ax.plot(xs, ys, color=AMP_COLORS[i], marker=AMP_MARKERS[i], linewidth=2, markersize=6,
                    label=mname if t == "sigma_e" else None)
        ax.axhline(cv_error(ampref, "mean", t), color="#b9b8b2", linestyle=":", linewidth=1.2)
        ax.axhline(noise[t], color=INK, linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(sets)), ["0", "8\nP", "44\n+A1", "81\n+A2", "96\n+B", "141\nall"])
        ax.set_title(title, color=INK)
        ax.set_xlabel("trace features given", color=INK2)
        style(ax, "error (%), A+B cross-validation" if t == "sigma_e" else None)
    handles = axes[0].get_legend_handles_labels()[0]
    fig.legend(handles=handles + [plt.Line2D([], [], color="#b9b8b2", linestyle=":", label="always the average"),
                                  plt.Line2D([], [], color=INK, linestyle="--", label="label noise")],
               loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig6_ladder.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("label noise (%):", {k: round(v, 1) for k, v in noise.items()}, "; wrote", out)


if __name__ == "__main__":
    main()
