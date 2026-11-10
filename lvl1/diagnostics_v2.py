"""
diagnostics_v2.py -- diagnostic plots of the level-1 models on v2 session D (setting-grouped cross-validation, the
saved predictions of lvl1/scorecard_v2.py), the change (relative) models first, then the absolute models, and what
the integration time (delayInt) does to all of them.  Each kind of model gets the plots that fit it.

CHANGE MODELS (new-amplifier cell: the amplifier predicted was never in training; pairs of one amplifier):
  change_scatter_<t>.png      predicted against true change for every change model; colour = whether delayInt changed
                              between the two traces; text = direction right on clear changes and size error
  direction_vs_size_<t>.png   direction right against the size of the true change (binned): how big a change must be
                              before each model calls it right
  classifier_reliability.png  the two direction CLASSIFIERS: predicted P(B worse) against the observed fraction (a
                              calibrated classifier lies on the diagonal) and the ROC area
  size_calibration_<t>.png    the SIZE models: mean predicted change against true change (binned) and the fitted slope
                              (slope < 1: the model shrinks changes, > 1: exaggerates)
  by_pair_type.png            direction per pair type (within packet, n -> n+1, n -> n+2, random)
  by_amplifier.png            direction per left-out amplifier
ABSOLUTE MODELS (new-amplifier cell):
  abs_pred_vs_true_<t>.png    predicted against measured per model, coloured by amplifier
  abs_residuals.png           residual against delayInt and against time (drift), per model
  abs_amp_bias.png            mean residual per left-out amplifier (the level a new amplifier needs)
THE INTEGRATION TIME:
  delayint_effect.png         the labels against delayInt per amplifier; share of each label's spread that delayInt
                              alone explains; the true changes before and after removing the part delayInt explains
  delayint_accuracy.png       direction of every change model against |d log2 delayInt| (binned)
  delayint_removed.png        experiment: change models trained (a) on all pairs, (b) on the fixed-delayInt pairs
                              only, (c) settings models without the delayInt input; scored on fixed-delayInt and on
                              changed-delayInt test pairs (new-amplifier CV as the scorecard)
Plus diagnostics.md with the numbers behind the plots.

    .venv/bin/python lvl1/diagnostics_v2.py --out processed_v2_trainD/lvl1_results/diagnostics
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "lvl1"), str(ROOT / "src")]
import funnel_shortlist as fs  # noqa: E402
import scorecard as sc  # noqa: E402
import scorecard_v2 as sv  # noqa: E402

warnings.filterwarnings("ignore")
T = ["sigma_e", "lambda", "gain"]
CLEAR = dict(zip(T, sc.CLEAR))
COL = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
FIXED = 0.1                       # |d log2 delayInt| below this: "delayInt fixed" (7 %)
CHANGE = {  # change models shown, and what kind they are
    "settings change: gp": "size", "settings change: trees": "size",
    "trace change: trees": "size", "trace: combined trees [A, B, B-A]": "size",
    "trace: direction classifier [A, B, B-A]": "classifier", "trace change: shared-encoder net": "size",
    "trace change: difference curves PCA + trees": "size", "trace: physics difference + correction (sigma_e)": "size",
    "settings + trace: combined trees": "size", "settings + trace: direction classifier": "classifier",
    "absolute trace: physics, differenced": "size", "ensemble: trace models": "size"}
SHORT = lambda m: (m.replace("trace: ", "").replace(" [A, B, B-A]", "").replace("trace change: ", "trace Δ: ")
                   .replace("settings change: ", "settings Δ: ").replace("absolute trace: ", "abs ")
                   .replace(" (sigma_e)", "").replace("difference curves PCA + trees", "Δcurves PCA"))
import matplotlib.cm as _cm
MCOL = {m: _cm.tab20(i / 20) for i, m in enumerate(CHANGE)}          # one fixed colour per change model, every plot
ACOL = {"settings: gp": "#1f77b4", "trace: physics": "#2ca02c", "trace: trees": "#9467bd", "trace: cnn": "#e377c2"}
ABS = ["settings: gp", "settings: trees", "trace: physics", "trace: trees", "trace: nn", "trace: cnn",
       "trace: pretrained cnn", "trace: gp"]


def load(run_dir: Path):
    data = (ROOT / "processed_v2_trainD").resolve()
    sv.cm.set_data(data)
    sc.DATA = data
    files, rows, d = sv.load(data, (0, 1, 3), None, False)
    Y, _, abs_preds, P, rel = pickle.loads((run_dir / "predictions.pkl").read_bytes())
    TR = sc.trace_table(files, rows, d)
    th = [c for c in d.columns if c.startswith("theta.")]
    i = th.index("theta.delayInt")
    P["dlog2_int"] = np.log2(TR["TH"][P["B"], i] / TR["TH"][P["A"], i])
    return d, Y, abs_preds, P, rel, TR, th


def direction(y, q, c):
    m = (np.abs(y) > c) & np.isfinite(q)
    return (float(np.mean(np.sign(q[m]) == np.sign(y[m]))) if m.sum() else np.nan), int(m.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=str(ROOT / "processed_v2_trainD" / "lvl1_results" / "taskB_scorecard_D_newmodels"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default="new")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, Y, abs_preds, P, rel, TR, th = load(Path(args.run))
    cell = args.cell
    Yp, typ, dl = P["Y"], P["type"], P["dlog2_int"]
    pamp = d.channel.to_numpy()[P["ra"]]
    fixed = np.abs(dl) < FIXED
    models = [m for m in CHANGE if m in rel[cell]]
    md = ["# Diagnostics of the v2 level-1 models (session D, setting-grouped CV, "
          f"cell '{cell}': the predicted amplifier was not in training)", "",
          f"Pairs: {dict(zip(sv.PAIR_TYPES, np.bincount(typ, minlength=5).tolist()))}. 'delayInt fixed' = |d log2 delayInt| "
          f"< {FIXED} (within 7 %): {int(fixed.sum())} pairs. Clear change: sigma_e > 5 %, lambda > 15 %, gain > 1.2 %.", ""]
    scored = lambda m: np.isfinite(rel[cell][m]).any(1)

    # ================================================================ change models: scatter
    for ti, t in enumerate(T):
        ms = [m for m in models if np.isfinite(rel[cell][m][:, ti]).any()]
        nc = 4
        nr = int(np.ceil(len(ms) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.9 * nr), squeeze=False)
        sel = typ != 1                                            # image-label pairs (strip pairs have their own models)
        for ax, m in zip(axes.flat, ms):
            q = rel[cell][m][:, ti]
            ok = sel & np.isfinite(q)
            y = 100 * Yp[ok, ti]
            qq = q[ok] * (100 if CHANGE[m] == "size" else 1)
            for f, c, lab in ((~fixed[ok], "#4c72b0", "delayInt changed"), (fixed[ok], "#dd8452", "delayInt fixed")):
                ax.plot(y[f], qq[f], ".", ms=3, alpha=0.4, color=c, label=lab)
            ax.axvline(0, color="k", lw=0.6)
            ax.axhline(0, color="k", lw=0.6)
            lim = np.nanpercentile(np.abs(y), 99)
            ax.set_xlim(-lim, lim)
            if CHANGE[m] == "size":
                ax.plot([-lim, lim], [-lim, lim], "k:", lw=0.8)
                ax.set_ylim(-lim, lim)
                ax.set_ylabel("predicted change %")
            else:
                ax.set_ylabel("predicted P(B worse) - 0.5")
            da, n = direction(Yp[ok, ti], q[ok], CLEAR[t])
            df, nf = direction(Yp[ok & fixed, ti], q[ok & fixed], CLEAR[t])
            txt = f"direction {100 * da:.0f} % (n {n})\ndelayInt fixed {100 * df:.0f} % (n {nf})"
            if CHANGE[m] == "size":
                txt += f"\nsize error {100 * np.sqrt(np.mean((q[ok] - Yp[ok, ti]) ** 2)):.1f} %"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=7,
                    bbox=dict(fc="white", alpha=0.8, lw=0))
            ax.set_title(SHORT(m), fontsize=9)
            ax.set_xlabel(f"true change of {t} %")
        for ax in axes.flat[len(ms):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"Change models, {t}: predicted against true change (new amplifier, image-label pairs; direction "
                     "counted on clear changes)")
        fig.tight_layout()
        fig.savefig(out / f"change_scatter_{t}.png", dpi=120)
        plt.close(fig)

    # ================================================================ direction against the size of the change
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    md += ["## Direction against the size of the true change (new amplifier, image-label pairs)", ""]
    for ti, (ax, t) in enumerate(zip(axes, T)):
        sel = typ != 1
        y = np.abs(Yp[:, ti])
        edges = np.quantile(y[sel], np.linspace(0, 1, 9))
        mids = 0.5 * (edges[1:] + edges[:-1])
        md += [f"{t}: bins of |true change| (%): " + ", ".join(f"{100 * e:.1f}" for e in edges), ""]
        for m in models:
            q = rel[cell][m][:, ti]
            if not np.isfinite(q).any():
                continue
            acc = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                b = sel & (y >= lo) & (y <= hi) & np.isfinite(q) & (y > 0)
                acc.append(np.mean(np.sign(q[b]) == np.sign(Yp[b, ti])) if b.sum() > 5 else np.nan)
            style = "--" if m.startswith("settings change") else "-"
            ax.plot(100 * mids, 100 * np.array(acc), style, marker="o", ms=3, label=SHORT(m), lw=1.2, color=MCOL[m])
        ax.axvline(100 * CLEAR[t], color="grey", lw=0.8, ls=":")
        ax.axhline(50, color="k", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(f"|true change of {t}| %  (dotted: the 'clear' threshold)")
        ax.set_ylabel("direction right %")
        ax.set_title(t)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("How big a change must be before each model calls its direction (dashed: settings only)")
    fig.tight_layout()
    fig.savefig(out / "direction_vs_size.png", dpi=120)
    plt.close(fig)

    # ================================================================ classifier reliability
    from sklearn.metrics import roc_auc_score
    cls = [m for m in models if CHANGE[m] == "classifier"]
    fig, axes = plt.subplots(len(cls), 3, figsize=(15, 4.3 * len(cls)), squeeze=False)
    md += ["## Direction classifiers: reliability and ROC area (new amplifier, image-label pairs, |change| above the "
           "label noise)", "", "| classifier | target | ROC area | pairs |", "|---|---|---|---|"]
    for r, m in enumerate(cls):
        for ti, t in enumerate(T):
            ax = axes[r, ti]
            q = rel[cell][m][:, ti]
            ok = (typ != 1) & np.isfinite(q) & (np.abs(Yp[:, ti]) > sc.cm.NOISE[ti])
            p = np.clip(q[ok] + 0.5, 0, 1)
            up = (Yp[ok, ti] > 0).astype(float)
            bins = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
            fr = [up[idx == k].mean() if (idx == k).sum() > 4 else np.nan for k in range(10)]
            cnt = [(idx == k).sum() for k in range(10)]
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.plot(0.5 * (bins[1:] + bins[:-1]), fr, "o-")
            ax2 = ax.twinx()
            ax2.bar(0.5 * (bins[1:] + bins[:-1]), cnt, width=0.09, alpha=0.2, color="grey")
            ax2.set_ylabel("pairs", fontsize=7)
            auc = roc_auc_score(up, p) if 0 < up.mean() < 1 else np.nan
            ax.set_title(f"{SHORT(m)}: {t}  (ROC area {auc:.2f})", fontsize=9)
            ax.set_xlabel("predicted P(B worse than A)")
            ax.set_ylabel("observed fraction B worse")
            md.append(f"| {SHORT(m)} | {t} | {auc:.2f} | {int(ok.sum())} |")
    md.append("")
    fig.suptitle("Direction classifiers: on the diagonal = the stated probability is right (grey: pairs per bin)")
    fig.tight_layout()
    fig.savefig(out / "classifier_reliability.png", dpi=120)
    plt.close(fig)

    # ================================================================ size calibration
    md += ["## Size models: slope of mean predicted against true change (1 = right size; < 1 shrinks)", "",
           "| model | " + " | ".join(T) + " |", "|---|---|---|---|"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    slopes = {}
    for ti, (ax, t) in enumerate(zip(axes, T)):
        for m in models:
            if CHANGE[m] != "size":
                continue
            q = rel[cell][m][:, ti]
            ok = (typ != 1) & np.isfinite(q)
            if not ok.any():
                continue
            y = Yp[ok, ti]
            edges = np.quantile(y, np.linspace(0, 1, 11))
            mids = [y[(y >= lo) & (y <= hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
            mq = [q[ok][(y >= lo) & (y <= hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
            slopes.setdefault(m, {})[t] = float(np.polyfit(y, q[ok], 1)[0])
            ax.plot(100 * np.array(mids), 100 * np.array(mq), "o-", ms=3, lw=1, label=SHORT(m), color=MCOL[m],
                    ls="--" if m.startswith("settings change") else "-")
        lim = ax.get_xlim()
        ax.plot(lim, lim, "k:", lw=0.8)
        ax.set_xlabel(f"true change of {t} % (bin means)")
        ax.set_ylabel("mean predicted change %")
        ax.set_title(t)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6.5)
    for m, s in slopes.items():
        md.append(f"| {SHORT(m)} | " + " | ".join(f"{s.get(t, np.nan):.2f}" for t in T) + " |")
    md.append("")
    fig.suptitle("Size models: do they get the size of a change right? (on the dotted line = yes)")
    fig.tight_layout()
    fig.savefig(out / "size_calibration.png", dpi=120)
    plt.close(fig)

    # ================================================================ by pair type and by amplifier
    for key, groups, fname, title in (
            ("type", [(k, typ == k) for k in range(5)], "by_pair_type.png", "pair type"),
            ("amp", [(a, (pamp == a) & (typ != 1)) for a in np.unique(pamp)], "by_amplifier.png", "left-out amplifier")):
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
        md += [f"## Direction by {title} (clear changes, %)", "",
               "| model | target | " + " | ".join(str(sv.PAIR_TYPES[g] if key == "type" else f"ch{g}") for g, _ in groups) + " |",
               "|---|---|" + "---|" * len(groups)]
        for ti, (ax, t) in enumerate(zip(axes, T)):
            w = 0.8 / len(models)
            for j, m in enumerate(models):
                q = rel[cell][m][:, ti]
                vals = [direction(Yp[g, ti], q[g], CLEAR[t])[0] for _, g in groups]
                if np.all(np.isnan(vals)):
                    continue
                ax.bar(np.arange(len(groups)) + (j - len(models) / 2) * w, 100 * np.array(vals), w, label=SHORT(m), color=MCOL[m])
                md.append(f"| {SHORT(m)} | {t} | " + " | ".join(f"{100 * v:.0f}" if np.isfinite(v) else "–" for v in vals) + " |")
            ax.set_xticks(range(len(groups)), [str(sv.PAIR_TYPES[g] if key == "type" else f"ch{g}") for g, _ in groups],
                          fontsize=8, rotation=15)
            ax.axhline(50, color="k", lw=0.6)
            ax.set_ylim(30, 100)
            ax.set_ylabel("direction right on clear changes %")
            ax.set_title(t)
            ax.grid(alpha=0.3, axis="y")
        axes[0].legend(fontsize=6, ncol=2)
        md.append("")
        fig.suptitle(f"Change models: direction by {title}")
        fig.tight_layout()
        fig.savefig(out / fname, dpi=120)
        plt.close(fig)

    # ================================================================ absolute models
    amp = d.channel.to_numpy()
    di = d["theta.delayInt"].to_numpy(float)
    ap_ = abs_preds[cell]
    am = [m for m in ABS if m in ap_]
    md += ["## Absolute models (new amplifier): typical error %, bias per left-out amplifier %", "",
           "| model | target | error | bias ch0 | bias ch1 | bias ch3 | residual slope per doubling of delayInt % |",
           "|---|---|---|---|---|---|---|"]
    for ti, t in enumerate(T):
        fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
        for ax, m in zip(axes.flat, am):
            p = ap_[m][:, ti]
            ok = np.isfinite(p)
            for a in np.unique(amp):
                s = ok & (amp == a)
                ax.plot(np.exp(Y[s, ti]), np.exp(p[s]), "o", ms=3, alpha=0.6, color=COL[a], label=f"ch{a}")
            lo, hi = np.exp(np.nanmin(Y[:, ti])) * 0.8, np.exp(np.nanmax(Y[:, ti])) * 1.2
            ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            e = p[ok] - Y[ok, ti]
            ax.set_title(f"{m}: error {100 * np.sqrt(np.mean(e ** 2)):.1f} %", fontsize=9)
            ax.set_xlabel(f"measured {t}")
            ax.set_ylabel(f"predicted {t}")
            bias = [100 * np.mean(p[ok & (amp == a)] - Y[ok & (amp == a), ti]) for a in (0, 1, 3)]
            sl = 100 * np.polyfit(np.log2(di[ok]), e, 1)[0]
            md.append(f"| {m} | {t} | {100 * np.sqrt(np.mean(e ** 2)):.1f} | " + " | ".join(f"{b:+.1f}" for b in bias)
                      + f" | {sl:+.1f} |")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"Absolute models, {t}: predicted against measured (new amplifier; colour = amplifier: an offset "
                     "per colour is the level a new amplifier needs)")
        fig.tight_layout()
        fig.savefig(out / f"abs_pred_vs_true_{t}.png", dpi=120)
        plt.close(fig)
    md.append("")
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for ti, t in enumerate(T):
        for m in ["settings: gp", "trace: physics", "trace: trees", "trace: cnn"]:
            if m not in ap_:
                continue
            e = 100 * (ap_[m][:, ti] - Y[:, ti])
            o = np.argsort(di)
            sm = pd.Series(e[o]).rolling(15, center=True, min_periods=5).mean()
            axes[0, ti].plot(di[o], sm, "-", label=m, color=ACOL[m], lw=1.6)
            axes[0, ti].plot(di, e, ".", ms=2, alpha=0.2, color=ACOL[m])
            h = d.hours.to_numpy()
            o2 = np.argsort(h)
            axes[1, ti].plot(h[o2], pd.Series(e[o2]).rolling(15, center=True, min_periods=5).mean(), "-", label=m,
                             color=ACOL[m], lw=1.6)
        axes[0, ti].set_xscale("log")
        axes[0, ti].set_xlabel("delayInt (cycles)")
        axes[1, ti].set_xlabel("hours into session D")
        for r in (0, 1):
            axes[r, ti].axhline(0, color="k", lw=0.6)
            axes[r, ti].set_ylabel(f"{t} residual % (running mean)")
            axes[r, ti].grid(alpha=0.3)
        axes[0, ti].set_title(t)
    for r in (0, 1):
        axes[r, 0].legend(fontsize=7)
    fig.suptitle("Absolute models: where they miss (top: against delayInt, bottom: against time = drift)")
    fig.tight_layout()
    fig.savefig(out / "abs_residuals.png", dpi=120)
    plt.close(fig)

    # ================================================================ the integration time
    # (1) the labels against delayInt, and what is left of the changes once delayInt's part is removed
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    md += ["## The integration time", "",
           "| target | spread of the label % | share explained by delayInt alone (per amplifier, CV) | RMS true change % "
           "(between-packet pairs) | ... after removing delayInt's part % | ... delayInt-fixed pairs % |",
           "|---|---|---|---|---|---|"]
    between = np.isin(typ, (2, 3, 4))
    for ti, t in enumerate(T):
        ax = axes[0, ti]
        r2 = []
        for a in np.unique(amp):
            s = amp == a
            x = np.log2(di[s])
            ax.plot(di[s], 100 * (Y[s, ti] - Y[s, ti].mean()), "o", ms=3, alpha=0.6, color=COL[a], label=f"ch{a}")
            c = np.polyfit(x, Y[s, ti], 2)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(2 ** xs, 100 * (np.polyval(c, xs) - Y[s, ti].mean()), "-", color=COL[a])
            from sklearn.model_selection import GroupKFold, cross_val_predict
            pr = cross_val_predict(H(max_iter=100, max_leaf_nodes=4, min_samples_leaf=5), x[:, None], Y[s, ti],
                                   groups=d.packet_id.to_numpy()[s], cv=GroupKFold(5))
            r2.append(1 - np.mean((Y[s, ti] - pr) ** 2) / np.var(Y[s, ti]))
        ax.set_xscale("log")
        ax.set_xlabel("delayInt (cycles, log)")
        ax.set_ylabel(f"{t} - amplifier mean, %")
        ax.set_title(f"{t}: delayInt alone explains {100 * np.mean(r2):.0f} % (CV)")
        ax.grid(alpha=0.3)
        # changes: before / after removing the part predicted from d log2 delayInt (+ the starting delayInt)
        Xd = np.column_stack([dl, np.log2(TR["TH"][P["A"], th.index("theta.delayInt")])])
        ok = between & np.isfinite(Yp[:, ti])
        from sklearn.model_selection import GroupKFold
        res = np.full(len(Yp), np.nan)
        g = d.packet_id.to_numpy()[P["ra"]]
        for tr_, te_ in GroupKFold(5).split(Xd[ok], Yp[ok, ti], g[ok]):
            i_tr, i_te = np.flatnonzero(ok)[tr_], np.flatnonzero(ok)[te_]
            f = H(max_iter=200, max_leaf_nodes=8, min_samples_leaf=10).fit(np.vstack([Xd[i_tr], -Xd[i_tr] * [1, 0] + Xd[i_tr] * [0, 1]]),
                                                                           np.concatenate([Yp[i_tr, ti], -Yp[i_tr, ti]]))
            res[i_te] = Yp[i_te, ti] - f.predict(Xd[i_te])
        ax = axes[1, ti]
        bins = np.linspace(-np.nanpercentile(np.abs(100 * Yp[ok, ti]), 99), np.nanpercentile(np.abs(100 * Yp[ok, ti]), 99), 50)
        ax.hist(100 * Yp[ok, ti], bins=bins, alpha=0.5, density=True, label=f"true changes (RMS {100 * np.std(Yp[ok, ti]):.1f} %)")
        ax.hist(100 * res[ok], bins=bins, alpha=0.5, density=True, label=f"delayInt's part removed (RMS {100 * np.nanstd(res[ok]):.1f} %)")
        ax.hist(100 * Yp[ok & fixed, ti], bins=bins, alpha=0.5, density=True, histtype="step", lw=1.5,
                label=f"delayInt-fixed pairs (RMS {100 * np.std(Yp[ok & fixed, ti]):.1f} %)")
        ax.axvline(-100 * CLEAR[t], color="grey", ls=":")
        ax.axvline(100 * CLEAR[t], color="grey", ls=":")
        ax.set_xlabel(f"change of {t} between two packets, %")
        ax.legend(fontsize=7)
        md.append(f"| {t} | {100 * np.std(Y[:, ti]):.1f} | {100 * np.mean(r2):.0f} % | {100 * np.std(Yp[ok, ti]):.1f} | "
                  f"{100 * np.nanstd(res[ok]):.1f} | {100 * np.std(Yp[ok & fixed, ti]):.1f} |")
    md.append("")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("The integration time: what delayInt does to each label (top) and how much of the changes is left "
                 "without it (bottom; dotted: the 'clear change' threshold)")
    fig.tight_layout()
    fig.savefig(out / "delayint_effect.png", dpi=120)
    plt.close(fig)

    # (2) direction against |d log2 delayInt|
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    edges = np.array([0, FIXED, 0.5, 1.0, 2.0, 4.0])
    md += ["## Direction against how much delayInt changed (clear changes, %; bins of |d log2 delayInt|: "
           + ", ".join(f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])) + ")", "",
           "| model | target | " + " | ".join(f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])) + " |",
           "|---|---|" + "---|" * (len(edges) - 1)]
    for ti, (ax, t) in enumerate(zip(axes, T)):
        for m in models:
            q = rel[cell][m][:, ti]
            if not np.isfinite(q).any():
                continue
            acc, ns = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                b = (typ != 1) & (np.abs(dl) >= lo) & (np.abs(dl) < hi)
                a_, n_ = direction(Yp[b, ti], q[b], CLEAR[t])
                acc.append(a_ if n_ >= 8 else np.nan)
                ns.append(n_)
            ax.plot(range(len(acc)), 100 * np.array(acc), marker="o", ms=4, lw=1.2, label=SHORT(m), color=MCOL[m],
                    ls="--" if m.startswith("settings change") else "-")
            md.append(f"| {SHORT(m)} | {t} | " + " | ".join(f"{100 * a:.0f} ({n})" if np.isfinite(a) else f"– ({n})"
                                                       for a, n in zip(acc, ns)) + " |")
        ax.set_xticks(range(len(edges) - 1), [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])])
        ax.set_xlabel("|d log2 delayInt| (0-0.1: fixed; 1 = a doubling)")
        ax.set_ylabel("direction right on clear changes %")
        ax.axhline(50, color="k", lw=0.6)
        ax.set_title(t)
        ax.grid(alpha=0.3)
    md.append("")
    axes[0].legend(fontsize=6.5)
    fig.suptitle("How much the change models depend on the integration time changing (dashed: settings only; "
                 "points with < 8 clear changes left out)")
    fig.tight_layout()
    fig.savefig(out / "delayint_accuracy.png", dpi=120)
    plt.close(fig)

    # (3) experiment: train with / without delayInt, and on fixed-delayInt pairs only
    exp = experiment(d, P, TR, th, fixed)
    md += ["## Experiment: what removing the integration time does (new-amplifier CV as the scorecard; image-label "
           "pairs; direction on clear changes % / size error %)", "",
           "| model | trained on | inputs | target | fixed-delayInt test pairs | changed-delayInt test pairs | training pairs |",
           "|---|---|---|---|---|---|---|"]
    for r in exp:
        md.append(f"| {r['model']} | {r['train']} | {r['inputs']} | {r['target']} | "
                  f"{r['fixed_dir']:.0f} ({r['fixed_n']}) / {r['fixed_size']:.1f} | "
                  f"{r['changed_dir']:.0f} ({r['changed_n']}) / {r['changed_size']:.1f} | {r['n_train']} |")
    md.append("")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ti, (ax, t) in enumerate(zip(axes, T)):
        rr = [r for r in exp if r["target"] == t]
        labels = [f"{r['model']}\n{r['train']}, {r['inputs']}" for r in rr]
        x = np.arange(len(rr))
        ax.bar(x - 0.2, [r["fixed_dir"] for r in rr], 0.4, label="delayInt-fixed test pairs", color="#dd8452")
        ax.bar(x + 0.2, [r["changed_dir"] for r in rr], 0.4, label="delayInt-changed test pairs", color="#4c72b0")
        ax.set_xticks(x, labels, fontsize=6.5, rotation=30, ha="right")
        ax.axhline(50, color="k", lw=0.6)
        ax.set_ylim(30, 100)
        ax.set_ylabel("direction right on clear changes %")
        ax.set_title(t)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7)
    fig.suptitle("Removing the integration time: models trained on all pairs, on delayInt-fixed pairs only, and "
                 "settings models without the delayInt input")
    fig.tight_layout()
    fig.savefig(out / "delayint_removed.png", dpi=120)
    plt.close(fig)

    (out / "diagnostics.md").write_text("\n".join(md) + "\n")
    (out / "experiment.json").write_text(json.dumps(exp, indent=1, default=float))
    print("\n".join(md))


def experiment(d, P, TR, th, fixed):
    """Train change models three ways and score them on fixed / changed delayInt test pairs (new-amplifier CV: test =
    one time block of one amplifier, train = other blocks of the other amplifiers), image-label pairs (types 0, 2, 3,
    4), both orders in training."""
    A, B, Yp, typ = P["A"], P["B"], P["Y"], P["type"]
    blk, amp = d.block.to_numpy(), d.channel.to_numpy()
    ba, bb, pamp = blk[P["ra"]], blk[P["rb"]], amp[P["ra"]]
    img = typ != 1
    i_int = th.index("theta.delayInt")
    TH = np.nan_to_num(TR["TH"])
    dT = TH[B] - TH[A]
    dT_noint = np.delete(dT, i_int, axis=1)
    F = TR["F"]
    Xc = np.hstack([F[A], F[B], F[B] - F[A]])
    Xcs = np.hstack([F[B], F[A], F[A] - F[B]])
    variants = [("settings Δ trees", "all pairs", "all settings", dT, -dT, None),
                ("settings Δ trees", "all pairs", "without delayInt", dT_noint, -dT_noint, None),
                ("settings Δ trees", "delayInt-fixed pairs only", "all settings", dT, -dT, fixed),
                ("trace combined trees", "all pairs", "traces", Xc, Xcs, None),
                ("trace combined trees", "delayInt-fixed pairs only", "traces", Xc, Xcs, fixed)]
    out = []
    for name, train, inputs, X, Xs, only in variants:
        pred = np.full(Yp.shape, np.nan)
        ntr = []
        for k in range(sv.N_BLOCKS):
            for a in np.unique(amp):
                te = np.flatnonzero(img & (ba == k) & (bb == k) & (pamp == a))
                tr = np.flatnonzero(img & (ba != k) & (bb != k) & (ba >= 0) & (bb >= 0) & (pamp != a)
                                    & (only if only is not None else True))
                if len(te) == 0 or len(tr) < 20:
                    continue
                f = fs.gbt(np.vstack([X[tr], Xs[tr]]), np.vstack([Yp[tr], -Yp[tr]]))
                pred[te] = 0.5 * (f(X[te]) - f(Xs[te]))
                ntr.append(len(tr))
        for ti, t in enumerate(T):
            r = {"model": name, "train": train, "inputs": inputs, "target": t, "n_train": int(np.mean(ntr)) if ntr else 0}
            for lab, m in (("fixed", img & fixed), ("changed", img & ~fixed)):
                ok = m & np.isfinite(pred[:, ti])
                a_, n_ = direction(Yp[ok, ti], pred[ok, ti], CLEAR[t])
                r[f"{lab}_dir"] = 100 * a_ if n_ else np.nan
                r[f"{lab}_n"] = n_
                r[f"{lab}_size"] = 100 * float(np.sqrt(np.mean((pred[ok, ti] - Yp[ok, ti]) ** 2))) if ok.any() else np.nan
            out.append(r)
        print(f"experiment: {name}, {train}, {inputs} done", flush=True)
    return out


if __name__ == "__main__":
    main()
