"""
model_diagnostics_v2.py -- one diagnostic figure per model, tailored to how that model works (v2 session D, the
amplifier predicted is never in training: 3 amplifiers x 5 time blocks, as lvl1/scorecard_v2.py's "new" cell).

ABSOLUTE MODELS (predict the value of sigma_e, lambda, gain)
  abs_settings_gp.png   settings-only Gaussian process: (a) predicted vs measured with its error bars; (b) are its error
                        bars honest (spread of misses in units of the stated error); (c) which settings it relies on
                        (ARD relevance = 1 / length scale)
  abs_physics.png       trace physics model, sigma_e = trace noise term / gain x correction: (a) the trace noise term
                        against the image's noise in ADU; (b) the learned gain; (c) the learned correction; (d) the
                        final sigma_e; (e) lambda (learned, no formula); (f) honesty of its error bars
  abs_trees.png         trace boosted trees: (a) predicted vs measured; (b) the trace features it relies on; (c) how its
                        prediction moves with the most important feature
CHANGE MODELS (predict the change between two packets of one amplifier)
  chg_settings_gp.png   settings-change Gaussian process: (a) predicted vs true change; (b) which setting changes it relies
                        on; (c) its learned response to a change of delayInt alone
  chg_physics.png       physics difference (sigma_e): (a) change of the trace's white noise vs true change; (b) after the
                        learned gain change; (c) after the learned correction; (d) how big the correction is
  chg_combined_trees.png  trees on [features A, features B, B - A]: (a) predicted vs true; (b) does it get the size
                        right; (c) which features, and whether through A, B or B - A
  chg_classifier.png    direction classifier: (a) is its confidence right; (b) its confidence for changes that truly go
                        up and down; (c) accuracy when only its confident answers are kept
Numbers: model_diagnostics.json and model_diagnostics.md.

    .venv/bin/python lvl1/model_diagnostics_v2.py --out processed_v2_trainD/lvl1_results/model_diagnostics
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "lvl1"), str(ROOT / "src")]
import change_test as ct  # noqa: E402
import funnel_shortlist as fs  # noqa: E402
import scorecard as sc  # noqa: E402
import scorecard_v2 as sv  # noqa: E402
import train_lvl1 as tl  # noqa: E402
from ml import models, structured as st  # noqa: E402

warnings.filterwarnings("ignore")
T = ["sigma_e", "lambda", "gain"]
NICE = {"sigma_e": "noise σe", "lambda": "dark charge λ", "gain": "gain"}
CLEAR = dict(zip(T, sc.CLEAR))
COL = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
RUN = ROOT / "processed_v2_trainD" / "lvl1_results" / "taskB_scorecard_D_newmodels"


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", fontsize=10, loc="left")


def group_names(names, X):
    """Columns that are identical (or move in lockstep) share one name; returns {group name: [column indices]}."""
    groups = []
    for j in range(X.shape[1]):
        for g in groups:
            a, b = X[:, j], X[:, g[0]]
            if np.allclose(a, b) or (np.std(a) > 0 and np.std(b) > 0 and abs(np.corrcoef(a, b)[0, 1]) > 0.999):
                g.append(j)
                break
        else:
            groups.append([j])
    return {(names[g[0]] + (f" (+{len(g) - 1})" if len(g) > 1 else "")): g for g in groups}


def coverage(y, m, s):
    z = (y - m) / s
    ok = np.isfinite(z)
    return z[ok], float(np.mean(np.abs(z[ok]) < 1)), float(np.mean(np.abs(z[ok]) < 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-physics", action="store_true", help="skip the slow physics refit (SAAS GP for lambda)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    from sklearn.inspection import partial_dependence, permutation_importance

    data = (ROOT / "processed_v2_trainD").resolve()
    sv.cm.set_data(data)
    sc.DATA = data
    files, rows, d = sv.load(data, (0, 1, 3), None, False)
    runner = sc.Runner(files, rows, False, tl.load_tiers(ct.TIERS))
    Y = runner.Y
    amp, blk, pk = d.channel.to_numpy(), d.block.to_numpy(), d.packet_id.to_numpy()
    splits = [(np.flatnonzero((blk != k) & (amp != a)), np.flatnonzero((blk == k) & (amp == a)))
              for a in np.unique(amp) for k in range(sv.N_BLOCKS)]
    splits = [s for s in splits if len(s[1])]
    _, _, abs_preds, P, rel = pickle.loads((RUN / "predictions.pkl").read_bytes())
    TR = sc.trace_table(files, rows, d)
    res = {}
    md = ["# Model diagnostics (v2 session D; the amplifier predicted was never in training)", ""]

    # ===================================================================== A1 settings GP
    runner.no_settings, runner.feature_spec = False, "all"
    th_all = runner.f("features")
    th = th_all["theta"]
    th_names = [str(n) for n in th_all["theta_names"]]
    m_ = np.full(Y.shape, np.nan)
    s_ = np.full(Y.shape, np.nan)
    for tr, te in splits:
        xtr, xte, _ = tl.prep_columns(th[tr], th[te])
        m_[te], s_[te] = models.GP(max_iter=150).fit(xtr, Y[tr]).predict(xte)
    xall, _, keep = tl.prep_columns(th, th)
    gp_all = models.GP(max_iter=150).fit(xall, Y)
    names = [n for n, k in zip(th_names, keep) if k]
    groups = group_names(names, xall)
    # a setting that never changed (its column is constant up to float rounding) has no learnable length scale:
    # left out of the relevance plot (it does not affect the predictions)
    groups = {g: idx for g, idx in groups.items() if np.std(xall[:, idx[0]]) > 1e-9 * max(1.0, abs(np.mean(xall[:, idx[0]])))}
    fig, axes = plt.subplots(3, 3, figsize=(15, 12.5))
    md += ["## Settings GP (absolute)", "", "| | " + " | ".join(NICE[t] for t in T) + " |", "|---|---|---|---|"]
    rows_md = {"typical error %": [], "inside its ±1σ (ideal 68 %)": [], "inside its ±2σ (ideal 95 %)": []}
    for ti, t in enumerate(T):
        ax = axes[0, ti]
        for a in np.unique(amp):
            s = amp == a
            ax.errorbar(np.exp(Y[s, ti]), np.exp(m_[s, ti]), yerr=[np.exp(m_[s, ti]) - np.exp(m_[s, ti] - s_[s, ti]),
                        np.exp(m_[s, ti] + s_[s, ti]) - np.exp(m_[s, ti])], fmt="o", ms=3, alpha=0.6, color=COL[a],
                        elinewidth=0.6, label=f"ch{a}")
        lo, hi = np.exp(Y[:, ti]).min() * 0.85, np.exp(Y[:, ti]).max() * 1.15
        ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"measured {NICE[t]}"); ax.set_ylabel(f"predicted {NICE[t]}")
        err = 100 * np.sqrt(np.nanmean((m_[:, ti] - Y[:, ti]) ** 2))
        panel(ax, "abc"[ti], f"{NICE[t]}: predicted vs measured (error {err:.0f} %)")
        z, c1, c2 = coverage(Y[:, ti], m_[:, ti], s_[:, ti])
        ax = axes[1, ti]
        ax.hist(np.clip(z, -6, 6), bins=np.linspace(-6, 6, 25), density=True, alpha=0.6, color="#4c72b0")
        xs = np.linspace(-6, 6, 200)
        ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "k-", lw=1, label="if the error bars were honest")
        ax.set_xlabel("miss ÷ stated error bar")
        panel(ax, "def"[ti], f"{NICE[t]}: honest error bars? {100 * c1:.0f} % within ±1σ (ideal 68)")
        ax.legend(fontsize=7)
        g_ls = gp_all.length_scales(ti)
        rel_ = {gname: float(np.sqrt(np.sum(1 / g_ls[idx] ** 2))) for gname, idx in groups.items()}
        top = sorted(rel_, key=rel_.get, reverse=True)[:10]
        ax = axes[2, ti]
        ax.barh(range(len(top))[::-1], [rel_[k] / max(rel_.values()) for k in top], color="#55a868")
        ax.set_yticks(range(len(top))[::-1], top, fontsize=8)
        ax.set_xlabel("relevance (1 / length scale, largest = 1)")
        panel(ax, "ghi"[ti], f"{NICE[t]}: which settings it uses")
        rows_md["typical error %"].append(f"{err:.1f}")
        rows_md["inside its ±1σ (ideal 68 %)"].append(f"{100 * c1:.0f}")
        rows_md["inside its ±2σ (ideal 95 %)"].append(f"{100 * c2:.0f}")
        res.setdefault("settings_gp", {})[t] = {"error": err, "cover1": c1, "cover2": c2, "relevance_top": {k: rel_[k] for k in top}}
    axes[0, 0].legend(fontsize=7)
    md += [f"| {k} | " + " | ".join(v) + " |" for k, v in rows_md.items()] + [""]
    md += ["Most relevant settings: " + "; ".join(f"{NICE[t]}: " + ", ".join(k.replace("theta.", "") for k in list(res["settings_gp"][t]["relevance_top"])[:3]) for t in T), ""]
    fig.suptitle("Settings-only Gaussian process (absolute): new amplifier", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "abs_settings_gp.png", dpi=120)
    plt.close(fig)
    print("settings GP done", flush=True)

    # ===================================================================== A2 physics model
    if not args.skip_physics:
        runner.no_settings, runner.feature_spec = True, tl.DEFAULT_MODEL_TIERS
        dphys = runner.f("physics")
        comp = {k: np.full(len(Y), np.nan) for k in ("phys_adu", "gain", "corr")}
        pm, ps = np.full(Y.shape, np.nan), np.full(Y.shape, np.nan)
        for i, (tr, te) in enumerate(splits):
            Xtr, Xte = runner._physics_X(dphys, tr, te)
            mdl = st.PhysicsStructured(lambda_kind="saas", lambda_kw=tl.SAAS_LAMBDA_KW, noise_source="white").fit(Xtr, Y[tr])
            pm[te], ps[te] = mdl.predict(Xte)
            comp["phys_adu"][te] = mdl.physics_log_sigma_adu(Xte)
            comp["gain"][te] = mdl.gain.predict(Xte)[0][:, 0]
            comp["corr"][te] = mdl.resid.predict(Xte)[0][:, 0]
            print(f"  physics split {i + 1}/{len(splits)}", flush=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        sig_adu = Y[:, 0] + Y[:, 2]                         # the image's noise in ADU = sigma_e x gain (logs)
        ax = axes[0, 0]
        for a in np.unique(amp):
            s = amp == a
            ax.plot(np.exp(sig_adu[s]), np.exp(comp["phys_adu"][s]), "o", ms=3, alpha=0.6, color=COL[a], label=f"ch{a}")
        lo, hi = np.exp(np.nanmin(sig_adu)) * 0.8, np.exp(np.nanmax(sig_adu)) * 1.2
        ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("image noise in ADU (σe × gain)"); ax.set_ylabel("trace: white noise per skip / √N (ADU)")
        r_ = np.corrcoef(sig_adu, comp["phys_adu"])[0, 1]
        panel(ax, "a", f"the formula's trace term (correlation {r_:.2f})")
        ax.legend(fontsize=7)
        ax = axes[0, 1]
        for a in np.unique(amp):
            s = amp == a
            ax.plot(np.exp(Y[s, 2]), np.exp(comp["gain"][s]), "o", ms=3, alpha=0.6, color=COL[a])
        lo, hi = np.exp(Y[:, 2]).min() * 0.98, np.exp(Y[:, 2]).max() * 1.02
        ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
        ax.set_xlabel("measured gain (ADU/e⁻)"); ax.set_ylabel("learned gain")
        eg = 100 * np.sqrt(np.nanmean((comp["gain"] - Y[:, 2]) ** 2))
        panel(ax, "b", f"the learned gain (error {eg:.1f} %)")
        ax = axes[0, 2]
        ax.hist(100 * comp["corr"], bins=30, color="#4c72b0", alpha=0.7)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("learned correction, % of σe")
        panel(ax, "c", f"the learned correction (typical size {100 * np.nanstd(comp['corr']):.0f} %)")
        for col_, ti in ((0, 0), (1, 1)):
            ax = axes[1, col_]
            for a in np.unique(amp):
                s = amp == a
                ax.plot(np.exp(Y[s, ti]), np.exp(pm[s, ti]), "o", ms=3, alpha=0.6, color=COL[a])
            lo, hi = np.exp(Y[:, ti]).min() * 0.8, np.exp(Y[:, ti]).max() * 1.2
            ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(f"measured {NICE[T[ti]]}"); ax.set_ylabel(f"predicted {NICE[T[ti]]}")
            e_ = 100 * np.sqrt(np.nanmean((pm[:, ti] - Y[:, ti]) ** 2))
            panel(ax, "de"[col_], f"final {NICE[T[ti]]} (error {e_:.0f} %)" + (": formula" if ti == 0 else ": learned, no formula"))
        ax = axes[1, 2]
        cov = {}
        for ti, t in enumerate(T):
            z, c1, c2 = coverage(Y[:, ti], pm[:, ti], ps[:, ti])
            cov[t] = (c1, c2)
        ax.bar(np.arange(3) - 0.2, [100 * cov[t][0] for t in T], 0.4, label="within ±1σ (ideal 68 %)", color="#4c72b0")
        ax.bar(np.arange(3) + 0.2, [100 * cov[t][1] for t in T], 0.4, label="within ±2σ (ideal 95 %)", color="#dd8452")
        ax.axhline(68, color="#4c72b0", ls=":"); ax.axhline(95, color="#dd8452", ls=":")
        ax.set_xticks(range(3), [NICE[t] for t in T]); ax.set_ylim(0, 100); ax.legend(fontsize=7)
        panel(ax, "f", "honest error bars?")
        fig.suptitle("Trace physics model (absolute): σe = trace noise term ÷ learned gain × learned correction; "
                     "new amplifier", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / "abs_physics.png", dpi=120)
        plt.close(fig)
        res["physics"] = {"trace_term_corr": float(r_), "gain_error": eg, "correction_sd": float(100 * np.nanstd(comp["corr"])),
                          "error": {t: float(100 * np.sqrt(np.nanmean((pm[:, i] - Y[:, i]) ** 2))) for i, t in enumerate(T)},
                          "coverage": {t: list(map(float, cov[t])) for t in T}}
        md += ["## Trace physics model (absolute)", "",
               f"Trace term vs image noise in ADU: correlation {r_:.2f}. Learned gain error {eg:.1f} %. Learned correction: "
               f"typical size {100 * np.nanstd(comp['corr']):.0f} %. Final error: " + ", ".join(
                   f"{NICE[t]} {res['physics']['error'][t]:.0f} %" for t in T) + ". Within ±1σ / ±2σ: " + ", ".join(
                   f"{NICE[t]} {100 * cov[t][0]:.0f} / {100 * cov[t][1]:.0f} %" for t in T) + ".", ""]
        print("physics done", flush=True)

    # ===================================================================== A3 trace trees
    runner.no_settings, runner.feature_spec = True, tl.MODEL_TIERS["trees"]
    dfe = runner.f("features")
    xs = tl.scalars(dfe, True, False)
    fnames = ([str(n) for n in dfe["theta_names"]] + [f"own: {n}" for n in dfe["feature_names"]]
              + [f"ref: {n}" for n in dfe["feature_names"]])
    xall, _, keep = tl.prep_columns(xs, xs)
    fnames = [n for n, k in zip(fnames, keep) if k]
    tp = abs_preds["new"]["trace: trees"]
    from sklearn.model_selection import GroupKFold
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    md += ["## Trace trees (absolute)", ""]
    res["trees"] = {}
    for ti, t in enumerate(T):
        ax = axes[0, ti]
        for a in np.unique(amp):
            s = amp == a
            ax.plot(np.exp(Y[s, ti]), np.exp(tp[s, ti]), "o", ms=3, alpha=0.6, color=COL[a], label=f"ch{a}")
        lo, hi = np.exp(Y[:, ti]).min() * 0.8, np.exp(Y[:, ti]).max() * 1.2
        ax.plot([lo, hi], [lo, hi], "k:", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"measured {NICE[t]}"); ax.set_ylabel(f"predicted {NICE[t]}")
        e_ = 100 * np.sqrt(np.nanmean((tp[:, ti] - Y[:, ti]) ** 2))
        panel(ax, "abc"[ti], f"{NICE[t]}: predicted vs measured (error {e_:.0f} %)")
        imps = []
        for tr, te in GroupKFold(5).split(xall, Y[:, ti], pk):
            m = H(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8, random_state=0).fit(xall[tr], Y[tr, ti])
            imps.append(permutation_importance(m, xall[te], Y[te, ti], n_repeats=3, random_state=0).importances_mean)
        imp = np.mean(imps, 0)
        top = np.argsort(-imp)[:8]
        ax = axes[1, ti]
        ax.barh(range(len(top))[::-1], imp[top], color="#55a868")
        ax.set_yticks(range(len(top))[::-1], [fnames[j][:48] for j in top], fontsize=7)
        ax.set_xlabel("importance (drop in R² when shuffled)")
        panel(ax, "def"[ti], f"{NICE[t]}: the trace features it uses")
        mfull = H(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8, random_state=0).fit(xall, Y[:, ti])
        j = top[0]
        ax = axes[2, ti]
        for a in np.unique(amp):
            s = amp == a
            ax.plot(xall[s, j], 100 * (Y[s, ti] - Y[:, ti].mean()), "o", ms=3, alpha=0.5, color=COL[a])
        pdp = partial_dependence(mfull, xall, [j], grid_resolution=30, kind="average")
        ax.plot(pdp["grid_values"][0], 100 * (pdp["average"][0] - Y[:, ti].mean()), "k-", lw=2, label="the model")
        ax.set_xlabel(fnames[j][:60], fontsize=8)
        ax.set_ylabel(f"{NICE[t]} − mean, %")
        ax.legend(fontsize=7)
        panel(ax, "ghi"[ti], f"{NICE[t]}: response to its top feature")
        res["trees"][t] = {"error": e_, "top": [fnames[k] for k in top[:5]]}
        md.append(f"* {NICE[t]}: error {e_:.0f} %; top features: " + "; ".join(fnames[k] for k in top[:4]))
    md.append("")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Trace boosted trees (absolute): new amplifier", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "abs_trees.png", dpi=120)
    plt.close(fig)
    print("trees done", flush=True)

    # ===================================================================== change models: shared bits
    Yp, typ = P["Y"], P["type"]
    img = typ != 1
    A_, B_ = P["A"], P["B"]
    THp = np.nan_to_num(TR["TH"])
    dTH = THp[B_] - THp[A_]
    tnames = [c[len("theta."):] for c in d.columns if c.startswith("theta.")]
    i_int = tnames.index("delayInt")
    dl = np.log2(THp[B_, i_int] / THp[A_, i_int])
    fixed = np.abs(dl) < 0.1

    def scatter_change(ax, q, ti, letter, name, size=True):
        ok = img & np.isfinite(q)
        y = 100 * Yp[ok, ti]
        qq = q[ok] * (100 if size else 1)
        ax.plot(y[~fixed[ok]], qq[~fixed[ok]], ".", ms=3, alpha=0.35, color="#4c72b0", label="delayInt changed")
        ax.plot(y[fixed[ok]], qq[fixed[ok]], ".", ms=4, alpha=0.7, color="#dd8452", label="delayInt unchanged")
        lim = np.nanpercentile(np.abs(y), 99)
        ax.set_xlim(-lim, lim)
        if size:
            ax.set_ylim(-lim, lim)
            ax.plot([-lim, lim], [-lim, lim], "k:", lw=0.8)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        m = np.abs(Yp[ok, ti]) > CLEAR[T[ti]]
        acc = np.mean(np.sign(q[ok][m]) == np.sign(Yp[ok, ti][m]))
        ax.set_xlabel(f"true change of {NICE[T[ti]]}, %")
        ax.set_ylabel("predicted change, %" if size else "predicted P(up) − 0.5")
        panel(ax, letter, f"{NICE[T[ti]]}: {name} (direction {100 * acc:.0f} %)")
        return acc

    # ===================================================================== C1 settings change GP
    q_sg = rel["new"]["settings change: gp"]
    tr_idx = np.flatnonzero(img)
    rng = np.random.default_rng(0)
    sub = rng.choice(tr_idx, min(800, len(tr_idx)), replace=False)
    Xg = np.vstack([dTH[sub], -dTH[sub]])
    xg, _, kg = tl.prep_columns(Xg, Xg)
    gpc = models.GP(max_points=800, max_iter=150).fit(xg, np.vstack([Yp[sub], -Yp[sub]]))
    gnames = [n for n, k in zip(tnames, kg) if k]
    ggroups = group_names(gnames, xg)
    ggroups = {g: idx for g, idx in ggroups.items() if np.std(xg[:, idx[0]]) > 1e-9}
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    res["settings_change_gp"] = {}
    md += ["## Settings change GP", ""]
    for ti, t in enumerate(T):
        acc = scatter_change(axes[0, ti], q_sg[:, ti], ti, "abc"[ti], "predicted vs true change")
        ls_ = gpc.length_scales(ti)
        rel_ = {g: float(np.sqrt(np.sum(1 / ls_[idx] ** 2))) for g, idx in ggroups.items()}
        top = sorted(rel_, key=rel_.get, reverse=True)[:10]
        ax = axes[1, ti]
        ax.barh(range(len(top))[::-1], [rel_[k] / max(rel_.values()) for k in top], color="#55a868")
        ax.set_yticks(range(len(top))[::-1], [f"Δ {k}" for k in top], fontsize=8)
        ax.set_xlabel("relevance (1 / length scale, largest = 1)")
        panel(ax, "def"[ti], f"{NICE[t]}: which setting changes it uses")
        # response to a change of delayInt alone, from three starting delayInt values
        ax = axes[2, ti]
        kint = [j for j, n in enumerate(gnames) if n == "delayInt"]
        for start, c in ((500, "#4c72b0"), (1000, "#55a868"), (2000, "#c44e52")):
            dd = np.linspace(-0.8 * start, 1.5 * start, 40)
            Xq = np.zeros((len(dd), len(gnames)))
            if kint:
                Xq[:, kint[0]] = dd
            mq, sq = gpc.predict(Xq)
            ax.plot(start + dd, 100 * mq[:, ti], "-", color=c, label=f"from delayInt {start}")
            ax.fill_between(start + dd, 100 * (mq[:, ti] - sq[:, ti]), 100 * (mq[:, ti] + sq[:, ti]), color=c, alpha=0.12)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("new delayInt (cycles)"); ax.set_ylabel(f"predicted change of {NICE[t]}, %")
        ax.legend(fontsize=7)
        panel(ax, "ghi"[ti], f"{NICE[t]}: its learned response to delayInt alone")
        res["settings_change_gp"][t] = {"direction": float(acc), "relevance_top": {k: rel_[k] for k in top[:5]}}
        md.append(f"* {NICE[t]}: direction {100 * acc:.0f} %; relies on " + ", ".join(top[:3]))
    md.append("")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Settings change Gaussian process: Δ(settings) → Δ(label); new amplifier", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "chg_settings_gp.png", dpi=120)
    plt.close(fig)
    print("settings change GP done", flush=True)

    # ===================================================================== C2 physics difference (sigma_e)
    dw = TR["white"][B_] - TR["white"][A_]
    q_p = rel["new"]["trace: physics difference (sigma_e)"][:, 0]
    q_pc = rel["new"]["trace: physics difference + correction (sigma_e)"][:, 0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    a1 = scatter_change(axes[0, 0], np.where(np.isfinite(q_p), dw, np.nan), 0, "a", "Δ trace white noise only")
    a2 = scatter_change(axes[0, 1], q_p, 0, "b", "− learned Δ gain")
    a3 = scatter_change(axes[1, 0], q_pc, 0, "c", "+ learned correction")
    ax = axes[1, 1]
    corr = q_pc - q_p
    ok = img & np.isfinite(corr)
    ax.plot(dl[ok], 100 * corr[ok], ".", ms=3, alpha=0.4, color="#4c72b0")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Δ log₂ delayInt (1 = doubled)"); ax.set_ylabel("learned correction, %")
    panel(ax, "d", f"size of the correction (typical {100 * np.nanstd(corr[ok]):.0f} %)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Physics difference model (σe change): Δσe = Δ(trace white noise) − Δ(gain) + learned correction; "
                 "new amplifier", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "chg_physics.png", dpi=120)
    plt.close(fig)
    res["physics_difference"] = {"direction_white_only": float(a1), "direction_with_gain": float(a2),
                                 "direction_with_correction": float(a3), "correction_sd": float(100 * np.nanstd(corr[ok]))}
    md += ["## Physics difference (σe change)", "",
           f"Direction on clear changes: trace white noise alone {100 * a1:.0f} %, minus the learned gain change "
           f"{100 * a2:.0f} %, plus the learned correction {100 * a3:.0f} %; correction typically {100 * np.nanstd(corr[ok]):.0f} %.", ""]
    print("physics difference done", flush=True)

    # ===================================================================== C3 combined trees
    q_ct = rel["new"]["trace: combined trees [A, B, B-A]"]
    F = TR["F"]
    tiers = tl.load_tiers(ct.TIERS)
    feats = [f for f, ts in tiers.items() if ts & {"P", "A1", "A2"} and f"trace.{f}" in d and f"ref.{f}" in d]
    nf = F.shape[1]
    Xc = np.hstack([F[A_], F[B_], F[B_] - F[A_]])
    Xcs = np.hstack([F[B_], F[A_], F[A_] - F[B_]])
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    res["combined_trees"] = {}
    md += ["## Combined trees (change)", ""]
    rng = np.random.default_rng(1)
    for ti, t in enumerate(T):
        acc = scatter_change(axes[0, ti], q_ct[:, ti], ti, "abc"[ti], "predicted vs true change")
        ax = axes[1, ti]
        ok = img & np.isfinite(q_ct[:, ti])
        y, q = Yp[ok, ti], q_ct[ok, ti]
        edges = np.quantile(y, np.linspace(0, 1, 11))
        mids = [y[(y >= a) & (y <= b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        mq = [q[(y >= a) & (y <= b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        slope = float(np.polyfit(y, q, 1)[0])
        ax.plot(100 * np.array(mids), 100 * np.array(mq), "o-", color="#4c72b0")
        lim = max(abs(100 * edges[0]), abs(100 * edges[-1]))
        ax.plot([-lim, lim], [-lim, lim], "k:", lw=0.8)
        ax.set_xlabel(f"true change of {NICE[t]}, % (bin means)"); ax.set_ylabel("mean predicted change, %")
        panel(ax, "def"[ti], f"{NICE[t]}: right size? slope {slope:.2f} (1 = right)")
        # importance: fit on all image pairs (both orders), permutation on held-out packets, grouped per feature
        idx = np.flatnonzero(img)
        tr_, te_ = idx[d.block.to_numpy()[P["ra"][idx]] != 0], idx[d.block.to_numpy()[P["ra"][idx]] == 0]
        f = fs.gbt(np.vstack([Xc[tr_], Xcs[tr_]]), np.vstack([Yp[tr_], -Yp[tr_]]))
        base = np.mean((f(Xc[te_])[:, ti] - Yp[te_, ti]) ** 2)
        blocks = {"A (before)": range(0, nf), "B (after)": range(nf, 2 * nf), "B − A (difference)": range(2 * nf, 3 * nf)}
        feat_imp = np.zeros(nf)
        block_imp = {}
        Xte = Xc[te_].copy()
        for bname, cols in blocks.items():
            Xp_ = Xte.copy()
            Xp_[:, list(cols)] = Xp_[rng.permutation(len(Xp_))][:, list(cols)]
            block_imp[bname] = float(np.mean((f(Xp_)[:, ti] - Yp[te_, ti]) ** 2) - base)
        for j in range(nf):
            Xp_ = Xte.copy()
            perm = rng.permutation(len(Xp_))
            for off in (0, nf, 2 * nf):
                Xp_[:, off + j] = Xte[perm, off + j]
            feat_imp[j] = np.mean((f(Xp_)[:, ti] - Yp[te_, ti]) ** 2) - base
        top = np.argsort(-feat_imp)[:7]
        ax = axes[2, ti]
        labels = [feats[j][:40] for j in top] + ["— block: " + b for b in blocks]
        vals = list(feat_imp[top] / base) + [block_imp[b] / base for b in blocks]
        cols_ = ["#55a868"] * len(top) + ["#8172b2"] * 3
        ax.barh(range(len(vals))[::-1], vals, color=cols_)
        ax.set_yticks(range(len(vals))[::-1], labels, fontsize=7)
        ax.set_xlabel("error increase when shuffled (× the model's error)")
        panel(ax, "ghi"[ti], f"{NICE[t]}: which features (green) and which part (purple)")
        res["combined_trees"][t] = {"direction": float(acc), "slope": slope, "top": [feats[j] for j in top[:5]],
                                    "blocks": block_imp}
        md.append(f"* {NICE[t]}: direction {100 * acc:.0f} %, size slope {slope:.2f}; top features: "
                  + "; ".join(feats[j] for j in top[:3]) + "; by part: " + ", ".join(f"{b} {v / base:.2f}" for b, v in block_imp.items()))
    md.append("")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Combined trees (change): trees on [features before, after, after − before]; new amplifier", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "chg_combined_trees.png", dpi=120)
    plt.close(fig)
    print("combined trees done", flush=True)

    # ===================================================================== C4 direction classifier
    q_cl = rel["new"]["trace: direction classifier [A, B, B-A]"]
    from sklearn.metrics import roc_auc_score
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5))
    res["classifier"] = {}
    md += ["## Direction classifier (change)", ""]
    for ti, t in enumerate(T):
        ok = img & np.isfinite(q_cl[:, ti]) & (np.abs(Yp[:, ti]) > sc.cm.NOISE[ti])
        p = np.clip(q_cl[ok, ti] + 0.5, 0, 1)
        up = Yp[ok, ti] > 0
        bins = np.linspace(0, 1, 11)
        k = np.clip(np.digitize(p, bins) - 1, 0, 9)
        fr = [up[k == b].mean() if (k == b).sum() > 4 else np.nan for b in range(10)]
        ax = axes[0, ti]
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.plot(0.5 * (bins[1:] + bins[:-1]), fr, "o-", color="#4c72b0")
        auc = roc_auc_score(up, p)
        ax.set_xlabel("stated probability that it goes up"); ax.set_ylabel("how often it really went up")
        panel(ax, "abc"[ti], f"{NICE[t]}: is its confidence right? (ROC area {auc:.2f})")
        ax = axes[1, ti]
        ax.hist(p[up], bins=bins, alpha=0.6, color="#c44e52", label="really went up")
        ax.hist(p[~up], bins=bins, alpha=0.6, color="#4c72b0", label="really went down")
        ax.set_xlabel("stated probability that it goes up"); ax.set_ylabel("pairs")
        ax.legend(fontsize=7)
        panel(ax, "def"[ti], f"{NICE[t]}: its answers for true ups and downs")
        ax = axes[2, ti]
        conf = np.abs(p - 0.5)
        o = np.argsort(-conf)
        right = (p[o] > 0.5) == up[o]
        kept = np.arange(1, len(o) + 1) / len(o)
        ax.plot(100 * kept, 100 * np.cumsum(right) / np.arange(1, len(o) + 1), "-", color="#55a868")
        ax.set_xlabel("pairs kept, % (most confident first)"); ax.set_ylabel("direction right, %")
        ax.set_ylim(50, 101)
        ax.grid(alpha=0.3)
        a50 = 100 * right[:len(o) // 2].mean()
        panel(ax, "ghi"[ti], f"{NICE[t]}: accuracy if only confident answers are used")
        res["classifier"][t] = {"auc": float(auc), "acc_all": float(100 * right.mean()), "acc_top_half": float(a50)}
        md.append(f"* {NICE[t]}: ROC area {auc:.2f}; direction {100 * right.mean():.0f} % on all pairs above the label "
                  f"noise, {a50:.0f} % on the most confident half")
    md.append("")
    fig.suptitle("Direction classifier (change): the probability that the label goes up from trace A to trace B; "
                 "new amplifier", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "chg_classifier.png", dpi=120)
    plt.close(fig)
    (out / "model_diagnostics.md").write_text("\n".join(md) + "\n")
    (out / "model_diagnostics.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
