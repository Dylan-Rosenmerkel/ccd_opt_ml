"""
story_figures.py -- the figures of the level-1 story (v1 -> v2 -> v3), one message per figure.

 1 What the first night (v1) taught us
   s1_amp_vs_settings.png     the dark charge and the gain sort by amplifier, the noise by the settings
   s1_fingerprint.png         a trace names its amplifier: which features carry the amplifier (eta^2)
   s1_known_new_onefix.png    so absolute models work on known amplifiers, fail on a new one, and one image fixes it
   s1_change_idea.png         the way out: predict CHANGES (an amplifier's level cancels); and why v2 changed the reference
 2 Which parameters move things (v2, the best models)
   s2_corr_settings.png       correlation of every setting with the labels, as levels and as changes
   s2_corr_traces.png         correlation of the settings with the key trace features (what the traces see)
   s2_model_effects.png       the effect of each setting: measured, and as the best change models see it
 3 What v2 showed
   s3_absolute.png            absolute models: known vs new amplifier
   s3_change.png              change models on a new amplifier (session D cross-validation and session E)
 4 The integration time did most of the work, and we knew it
   s4_physics.png             the labels against delayInt with the physics expectation
   s4_shares.png              how much delayInt alone explains, against all settings and the traces
   s4_without.png             the change models where delayInt does not change, and a settings model not told delayInt
 5 What the next data tests
   s5_design_corr.png         which settings moved together: v2 against the v3 noise set
   s5_coverage.png            packets that change only voltages (timing fixed), and the voltage ranges
   s5_detectable.png          the smallest voltage effect each design can see
Numbers: story.json.

    .venv/bin/python lvl1/story_figures.py --out docs/overleaf_level1_story_2026-11-10/figures
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
import lambda_settings_figures as LSF  # noqa: E402
import parameter_influence as pi  # noqa: E402

warnings.filterwarnings("ignore")
T = ["sigma_e", "lambda", "gain"]
NICE = {"sigma_e": "noise σe", "lambda": "dark charge λ", "gain": "gain"}
COL = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
CLEAR = {"sigma_e": 0.05, "lambda": 0.15, "gain": 0.012}
V1 = dict(data=ROOT / "processed_v1", plan=[ROOT / "lvl1" / "packet_plan" / "plan.json"], sessions=["A", "B", "C"])
V2 = dict(data=ROOT / "processed_v2", plan=[ROOT / "lvl1" / "packet_plan_v2" / "plan.json"], sessions=["D", "E"])
RUN = ROOT / "processed_v2_trainD" / "lvl1_results" / "taskB_scorecard_D_newmodels"
DIAG = ROOT / "processed_v2_trainD" / "lvl1_results" / "diagnostics"


def gbt():
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    return H(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8, random_state=0)


def trace_cols(d):
    return [c for c in d.columns if (c.startswith("trace.") or c.startswith("ref.")) and pd.api.types.is_numeric_dtype(d[c])
            and d[c].notna().mean() > 0.9 and d[c].nunique() > 5]


def r2(y, p):
    return float(1 - np.nanmean((y - p) ** 2) / np.nanvar(y))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GroupKFold
    res = {}
    plt.rcParams.update({"font.size": 10})

    # ================================================================== 1. v1
    d1 = pi.load(V1["data"], V1["plan"], V1["sessions"], [0, 1, 3])
    K1, _, _ = pi.knobs(d1)
    ch1 = d1.channel.to_numpy().astype(int)
    pk1 = d1.packet_id.to_numpy()
    oh1 = np.eye(4)[ch1][:, [0, 1, 3]]
    # 1a: sort by amplifier or by settings
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.2]})
    shares = {}
    for ax, t in zip(axes[:3], T):
        for j, a in enumerate((0, 1, 3)):
            v = d1.loc[ch1 == a, f"log_{t}"]
            ax.plot(np.full(len(v), j) + np.random.default_rng(a).uniform(-0.15, 0.15, len(v)), np.exp(v), "o", ms=3,
                    alpha=0.5, color=COL[a])
        ax.set_xticks(range(3), ["ch0", "ch1", "ch3"])
        ax.set_yscale("log")
        ax.set_title(NICE[t])
        ax.grid(alpha=0.3, axis="y")
        y = d1[f"log_{t}"].to_numpy()
        shares[t] = {"amplifier": LSF.cv_r2(oh1, y, pk1)[0], "settings": LSF.cv_r2(K1.to_numpy(), y, pk1)[0]}
    ax = axes[3]
    x = np.arange(3)
    ax.bar(x - 0.2, [max(0, shares[t]["amplifier"]) for t in T], 0.4, color="#c44e52", label="which amplifier")
    ax.bar(x + 0.2, [max(0, shares[t]["settings"]) for t in T], 0.4, color="#4c72b0", label="the settings")
    ax.set_xticks(x, [NICE[t] for t in T])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of the spread explained")
    ax.legend(fontsize=8)
    ax.set_title("explained by ...")
    fig.suptitle("v1: every packet of the first night, per amplifier. Gain and dark charge sort by amplifier "
                 "(the settings explain almost none of them); the noise depends on the settings as much as on the amplifier",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "s1_amp_vs_settings.png", dpi=130)
    plt.close(fig)
    res["v1_shares"] = shares

    # 1b: fingerprints
    tc = trace_cols(d1)
    eta = {}
    for c in tc:
        x_ = d1[c].astype(float)
        ok = x_.notna()
        g = x_[ok].groupby(ch1[ok.to_numpy()])
        eta[c] = float(g.size().mul((g.mean() - x_[ok].mean()) ** 2).sum() / ((x_[ok] - x_[ok].mean()) ** 2).sum())
    own = [eta[c] for c in tc if c.startswith("trace.")]
    ref = [eta[c] for c in tc if c.startswith("ref.")]
    from sklearn.ensemble import HistGradientBoostingClassifier as HC
    from sklearn.model_selection import cross_val_predict
    acc = {}
    for name, cols in (("own trace", [c for c in tc if c.startswith("trace.")]),
                       ("reference trace (at nominal)", [c for c in tc if c.startswith("ref.")])):
        p = cross_val_predict(HC(max_iter=200, random_state=0), d1[cols].to_numpy(float), ch1, groups=pk1, cv=GroupKFold(5))
        acc[name] = float(np.mean(p == ch1))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].hist([own, ref], bins=np.linspace(0, 1, 21), label=["own-trace features", "reference-trace features"],
                 color=["#4c72b0", "#c44e52"])
    axes[0].axvline(0.9, color="k", ls=":")
    axes[0].set_xlabel("η²: share of the feature's spread that is just 'which amplifier'")
    axes[0].set_ylabel("trace features")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{sum(e > 0.9 for e in eta.values())} features are amplifier fingerprints (η² > 0.9)")
    axes[1].barh(range(len(acc)), [100 * v for v in acc.values()], color=["#4c72b0", "#c44e52"])
    axes[1].set_yticks(range(len(acc)), list(acc))
    axes[1].axvline(100 / 3, color="k", ls=":")
    axes[1].text(100 / 3 + 1, -0.4, "guessing", fontsize=8)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("amplifier named correctly from one trace, % (unseen packets)")
    axes[1].set_title("a trace names its amplifier")
    fig.suptitle("v1: the traces carry the amplifier's identity", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "s1_fingerprint.png", dpi=130)
    plt.close(fig)
    res["v1_fingerprint"] = {"n_eta_gt_0.9": int(sum(e > 0.9 for e in eta.values())), "n_features": len(eta), "accuracy": acc}

    # 1c: known vs new amplifier vs one image (trace model: trees on own + reference trace features)
    X1 = d1[tc].to_numpy(float)
    rng = np.random.default_rng(0)
    errs = {t: {} for t in T}
    for ti, t in enumerate(T):
        y = d1[f"log_{t}"].to_numpy()
        pk_known = np.full(len(y), np.nan)
        for tr, te in GroupKFold(5).split(X1, y, pk1):
            pk_known[te] = gbt().fit(X1[tr], y[tr]).predict(X1[te])
        p_new = np.full(len(y), np.nan)
        for a in (0, 1, 3):
            te, tr = ch1 == a, ch1 != a
            p_new[te] = gbt().fit(X1[tr], y[tr]).predict(X1[te])
        fix = []
        for a in (0, 1, 3):
            idx = np.flatnonzero(ch1 == a)
            for i in idx:
                others = idx[pk1[idx] != pk1[i]]
                j = rng.choice(others, 20)
                fix.append(np.mean([(p_new[i] + (y[k] - p_new[k]) - y[i]) ** 2 for k in j]))
        mean_err = np.sqrt(np.mean((y - y.mean()) ** 2))
        errs[t] = {"average (no model)": 100 * mean_err, "known amplifier": 100 * np.sqrt(np.nanmean((pk_known - y) ** 2)),
                   "new amplifier": 100 * np.sqrt(np.nanmean((p_new - y) ** 2)),
                   "new amplifier + 1 image": 100 * np.sqrt(np.mean(fix))}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    keys = list(errs["sigma_e"])
    cols = ["#999999", "#55a868", "#c44e52", "#8172b2"]
    for ax, t in zip(axes, T):
        ax.bar(range(4), [errs[t][k] for k in keys], color=cols)
        ax.set_xticks(range(4), [k.replace(" + ", "\n+ ") for k in keys], fontsize=8)
        for i, k in enumerate(keys):
            ax.text(i, errs[t][k], f"{errs[t][k]:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("typical error %")
        ax.set_title(NICE[t])
    fig.suptitle("v1: a trace model is good on the amplifiers it knows, poor on a new one; one image of the new one "
                 "fixes the noise and the gain, and part of the dark charge", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "s1_known_new_onefix.png", dpi=130)
    plt.close(fig)
    res["v1_errors"] = errs

    # 1d: the change idea + the reference problem
    nom1 = np.full((len(d1), 3), np.nan)
    for a in (0, 1, 3):
        g = d1[ch1 == a]
        n = g[g.nominal].sort_values("hours")
        for ti, t in enumerate(T):
            nom1[g.index, ti] = np.interp(g.hours, n.hours, n[f"log_{t}"])
    Yv1 = np.column_stack([d1[f"log_{t}"] for t in T])
    dlt = Yv1 - nom1                                     # each packet against nominal (v1's reference)
    nn = ~d1.nominal.to_numpy()
    up_v1 = {t: float(np.mean(dlt[nn & (np.abs(dlt[:, i]) > CLEAR[t]), i] > 0)) for i, t in enumerate(T)}
    sc2 = json.loads((RUN / "scorecard.json").read_text())
    up_v2 = {t: sc2["relative"]["new"]["no change"]["random pairs"][t]["clear_up"] for t in T}
    # absolute new-amplifier error split into the amplifier's offset and the rest (v1, trace model)
    split = {}
    for ti, t in enumerate(T):
        y = Yv1[:, ti]
        p_new = np.full(len(y), np.nan)
        for a in (0, 1, 3):
            te, tr = ch1 == a, ch1 != a
            p_new[te] = gbt().fit(X1[tr], y[tr]).predict(X1[te])
        e = p_new - y
        off = pd.Series(e).groupby(ch1).transform("mean").to_numpy()
        split[t] = {"amplifier offset": 100 * np.sqrt(np.mean(off ** 2)), "the rest": 100 * np.sqrt(np.mean((e - off) ** 2))}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    x = np.arange(3)
    axes[0].bar(x, [split[t]["amplifier offset"] for t in T], 0.5, color="#c44e52", label="the amplifier's offset")
    axes[0].bar(x, [split[t]["the rest"] for t in T], 0.5, bottom=[split[t]["amplifier offset"] for t in T],
                color="#4c72b0", label="everything else")
    axes[0].set_xticks(x, [NICE[t] for t in T])
    axes[0].set_ylabel("error on a new amplifier, % (stacked)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("most of the new-amplifier error is one number per amplifier,\nwhich cancels when two settings of "
                      "the same amplifier are compared")
    axes[1].bar(x - 0.2, [100 * up_v1[t] for t in T], 0.4, color="#dd8452", label="v1: every packet against nominal")
    axes[1].bar(x + 0.2, [100 * up_v2[t] for t in T], 0.4, color="#55a868", label="v2: any two packets")
    axes[1].axhline(50, color="k", ls=":")
    axes[1].set_xticks(x, [NICE[t] for t in T])
    axes[1].set_ylabel("clear changes that go up, %")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8)
    axes[1].set_title("v1's changes were all 'away from nominal' (nominal is near the best noise):\n"
                      "saying 'up' was enough, so v2 compares packets both ways")
    fig.suptitle("The change models: compare two settings of the same amplifier instead of predicting a value", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "s1_change_idea.png", dpi=130)
    plt.close(fig)
    res["v1_change"] = {"offset_split": split, "up_v1": up_v1, "up_v2": up_v2}
    print("chapter 1 done", flush=True)

    # ================================================================== 2. parameters (v2)
    d2 = pi.load(V2["data"], V2["plan"], V2["sessions"], [0, 1, 3])
    K2, mem2, tim2 = pi.knobs(d2)
    ch2_ = d2.channel.to_numpy().astype(int)
    pk2 = d2.packet_id.to_numpy()
    names = list(K2.columns)
    # pairs within amplifier and session
    P = []
    for (a, s), g in d2.groupby(["channel", "session"]):
        idx = g.index.to_numpy()
        i, j = np.triu_indices(len(idx), 1)
        P.append(np.column_stack([idx[i], idx[j]]))
    P = np.vstack(P)
    dK = K2.to_numpy()[P[:, 1]] - K2.to_numpy()[P[:, 0]]
    from scipy.stats import spearmanr
    lev = np.array([[spearmanr(K2[k], d2[f"log_{t}"])[0] for t in T] for k in names])
    chg = np.array([[spearmanr(dK[:, i], d2[f"log_{t}"].to_numpy()[P[:, 1]] - d2[f"log_{t}"].to_numpy()[P[:, 0]])[0]
                     for t in T] for i in range(len(names))])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, M, ttl in ((axes[0], lev, "levels: setting vs label (every packet)"),
                       (axes[1], chg, "changes: Δ setting vs Δ label (two packets, one amplifier)")):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(3), [NICE[t] for t in T])
        ax.set_yticks(range(len(names)), names, fontsize=8)
        for i in range(M.shape[0]):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(M[i, j]) > 0.5 else "black")
        ax.set_title(ttl, fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.03, label="correlation (Spearman)")
    fig.suptitle("v2: which settings go with which label (red: up together, blue: opposite)", fontsize=11)
    fig.savefig(out / "s2_corr_settings.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    res["corr_levels"] = {n: dict(zip(T, map(float, r))) for n, r in zip(names, lev)}
    res["corr_changes"] = {n: dict(zip(T, map(float, r))) for n, r in zip(names, chg)}

    key = {"white noise per skip": "trace.fit: white noise per skip (ADU)",
           "noise of 50-skip average": "trace.skip: noise of 50-skip average",
           "1/f corner": "trace.noise: 1/f corner frequency (kHz)",
           "slow / fast CDS power": "trace.noise: CDS power <1 kHz / >10 kHz",
           "power 0.1-1 MHz": "trace.skip: noise power 0.1-1 MHz", "power 1-3 MHz": "trace.skip: noise power 1-3 MHz",
           "power 3-7.5 MHz": "trace.skip: noise power 3-7.5 MHz", "reset-level scatter": "trace.noise: reset-level scatter (ADU)",
           "baseline droop amplitude": "trace.fit: baseline droop amplitude",
           "pedestal-window settling": "trace.noise: pedestal-window settling (ADU)"}
    key = {k: v for k, v in key.items() if v in d2}
    tf = list(key) + [NICE[t] for t in T]
    vals = {k: d2[v].astype(float) for k, v in key.items()}
    vals.update({NICE[t]: d2[f"log_{t}"] for t in T})
    M = np.array([[spearmanr(K2[k], vals[f], nan_policy="omit")[0] for f in tf] for k in names])
    fig, ax = plt.subplots(figsize=(13, 6.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tf)), tf, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.axvline(len(key) - 0.5, color="k", lw=2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(M[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, label="correlation (Spearman)")
    ax.set_title("v2: the settings against what the trace measures (left of the line) and the image labels (right)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "s2_corr_traces.png", dpi=130)
    plt.close(fig)

    # effects per setting: measured vs implied by the best change models (session D CV, new amplifier)
    sys.path.insert(0, str(ROOT / "lvl1"))
    import scorecard as sc
    import scorecard_v2 as sv
    data = (ROOT / "processed_v2_trainD").resolve()
    sv.cm.set_data(data)
    sc.DATA = data
    files, rows, dD = sv.load(data, (0, 1, 3), None, False)
    Yd, _, abs_preds, PP, rel = pickle.loads((RUN / "predictions.pkl").read_bytes())
    KD, _, _ = pi.knobs(dD.assign(**{f"log_{t}": 0.0 for t in T}))
    # the knob values of the two traces of each pair (trace n + r is row r's reference; v2 references: nominal or the
    # previous packet's settings -> their knobs from the row's delta columns)
    thc = [c for c in dD.columns if c.startswith("theta.")]
    th_own = dD[thc].to_numpy(float)
    th_ref = th_own - dD[["delta." + c[6:] for c in thc]].to_numpy(float)
    TH = np.vstack([th_own, th_ref])
    kn_cols = {k: thc.index("theta." + mem[0]) for k, mem in pi.knobs(dD.assign(**{f"log_{t}": 0.0 for t in T}))[1].items()}
    kn = list(kn_cols)
    Zp = []
    for k in kn:
        x = TH[:, kn_cols[k]]
        x = np.log2(np.clip(x, 1e-9, None)) if pi.knobs(dD.assign(**{f"log_{t}": 0.0 for t in T}))[2][k] else x
        Zp.append(x)
    Zp = np.column_stack(Zp)
    sd = np.nanstd(Zp[:len(dD)], 0)
    sd[sd == 0] = 1
    dZ = (Zp[PP["B"]] - Zp[PP["A"]]) / sd
    img = PP["type"] != 1
    eff = {}
    models = {"measured (images)": None, "settings change GP": "settings change: gp",
              "trace combined trees": "trace: combined trees [A, B, B-A]"}
    for ti, t in enumerate(T):
        eff[t] = {}
        for lab, m in models.items():
            y = PP["Y"][:, ti] if m is None else rel["new"][m][:, ti]
            ok = img & np.isfinite(y) & np.isfinite(dZ).all(1)
            b, *_ = np.linalg.lstsq(dZ[ok], y[ok], rcond=None)
            eff[t][lab] = dict(zip(kn, (100 * b).tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    order = sorted(kn, key=lambda k: -abs(eff["lambda"]["measured (images)"][k]) - abs(eff["sigma_e"]["measured (images)"][k]))
    for ax, t in zip(axes, T[:2]):
        w = 0.27
        for j, (lab, c) in enumerate(zip(models, ["#333333", "#4c72b0", "#55a868"])):
            ax.barh(np.arange(len(order))[::-1] + (1 - j) * w, [eff[t][lab][k] for k in order], w, color=c, label=lab)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(np.arange(len(order))[::-1], order, fontsize=8)
        ax.set_xlabel(f"change of {NICE[t]} per 1 SD change of the setting, %")
        ax.set_title(NICE[t])
        ax.grid(alpha=0.3, axis="x")
    axes[0].legend(fontsize=8)
    fig.suptitle("v2: the effect of each setting on a change, measured and as the best change models predict it "
                 "(all settings fitted together; new amplifier)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "s2_model_effects.png", dpi=130)
    plt.close(fig)
    res["effects_per_setting"] = eff
    print("chapter 2 done", flush=True)

    # ================================================================== 3. v2 results
    A = sc2["absolute"]
    am = ["mean", "settings: gp", "settings + amplifier: gp", "trace: physics", "trace: trees"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, t in zip(axes, T):
        xs = np.arange(len(am))
        kv = [A["trained"].get(m, {}).get(t, {}).get("error", np.nan) for m in am]
        nv = [A["new"].get(m, {}).get(t, {}).get("error", np.nan) for m in am]
        ax.bar(xs - 0.2, kv, 0.4, color="#55a868", label="amplifier seen in training")
        ax.bar(xs + 0.2, nv, 0.4, color="#c44e52", label="new amplifier")
        for i in range(len(am)):
            for v, off in ((kv[i], -0.2), (nv[i], 0.2)):
                if np.isfinite(v):
                    ax.text(i + off, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs, ["average", "settings", "settings +\nknown amp", "trace\nphysics", "trace\ntrees"], fontsize=8)
        ax.set_ylabel("typical error %")
        ax.set_title(NICE[t])
    axes[0].legend(fontsize=8)
    fig.suptitle("v2 absolute models (session D cross-validation): the traces help the noise on a new amplifier; "
                 "nothing predicts a new amplifier's dark charge", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "s3_absolute.png", dpi=130)
    plt.close(fig)
    scE = json.loads((ROOT / "processed_v2" / "lvl1_results" / "sessionE" / "taskB_scorecard" / "scorecard.json").read_text())
    cm_ = ["settings change: gp", "trace change: trees", "trace: combined trees [A, B, B-A]",
           "trace: direction classifier [A, B, B-A]"]
    lab_ = ["settings\nchange", "trace\nchange", "trace\ncombined", "trace\nclassifier"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, t in zip(axes, T):
        xs = np.arange(len(cm_))
        dv = [100 * sc2["relative"]["new"][m]["random pairs"][t]["direction"] for m in cm_]
        ev = [100 * scE["relative"]["new E"][m]["random pairs"][t]["direction"] for m in cm_]
        ax.bar(xs - 0.2, dv, 0.4, color="#4c72b0", label="session D (cross-validation)")
        ax.bar(xs + 0.2, ev, 0.4, color="#dd8452", label="session E (held-out test)")
        for i in range(len(cm_)):
            ax.text(i - 0.2, dv[i], f"{dv[i]:.0f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + 0.2, ev[i], f"{ev[i]:.0f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(90, color="k", ls="--", lw=0.8)
        ax.text(len(cm_) - 0.5, 90.5, "pass mark 90 %", fontsize=7, ha="right")
        ax.set_ylim(50, 102)
        ax.set_xticks(xs, lab_, fontsize=8)
        ax.set_ylabel("direction right on clear changes, %")
        ax.set_title(NICE[t])
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("v2 change models on a new amplifier (random pairs of packets): the traces beat the settings for the "
                 "noise; for λ and gain the settings already get it", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "s3_change.png", dpi=130)
    plt.close(fig)
    print("chapter 3 done", flush=True)

    # ================================================================== 4. delayInt
    from make_packet_plan import TIMING, seconds_per_skip
    th = {k: d2.get(f"theta.{k}", pd.Series(v, index=d2.index)).fillna(v).to_numpy(float) for k, v in TIMING.items()}
    tcyc = np.array([seconds_per_skip({k: th[k][i] for k in TIMING}) for i in range(len(d2))])
    di = th["delayInt"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, t in zip(axes, T):
        for a in (0, 1, 3):
            s = ch2_ == a
            y = d2.loc[s, f"log_{t}"].to_numpy()
            ax.plot(di[s], np.exp(y), "o", ms=3, alpha=0.5, color=COL[a], label=f"ch{a}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("delayInt (cycles)")
        ax.set_ylabel(NICE[t])
        ax.grid(alpha=0.3, which="both")
    # expectations: white noise averages down as 1/sqrt(window) (short delayInt); dark charge grows with readout time
    xs = np.linspace(300, 3000, 50)
    s0 = d2[(di < 400)].groupby("channel")["log_sigma_e"].mean().mean()
    axes[0].plot(xs, np.exp(s0) * (xs / 330) ** -0.5, "k--", lw=1.5, label="white noise only: ∝ 1/√delayInt")
    l0 = d2[(np.abs(di - 1000) < 1)].groupby("channel")["log_lambda"].mean()
    tc_x = np.array([seconds_per_skip({**TIMING, "delayInt": x}) for x in xs])
    for a in (0, 1, 3):
        axes[1].plot(xs, np.exp(l0[a]) * tc_x / seconds_per_skip(TIMING), "--", color=COL[a], lw=1.2)
    axes[1].plot([], [], "k--", label="∝ readout time per pixel")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    axes[0].set_title("noise: falls as expected for white noise, then flattens (slow noise)", fontsize=9)
    axes[1].set_title("dark charge: grows with the readout time, as expected", fontsize=9)
    axes[2].set_title("gain: no dependence (as expected)", fontsize=9)
    fig.suptitle("v2: the integration time does what the physics says", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "s4_physics.png", dpi=130)
    plt.close(fig)
    # shares
    timing_k = [k for k in names if tim2[k]]
    oh2 = np.eye(4)[ch2_][:, [0, 1, 3]]
    tcl = [c for c in trace_cols(d2) if c.startswith("trace.")]
    sh = {}
    for t in T:
        y = d2[f"log_{t}"].to_numpy()
        sh[t] = {"amplifier alone": LSF.cv_r2(oh2, y, pk2)[0],
                 "+ delayInt": LSF.cv_r2(np.column_stack([np.log2(di), oh2]), y, pk2)[0],
                 "+ all timing": LSF.cv_r2(np.column_stack([K2[timing_k].to_numpy(), oh2]), y, pk2)[0],
                 "+ all settings": LSF.cv_r2(np.column_stack([K2.to_numpy(), oh2]), y, pk2)[0],
                 "+ all settings + traces": LSF.cv_r2(np.column_stack([K2.to_numpy(), oh2, d2[tcl].to_numpy(float)]), y, pk2)[0]}
    fig, ax = plt.subplots(figsize=(10, 4.4))
    keys = list(sh["sigma_e"])
    for j, (k, c) in enumerate(zip(keys, ["#999999", "#dd8452", "#c44e52", "#4c72b0", "#55a868"])):
        v = [max(0, sh[t][k]) for t in T]
        ax.bar(np.arange(3) + (j - 2) * 0.16, v, 0.16, color=c, label=k)
        for i, vv in enumerate(v):
            ax.text(i + (j - 2) * 0.16, vv, f"{vv:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(3), [NICE[t] for t in T])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of the spread explained\n(cross-validated by packet)")
    ax.legend(fontsize=8)
    ax.set_title("v2: on top of the amplifier, delayInt alone adds almost everything the settings add to the noise "
                 "and the dark charge; the gain needs VDD and VR", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "s4_shares.png", dpi=130)
    plt.close(fig)
    res["shares"] = sh
    # without delayInt
    exp = json.loads((DIAG / "experiment.json").read_text())
    dl = np.log2(np.nan_to_num(TH[PP["B"], thc.index("theta.delayInt")]) / np.nan_to_num(TH[PP["A"], thc.index("theta.delayInt")]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    edges = np.array([0, 0.1, 0.5, 1.0, 2.0, 4.0])
    for m, c, ls in (("settings change: gp", "#4c72b0", "--"), ("trace: direction classifier [A, B, B-A]", "#55a868", "-")):
        for ti, (t, mk) in enumerate(zip(T[:2], ("o", "s"))):
            q = rel["new"][m][:, ti]
            acc = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                b = img & (np.abs(dl) >= lo) & (np.abs(dl) < hi) & np.isfinite(q) & (np.abs(PP["Y"][:, ti]) > CLEAR[t])
                acc.append(np.mean(np.sign(q[b]) == np.sign(PP["Y"][b, ti])) if b.sum() >= 8 else np.nan)
            axes[0].plot(range(len(acc)), 100 * np.array(acc), ls, marker=mk, color=c if t == "sigma_e" else c,
                         alpha=1 if t == "sigma_e" else 0.5,
                         label=f"{'settings change GP' if 'settings' in m else 'trace classifier'}: {NICE[t]}")
    axes[0].set_xticks(range(len(edges) - 1), ["unchanged", "×1.07-1.4", "×1.4-2", "×2-4", "×4-16"])
    axes[0].set_xlabel("how much delayInt changed between the two packets")
    axes[0].set_ylabel("direction right on clear changes, %")
    axes[0].axhline(50, color="k", lw=0.6)
    axes[0].legend(fontsize=7)
    axes[0].set_title("noise, small or no delayInt change: settings model 53-69 %, trace classifier 73-98 %;\n"
                      "dark charge: both 84-100 % (delayRGH and VSUB still move it)", fontsize=9)
    rows_ = [r for r in exp if r["model"] == "settings Δ trees" and r["train"] == "all pairs"]
    for j, inp in enumerate(("all settings", "without delayInt")):
        v = [next(r["changed_dir"] for r in rows_ if r["inputs"] == inp and r["target"] == t) for t in T]
        axes[1].bar(np.arange(3) + (j - 0.5) * 0.4, v, 0.4, color=["#4c72b0", "#c44e52"][j],
                    label=f"settings model, {inp}")
        for i, vv in enumerate(v):
            axes[1].text(i + (j - 0.5) * 0.4, vv, f"{vv:.0f}", ha="center", va="bottom", fontsize=8)
    axes[1].axhline(50, color="k", ls=":")
    axes[1].set_xticks(range(3), [NICE[t] for t in T])
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("direction right on changes where delayInt moved, %")
    axes[1].legend(fontsize=8)
    axes[1].set_title("a settings model not told delayInt drops to a coin flip on noise and dark charge", fontsize=9)
    fig.suptitle("v2: the integration time carries the change models", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "s4_without.png", dpi=130)
    plt.close(fig)
    print("chapter 4 done", flush=True)

    # ================================================================== 5. the next data (v3 noise set)
    from ccd_limits import Limits
    L = Limits.load(ROOT / "configs" / "limits.toml")
    nom = {**dict(L.nominal), **TIMING}
    p3 = json.loads((ROOT / "lvl1" / "packet_plan_v3n" / "plan.json").read_text())
    G = p3["groups"]
    rows3 = [{**nom, **p["changes"]} for p in p3["packets"] if p["session"] == "G"]
    R3 = pd.DataFrame(rows3)
    knobs3 = {g: (R3[pins[0]] - nom[pins[0]]) for g, pins in G.items()}
    knobs3["delayInt"] = np.log2(R3["delayInt"] / nom["delayInt"])
    knobs3["delayRGH"] = np.log2(R3["delayRGH"] / nom["delayRGH"])
    knobs3["delay_H"] = np.log2(R3["delay_Ha"] / nom["delay_Ha"])
    K3 = pd.DataFrame(knobs3)
    # v2's knobs in the same groups (session D packets)
    pD = d2[d2.session == "D"].drop_duplicates("packet_id")
    nomD = pD[pD.nominal].median(numeric_only=True)          # v2's own nominal (VSUB reads back 45.1, not 45.0)
    k2 = {}
    for g, pins in G.items():
        c = f"theta.{pins[0]}"
        k2[g] = (pD[c].astype(float) - nomD[c]).round(3) if c in pD else pd.Series(0.0, index=pD.index)
    k2["delayInt"] = np.log2(pD["theta.delayInt"].astype(float) / nomD["theta.delayInt"])
    k2["delayRGH"] = np.log2(pD["theta.delayRGH"].astype(float) / nomD["theta.delayRGH"])
    k2["delay_H"] = np.log2(pD["theta.delay_Ha"].astype(float) / nomD["theta.delay_Ha"])
    K2g = pd.DataFrame(k2)
    order5 = ["delayInt", "delayRGH", "delay_H"] + list(G)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, KK, ttl in ((axes[0], K2g, f"v2 session D ({len(K2g)} packets)"),
                        (axes[1], K3, f"v3 noise set, training ({len(K3)} packets)")):
        C = KK[order5].corr().to_numpy()
        C = np.nan_to_num(C)
        im = ax.imshow(np.abs(C), cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks(range(len(order5)), order5, rotation=90, fontsize=7)
        ax.set_yticks(range(len(order5)), order5, fontsize=7)
        var = [KK[k].std() > 1e-9 for k in order5]
        for i, v in enumerate(var):
            if not v:
                ax.axhspan(i - 0.5, i + 0.5, color="#ffe0e0", alpha=0.7)
        ax.set_title(ttl + "\n(pink rows: setting never varied)", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.025, label="|correlation| between two settings over the packets")
    fig.suptitle("Which settings moved, and together: v2 moved timing in every packet and locked clock phases "
                 "together; the v3 noise set keeps timing fixed and moves each voltage on its own", fontsize=10)
    fig.savefig(out / "s5_design_corr.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # coverage
    t2 = (np.abs(K2g[["delayInt", "delayRGH", "delay_H"]]) < 1e-6).all(axis=1)
    t3 = (np.abs(K3[["delayInt", "delayRGH", "delay_H"]]) < 1e-6).all(axis=1)
    groups = list(G)
    n2 = [int(((np.abs(K2g[g]) > 0.05) & t2).sum()) for g in groups]
    n3 = [int(((np.abs(K3[g]) > 0.05) & t3).sum()) for g in groups]
    rg2 = [float(K2g[g].max() - K2g[g].min()) for g in groups]
    rg3 = [float(K3[g].max() - K3[g].min()) for g in groups]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    x = np.arange(len(groups))
    axes[0].bar(x - 0.2, n2, 0.4, color="#999999", label="v2 session D")
    axes[0].bar(x + 0.2, n3, 0.4, color="#55a868", label="v3 noise set (training)")
    axes[0].set_xticks(x, groups, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("packets where this voltage changed\nand the timing did not")
    axes[0].legend(fontsize=8)
    axes[0].set_title("v2: none. v3: every voltage, many times", fontsize=9)
    axes[1].bar(x - 0.2, rg2, 0.4, color="#999999", label="v2")
    axes[1].bar(x + 0.2, rg3, 0.4, color="#55a868", label="v3 noise set")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, groups, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("range covered (V, log scale)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("and over wider ranges (VSUB 35-70 V; clock lows and swings independently)", fontsize=9)
    fig.suptitle("What the next data adds: voltage changes at a fixed integration time", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "s5_coverage.png", dpi=130)
    plt.close(fig)
    res["coverage"] = {"groups": groups, "v2_fixed_timing": n2, "v3_fixed_timing": n3, "range_v2": rg2, "range_v3": rg3}

    # detectable effects: 2 sigma of the fitted slope x the voltage's range, pooled over 3 amplifiers
    def detectable(KK, cols, sigma, span):
        Xd = np.column_stack([KK[c].to_numpy(float) for c in cols])
        keep = Xd.std(0) > 1e-9
        # settings that always moved in lockstep with another cannot be told apart: no estimate for them
        Cc = np.corrcoef(Xd[:, keep].T) if keep.sum() > 1 else np.ones((1, 1))
        kidx = np.flatnonzero(keep)
        locked = np.zeros(len(cols), bool)
        for a_ in range(len(kidx)):
            for b_ in range(len(kidx)):
                if a_ != b_ and abs(Cc[a_, b_]) > 0.999:
                    locked[kidx[a_]] = True
        keep &= ~locked
        Xd = Xd[:, keep]
        n = len(Xd)
        A_ = np.hstack([np.kron(np.eye(3), np.ones((n, 1))), np.kron(np.ones((3, 1)), Xd)])
        Cv = np.linalg.pinv(A_.T @ A_) * sigma ** 2
        se = np.sqrt(np.diag(Cv))[3:]
        outv = np.full(len(cols), np.nan)
        outv[keep] = 200 * se * np.array([span[c] for c, k in zip(cols, keep) if k])
        outv[locked] = -1.0
        return outv
    cols5 = ["delayInt", "delayRGH", "delay_H"] + groups
    span = {g: rg3[groups.index(g)] if g in groups else 1.0 for g in cols5}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, t, sg in ((axes[0], "lambda", 0.035), (axes[1], "sigma_e", 0.015)):
        dv2 = detectable(K2g, cols5, sg, span)[3:]
        dv3 = detectable(K3, cols5, sg, span)[3:]
        ax.bar(x - 0.2, np.where(np.isfinite(dv2) & (dv2 > 0), dv2, 0), 0.4, color="#999999", label="v2 design")
        ax.bar(x + 0.2, np.where(np.isfinite(dv3) & (dv3 > 0), dv3, 0), 0.4, color="#55a868", label="v3 noise set")
        for i, v in enumerate(dv2):
            if not np.isfinite(v) or v < 0:
                ax.text(i - 0.2, 0.45, "never varied" if not np.isfinite(v) else "locked together", ha="center",
                        va="bottom", fontsize=6.5, rotation=90, color="#c44e52")
        for i, v in enumerate(dv3):
            if not np.isfinite(v):
                ax.text(i + 0.2, 0.45, "held at nominal", ha="center", va="bottom", fontsize=6.5, rotation=90, color="#c44e52")
        ax.set_ylim(0.4, None)
        ax.set_xticks(x, groups, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"smallest effect on {NICE[t]} it can see, %\n(over the v3 range of the voltage, 2σ)")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.set_title(f"{NICE[t]} (label noise per image ≈ {100 * sg:.1f} %)", fontsize=10)
        res.setdefault("detectable", {})[t] = {"v2": dict(zip(groups, map(float, np.nan_to_num(dv2, nan=-1)))),
                                               "v3": dict(zip(groups, map(float, dv3)))}
    fig.suptitle("How small a voltage effect each data set can measure (linear fit, all settings together, "
                 "3 amplifiers)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "s5_detectable.png", dpi=130)
    plt.close(fig)
    (out.parent / "story.json").write_text(json.dumps(res, indent=1, default=float))
    print("chapter 5 done")


if __name__ == "__main__":
    main()
