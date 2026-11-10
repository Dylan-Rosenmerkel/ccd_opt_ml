"""
lambda_settings_figures.py -- why the dark-current (lambda) change models do so well from the settings alone, which
settings drive them, whether that would survive many more amplifiers, and why the traces add little for lambda (with
sigma_e and gain side by side for contrast).

Data: board 101, all four amplifiers, sessions D and E, label.ok, no fractional-window rows (biased image labels).
Settings are grouped into the independent knobs that were actually varied (columns that always move together, e.g.
delay_Ha/Hb/Hc or RG_1_H/RG_2_H, are one knob).

Figures (in --out):
 1 knob_importance.png      which knobs a settings-only change model uses: permutation importance (drop in R^2 of
                            d log y when the knob's change is shuffled), cross-validated by packet, per target
 2 lambda_vs_knobs.png      log lambda against the top knobs, per amplifier, with each amplifier's own straight-line
                            fit: same slope, different offset = a change model transfers, an absolute model does not
 3 slopes_per_amp.png       the per-amplifier slope of every top knob (multiple regression, 68 % intervals): the
                            amplifier-to-amplifier spread of the slopes is what would break a change model on many
                            amplifiers
 4 variance_split.png       cross-validated R^2 of log y from: amplifier only, settings only, settings + amplifier
                            offset, settings x amplifier (separate model per amplifier), settings + amplifier + traces
 5 traces_vs_residual.png   what the traces add on top of the settings and the amplifier: R^2 of the residual
                            (log y - settings+amplifier model) from the traces, per target
 6 lambda_D_vs_E.png        lambda changes in D and in E and the knob ranges of each session (does E extrapolate?)
 7 amp_transfer.png         per knob, the typical change it causes (shared slope x typical knob change) against the
                            part that differs between amplifiers (spread of the slopes x typical knob change): the
                            second is the error a change model makes on an amplifier it has never seen
 8 lambda_residuals.png     measured against settings + amplifier predicted lambda per packet (D, E), outliers named

    .venv/bin/python lvl1/lambda_settings_figures.py --data processed_v2 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lvl1"))
import change_test as ct  # noqa: E402

LAB = {"sigma_e": "label.sigma_e", "lambda": "label.lambda_e_per_bin", "gain": "label.gain_adu_per_e"}
PLAN = ROOT / "lvl1" / "packet_plan_v2" / "plan.json"
FP = ROOT / "processed_v2_trainD" / "feature_study" / "amp_fingerprint_eta90.txt"
COL = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
warnings.filterwarnings("ignore")


def load(data: Path, plans=(PLAN,), sessions=("D", "E")):
    d = pd.read_csv(data / "dataset.csv", low_memory=False)
    plan = [p for pl in plans for p in json.loads(Path(pl).read_text())["packets"]]
    session = {p["packet_id"]: str(p["session"]) for p in plan}
    d["session"] = [session.get(ct.planned_id(p, session), "?") for p in d.packet_id]
    ok = (d.board == 101) & d.session.isin(list(sessions)) & d["label.ok"].astype(str).str.lower().eq("true")
    ok &= ~d["label.fractional_window"].astype(str).str.lower().eq("true")
    d = d[ok].reset_index(drop=True)
    for t, c in LAB.items():
        d[f"log_{t}"] = np.log(d[c].astype(float))
    d = d[np.isfinite(d[[f"log_{t}" for t in LAB]]).all(1)].reset_index(drop=True)
    return d


def knobs(d):
    """The varied settings, columns that always move together merged into one knob (named after its members)."""
    th = [c for c in d.columns if c.startswith("theta.") and d[c].nunique() > 1]
    groups = []
    for c in th:
        for g in groups:
            if np.allclose(d[c].to_numpy(float), d[g[0]].to_numpy(float), equal_nan=True):
                g.append(c)
                break
        else:
            groups.append([c])
    names = []
    for g in groups:
        short = [c[len("theta."):] for c in g]
        names.append(short[0] if len(short) == 1 else f"{short[0]} (+{len(short) - 1})")
    K = pd.DataFrame({n: d[g[0]].to_numpy(float) for n, g in zip(names, groups)})
    return K, dict(zip(names, [[c[len("theta."):] for c in g] for g in groups]))


def gbt():
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    return H(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8, random_state=0)


def cv_r2(X, y, groups, make=gbt, folds=5):
    """Out-of-fold predictions grouped by packet, and their R^2."""
    from sklearn.model_selection import GroupKFold
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(X, y, groups):
        p[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return 1 - np.mean((y - p) ** 2) / np.var(y), p


def pairs(d, K):
    """All pairs of packets within one amplifier and one session: d knob, d log y (drift is small: <0.2 %/h)."""
    rows = []
    for (ch, s), g in d.groupby(["channel", "session"]):
        idx = g.index.to_numpy()
        a, b = np.triu_indices(len(idx), 1)
        rows.append(pd.DataFrame({"a": idx[a], "b": idx[b], "channel": ch, "session": s}))
    P = pd.concat(rows, ignore_index=True)
    dK = K.to_numpy()[P.b] - K.to_numpy()[P.a]
    dY = {t: d[f"log_{t}"].to_numpy()[P.b] - d[f"log_{t}"].to_numpy()[P.a] for t in LAB}
    return P, dK, dY


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(ROOT / "processed_v2"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pairs", type=int, default=6000, help="pairs subsampled for the importance fits")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance

    d = load(Path(args.data))
    K, members = knobs(d)
    knames = list(K.columns)
    pk = d.packet_id.to_numpy()
    res = {"n_rows": len(d), "n_packets": int(d.packet_id.nunique()), "knobs": members}
    print(f"{len(d)} rows, {d.packet_id.nunique()} packets, {len(knames)} knobs: {knames}")

    # ---------------------------------------------------------------- 1: knob importance for the change models
    P, dK, dY = pairs(d, K)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(P), min(args.n_pairs, len(P)), replace=False)
    pa = pk[P.a.to_numpy()][sub]
    imp = {}
    from sklearn.model_selection import GroupKFold
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    order = None
    for ax, t in zip(axes, LAB):
        X, y = np.vstack([dK[sub], -dK[sub]]), np.concatenate([dY[t][sub], -dY[t][sub]])
        g = np.concatenate([pa, pa])
        vals = np.zeros((5, len(knames)))
        r2s = []
        for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
            m = gbt().fit(X[tr], y[tr])
            r2s.append(m.score(X[te], y[te]))
            vals[k] = permutation_importance(m, X[te], y[te], n_repeats=5, random_state=0).importances_mean
        v = vals.mean(0)
        imp[t] = {"r2_cv": float(np.mean(r2s)), "importance": dict(zip(knames, v.tolist()))}
        if order is None:
            order = np.argsort(v)                       # sort every panel by the lambda ranking (first target: sigma_e)
        ax.barh(np.arange(len(knames)), v[order], xerr=vals.std(0)[order], color="#4c72b0")
        ax.set_title(f"{t}: settings-only change model\nCV R$^2$ of d log {t} = {np.mean(r2s):.2f}")
        ax.set_xlabel("permutation importance (drop in R$^2$)")
        ax.grid(alpha=0.3, axis="x")
    axes[0].set_yticks(np.arange(len(knames)), [knames[i] for i in order], fontsize=8)
    # re-sort by lambda for readability
    fig.suptitle("Which settings drive the change of each label (all within-amplifier packet pairs, D+E, board 101)")
    fig.tight_layout()
    fig.savefig(out / "knob_importance.png", dpi=140)
    plt.close(fig)
    res["change_importance"] = imp
    lam_rank = sorted(knames, key=lambda k: -imp["lambda"]["importance"][k])
    top = lam_rank[:4]
    print("lambda change: CV R2", round(imp["lambda"]["r2_cv"], 3), "top knobs", top)

    # ---------------------------------------------------------------- 2: log lambda vs the top knobs, per amplifier
    fig, axes = plt.subplots(2, len(top), figsize=(4.2 * len(top), 7.5))
    Kn = (K - K.mean()) / K.std()
    for j, kn in enumerate(top):
        for ch, g in d.groupby("channel"):
            x = K.loc[g.index, kn].to_numpy()
            # partial residual: log lambda minus the per-amp linear effect of the OTHER top knobs
            others = [o for o in top if o != kn]
            Xo = np.column_stack([np.ones(len(g)), Kn.loc[g.index, others].to_numpy(), Kn.loc[g.index, kn].to_numpy()])
            beta = np.linalg.lstsq(Xo, g["log_lambda"].to_numpy(), rcond=None)[0]
            partial = g["log_lambda"].to_numpy() - Xo[:, 1:-1] @ beta[1:-1]
            for row, (yv, lab) in enumerate(((g["log_lambda"].to_numpy(), "log lambda (raw)"),
                                             (partial, "log lambda, other top knobs removed"))):
                ax = axes[row, j]
                ax.plot(x, yv, "o", ms=3, alpha=0.6, color=COL[ch], label=f"ch{ch}")
                xs = np.linspace(x.min(), x.max(), 20)
                c = np.polyfit(x, yv, 1)
                ax.plot(xs, np.polyval(c, xs), "-", color=COL[ch], lw=1.5)
                ax.set_xlabel(kn)
                ax.set_ylabel(lab)
                ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("log lambda against the knobs that drive its change, per amplifier: parallel lines = the same "
                 "effect, shifted (an offset cancels in a change)")
    fig.tight_layout()
    fig.savefig(out / "lambda_vs_knobs.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- 3: per-amplifier slopes (multiple regression)
    tops = {t: sorted(knames, key=lambda k: -imp[t]["importance"][k])[:6] for t in LAB}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    res["slopes"] = {}
    for ax, t in zip(axes, LAB):
        ks = tops[t]
        res["slopes"][t] = {}
        for c_i, (ch, g) in enumerate(d.groupby("channel")):
            X = np.column_stack([np.ones(len(g)), Kn.loc[g.index, ks].to_numpy()])
            y = g[f"log_{t}"].to_numpy()
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
            cov = np.linalg.pinv(X.T @ X) * (r @ r) / max(len(y) - X.shape[1], 1)
            se = np.sqrt(np.diag(cov))
            res["slopes"][t][f"ch{ch}"] = {k: [float(beta[i + 1]), float(se[i + 1])] for i, k in enumerate(ks)}
            ax.errorbar(np.arange(len(ks)) + (c_i - 1.5) * 0.15, 100 * beta[1:], yerr=100 * se[1:], fmt="o",
                        color=COL[ch], label=f"ch{ch}", capsize=2)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(np.arange(len(ks)), ks, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(f"% change of {t} per 1 SD of the knob")
        ax.set_title(f"{t}: slope of each top knob, per amplifier")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "slopes_per_amp.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- 4 + 5: variance split and what the traces add
    excl = set(Path(FP).read_text().split()) if FP.exists() else set()
    tcols = [c for c in d.columns if c.startswith("trace.") and c[len("trace."):] not in excl
             and pd.api.types.is_numeric_dtype(d[c]) and d[c].notna().mean() > 0.95 and d[c].nunique() > 5]
    TRc = d[tcols].to_numpy(float)
    pos = np.nanmin(TRc, 0) > 0
    TRc[:, pos] = np.log(TRc[:, pos])
    onehot = np.eye(4)[d.channel.to_numpy().astype(int)]
    Kx = K.to_numpy()
    variants = {"amplifier only": onehot, "settings only": Kx, "settings + amplifier": np.hstack([Kx, onehot]),
                "traces only": TRc, "settings + amplifier + traces": np.hstack([Kx, onehot, TRc])}
    res["r2"], res["resid"] = {}, {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    w = 0.15
    for i, t in enumerate(LAB):
        y = d[f"log_{t}"].to_numpy()
        r2 = {k: cv_r2(X, y, pk)[0] for k, X in variants.items()}
        # settings x amplifier: a separate settings model per amplifier (out-of-fold, by packet)
        p = np.full(len(y), np.nan)
        for ch in range(4):
            m = d.channel.to_numpy() == ch
            p[m] = cv_r2(Kx[m], y[m], pk[m])[1]
        r2["settings x amplifier (per-amp model)"] = 1 - np.mean((y - p) ** 2) / np.var(y)
        res["r2"][t] = r2
        for j, (k, v) in enumerate(r2.items()):
            axes[0].bar(i + (j - 2.5) * w, v, w, label=k if i == 0 else None, color=plt.cm.tab10(j))
        # residual after settings + amplifier, explained by the traces?
        _, p_sa = cv_r2(np.hstack([Kx, onehot]), y, pk)
        resid = y - p_sa
        r2_res, _ = cv_r2(TRc, resid, pk)
        noise = {"sigma_e": 0.012, "lambda": 0.025, "gain": 0.003}[t]     # task-D repeat scatter (nominal, detrended)
        res["resid"][t] = {"resid_rms_pct": float(100 * np.std(resid)), "r2_from_traces": float(r2_res),
                           "label_noise_pct": 100 * noise}
        axes[1].bar(i - 0.2, 100 * np.std(resid), 0.4, color="#999999", label="residual after settings + amplifier" if i == 0 else None)
        axes[1].bar(i + 0.2, 100 * np.std(resid) * np.sqrt(max(1 - r2_res, 0)), 0.4, color="#4c72b0",
                    label="... after the traces too (CV)" if i == 0 else None)
        axes[1].hlines(100 * noise, i - 0.4, i + 0.4, colors="k", linestyles=":", label="label repeat scatter" if i == 0 else None)
    axes[0].set_xticks(range(3), list(LAB))
    axes[0].set_ylabel("cross-validated R$^2$ of log y (grouped by packet)")
    axes[0].set_ylim(min(0, axes[0].get_ylim()[0]), 1)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].legend(fontsize=7, loc="lower left")
    axes[0].set_title("How much of each label the settings, the amplifier and the traces explain")
    axes[1].set_xticks(range(3), list(LAB))
    axes[1].set_ylabel("RMS of log y, %")
    axes[1].set_title("What is left after settings + amplifier, and what the traces remove")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "variance_split.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- 6: D vs E
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for s, c in (("D", "#4c72b0"), ("E", "#dd8452")):
        m = (P.session == s).to_numpy()
        axes[0].hist(100 * dY["lambda"][m], bins=60, alpha=0.6, density=True, color=c,
                     label=f"session {s}: RMS {100 * np.std(dY['lambda'][m]):.0f} %")
    axes[0].set_xlabel("d log lambda between two packets of one amplifier, %")
    axes[0].set_title("Size of the lambda changes, D vs E")
    axes[0].legend()
    ks = lam_rank[:8]
    for j, kn in enumerate(ks):
        for s, c, off in (("D", "#4c72b0", -0.15), ("E", "#dd8452", 0.15)):
            v = Kn.loc[d.session == s, kn]
            axes[1].plot([j + off] * 2, [v.min(), v.max()], "-", color=c, lw=4, label=f"session {s}" if j == 0 else None)
            axes[1].plot(np.full(len(v), j + off), v, "_", color="k", ms=6, alpha=0.4)
    axes[1].set_xticks(range(len(ks)), ks, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("knob value (standardised)")
    axes[1].set_title("Range of the top lambda knobs in D and E")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "lambda_D_vs_E.png", dpi=140)
    plt.close(fig)
    res["lambda_change_rms_pct"] = {s: float(100 * np.std(dY["lambda"][(P.session == s).to_numpy()])) for s in ("D", "E")}
    res["knob_range_E_outside_D"] = {kn: bool((K.loc[d.session == "E", kn].max() > K.loc[d.session == "D", kn].max() + 1e-9)
                                               or (K.loc[d.session == "E", kn].min() < K.loc[d.session == "D", kn].min() - 1e-9))
                                      for kn in knames}
    # ---------------------------------------------------------------- 7: amplifier-to-amplifier transfer, per knob
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    res["transfer"] = {}
    rng2 = np.random.default_rng(1)
    for ax, t in zip(axes, LAB):
        ks = tops[t]
        sl = np.array([[res["slopes"][t][f"ch{ch}"][k][0] for k in ks] for ch in range(4)])      # amps x knobs
        se = np.array([[res["slopes"][t][f"ch{ch}"][k][1] for k in ks] for ch in range(4)])
        dk = np.abs(np.subtract(*rng2.choice(Kn[ks].to_numpy(), 2, axis=0)) if False else
                    np.median(np.abs(Kn[ks].to_numpy()[:, None, :] - Kn[ks].to_numpy()[None, :, :]).reshape(-1, len(ks)), 0))
        shared = 100 * np.abs(sl.mean(0)) * dk
        # between-amp spread of the slopes, minus what their own errors explain (0 if within the errors)
        spread = np.sqrt(np.maximum(sl.var(0, ddof=1) - (se ** 2).mean(0), 0))
        amp_part = 100 * spread * dk
        res["transfer"][t] = {k: {"shared_pct": float(a), "amp_specific_pct": float(b)} for k, a, b in zip(ks, shared, amp_part)}
        x = np.arange(len(ks))
        ax.bar(x - 0.2, shared, 0.4, color="#4c72b0", label="shared effect (all amps agree)")
        ax.bar(x + 0.2, amp_part, 0.4, color="#c44e52", label="amplifier-specific part (beyond the slope errors)")
        tot = np.sqrt((amp_part ** 2).sum())
        ax.set_title(f"{t}: typical change per knob\namp-specific error on a new amp ~ {tot:.1f} % (all top knobs)")
        ax.set_xticks(x, ks, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("% change of the label for a typical knob change")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Would a settings-based change model survive many amplifiers? Blue = what transfers, red = what a new "
                 "amplifier would get wrong (estimated from the 4 amplifiers of board 101)")
    fig.tight_layout()
    fig.savefig(out / "amp_transfer.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- 8: lambda per packet, settings + amp model
    y = d["log_lambda"].to_numpy()
    _, p_sa = cv_r2(np.hstack([Kx, onehot]), y, pk)
    d["_res"] = 100 * (y - p_sa)
    d["_pred"] = p_sa
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for s, c in (("D", "#4c72b0"), ("E", "#dd8452")):
        m = d.session == s
        axes[0].plot(np.exp(d.loc[m, "_pred"]), np.exp(y[m]), "o", ms=4, color=c, alpha=0.7, label=f"session {s}")
    lim = [np.exp(y).min() * 0.8, np.exp(y).max() * 1.2]
    axes[0].plot(lim, lim, "k:", lw=1)
    axes[0].set_xscale("log"), axes[0].set_yscale("log")
    axes[0].set_xlabel("lambda predicted from settings + amplifier (CV by packet)")
    axes[0].set_ylabel("lambda measured (image)")
    pr = d.groupby(["packet_id", "session"])["_res"].mean().reset_index()
    pr = pr.join(K.groupby(d.packet_id).first(), on="packet_id").sort_values("_res")
    for _, r in pr[np.abs(pr._res) > 20].iterrows():
        sel = d.packet_id == r.packet_id
        axes[0].annotate(r.packet_id, (np.exp(d.loc[sel, "_pred"]).max(), np.exp(y[sel]).max()), fontsize=8)
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")
    axes[0].set_title("lambda per row: settings + amplifier model")
    xs = Kn["delayRGH"] + Kn["VSUB"]
    for s, c in (("D", "#4c72b0"), ("E", "#dd8452")):
        m = d.session == s
        axes[1].scatter(xs[m], d.loc[m, "_res"], s=12 + 30 * Kn.loc[m, "delayInt"].clip(-1, 2).add(1), color=c, alpha=0.6,
                        label=f"session {s}")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("delayRGH + VSUB (standardised; both high = the untested corner)")
    axes[1].set_ylabel("lambda residual, %  (marker size: delayInt)")
    axes[1].set_title("Where the settings model misses lambda")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "lambda_residuals.png", dpi=140)
    plt.close(fig)
    res["lambda_resid_by_packet"] = {r.packet_id: float(r._res) for _, r in pr.iterrows()}
    res["lambda_loglog_slope_delayInt"] = {f"ch{ch}": float(np.polyfit(np.log(K.loc[g.index, "delayInt"]), g.log_lambda, 1)[0])
                                           for ch, g in d.groupby("channel")}

    (out / "lambda_figures.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("transfer", "lambda_loglog_slope_delayInt", "lambda_change_rms_pct")}, indent=1, default=float))
    print("lambda importance:", {k: round(v, 3) for k, v in sorted(imp["lambda"]["importance"].items(), key=lambda kv: -kv[1])[:8]})
    print("sigma_e importance:", {k: round(v, 3) for k, v in sorted(imp["sigma_e"]["importance"].items(), key=lambda kv: -kv[1])[:8]})
    print("gain importance:", {k: round(v, 3) for k, v in sorted(imp["gain"]["importance"].items(), key=lambda kv: -kv[1])[:8]})


if __name__ == "__main__":
    main()
