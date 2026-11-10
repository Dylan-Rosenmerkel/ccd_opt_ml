"""
voltage_dark_current.py -- the dark current (lambda) against the VOLTAGES, with the timing removed.

lambda changes mostly with the readout time (delayInt, delayRGH, delay_H*: lambda ~ delayInt^0.5 on every amplifier),
which hides the smaller voltage effects.  Here the timing is removed first (per amplifier: log lambda against log
delayInt, delayRGH and delay_Ha, a straight line each, plus the amplifier's offset), and what is left (the
"timing-corrected" log lambda) is studied against the voltages only:

 1 voltage_slopes.png     per voltage knob, the slope of the timing-corrected log lambda per amplifier (multiple
                          regression on all voltages at once; % per 1 SD of the knob and per volt, 68 % intervals)
 2 voltage_importance.png how much of the timing-corrected lambda the voltages explain (CV R^2 by packet, linear and
                          trees) against the label noise, and which voltages a model uses (permutation importance)
 3 voltage_partial.png    timing-corrected log lambda against each of the top voltages, per amplifier
 4 voltage_design.png     how the voltages were sampled: correlation with the timing knobs and among themselves, and
                          the number of packets that changed ONLY voltages (timing fixed)

    .venv/bin/python lvl1/voltage_dark_current.py --out <dir>
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
import lambda_settings_figures as L  # noqa: E402

TIMING = ("delayInt", "delayRGH", "delay_Ha (+2)")
COL = L.COL
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(ROOT / "processed_v2"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop", nargs="*", default=["L2_72"], help="packets left out (the untested corner)")
    ap.add_argument("--plan", nargs="+", default=[str(L.PLAN)])
    ap.add_argument("--sessions", nargs="+", default=["D", "E"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import RidgeCV

    d = L.load(Path(args.data), args.plan, args.sessions)
    d = d[~d.packet_id.isin(args.drop)].reset_index(drop=True)
    K, members = L.knobs(d)
    volts = [k for k in K.columns if k not in TIMING]
    pk = d.packet_id.to_numpy()
    ch = d.channel.to_numpy().astype(int)
    res = {"n_rows": len(d), "n_packets": int(d.packet_id.nunique()), "voltages": volts, "dropped": args.drop}

    # ---- remove the timing: per amplifier, log lambda ~ a + b log delayInt + c delayRGH + e delay_Ha
    Tm = np.column_stack([np.log(K["delayInt"]), K["delayRGH"] / 1000, K["delay_Ha (+2)"] / 1000])
    y = d["log_lambda"].to_numpy()
    Vs = (K[volts] - K[volts].mean()) / K[volts].std()
    corr = np.full(len(y), np.nan)
    res["timing_fit"] = {}
    for a in range(4):
        m = ch == a
        # timing AND voltages fitted together (so the timing slope is not biased by correlated voltages); then only
        # the timing part is subtracted
        X = np.column_stack([np.ones(m.sum()), Tm[m], Vs[m].to_numpy()])
        beta = np.linalg.lstsq(X, y[m], rcond=None)[0]
        corr[m] = y[m] - X[:, 1:4] @ beta[1:4] - beta[0]
        res["timing_fit"][f"ch{a}"] = {"d log lambda / d log delayInt": float(beta[1]),
                                       "per 1000 delayRGH": float(beta[2]), "per 1000 delay_Ha": float(beta[3])}
    d["corr"] = corr
    res["rms_raw_pct"] = float(100 * np.std(y - pd.Series(y).groupby(ch).transform("mean")))
    res["rms_timing_corrected_pct"] = float(100 * np.std(corr))

    # ---- 1: voltage slopes per amplifier (multiple regression on all voltages)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    res["slopes"] = {}
    for a in range(4):
        m = ch == a
        X = np.column_stack([np.ones(m.sum()), Vs[m].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, corr[m], rcond=None)
        r = corr[m] - X @ beta
        cov = np.linalg.pinv(X.T @ X) * (r @ r) / max(m.sum() - X.shape[1], 1)
        se = np.sqrt(np.diag(cov))[1:]
        res["slopes"][f"ch{a}"] = {v: {"pct_per_sd": 100 * float(b), "se": 100 * float(s),
                                       "pct_per_volt": 100 * float(b / K[v].std())} for v, b, s in zip(volts, beta[1:], se)}
        ax.errorbar(np.arange(len(volts)) + (a - 1.5) * 0.15, 100 * beta[1:], yerr=100 * se, fmt="o", color=COL[a],
                    capsize=2, label=f"ch{a}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(volts)), [f"{v}\n(1 SD = {K[v].std():.2f} V)" for v in volts], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("% change of lambda per 1 SD (timing removed)")
    ax.set_title("Dark current against the voltages, timing removed: slope per amplifier (all voltages fitted together)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "voltage_slopes.png", dpi=140)
    plt.close(fig)

    # ---- 2: how much the voltages explain
    Xv = np.hstack([Vs.to_numpy(), np.eye(4)[ch]])
    ridge = lambda: RidgeCV(alphas=np.logspace(-2, 3, 20))
    r2_lin, _ = L.cv_r2(Xv, corr, pk, make=ridge)
    r2_gbt, _ = L.cv_r2(Xv, corr, pk)
    # per-amplifier linear models (amplifier-specific voltage slopes)
    p = np.full(len(corr), np.nan)
    for a in range(4):
        m = ch == a
        p[m] = L.cv_r2(Vs[m].to_numpy(), corr[m], pk[m], make=ridge)[1]
    r2_amp = 1 - np.mean((corr - p) ** 2) / np.var(corr)
    res["r2_timing_corrected"] = {"voltages, linear (shared slopes)": float(r2_lin), "voltages, trees": float(r2_gbt),
                                  "voltages, linear per amplifier": float(r2_amp)}
    noise = 0.025
    res["label_noise_pct"] = 100 * noise
    res["explainable_fraction"] = float(max(0, 1 - noise ** 2 / np.var(corr)))
    from sklearn.model_selection import GroupKFold
    vals = []
    for tr, te in GroupKFold(5).split(Xv, corr, pk):
        m = ridge().fit(Xv[tr], corr[tr])
        vals.append(permutation_importance(m, Xv[te], corr[te], n_repeats=10, random_state=0).importances_mean[:len(volts)])
    vi = np.mean(vals, 0)
    res["voltage_importance"] = dict(zip(volts, vi.tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    names = list(res["r2_timing_corrected"]) + ["ceiling (label noise)"]
    vals_ = list(res["r2_timing_corrected"].values()) + [res["explainable_fraction"]]
    axes[0].bar(range(len(names)), vals_, color=["#4c72b0", "#55a868", "#8172b2", "#999999"])
    axes[0].set_xticks(range(len(names)), names, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("CV R$^2$ of timing-corrected log lambda")
    axes[0].set_title(f"Timing-corrected lambda: RMS {res['rms_timing_corrected_pct']:.1f} % "
                      f"(raw {res['rms_raw_pct']:.0f} %), label noise {100 * noise:.1f} %")
    axes[0].grid(alpha=0.3, axis="y")
    o = np.argsort(vi)
    axes[1].barh(range(len(volts)), vi[o], color="#4c72b0")
    axes[1].set_yticks(range(len(volts)), [volts[i] for i in o], fontsize=8)
    axes[1].set_xlabel("permutation importance (drop in R$^2$), linear model")
    axes[1].set_title("Which voltages move the dark current (timing removed)")
    axes[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out / "voltage_importance.png", dpi=140)
    plt.close(fig)

    # ---- 3: partial plots of the top voltages
    top = [volts[i] for i in np.argsort(-vi)[:4]]
    fig, axes = plt.subplots(1, len(top), figsize=(4.3 * len(top), 4))
    for ax, v in zip(axes, top):
        for a in range(4):
            m = ch == a
            others = [w for w in volts if w != v]
            X = np.column_stack([np.ones(m.sum()), Vs.loc[m, others].to_numpy()])
            b = np.linalg.lstsq(X, corr[m], rcond=None)[0]
            part = corr[m] - X[:, 1:] @ b[1:]
            x = K.loc[m, v].to_numpy()
            ax.plot(x, 100 * (part - part.mean()), "o", ms=3, alpha=0.6, color=COL[a], label=f"ch{a}")
            c = np.polyfit(x, 100 * (part - part.mean()), 1)
            xs = np.linspace(x.min(), x.max(), 10)
            ax.plot(xs, np.polyval(c, xs), "-", color=COL[a])
        ax.set_xlabel(f"{v} (V)")
        ax.set_ylabel("lambda, timing + other voltages removed, %")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Dark current against the top voltages (timing and the other voltages removed), per amplifier")
    fig.tight_layout()
    fig.savefig(out / "voltage_partial.png", dpi=140)
    plt.close(fig)

    # ---- 4: the design
    pkt = K.groupby(d.packet_id).first()
    pv = (pkt[volts] - pkt[volts].mean()) / pkt[volts].std()
    pt = (pkt[list(TIMING)] - pkt[list(TIMING)].mean()) / pkt[list(TIMING)].std()
    C = np.corrcoef(np.hstack([pt.to_numpy(), pv.to_numpy()]).T)
    nom = pkt.median()
    t_fixed = (np.abs(pkt[list(TIMING)] - nom[list(TIMING)]) < 50).all(axis=1)
    v_moved = (np.abs(pkt[volts] - nom[volts]) > 0.5 * pkt[volts].std()).any(axis=1)
    res["design"] = {"packets": int(len(pkt)), "timing at nominal": int(t_fixed.sum()),
                     "timing at nominal AND some voltage moved": int((t_fixed & v_moved).sum()),
                     "max |corr| voltage-voltage": float(np.max(np.abs(C[3:, 3:] - np.eye(len(volts))))),
                     "max |corr| voltage-timing": float(np.max(np.abs(C[:3, 3:])))}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    im = axes[0].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    labs = list(TIMING) + volts
    axes[0].set_xticks(range(len(labs)), labs, rotation=90, fontsize=7)
    axes[0].set_yticks(range(len(labs)), labs, fontsize=7)
    axes[0].set_title("Correlation of the knobs over the packets (D+E)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    ndev = (np.abs(pv) > 0.5).sum(1)
    axes[1].hist([ndev[t_fixed], ndev[~t_fixed]], bins=np.arange(len(volts) + 2) - 0.5, stacked=True,
                 label=["timing at nominal", "timing moved"], color=["#55a868", "#4c72b0"])
    axes[1].set_xlabel("number of voltages more than 0.5 SD from their mean in the packet")
    axes[1].set_ylabel("packets")
    axes[1].set_title(f"{int(t_fixed.sum())} of {len(pkt)} packets have the timing at nominal")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "voltage_design.png", dpi=140)
    plt.close(fig)
    (out / "voltage_dark_current.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("timing_fit", "rms_raw_pct", "rms_timing_corrected_pct", "r2_timing_corrected",
                                          "explainable_fraction", "design")}, indent=1))
    print("importance:", {k: round(v, 3) for k, v in sorted(res["voltage_importance"].items(), key=lambda kv: -kv[1])})
    print("slopes % per SD:", {v: [round(res["slopes"][f"ch{a}"][v]["pct_per_sd"], 1) for a in range(4)] for v in volts})


if __name__ == "__main__":
    main()
