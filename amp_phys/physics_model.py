"""
physics_model.py -- physics equations for noise, gain and dark charge, tested leave-one-amplifier-out (LOAO) on the v3
voltage scan (board 101, sessions G+H): fit on three amplifiers, predict the fourth, against the ML models' new-amplifier
numbers.  Few parameters, each with a physical meaning; "one image" variants use a single nominal image of the new
amplifier (what one would take anyway to commission it).

NOISE  sigma_e = sqrt(sigma_read^2 + kappa * lambda) / g
       sigma_read: the trace's own noise extrapolated to the image's N = 500 skips (Allan deviation at 50 skips scaled
       with the measured Allan slope: white -> -1/2; flat = 1/f floor);  kappa * lambda: the charge part (the WADERs
       noise absorbs part of the dark/CIC charge: it moves with VSUB through lambda, not through the amplifier).
       g: the gain (other amplifiers' mean if no image, else the one-image gain).
       Variants: N0 no fitted parameter, no image; N1 + global scale and kappa (fitted on the 3 others), no image
       (lambda from the others' mean at that setting); N2 = N1 with the new amplifier's one-image lambda and gain.
GAIN   ln g = ln g_nom(amp) + b_VDD (VDD - VDD0) + b_VR (VR - VR0) + c_VR (VR - VR0)^2
       b, c shared (fitted on the others: the source follower's operating point); g_nom: the others' mean (G0) or the
       new amplifier's one nominal image (G1).
DARK   ln lambda = ln lambda_nom(amp) + beta (VSUB - V0) + gamma max(0, VSUB - V_knee) + sum_k b_k (V_k - V_k0)
       slopes shared; lambda_nom: the others' mean (L0) or one nominal image (L1).  The image agent found that each
       channel reads its OWN CCD, so lambda_nom is a property of that silicon: no amplifier trace can know it.

    .venv/bin/python amp_phys/physics_model.py --data processed_v3n --plan lvl1/packet_plan_v3n/plan.json \\
        --out amp_phys/results/v3n
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lvl1"))
import change_test as ct  # noqa: E402

N_SKIPS = 500
KNOBS = ["theta.VDD1", "theta.VR1", "theta.SW_1_L", "theta.RG_1_H", "theta.RG_1_L", "theta.V1_B_L", "theta.H3_B_H"]


def num(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def load(data, plan):
    d = pd.read_csv(data / "dataset.csv", low_memory=False)
    pk = json.loads(Path(plan).read_text())["packets"]
    ses = {p["packet_id"]: str(p["session"]) for p in pk}
    pur = {p["packet_id"]: str(p.get("purpose", "")) for p in pk}
    pid = [ct.planned_id(p, ses) for p in d.packet_id]
    d["session"] = [ses.get(p, "?") if p else "?" for p in pid]
    d["nominal"] = [pur.get(p, "").startswith("nominal") if p else False for p in pid]
    ok = (d.board == 101) & d["label.ok"].astype(str).str.lower().eq("true") & d.session.isin(["G", "H"])
    ok &= num(d["label.lambda_e_per_bin"]) > 0.005
    ok &= ~d["label.fractional_window"].astype(str).str.lower().eq("true")
    d = d[ok].reset_index(drop=True)
    d["date"] = pd.to_datetime(d.date_start)
    d["sig"] = num(d["label.sigma_e"])
    d["lam"] = num(d["label.lambda_e_per_bin"])
    d["g"] = num(d["label.gain_adu_per_e"])
    a50 = num(d["trace.noise: Allan deviation n=50 (ADU)"])
    sl = num(d["trace.noise: Allan slope n=10-100"]).clip(-0.5, 0.0)
    d["read_adu"] = a50 * (N_SKIPS / 50) ** sl                    # Allan deviation extrapolated to 500 skips
    d["read_white_adu"] = num(d["trace.fit: white noise per skip (ADU)"]) / np.sqrt(N_SKIPS)
    return d.dropna(subset=["sig", "lam", "g", "read_adu"]).reset_index(drop=True)


def rms_pct(pred, true):
    e = np.log(pred) - np.log(true)
    e = e[np.isfinite(e)]
    return 100 * np.sqrt(np.mean(e ** 2))


def setting_key(d):
    return d[KNOBS + ["theta.VSUB"]].round(3).astype(str).agg("|".join, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    d = load(Path(args.data), args.plan)
    d["key"] = setting_key(d)
    amps = sorted(d.channel.unique())
    nomV = {k: num(d.loc[d.nominal, k]).median() for k in KNOBS + ["theta.VSUB"]}
    res, preds = [], []
    for a in amps:
        tr, te = d[d.channel != a].copy(), d[d.channel == a].copy()
        # one nominal image of the new amplifier: its FIRST nominal packet
        first_nom = te[te.nominal].sort_values("date").iloc[0]
        others_by_key = tr.groupby("key")[["lam", "g"]].apply(lambda x: np.exp(np.log(x).mean()))
        # ---------- gain
        def gX(df):
            v, r = num(df["theta.VDD1"]) - nomV["theta.VDD1"], num(df["theta.VR1"]) - nomV["theta.VR1"]
            return np.column_stack([v, r, r ** 2])
        amp_off = tr.groupby("channel").g.transform(lambda x: np.log(x[tr.loc[x.index, "nominal"]]).mean())
        bg = np.linalg.lstsq(gX(tr), np.log(tr.g) - amp_off, rcond=None)[0]
        g_others_nom = np.exp(np.log(tr.loc[tr.nominal, "g"]).mean())
        G0 = g_others_nom * np.exp(gX(te) @ bg)
        G1 = first_nom.g * np.exp(gX(te) @ bg)
        # ---------- dark charge
        def lX(df):
            V = num(df["theta.VSUB"])
            cols = [V - nomV["theta.VSUB"], np.maximum(0, V - 56.0)]
            cols += [num(df[k]) - nomV[k] for k in KNOBS]
            return np.column_stack(cols)
        lam_off = tr.groupby("channel").lam.transform(lambda x: np.log(x[tr.loc[x.index, "nominal"]]).mean())
        bl = np.linalg.lstsq(lX(tr), np.log(tr.lam) - lam_off, rcond=None)[0]
        lam_others_nom = np.exp(np.log(tr.loc[tr.nominal, "lam"]).mean())
        L0 = lam_others_nom * np.exp(lX(te) @ bl)
        L1 = first_nom.lam * np.exp(lX(te) @ bl)
        # ---------- noise
        N0 = te.read_adu / g_others_nom
        # N1: sigma^2 g^2 = s^2 read^2 + kappa lambda g^2  (fit s, kappa on the others with their own g, lambda)
        from scipy.optimize import least_squares
        def resid(q, df=tr):
            pred = np.sqrt(q[0] ** 2 * df.read_adu ** 2 + q[1] ** 2 * df.lam * df.g ** 2) / df.g
            return np.log(pred) - np.log(df.sig)
        fit = least_squares(resid, [1.0, 0.05], loss="soft_l1", f_scale=0.1)
        s2, kap = fit.x[0] ** 2, fit.x[1] ** 2
        lam_settings = te.key.map(others_by_key.lam).fillna(pd.Series(L0, index=te.index))
        N1 = np.sqrt(np.maximum(s2 * te.read_adu ** 2 + kap * lam_settings * G0 ** 2, 1e-12)) / G0
        N2 = np.sqrt(np.maximum(s2 * te.read_adu ** 2 + kap * L1 * G1 ** 2, 1e-12)) / G1
        N_oracle = np.sqrt(np.maximum(s2 * te.read_adu ** 2 + kap * te.lam * te.g ** 2, 1e-12)) / te.g
        m = ~te.index.isin([first_nom.name])                                     # score without the calibration image
        r = {"amp": f"ch{a}", "rows": int(m.sum()),
             "noise N0 (trace, no fit, no image)": rms_pct(N0[m], te.sig[m]),
             "noise N1 (trace + 2 global constants, no image)": rms_pct(N1[m], te.sig[m]),
             "noise N2 (N1 + one image)": rms_pct(N2[m], te.sig[m]),
             "noise, true lambda and gain (limit)": rms_pct(N_oracle[m], te.sig[m]),
             "gain G0 (shared slopes, no image)": rms_pct(G0[m], te.g[m]),
             "gain G1 (shared slopes + one image)": rms_pct(G1[m], te.g[m]),
             "lambda L0 (shared slopes, no image)": rms_pct(L0[m], te.lam[m]),
             "lambda L1 (shared slopes + one image)": rms_pct(L1[m], te.lam[m]),
             "fitted scale s": np.sqrt(s2), "fitted kappa": kap,
             "gain slope VDD (%/V)": 100 * bg[0], "gain slope VR (%/V)": 100 * bg[1],
             "lambda slope VSUB below / above 56 V (%/V)": f"{100 * bl[0]:.1f} / {100 * (bl[0] + bl[1]):.1f}"}
        res.append(r)
        preds.append(te.assign(N1=N1, N2=N2, G1=G1, L1=L1, L0=L0, N0=N0))
    R = pd.DataFrame(res).set_index("amp")
    R.to_csv(out / "physics_model_loao.csv")
    num_cols = [c for c in R.columns if R[c].dtype.kind == "f" and not c.startswith("fitted")]
    mean = R[num_cols].mean()
    L = ["# Physics equations, leave one amplifier out (board 101, v3 sessions G+H)", "",
         "Error = RMS of log(predicted/true), %, on the held-out amplifier (its calibration image left out).", "",
         R.T.to_markdown(floatfmt=".3g"), "", "## Mean over the four held-out amplifiers vs the ML models (new amplifier)", "",
         "| target | physics, no image | physics, one image | best ML, new amplifier (session G) |", "|---|---|---|---|",
         f"| noise sigma_e | {mean['noise N1 (trace + 2 global constants, no image)']:.1f} (N0 {mean['noise N0 (trace, no fit, no image)']:.1f}) | "
         f"{mean['noise N2 (N1 + one image)']:.1f} | 11.9 (trace trees) |",
         f"| gain | {mean['gain G0 (shared slopes, no image)']:.1f} | {mean['gain G1 (shared slopes + one image)']:.1f} | 3.7 (settings trees) |",
         f"| dark charge lambda | {mean['lambda L0 (shared slopes, no image)']:.1f} | {mean['lambda L1 (shared slopes + one image)']:.1f} | 39.9 (trace trees) |"]
    (out / "physics_model.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    P = pd.concat(preds)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    cols = {0: "#2a78d6", 1: "#eb6834", 2: "#8a8984", 3: "#1baf7a"}
    for ax, (t, p, lab) in zip(axs, [("sig", "N2", "noise σe (e⁻)"), ("g", "G1", "gain (ADU/e⁻)"), ("lam", "L1", "λ (e⁻/pix)")]):
        for a, g_ in P.groupby("channel"):
            ax.scatter(g_[t], g_[p], s=10, color=cols[a], label=f"ch{a}", alpha=0.8)
        lim = [min(P[t].min(), P[p].min()), max(P[t].max(), P[p].max())]
        ax.plot(lim, lim, color="#0b0b0b", lw=0.8)
        if t == "lam":
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(f"measured {lab}")
        ax.set_ylabel("physics prediction (amp left out, one image)")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Physics equations, each amplifier predicted from the other three plus one nominal image of its own",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "physics_model.png", dpi=150)


if __name__ == "__main__":
    main()
