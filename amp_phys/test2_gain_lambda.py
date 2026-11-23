"""
test2_gain_lambda.py -- test 2: physics models for the GAIN and the DARK CHARGE lambda, like test 1 for sigma_e.
Train on session G, score once on sealed session H (board 101 ch0/1/3, valid rows); cells "new amplifier" (left out of
training) and "trained".  Scores: typical error (RMS log, %), change rho over all pairs of H rows, change size error.

GAIN (ADU/e = A G q / C_sn); the trace's deterministic signals are known charges times the gain:
  physics, no ML (reset)    log g = log|reset jump| - log(RG_high - VR - V*) + c      (V*, c fitted on training rows)
  physics, no ML (SW)       log g = log|CDS offset| - log(SW swing) + c
  physics features: trees   trees on the gain signals (reset jump, CDS offset, skip pulse, H waveform) + overdrive, swings
  physics + settings: trees
LAMBDA (dark charge = bulk dark current + clock-induced charge):
  physics form, no ML       lambda = exp(a + b VSUB) + exp(c + d H_swing + e V_swing + f SW_swing + h overdrive)
                            (shared shape and level, fitted on training rows; log-space least squares)
  components: trees         log lambda_dark and log lambda_cic predicted separately (trees on the settings), summed
  settings + trace + setup  trees on the settings, every trace-physics feature (tests 1-2) and the gain signals
Reference: settings: trees.

    .venv/bin/python amp_phys/test2_gain_lambda.py --data processed_v3n --features <physics_features_v3.csv> \
        --gain-features <gain_features_v3.csv> --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test1_physics_model import load as load1, phys_cols, score, trees  # noqa: E402


def num(d, c):
    return pd.to_numeric(d[c], errors="coerce").astype(float)


def derived(d):
    g = lambda h: num(d, f"theta.{h}")  # noqa: E731
    D = pd.DataFrame(index=d.index)
    D["VSUB"] = g("VSUB")
    D["overdrive RG_high - VR"] = g("RG_1_H") - g("VR1")
    D["SW swing"] = g("SW_1_H") - g("SW_1_L")
    D["RG swing"] = g("RG_1_H") - g("RG_1_L")
    D["H swing"] = np.mean([g(f"{h}_H") - g(f"{h}_L") for h in ("H1_B", "H2_C", "H3_B")], 0)
    D["V swing"] = np.mean([g(f"{h}_H") - g(f"{h}_L") for h in ("V1_B", "V2_C", "V3_B")], 0)
    D["VDD"] = g("VDD1")
    return D


def run(d, knobs, tr, te):
    S = d[knobs].apply(pd.to_numeric, errors="coerce").fillna(0)
    D = derived(d)
    G = pd.DataFrame({"log|reset jump|": np.log(num(d, "reset jump (ADU)").abs()),
                      "log|CDS offset|": np.log(num(d, "CDS offset (ADU)").abs().clip(1e-3)),
                      "log skip pulse": np.log(num(d, "skip pulse (ADU)").clip(1)),
                      "log H p2p": np.log(num(d, "H waveform p2p (ADU)").clip(1))})
    P = phys_cols(d)
    yg = np.log(num(d, "label.gain_adu_per_e"))
    yl = np.log(num(d, "label.lambda_e_per_bin"))
    pg, pl = {}, {}
    # ---------------- gain
    pg["settings: trees"] = trees().fit(S[tr], yg[tr]).predict(S[te])
    x = D["overdrive RG_high - VR"].to_numpy()
    J = G["log|reset jump|"].to_numpy()
    f = lambda p, m: J[m] - np.log(np.clip(x[m] - p[0], 1e-3, None)) + p[1]  # noqa: E731
    r = least_squares(lambda p: f(p, tr) - yg[tr], [9.5, 0.0], bounds=([5, -10], [x[tr].min() - 0.2, 10]))
    pg["physics, no ML (reset)"] = f(r.x, te)
    vstar = r.x[0]
    c2 = np.median(yg[tr] - (G["log|CDS offset|"][tr] - np.log(D["SW swing"][tr])))
    pg["physics, no ML (SW)"] = (G["log|CDS offset|"] - np.log(D["SW swing"]))[te] + c2
    GF = pd.concat([G, D[["overdrive RG_high - VR", "SW swing", "VDD"]]], axis=1)
    pg["physics features: trees"] = trees().fit(GF[tr], yg[tr]).predict(GF[te])
    GS = pd.concat([GF, S], axis=1)
    pg["physics + settings: trees"] = trees().fit(GS[tr], yg[tr]).predict(GS[te])
    # ---------------- lambda
    pl["settings: trees"] = trees().fit(S[tr], yl[tr]).predict(S[te])
    Z = D[["VSUB", "H swing", "V swing", "SW swing", "overdrive RG_high - VR"]].to_numpy()
    Zm, Zs = Z[tr].mean(0), Z[tr].std(0) + 1e-9
    Zn = (Z - Zm) / Zs
    form = lambda p, m: np.log(np.exp(p[0] + p[1] * Zn[m, 0]) + np.exp(p[2] + Zn[m, 1:] @ p[3:7]))  # noqa: E731
    r = least_squares(lambda p: form(p, tr) - yl[tr], np.r_[np.log(0.1), 0.5, np.log(0.1), np.zeros(4)])
    pl["physics form, no ML"] = form(r.x, te)
    yd = np.log(num(d, "label.lambda_dark").clip(1e-4))
    yc = np.log(num(d, "label.lambda_cic").clip(1e-4))
    ok = np.isfinite(yd) & np.isfinite(yc)
    trc = tr & ok.to_numpy()
    if trc.sum() > 20:
        ld = trees().fit(S[trc], yd[trc]).predict(S[te])
        lc = trees().fit(S[trc], yc[trc]).predict(S[te])
        pl["components (dark + CIC): trees"] = np.log(np.exp(ld) + np.exp(lc))
    ALL = pd.concat([S, D, P, G], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)
    pl["settings + trace + setup: trees"] = trees().fit(ALL[tr], yl[tr]).predict(ALL[te])
    return {"gain": (yg[te].to_numpy(), pg), "lambda": (yl[te].to_numpy(), pl)}, vstar, r.x


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--gain-features", required=True)
    ap.add_argument("--plan", default="lvl1/packet_plan_v3n/plan.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d, knobs = load1(a)
    d = d.merge(pd.read_csv(a.gain_features), on=["packet_id", "board", "channel"], how="inner").reset_index(drop=True)
    d = d[np.isfinite(num(d, "reset jump (ADU)")) & np.isfinite(num(d, "label.lambda_e_per_bin"))
          & (num(d, "label.lambda_e_per_bin") > 0)].reset_index(drop=True)
    print(f"{len(d)} rows ({(d.session == 'G').sum()} G, {(d.session == 'H').sum()} H)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = ["# Test 2: physics models for gain and dark charge (train G, sealed H, board 101 ch0/1/3)", ""]
    for cell in ("new amplifier", "trained"):
        rows = {"gain": [], "lambda": []}
        vst = []
        for amp in (0, 1, 3):
            te = (d.session.eq("H") & d.channel.eq(amp)).to_numpy()
            tr = d.session.eq("G").to_numpy() & (~d.channel.eq(amp).to_numpy() if cell == "new amplifier" else True)
            res, vstar, _ = run(d, knobs, tr, te)
            vst.append(vstar)
            for t in rows:
                rows[t].append((res[t][0], res[t][1], amp))
        for t in rows:
            R = score(rows[t])
            R.to_csv(out / f"test2_{t}_{cell.replace(' ', '_')}.csv", index=False)
            L += [f"## {t}, {cell}", "", R.round(3).to_markdown(index=False), ""]
        L += [f"(fitted V* of the reset formula: {np.round(vst, 2)} V)", ""]
    L += ["Scorecard v3 on H, new amplifier: gain settings gp 3.9 %, trees 3.6 %; lambda settings gp 71.7 %, trees 66.4 %,",
          "trace physics 50.9 %, trace trees 49.5 %.  Trained: gain settings + amplifier gp 0.6 %; lambda settings + amplifier",
          "trees 10.5 %.  Change rho (new amp): gain settings 0.87; lambda settings + trace combined trees 0.94."]
    (out / "test2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
