"""
chain_calibration.py -- absolute calibration of the readout chain and a first inversion of the output amplifier, from
the phase-1 datasheet (amp_phys/datasheet.py) and the priors (amp_phys/priors.toml: the frontend schematic, the ACM,
the process recipe).

1  CHAIN GAIN (ADU per uV at the video input AIN), two independent ways:
   a) dead channels: with no amplifier the input sees only the frontend's load network (20 k to GND, 100 k after the
      coupling capacitor): a Johnson-noise source of known density sqrt(4 k T R) at the board temperature, plus the
      ADA4817's 4 nV/rtHz.  The CDS of two boxcar windows of length tau gives sigma_V = e_n / sqrt(tau); the measured
      white noise per skip (ADU) / sigma_V = ADU per volt.
   b) design: frontend gain (13.64 x 2.00, differential) / ADC LSB (8.192 V / 2^18 = 31.25 uV, unity ACM buffer).
   Agreement of a) and b) checks both the noise model and the units of the traces.
2  CONVERSION GAIN at AIN (uV per electron) = image gain (ADU/e) / chain gain.
3  AMPLIFIER NOISE at AIN (nV/rtHz) = white noise per skip -> volts -> density, minus the live electronics floor.
   On a live channel the source follower's output resistance r_o ~ 1/gm shunts the load resistor, so its Johnson
   noise at AIN is sqrt(4kT/R) * (R || r_o), much smaller than on a dead channel.
4  SOURCE-FOLLOWER INVERSION (one equation, one unknown per amplifier): the amplifier's channel thermal noise
   e_n,node^2 = (8/3) k T_ccd / gm, seen at AIN through the follower gain A = gm R / (1 + gm R) (R = load), plus the
   live floor, must equal the measured density -> gm, A, r_o; then the sense-node conversion gain = CG_AIN / A,
   C_sn = q / CG_node, and the gate area bound W L >= C_sn / C_ox (C_sn includes overlap and wiring, so this is an
   upper bound on the area the gate alone can have... reported as the area-equivalent of C_sn).
   Everything assuming thermal (white) noise only in the fitted white term; 1/f is fitted separately in the traces.

    .venv/bin/python amp_phys/chain_calibration.py --datasheet amp_phys/results/v3n/datasheet.csv \\
        --priors amp_phys/priors.toml --out amp_phys/results/v3n
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

K, Q = 1.380649e-23, 1.602176634e-19


def row(T, name):
    hit = [i for i in T.index if i.endswith("| " + name)]
    return T.loc[hit[0]] if hit else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasheet", required=True)
    ap.add_argument("--priors", required=True)
    ap.add_argument("--t-ccd", type=float, nargs=3, default=[120.0, 140.0, 160.0], help="CCD temperature K: low nominal high")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    P = tomllib.loads(Path(args.priors).read_text())
    fe, acm, tim, proc = P["readout"]["frontend"], P["readout"]["acm"], P["timing"], P["process"]
    T = pd.read_csv(args.datasheet, index_col=0)
    amps = [c for c in T.columns if not c.endswith(" err")]
    dead = [a for a in amps if a.startswith("102") and a != "102 ch3"]
    working = [a for a in amps if a not in dead]
    tau = tim["delayInt_cycles"] * tim["cycle_ns"] * 1e-9
    R_L, R_b = fe["load_resistor_ohm"], fe["bias_resistor_ohm"]
    R_par = 1 / (1 / R_L + 1 / R_b)
    T_b = fe["board_temperature_K"]
    e_op = fe["preamp_en_nV"] * 1e-9
    white, gain = row(T, "white noise per skip (ADU)"), row(T, "gain (ADU/e)")

    # 1 chain gain
    e_dead = np.sqrt(4 * K * T_b * R_par + e_op ** 2)              # V/rtHz at AIN, open input
    sig_dead_V = e_dead / np.sqrt(tau)
    G_dead = {a: white[a] / (sig_dead_V * 1e6) for a in dead}       # ADU per uV
    G_dead_err = {a: white[a + " err"] / (sig_dead_V * 1e6) for a in dead}
    G102 = float(np.mean(list(G_dead.values())))
    G102_spread = float(np.std(list(G_dead.values()), ddof=1))
    G_design = fe["frontend_gain_diff"] / acm["adc_lsb_uV"]
    L = ["# Readout-chain calibration and source-follower inversion", "",
         f"CDS window tau = {tau * 1e6:.1f} us; load {R_L / 1e3:.0f} k || {R_b / 1e3:.0f} k = {R_par / 1e3:.2f} k at "
         f"{T_b:.0f} K; ADA4817 {e_op * 1e9:.0f} nV/rtHz.", "",
         "## 1. Chain gain (ADU per uV at the video input)", "",
         f"Dead channel (open input): e_n = sqrt(4kTR + e_op^2) = {e_dead * 1e9:.1f} nV/rtHz -> sigma per skip "
         f"{sig_dead_V * 1e6:.2f} uV.", "", "| channel | white noise per skip (ADU) | chain gain (ADU/uV) |", "|---|---|---|"]
    for a in dead:
        L.append(f"| {a} (dead) | {white[a]:.3f} ± {white[a + ' err']:.3f} | {G_dead[a]:.3f} ± {G_dead_err[a]:.3f} |")
    L += ["", f"**Board 102 from the dead channels: {G102:.3f} ± {G102_spread:.3f} ADU/uV** (spread between the three). "
          f"**Design value: {G_design:.3f} ADU/uV** ({fe['frontend_gain_diff']:.2f} V/V / {acm['adc_lsb_uV']} uV per LSB). "
          f"Ratio measured/design = {G102 / G_design:.3f}.", "",
          "Board 101 has no dead channel; it uses the design value (same hardware), with the board-102 "
          "measured/design ratio as its uncertainty.", ""]
    G = {a: (G102 if a.startswith("102") else G_design) for a in working}
    G_rel_err = abs(G102 / G_design - 1) + G102_spread / G102

    # 2-4 per amplifier: conversion gain and noise at AIN (chain gain only), then a gm scan (the noise alone cannot
    # fix gm: the follower's channel noise seen at AIN, A^2 (8/3) k T / gm, peaks at gm R = 1 at only a few nV/rtHz)
    Tc_lo, Tc, Tc_hi = args.t_ccd
    rise = row(T, "rise time (ns)")
    rise_dead = float(np.mean([rise[a] for a in dead]))
    rows, scan = [], []
    gms = np.array([5, 10, 20, 50, 100, 200, 500]) * 1e-6
    th_max = np.sqrt(0.25 * (8 / 3) * K * Tc / (1 / R_L))           # max channel-noise density at AIN (gm R = 1)
    for a in working:
        cg_ain = gain[a] / G[a]
        e_meas = white[a] / G[a] * 1e-6 * np.sqrt(tau)
        excess_dead = np.sqrt(max(e_meas ** 2 - e_dead ** 2, 0))      # above the open-input floor
        rows.append({"amp": a, "chain gain (ADU/uV)": G[a], "conversion gain at AIN (uV/e)": cg_ain,
                     "white noise at AIN (nV/rtHz)": e_meas * 1e9,
                     "above the dead-channel floor (nV/rtHz)": excess_dead * 1e9,
                     "electronics share of the variance if unshunted (%)": 100 * min(1, e_dead ** 2 / e_meas ** 2),
                     "rise time (ns)": rise[a]})
        for gm in gms:
            A = gm * R_L / (1 + gm * R_L)
            R_eff = 1 / (1 / R_par + gm)
            floor = np.sqrt(4 * K * T_b / R_par * R_eff ** 2 + e_op ** 2)
            chan = A * np.sqrt((8 / 3) * K * Tc / gm)
            other = np.sqrt(max(e_meas ** 2 - floor ** 2 - chan ** 2, 0))
            scan.append({"amp": a, "gm (uS)": gm * 1e6, "follower gain A": A, "output R (k)": R_eff / 1e3,
                         "live floor (nV/rtHz)": floor * 1e9, "channel noise at AIN (nV/rtHz)": chan * 1e9,
                         "unexplained (nV/rtHz)": other * 1e9, "sense-node CG (uV/e)": cg_ain / A,
                         "C_sn (fF)": Q / (cg_ain / A * 1e-6) * 1e15,
                         "C_sn area-equivalent (um^2)": Q / (cg_ain / A * 1e-6) * 1e15 / proc["cox_fF_per_um2"],
                         "predicted rise / dead rise": R_eff / R_par})
    S = pd.DataFrame(scan)
    S.round(3).to_csv(Path(args.out) / "sf_gm_scan.csv", index=False)
    R = pd.DataFrame(rows).set_index("amp")
    R.round(4).to_csv(Path(args.out) / "chain_calibration.csv")
    L += ["## 2-3. Per amplifier (needs only the chain gain)", "",
          f"Chain-gain uncertainty: {100 * G_rel_err:.0f} % (propagates 1:1 to the conversion gains).", "",
          R.T.to_markdown(floatfmt=".3g"), "",
          "## 4. Source follower: what the noise can and cannot say", "",
          f"With a {R_L / 1e3:.0f} k load at {Tc:.0f} K, the follower's channel thermal noise seen at AIN is at most "
          f"{th_max * 1e9:.1f} nV/rtHz (at gm R = 1), whatever gm is. The measured 19-22 nV/rtHz is therefore mostly the "
          f"frontend's load network and preamp ({e_dead * 1e9:.1f} nV/rtHz open), unless the follower's output "
          "resistance shunts it, which needs a large gm. The noise alone cannot fix gm; the scan below shows each gm's "
          "consequences, and a second observable is needed (rise-time ratio, or a direct follower-gain measurement).", "",
          f"Rise-time check (board 102): the edge rise time scales with the output resistance (R_L || R_b || 1/gm) "
          f"times the input capacitance. Dead channels (open): {rise_dead:.0f} ns; 102 ch3 (live, same board): "
          f"{rise['102 ch3']:.0f} ns, ratio {rise['102 ch3'] / rise_dead:.2f}. No shortening means 1/gm is not much below "
          f"{R_par / 1e3:.0f} k, i.e. gm of order 10-30 uS or less, a follower gain A of 0.2-0.4, and then a sense-node "
          "conversion gain of 4-8 uV/e (C_sn 20-40 fF). Caveat: the dead channels' edges are pickup, whose own shape "
          "may differ.", "",
          "| gm (uS) | A | output R (k) | live floor | channel noise | unexplained | CG node (uV/e) | C_sn (fF) | rise/dead |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in S[S.amp == working[0]].iterrows():
        L.append(f"| {r['gm (uS)']:.0f} | {r['follower gain A']:.2f} | {r['output R (k)']:.1f} | {r['live floor (nV/rtHz)']:.1f} | "
                 f"{r['channel noise at AIN (nV/rtHz)']:.1f} | {r['unexplained (nV/rtHz)']:.1f} | {r['sense-node CG (uV/e)']:.2f} | "
                 f"{r['C_sn (fF)']:.0f} | {r['predicted rise / dead rise']:.2f} |")
    L += ["", f"(scan shown for {working[0]}; all amplifiers in sf_gm_scan.csv)"]
    (Path(args.out) / "chain_calibration.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
