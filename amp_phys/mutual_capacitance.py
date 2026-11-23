"""
mutual_capacitance.py -- mutual capacitances between the channels from the crosstalk measured at several CDS window
lengths (v2 varied delayInt).  Model per source->target pair:
    k(tau) = a / tau + b
  a / tau : capacitive coupling between the VIDEO LINES after the source followers. A step dV on the source line
            injects a pulse of area R_out C_m dV into the target line (R_out = the target follower's output resistance
            || load); the CDS window of length tau averages it -> k = R_out C_m / tau  ->  C_m = a / R_out.
  b       : tau-independent coupling: charge sharing at the sense nodes / bond wires (k = C_m,sn / C_sn) or a shared
            return impedance (negative).  If read as sense-node coupling: C_m,sn = b x C_sn.
Crosstalk k is target electrons per source electron (equal conversion gains to ~6 %).

    .venv/bin/python amp_phys/mutual_capacitance.py --groups short:441:amp_phys/results/v2/crosstalk_101_short.csv \\
        nominal:1000:... long:2046:... --rout-kohm 2.5 5 --csn-ff 85 --out amp_phys/results/v2/mutual_capacitance
"""
import argparse
import numpy as np, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--groups", nargs="+", required=True)
ap.add_argument("--rout-kohm", type=float, nargs=2, default=[2.5, 5.0]); ap.add_argument("--csn-ff", type=float, default=85.0)
ap.add_argument("--cycle-ns", type=float, default=10.0); ap.add_argument("--out", required=True); a = ap.parse_args()
frames = []
for g in a.groups:
    name, tau, path = g.split(":")
    r = pd.read_csv(path); r["tau_s"] = float(tau) * a.cycle_ns * 1e-9; r["group"] = name
    frames.append(r)
D = pd.concat(frames)
rows = []
for (s, t), g in D.groupby(["source", "target"]):
    k = g["crosstalk (1e-4)"].to_numpy() * 1e-4
    w = 1 / np.maximum((g["hi"] - g["lo"]).to_numpy() * 1e-4 / 2, 1e-6) ** 2
    X = np.column_stack([1 / g.tau_s, np.ones(len(g))])
    W = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(X * W, k * np.sqrt(w), rcond=None)
    cov = np.linalg.inv((X * W).T @ (X * W))
    aa, bb = coef
    ea, eb = np.sqrt(np.diag(cov))
    rows.append({"pair": f"{s}->{t}", "a (ns)": aa * 1e9, "a err": ea * 1e9, "b (1e-4)": bb * 1e4, "b err": eb * 1e4,
                 "C_m video lines (pF)": f"{aa / (a.rout_kohm[1] * 1e3) * 1e12:.2f}-{aa / (a.rout_kohm[0] * 1e3) * 1e12:.2f}",
                 "C_m sense nodes from b (fF)": bb * a.csn_ff})
R = pd.DataFrame(rows)
out = Path(a.out); R.round(4).to_csv(out.with_suffix(".csv"), index=False)
L = ["# Mutual capacitances between the channels (board 101)", "",
     f"k(tau) = a/tau + b fitted on {len(a.groups)} CDS window lengths; R_out {a.rout_kohm[0]}-{a.rout_kohm[1]} k; C_sn {a.csn_ff} fF.", "",
     R.round(3).to_markdown(index=False)]
out.with_suffix(".md").write_text("\n".join(L) + "\n")
print("\n".join(L))
