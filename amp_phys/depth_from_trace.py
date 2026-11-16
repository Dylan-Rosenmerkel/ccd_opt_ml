"""
depth_from_trace.py -- buried-channel depth from ONE raw trace, without a voltage scan.

When the reset transistor switches, the charge in its channel, C_eff W L (RG_high - VR - V*), is injected roughly half
onto the sense node.  The scan (80 v3 settings) measures the slope s = 1/2 C_eff W L and the effective threshold V*.
V* turned out nearly identical on all four PA03 amplifiers (9.38-9.57 V), so it is a property of the reset-FET design,
not of each chip.  With V* fixed, one trace at a known RG_high - VR gives
    s = Q_reset / (RG_high - VR - V*),   C_eff = 2 s / (W L),   x_ch = eps_Si (1/C_eff - 1/C_ox)
Q_reset: the reset edge in the median horizontal-transfer waveform (clock_edges.py, edge 0), divided by the gain.
Validated on PA03 board 101 with V* from the other three amplifiers: a nominal trace gives 0.56-0.70 um against the
scan's 0.51-0.62 um; the median over single traces is within 3-6 % of the scan slope.
Uncertainties printed: from V* (+- --vstar-err) and from the injection split (30-70 % instead of 50 %).
Valid for the PA03 sequencer only (the reset edge ~278 samples before the H flag); other sequencers must be mapped first.

    .venv/bin/python amp_phys/depth_from_trace.py --traces <npz> ... --rg-high 8.88 --vr -3.0 --gain 1.39 --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clock_edges as ce  # noqa: E402
from trace_anatomy import load  # noqa: E402

Q_FC = 1.602e-4                       # fC per electron
WL, C_OX, EPS_SI = 36.0, 0.454, 0.1036   # um^2 (RG 6x6 um, layout), fF/um^2 (recipe 76 nm), fF/um


def depth(slope):
    return EPS_SI * (1 / (2 * slope / WL) - 1 / C_OX)


def reset_jump(path):
    v, fl = load(path)
    e = ce.flag_edges(fl, 6)
    e = e[(e > ce.PRE) & (e < len(v) - ce.POST)]
    if len(e) < 20:
        return np.nan
    H = np.array([v[k - ce.PRE:k + ce.POST] for k in e])
    H -= np.median(H[:, ce.PRE - 250:ce.PRE - 50], axis=1, keepdims=True)
    w = np.median(H, 0)
    edges = [k for k in ce.find_edges(w) if abs((k - ce.PRE) + 278) < 40]       # the PA03 reset edge
    return ce.jump(w, edges[0]) if edges else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--rg-high", type=float, required=True)
    ap.add_argument("--vr", type=float, required=True)
    ap.add_argument("--gain", type=float, required=True, help="ADU per electron (image)")
    ap.add_argument("--vstar", type=float, default=9.47, help="effective reset threshold (PA03 scan mean)")
    ap.add_argument("--vstar-err", type=float, default=0.08, help="spread of V* between amplifiers")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    od = a.rg_high - a.vr - a.vstar
    rows = []
    for t in a.traces:
        q = reset_jump(t) / a.gain * Q_FC
        s = q / od
        lo, hi = q / (od + a.vstar_err), q / (od - a.vstar_err)
        rows.append({"label": a.label, "file": t, "reset charge (fC)": q, "overdrive above V* (V)": od,
                     "slope 1/2 C_eff WL (fF)": s, "C_eff (fF/um2)": 2 * s / WL, "depth (um)": depth(s),
                     "depth range, V* err (um)": f"{depth(hi):.2f}-{depth(lo):.2f}",
                     "depth range, split 30-70 % (um)": f"{depth(s * 0.5 / 0.7):.2f}-{depth(s * 0.5 / 0.3):.2f}"})
    R = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "depth_from_trace.csv"
    R.to_csv(p, mode="a", header=not p.exists(), index=False)
    print(R.drop(columns=["file"]).to_string(index=False, float_format=lambda x: f"{x:.3g}"))


if __name__ == "__main__":
    main()
