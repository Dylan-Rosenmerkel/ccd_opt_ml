"""
sense_node_coupling.py -- mutual capacitances between each amplifier's SENSE NODE and every voltage line, from the v3
scan (board 101: every rail's high and low varied independently, 18 voltages).

Two measures, both regressed on all varied rails at once (each coefficient = that rail alone):
 1  WAVEFORM: the mean skip cycle ("skip: mean shape", pedestal level removed, theta traces of each packet averaged).
    d(waveform)/dV_k vs time in the cycle shows WHERE line k couples (its edges) and how strongly.
 2  SETTLED OFFSET: the CDS of an empty skip ("skip: CDS mean (skips>=10)"; signal-window minus pedestal-window level).
    A capacitive step onto the sense node persists through the 10 us window; pickup into the video cable decays in
    ~1 us (R_out C_line), so the settled offset is pickup-free.  A line whose level differs between the pedestal and
    the signal windows (SW, OG; RG through the reset) gives dCDS/dV_k = A C_k / C_sn (in ADU via the gain) ->
        C_k = (dCDS/dV_k [ADU/V]) / (gain [ADU/e]) x q      (equivalent charge per volt on the sense node)
    A rail that is held constant through the cycle (VR, VDD, H/V rails during the skips) gives no settled offset;
    its coupling shows only through its noise (common_mode.py).
 Also: the reset-pulse height and SW dip vs RG/SW high and low separately -> the voltage dependence of the overlap
 capacitance, a C-V curve of the gate edge (C at each rail value from the local slope, quadratic fit).

    .venv/bin/python amp_phys/sense_node_coupling.py --data processed_v3n --plan lvl1/packet_plan_v3n/plan.json \\
        --out amp_phys/results/v3n
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasheet as ds  # noqa: E402

Q = 1.602176634e-19
C_SN_FF = 85.0


def waveforms(data: Path, packet: str, ch: int):
    W = []
    for f in sorted(glob.glob(str(data / packet / "traces" / f"theta*_acm101_ch{ch}.npz"))):
        z = np.load(f, allow_pickle=True)
        k = "curve: skip: mean shape"
        if k in z.files:
            W.append(z[k].astype(float))
    return np.mean(W, 0) if W else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    d = ds.load(data, Path(args.plan))
    scan = d.board.eq(101) & d.session.isin(["G", "H"])
    rows, curves = [], {}
    for ch in range(4):
        g = d[scan & d.channel.eq(ch)].reset_index(drop=True)
        rails = []
        for r in ds.knob_matrix(g):                        # merge rails that moved in lockstep (A/B pins, offset)
            x = ds.num(g[r]).to_numpy()
            if not any(abs(np.corrcoef(x, ds.num(g[q]).to_numpy())[0, 1]) > 0.999 for q in rails):
                rails.append(r)
        X = np.column_stack([ds.num(g[r]) - ds.num(g[r]).median() for r in rails] + [np.ones(len(g))])
        gain = ds.num(g["label.gain_adu_per_e"]).median()
        # 2 settled offset
        y = ds.num(g["trace.skip: CDS mean (skips>=10)"]).to_numpy()
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        res = y[ok] - X[ok] @ b
        cov = np.linalg.inv(X[ok].T @ X[ok]) * res.var(ddof=X.shape[1])
        for i, r in enumerate(rails):
            e_per_v = b[i] / gain
            rows.append({"amp": f"101 ch{ch}", "line": r.replace("theta.", ""), "measure": "settled CDS offset",
                         "ADU/V": b[i], "err ADU/V": np.sqrt(cov[i, i]), "e/V": e_per_v,
                         "C to sense node (fF)": abs(e_per_v) * Q * 1e15,
                         "sign": "+" if b[i] > 0 else "-", "significance (sigma)": abs(b[i]) / np.sqrt(cov[i, i])})
        # amplitude features vs the rails (pulse = RG edge, dip = SW edge): the C-V of the overlap
        for feat in ("trace.skip: pulse peak height", "trace.skip: dip height"):
            yy = ds.num(g[feat]).to_numpy()
            ok2 = np.isfinite(yy) & np.isfinite(X).all(1)
            bb = np.linalg.lstsq(X[ok2], yy[ok2], rcond=None)[0]
            for i, r in enumerate(rails):
                rows.append({"amp": f"101 ch{ch}", "line": r.replace("theta.", ""), "measure": feat.split(": ")[1],
                             "ADU/V": bb[i], "e/V": bb[i] / gain, "C to sense node (fF)": abs(bb[i] / gain) * Q * 1e15})
        # 1 waveform regression
        W = [waveforms(data, p, ch) for p in g.packet_id]
        okw = np.array([w is not None for w in W]) & np.isfinite(X).all(1)
        Y = np.array([w for w, o in zip(W, okw) if o])
        B = np.linalg.lstsq(X[okw], Y, rcond=None)[0]
        curves[ch] = (rails, B[:-1] / gain * Q * 1e15)                   # fF-equivalent per V vs time
    R = pd.DataFrame(rows)
    R.round(5).to_csv(out / "sense_node_coupling.csv", index=False)
    S = R[R.measure == "settled CDS offset"]
    P = S.pivot(index="line", columns="amp", values="C to sense node (fF)")
    Z = S.pivot(index="line", columns="amp", values="significance (sigma)")
    sg = S.pivot(index="line", columns="amp", values="sign")
    L = ["# Sense node <-> voltage-line mutual capacitances (board 101, v3 scan)", "",
         "## Settled (pickup-free) coupling: C_k = |dCDS/dV_k| / gain x q, fF.  (sign, significance)", "",
         "| line | " + " | ".join(P.columns) + " |", "|---|" + "---|" * len(P.columns)]
    for line in P.index:
        cells = [f"{P.loc[line, a]:.4f} ({sg.loc[line, a]}{Z.loc[line, a]:.1f}σ)" for a in P.columns]
        L.append(f"| {line} | " + " | ".join(cells) + " |")
    for feat in ("pulse peak height", "dip height"):
        F = R[R.measure == feat].pivot(index="line", columns="amp", values="C to sense node (fF)")
        L += ["", f"## Edge amplitude '{feat}' per volt of each rail (fF-equivalent, includes cable pickup)", "",
              F.round(3).to_markdown()]
    (out / "sense_node_coupling.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    # figure: waveform response curves for the strongest rails
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, ch in zip(axs.ravel(), range(4)):
        rails, C = curves[ch]
        strength = np.abs(C).max(1)
        for i in np.argsort(-strength)[:6]:
            ax.plot(C[i], lw=1.2, label=rails[i].replace("theta.", ""))
        ax.axhline(0, color="#8a8984", lw=0.6)
        ax.set_title(f"101 ch{ch}: d(skip waveform)/dV per rail (fF-equivalent)", loc="left", fontsize=9)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs[1]:
        ax.set_xlabel("time in the skip cycle (binned samples; pedestal window first)")
    fig.tight_layout()
    fig.savefig(out / "sense_node_coupling.png", dpi=150)


if __name__ == "__main__":
    main()
