"""
clock_edges.py -- the horizontal (and vertical) clock edges seen directly in the raw traces, and from them the
mutual capacitance between each H / V line and the sense node.

The raw samples ('values', 15 MS/s) carry the sequencer flags: bit 6 = horizontal-transfer start, bit 7 = vertical.
Per trace: the median waveform over all horizontal transfers (bit-6 edge at 0, -300..+1700 samples); its edges are
found from the waveform (largest slope changes); for each edge the FAST jump (mean of samples +3..+12 minus -12..-3:
the capacitive part, before the ~7-10 us relaxation) and the relaxation time constant.

Across the v3 packets (board 101: every H/V rail's high and low varied), each edge's jump is regressed on all varied
rails at once -> ADU per volt of each rail at each edge.  The edge belongs to the rail with the dominant coefficient;
with the gain, C_line->sense node = |ADU/V| / gain x q (equivalent charge per volt).  Board 102's dead channels
(rails fixed) give the pickup level of the same edges for comparison.

    .venv/bin/python amp_phys/clock_edges.py --data processed_v3n --plan lvl1/packet_plan_v3n/plan.json \\
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
from scipy.optimize import curve_fit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasheet as ds  # noqa: E402

Q = 1.602176634e-19
PRE, POST = 300, 1700


def flag_edges(fl, bit):
    x = (fl >> bit) & 1
    return np.flatnonzero(np.diff(x) == 1) + 1


def h_waveform(f):
    z = np.load(f, allow_pickle=True)
    if "values" not in z.files:
        return None
    v, fl = z["values"].astype(float), z["flags"].astype(np.int64)
    e = flag_edges(fl, 6)
    e = e[(e > PRE) & (e < len(v) - POST)]
    if len(e) < 20:
        return None
    H = np.array([v[k - PRE:k + POST] for k in e])
    H -= np.median(H[:, PRE - 250:PRE - 50], axis=1, keepdims=True)
    return np.median(H, 0)


def find_edges(w, n=7, min_gap=150):
    d2 = np.abs(np.diff(w, 2))
    d2 = np.convolve(d2, np.ones(3) / 3, mode="same")
    idx = []
    for i in np.argsort(-d2):
        if all(abs(i - j) > min_gap for j in idx):
            idx.append(i)
        if len(idx) == n:
            break
    return sorted(int(i) + 1 for i in idx)


def jump(w, e):
    return w[e + 3:e + 12].mean() - w[e - 12:e - 3].mean()


def tau_after(w, e, nxt):
    seg = w[e + 5:min(nxt - 5, e + 600)]
    t = np.arange(len(seg))
    try:
        p, _ = curve_fit(lambda t, a, b, c: a + b * np.exp(-t / c), t, seg, p0=[seg[-1], seg[0] - seg[-1], 100],
                         maxfev=4000)
        return p[2] / 15.0                                        # us (15 MS/s)
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    d = ds.load(data, Path(args.plan))
    rows, wf = [], {}
    for board in (101, 102):
        for ch in range(4):
            for p in sorted(d.packet_id.unique()):
                W = [h_waveform(f) for f in sorted(glob.glob(str(data / p / "traces" / f"theta*_acm{board}_ch{ch}.npz")))]
                W = [w for w in W if w is not None]
                if W:
                    wf[(board, ch, p)] = np.median(W, 0)
    # common edge positions from the board-101 grand mean
    grand = np.median([w for (b, c, p), w in wf.items() if b == 101], 0)
    E = find_edges(grand)
    E = [e for e in E if 15 < e < len(grand) - 15]
    for (b, c, p), w in wf.items():
        r = {"board": b, "channel": c, "packet_id": p}
        for i, e in enumerate(E):
            r[f"edge{i} jump"] = jump(w, e)
            r[f"edge{i} tau_us"] = tau_after(w, e, E[i + 1] if i + 1 < len(E) else len(w))
        rows.append(r)
    J = pd.DataFrame(rows)
    J.to_csv(out / "clock_edges_per_trace.csv", index=False)
    # regression on the rails, board 101
    res = []
    scan = d[d.board.eq(101) & d.session.isin(["G", "H"])]
    for ch in range(4):
        g = scan[scan.channel.eq(ch)].drop_duplicates("packet_id").set_index("packet_id")
        rails = []
        for k in ds.knob_matrix(g.reset_index()):
            x = ds.num(g[k]).to_numpy()
            if not any(abs(np.corrcoef(x, ds.num(g[q]).to_numpy())[0, 1]) > 0.999 for q in rails):
                rails.append(k)
        jj = J[(J.board == 101) & (J.channel == ch)].set_index("packet_id").reindex(g.index)
        X = np.column_stack([ds.num(g[k]) - ds.num(g[k]).median() for k in rails] + [np.ones(len(g))])
        gain = ds.num(g["label.gain_adu_per_e"]).median()
        for i in range(len(E)):
            y = jj[f"edge{i} jump"].to_numpy()
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            rr = y[ok] - X[ok] @ b
            se = np.sqrt(np.diag(np.linalg.pinv(X[ok].T @ X[ok])) * rr.var(ddof=X.shape[1]))
            for k, bb, s in zip(rails, b, se):
                res.append({"amp": f"101 ch{ch}", "edge": i, "edge sample": E[i], "rail": k.replace("theta.", ""),
                            "ADU/V": bb, "err": s, "z": bb / s if s > 0 else np.nan,
                            "C (fF, |ADU/V|/gain*q)": abs(bb) / gain * Q * 1e15,
                            "mean jump (ADU)": np.nanmean(y), "tau (us)": np.nanmedian(jj[f"edge{i} tau_us"])})
    R = pd.DataFrame(res)
    R.round(5).to_csv(out / "clock_edges_regression.csv", index=False)
    # per edge: the dominant rails
    L = ["# Clock edges in the traces -> line <-> sense-node coupling (board 101, v3)", "",
         f"Edges (samples after the bit-6 flag, 15 MS/s): {E}", "",
         "Mean jump per edge (ADU) and relaxation time constant, board 101 vs the board-102 dead channels (pickup):", "",
         "| edge | sample | " + " | ".join(f"101 ch{c}" for c in range(4)) + " | 102 dead (mean) | tau 101 (us) |",
         "|---|---|---|---|---|---|---|---|"]
    for i, e in enumerate(E):
        cells = [f"{J[(J.board == 101) & (J.channel == c)][f'edge{i} jump'].median():.0f}" for c in range(4)]
        dead = J[(J.board == 102) & (J.channel < 3)][f"edge{i} jump"].median()
        tau = J[J.board == 101][f"edge{i} tau_us"].median()
        L.append(f"| {i} | {e} | " + " | ".join(cells) + f" | {dead:.0f} | {tau:.1f} |")
    L += ["", "Per edge, the rails with |z| > 4 (ADU/V; C in fF-equivalent), all four amplifiers:", "",
          "| edge | rail | " + " | ".join(f"101 ch{c}" for c in range(4)) + " |", "|---|---|---|---|---|---|"]
    for (i, rail), gg in R.groupby(["edge", "rail"]):
        if (gg.z.abs() > 4).sum() >= 2:
            cells = [f"{q['ADU/V'].iloc[0]:+.0f} ({q['C (fF, |ADU/V|/gain*q)'].iloc[0]:.3f} fF)" if len(q) else "–"
                     for q in (gg[gg.amp == f"101 ch{c}"] for c in range(4))]
            L.append(f"| {i} | {rail} | " + " | ".join(cells) + " |")
    (out / "clock_edges.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    t = np.arange(-PRE, POST)
    for c in range(4):
        ax.plot(t, np.median([w for (b, cc, p), w in wf.items() if b == 101 and cc == c], 0), lw=1, label=f"101 ch{c}")
    ax.plot(t, np.median([w for (b, cc, p), w in wf.items() if b == 102 and cc < 3], 0), lw=1, color="#8a8984",
            label="102 dead (pickup)")
    for i, e in enumerate(E):
        ax.axvline(e - PRE, color="#0b0b0b", lw=0.5, ls=":")
        ax.text(e - PRE, ax.get_ylim()[1] * 0.9, str(i), fontsize=8)
    ax.set_xlabel("samples after the horizontal-transfer flag (15 MS/s)")
    ax.set_ylabel("ADU")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Mean horizontal-transfer waveform (all v3 packets): numbered edges", loc="left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "clock_edges.png", dpi=150)


if __name__ == "__main__":
    main()
