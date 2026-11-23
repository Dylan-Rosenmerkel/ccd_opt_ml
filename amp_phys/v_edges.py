"""
v_edges.py -- the vertical-clock edges in the raw traces (flag bit 7 = vertical transfer), regressed on every rail of
the v3 scan (board 101) -> the V1/V2/V3 (and TG) lines' coupling to the output, in ADU/V and fF-equivalent at the
sense node.  Same method as clock_edges.py, window -500..+3000 samples around each bit-7 edge.

    .venv/bin/python amp_phys/v_edges.py --data processed_v3n --plan lvl1/packet_plan_v3n/plan.json --out amp_phys/results/v3n
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasheet as ds
from clock_edges import flag_edges, find_edges, jump, Q

PRE, POST = 500, 3000
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--plan", required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
data, out = Path(a.data), Path(a.out)
d = ds.load(data, Path(a.plan))
wf = {}
for board in (101, 102):
    for ch in range(4):
        for p in sorted(d.packet_id.unique()):
            W = []
            for f in sorted(glob.glob(str(data / p / "traces" / f"theta*_acm{board}_ch{ch}.npz"))):
                z = np.load(f, allow_pickle=True)
                if "values" not in z.files:
                    continue
                v, fl = z["values"].astype(float), z["flags"].astype(np.int64)
                for e in flag_edges(fl, 7):
                    if PRE < e < len(v) - POST and ((fl[e - PRE:e] >> 0) & 1).sum() == 0:   # no pixel window before
                        w = v[e - PRE:e + POST]
                        W.append(w - np.median(w[:PRE - 50]))
            if W:
                wf[(board, ch, p)] = np.median(W, 0)
grand = np.median([w for (b, c, p), w in wf.items() if b == 101], 0)
E = [e for e in find_edges(grand, n=10, min_gap=60) if 15 < e < len(grand) - 15]
rows = [{"board": b, "channel": c, "packet_id": p, **{f"edge{i}": jump(w, e) for i, e in enumerate(E)}} for (b, c, p), w in wf.items()]
J = pd.DataFrame(rows)
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
        y = jj[f"edge{i}"].to_numpy(); ok = np.isfinite(y) & np.isfinite(X).all(1)
        if ok.sum() < X.shape[1] + 10:
            continue
        b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None); rr = y[ok] - X[ok] @ b
        se = np.sqrt(np.diag(np.linalg.pinv(X[ok].T @ X[ok])) * rr.var(ddof=X.shape[1]))
        for k, bb, s in zip(rails, b, se):
            res.append({"amp": f"101 ch{ch}", "edge": i, "sample": E[i] - PRE, "rail": k.replace("theta.", ""), "ADU/V": bb,
                        "z": bb / s if s > 0 else np.nan, "C fF-eq": abs(bb) / gain * Q * 1e15})
R = pd.DataFrame(res); R.round(5).to_csv(out / "v_edges_regression.csv", index=False)
L = ["# Vertical-clock edges in the traces (board 101, v3)", "", f"edges (samples after the bit-7 flag): {[e - PRE for e in E]}", "",
     "| edge | sample | " + " | ".join(f"101 ch{c} jump" for c in range(4)) + " | 102 dead |", "|---|---|---|---|---|---|---|"]
for i, e in enumerate(E):
    L.append(f"| {i} | {e - PRE} | " + " | ".join(f"{J[(J.board == 101) & (J.channel == c)][f'edge{i}'].median():.0f}" for c in range(4))
             + f" | {J[(J.board == 102) & (J.channel < 3)][f'edge{i}'].median():.0f} |")
L += ["", "Rails with |z| > 4 on >= 2 amplifiers (ADU/V, fF-equivalent):", "", "| edge | rail | " + " | ".join(f"101 ch{c}" for c in range(4)) + " |", "|---|---|---|---|---|---|"]
for (i, rail), gg in R.groupby(["edge", "rail"]):
    if (gg.z.abs() > 4).sum() >= 2:
        L.append(f"| {i} | {rail} | " + " | ".join(f"{q['ADU/V'].iloc[0]:+.0f} ({q['C fF-eq'].iloc[0]:.3f})" if len(q) else "–" for q in (gg[gg.amp == f"101 ch{c}"] for c in range(4))) + " |")
(out / "v_edges.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
