"""
channel_knees.py -- channel potentials without SIMS: for every rail scanned in v3 (board 101), a hinge fit of each output
(sigma_e, lambda, gain, the trace white noise, CIC) against that rail with all other rails regressed out linearly.  A
significant hinge (F-test against the straight line, p < 0.01) marks a voltage where the device physics changes: a
barrier or well potential being crossed (transfer starts to fail, reset loses headroom, the follower leaves saturation,
a surface inverts).  Knee voltage with a bootstrap error (packets), per amplifier (= per CCD).

    .venv/bin/python amp_phys/channel_knees.py --data processed_v3n --plan lvl1/packet_plan_v3n/plan.json \\
        --out amp_phys/results/v3n
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import f as fdist
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datasheet as ds

OUT = {"sigma_e": "label.sigma_e", "lambda": "label.lambda_e_per_bin", "gain": "label.gain_adu_per_e",
       "CIC": "label.lambda_cic", "white noise": "trace.fit: white noise per skip (ADU)"}
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--plan", required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
d = ds.load(Path(a.data), Path(a.plan))
scan = d.board.eq(101) & d.ok & d.session.isin(["G", "H"]) & (ds.num(d["label.lambda_e_per_bin"]) > 0.005)
rows = []
rng = np.random.default_rng(1)
for ch in range(4):
    g = d[scan & d.channel.eq(ch)]
    ks = ds.knob_matrix(g)
    for rail in ks:
        x = ds.num(g[rail]).to_numpy()
        grid = np.quantile(x, np.linspace(0.1, 0.9, 33))
        others = [k for k in ks if k != rail]
        Xo = np.column_stack([ds.num(g[k]) for k in others] + [x, np.ones(len(g))])
        for oname, col in OUT.items():
            y = np.log(ds.num(g[col]).to_numpy())
            ok = np.isfinite(y) & np.isfinite(Xo).all(1)
            X0, y0, x0, pk = Xo[ok], y[ok], x[ok], g.packet_id.to_numpy()[ok]
            if len(y0) < Xo.shape[1] + 15:
                continue
            r0 = y0 - X0 @ np.linalg.lstsq(X0, y0, rcond=None)[0]
            best = (np.inf, None)
            for k in grid:
                A = np.column_stack([X0, np.maximum(0, x0 - k)])
                r = y0 - A @ np.linalg.lstsq(A, y0, rcond=None)[0]
                if r @ r < best[0]:
                    best = (r @ r, k)
            dfree = len(y0) - X0.shape[1] - 2
            F = (r0 @ r0 - best[0]) / (best[0] / dfree)
            p = 1 - fdist.cdf(F, 2, dfree)
            if p > 0.01:
                continue
            ks_b = []
            u = np.unique(pk)
            for _ in range(60):
                s = np.concatenate([np.flatnonzero(pk == q) for q in rng.choice(u, len(u))])
                bb = (np.inf, None)
                for k in grid:
                    A = np.column_stack([X0[s], np.maximum(0, x0[s] - k)])
                    r = y0[s] - A @ np.linalg.lstsq(A, y0[s], rcond=None)[0]
                    if r @ r < bb[0]:
                        bb = (r @ r, k)
                ks_b.append(bb[1])
            A = np.column_stack([X0, np.maximum(0, x0 - best[1])])
            co = np.linalg.lstsq(A, y0, rcond=None)[0]
            rows.append({"amp": f"101 ch{ch}", "rail": rail.replace("theta.", ""), "output": oname,
                         "knee (V)": best[1], "knee err (V)": np.std(ks_b), "p": p,
                         "slope below (%/V)": 100 * co[-3], "slope above (%/V)": 100 * (co[-3] + co[-1])})
R = pd.DataFrame(rows).sort_values(["rail", "output", "amp"])
R.round(4).to_csv(Path(a.out) / "channel_knees.csv", index=False)
# keep knees found on >= 2 amplifiers (a device property, not noise)
cnt = R.groupby(["rail", "output"]).amp.nunique()
keep = cnt[cnt >= 2].index
S = R.set_index(["rail", "output"]).loc[keep].reset_index()
L = ["# Knees (significant hinges, p < 0.01, found on >= 2 amplifiers)", "",
     "| rail | output | " + " | ".join(f"101 ch{c}" for c in range(4)) + " |", "|---|---|---|---|---|---|"]
for (rail, o), gg in S.groupby(["rail", "output"]):
    cells = []
    for c in range(4):
        q = gg[gg.amp == f"101 ch{c}"]
        cells.append(f"{q['knee (V)'].iloc[0]:.2f} ± {q['knee err (V)'].iloc[0]:.2f} ({q['slope below (%/V)'].iloc[0]:+.0f}→{q['slope above (%/V)'].iloc[0]:+.0f}%/V)" if len(q) else "–")
    L.append(f"| {rail} | {o} | " + " | ".join(cells) + " |")
(Path(a.out) / "channel_knees.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
