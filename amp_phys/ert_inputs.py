"""
ert_inputs.py -- the test-1 extremely randomised trees (amp_phys/test1_physics_model.trees) on every input type, trained on
session G and scored once on session H, board 101 ch0/1/3, for sigma_e, lambda and gain, new amplifier (left out of
training) and seen amplifier (all three in training, no amplifier identity).  Inputs: voltages (theta), the 9 trace
physics values (test 1), the 4 signal sizes (gain_features.py), the 141 general trace features (trace.* of dataset.csv)
and their combinations.  No derived physics-prediction feature (test 1's 8.6 % model has one).  Rows: as test 1 plus
lambda > 0.005.  Score: RMS of the log error (%), with a 95 % packet bootstrap (Poisson(1) weight per H packet, 1000 draws).

    .venv/bin/python amp_phys/ert_inputs.py <worktree> <out dir>
"""
import sys, types
from pathlib import Path
import numpy as np, pandas as pd
WT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sys.path.insert(0, str(WT / "amp_phys"))
from test1_physics_model import load, phys_cols, trees
a = types.SimpleNamespace(data=str(WT / "processed_v3n"), features=str(WT / "amp_phys/results/per_channel/physics_features_v3.csv"),
                          plan=str(WT / "lvl1/packet_plan_v3n/plan.json"))
import os; os.chdir(WT)
d, knobs = load(a)
G = pd.read_csv(WT / "amp_phys/results/per_channel/gain_features_v3.csv").groupby(["packet_id", "board", "channel"], as_index=False).first()
d = d.merge(G, on=["packet_id", "board", "channel"], how="left")
lab = {"sigma_e": "label.sigma_e", "lambda": "label.lambda_e_per_bin", "gain": "label.gain_adu_per_e"}
Y = {t: np.log(pd.to_numeric(d[c], errors="coerce").where(lambda x: x > 0)) for t, c in lab.items()}
V = d[knobs].apply(pd.to_numeric, errors="coerce").fillna(0)
P = phys_cols(d)
gcols = ["reset jump (ADU)", "H waveform p2p (ADU)", "skip pulse (ADU)", "CDS offset (ADU)"]
S = pd.DataFrame({c: (np.log(np.abs(d[c]).clip(1e-9)) if c != "CDS offset (ADU)" else d[c]) for c in gcols})
tcols = [c for c in d.columns if c.startswith("trace.") and c not in ("trace.n_usable", "trace.n_rejected_values")]
T = d[tcols].apply(pd.to_numeric, errors="coerce")
sets = {"voltages (31)": [V], "trace physics values (9)": [P], "signal sizes (4)": [S], "141 trace features": [T],
        "physics + signal sizes (13)": [P, S], "physics + 141": [P, T],
        "physics + voltages": [P, V], "signal sizes + voltages": [S, V], "141 + voltages": [T, V],
        "physics + signal sizes + voltages": [P, S, V], "everything (physics + sizes + 141 + voltages)": [P, S, T, V]}
rows = []
for target, y in Y.items():
    lam_ok = (Y["lambda"] > np.log(0.005)).to_numpy()
    fin = np.isfinite(y.to_numpy()) & lam_ok
    for cell in ("new amplifier", "seen amplifier"):
        for name, parts in sets.items():
            X = pd.concat(parts, axis=1)
            X.columns = [f"c{i}" for i in range(X.shape[1])]
            e_all, p_all = [], []
            for amp in (0, 1, 3):
                te = d.session.eq("H").to_numpy() & d.channel.eq(amp).to_numpy() & fin
                tr = d.session.eq("G").to_numpy() & fin & (~d.channel.eq(amp).to_numpy() if cell == "new amplifier" else True)
                Xt = X[tr]
                keep = Xt.columns[(Xt.isna().mean() <= 0.2) & (Xt.std() > 0)]
                med = Xt[keep].median()
                m = trees().fit(Xt[keep].fillna(med), y[tr])
                e_all.append(m.predict(X.loc[te, keep].fillna(med)) - y[te].to_numpy())
                p_all.append(d.packet_id[te].to_numpy())
            e = np.concatenate(e_all)
            pk = np.concatenate(p_all)
            up, inv = np.unique(pk, return_inverse=True)
            W = np.random.default_rng(0).poisson(1.0, (1000, len(up)))[:, inv]
            bs = 100 * np.sqrt((W * e ** 2).sum(1) / np.maximum(W.sum(1), 1))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(target=target, cell=cell, inputs=name, n_test=len(e), rms=100 * np.sqrt(np.mean(e ** 2)),
                             median=100 * np.median(np.abs(e)), lo=lo, hi=hi))
            print(rows[-1], flush=True)
R = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); R.to_csv(OUT / "ert_inputs.csv", index=False)
print(R.pivot_table(index="inputs", columns=["cell", "target"], values="rms", sort=False).round(1).to_string())
