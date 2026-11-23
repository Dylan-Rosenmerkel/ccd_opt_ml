import sys, types, os
import numpy as np, pandas as pd
WT = sys.argv[1]; sys.path.insert(0, WT + "/amp_phys"); os.chdir(WT)
from test1_physics_model import load, phys_cols
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
a = types.SimpleNamespace(data="processed_v3n", features="amp_phys/results/per_channel/physics_features_v3.csv", plan="lvl1/packet_plan_v3n/plan.json")
d, knobs = load(a)
y = np.log(pd.to_numeric(d["label.sigma_e"])); g = np.log(pd.to_numeric(d["label.gain_adu_per_e"]))
S = d[knobs].apply(pd.to_numeric, errors="coerce").fillna(0); P = phys_cols(d)
M = {"EDT": lambda s: ExtraTreesRegressor(400, min_samples_leaf=3, max_features=0.5, random_state=s, n_jobs=-1),
     "random forest": lambda s: RandomForestRegressor(400, min_samples_leaf=3, max_features=0.5, random_state=s, n_jobs=-1)}
for name, mk in M.items():
    for cell in ("new amplifier", "seen amplifier"):
        errs = []
        for seed in range(5):
            e = []
            for amp in (0, 1, 3):
                te = (d.session.eq("H") & d.channel.eq(amp)).to_numpy()
                tr = (d.session.eq("G") & (~d.channel.eq(amp) if cell == "new amplifier" else True)).to_numpy()
                gm = mk(seed).fit(S[tr], g[tr])
                X = pd.concat([P, S], axis=1); X["b"] = P["log s500"] - gm.predict(S)
                m = mk(seed).fit(X[tr], y[tr]); e.append(m.predict(X[te]) - y[te].to_numpy())
            e = np.concatenate(e); errs.append(100 * np.sqrt(np.mean(e ** 2)))
        print(f"{name:14s} {cell:15s} {np.mean(errs):5.1f} %  (seeds {min(errs):.1f}-{max(errs):.1f})")
