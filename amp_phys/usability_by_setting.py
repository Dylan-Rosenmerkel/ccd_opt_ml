import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
d = pd.read_csv("dataset.csv", low_memory=False).copy()
packets = json.load(open("plan.json"))["packets"]
session = {q["packet_id"]: str(q.get("session", "?")) for q in packets}
def pid0(pid):
    if pid in session: return pid
    for q in session:
        if pid.startswith(q) and not pid[len(q)].isdigit(): return q
d["session"] = [session.get(pid0(str(x)), "?") for x in d.packet_id]
d = d[d.board.eq(101) & d.channel.isin([0, 1, 3]) & d.session.isin(["G", "H"])].copy()
d["ok"] = d["label.ok"].astype(str).str.lower().eq("true")
F = pd.read_csv("pf.csv"); d = d.merge(F, on=["packet_id", "board", "channel"], how="left")
knobs = [c for c in d.columns if c.startswith("theta.") and pd.to_numeric(d[c], errors="coerce").std() > 1e-6]
V = d[knobs].apply(pd.to_numeric, errors="coerce").fillna(0)
P = pd.DataFrame({"s500": np.log(d["sigma_500 pred (ADU)"].clip(1e-6)), "Sw": np.log(d["S_w"].clip(1e-12)), "K": np.log(d["K"].clip(1e-3))})
for c in [c for c in d.columns if c.startswith("sigma_1 tau=")]: P[c] = np.log(d[c].clip(1e-6))
P = P.fillna(P.median())
T = d[[c for c in d.columns if c.startswith("trace.") and c not in ("trace.n_usable", "trace.n_rejected_values")]].apply(pd.to_numeric, errors="coerce")
T = T.fillna(T.median())
print("rows", len(d), "unusable", (~d.ok).sum(), "missing physics", d["S_w"].isna().sum())
# per (session, theta) how many amps usable
nk = d.groupby(["session", "theta_id"]).ok.transform("sum"); na = d.groupby(["session", "theta_id"]).ok.transform("size")
d["kind"] = np.where(nk == 0, "broken on all", np.where(nk == na, "usable on all", "mixed"))
for name, X in [("voltages", V), ("voltages + physics(9)", pd.concat([V, P], axis=1)), ("voltages + 141 trace", pd.concat([V, T], axis=1)),
                ("141 trace only", T), ("physics(9) only", P)]:
    out = []
    for a in (0, 1, 3):
        tr = d.session.eq("G") & ~d.channel.eq(a); te = d.session.eq("H") & d.channel.eq(a)
        c = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1).fit(X[tr], d.ok[tr])
        p = c.predict_proba(X[te])[:, 1]
        out.append(pd.DataFrame({"p": p, "ok": d.ok[te].values, "kind": d.kind[te].values}))
    o = pd.concat(out)
    bad = o[~o.ok]
    print(f"{name:25s} AUC {roc_auc_score(o.ok, o.p):.3f}  caught {(bad.p<0.5).sum()}/{len(bad)}  "
          f"(broken-on-all {((bad.p<0.5)&(bad.kind=='broken on all')).sum()}/{(bad.kind=='broken on all').sum()}, "
          f"mixed {((bad.p<0.5)&(bad.kind=='mixed')).sum()}/{(bad.kind=='mixed').sum()})  usable blocked {((o.p<0.5)&o.ok).sum()}/{o.ok.sum()}")
print(d[d.session.eq("H") & d.kind.eq("mixed")][["theta_id", "channel", "ok"]].to_string())
m = d[d.kind.eq("mixed")]
print(m[["session","theta_id","channel","ok","label.flags","label.lambda_e_per_bin","label.sigma_e","label.rows_chi2_dof"]].to_string())
print(d[~d.ok].groupby("kind").size())
