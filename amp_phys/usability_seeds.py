exec(open("orig8.py").read().split("res = {}")[0])
h = d.session.eq("H"); bad = h & ~d.ok; ball = bad & d.kind.eq("broken on all")
P = {}
for name, X in [("v", Zv), ("vt", np.hstack([Zv, X8]))]:
    cs = []; acc = pd.Series(0.0, index=d.index)
    for seed in range(10):
        p = pd.Series(np.nan, index=d.index)
        for a in (0,1,3):
            tr = (d.session.eq("G") & ~d.channel.eq(a)).values; te = (h & d.channel.eq(a)).values
            c = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1).fit(X[tr], d.ok[tr])
            p[te] = c.predict_proba(X[te])[:, list(c.classes_).index(True)]
        cs.append(((p[ball]<0.5).sum(), (p[bad & ~d.kind.eq('broken on all')]<0.5).sum(), (p[h&d.ok]<0.5).sum()))
        acc[h] += p[h] / 10
    print(name, "per seed (all-broken caught /18, one-amp caught /2, good blocked /67):", cs)
    P[name] = acc
out = d[h][["theta_id","channel","ok","kind"]].copy(); out["p_v"] = P["v"][h]; out["p_vt"] = P["vt"][h]
out.to_csv("pgood_H.csv"); print(out[out.kind.eq("broken on all")].round(2).to_string())
