exec(open("orig8.py").read().split("PH = [")[0])
nom = d.theta_id.value_counts().index[0]
n = d[d.theta_id.eq(nom) & d.ok]
print("nominal theta", nom, "packets", n.packet_id.nunique(), "rows per ch", n.channel.value_counts().to_dict())
cols = [c for c in d.columns if c.startswith("sigma_1 tau=")] + ["S_w", "K", "sigma_500 pred (ADU)", "sigma_500 white only (ADU)",
        "reset jump (ADU)", "H waveform p2p (ADU)", "skip pulse (ADU)", "CDS offset (ADU)", "label.sigma_adu", "label.gain_adu_per_e", "label.sigma_e"]
M = n.groupby("channel")[cols].median().T
S = n.groupby("channel")[cols].std().T / M.abs()      # repeat scatter, relative
out = M.copy(); out.columns = [f"ch{c}" for c in out.columns]
for c in out.columns: out[c + " rel. scatter"] = S[int(c[2:])]
pd.set_option("display.width", 200); print(out.to_string(float_format=lambda v: f"{v:.4g}"))
out.to_csv("feat_examples.csv")
