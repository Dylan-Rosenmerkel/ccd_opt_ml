exec(open("usab2.py").read().split("for name, X in")[0])
from scipy.stats import pearsonr
REP = [("SW low", "theta.SW_1_L"), ("RG high", "theta.RG_1_H"), ("RG low", "theta.RG_1_L"),
       ("V1 high", "theta.V1_A_H"), ("V1 low", "theta.V1_A_L"), ("V2 high", "theta.V2_C_H"), ("V2 low", "theta.V2_C_L"),
       ("V3 high", "theta.V3_A_H"), ("V3 low", "theta.V3_A_L"), ("H1 high", "theta.H1_A_H"), ("H1 low", "theta.H1_A_L"),
       ("H2 high", "theta.H2_C_H"), ("H2 low", "theta.H2_C_L"), ("H3 high", "theta.H3_A_H"), ("H3 low", "theta.H3_A_L"),
       ("VSUB", "theta.VSUB"), ("VDD", "theta.VDD1"), ("VR", "theta.VR1")]
# check paired columns move together
for a, b in [("theta.SW_1_L","theta.SW_2_L"),("theta.V1_A_L","theta.V1_B_L"),("theta.H1_A_H","theta.H1_B_H"),("theta.VDD1","theta.VDD2")]:
    print(a, b, round(np.corrcoef(V[a], V[b])[0,1], 4))
g = d[d.ok].copy()
g["lnl"] = np.log(g["label.lambda_e_per_bin"].astype(float)); g["lns"] = np.log(g["label.sigma_e"].astype(float))
Vg = V.loc[g.index]
rows = []
for name, c in REP:
    r1 = pearsonr(Vg[c], g.lnl)[0]; r2 = pearsonr(Vg[c], g.lns)[0]
    pa1 = [pearsonr(Vg[c][g.channel == a], g.lnl[g.channel == a])[0] for a in (0, 1, 3)]
    pa2 = [pearsonr(Vg[c][g.channel == a], g.lns[g.channel == a])[0] for a in (0, 1, 3)]
    rows.append((name, r1, min(pa1), max(pa1), r2, min(pa2), max(pa2)))
T = pd.DataFrame(rows, columns=["voltage", "r_dc", "dc_lo", "dc_hi", "r_noise", "n_lo", "n_hi"])
print("n good images", len(g), "per amp", g.channel.value_counts().to_dict())
print("r(ln DC, ln noise) =", round(np.corrcoef(g.lnl, g.lns)[0,1], 2))
cv = np.corrcoef(Vg[[c for _, c in REP]].values.T); np.fill_diagonal(cv, 0); print("max |r| between voltages:", round(np.abs(cv).max(), 2))
print(T.round(2).to_string()); T.to_csv("volt_corr.csv", index=False)
names = [n for n, _ in REP]; C = pd.DataFrame(np.corrcoef(Vg[[c for _, c in REP]].values.T), index=names, columns=names)
pairs = [(names[i], names[j], C.iloc[i, j]) for i in range(len(names)) for j in range(i + 1, len(names)) if abs(C.iloc[i, j]) > 0.3]
for p in sorted(pairs, key=lambda t: -abs(t[2])): print(p[0], "|", p[1], round(p[2], 2))
C.to_csv("volt_volt_corr.csv")
