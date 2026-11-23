import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
plt.rcParams.update({"font.size": 13, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
OUT = sys.argv[1]
# --- forest average + extrapolation caveat
rng = np.random.default_rng(1)
f = lambda x: 0.15 + 0.03 * np.sin(x / 1.6) + 0.004 * x
x = np.sort(rng.uniform(0, 7, 45)); y = f(x) + rng.normal(0, 0.006, len(x))
xx = np.linspace(0, 10, 600)
et = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, random_state=0).fit(x[:, None], y)
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.axvspan(7, 10, color="0.92", zorder=0)
ax.text(8.5, 0.205, "no training\nimages here", ha="center", va="top", fontsize=11)
for est in et.estimators_[:6]:
    ax.plot(xx, est.predict(xx[:, None]), color="C1", lw=0.8, alpha=0.6)
ax.plot(x, y, "o", color="0.5", ms=4, label="training images")
ax.plot(xx, f(xx), "k--", lw=1.2, label="true response")
ax.plot(xx, et.predict(xx[:, None]), color="C3", lw=2.5, label="average of 400 random trees")
ax.plot([], [], color="C1", lw=0.8, label="6 of the random trees")
ax.set_ylim(0.11, 0.21)
ax.set_title("Averaged trees: smooth inside, flat outside", fontsize=13); ax.set_xlabel("one input"); ax.set_ylabel("label")
ax.legend(loc="lower left", fontsize=9, ncol=2); fig.tight_layout(); fig.savefig(f"{OUT}/tree_extrap.png"); plt.close(fig)
# --- noise and gain bars
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 4.2), gridspec_kw={"width_ratios": [3, 2]})
lab = ["EDT,\nvoltages\nonly", "formula\nσ₅₀₀ ÷ gain\n(no ML)", "EDT,\ntrace +\nvoltages"]
v = [18.2, 11.9, 8.6]
b = a1.bar(range(3), v, color=["0.6", "C0", "C3"], edgecolor="k", width=0.6)
for r in b: a1.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.1f} %", ha="center", fontsize=12)
a1.set_xticks(range(3)); a1.set_xticklabels(lab, fontsize=11); a1.set_ylabel("typical error, new amplifier (%)"); a1.set_ylim(0, 21)
a1.set_title("Image noise σe", fontsize=13)
g = [3.6, 3.4]
b = a2.bar(range(2), g, color=["0.6", "C3"], edgecolor="k", width=0.6)
for r in b: a2.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.1f} %", ha="center", fontsize=12)
a2.set_xticks(range(2)); a2.set_xticklabels(["EDT,\nvoltages\nonly", "EDT,\nsignal sizes\n+ voltages"], fontsize=11); a2.set_ylim(0, 21)
a2.set_title("Gain g", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/noise_gain_bars.png"); plt.close(fig)
