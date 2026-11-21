import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
lab = ["new amplifier\n(sealed test)", "amplifier seen\nin training"]
v = [18.4, 13.6]; a = [9.2, 6.4]
x = np.arange(2); w = 0.36
fig, ax = plt.subplots(figsize=(5.8, 4.3))
b1 = ax.bar(x - w/2, v, w, color="0.6", edgecolor="k", label="voltages only")
b2 = ax.bar(x + w/2, a, w, color="C3", edgecolor="k", label="trace physics + trace features\n+ voltages")
for bars in (b1, b2):
    for r in bars: ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.3, f"{r.get_height():.1f} %", ha="center", fontsize=12)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("typical error on σe (%)"); ax.set_ylim(0, 22)
ax.set_title("Predicting image noise"); ax.legend(fontsize=10, loc="upper right")
fig.tight_layout(); fig.savefig(sys.argv[1] + "/noise_bars2.png")
