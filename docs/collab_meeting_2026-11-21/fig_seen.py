"""Noise EDT, two separate bar plots: new amplifier (test 1) and seen amplifier (test 1 + ert_inputs 'everything')."""
import sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
OUT = sys.argv[1]
for fn, ttl, lab, v, cols in [
        ("noise_new.png", "New amplifier (not in training)", ["voltages\nonly", "trace +\nvoltages"], [18.2, 8.6], ["0.6", "C3"]),
        ("noise_seen.png", "Amplifier seen in training", ["voltages\nonly", "trace +\nvoltages", "everything\n(+ 141 features,\nsignal sizes)"], [13.8, 6.5, 6.1], ["0.6", "C3", "C2"])]:
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    b = ax.bar(range(len(v)), v, color=cols, edgecolor="k", width=0.6)
    for r in b: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.3, f"{r.get_height():.1f} %", ha="center", fontsize=12)
    ax.set_xticks(range(len(v))); ax.set_xticklabels(lab, fontsize=10.5); ax.set_ylim(0, 21)
    ax.set_ylabel("typical error (%)"); ax.set_title(ttl, fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fn}"); plt.close(fig)
