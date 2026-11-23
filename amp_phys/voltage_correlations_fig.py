exec(open("corr.py").read().split("names = [n for")[0])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, sys
names = [n for n, _ in REP] + ["ln DC", "ln noise"]
M = np.column_stack([Vg[c].values for _, c in REP] + [g.lnl.values, g.lns.values])
C = np.corrcoef(M.T)
fig, ax = plt.subplots(figsize=(6.6, 5.8))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=9)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
ax.axhline(17.5, color="k", lw=1); ax.axvline(17.5, color="k", lw=1); ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="correlation r")
ax.set_title("Correlation matrix: 18 voltages, dark charge, noise", fontsize=12)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=200); plt.close(fig)
