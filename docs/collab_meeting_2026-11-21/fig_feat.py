"""Where the signal-size model inputs come from, drawn on one nominal trace (same steps as amp_phys/gain_features.py).
usage: python fig_feat.py AMP_PHYS_DIR TRACE.npz OUT"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[1])
import clock_edges as ce
from trace_anatomy import load, rising
plt.rcParams.update({"font.size": 13, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
FS = 15e6
v, fl = load(sys.argv[2]); OUT = sys.argv[3]

# ---- median skip cycle: CDS offset and skip pulse
p0 = rising(fl, 0); P = int(np.median(np.diff(p0)))
s0 = int(np.median([s for k in p0[:300] for s in rising(fl[k:k + P], 1)[:1]]))
C = np.array([v[k:k + P] for k in p0[50:2000] if k + P <= len(v)]); tm = np.median(C, 0)
base = np.median(tm[5:145]); sig = np.median(tm[s0 + 5:s0 + 145])
mask = np.ones(P, bool); mask[:150] = False; mask[s0:s0 + 150] = False
pk = int(np.argmax(np.where(mask, np.abs(tm - base), -1)))
t = np.arange(P) / FS * 1e6
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(t, (tm - base) / 1e3, color="C0", lw=1.2)
ax.axvspan(t[5], t[145], color="C2", alpha=0.15); ax.axvspan(t[s0 + 5], t[min(s0 + 145, P - 1)], color="C1", alpha=0.15)
ax.text(t[75], 0.93, "pedestal\nwindow", ha="center", va="top", fontsize=10, transform=ax.get_xaxis_transform())
ax.text(t[min(s0 + 75, P - 1)], 0.93, "signal\nwindow", ha="center", va="top", fontsize=10, transform=ax.get_xaxis_transform())
ax.annotate("", xy=(t[pk], (tm[pk] - base) / 1e3), xytext=(t[pk], 0), arrowprops=dict(arrowstyle="<->", color="C3", lw=1.5))
ax.text(t[pk] + 0.4, (tm[pk] - base) / 2e3, "skip pulse", color="C3", fontsize=11)
ax.set_xlabel("time within one skip (µs)"); ax.set_ylabel("kADU relative to pedestal")
ax.set_title("One skip (median of ≈2000): pulse size and CDS offset", fontsize=12)
ins = ax.inset_axes([0.07, 0.3, 0.4, 0.38])
z0, z1 = 5, min(s0 + 145, P - 1)
ins.plot(t[z0:z1], tm[z0:z1] - base, color="C0", lw=0.8)
ins.axhline(0, color="C2", lw=1); ins.axhline(sig - base, color="C1", lw=1)
ins.set_ylim(min(sig - base, 0) - 60, max(sig - base, 0) + 60); ins.tick_params(labelsize=8)
ins.set_title(f"CDS offset = {sig - base:.0f} ADU", fontsize=9); ins.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/feat_skip.png"); plt.close(fig)
print("CDS offset", sig - base, "skip pulse", abs(tm[pk] - base))

# ---- median horizontal-transfer waveform: peak-to-peak and reset jump
e = ce.flag_edges(fl, 6); e = e[(e > ce.PRE) & (e < len(v) - ce.POST)]
H = np.array([v[k - ce.PRE:k + ce.POST] for k in e]); H -= np.median(H[:, ce.PRE - 250:ce.PRE - 50], axis=1, keepdims=True)
w = np.median(H, 0); ed = [k for k in ce.find_edges(w) if abs((k - ce.PRE) + 278) < 40]
tt = (np.arange(len(w)) - ce.PRE) / FS * 1e6
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(tt, w / 1e3, color="C1", lw=1.2)
ax.axhline(w.max() / 1e3, color="k", ls=":", lw=1); ax.axhline(w.min() / 1e3, color="k", ls=":", lw=1)
xm = tt[-1] - 8
ax.annotate("", xy=(xm, w.max() / 1e3), xytext=(xm, w.min() / 1e3), arrowprops=dict(arrowstyle="<->", color="k", lw=1.5))
ax.text(xm - 1.5, w.max() * 0.8 / 1e3, "peak-to-peak", ha="right", fontsize=11)
if ed:
    k = ed[0]; j = ce.jump(w, k)
    ax.annotate("", xy=(tt[k], (w[k - 8:k - 3].mean() + j) / 1e3), xytext=(tt[k], w[k - 8:k - 3].mean() / 1e3),
                arrowprops=dict(arrowstyle="<->", color="C3", lw=1.5))
    ax.text(tt[k] + 1, (w[k - 8:k - 3].mean() + j / 2) / 1e3, "reset jump", color="C3", fontsize=11)
    print("reset jump", j)
ax.set_xlabel("time from the H clock flag (µs)"); ax.set_ylabel("kADU relative")
ax.set_title("Horizontal transfer (median): peak-to-peak and reset jump", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/feat_hwave.png"); plt.close(fig)
print("H p2p", np.ptp(w))
