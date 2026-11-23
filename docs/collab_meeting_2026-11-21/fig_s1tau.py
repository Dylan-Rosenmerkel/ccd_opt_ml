"""How sigma_1(tau) is measured: two skip cycles, and the LAST tau of each 10 us window (amp_phys/trap_from_trace.sigmas).
usage: python fig_s1tau.py AMP_PHYS_DIR TRACE.npz OUT"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[1])
from trace_anatomy import load
plt.rcParams.update({"font.size": 13, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
FS = 15e6
v, fl = load(sys.argv[2]); OUT = sys.argv[3]
def runs(bit):
    on = ((fl >> bit) & 1).astype(int); s = np.flatnonzero(np.diff(on) == 1) + 1; e = np.flatnonzero(np.diff(on) == -1) + 1
    e = e[e > s[0]]; m = min(len(s), len(e)); return s[:m], e[:m]
ps, pe = runs(0); ss, se = runs(1)
k0 = 300                                            # a skip well inside a pixel
p_s, p_e = ps[k0], pe[k0]
j = np.searchsorted(ss, p_e); s_s, s_e = ss[j], se[j]
P = ps[k0 + 1] - ps[k0]
a, b = p_s - 30, p_s + 2 * P + 30
t = lambda i: (i - a) / FS * 1e6
base = v[p_s:p_e].mean()
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(t(np.arange(a, b)), (v[a:b] - base) / 1e3, color="k", lw=0.9)
taus = [(2.0, "C2"), (4.0, "C1"), (9.3, "C4")]
for n in range(2):
    for (w0, w1, col, lab) in [(ps[k0 + n], pe[k0 + n], "C0", "pedestal"), (ss[j + n], se[j + n], "C3", "signal")]:
        ax.axvspan(t(w0), t(w1), color=col, alpha=0.12, lw=0)
        if n == 0:
            ax.text((t(w0) + t(w1)) / 2, 1.2, f"{lab}\nwindow", ha="center", va="bottom", fontsize=10)
        for i, (tau, c) in enumerate(taus):
            y = -9 - 2.2 * i
            ax.plot([t(w1) - tau, t(w1)], [y, y], color=c, lw=4, solid_capstyle="butt")
for i, (tau, c) in enumerate(taus):
    ax.text(t(se[j + 1]) + 0.8, -9 - 2.2 * i, f"last τ = {tau:g} µs", color=c, fontsize=10, va="center")
ax.set_ylim(-17, 16); ax.set_xlim(t(a), t(b) + 9)
ax.set_xlabel("time (µs)"); ax.set_ylabel("kADU relative to pedestal")
ax.set_title("σ₁(τ): use only the last τ of each 10 µs window", fontsize=13)
ax.text(0.01, 0.97, "CDS(τ) = mean(last τ of pedestal) − mean(last τ of signal)\n"
        "σ₁(τ) = spread of [CDS(τ) of a skip − CDS(τ) of the next skip] ÷ √2\n"
        "6 window lengths: τ = 2, 3, 4, 6, 8, 9.3 µs (3 shown)",
        transform=ax.transAxes, fontsize=9.5, va="top", bbox=dict(boxstyle="round", fc="white", ec="0.6"))
fig.tight_layout(); fig.savefig(f"{OUT}/sigma1_tau.png"); plt.close(fig)
