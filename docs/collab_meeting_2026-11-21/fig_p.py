"""The 8 protected trace features (tier P, given to every model), drawn on one nominal trace (PA03 ch0).
Follows the definitions in src/trace_features.py and src/trace_features_extra.py.
usage: python fig_p.py AMP_PHYS_DIR TRACE.npz OUT"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[1])
from trace_anatomy import load, rising
plt.rcParams.update({"font.size": 13, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
FS = 15e6
v, fl = load(sys.argv[2]); OUT = sys.argv[3]
p0 = rising(fl, 0); P = int(np.median(np.diff(p0)))
s0 = int(np.median([s for k in p0[:300] for s in rising(fl[k:k + P], 1)[:1]]))
Z = np.load(sys.argv[2], allow_pickle=True)
cds = [row[np.isfinite(row)] for row in Z["cds"]]                 # the CDS stored by the trace processing: pixels x skips
cyc = np.array([v[k:k + P] for k in p0[50:2000] if k + P <= len(v)])
c = cds[1][10:]; n = np.arange(10, 10 + len(c))
poly = np.polyval(np.polyfit(n, c, 4), n); r = c - poly
ad = np.abs(np.diff(r)); med_rv = np.pi / (6 - 4 * np.sqrt(3) + np.pi) * len(r) / (len(r) - 2) * np.sum(np.median(np.c_[ad[:-2], ad[1:-1], ad[2:]], 1) ** 2)
jr = np.sqrt(med_rv / len(r))

# ---- 1: CDS of one pixel along its skips
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(n, c, ".", ms=3, color="C0", label="CDS of each skip")
ax.plot(n, poly, color="k", lw=1.5, label="slow trend (4th-order polynomial)")
ax.fill_between(n, poly - r.std(), poly + r.std(), color="C0", alpha=0.15, label="± CDS rms (detrended)")
ax.set_xlabel("skip number"); ax.set_ylabel("CDS (ADU)"); ax.legend(fontsize=9, loc="upper right")
ax.set_title("One pixel: CDS of each skip (jump-robust σ: same, ignoring jumps)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/p_cds.png"); plt.close(fig)

# ---- 2: noise of N-skip averages vs N: white noise per skip and Allan deviation
Ns = np.array([1, 2, 5, 10, 25, 50]); res = []
for cc in cds:
    rr = cc[10:] - np.polyval(np.polyfit(np.arange(len(cc[10:])), cc[10:], 4), np.arange(len(cc[10:])))
    res.append([np.std([rr[i:i + N].mean() for i in range(0, len(rr) - N + 1, N)]) for N in Ns])
sN = np.median(res, 0)
A = np.c_[1 / Ns, np.ones(len(Ns))] / sN[:, None] ** 2; sw2, sf2 = np.linalg.lstsq(A, np.ones(len(Ns)), rcond=None)[0]; sw2 = max(sw2, 0); sf2 = max(sf2, 0)
al = []
for cc in cds:
    rr = cc[10:] - np.polyval(np.polyfit(np.arange(len(cc[10:])), cc[10:], 4), np.arange(len(cc[10:])))
    m = np.array([rr[i:i + 50].mean() for i in range(0, len(rr) - 49, 50)]); al.append(np.sqrt(np.mean(np.diff(m) ** 2) / 2))
NN = np.logspace(0, np.log10(60), 100)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.loglog(Ns, sN, "ko", ms=7, label="noise of N-skip averages")
ax.loglog(NN, np.sqrt(sw2 / NN + sf2), "C0-", label="fit σ_w²/N + σ_f²: white noise per skip σ_w")
ax.loglog([50], [np.median(al)], "*", color="C3", ms=14, label="Allan deviation, n = 50")
ax.set_xlabel("N, skips averaged"); ax.set_ylabel("noise (ADU)"); ax.legend(fontsize=9, loc="lower left")
ax.set_title("White noise per skip and Allan deviation", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/p_avg.png"); plt.close(fig)

# ---- 3: pedestal window: reset-level scatter (kTC) and the null CDS; pedestal windows = runs of flag bit 0
on = (fl & 1).astype(int); st = np.flatnonzero(np.diff(on) == 1) + 1; en = np.flatnonzero(np.diff(on) == -1) + 1
en = en[en > st[0]]; m = min(len(st), len(en)); W = [v[a0:b0] for a0, b0 in zip(st[:m], en[:m]) if 100 < b0 - a0 < 200]
L = min(len(w) for w in W); Q = np.array([w[:L] for w in W])
brk = np.flatnonzero(np.diff(st[:len(Q)]) > 1.5 * P) + 1
grp = [g for g in np.split(np.arange(len(Q)), brk) if len(g) > 400][1]
q = Q[grp][10:]; q = q - q.mean(0)                                  # remove the mean window shape
lvl = q.mean(1); lvl_d = lvl - np.polyval(np.polyfit(np.arange(len(lvl)), lvl, 4), np.arange(len(lvl)))
h = L // 2; nul = q[:, :h].mean(1) - q[:, h:2 * h].mean(1)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
b = np.linspace(-500, 500, 201)
ax.hist(lvl_d, bins=b, color="C1", alpha=0.6, label="pedestal level, skip to skip: reset-level scatter")
ax.hist(nul, bins=b, color="C2", alpha=0.6, label="first − second half of one pedestal: null-CDS noise")
ax.set_xlabel("ADU"); ax.set_ylabel("skips"); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title("Pedestal window: each reset sets a new level", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/p_ped.png"); plt.close(fig)

# ---- 4: skip cycle: SW dip and RG pulse heights (÷ the clock swings)
tm = np.median(cyc, 0); base = np.median(tm[s0 + 5:s0 + 145])
t = np.arange(P) / FS * 1e6
kd = int(np.argmin(tm[150:s0])) + 150
mask = np.ones(P, bool); mask[:150] = False; mask[s0:s0 + 150] = False
kp = int(np.argmax(np.where(mask, tm, -np.inf)))
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(t, (tm - base) / 1e3, color="C0", lw=1.2)
ax.axhline(0, color="k", ls=":", lw=1)
ax.annotate("", xy=(t[kd], (tm[kd] - base) / 1e3), xytext=(t[kd], 0), arrowprops=dict(arrowstyle="<->", color="C2", lw=1.5))
ax.text(t[kd] + 0.6, (tm[kd] - base) / 1e3, "SW dip\n÷ SW swing", color="C2", fontsize=11, va="bottom")
ax.annotate("", xy=(t[kp], (tm[kp] - base) / 1e3), xytext=(t[kp], 0), arrowprops=dict(arrowstyle="<->", color="C3", lw=1.5))
ax.text(t[kp] - 0.6, (tm[kp] - base) / 2e3, "RG pulse\n÷ RG swing", color="C3", fontsize=11, ha="right")
ax.set_xlabel("time within one skip (µs)"); ax.set_ylabel("kADU relative to signal level")
ax.set_title("Clock feedthrough per volt (gain-related)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/p_edges.png"); plt.close(fig)
print("cds rms", r.std(), "jump-robust", jr, "sigma_w", np.sqrt(sw2), "allan50", np.median(al), "reset scatter", lvl_d.std(), "null", nul.std(),
      "dip", tm[kd] - base, "pulse", tm[kp] - base)
