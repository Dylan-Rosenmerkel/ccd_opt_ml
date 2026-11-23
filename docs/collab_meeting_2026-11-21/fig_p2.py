"""Tier P figures, round 2: jump-robust vs plain CDS rms, reset level vs CDS, null CDS.
Definitions as in src/trace_features.py / trace_features_extra.py (med_rv on the detrended CDS of skips 10-499).
usage: python fig_p2.py AMP_PHYS_DIR NOMINAL_TRACE.npz JUMPY_TRACE.npz JUMPY_PIXEL OUT"""
import glob, math, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[1])
from trace_anatomy import load
plt.rcParams.update({"font.size": 13, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
FS = 15e6
NOM, JMP, JPIX, OUT = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]


def med_rv(e):
    a = np.abs(e); m = np.median(np.vstack([a[:-2], a[1:-1], a[2:]]), 0); n = len(a)
    return math.pi / (6 - 4 * math.sqrt(3) + math.pi) * n / (n - 2) * np.sum(m ** 2)


def detr(c):
    c = c[10:]; c = c[np.isfinite(c)]; n = np.arange(len(c)); return c - np.polyval(np.polyfit(n, c, 4), n)


# ---- 1: the pixel with the largest plain/robust difference on PA03
e = detr(np.load(JMP, allow_pickle=True)["cds"][JPIX])
rms, rob = e.std(), math.sqrt(med_rv(e) / len(e))
n = np.arange(10, 10 + len(e)); out = np.abs(e) > 4 * rob
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(n[~out], e[~out], ".", ms=3, color="C0", label="detrended CDS of each skip")
ax.plot(n[out], e[out], "o", ms=6, mfc="none", color="C3", label=f"outlying skips ({out.sum()})")
for y, c, lab in [(rms, "k", "plain rms: every skip counts"), (rob, "C2", "jump-robust: outliers ignored")]:
    ax.axhline(y, color=c, lw=1.5, ls="-" if c == "k" else "--", label=lab); ax.axhline(-y, color=c, lw=1.5, ls="-" if c == "k" else "--")
ax.set_xlabel("skip number"); ax.set_ylabel("CDS − slow trend (ADU)"); ax.legend(fontsize=9, loc="upper center", ncol=2)
ax.set_ylim(min(-5.5 * rob, e.min() * 1.15), 8.5 * rob)
ax.set_title("PA03 pixel with the most outliers: plain 10 % above robust", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/p_jump.png"); plt.close(fig)

# ---- 2: plain / robust over every PA03 pixel
R = []
for f in sorted(glob.glob("/Users/dylanrosenmerkel/Documents/Software/ccd_opt_ml/processed/L1_*/traces/*acm101_ch[013].npz")):
    try:
        C = np.load(f, allow_pickle=True)["cds"]
    except Exception:
        continue
    for row in C:
        if np.isfinite(row[10:]).sum() < 400:
            continue
        x = detr(row); r_ = math.sqrt(med_rv(x) / len(x))
        if r_ > 2:
            R.append(x.std() / r_)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.hist(R, bins=np.linspace(0.85, 1.15, 61), color="C0")
ax.axvline(1, color="k", ls=":", lw=1)
ax.set_xlabel("plain CDS rms ÷ jump-robust σ"); ax.set_ylabel("pixels")
ax.set_title(f"All {len(R)} PA03 pixels: the two agree (no jumps)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/p_jumpratio.png"); plt.close(fig)

# ---- pedestal and signal windows of the nominal trace (flag bits 0 and 1)
v, fl = load(NOM)
def runs(bit):
    on = ((fl >> bit) & 1).astype(int); s = np.flatnonzero(np.diff(on) == 1) + 1; t = np.flatnonzero(np.diff(on) == -1) + 1
    t = t[t > s[0]]; m = min(len(s), len(t)); return s[:m], t[:m]
ps, pe = runs(0); ss, se = runs(1)
pairs = []
j = 0
for a, b in zip(ps, pe):
    while j < len(ss) and ss[j] < b:
        j += 1
    if j < len(ss) and ss[j] - b < 400:
        pairs.append((a, b, ss[j], se[j]))
pairs = np.array(pairs)
brk = np.flatnonzero(np.diff(pairs[:, 0]) > 1500) + 1
pix = [g for g in np.split(np.arange(len(pairs)), brk) if len(g) > 400][1]
pr = pairs[pix][10:]
ped = np.array([v[a:b].mean() for a, b, _, _ in pr]); sig = np.array([v[c:d].mean() for _, _, c, d in pr])
cds = ped - sig; k = np.arange(10, 10 + len(pr))

# ---- 3: reset level vs CDS, same scale
fig, ax = plt.subplots(figsize=(6.4, 3.9))
pd_ = ped - np.polyval(np.polyfit(k, ped, 4), k)
ax.plot(k, pd_, color="C1", lw=0.8, label="pedestal level after each reset (slow drift removed)")
ax.plot(k, cds - cds.mean(), color="C0", lw=0.8, label="CDS = pedestal − signal")
ax.set_xlabel("skip number"); ax.set_ylabel("ADU"); ax.legend(fontsize=9.5, loc="upper center", ncol=1); ax.set_ylim(-1.6 * np.abs(pd_).max(), 1.9 * np.abs(pd_).max())
ax.set_title("Each reset leaves a new random level; the CDS cancels it", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/p_reset.png"); plt.close(fig)

# ---- 4: null CDS: one pedestal window split in halves
a, b, c, d = pr[200]
w = v[a:b]; h = len(w) // 2; t = np.arange(len(w)) / FS * 1e6
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(t, w - w.mean(), color="0.4", lw=0.8, label="samples of one pedestal window")
ax.axvspan(t[0], t[h - 1], color="C2", alpha=0.12); ax.axvspan(t[h], t[-1], color="C4", alpha=0.12)
m1, m2 = w[:h].mean() - w.mean(), w[h:2 * h].mean() - w.mean()
ax.plot([t[0], t[h - 1]], [m1, m1], color="C2", lw=2.5, label="mean of first half")
ax.plot([t[h], t[-1]], [m2, m2], color="C4", lw=2.5, label="mean of second half")
ax.set_xlabel("time within the pedestal window (µs)"); ax.set_ylabel("ADU (window mean removed)")
ax.set_title("Null CDS = first half − second half (no charge moved)", fontsize=12); ax.legend(fontsize=9, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/p_null.png"); plt.close(fig)
print("jump pixel rms %.2f robust %.2f outliers %d; pixels %d max ratio %.3f; reset sd %.0f cds sd %.1f" % (rms, rob, out.sum(), len(R), max(R), pd_.std(), cds.std()))
