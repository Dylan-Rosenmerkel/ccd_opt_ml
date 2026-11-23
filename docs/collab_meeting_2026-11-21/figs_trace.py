"""Trace-breakdown figures for the collaboration-meeting deck (plain matplotlib style)."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(ROOT, "src"))
from decode_trace import decode_trace

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11,
                     "figure.dpi": 100, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
FS = 15e6


def segs(bit, flags, length=None):
    on = (flags >> bit) & 1
    d = np.diff(np.r_[0, on, 0]); st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
    if length is not None:
        keep = (en - st) == length; st, en = st[keep], en[keep]
    return st, en


def oct12(ts):
    f = glob.glob(os.path.join(ROOT, "data/2026-10-12-PA03_PQ04_Packet_Tests", f"trace_*_ch0_101_{ts}_*.meta"))[0]
    m = json.load(open(f)); t = decode_trace(f[:-5] + ".RJ45ADCDATA0")
    return m, t.values.astype(float), t.flags.astype(int)


def mean_wave(v, fl, bit, length, pad=0):
    st, en = segs(bit, fl, length)
    W = np.array([v[s - pad:s + length + pad] for s in st if s - pad >= 0 and s + length + pad <= len(v)])
    W = W - W[:, :5].mean(1, keepdims=True)
    return np.median(W, 0)


m0, v0, f0 = oct12("20261012_185604")      # nominal
mV, vV, fV = oct12("20261012_204040")      # V low rails and TG lowered
mH, vH, fH = oct12("20261012_205952")      # H low rails -0.5 V
print("nominal V1_A_L", m0["V1_A_L"], "TG_A_L", m0["TG_A_L"], "H1_A_L", m0["H1_A_L"])
print("V-changed V1_A_L", mV["V1_A_L"], "TG_A_L", mV["TG_A_L"], " H-changed H1_A_L", mH["H1_A_L"])

# ---- vertical: zoom of the raw trace and the mean waveform at two settings
st, en = segs(7, f0)
a, b = st[0] - 300, st[3] + 300
t = (np.arange(a, b) - a) / FS * 1e3
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, v0[a:b] / 1e3, lw=0.6, color="C0")
for s, e in zip(st[:3], en[:3]):
    ax.axvspan((s - a) / FS * 1e3, (e - a) / FS * 1e3, color="C0", alpha=0.12)
ax.set_xlabel("time (ms)"); ax.set_ylabel("ADC value (kADU)")
ax.set_title("Vertical transfers in the raw trace (shaded)")
fig.tight_layout(); fig.savefig(f"{OUT}/v_zoom.png"); plt.close(fig)

L = 2700
w0, w1 = mean_wave(v0, f0, 7, L), mean_wave(vV, fV, 7, L)
tt = np.arange(L) / FS * 1e6
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(tt, w0 / 1e3, label=f"nominal (V low {m0['V1_A_L']} V, TG {m0['TG_A_L']} V)", color="C0")
ax.plot(tt, w1 / 1e3, label=f"V low {mV['V1_A_L']} V, TG {mV['TG_A_L']} V", color="C3")
ax.set_xlabel("time within one vertical transfer (µs)"); ax.set_ylabel("ADU (relative, k)")
ax.set_title("Mean vertical-transfer waveform, two settings")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(f"{OUT}/v_setting.png"); plt.close(fig)
print("V p2p nominal/changed (kADU): %.1f %.1f" % (np.ptp(w0) / 1e3, np.ptp(w1) / 1e3))

# ---- horizontal
st, en = segs(6, f0, 810)
a, b = st[0] - 150, st[6] + 150
t = (np.arange(a, b) - a) / FS * 1e6
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, v0[a:b] / 1e3, lw=0.7, color="C1")
for s, e in zip(st[:6], en[:6]):
    ax.axvspan((s - a) / FS * 1e6, (e - a) / FS * 1e6, color="C1", alpha=0.12)
ax.set_xlabel("time (µs)"); ax.set_ylabel("ADC value (kADU)")
ax.set_title("Horizontal transfers in the raw trace (shaded)")
fig.tight_layout(); fig.savefig(f"{OUT}/h_zoom.png"); plt.close(fig)

L = 810
h0, h1 = mean_wave(v0, f0, 6, L), mean_wave(vH, fH, 6, L)
tt = np.arange(L) / FS * 1e6
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(tt, h0 / 1e3, label=f"nominal (H low {m0['H1_A_L']} V)", color="C1")
ax.plot(tt, h1 / 1e3, label=f"H low {mH['H1_A_L']} V", color="C3")
ax.set_xlabel("time within one horizontal transfer (µs)"); ax.set_ylabel("ADU (relative, k)")
ax.set_title("Mean horizontal-transfer waveform, two settings")
ax.legend(loc="lower left"); fig.tight_layout(); fig.savefig(f"{OUT}/h_setting.png"); plt.close(fig)
print("H p2p nominal/changed (kADU): %.1f %.1f" % (np.ptp(h0) / 1e3, np.ptp(h1) / 1e3))

# ---- skip region (Nov 3, unbinned trace, nominal)
z = np.load(os.path.join(ROOT, "processed/L1_23/traces/ref1_acm101_ch0.npz"))
v = z["values"].astype(float); fl = z["flags"].astype(int)
ps, pe = segs(0, fl, 150); ss, se = segs(1, fl, 150)
# pixel boundaries: gaps between pedestal windows larger than 2 cycles
gap = np.where(np.diff(ps) > 1000)[0]
p_first = ps[gap[1] + 1]          # second pixel start (first full pixel after a row start)
p_last = ps[gap[2]]
a = p_first - 2000; b = p_first + int(3e-3 * FS)
t = (np.arange(a, b) - a) / FS * 1e3
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, v[a:b] / 1e3, lw=0.5, color="C2")
ax.axvspan(0, (p_first - a) / FS * 1e3, color="C1", alpha=0.12, label="horizontal transfer")
ax.set_xlabel("time (ms)"); ax.set_ylabel("ADC value (kADU)")
ax.set_title("Start of a pixel: the skip cycles begin")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(f"{OUT}/skips_zoom.png"); plt.close(fig)

# per-skip pedestal level and CDS along one pixel
sel = (ps >= p_first) & (ps <= p_last)
pst = ps[sel]
ped = np.array([v[s:s + 150].mean() for s in pst])
sig = np.array([v[s2:s2 + 150].mean() for s2 in ss[(ss > p_first) & (ss < p_last + 400)][:len(pst)]])
n = min(len(ped), len(sig)); ped, sig = ped[:n], sig[:n]
cds = ped - sig
k = np.arange(n)
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(k, (ped - ped[-1]) / 1e3, color="C0", label="pedestal level")
ax1.set_xlabel("skip number"); ax1.set_ylabel("pedestal level − final (kADU)", color="C0")
ax2 = ax1.twinx(); ax2.grid(False)
ax2.plot(k[3:], cds[3:] - np.median(cds), color="C3", lw=0.6, label="CDS = pedestal − signal")
ax2.set_ylabel("CDS − median (ADU)", color="C3")
ax1.set_title("One pixel: the baseline drifts, the CDS stays flat")
fig.tight_layout(); fig.savefig(f"{OUT}/skips_drift.png"); plt.close(fig)
print("pedestal droop over pixel (kADU): %.1f ; CDS rms (ADU): %.1f" % ((ped[0] - ped[-1]) / 1e3, cds[20:].std()))

# ---- two skips with windows
s0 = pst[100]
a, b = s0 - 30, s0 + 2 * 489 + 30
t = (np.arange(a, b) - a) / FS * 1e6
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, (v[a:b] - v[s0:s0 + 150].mean()) / 1e3, color="k", lw=0.9)
for j in range(2):
    p0 = s0 + j * 489
    sp = ps[np.argmin(np.abs(ps - p0))]; sg = ss[np.argmin(np.abs(ss - (p0 + 207)))]
    ax.axvspan((sp - a) / FS * 1e6, (sp + 150 - a) / FS * 1e6, color="C0", alpha=0.2, label="pedestal window (10 µs)" if j == 0 else None)
    ax.axvspan((sg - a) / FS * 1e6, (sg + 150 - a) / FS * 1e6, color="C3", alpha=0.2, label="signal window (10 µs)" if j == 0 else None)
ax.set_xlabel("time (µs)"); ax.set_ylabel("ADU relative to pedestal (k)")
ax.set_title("Two skip cycles (32.6 µs each)")
ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(f"{OUT}/two_skips.png"); plt.close(fig)

# ---- Allan deviation and N-skip averaging from the stored CDS of this trace
C = np.asarray(z["cds"], float)[[1, 2, 3, 4, 6, 7, 8, 9], 20:]
C = C - C.mean(1, keepdims=True)
C_raw = C.copy()
C = C - C.mean(0, keepdims=True)   # remove the pattern common to all pixels (same in every pixel, cancels in images)
Ns = np.array([1, 2, 4, 8, 16, 30, 60, 120, 240])
allan, avg = [], []
for N in Ns:
    nb = C.shape[1] // N
    Y = C[:, :nb * N].reshape(C.shape[0], nb, N).mean(2)
    allan.append(np.sqrt(0.5 * np.mean(np.diff(Y, axis=1) ** 2)))
allan = np.array(allan)
s1 = np.std(np.diff(C, axis=1)) / np.sqrt(2)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
araw=[]
for N in Ns:
    nb = C_raw.shape[1] // N
    Y = C_raw[:, :nb * N].reshape(C_raw.shape[0], nb, N).mean(2)
    araw.append(np.sqrt(0.5 * np.mean(np.diff(Y, axis=1) ** 2)))
ax.loglog(Ns, allan, "o-", color="C0", label="Allan deviation (common pattern removed)")
ax.loglog(Ns, araw, "o:", color="C1", ms=4, label="Allan deviation, raw CDS")
ax.loglog(Ns, s1 / np.sqrt(Ns), "--", color="gray", label=r"white noise: $\sigma_1/\sqrt{N}$")
ax.set_xlabel("N (skips averaged)"); ax.set_ylabel("ADU")
ax.set_title("Noise vs number of skips averaged")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/allan.png"); plt.close(fig)
print("sigma1 (ADU) %.2f ; Allan at N=240 %.3f vs white %.3f" % (s1, allan[-1], s1 / np.sqrt(240)))
