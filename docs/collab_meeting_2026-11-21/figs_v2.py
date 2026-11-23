"""Revised figures (round 2): full-period clock shading, clearer drift and Allan plots, single PCA panel,
sigma_1(tau) fit, decision-tree illustrations, simpler BO results."""
import glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT, OUT, WT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(ROOT, "src"))
from decode_trace import decode_trace

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11,
                     "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
FS = 15e6
which = sys.argv[4:] or ["all"]
def want(k): return "all" in which or k in which


def starts(bit, fl):
    on = (fl >> bit) & 1
    d = np.diff(np.r_[0, on, 0]); return np.where(d == 1)[0]


def oct12(ts):
    f = glob.glob(os.path.join(ROOT, "data/2026-10-12-PA03_PQ04_Packet_Tests", f"trace_*_ch0_101_{ts}_*.meta"))[0]
    m = json.load(open(f)); t = decode_trace(f[:-5] + ".RJ45ADCDATA0")
    return m, t.values.astype(float), t.flags.astype(int)


def cycles(fl, bit, period):
    st = starts(bit, fl)
    return st[np.r_[np.diff(st), 0] == period]          # transfers followed by another one exactly one period later


def spike_starts(v, st, look):
    """True start of each transfer: the short spike (largest |2nd difference|) in the `look` samples before the flag rises."""
    out = []
    for s0 in st:
        seg = v[max(s0 - look, 1):s0 + 1]
        d2 = np.abs(np.diff(seg, 2))
        out.append(max(s0 - look, 1) + 1 + int(np.argmax(d2)))
    return np.array(out)


def mean_cycle(v, fl, bit, period):
    st = spike_starts(v, cycles(fl, bit, period), period // 3)
    W = np.array([v[s:s + period] for s in st]); W = W - W[:, :5].mean(1, keepdims=True)
    return np.median(W, 0)


if want("clocks"):
    m0, v0, f0 = oct12("20261012_185604"); mV, vV, fV = oct12("20261012_204040"); mH, vH, fH = oct12("20261012_205952")
    for bit, per, n, col, name, lab in [(7, 5404, 3, "C0", "v", "Vertical"), (6, 1894, 6, "C1", "h", "Horizontal")]:
        allst = starts(bit, f0)
        allst = allst[allst >= cycles(f0, bit, per)[0]][:n + 1]     # consecutive transfers, whatever their spacing
        allst = spike_starts(v0, allst, per // 3)                  # move each boundary to the spike that starts the transfer
        st = allst[:n]; ends = allst[1:n + 1]
        a, b = max(st[0] - per // 12, 0), ends[-1] + per // 12
        scale = 1e3 if bit == 7 else 1e6
        t = (np.arange(a, b) - a) / FS * scale
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, v0[a:b] / 1e3, lw=0.7, color=col)
        for i, (s0, e0) in enumerate(zip(st, ends)):
            ax.axvspan((s0 - a) / FS * scale, (e0 - a) / FS * scale, color=col, alpha=0.10 if i % 2 == 0 else 0.24, lw=0)
        ax.set_xlabel("time (ms)" if bit == 7 else "time (µs)"); ax.set_ylabel("ADC value (kADU)")
        ax.set_title(f"{lab} transfers in the raw trace (one band = one transfer)", fontsize=13)
        fig.tight_layout(); fig.savefig(f"{OUT}/{name}_zoom.png"); plt.close(fig)
        vB, fB, mB = (vV, fV, mV) if bit == 7 else (vH, fH, mH)
        w0, w1 = mean_cycle(v0, f0, bit, per), mean_cycle(vB, fB, bit, per)
        tt = np.arange(per) / FS * 1e6
        fig, ax = plt.subplots(figsize=(7, 4))
        if bit == 7:
            ax.plot(tt, w0 / 1e3, color=col, label=f"nominal (V low {m0['V1_A_L']} V, TG low {m0['TG_A_L']} V)")
            ax.plot(tt, w1 / 1e3, color="C3", label=f"V low {mB['V1_A_L']} V, TG low {mB['TG_A_L']} V")
        else:
            ax.plot(tt, w0 / 1e3, color=col, label=f"nominal (H low {m0['H1_A_L']} V)")
            ax.plot(tt, w1 / 1e3, color="C3", label=f"H low {mB['H1_A_L']} V")
        ax.set_xlabel(f"time within one {lab.lower()} transfer (µs)"); ax.set_ylabel("kADU relative")
        ax.set_title(f"Average {lab.lower()}-transfer waveform, two settings")
        ax.legend(loc="best", fontsize=10); fig.tight_layout(); fig.savefig(f"{OUT}/{name}_setting.png"); plt.close(fig)
        print(name, "period us %.1f" % (per / FS * 1e6), "p2p nominal/changed kADU %.1f %.1f" % (np.ptp(w0) / 1e3, np.ptp(w1) / 1e3))

z = np.load(os.path.join(ROOT, "processed/L1_23/traces/ref1_acm101_ch0.npz"))
if want("drift"):
    v = z["values"].astype(float); fl = z["flags"].astype(int)
    ps = starts(0, fl); ss = starts(1, fl)
    gap = np.where(np.diff(ps) > 1000)[0]
    p_first, p_last = ps[gap[1] + 1], ps[gap[2]]
    pst = ps[(ps >= p_first) & (ps <= p_last)]
    sst = ss[(ss > p_first) & (ss < p_last + 400)]
    n = min(len(pst), len(sst))
    ped = np.array([v[s:s + 150].mean() for s in pst[:n]]); sig = np.array([v[s:s + 150].mean() for s in sst[:n]])
    cds = ped - sig; k = np.arange(n)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
    a1.plot(k, (ped - ped[-1]) / 1e3, color="C0"); a1.set_ylabel("baseline\n(kADU)")
    a1.set_title("One pixel, 500 skips")
    a1.text(250, -2.5, "baseline (pedestal-window level) changes by ≈3000 ADU", ha="center", fontsize=11, color="C0")
    a2.plot(k[3:], cds[3:] - np.median(cds), color="C3", lw=0.7); a2.set_ylabel("CDS\n(ADU)")
    a2.set_ylim(-35, 35)
    a2.text(250, 25, "CDS = pedestal − signal: flat, only noise", ha="center", fontsize=11, color="C3")
    a2.set_xlabel("skip number")
    fig.tight_layout(); fig.savefig(f"{OUT}/skips_drift.png"); plt.close(fig)

if want("allan"):
    X = []
    for f in sorted(glob.glob(os.path.join(ROOT, "processed/L1_*/traces/ref*_acm101_ch0.npz"))):
        zz = np.load(f)
        if "cds" in zz.files and zz["cds"].shape == (10, 500):
            X.append(np.asarray(zz["cds"], float)[[1, 2, 3, 4, 6, 7, 8, 9], 20:])
    X = np.array(X); n = X.shape[2]; k = np.arange(n)
    A = X.reshape(-1, n); p = np.polyfit(k, A.T, 1)
    D = A - (np.outer(p[0], k) + p[1][:, None])                 # remove a straight-line drift from each pixel
    s1 = np.std(np.diff(A, axis=1)) / np.sqrt(2)
    Ns = np.array([1, 2, 4, 8, 16, 30, 60, 120]); al = []
    for N in Ns:
        nb = n // N; Y = D[:, :nb * N].reshape(D.shape[0], nb, N).mean(2)
        al.append(np.sqrt(0.5 * np.mean(np.diff(Y, axis=1) ** 2)))
    img = []
    for dd in sorted(glob.glob(os.path.join(ROOT, "processed/L1_*"))):
        L = json.load(open(dd + "/labels.json"))
        img += [l["sigma_e"] * l["gain_adu_per_e"] for im in L["images"] if im["board"] == 101 for l in im["labels"] if l["channel"] == 0 and not l["flags"]]
    im500 = float(np.median(img))
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    NN = np.array([1, 500])
    ax.loglog(NN, s1 / np.sqrt(NN), "--", color="gray", label=f"pure white noise: σ₁/√N, σ₁ = {s1:.1f} ADU")
    ax.loglog(Ns, al, "o-", color="C0", label=f"measured in the trace ({X.shape[0]} traces)")
    ax.loglog([500], [im500], "*", color="C3", ms=16, label=f"measured in the 500-skip images: {im500:.2f} ADU")
    ax.set_xlabel("N, number of skips averaged"); ax.set_ylabel("noise of the N-skip average (ADU)")
    ax.set_title("Averaging more skips keeps reducing the noise")
    ax.legend(fontsize=10); fig.tight_layout(); fig.savefig(f"{OUT}/allan.png"); plt.close(fig)
    print("traces", X.shape, "sigma1 %.2f; allan" % s1, np.round(al, 3), "image500 %.3f white500 %.3f" % (im500, s1 / np.sqrt(500)))

if want("pca"):
    im = Image.open(os.path.join(OUT, "oct12_feature_space.png")); W, H = im.size
    im.crop((0, 0, W // 2, H)).save(os.path.join(OUT, "pca_ch0.png"))

if want("sigtau"):
    d = pd.read_csv(os.path.join(WT, "amp_phys/results/trap_from_trace_v3/trap_from_trace.csv"))
    r = d[d.label == "PA03 101 ch1"].iloc[0]
    taus = np.array([2.0, 3.0, 4.0, 6.0, 8.0, 9.3])
    s1 = np.array([r[f"sigma_1 tau={t}us (ADU)"] for t in taus])
    f = np.logspace(0, np.log10(7.5e6), 300000)
    dt, T = 13.8e-6, 32.6e-6
    def I(tau):
        h = 4 * np.sin(np.pi * f * dt) ** 2 * np.sinc(f * tau) ** 2 * 2 * np.sin(np.pi * f * T) ** 2
        return np.trapezoid(h, f), np.trapezoid(h / f, f)
    Iw, If = np.array([I(t * 1e-6) for t in taus]).T
    A = np.c_[Iw, If] / (s1 ** 2)[:, None]                     # relative residuals
    Sw, K = np.linalg.lstsq(A, np.ones(len(taus)), rcond=None)[0]
    tt = np.linspace(1.8, 10, 60); Iwt, Ift = np.array([I(t * 1e-6) for t in tt]).T
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(taus, s1 ** 2, "ko", ms=7, label="measured σ₁² (one trace)", zorder=3)
    ax.plot(tt, Sw * Iwt + K * Ift, "k-", label="fit: white + 1/f")
    ax.plot(tt, Sw * Iwt, "--", color="C0", label="white part (falls as 1/τ)")
    ax.plot(tt, K * Ift, "--", color="C3", label=f"1/f part (K = {K:.1f} ADU²)")
    ax.set_xlabel("window length τ (µs)"); ax.set_ylabel("σ₁² (ADU²)"); ax.set_ylim(0, None)
    ax.set_title("PA03 ch1: noise vs window length")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/sigtau.png"); plt.close(fig)
    print("sigtau fit K %.2f (csv %.2f), Sw %.3e" % (K, r["1/f K (ADU^2)"], Sw))

if want("trees"):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 10, 60)); y = 0.15 + 0.03 * np.sin(x / 1.6) + 0.004 * x + rng.normal(0, 0.006, len(x))
    xx = np.linspace(0, 10, 500)
    dt = DecisionTreeRegressor(max_depth=2).fit(x[:, None], y)
    fig, a1 = plt.subplots(figsize=(6, 3.2))
    a1.plot(x, y, "o", color="0.5", ms=4, label="training images"); a1.plot(xx, dt.predict(xx[:, None]), color="C0", lw=2.5, label="one tree (4 leaves)")
    for thr in dt.tree_.threshold[dt.tree_.threshold > -2]:
        a1.axvline(thr, color="C0", ls=":", lw=1)
    a1.set_title("One tree's prediction: one step per leaf", fontsize=13); a1.set_xlabel("one input"); a1.set_ylabel("label")
    a1.legend(loc="upper right", fontsize=10); fig.tight_layout(); fig.savefig(f"{OUT}/tree_one.png"); plt.close(fig)
    et = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, random_state=0).fit(x[:, None], y)
    fig, a2 = plt.subplots(figsize=(6, 3.2))
    for est in et.estimators_[:6]:
        a2.plot(xx, est.predict(xx[:, None]), color="C1", lw=0.8, alpha=0.6)
    a2.plot(x, y, "o", color="0.5", ms=4); a2.plot(xx, et.predict(xx[:, None]), color="C3", lw=2.5, label="average of 400 random trees")
    a2.plot([], [], color="C1", lw=0.8, label="6 of the random trees")
    a2.set_title("Many random trees, averaged", fontsize=13); a2.set_xlabel("one input"); a2.set_ylabel("label")
    a2.legend(loc="upper right", fontsize=10); fig.tight_layout(); fig.savefig(f"{OUT}/tree_forest.png"); plt.close(fig)

if want("results"):
    labs = ["random order", "by hand\n(hill-climb)", "BO, new amplifier's\nimages only", "BO, multi-amplifier\nmodel + EDT gate"]
    vals = [38.5, 26.8, 4.0, 1.8]; wasted = [5.7, 3.4, 0.5, 0.0]; cols = ["0.6", "0.6", "C0", "C3"]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.bar(range(4), vals, color=cols, edgecolor="k", width=0.6)
    for i, (v, w) in enumerate(zip(vals, wasted)):
        ax.text(i, v + 0.8, f"{v}", ha="center", fontsize=14)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=11)
    ax.set_ylabel("images needed"); ax.set_ylim(0, 44)
    ax.set_title("Images to get within 2 % of the lowest dark charge")
    fig.tight_layout(); fig.savefig(f"{OUT}/results.png"); plt.close(fig)
