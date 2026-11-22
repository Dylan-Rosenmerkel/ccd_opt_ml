"""v7 figures: every side-by-side plot as its own image; P(good image) plot; GP kernel and multi-amplifier training steps.
usage: python fig_v7.py ROOT OUT PGOOD_CSV [which ...]"""
import glob, json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT, OUT, PG = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(ROOT, "src"))
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11,
                     "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
FS = 15e6
which = sys.argv[4:] or ["all"]
def want(k): return "all" in which or k in which
def toy(x): return 0.45 * np.cos(1.6 * x) + 0.08 * (x - 2.6) ** 2

# ---------------------------------------------------------------- good vs bad image: two separate images
if want("usable"):
    from astropy.io import fits
    def load(pk, ch):
        f = sorted(glob.glob(os.path.join(ROOT, f"processed/{pk}/fits/*_101_*.fits")))[0]
        X = np.array(fits.open(f)[ch + 1].data, float)
        L = json.load(open(os.path.join(ROOT, f"processed/{pk}/labels.json")))
        lab = [l for im in L["images"] if im["board"] == 101 for l in im["labels"] if l["channel"] == ch][0]
        return X, lab
    for (pk, ch, name, col, fn, note) in [("L1_23", 0, "good image (PA03 ch0, nominal)", "C0", "img_good.png", "fit OK"),
                                         ("L1_03", 2, "bad image (PA03 ch2, H delays halved)", "C3", "img_bad.png", "χ²/ndf = 11")]:
        X, lab = load(pk, ch)
        act = X[:, 10:860] / lab["gain_adu_per_e"]; act = act - np.median(act)
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(6, 5.6), gridspec_kw={"height_ratios": [1, 1.25]})
        a0.imshow(act[:, :400], aspect="auto", cmap="viridis", vmin=-0.6, vmax=2.5, interpolation="nearest")
        a0.set_title(name, fontsize=13); a0.grid(False); a0.set_xlabel("column"); a0.set_ylabel("row")
        a1.hist(act.ravel(), bins=np.arange(-1, 4, 0.04), color=col)
        a1.set_yscale("log"); a1.set_xlabel("pixel value (e⁻)"); a1.set_ylabel("pixels")
        a1.set_title(f"σe = {lab['sigma_e']:.2f} e⁻, λ = {lab['lambda_e_per_bin']:.2f} e⁻/pix  ({note})", fontsize=12)
        fig.tight_layout(); fig.savefig(f"{OUT}/{fn}"); plt.close(fig)

# ---------------------------------------------------------------- amplifiers apart: two separate images
if want("amps"):
    from decode_trace import decode_trace
    f1, ax = plt.subplots(figsize=(7.4, 4.3)); decay = {}
    for ch in range(4):
        metas = sorted(glob.glob(os.path.join(ROOT, f"data/2026-10-08-PA03_PQ04_traces/trace_*_ch{ch}_101_*.meta")))
        decay[ch] = []
        for i, meta in enumerate(metas):
            try:
                t = decode_trace(meta[:-5] + f".RJ45ADCDATA{ch}"); cyc, P, pix = t.skip_cycles()
            except Exception:
                continue
            w = np.median(cyc.astype(float), 0); base = np.median(w[20:120])
            k0 = int(np.argmax(w)); amp = w[k0] - base; tail = w[k0:] - base
            kd = int(np.argmax(tail < amp / np.e)); frac = (tail[kd - 1] - amp / np.e) / (tail[kd - 1] - tail[kd])
            decay[ch].append((kd - 1 + frac) / FS * 1e9)
            seg = w[k0 - 10:k0 + 60] - base
            ax.plot((np.arange(len(seg)) - 10) / FS * 1e6, seg / 1e3, color=f"C{ch}", lw=1.2, alpha=0.8, label=f"ch{ch}" if i == 0 else None)
    ax.set_xlabel("time after the reset-pulse peak (µs)"); ax.set_ylabel("ADU above baseline (k)")
    ax.set_title("Reset pulse, 4 amplifiers × 5 repeats (same voltages)"); ax.legend(title="amplifier")
    f1.tight_layout(); f1.savefig(f"{OUT}/amps_pulse.png"); plt.close(f1)
    f2, bx = plt.subplots(figsize=(4.6, 4.3))
    for ch in range(4):
        y = np.array(decay[ch]); bx.plot(np.full(len(y), ch) + np.linspace(-0.12, 0.12, len(y)), y, "o", color=f"C{ch}")
    bx.set_xticks(range(4)); bx.set_xticklabels([f"ch{c}" for c in range(4)])
    bx.set_ylabel("pulse decay time to 1/e (ns)"); bx.set_title("One number: repeats vs amplifiers")
    f2.tight_layout(); f2.savefig(f"{OUT}/amps_decay.png"); plt.close(f2)

# ---------------------------------------------------------------- GP and EI: two separate images
if want("gp"):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    from scipy.stats import norm
    x = np.linspace(0, 5, 400)[:, None]
    X = np.array([0.4, 1.3, 2.1, 3.6, 4.6])[:, None]; y = toy(X.ravel())
    gp = GaussianProcessRegressor(ConstantKernel(0.3) * Matern(0.9, nu=2.5), alpha=1e-4, optimizer=None).fit(X, y - y.mean())
    m, s = gp.predict(x, return_std=True); m = m + y.mean()
    b = y.min(); z = (b - m) / np.maximum(s, 1e-9); ei = (b - m) * norm.cdf(z) + s * norm.pdf(z)
    fig, a1 = plt.subplots(figsize=(7, 3.4))
    a1.plot(x, m, color="C0", label="GP mean m(x)"); a1.fill_between(x.ravel(), m - 2 * s, m + 2 * s, color="C0", alpha=0.2, label="± 2 s(x)")
    a1.plot(X, y, "o", color="C3", ms=7, label="images so far"); a1.axhline(b, color="C3", ls=":", lw=1)
    a1.text(0.05, b - 0.13, "best so far b", color="C3", fontsize=11)
    a1.set_ylabel("J"); a1.set_xlabel("one voltage (standardised)"); a1.set_title("1. The GP's guess m(x) and uncertainty s(x)")
    a1.legend(fontsize=9, loc="upper center", ncol=3)
    fig.tight_layout(); fig.savefig(f"{OUT}/bo_gp.png"); plt.close(fig)
    fig, a2 = plt.subplots(figsize=(7, 2.9))
    a2.plot(x, ei, color="C2"); xi = x.ravel()[np.argmax(ei)]; a2.axvline(xi, color="C2", ls="--", lw=1)
    a2.text(xi + 0.05, ei.max() * 0.8, "next image", color="C2", fontsize=11)
    a2.set_ylabel("EI(x)"); a2.set_xlabel("one voltage (standardised)"); a2.set_title("2. Expected improvement: image where it is largest")
    fig.tight_layout(); fig.savefig(f"{OUT}/bo_ei.png"); plt.close(fig)
    # what the kernel says: correlation of J at two settings vs their distance
    r = np.linspace(0, 4, 200)
    mat = lambda r: (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for l, c in [(0.5, "C3"), (1.5, "C0")]:
        ax.plot(r, mat(r / l), color=c, lw=2, label=f"ℓ = {l}: {'J changes fast' if l < 1 else 'J changes slowly'}")
    ax.set_xlabel("distance between two settings in one voltage"); ax.set_ylabel("correlation of their J")
    ax.set_title("What the GP assumes: close settings, similar J"); ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/gp_kernel.png"); plt.close(fig)

# ---------------------------------------------------------------- multi-amplifier: training steps and the new amplifier
if want("multi"):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    xx = np.linspace(0, 5, 300); f = toy(xx); rng = np.random.default_rng(3)
    lv = {"A": 0.55, "B": 0.0}; pts = {}
    for i, a in enumerate(lv):
        xs = np.sort(rng.uniform(0, 5, 8)); pts[a] = (xs, toy(xs) + lv[a] + 0.05 * np.sin(3.1 * xs + i) + rng.normal(0, 0.03, 8))
    Xa = np.concatenate([pts[a][0] for a in lv]); ya = np.concatenate([pts[a][1] for a in lv])
    gp = GaussianProcessRegressor(ConstantKernel(0.25) * Matern(1.2, nu=2.5) + WhiteKernel(0.08), optimizer=None).fit(Xa[:, None], ya - ya.mean())
    m, s = gp.predict(xx[:, None], return_std=True); m = m + ya.mean()
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.plot(Xa, ya, "o", color="0.45", ms=6, label="images of amps A and B,\nlabels ignored")
    ax.plot(xx, m, color="k", lw=2, label="one GP fit"); ax.fill_between(xx, m - 2 * s, m + 2 * s, color="0.5", alpha=0.15)
    ax.set_title("Step 1: pool all amplifiers\n→ the shape (length scales ℓⱼ)", fontsize=13); ax.set_xlabel("one voltage (standardised)"); ax.set_ylabel("J")
    ax.legend(fontsize=9, loc="upper center"); ax.set_ylim(-0.6, 1.9)
    fig.tight_layout(); fig.savefig(f"{OUT}/mt_step1.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for i, a in enumerate(lv):
        ax.plot(*pts[a], "o", color=f"C{i}", ms=6); ax.plot(xx, f + lv[a], color=f"C{i}", lw=2, label=f"amp {a}: shape + level c_{a}")
    ax.annotate("", xy=(4.4, toy(4.4) + lv["A"]), xytext=(4.4, toy(4.4) + lv["B"]), arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text(3.45, toy(4.4) + 0.22, "levels\ndiffer", fontsize=10)
    ax.set_title("Step 2: keep the shape, fit how far\nthe levels differ (c²) and how alike (ρ)", fontsize=13)
    ax.set_xlabel("one voltage (standardised)"); ax.set_ylabel("J"); ax.legend(fontsize=9, loc="upper center"); ax.set_ylim(-0.6, 1.9)
    fig.tight_layout(); fig.savefig(f"{OUT}/mt_step2.png"); plt.close(fig)
    cnew, xnew = 0.3, 3.3
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.plot(xx, f + 0.25, color="0.5", lw=1.5, label="before any image: shape only")
    ax.fill_between(xx, f + 0.25 - 0.6, f + 0.25 + 0.6, color="0.5", alpha=0.15)
    ax.plot(xx, f + cnew, color="C2", lw=2, label="after one image: level fixed")
    ax.fill_between(xx, f + cnew - 0.12, f + cnew + 0.12, color="C2", alpha=0.2)
    ax.plot([xnew], [toy(xnew) + cnew], "o", color="C2", ms=9)
    imin = np.argmin(f); ax.axvline(xx[imin], color="C3", ls=":", lw=1.2); ax.text(xx[imin] + 0.08, -0.5, "best setting already known", color="C3", fontsize=10)
    ax.set_title("New amplifier C:\none image fixes its level", fontsize=13); ax.set_xlabel("one voltage (standardised)"); ax.set_ylabel("J")
    ax.legend(fontsize=9, loc="upper center"); ax.set_ylim(-0.6, 1.9)
    fig.tight_layout(); fig.savefig(f"{OUT}/mt_new.png"); plt.close(fig)

# ---------------------------------------------------------------- noise: EDT only (main) and with the formula (backup)
if want("noise"):
    for fn, lab, v, cols in [("noise_bars_v7.png", ["EDT,\nvoltages only", "EDT,\ntrace + voltages"], [18.2, 8.6], ["0.6", "C3"]),
                             ("noise_formula.png", ["EDT,\nvoltages only", "formula\nσ₅₀₀ ÷ gain\n(no ML)", "EDT,\ntrace + voltages"], [18.2, 11.9, 8.6], ["0.6", "C0", "C3"])]:
        fig, ax = plt.subplots(figsize=(1.9 * len(v) + 1.2, 4.2))
        b = ax.bar(range(len(v)), v, color=cols, edgecolor="k", width=0.6)
        for r in b: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.3, f"{r.get_height():.1f} %", ha="center", fontsize=12)
        ax.set_xticks(range(len(v))); ax.set_xticklabels(lab, fontsize=11); ax.set_ylim(0, 21)
        ax.set_ylabel("typical error, new amplifier (%)"); ax.set_title("Image noise σe", fontsize=13)
        fig.tight_layout(); fig.savefig(f"{OUT}/{fn}"); plt.close(fig)

# ---------------------------------------------------------------- P(good image) on the sealed test
if want("pgood"):
    d = pd.read_csv(PG)
    fig, ax = plt.subplots(figsize=(6.4, 5.0)); rng = np.random.default_rng(0)
    for j, col in enumerate(["p_v", "p_vt"]):
        g = d[d.ok]; b = d[d.kind.eq("broken on all")]
        ax.plot(j - 0.17 + rng.uniform(-0.09, 0.09, len(g)), g[col], "o", color="C0", ms=5, alpha=0.6, label="good images (67)" if j == 0 else None)
        ax.plot(j + 0.17 + rng.uniform(-0.09, 0.09, len(b)), b[col], "o", color="C3", ms=6, alpha=0.85,
                label="bad: setting broken on all amplifiers (18)" if j == 0 else None)
    ax.axhline(0.5, color="k", ls="--", lw=1); ax.text(1.42, 0.52, "gate", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["voltages only", "voltages + trace"]); ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel("predicted P(good image)"); ax.set_ylim(0, 1.02)
    ax.set_title("Sealed test: new settings, new amplifier", fontsize=13); ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    fig.tight_layout(); fig.savefig(f"{OUT}/pgood.png"); plt.close(fig)
