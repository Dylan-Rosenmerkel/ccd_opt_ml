"""Remaining figures for the collaboration-meeting deck (plain matplotlib style)."""
import glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT, OUT, WT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(ROOT, "src"))
from decode_trace import decode_trace

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11,
                     "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.3})
FS = 15e6
which = sys.argv[4:] or ["all"]
def want(k): return "all" in which or k in which

# ---------------------------------------------------------------- amplifiers apart (Oct 8: 5 repeats x 4 amps, same voltages)
if want("amps"):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
    decay = {}
    for ch in range(4):
        metas = sorted(glob.glob(os.path.join(ROOT, f"data/2026-10-08-PA03_PQ04_traces/trace_*_ch{ch}_101_*.meta")))
        decay[ch] = []
        for i, meta in enumerate(metas):
            try:
                t = decode_trace(meta[:-5] + f".RJ45ADCDATA{ch}")
                cyc, P, pix = t.skip_cycles()
            except Exception:
                continue
            w = np.median(cyc.astype(float), 0)
            base = np.median(w[20:120])
            k0 = int(np.argmax(w)); amp = w[k0] - base
            tail = w[k0:] - base
            kd = int(np.argmax(tail < amp / np.e))
            frac = (tail[kd - 1] - amp / np.e) / (tail[kd - 1] - tail[kd])   # linear interpolation between samples
            decay[ch].append((kd - 1 + frac) / FS * 1e9)
            seg = w[k0 - 10:k0 + 60] - base
            ax.plot((np.arange(len(seg)) - 10) / FS * 1e6, seg / 1e3, color=f"C{ch}", lw=1.2, alpha=0.8,
                    label=f"ch{ch}" if i == 0 else None)
    ax.set_xlabel("time after the reset-pulse peak (µs)"); ax.set_ylabel("ADU above baseline (k)")
    ax.set_title("Reset pulse, 4 amplifiers × 5 repeats (same voltages)")
    ax.legend(title="amplifier")
    for ch in range(4):
        y = np.array(decay[ch]); bx.plot(np.full(len(y), ch) + np.linspace(-0.12, 0.12, len(y)), y, "o", color=f"C{ch}")
    bx.set_xticks(range(4)); bx.set_xticklabels([f"ch{c}" for c in range(4)])
    bx.set_ylabel("pulse decay time to 1/e (ns)"); bx.set_title("One number: repeats vs amplifiers")
    fig.tight_layout(); fig.savefig(f"{OUT}/amps.png"); plt.close(fig)
    print("decay ns per ch:", {c: np.round(v, 0).tolist() for c, v in decay.items()})

# ---------------------------------------------------------------- transfer function
if want("transfer"):
    f = np.logspace(2, np.log10(7.5e6), 4000)
    dt, tau, T = 13.8e-6, 10e-6, 32.6e-6
    a = 4 * np.sin(np.pi * f * dt) ** 2; b = np.sinc(f * tau) ** 2; c = 2 * np.sin(np.pi * f * T) ** 2
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.loglog(f, b, color="C0", lw=1, label=r"window average: sinc$^2(f\tau)$")
    ax.loglog(f, a * b, color="C1", lw=1, label=r"CDS: $4\sin^2(\pi f\Delta t)\,$sinc$^2(f\tau)$")
    ax.loglog(f, a * b * c, color="k", lw=1.3, label=r"$\sigma_1$: × skip difference $2\sin^2(\pi f T)$")
    ax.set_ylim(1e-6, 20); ax.set_xlim(1e2, 7.5e6)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel(r"$|H(f)|^2$")
    ax.set_title(r"Which noise frequencies reach $\sigma_1$ ($\tau$=10 µs)")
    ax.legend(loc="lower right", fontsize=10); fig.tight_layout(); fig.savefig(f"{OUT}/transfer.png"); plt.close(fig)

# ---------------------------------------------------------------- K table numbers
if want("ktable"):
    d = pd.read_csv(os.path.join(WT, "amp_phys/results/trap_from_trace_v3/trap_from_trace.csv"))
    g = d.groupby("label").agg(n=("file", "size"), K=("1/f K (ADU^2)", "median"), white=("white ASD (nV/rtHz)", "median"),
                               corner=("1/f corner (kHz)", "median"), Nt=("N_t A=0.85 C_ox (eV^-1 cm^-3)", "median"),
                               s9=("sigma_1 tau=9.3us (ADU)", "median"))
    print(g.round(3).to_string())

# ---------------------------------------------------------------- decision tree diagram
if want("tree"):
    fig, ax = plt.subplots(figsize=(8, 4.6)); ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    def box(x, y, txt, fc):
        ax.add_patch(FancyBboxPatch((x - 1.35, y - 0.42), 2.7, 0.84, boxstyle="round,pad=0.05", fc=fc, ec="k", lw=1))
        ax.text(x, y, txt, ha="center", va="center", fontsize=12)
    def arrow(x0, y0, x1, y1, lab):
        ax.annotate("", (x1, y1 + 0.45), (x0, y0 - 0.45), arrowprops=dict(arrowstyle="->", lw=1.2))
        ax.text((x0 + x1) / 2 + (0.25 if x1 > x0 else -0.25), (y0 + y1) / 2, lab, ha="center", va="center", fontsize=11, color="0.3")
    box(5, 5.2, r"$\sigma_{500}$ < 1.9 ADU ?", "#dbe9f6")
    box(2.6, 3.3, "VDD < −19.5 V ?", "#dbe9f6"); box(7.4, 3.3, "K < 3 ADU² ?", "#dbe9f6")
    for x, t in [(1.2, "leaf:\nσe ≈ 0.14 e⁻"), (4.0, "leaf:\nσe ≈ 0.16 e⁻"), (6.0, "leaf:\nσe ≈ 0.18 e⁻"), (8.8, "leaf:\nσe ≈ 0.22 e⁻")]:
        ax.add_patch(FancyBboxPatch((x - 0.85, 0.8), 1.7, 0.95, boxstyle="round,pad=0.05", fc="#e8f4e0", ec="k", lw=1))
        ax.text(x, 1.27, t, ha="center", va="center", fontsize=11)
    arrow(5, 5.2, 2.6, 3.3, "yes"); arrow(5, 5.2, 7.4, 3.3, "no")
    for x0, xs in [(2.6, (1.2, 4.0)), (7.4, (6.0, 8.8))]:
        ax.annotate("", (xs[0], 1.8), (x0, 2.88), arrowprops=dict(arrowstyle="->", lw=1.2)); ax.text((x0 + xs[0]) / 2 - 0.3, 2.35, "yes", color="0.3", fontsize=11)
        ax.annotate("", (xs[1], 1.8), (x0, 2.88), arrowprops=dict(arrowstyle="->", lw=1.2)); ax.text((x0 + xs[1]) / 2 + 0.3, 2.35, "no", color="0.3", fontsize=11)
    ax.text(5, 0.2, "illustration: cut values and leaf values are made up", ha="center", fontsize=10, color="0.4", style="italic")
    fig.tight_layout(); fig.savefig(f"{OUT}/tree.png"); plt.close(fig)

# ---------------------------------------------------------------- noise model bars
if want("noisebars"):
    labs = ["voltages only\n(the standard)", "trace physics\n+ voltages"]
    vals = [18.2, 8.6]; cols = ["0.6", "C3"]
    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    ax.bar(range(2), vals, color=cols, edgecolor="k", width=0.6)
    for i, v in enumerate(vals): ax.text(i, v + 0.3, f"{v} %", ha="center", fontsize=14)
    ax.set_xticks(range(2)); ax.set_xticklabels(labs)
    ax.set_ylabel("typical error on σe (%)"); ax.set_ylim(0, 21)
    ax.set_title("Image noise on a new amplifier")
    fig.tight_layout(); fig.savefig(f"{OUT}/noise_bars.png"); plt.close(fig)

# ---------------------------------------------------------------- usable vs unusable image
if want("usable"):
    from astropy.io import fits
    def load(pk, ch):
        f = sorted(glob.glob(os.path.join(ROOT, f"processed/{pk}/fits/*_101_*.fits")))[0]
        h = fits.open(f); X = np.array(h[ch + 1].data, float)
        L = json.load(open(os.path.join(ROOT, f"processed/{pk}/labels.json")))
        lab = [l for im in L["images"] if im["board"] == 101 for l in im["labels"] if l["channel"] == ch][0]
        return X, lab
    good, lg = load("L1_23", 0); bad, lb = load("L1_03", 2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 5.6), gridspec_kw={"height_ratios": [1, 1.25]})
    for j, (X, lab, name) in enumerate([(good, lg, "usable (Nov 3, ch0)"), (bad, lb, "unusable (Nov 2, ch2)")]):
        g = lab["gain_adu_per_e"]; act = X[:, 10:860] / g
        act = act - np.median(act)
        axs[0, j].imshow(act[:, :400], aspect="auto", cmap="viridis", vmin=-0.6, vmax=2.5, interpolation="nearest")
        axs[0, j].set_title(f"{name}: 16 rows × 400 columns (e⁻)"); axs[0, j].grid(False)
        axs[0, j].set_xlabel("column"); axs[0, j].set_ylabel("row")
        axs[1, j].hist(act.ravel(), bins=np.arange(-1, 4, 0.04), color="C0" if j == 0 else "C3")
        axs[1, j].set_yscale("log"); axs[1, j].set_xlabel("pixel value (e⁻)"); axs[1, j].set_ylabel("pixels")
        fl = "; ".join(lab["flags"]) if lab["flags"] else "fit OK"
        axs[1, j].set_title(f"σe = {lab['sigma_e']:.2f} e⁻, λ = {lab['lambda_e_per_bin']:.2f} e⁻/pix  ({'fit OK' if not lab['flags'] else 'χ²/ndf = 11'})", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/usable.png"); plt.close(fig)

# ---------------------------------------------------------------- dark charge per amplifier
if want("dc"):
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "processed/L1_*"))):
        L = json.load(open(d + "/labels.json"))
        for im in L["images"]:
            if im["board"] != 101: continue
            for l in im["labels"]:
                if not l["flags"] and l.get("lambda_e_per_bin"):
                    rows.append((os.path.basename(d), l["channel"], l["lambda_e_per_bin"]))
    D = pd.DataFrame(rows, columns=["packet", "ch", "lam"])
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for ch in range(4):
        y = D[D.ch == ch].lam.values
        ax.plot(np.full(len(y), ch) + np.random.default_rng(ch).uniform(-0.18, 0.18, len(y)), y, "o", ms=4, alpha=0.7, color=f"C{ch}")
        ax.plot([ch - 0.3, ch + 0.3], [np.median(y)] * 2, "k-", lw=2)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"ch{c}" for c in range(4)])
    ax.set_ylabel("dark charge λ (e⁻/pixel)"); ax.set_yscale("log")
    ax.set_title("Same 43 settings, four amplifiers (Nov 2–3)")
    fig.tight_layout(); fig.savefig(f"{OUT}/dc_amp.png"); plt.close(fig)
    print("median lambda per ch:", D.groupby("ch").lam.median().round(3).to_dict())

# ---------------------------------------------------------------- GP illustration and EI
def toy(x): return 0.45 * np.cos(1.6 * x) + 0.08 * (x - 2.6) ** 2
if want("gp"):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    from scipy.stats import norm
    x = np.linspace(0, 5, 400)[:, None]
    X = np.array([0.4, 1.3, 2.1, 3.6, 4.6])[:, None]; y = toy(X.ravel())
    gp = GaussianProcessRegressor(ConstantKernel(0.3) * Matern(0.9, nu=2.5), alpha=1e-4, optimizer=None, normalize_y=False).fit(X, y - y.mean())
    m, s = gp.predict(x, return_std=True); m = m + y.mean()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, toy(x.ravel()), "k--", lw=1, label="true response (unknown)")
    ax.plot(x, m, color="C0", label="GP mean m(x)")
    ax.fill_between(x.ravel(), m - 2 * s, m + 2 * s, color="C0", alpha=0.2, label="± 2 s(x)")
    ax.plot(X, y, "o", color="C3", ms=8, label="measured images")
    ax.set_xlabel("one voltage (standardised)"); ax.set_ylabel("J  (e.g. ln λ)")
    ax.set_title("Gaussian process: a guess and its uncertainty"); ax.legend(fontsize=10, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{OUT}/gp.png"); plt.close(fig)
    b = y.min(); z = (b - m) / np.maximum(s, 1e-9); ei = (b - m) * norm.cdf(z) + s * norm.pdf(z)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    a1.plot(x, m, color="C0"); a1.fill_between(x.ravel(), m - 2 * s, m + 2 * s, color="C0", alpha=0.2)
    a1.plot(X, y, "o", color="C3", ms=7); a1.axhline(b, color="C3", ls=":", lw=1); a1.text(0.05, b - 0.12, "best so far b", color="C3", fontsize=11)
    a1.set_ylabel("J"); a1.set_title("Expected improvement picks the next image")
    a2.plot(x, ei, color="C2"); xi = x.ravel()[np.argmax(ei)]; a2.axvline(xi, color="C2", ls="--", lw=1)
    a2.text(xi + 0.05, ei.max() * 0.8, "next image", color="C2", fontsize=11)
    a2.set_ylabel("EI(x)"); a2.set_xlabel("one voltage (standardised)")
    fig.tight_layout(); fig.savefig(f"{OUT}/ei.png"); plt.close(fig)

# ---------------------------------------------------------------- multi-amplifier illustration
if want("multi"):
    x = np.linspace(0, 5, 300); f = toy(x)
    rng = np.random.default_rng(3)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for i, (c, lab) in enumerate([(0.9, "amp A"), (0.2, "amp B")]):
        own = 0.08 * np.sin(3.1 * x + i)
        a1.plot(x, f + c + own, color=f"C{i}", label=f"{lab} = shared shape + level c_{lab[-1]}")
        xs = np.sort(rng.uniform(0, 5, 7)); a1.plot(xs, toy(xs) + c + 0.08 * np.sin(3.1 * xs + i) + rng.normal(0, 0.03, 7), "o", color=f"C{i}")
    a1.plot(x, f - 0.4, "k--", lw=1, label="shared shape f(x)")
    a1.set_title("Measured amplifiers: same shape, own level"); a1.set_xlabel("one voltage (standardised)"); a1.set_ylabel("J")
    a1.legend(fontsize=10, loc="lower left")
    cnew = 0.55; xnew = 3.3
    a2.plot(x, f, color="0.5", lw=1.5, label="before any image: shape known, level not")
    a2.fill_between(x, f - 0.7, f + 0.7, color="0.5", alpha=0.15)
    a2.plot(x, f + cnew, color="C2", lw=2, label="after one image: level fixed")
    a2.fill_between(x, f + cnew - 0.15, f + cnew + 0.15, color="C2", alpha=0.2)
    a2.plot([xnew], [toy(xnew) + cnew], "o", color="C2", ms=9)
    imin = np.argmin(f); a2.axvline(x[imin], color="C3", ls=":", lw=1.2); a2.text(x[imin] + 0.08, a2.get_ylim()[1] - 0.35, "minimum in the\nsame place", color="C3", fontsize=11)
    a2.set_title("New amplifier"); a2.set_xlabel("one voltage (standardised)"); a2.legend(fontsize=10, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{OUT}/multi.png"); plt.close(fig)

# ---------------------------------------------------------------- BO results (dark charge only)
if want("results"):
    labs = ["random\norder", "by hand\n(hill-climb)", "multi-amp\nranking only", "trace\nranking", "BO, new amp\ndata only", "multi-amp\nBO + gate", "hybrid"]
    vals = [38.5, 26.8, 8.3, 7.3, 4.0, 1.8, 1.8]; cols = ["0.6", "0.6", "C0", "C0", "C0", "C3", "C3"]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.bar(range(len(vals)), vals, color=cols, edgecolor="k")
    for i, v in enumerate(vals): ax.text(i, v + 0.6, f"{v}", ha="center", fontsize=12)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, fontsize=10.5)
    ax.set_ylabel("images to reach 2 % of best"); ax.set_ylim(0, 43)
    ax.set_title("Dark charge only, nothing known about the new amplifier (mean of 6 searches)")
    fig.tight_layout(); fig.savefig(f"{OUT}/results.png"); plt.close(fig)
