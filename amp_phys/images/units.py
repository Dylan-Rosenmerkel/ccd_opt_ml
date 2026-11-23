"""Phase 3.0: units and noise per amplifier from skipper images.

For each image (skip-averaged, avg_img_*) and each channel:
  1. baseline: per-row median of the overscan (columns past the active area) -> subtract; then per-column median of
     the 0-e population (removes the row-start dip / column transients);
  2. gain g [ADU/e-] from a fit of the pixel histogram with a comb of Gaussians mu0 + k*g, k = 0..K
     (free amplitudes, common sigma), when single electrons are resolved (sigma < ~0.35 g);
  3. read noise sigma [ADU] from the 0-e peak (resolved) and, for every image, from the overscan
     (robust MAD of the baseline-subtracted overscan: pure readout, no array charge);
  4. noise vs number of skips N: sigma_e(N) = sqrt(s1^2/N + floor^2) fitted per amplifier.

Usage:
    python units.py [--run 2026-10-14-PA03_PQ04_Skip_Noise] [--extra 400x1500x500] [--out results/images]
Outputs: units_fits.csv (per image x channel), units_summary.csv (per amplifier), noise_vs_skips.png,
         peaks_<board>.png.
"""
import argparse, os, sys, re
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, OUT, load, find, mad_std, ALL, WORKING, amp_name  # noqa

PRESCAN, NACT = 8, 6144


def active_cols(hdr):
    """Index range (in the read image) of active columns; the rest (if any) is overscan."""
    c0 = int(hdr.get("NCOLPRE", 0))
    nsb = int(hdr.get("NSBIN", 1))
    ncol = int(hdr["NCOL"])
    first_ovs = (PRESCAN + NACT - c0) // nsb  # read index of the first overscan column
    a0 = max(0, (PRESCAN - c0) // nsb) if c0 < PRESCAN else 0
    return a0, min(ncol, first_ovs)


def baseline_subtract(x, hdr, ovs_min=30):
    """Return image with per-row and per-column baselines removed and the overscan slice."""
    a0, a1 = active_cols(hdr)
    x = x.astype(np.float64)
    if x.shape[1] - a1 >= ovs_min:
        ovs = x[:, a1 + 5:]
        rb = np.median(ovs, axis=1, keepdims=True)
    else:
        rb = np.median(x[:, a0:a1], axis=1, keepdims=True)
    y = x - rb
    # column baseline from pixels near zero (robust to charge): median of values below the row-median + 1 MAD
    s = mad_std(y[:, a0:a1])
    m = np.where(np.abs(y) < 3 * max(s, 1e-3), y, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        cb = np.nanmedian(m, axis=0, keepdims=True)
    cb = np.where(np.isfinite(cb), cb, 0)
    y = y - cb
    return y, (a0, a1)


def comb_gain(v, gmin=0.5, gmax=8.0):
    """Initial gain from the coherence of exp(2 pi i v / g) (largest strongly coherent period)."""
    v = v[(v > -3) & (v < 40)]
    h, e = np.histogram(v, np.arange(-3, 40, 0.01))
    c = 0.5 * (e[1:] + e[:-1])
    gs = np.linspace(gmin, gmax, 1500)
    S = np.abs(np.exp(2j * np.pi * c[None, :] / gs[:, None]) @ h) / max(h.sum(), 1)
    # local maxima
    loc = [i for i in range(1, len(gs) - 1) if S[i] >= S[i - 1] and S[i] >= S[i + 1]]
    if not loc:
        return None, 0
    smax = max(S[i] for i in loc)
    cand = [i for i in loc if S[i] > 0.6 * smax]
    i = max(cand, key=lambda j: gs[j])
    return gs[i], S[i]


def fit_comb(v, g0, K=6):
    """Fit histogram with sum_k A_k N(mu0 + k g, sigma). Returns dict with values and errors."""
    lo, hi = -4 * min(g0, 3), (K + 0.5) * g0
    bw = g0 / 40
    bins = np.arange(lo, hi, bw)
    h, e = np.histogram(v, bins)
    c = 0.5 * (e[1:] + e[:-1])
    A0 = [max(h[np.abs(c - k * g0) < g0 / 4].max() if np.any(np.abs(c - k * g0) < g0 / 4) else 1, 1) for k in range(K + 1)]

    def model(p):
        mu0, g, s = p[:3]
        A = p[3:]
        return sum(A[k] * np.exp(-0.5 * ((c - mu0 - k * g) / s) ** 2) for k in range(K + 1))

    p0 = np.r_[0.0, g0, g0 / 6, A0]
    err = np.sqrt(np.maximum(h, 1))
    r = least_squares(lambda p: (model(p) - h) / err, p0,
                      bounds=(np.r_[-g0 / 2, g0 * 0.7, g0 / 100, np.zeros(K + 1)],
                              np.r_[g0 / 2, g0 * 1.3, g0, np.full(K + 1, np.inf)]))
    J = r.jac
    dof = max(len(h) - len(p0), 1)
    chi2 = np.sum(r.fun ** 2) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * max(chi2, 1)
        pe = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        pe = np.full(len(p0), np.nan)
    p = r.x
    return dict(mu0=p[0], gain=p[1], gain_err=pe[1], sigma=p[2], sigma_err=pe[2], A=p[3:], chi2=chi2,
                lam=p[4] / p[3] if p[3] > 0 else np.nan), (c, h, model(p))


def analyse_image(path, board):
    hdr, data = load(path)
    N = int(hdr["NDCM"])
    rows = []
    curves = {}
    for ch, x in enumerate(data):
        y, (a0, a1) = baseline_subtract(x, hdr)
        act = y[:, a0 + 2:a1]
        ovs = y[:, a1 + 5:] if y.shape[1] - a1 > 30 else None
        s_ovs = float(mad_std(ovs)) if ovs is not None else np.nan
        row = dict(file=os.path.basename(path), board=board, ch=ch, NDCM=N, nrow=y.shape[0], ncol=y.shape[1],
                   sigma_ovs_adu=s_ovs)
        g0, coh = comb_gain(act.ravel())
        row["comb_coh"] = coh
        if g0 is not None and coh > 0.15:
            try:
                f, cur = fit_comb(act.ravel(), g0)
                resolved = f["sigma"] < 0.4 * f["gain"] and f["A"][1] > 20
                if resolved:
                    row.update(gain=f["gain"], gain_err=f["gain_err"], sigma0_adu=f["sigma"],
                               sigma0_err=f["sigma_err"], lam_A1A0=f["lam"], chi2=f["chi2"])
                    curves[ch] = cur
            except Exception as e:  # noqa
                row["note"] = str(e)
        rows.append(row)
    return rows, curves, hdr


def noise_model(N, s1, fl):
    return np.sqrt(s1 ** 2 / N + fl ** 2)


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="2026-10-14-PA03_PQ04_Skip_Noise")
    ap.add_argument("--extra", nargs="*", default=["2026-10-22-PA03_PQ04_PP_Testing:400x1500x500_1x1",
                                                    "2026-10-16-PA03_PQ04_16rows:16x1000x500_1x1"])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    allcurves = {}
    jobs = []
    for b in (101, 102):
        for f in find(a.run, "*", b):
            jobs.append((a.run, f, b))
        for ex in a.extra:
            run, pat = ex.split(":")
            for f in find(run, pat, b):
                jobs.append((run, f, b))
    for run, f, b in jobs:
        r, cur, hdr = analyse_image(f, b)
        for x in r:
            x["run"] = run
            x["EXPOSURE"] = hdr.get("EXPOSURE")
        rows += r
        allcurves[(b, os.path.basename(f), int(hdr["NDCM"]))] = cur
        print(os.path.basename(f), [(x["ch"], round(x.get("gain", np.nan), 3), round(x.get("sigma0_adu", np.nan), 3),
                                     round(x["sigma_ovs_adu"], 3)) for x in r])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "units_fits.csv"), index=False)

    # --- summary per amplifier: gain from the skip-noise run (N in 500..5000; 10000 has a different ADC scale) ---
    from scipy.optimize import curve_fit
    sk = df[df.run == a.run]
    summ = []
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    nn = np.logspace(0, 4.1, 100)
    for i, (b, ch) in enumerate(ALL):
        d = sk[(sk.board == b) & (sk.ch == ch)].sort_values("NDCM")
        good = d[(d.NDCM >= 500) & (d.NDCM <= 5000) & d.gain.notna()]
        if len(good):
            w = 1 / good.gain_err ** 2
            gm = np.sum(w * good.gain) / np.sum(w)
            gerr = max(np.sqrt(1 / np.sum(w)), good.gain.std(ddof=1) / np.sqrt(len(good)) if len(good) > 1 else 0)
        else:
            gm, gerr = np.nan, np.nan
        g10k = d[d.NDCM == 10000].gain.values
        g10k = g10k[0] if len(g10k) and np.isfinite(g10k[0]) else np.nan
        rec = dict(board=b, ch=ch, amp=amp_name(b, ch), gain=gm, gain_err=gerr, n_gain_images=len(good),
                   gain_10000skips=g10k, adc_scale_10000=g10k / gm if np.isfinite(gm) else np.nan)
        # electronics view (ADU, overscan, all channels; 10000-skip point rescaled by the ADC-scale anomaly 3.33)
        Na = d.NDCM.values.astype(float)
        sa = d.sigma_ovs_adu.values.copy()
        sa[Na == 10000] /= (10000 / 3000)
        p1 = sa[Na == 1][0] if np.any(Na == 1) else np.nan
        rec["sigma_ovs_adu_N1"] = p1
        rec["ovs_adu_N5000_over_white"] = sa[Na == 5000][0] / (p1 / np.sqrt(5000)) if np.any(Na == 5000) else np.nan
        ax[2].loglog(Na, sa, "o-", color=cols[i % 10], ms=4, lw=0.8, label=rec["amp"])
        if np.isfinite(gm):
            # e-: 0-e peak width where resolved (N >= 1000), overscan MAD for N <= 100 (row drift negligible there)
            Ns, se = [], []
            for _, r in d.iterrows():
                gg = g10k if r.NDCM == 10000 else gm
                if r.NDCM >= 1000 and np.isfinite(r.get("sigma0_adu", np.nan)):
                    Ns.append(r.NDCM); se.append(r.sigma0_adu / gg)
                elif r.NDCM <= 100 and np.isfinite(r.sigma_ovs_adu):
                    Ns.append(r.NDCM); se.append(r.sigma_ovs_adu / gg)
            Ns, se = np.array(Ns, float), np.array(se)
            try:
                p, c = curve_fit(noise_model, Ns, se, p0=[se[0], 0.1], sigma=0.08 * se, bounds=([0, 0], [np.inf, np.inf]))
                pe = np.sqrt(np.diag(c))
            except Exception:
                p, pe = [np.nan, np.nan], [np.nan, np.nan]
            rec.update(sigma1_e=p[0], sigma1_e_err=pe[0], floor_e=p[1], floor_e_err=pe[1],
                       sigma_e_N1000=se[Ns == 1000][0] if np.any(Ns == 1000) else np.nan,
                       sigma_e_N5000=se[Ns == 5000][0] if np.any(Ns == 5000) else np.nan,
                       sigma_e_N10000=se[Ns == 10000][0] if np.any(Ns == 10000) else np.nan,
                       N_floor_equals_white=(p[0] / p[1]) ** 2 if p[1] > 0 else np.nan)
            ax[0].loglog(Ns, se, "o", color=cols[i % 10], label=rec["amp"])
            ax[0].loglog(nn, noise_model(nn, *p), "-", color=cols[i % 10], lw=1)
            ax[0].loglog(nn, p[0] / np.sqrt(nn), ":", color=cols[i % 10], lw=0.7)
            ax[1].errorbar(d.NDCM, d.gain, d.gain_err, fmt="o-", color=cols[i % 10], label=rec["amp"], ms=4)
        summ.append(rec)
    ax[0].set_xlabel("number of skips N")
    ax[0].set_ylabel("read noise σ (e⁻)")
    ax[0].set_title("Noise vs skips (N≤100: overscan; N≥1000: 0-e⁻ peak)\nsolid: √(σ₁²/N + floor²), dotted: σ₁/√N", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("number of skips N")
    ax[1].set_ylabel("gain from single-electron peaks (ADU/e⁻)")
    ax[1].set_title("Gain per image (N = 10000: ADC scale ×3.33)", fontsize=10)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[2].loglog(nn, 5.0 / np.sqrt(nn), "k:", lw=1, label="5 ADU/√N")
    ax[2].set_xlabel("number of skips N")
    ax[2].set_ylabel("overscan σ (ADU)")
    ax[2].set_title("Overscan noise in ADU, all 8 channels\n(incl. within-row baseline drift)", fontsize=10)
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "noise_vs_skips.png"), dpi=110)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(a.out, "units_summary.csv"), index=False)
    print(sdf.to_string())

    # peak plots for the 1000-skip and 5000-skip images
    for b in (101, 102):
        keys = [k for k in allcurves if k[0] == b and k[2] in (1000, 5000) and k[1].startswith("avg_img_2") or
                (k[0] == b and k[2] == 5000)]
        keys = sorted(set(keys), key=lambda k: k[2])
        if not keys:
            continue
        fig, ax = plt.subplots(len(keys), 4, figsize=(16, 3.2 * len(keys)), squeeze=False)
        for r_, k in enumerate(keys):
            for ch in range(4):
                axx = ax[r_, ch]
                if ch in allcurves[k]:
                    c, h, m = allcurves[k][ch]
                    axx.semilogy(c, np.maximum(h, 0.5), drawstyle="steps-mid", lw=0.8, label="data")
                    axx.semilogy(c, np.maximum(m, 0.5), "r-", lw=0.8, label="comb fit")
                    axx.set_ylim(0.5, None)
                else:
                    axx.text(0.5, 0.5, "single e⁻ not resolved\n(no charge / too noisy)", ha="center",
                             transform=axx.transAxes)
                axx.set_title(f"{amp_name(b, ch)}, N={k[2]}", fontsize=9)
                axx.set_xlabel("pixel value − baseline (ADU)")
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, f"peaks_{b}.png"), dpi=90)


if __name__ == "__main__":
    main()
