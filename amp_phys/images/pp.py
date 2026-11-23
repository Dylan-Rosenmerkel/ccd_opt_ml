"""Phase 3.3: pocket-pumping trap dipoles per amplifier (CCD).

A charge trap under a vertical phase, pumped N times, moves charge from one pixel to its vertical neighbour:
a 'dipole' (one pixel bright, the adjacent one dark by the same amount, A ~ N * P_capture-emission, capped by
the available charge).  For each image / channel:
  * row baseline from the overscan, column baseline (glow, hot columns) from the column median; convert to e-;
  * mask clusters (> 8 sigma, dilated 2 px) and noisy columns;
  * local background m = median of the column over rows r-6..r+7 (excluding r, r+1);
  * dipole candidates: (x_r - m) > k*s and (x_{r+1} - m) < -k*s (type '+-') or the reverse ('-+'),
    s = noise of the image (overscan), k = 3;  amplitude A = (|x_r - m| + |x_{r+1} - m|)/2;
  * the same count on horizontal pairs (c, c+1) is the control (vertical pumping should not make horizontal dipoles);
  * density = candidates per active pixel; excess over the unpumped reference image of the same geometry.
The series PP_400x6300x5 has N = 15, 50, 150, 500, 5000, 50000 at one phase time (delay_V = 6000 clocks = 60 us):
emission time constants (which need a scan of the phase time) are NOT determinable from these data.

Usage:
    python pp.py [--jobs 32]
"""
import argparse, os, sys
import numpy as np
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, load, find, mad_std, IMG_CHARGE, amp_name  # noqa
from units import active_cols  # noqa
from tracks import gains  # noqa

RUN = "2026-10-22-PA03_PQ04_PP_Testing"
K = 3.0
CHARGE_BINS = [(-5, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 200)]


def analyse_channel(x, hdr, g):
    a0, a1 = active_cols(hdr)
    x = x.astype(np.float64)
    x = x - np.median(x[:, a1 + 5:], axis=1, keepdims=True)
    s_ovs = mad_std(x[10:, a1 + 5:]) / g
    A = x[:, a0 + 2:a1] / g
    A = A - np.median(A, axis=0, keepdims=True)
    s_act = mad_std(A)
    big = A > 8 * max(s_act, s_ovs)
    mask = ndimage.binary_dilation(big, iterations=2)
    colmad = mad_std(A, axis=0)
    mask |= (colmad > 2 * np.median(colmad))[None, :]
    mask[:10] = True
    # local background per column (running median over 14 rows) ~ use median filter along rows
    m = ndimage.median_filter(A, size=(15, 1), mode="nearest")
    d = A - m
    s = max(s_ovs, 1e-3)
    out = {}
    for name, (u, v, mu) in {"vert": (d[:-1], d[1:], mask[:-1] | mask[1:]),
                             "horz": (d[:, :-1], d[:, 1:], mask[:, :-1] | mask[:, 1:])}.items():
        pm = (u > K * s) & (v < -K * s) & ~mu
        mp = (u < -K * s) & (v > K * s) & ~mu
        amp = np.r_[(u[pm] - v[pm]) / 2, (v[mp] - u[mp]) / 2]
        npix = (~mu).sum()
        out[name] = dict(n_pm=int(pm.sum()), n_mp=int(mp.sum()), npix=int(npix), amps=amp)
    # lag-1 correlations of unmasked pixels
    dd = np.where(mask, 0, d)
    # density of vertical '+-' / '-+' pairs vs the local charge level (column median, e-)
    cm = np.median((x[10:, a0 + 2:a1] / g), axis=0)
    u, v, mu = d[:-1], d[1:], mask[:-1] | mask[1:]
    pair = (((u > K * s) & (v < -K * s)) | ((u < -K * s) & (v > K * s))) & ~mu
    for lo, hi in CHARGE_BINS:
        sel = (cm >= lo) & (cm < hi)
        n_ok = (~mu[10:, sel]).sum()
        out[f"dens_{lo}_{hi}"] = float(pair[10:, sel].sum() / n_ok) if n_ok > 1000 else np.nan
        out[f"npix_{lo}_{hi}"] = int(n_ok)
    darkpix = (x[1:, a0 + 2:a1] / g)[(u > K * s) & (v < -K * s) & ~mu]
    out["abs_dark_pixel_median_e"] = float(np.median(darkpix)) if len(darkpix) else np.nan
    out["rho_v"] = float(np.sum(dd[1:] * dd[:-1]) / np.sum(dd * dd))
    out["rho_h"] = float(np.sum(dd[:, 1:] * dd[:, :-1]) / np.sum(dd * dd))
    out["s_ovs"] = s_ovs
    out["s_act"] = s_act
    out["mean_charge"] = float(np.mean((x[10:, a0 + 2:a1] / g)[~mask[10:]]))
    return out


def one(args):
    path, board, gm = args
    hdr, data = load(path)
    rows = []
    for ch, x in enumerate(data):
        if (board, ch) not in gm:
            continue
        o = analyse_channel(x, hdr, gm[(board, ch)])
        pump = "PP_" in os.path.basename(path)
        for kind in ("vert", "horz"):
            r = o[kind]
            amps = r["amps"]
            rows.append(dict(file=os.path.basename(path), board=board, ch=ch, pumped=pump,
                             N=int(hdr.get("PP_NCYCLES", 0)) if pump else 0, nrow=int(hdr["NROW"]),
                             ndcm=int(hdr["NDCM"]), exposure=float(hdr["EXPOSURE"]), kind=kind,
                             n_pm=r["n_pm"], n_mp=r["n_mp"], npix=r["npix"],
                             density=(r["n_pm"] + r["n_mp"]) / r["npix"],
                             amp_med=float(np.median(amps)) if len(amps) else np.nan,
                             n_amp_gt20=int((amps > 20).sum()), n_amp_gt50=int((amps > 50).sum()),
                             rho_v=o["rho_v"], rho_h=o["rho_h"], s_ovs=o["s_ovs"], s_act=o["s_act"],
                             mean_charge=o["mean_charge"], abs_dark_pixel_median_e=o["abs_dark_pixel_median_e"],
                             **{k_: o[k_] for k_ in o if k_.startswith(("dens_", "npix_"))}))
    return rows


def main(argv=None):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    gm = gains()
    jobs = []
    for b in (101, 102):
        for f in find(RUN, "*400x6300x5_1x1", b) + find(RUN, "PP_1300x6300x5_1x1", b) + \
                find("2026-10-12-PA03_PQ04_PocketPump1", "PP_1300x6300x*_1x1", b):
            jobs.append((f, b, gm))
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(one, jobs):
            rows += r
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "pp_dipoles.csv"), index=False)
    pd.set_option("display.width", 250)
    s = df[(df.nrow == 400)].sort_values(["board", "ch", "kind", "N"])
    print(s[["board", "ch", "kind", "pumped", "N", "density", "n_pm", "n_mp", "amp_med", "n_amp_gt20", "rho_v",
             "rho_h", "s_ovs", "s_act", "mean_charge"]].to_string())
    # summary: excess density over the unpumped reference, per amplifier
    summ = []
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, (b, ch) in enumerate([x for x in IMG_CHARGE if x in gm]):
        d = s[(s.board == b) & (s.ch == ch)]
        for kind, mk in (("vert", "o-"), ("horz", "s--")):
            dk = d[d.kind == kind]
            ref = dk[~dk.pumped]
            pp = dk[dk.pumped].sort_values("N")
            if not len(ref) or not len(pp):
                continue
            r0 = ref.density.values[0]
            r0e = np.sqrt(ref.n_pm.values[0] + ref.n_mp.values[0]) / ref.npix.values[0]
            exc = pp.density.values - r0
            exce = np.sqrt((pp.n_pm.values + pp.n_mp.values) / pp.npix.values ** 2 + r0e ** 2)
            ax[0].errorbar(pp.N, exc * 1e6, exce * 1e6, fmt=mk, color=cols[k], ms=4,
                           label=f"{amp_name(b, ch)} {kind}")
            if kind == "vert":
                asym = (pp.n_pm - pp.n_mp) / (pp.n_pm + pp.n_mp)
                ax[1].plot(pp.N, asym, mk, color=cols[k], label=amp_name(b, ch))
                ax[2].plot(pp.N, pp.s_act / ref.s_act.values[0], mk, color=cols[k], label=amp_name(b, ch))
                # a trap population growing with N would make exc rise with N: fit exc = c0 + c1*log10(N)
                c = np.polyfit(np.log10(pp.N), exc, 1, w=1 / exce)
                summ.append(dict(board=b, ch=ch, amp=amp_name(b, ch), ref_density_per_Mpix=r0 * 1e6,
                                 mean_excess_per_Mpix=np.average(exc, weights=1 / exce ** 2) * 1e6,
                                 mean_excess_err=1e6 / np.sqrt(np.sum(1 / exce ** 2)),
                                 slope_per_decade_N_per_Mpix=c[0] * 1e6,
                                 excess_N50000_per_Mpix=exc[pp.N.values == 50000][0] * 1e6 if np.any(pp.N.values == 50000) else np.nan,
                                 horz_excess_note="see pp_dipoles.csv",
                                 s_act_ratio_pp_over_ref=float(np.mean(pp.s_act / ref.s_act.values[0])),
                                 mean_charge_ref=float(ref.mean_charge.values[0]),
                                 mean_charge_pp=float(pp.mean_charge.mean())))
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("number of pumps N")
    ax[0].set_ylabel("dipole candidates − unpumped reference (per Mpix)")
    ax[0].set_title(f"Dipole-pair excess (|Δ| > {K:.0f}σ on both pixels)", fontsize=10)
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("number of pumps N")
    ax[1].set_ylabel("(N₊₋ − N₋₊)/(N₊₋ + N₋₊), vertical")
    ax[1].set_title("Dipole orientation asymmetry", fontsize=10)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("number of pumps N")
    ax[2].set_ylabel("active-pixel scatter: pumped / unpumped")
    ax[2].set_title("Pixel scatter in pumped vs unpumped image", fontsize=10)
    for x_ in ax:
        x_.grid(alpha=0.3)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "pp_dipoles.png"), dpi=110)
    # charge-binned density table (vertical pairs), N = 50000 and unpumped reference
    tb = []
    for (b, ch) in [x for x in IMG_CHARGE if x in gm]:
        d = s[(s.board == b) & (s.ch == ch) & (s.kind == "vert")]
        for lab, r in (("ref", d[~d.pumped]), ("N=15", d[d.N == 15]), ("N=50000", d[d.N == 50000])):
            if len(r):
                tb.append(dict(amp=amp_name(b, ch), image=lab, **{f"{lo}..{hi} e": r[f"dens_{lo}_{hi}"].values[0]
                                                                for lo, hi in CHARGE_BINS},
                               dark_pixel_abs_e=r.abs_dark_pixel_median_e.values[0]))
    tb = pd.DataFrame(tb)
    tb.to_csv(os.path.join(a.out, "pp_density_vs_charge.csv"), index=False)
    print(tb.to_string())
    sm = pd.DataFrame(summ)
    sm.to_csv(os.path.join(a.out, "pp_summary.csv"), index=False)
    print(sm.T.to_string())
    print(df[df.nrow == 1300].groupby(["board", "ch", "kind", "N", "ndcm"])[["density", "rho_v", "rho_h", "s_act", "s_ovs"]].mean())


if __name__ == "__main__":
    main()
