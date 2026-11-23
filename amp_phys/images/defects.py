"""Phase 3.2b: defect maps per amplifier (CCD): glow regions, hot columns, hot pixels, long-exposure dark rate.

Inputs (read only):
  * dark image  : 2026-10-22 avg_img_400x6300x5 (AcquireImage, EXPOSURE 3708 s, no pumping);
  * persistence : 2026-10-12 avg_img_PP_1300x6300x1 x10 (EXPOSURE 100 s, 45 min readout; the 5000-pump
                  sequence there shows no pair artifact, see pp.py) -> hot pixels that are bright in >= 8 of 10.
Per CCD:
  * column profile (median over rows, e-/pix): glow = running median (151 col) above the 10th percentile floor;
    hot columns = column median - running median > max(5 robust sigma, 1 e-);
  * array dark rate from the row profile of glow-free columns: slope / t_row (rows read later wait longer);
  * hot pixels: > HOT_E e- in >= 8 of 10 persistence images, outside hot columns.
Usage:
    python defects.py
"""
import argparse, os, sys
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, load, find, mad_std, IMG_CHARGE, amp_name  # noqa
from units import active_cols  # noqa
from tracks import gains  # noqa
from dark import timing  # noqa

HOT_E = 25.0


def to_e(x, hdr, g):
    a0, a1 = active_cols(hdr)
    x = x.astype(np.float64)
    x = x - np.median(x[:, a1 + 5:], axis=1, keepdims=True)
    return x[:, a0 + 2:a1] / g, a0 + 2


def main(argv=None):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    gm = gains()
    amps = [x for x in IMG_CHARGE if x in gm]
    res = []
    hotcols_all = []
    fig, ax = plt.subplots(len(amps), 2, figsize=(16, 3.0 * len(amps)), squeeze=False)
    for k, (b, ch) in enumerate(amps):
        g = gm[(b, ch)]
        f = [x for x in find("2026-10-22-PA03_PQ04_PP_Testing", "400x6300x5_1x1", b) if "/avg_img_400" in x][0]
        hdr, data = load(f)
        E, c0 = to_e(data[ch], hdr, g)
        t_row, t_pix, ro = timing(hdr)
        E = E[10:]                                       # skip the first rows (row-start transient)
        colp = np.median(E, axis=0)
        smooth = ndimage.median_filter(colp, size=151, mode="nearest")
        floor = np.percentile(smooth, 10)
        glow = smooth - floor
        resid = colp - smooth
        rs = mad_std(resid)
        hot = (resid > max(5 * rs, 1.0))
        hc = np.nonzero(hot)[0]
        for c in hc:
            hotcols_all.append(dict(amp=amp_name(b, ch), col=int(c + c0 - 8), excess_e=float(resid[c]),
                                    excess_e_per_s=float(resid[c] / (float(hdr["EXPOSURE"]) + 0.5 * ro))))
        # glow description
        ip = int(np.argmax(glow))
        half = glow > 0.5 * glow[ip]
        lab, _ = ndimage.label(half)
        fw = int((lab == lab[ip]).sum()) if glow[ip] > 2 else 0
        # array dark from the row profile (glow-free, non-hot columns)
        quiet = (glow < 1.0) & ~ndimage.binary_dilation(hot, iterations=3)
        rowp = np.array([np.mean(np.clip(r[quiet], -20, 20)) for r in E])   # clipped mean (tracks)
        rr = np.arange(len(rowp)) + 10
        c, cv = np.polyfit(rr, rowp, 1, cov=True)
        D_slope = c[0] / t_row
        D_slope_err = np.sqrt(cv[0, 0]) / t_row
        lam0 = c[1]
        res.append(dict(board=b, ch=ch, amp=amp_name(b, ch), exposure_s=float(hdr["EXPOSURE"]), t_row_s=t_row,
                        floor_e=float(floor), quiet_cols_frac=float(quiet.mean()),
                        glow_peak_col=int(ip + c0 - 8), glow_peak_e=float(glow[ip]), glow_fwhm_cols=fw,
                        glow_integral_e_per_row=float(np.sum(glow[glow > 0.5])),
                        n_hot_columns=int(hot.sum()), hot_col_list=";".join(str(int(x + c0 - 8)) for x in hc[:15]),
                        D_arr_rowslope_e_per_pix_s=D_slope, D_arr_rowslope_err=D_slope_err,
                        D_arr_rowslope_e_per_pix_day=D_slope * 86400, lam_row0_e=lam0,
                        D_from_row0_over_exposure=lam0 / float(hdr["EXPOSURE"])))
        # figure: column profile
        x_ = np.arange(len(colp)) + c0 - 8
        ax[k, 0].plot(x_, colp, lw=0.5, color="0.4", label="column median")
        ax[k, 0].plot(x_, smooth, lw=1.2, color="C1", label="running median (151)")
        ax[k, 0].plot(x_[hot], colp[hot], "rv", ms=4, label=f"hot columns ({hot.sum()})")
        ax[k, 0].set_yscale("symlog", linthresh=10)
        ax[k, 0].set_ylabel("e⁻/pix")
        ax[k, 0].set_title(f"{amp_name(b, ch)}: 400×6300×5 dark ({hdr['EXPOSURE']} s + {ro / 60:.0f} min readout)",
                           fontsize=9)
        ax[k, 0].legend(fontsize=7, loc="upper right")
        ax[k, 1].plot(rr, rowp, ".", ms=2, color="0.4")
        ax[k, 1].plot(rr, np.polyval(c, rr), "C0-", label=f"D = {D_slope * 86400:.1f} ± {D_slope_err * 86400:.1f} e⁻/pix/day")
        ax[k, 1].set_ylabel("mean e⁻/pix (quiet cols)")
        ax[k, 1].legend(fontsize=8)
    ax[-1, 0].set_xlabel("physical column")
    ax[-1, 1].set_xlabel("read row (read later →)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "defects_profiles.png"), dpi=100)

    # ---------------- hot pixels from the persistence set
    pers = {}
    for b in (101, 102):
        fs = find("2026-10-12-PA03_PQ04_PocketPump1", "PP_1300x6300x1_1x1", b)
        for f in fs:
            hdr, data = load(f)
            for ch, x in enumerate(data):
                if (b, ch) not in gm:
                    continue
                E, c0 = to_e(x, hdr, gm[(b, ch)])
                cmed = np.median(E, axis=0, keepdims=True)
                E = E - cmed
                # exclude glow cores / hot-spot columns (column median > 20 e-) and noisy columns
                bad = (cmed[0] > 20) | (mad_std(E, axis=0) > 3 * np.median(mad_std(E, axis=0)))
                bad = ndimage.binary_dilation(bad, iterations=3)
                pers.setdefault((b, ch), []).append((E > HOT_E) & ~bad[None, :])
                pers.setdefault(("bad", b, ch), []).append(bad)
    fig2, ax2 = plt.subplots(1, len(amps), figsize=(4 * len(amps), 4), squeeze=False)
    for k, (b, ch) in enumerate(amps):
        if (b, ch) not in pers:
            continue
        st = np.sum(pers[(b, ch)], axis=0)
        nimg = len(pers[(b, ch)])
        hp = st >= int(np.ceil(0.8 * nimg))
        r = [x for x in res if x["board"] == b and x["ch"] == ch][0]
        hcols = [int(c) for c in r["hot_col_list"].split(";") if c]
        r["n_images_persist"] = nimg
        r["frac_cols_excluded_hotspot_or_noisy"] = float(np.mean(np.any(pers[("bad", b, ch)], axis=0)))
        r["n_hot_pixels"] = int(hp.sum())
        r["hot_pixels_per_Mpix"] = float(hp.sum() / hp.size * 1e6)
        # chance coincidence from random (cosmic) hits: p^8 with p = occupancy
        p = np.mean([m.mean() for m in pers[(b, ch)]])
        r["random_occupancy"] = float(p)
        rr_, cc_ = np.nonzero(hp)
        ax2[0, k].plot(cc_, rr_, ".", ms=2)
        ax2[0, k].set_xlim(0, hp.shape[1])
        ax2[0, k].set_ylim(hp.shape[0], 0)
        ax2[0, k].set_title(f"{amp_name(b, ch)}: {hp.sum()} hot pixels (>{HOT_E:.0f} e⁻ in ≥{int(np.ceil(0.8 * nimg))}/{nimg})",
                            fontsize=8)
        ax2[0, k].set_xlabel("column")
    ax2[0, 0].set_ylabel("row")
    fig2.tight_layout()
    fig2.savefig(os.path.join(a.out, "defects_hot_pixels.png"), dpi=100)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(a.out, "defects_summary.csv"), index=False)
    pd.DataFrame(hotcols_all).to_csv(os.path.join(a.out, "defects_hot_columns.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
