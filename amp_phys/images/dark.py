"""Phase 3.2a: dark charge decomposed into array, serial-register and time-independent parts, per amplifier,
from electron-counting skipper images; and its VSUB dependence (v3 / New_Packet 32x1000x500 scans).

Model for the mean charge of an unmasked active pixel at read row r, read column j:
    lambda(r, j) = a + D_arr * r * t_row + D_ser * j * t_pix
  D_arr : array dark rate (e-/pix/s), from the growth along rows (rows read later wait longer in the array);
  D_ser : serial-register dark rate (e-/pix/s): a pixel read at column j waited j*t_pix in the serial register;
  a     : charge at the first read pixel = exposure dark + clock-induced charge (CIC) + spurious charge.
The overscan level lambda_ovs (empty pixels clocked through the register) is reported separately.
Zero level: the 0-e- peak position of a comb fit (single electrons resolved); pixels >= 4 e- (tracks, hot pixels)
and their neighbours are masked.  t_pix and t_row come from the file timestamps (readout time / pixels).

Usage:
    python dark.py [--jobs 48]          (all images; writes dark_images.csv, dark_vsub.csv, dark_*.png)
"""
import argparse, os, sys, glob
import numpy as np
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, OUT, load, find, mad_std, IMG_CHARGE, amp_name  # noqa
from units import baseline_subtract, comb_gain, fit_comb, active_cols  # noqa

SETS = [("2026-10-14-PA03_PQ04_Skip_Noise", "250x1000x*_1x1"), ("2026-10-14-PA03_PQ04_Skip_Noise", "100x500x*_1x1"),
        ("2026-10-22-PA03_PQ04_PP_Testing", "400x1500x500_1x1"), ("2026-10-16-PA03_PQ04_16rows", "16x1000x500_1x1"),
        ("2026-11-05-PA03_PQ04_New_Packet_Tests", "32x1000x500_1x1"), ("2026-11-10-PA03_PQ04_v3_packets", "32x1000x500_1x1")]
VKEYS = None


def timing(hdr):
    from datetime import datetime
    t0 = datetime.fromisoformat(hdr["DATEINI"])
    t1 = datetime.fromisoformat(hdr["DATEEND"])
    ro = (t1 - t0).total_seconds() - float(hdr["EXPOSURE"])
    nrow, ncol = int(hdr["NROW"]), int(hdr["NCOL"])
    t_row = ro / nrow
    t_pix = int(hdr["NDCM"]) * 33.1e-6 + 0.30e-3     # from the timing table (see IMAGES_REPORT.md)
    return t_row, t_pix, ro


def one(args):
    path, board = args
    hdr, data = load(path)
    N = int(hdr["NDCM"])
    t_row, t_pix, ro = timing(hdr)
    rows = []
    for ch, x in enumerate(data):
        if (board, ch) not in IMG_CHARGE:
            continue
        y, (a0, a1) = baseline_subtract(x, hdr)
        act = y[:, a0 + 2:a1]
        g0, coh = comb_gain(act.ravel())
        rec = dict(file=os.path.basename(path), run=os.path.basename(os.path.dirname(path)), board=board, ch=ch,
                   NDCM=N, nrow=y.shape[0], t_row=t_row, t_pix=t_pix, exposure=float(hdr["EXPOSURE"]),
                   VSUB=float(hdr.get("VSUB", np.nan)))
        if g0 is None or coh < 0.15:
            rows.append(rec)
            continue
        try:
            f, _ = fit_comb(act.ravel(), g0)
        except Exception:
            rows.append(rec)
            continue
        if not (f["sigma"] < 0.4 * f["gain"]):
            rows.append(rec)
            continue
        g = f["gain"]
        # per-row zero level = 0-e- peak of that row (overscan median is biased when overscan carries charge)
        z = x.astype(np.float64)
        z = z - np.median(z[:, a0 + 2:a1], axis=1, keepdims=True)
        zz = z[:, a0 + 2:]
        b = np.percentile(zz, 10, axis=1, keepdims=True)
        for _ in range(3):
            w = np.abs(zz - b) < 0.5 * g
            b = np.array([[np.median(r[m]) if m.sum() > 10 else bb[0]] for r, m, bb in zip(zz, w, b)])
        y = z.copy()
        y[:, a0 + 2:] -= b
        # column zero (row-start transients): median per column of 0-e pixels
        w = np.abs(y) < 0.5 * g
        cz = np.array([np.median(c[m]) if m.sum() > 3 else 0.0 for c, m in zip(y.T, w.T)])
        y = y - cz[None, :]
        e = y / g
        big = e > 3.5
        mask = ndimage.binary_dilation(big, iterations=1)
        # integer electron counts (rounded), clipped
        n = np.clip(np.round(e), 0, None)
        A = n[:, a0 + 2:a1]
        M = ~mask[:, a0 + 2:a1]
        rr, jj = np.nonzero(M)
        jj = jj + 2  # read index of column relative to a0
        v = A[M]
        X = np.vstack([np.ones_like(v), rr * t_row, (jj + a0) * t_pix]).T
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        res = v - X @ beta
        cov = np.linalg.inv(X.T @ X) * np.var(res)
        ovs = n[:, a1 + 5:]
        movs = ~mask[:, a1 + 5:]
        rec.update(gain=f["gain"], sigma_e=f["sigma"] / f["gain"], lam_act=float(v.mean()),
                   lam_ovs=float(ovs[movs].mean()) if movs.sum() else np.nan,
                   a=beta[0], a_err=np.sqrt(cov[0, 0]), D_arr=beta[1], D_arr_err=np.sqrt(cov[1, 1]),
                   D_ser=beta[2], D_ser_err=np.sqrt(cov[2, 2]), frac_masked=float(mask[:, a0 + 2:a1].mean()))
        # header voltages
        for k in hdr:
            if k.endswith(("_H", "_L")) or k.startswith(("VDD", "VR", "VDRAIN", "VSUB")):
                try:
                    rec["hv_" + k] = float(hdr[k])
                except Exception:
                    pass
        rows.append(rec)
    return rows


def main(argv=None):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args(argv)
    p = os.path.join(a.out, "dark_images.csv")
    if not a.analyse_only:
        jobs = []
        for run, pat in SETS:
            for b in (101, 102):
                jobs += [(f, b) for f in find(run, pat, b)]
        print(len(jobs), "images")
        rows = []
        with ProcessPoolExecutor(a.jobs) as ex:
            for r in ex.map(one, jobs, chunksize=2):
                rows += r
        df = pd.DataFrame(rows)
        df.to_csv(p, index=False)
    df = pd.read_csv(p)
    df = df[df.gain.notna()]
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ------------- 1) dark decomposition at nominal, per amplifier (Oct runs + nominal packets)
    hv = [c for c in df.columns if c.startswith("hv_") and c != "hv_VSUB"]
    pk = df[df.run.str.contains("Packet|v3")].copy()
    nominal = {}
    if len(pk):
        modes = pk[hv].round(2).mode().iloc[0]
        dev = (pk[hv].round(2) != modes).sum(axis=1)
        pk["n_dev"] = dev
        pk["dev_keys"] = (pk[hv].round(2) != modes).apply(lambda r: ",".join(k[3:] for k, v in r.items() if v), axis=1)
    summ = []
    for b, ch in IMG_CHARGE:
        d = df[(df.board == b) & (df.ch == ch)]
        rec = dict(board=b, ch=ch, amp=amp_name(b, ch))
        for lab, sel in [("skip", d.run.str.contains("Skip_Noise") & (d.NDCM >= 500)),
                         ("pp400x1500", d.file.str.contains("400x1500")),
                         ("packets_nominal", d.run.str.contains("Packet|v3") & (d.file.isin(pk[(pk.n_dev == 0) & (np.abs(pk.VSUB - 45) < 0.6)].file) if len(pk) else False))]:
            s = d[sel]
            if not len(s):
                continue
            for q in ("D_arr", "D_ser", "a", "lam_act", "lam_ovs", "sigma_e"):
                v = s[q].values
                if q + "_err" in s:
                    w = 1 / np.maximum(s[q + "_err"].values, 1e-12) ** 2
                    m = np.sum(w * v) / np.sum(w)
                    err = max(np.sqrt(1 / np.sum(w)), np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
                else:
                    m, err = np.mean(v), (np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)
                rec[f"{lab}_{q}"] = m
                rec[f"{lab}_{q}_err"] = err
            rec[f"{lab}_n"] = len(s)
        summ.append(rec)
    sd = pd.DataFrame(summ)
    sd.to_csv(os.path.join(a.out, "dark_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(sd.T.to_string())

    # ------------- 2) VSUB dependence from the packet scans: all voltages vary together -> multivariate OLS
    out = []
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    qs = [("a", "a: CIC + time-independent (e⁻/pix)"), ("D_arr", "array dark D_arr (e⁻/pix/s)"),
          ("D_ser", "serial-register dark D_ser (e⁻/pix/s)"), ("lam_ovs", "overscan λ (e⁻/pix)")]
    for run in ("2026-11-10-PA03_PQ04_v3_packets", "2026-11-05-PA03_PQ04_New_Packet_Tests"):
        pr = df[(df.run == run) & (df.board == 101)]
        if not len(pr):
            continue
        cand = [c for c in hv + ["hv_VSUB"] if not c.endswith(("_LDO", "_SW", "_filter", "_divider"))
                and "_2_" not in c and c not in ("hv_VR2", "hv_VDD2", "hv_VDRAIN2")]
        ref = pr[pr.ch == 0]
        keep = []
        for c in cand:
            v = ref[c].values
            if np.nanstd(v) < 1e-3 or not np.all(np.isfinite(v)):
                continue
            if any(abs(np.corrcoef(v, ref[k].values)[0, 1]) > 0.98 for k in keep):
                continue
            keep.append(c)
        if "hv_VSUB" not in keep:
            keep = ["hv_VSUB"] + [k for k in keep if abs(np.corrcoef(ref.hv_VSUB, ref[k])[0, 1]) < 0.98]
        for k_, (b, ch) in enumerate([(101, 0), (101, 1), (101, 2), (101, 3)]):
            d = pr[pr.ch == ch]
            rec = dict(run=run, board=b, ch=ch, amp=amp_name(b, ch), n_images=len(d), n_regressors=len(keep),
                       VSUB_min=d.VSUB.min(), VSUB_max=d.VSUB.max())
            Xc = d[keep].values - d[keep].median().values
            X = np.hstack([np.ones((len(d), 1)), Xc])
            iv = keep.index("hv_VSUB") + 1
            for i, (q, lab) in enumerate(qs):
                y = d[q].values
                ok = np.isfinite(y)
                beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
                res = y[ok] - X[ok] @ beta
                dof = max(ok.sum() - X.shape[1], 1)
                cov = np.linalg.pinv(X[ok].T @ X[ok]) * np.sum(res ** 2) / dof
                rec[f"{q}_nom"] = beta[0]
                rec[f"{q}_nom_err"] = np.sqrt(cov[0, 0])
                rec[f"{q}_dVSUB"] = beta[iv]
                rec[f"{q}_dVSUB_err"] = np.sqrt(cov[iv, iv])
                rec[f"{q}_dVSUB_pct"] = 100 * beta[iv] / beta[0]
                rec[f"{q}_dVSUB_pct_err"] = 100 * np.sqrt(cov[iv, iv]) / abs(beta[0])
                if run.startswith("2026-11-10"):
                    # partial residual plot: y - (all other terms) vs VSUB
                    part = y[ok] - X[ok] @ beta + beta[0] + beta[iv] * Xc[ok, iv - 1]
                    ax[i].plot(d.VSUB.values[ok], part, "o", ms=3, color=cols[k_], alpha=0.5, label=amp_name(b, ch))
                    xx = np.linspace(d.VSUB.min(), d.VSUB.max(), 10)
                    ax[i].plot(xx, beta[0] + beta[iv] * (xx - d.hv_VSUB.median()), "-", color=cols[k_])
                    ax[i].set_title(lab, fontsize=10)
                    ax[i].set_xlabel("VSUB (V)  [partial residual: other voltages regressed out]")
            out.append(rec)
    ax[0].legend(fontsize=8)
    for x_ in ax:
        x_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "dark_vs_vsub.png"), dpi=110)
    od = pd.DataFrame(out)
    od.to_csv(os.path.join(a.out, "dark_vsub.csv"), index=False)
    print(od.T.to_string())


if __name__ == "__main__":
    main()
