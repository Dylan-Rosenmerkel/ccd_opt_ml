"""Phase 3.1: cosmic-ray tracks -> thickness, diffusion, pixel-size and CTI proxies, per amplifier (CCD).

Per image and channel:
  * baseline: per-row median of the overscan, then per-column median of the active rows (removes glow, hot columns);
  * ADU -> e- with the gain from units_summary.csv (single-electron peaks);
  * clusters: pixels > THR*sigma, 8-connected; per cluster: charge Q (on the cluster dilated by 2 px), charge-weighted
    PCA (major/minor variance), length L (moment estimate sqrt(12*(var_major - sigma_mean^2 - 1/12)) and threshold
    extent), transverse width sigma(u) in segments along the track, leading/trailing edge charge (serial and vertical).
Straight, uncontained-by-edges tracks with L >= LMIN are 'muon candidates'.

Thickness: for a muon crossing the full depth T at projected length L, Q = q0 * sqrt(L^2 + T^2) (q0 = e- per pixel
of path length). A robust fit of Q vs L gives T in pixels (independent of the gain); q0 compared with the most-probable
MIP deposit (~ 0.28 keV/um / 3.75 eV) gives an estimate of the pixel pitch.

Usage:
    python tracks.py [--jobs 32] [--out results/images]      (find clusters in all listed images; writes tracks_clusters.csv)
    python tracks.py --analyse-only                            (fits + figures from tracks_clusters.csv)
"""
import argparse, os, sys, json
import numpy as np
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, load, find, mad_std, ALL, WORKING, amp_name  # noqa
from units import active_cols  # noqa

IMAGES = [  # (run, pattern) with long exposures/readouts, full width
    ("2026-10-22-PA03_PQ04_PP_Testing", "400x6300x5_1x1"),
    ("2026-10-22-PA03_PQ04_PP_Testing", "PP_400x6300x5_1x1"),
    ("2026-10-22-PA03_PQ04_PP_Testing", "PP_1300x6300x5_1x1"),
    ("2026-10-22-PA03_PQ04_PP_Testing", "400x1500x500_1x1"),
    ("2026-10-12-PA03_PQ04_PocketPump1", "PP_1300x6300x1_1x1"),
    ("2026-10-12-PA03_PQ04_PocketPump1", "PP_1300x6300x5_1x1"),
]
THR = 4.0       # cluster threshold in sigma
LMIN = 0.0
SAT_ADU = 16000.0
SEG = 4.0       # segment length along the track (px) for width profile
MIP_E_PER_UM = 305.0 / 3.75   # median MIP deposit per um for ~0.6-2 mm Si paths (~0.30-0.31 keV/um), +-10 %


def gains():
    import pandas as pd
    p = os.path.join(OUT, "units_summary.csv")
    g = {}
    if os.path.exists(p):
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            if np.isfinite(r.gain):
                g[(int(r.board), int(r.ch))] = float(r.gain)
    return g


def prep(x, hdr):
    a0, a1 = active_cols(hdr)
    x = x.astype(np.float64)
    sat = x > SAT_ADU
    ovs = x[:, a1 + 5:]
    rb = np.median(ovs, axis=1, keepdims=True) if ovs.shape[1] > 20 else np.median(x[:, a0:a1], axis=1, keepdims=True)
    y = x - rb
    act = y[:, a0 + 1:a1]
    cm = np.median(act, axis=0)
    y[:, a0 + 1:a1] -= cm[None, :]
    # row-dependent array dark (rows read later collect more): remove the per-row median of the active area
    y[:, a0 + 1:a1] -= np.median(y[:, a0 + 1:a1], axis=1, keepdims=True)
    return y, (a0 + 1, a1), cm, sat


def cluster_features(e, lab, sl, idx, sigma, img_shape, act, sat):
    """Features of one cluster. e: image in e-, lab: label image, sl: slice, idx: label id."""
    r0, c0 = sl[0].start, sl[1].start
    pad = 3
    R0, R1 = max(r0 - pad, 0), min(sl[0].stop + pad, img_shape[0])
    C0, C1 = max(c0 - pad, act[0]), min(sl[1].stop + pad, act[1])
    sub = e[R0:R1, C0:C1]
    m = lab[R0:R1, C0:C1] == idx
    md = ndimage.binary_dilation(m, iterations=2) & ((lab[R0:R1, C0:C1] == 0) | m)
    rr, cc = np.nonzero(md)
    w = sub[rr, cc]
    Q = w.sum()
    npx = int(m.sum())
    edge = (sl[0].start <= 1 or sl[0].stop >= img_shape[0] - 1 or sl[1].start <= act[0] + 1 or sl[1].stop >= act[1] - 1)
    nsat = int(sat[R0:R1, C0:C1][m].sum())
    rc, ccm = np.nonzero(m)
    wm = np.clip(sub[rc, ccm], 0, None)
    if wm.sum() <= 0:
        return None
    yc = np.sum(wm * rc) / wm.sum()
    xc = np.sum(wm * ccm) / wm.sum()
    cov = np.cov(np.vstack([ccm - xc, rc - yc]), aweights=wm) if npx > 2 else np.eye(2) * 0.1
    ev, evec = np.linalg.eigh(cov)
    l1, l2 = ev[1], ev[0]
    d = evec[:, 1]  # (dx, dy) of major axis
    n = np.array([-d[1], d[0]])
    # coordinates of the dilated pixels
    u = (cc - xc) * d[0] + (rr - yc) * d[1]
    v = (cc - xc) * n[0] + (rr - yc) * n[1]
    um = (ccm - xc) * d[0] + (rc - yc) * d[1]
    L_thr = um.max() - um.min() if npx > 1 else 0.0
    # transverse width in segments
    prof = []
    if L_thr >= 2 * SEG:
        nseg = int(L_thr // SEG)
        edges = np.linspace(um.min(), um.max(), nseg + 1)
        for k in range(nseg):
            s = (u >= edges[k]) & (u < edges[k + 1]) & (np.abs(v) < 4)
            ws = w[s]
            if ws.sum() <= 20 * sigma:
                prof.append((float(0.5 * (edges[k] + edges[k + 1])), np.nan, float(ws.sum())))
                continue
            vm = np.sum(ws * v[s]) / ws.sum()
            var = np.sum(ws * (v[s] - vm) ** 2) / ws.sum() - 1.0 / 12
            prof.append((float(0.5 * (edges[k] + edges[k + 1])), float(var), float(ws.sum())))
    # charge-weighted sigma^2 over all (for L moment correction)
    s_in = np.abs(v) < 4
    wv = np.clip(w[s_in], 0, None)
    var_perp = np.sum(wv * v[s_in] ** 2) / wv.sum() - 1 / 12 if wv.sum() > 0 else np.nan
    var_par = np.sum(wv * u[s_in] ** 2) / wv.sum() if wv.sum() > 0 else np.nan
    L_mom = np.sqrt(max(12 * (var_par - max(var_perp, 0) - 1 / 12), 0))
    far = np.sum(np.clip(w[np.abs(v) >= 3], 0, None)) / max(np.sum(np.clip(w, 0, None)), 1)
    # leading / trailing edges: serial (along a row, read order = increasing column) and vertical (increasing row)
    lead_s = trail_s = edge_s = 0.0
    lead_v = trail_v = edge_v = 0.0
    full = lab[R0:R1, C0:C1]
    for i in np.unique(rc):
        cols = ccm[rc == i]
        a, b = cols.min(), cols.max()
        if a - 2 >= 0 and b + 2 < sub.shape[1] and np.all(full[i, a - 2:a] == 0) and np.all(full[i, b + 1:b + 3] == 0):
            lead_s += sub[i, a - 2:a].sum()
            trail_s += sub[i, b + 1:b + 3].sum()
            edge_s += sub[i, a] + sub[i, b]
    for j in np.unique(ccm):
        rows = rc[ccm == j]
        a, b = rows.min(), rows.max()
        if a - 2 >= 0 and b + 2 < sub.shape[0] and np.all(full[a - 2:a, j] == 0) and np.all(full[b + 1:b + 3, j] == 0):
            lead_v += sub[a - 2:a, j].sum()
            trail_v += sub[b + 1:b + 3, j].sum()
            edge_v += sub[a, j] + sub[b, j]
    return dict(row=float(yc + R0), col=float(xc + C0), Q=float(Q), npix=npx, l1=float(l1), l2=float(l2),
                dx=float(d[0]), dy=float(d[1]), L_thr=float(L_thr), L_mom=float(L_mom), var_perp=float(var_perp),
                far=float(far), qmax=float(sub[rc, ccm].max()), edge=bool(edge), nsat=nsat,
                lead_s=float(lead_s), trail_s=float(trail_s), edge_s=float(edge_s),
                lead_v=float(lead_v), trail_v=float(trail_v), edge_v=float(edge_v),
                prof=json.dumps(prof))


def process(args):
    path, board, gmap = args
    hdr, data = load(path)
    out = []
    for ch, x in enumerate(data):
        g = gmap.get((board, ch))
        if g is None:
            continue
        y, act, cm, sat = prep(x, hdr)
        e = y / g
        a = e[:, act[0]:act[1]]
        sigma = float(mad_std(a))
        # bad columns: column median excess (glow/hot) is removed, but very hot / noisy columns are masked
        colmad = mad_std(a, axis=0)
        bad = np.zeros(e.shape[1], bool)
        bad[act[0]:act[1]] = (colmad > 3 * np.median(colmad)) | (np.abs(cm / g) > 50)
        bad[:act[0]] = True
        bad[act[1]:] = True
        bad = ndimage.binary_dilation(bad, iterations=2)
        mask = (e > THR * sigma) & ~bad[None, :]
        lab, nl = ndimage.label(mask, structure=np.ones((3, 3)))
        sls = ndimage.find_objects(lab)
        for idx, sl in enumerate(sls, start=1):
            if sl is None:
                continue
            npx = (lab[sl] == idx).sum()
            if npx < 3:
                continue
            f = cluster_features(e, lab, sl, idx, sigma, e.shape, act, sat)
            if f is None:
                continue
            f.update(file=os.path.basename(path), board=board, ch=ch, sigma_e=sigma, gain=g,
                     nrow=e.shape[0], NDCM=int(hdr["NDCM"]))
            out.append(f)
    return out


def image_list():
    jobs = []
    for run, pat in IMAGES:
        for b in (101, 102):
            for f in find(run, pat, b):
                jobs.append((f, b))
    return jobs


def find_all(jobs_n, out):
    import pandas as pd
    gm = gains()
    gm.setdefault((102, 2), np.nan)
    gm = {k: v for k, v in gm.items() if np.isfinite(v)}
    print("gains:", gm)
    jobs = [(f, b, gm) for f, b in image_list()]
    rows = []
    with ProcessPoolExecutor(jobs_n) as ex:
        for r in ex.map(process, jobs):
            rows += r
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "tracks_clusters.csv.gz"), index=False)
    print(len(df), "clusters")
    return df


# ---------------------------------------------------------------- analysis
CLEAN_TYPES = ("400x6300x5", "400x1500x500", "PP_1300x6300x1")  # no pumping, or exposure (100 s) << readout (45 min)
ALPHA_SYST = 0.045   # d ln(Q/s) / d ln s of the median MIP deposit for 0.5-2.5 mm Si paths (Bichsel), systematic


def select_muons(d, lmin=4.0):
    return d[(~d.edge) & (d.nsat == 0) & (d.far < 0.03) & (d.l2 < 2.0) & (d.L_mom >= lmin) & (d.npix >= 6)]


def fit_thickness(L, Q, alpha=0.0, nboot=200, rng=None):
    """Median (L1 in log) fit of Q = q0 * s * (s/50)^alpha, s = sqrt(L^2 + T^2) [px]. Returns q0, T, errors."""
    from scipy.optimize import minimize
    rng = rng or np.random.default_rng(1)

    def fit(L, Q):
        lq = np.log(Q)

        def cost(p):
            q0, T = np.exp(p)
            s = np.sqrt(L ** 2 + T ** 2)
            return np.sum(np.abs(lq - np.log(q0 * s * (s / 50.0) ** alpha)))
        best = None
        for T0 in (15, 30, 45, 70):
            q00 = np.median(Q / np.sqrt(L ** 2 + T0 ** 2))
            r = minimize(cost, np.log([q00, T0]), method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-7))
            if best is None or r.fun < best.fun:
                best = r
        return np.exp(best.x)
    p = fit(L, Q)
    bs = np.array([fit(L[i], Q[i]) for i in (rng.integers(0, len(L), len(L)) for _ in range(nboot))])
    return p[0], p[1], bs[:, 0].std(), bs[:, 1].std()


def boot_ratio(x, y, rng, n=300):
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return np.nan, np.nan
    r = x.sum() / y.sum()
    bs = [x[i].sum() / y[i].sum() for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return r, np.std(bs)


def analyse(out, lmin_fit=20.0):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(os.path.join(out, "tracks_clusters.csv.gz"))
    df["typ"] = df.file.str.extract(r"avg_img_(.*?)_1x1")[0]
    clean = df[df.typ.isin(CLEAN_TYPES)]
    res = []
    amps = [a for a in ALL if ((df.board == a[0]) & (df.ch == a[1])).sum() > 100]
    figQ, axQ = plt.subplots(1, len(amps), figsize=(4.3 * len(amps), 4.2), squeeze=False)
    figW, axW = plt.subplots(1, 2, figsize=(14, 4.8))
    figA, axA = plt.subplots(1, 3, figsize=(18, 4.6))
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rng = np.random.default_rng(2)
    for k, (b, ch) in enumerate(amps):
        d = clean[(clean.board == b) & (clean.ch == ch)]
        mu = select_muons(d)
        L, Q = mu.L_mom.values, mu.Q.values
        # muon band: iterate model, keep |log residual| < 0.45 for L >= lmin_fit
        def band_fit(lmin, alpha, nboot):
            q0, T = 1300.0, 40.0
            for it in range(5):
                s = np.sqrt(L ** 2 + T ** 2)
                band = (L >= lmin) & (np.abs(np.log(Q / (q0 * s * (s / 50.0) ** alpha))) < 0.45)
                q0, T, q0e, Te = fit_thickness(L[band], Q[band], alpha, nboot=nboot if it == 4 else 1, rng=rng)
            return q0, T, q0e, Te, band
        grid = [band_fit(lm, al, 1)[1] for lm in (12, 20, 30) for al in (0.0, ALPHA_SYST)]
        q0, T, q0e, Te, band = band_fit(lmin_fit, ALPHA_SYST / 2, 200)
        T_syst = 0.5 * (max(grid) - min(grid))
        Ta = np.mean(grid)
        nb = int(band.sum())
        ax = axQ[0, k]
        ax.plot(d.L_mom[d.L_mom < 160], d.Q[d.L_mom < 160], ",", color="0.75", alpha=0.6, label="all clusters")
        ax.plot(L[band], Q[band], ".", ms=2.5, color=cols[k], label=f"muon band, L≥{lmin_fit:.0f} (n={nb})")
        # binned medians
        bins = np.array([0, 3, 6, 9, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160])
        sm = select_muons(d)
        bm = [np.median(sm.Q[(sm.L_mom >= bins[i]) & (sm.L_mom < bins[i + 1]) & (sm.Q > 5000)])
              if ((sm.L_mom >= bins[i]) & (sm.L_mom < bins[i + 1])).sum() > 5 else np.nan for i in range(len(bins) - 1)]
        ax.plot(0.5 * (bins[1:] + bins[:-1]), bm, "kD", ms=4, mfc="w", label="median of straight clusters")
        ll = np.linspace(0, 160, 200)
        ax.plot(ll, q0 * np.sqrt(ll ** 2 + T ** 2), "k-", lw=1.3, label=f"q0·√(L²+T²): T={T:.1f}±{Te:.1f} px")
        ax.plot(ll, q0 * ll, "k:", lw=0.8, label="q0·L (T = 0)")
        ax.set_xlim(0, 160)
        ax.set_ylim(3e3, 3e5)
        ax.set_yscale("log")
        ax.set_xlabel("projected length L (px)")
        ax.set_ylabel("cluster charge Q (e⁻)")
        ax.set_title(amp_name(b, ch))
        ax.legend(fontsize=6.5, loc="lower right")
        # ---- diffusion: stacked width profile
        sel = mu[band]
        sel = sel[sel.L_thr >= 12]
        prof_t, prof_s2, ends = [], [], []
        for _, r in sel.iterrows():
            p = np.array(json.loads(r.prof), float)
            if len(p) < 4:
                continue
            ok = np.isfinite(p[:, 1])
            if ok.sum() < 4:
                continue
            t = (p[:, 0] - p[:, 0].min()) / (p[:, 0].max() - p[:, 0].min())
            ev, od = ok & (np.arange(len(p)) % 2 == 0), ok & (np.arange(len(p)) % 2 == 1)
            if ev.sum() >= 2 and od.sum() >= 2:
                sl = np.polyfit(t[ev], p[ev, 1], 1)[0]
                prof_t += list(t[od] if sl > 0 else 1 - t[od])
                prof_s2 += list(p[od, 1])
            ends.append(np.nanmean(p[ok, 1]))
        prof_t, prof_s2, ends = np.array(prof_t), np.array(prof_s2), np.array(ends)
        tb = np.linspace(0, 1, 9)
        bc = 0.5 * (tb[1:] + tb[:-1])
        med, err = [], []
        for i in range(len(tb) - 1):
            v = np.sqrt(np.clip(prof_s2[(prof_t >= tb[i]) & (prof_t <= tb[i + 1])], 0, None))
            med.append(np.median(v) if len(v) > 10 else np.nan)
            err.append(np.std([np.median(rng.choice(v, len(v))) for _ in range(200)]) if len(v) > 10 else np.nan)
        med, err = np.array(med), np.array(err)
        okb = np.isfinite(med)
        c = np.polyfit(bc[okb], med[okb], 1, w=1 / err[okb])
        s_front, s_back = c[1], c[1] + c[0]
        # errors by bootstrap over tracks is costly; use the bin errors propagated through the line fit
        A = np.vstack([np.ones(okb.sum()), bc[okb]]).T
        W = np.diag(1 / err[okb] ** 2)
        cov = np.linalg.inv(A.T @ W @ A)
        s_front_e = np.sqrt(cov[0, 0])
        s_back_e = np.sqrt(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1])
        axW[0].errorbar(bc + 0.005 * k, med, err, fmt="o-", color=cols[k], ms=4, label=amp_name(b, ch))
        # ---- direction / length distributions (clean sample)
        ang = np.degrees(np.arctan2(np.abs(sel.dy), np.abs(sel.dx)))
        axA[0].hist(ang, bins=np.linspace(0, 90, 10), histtype="step", density=True, color=cols[k], label=amp_name(b, ch))
        axA[1].hist(L[band], bins=np.linspace(0, 200, 41), histtype="step", density=True, color=cols[k],
                    label=amp_name(b, ch))
        # ---- deferred charge (CTI proxy) from bright clusters in clean images
        cl = d[(~d.edge) & (d.nsat == 0) & (d.Q > 3000)]
        rs, rse = boot_ratio((cl.trail_s - cl.lead_s).values, cl.edge_s.values, rng)
        rv, rve = boot_ratio((cl.trail_v - cl.lead_v).values, cl.edge_v.values, rng)
        # vs number of transfers: serial ~ column + 8, vertical ~ row
        cb = np.linspace(0, 6150, 7)
        ds = [boot_ratio((cl.trail_s - cl.lead_s).values[(cl.col >= cb[i]) & (cl.col < cb[i + 1])],
                         cl.edge_s.values[(cl.col >= cb[i]) & (cl.col < cb[i + 1])], rng, 100) for i in range(6)]
        ds = np.array(ds)
        cc = 0.5 * (cb[1:] + cb[:-1]) + 8
        okd = np.isfinite(ds[:, 0]) & (ds[:, 1] > 0)
        cs = np.polyfit(cc[okd], ds[okd, 0], 1, w=1 / ds[okd, 1], cov=True) if okd.sum() > 2 else (np.full(2, np.nan), np.full((2, 2), np.nan))
        rb_ = np.linspace(0, 1300, 6)
        dv = np.array([boot_ratio((cl.trail_v - cl.lead_v).values[(cl.row >= rb_[i]) & (cl.row < rb_[i + 1])],
                                  cl.edge_v.values[(cl.row >= rb_[i]) & (cl.row < rb_[i + 1])], rng, 100) for i in range(5)])
        rc_ = 0.5 * (rb_[1:] + rb_[:-1]) + 1
        okv = np.isfinite(dv[:, 0]) & (dv[:, 1] > 0)
        cvf = np.polyfit(rc_[okv], dv[okv, 0], 1, w=1 / dv[okv, 1], cov=True) if okv.sum() > 2 else (np.full(2, np.nan), np.full((2, 2), np.nan))
        axA[2].errorbar(cc + 20 * k, ds[:, 0], ds[:, 1], fmt="o-", color=cols[k], ms=4, label=amp_name(b, ch))
        pitch = q0 / MIP_E_PER_UM
        pitch_err = pitch * np.hypot(q0e / q0, 0.10)
        res.append(dict(board=b, ch=ch, amp=amp_name(b, ch), n_clusters_clean=len(d), n_muon_band=nb,
                        T_px=T, T_px_stat=Te, T_px_syst=T_syst, T_px_grid_mean=Ta, T_px_grid_min=min(grid), T_px_grid_max=max(grid),
                        T_um_15=15 * T, T_um_15_err=15 * np.hypot(Te, T_syst),
                        q0_e_per_px=q0, q0_err=q0e, pitch_um_from_dEdx=pitch, pitch_err=pitch_err,
                        T_um_dEdx=T * pitch, T_um_dEdx_err=T * pitch * np.hypot(np.hypot(Te, T_syst) / T, pitch_err / pitch),
                        sigma_front_px=s_front, sigma_front_err=s_front_e, sigma_back_px=s_back, sigma_back_err=s_back_e,
                        sigma_mean_px=np.sqrt(np.median(ends)) if len(ends) else np.nan, n_width_tracks=len(ends),
                        frac_azimuth_gt45=float(np.mean(ang > 45)), n_dir=len(ang),
                        defer_serial=rs, defer_serial_err=rse, defer_vertical=rv, defer_vertical_err=rve,
                        cti_serial_per_transfer=cs[0][0], cti_serial_err=np.sqrt(cs[1][0, 0]),
                        cti_vertical_per_transfer=cvf[0][0], cti_vertical_err=np.sqrt(cvf[1][0, 0]),
                        defer_vertical_intercept=cvf[0][1],
                        sigma_e_img=float(d.sigma_e.median())))
    axW[0].set_xlabel("relative position along track (0 = narrow end ≈ front/gates, 1 = wide end ≈ back)")
    axW[0].set_ylabel("transverse σ (px)")
    axW[0].set_title("Diffusion profile of muon tracks (stacked medians)", fontsize=10)
    axW[0].legend(fontsize=8)
    axW[0].grid(alpha=0.3)
    for k, r in enumerate(res):
        axW[1].errorbar(k, r["T_um_15"], r["T_um_15_err"], fmt="o", color=cols[k])
        axW[1].errorbar(k + 0.2, r["T_um_dEdx"], r["T_um_dEdx_err"], fmt="s", color=cols[k], mfc="none")
    axW[1].plot([], [], "ko", label="T_px × 15 µm")
    axW[1].plot([], [], "ks", mfc="none", label="T_px × pitch from dE/dx (±10 %)")
    axW[1].axhline(670, color="k", ls="--", lw=0.8, label="DAMIC-M nominal 670 µm")
    axW[1].set_xticks(range(len(res)))
    axW[1].set_xticklabels([r["amp"] for r in res], fontsize=8)
    axW[1].set_ylabel("active (charge-collecting) thickness (µm)")
    axW[1].set_title("Thickness from muon Q vs L", fontsize=10)
    axW[1].legend(fontsize=8)
    axW[1].grid(alpha=0.3)
    figW.tight_layout()
    figW.savefig(os.path.join(out, "tracks_diffusion_thickness.png"), dpi=110)
    figQ.tight_layout()
    figQ.savefig(os.path.join(out, "tracks_Q_vs_L.png"), dpi=110)
    axA[0].set_xlabel("track direction (deg; 0 = along a row/serial, 90 = along a column)")
    axA[0].set_ylabel("density")
    axA[0].set_title("Track azimuth (horizontal CCD + isotropic → flat)", fontsize=10)
    axA[0].legend(fontsize=8)
    axA[1].set_xlabel("projected length L (px)")
    axA[1].set_title("Projected length (muon band)", fontsize=10)
    axA[1].legend(fontsize=8)
    axA[2].axhline(0, color="k", lw=0.6)
    axA[2].set_xlabel("serial transfers (column + prescan)")
    axA[2].set_ylabel("(trailing − leading) / edge charge")
    axA[2].set_title("Serial deferred-charge proxy vs column", fontsize=10)
    axA[2].legend(fontsize=8)
    figA.tight_layout()
    figA.savefig(os.path.join(out, "tracks_angles_cti.png"), dpi=110)
    r = pd.DataFrame(res)
    r.to_csv(os.path.join(out, "tracks_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(r.T.to_string())
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.analyse_only:
        find_all(a.jobs, a.out)
    analyse(a.out)


if __name__ == "__main__":
    main()
