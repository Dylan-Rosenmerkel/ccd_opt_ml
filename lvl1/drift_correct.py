"""
drift_correct.py -- detect the along-row baseline drift of an averaged skipper image and, if present, correct it.

What is known about the drift (lvl1/real_image_checks/2026-10-16_PA03_PQ04_16rows.md):
* the zero-electron level of the active region starts up to ~1 ADU below the row's overscan pedestal at the start of
  the readout and recovers over a few hundred columns; the same in every row;
* the whole electron comb shifts with it: the gain (1e- − 0e- spacing) does not change along the row, so in columns
  where only the level is shifted, subtracting it is a pedestal correction like the per-row one WADERs already makes;
* the noise (width of the zero-electron peak) is raised at the start of the row (+30 % in the first 25 columns): no
  subtraction can repair that, so those columns are CUT, not corrected.

The method, per channel:
1. per row, subtract the overscan median (as WADERs does);
2. in column blocks (fine at the start, coarser later), fit a Gaussian to the zero-electron peak: its position
   (offset) and width, with errors -- no fixed window, so a shifted peak is followed, not clipped;
3. DRIFT DETECTED if the offsets are inconsistent with a constant (chi2 of a flat line, p < 1e-6) AND the full range of
   the offsets exceeds 0.1 x the zero-peak width -- otherwise the image is left untouched;
4. CUT the first columns, where the noise is raised.  DEFAULT: a FIXED cut of 200 columns, the same for every packet (the
   columns used must be the same in every packet: λ grows along the row).  It was calibrated on the five 2026-10-16
   images (80 rows of board 101, drift removed first): the zero-peak width is +15-18 % in columns 0-24, +5-10 % up to
   ~175, and consistent with the end of the row (within ±3 %) from column 200 in ch0, ch1 and ch3.  ``--cut auto`` instead
   finds a cut per image (the first 100-column window within ``width_tol`` of the row end): a diagnostic, too noisy on a
   single 16-row image to define the labels;
5. CORRECT the remaining columns: fit offset(c) = A1 exp(-c/t1) + A2 exp(-c/t2) + C to the block offsets (two
   exponential recoveries and a constant); if that fit is poor (chi2/dof > 3), a smooth spline through the blocks is used
   instead; subtract the curve from every row (the overscan is not changed);
6. CHECK: the drift profile of the first and second half of the rows must agree (chi2/dof of their difference < 5; else
   a WARNING in the output and the header, DRIFTWRN: the assumption "the same in every row" fails and the correction is
   not trustworthy for that image), and the offsets and widths after the correction are reported.

The corrected image keeps the original format: the cut columns are removed and the header's NCOLPRE / NCOL are adjusted
(they are read before the kept columns), so WADERs and run_WADERs find the right geometry.  Header keywords: DRIFTCOR
(T/F), DRIFTCUT (columns cut), and per extension DRIFTA1/T1/A2/T2/C (the model) or DRIFTSPL (spline used).  The cut is the
same for all channels of an image (the largest needed), so the channels stay aligned.

    python lvl1/drift_correct.py <avg image(s)> --out <dir> [--cut auto|N] [--width-tol 0.05] [--max-cut 400]
           [--channels 0 1 3] [--report report.json]

For every image it prints the decision and writes <out>/<image>.fits (corrected, or an unchanged copy) and a JSON
report.  It does not change anything in data/.
"""

from __future__ import annotations

import argparse
import bz2
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist

GAUSS = lambda x, a, m, s: a * np.exp(-0.5 * ((x - m) / s) ** 2)
EXPS = lambda c, a1, t1, a2, t2, k: a1 * np.exp(-c / t1) + a2 * np.exp(-c / t2) + k


def open_any(path: Path):
    from astropy.io import fits
    raw = path.read_bytes()
    return fits.open(io.BytesIO(bz2.decompress(raw) if path.suffix == ".bz2" else raw))


def n_active(header, ncol: int) -> int:
    nsbin = int(header.get("NSBIN", 1) or 1)
    return min(int((header.get("PHYSCOL", 6144) + header.get("PRESCAN", 8) - header.get("NCOLPRE", 0)) / nsbin), ncol)


def block_edges(na: int) -> np.ndarray:
    """Finer blocks where the drift changes fast: 10 columns up to 100, 25 up to 400, 50 beyond."""
    e = list(range(0, min(100, na), 10)) + list(range(100, min(400, na), 25)) + list(range(400, na, 50))
    return np.array(sorted(set(e + [na])))


def zero_peak(x: np.ndarray, s_guess: float) -> tuple[float, float, float, float]:
    """Position and width of the zero-electron peak (and their errors) from a Gaussian fit around the histogram mode."""
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, 1) - 1, np.percentile(x, 60) + 1
    h, e = np.histogram(x, max(40, int((hi - lo) / (s_guess / 5))), (lo, hi))
    c = 0.5 * (e[1:] + e[:-1])
    m = c[np.argmax(np.convolve(h, np.ones(5) / 5, "same"))]
    for _ in range(3):                                           # re-centre the fit window on the peak
        sel = np.abs(x - m) < 2.5 * s_guess
        hh, ee = np.histogram(x[sel], 30, (m - 2.5 * s_guess, m + 2.5 * s_guess))
        cc = 0.5 * (ee[1:] + ee[:-1])
        try:
            p, cov = curve_fit(GAUSS, cc, hh, p0=[hh.max(), m, s_guess], sigma=np.sqrt(np.maximum(hh, 1)), maxfev=5000)
        except RuntimeError:
            return np.nan, np.nan, np.nan, np.nan
        m, s = p[1], abs(p[2])
        s_guess = s
    err = np.sqrt(np.diag(cov))
    return float(m), float(err[1]), float(s), float(err[2])


def profile(act: np.ndarray, s_guess: float, edges: np.ndarray) -> dict:
    out = {"c": [], "off": [], "off_err": [], "width": [], "width_err": []}
    for a, b in zip(edges[:-1], edges[1:]):
        m, me, s, se = zero_peak(act[:, a:b].ravel(), s_guess)
        out["c"].append(0.5 * (a + b - 1)); out["off"].append(m); out["off_err"].append(me)
        out["width"].append(s); out["width_err"].append(se)
    return {k: np.array(v) for k, v in out.items()}


def analyse_channel(d: np.ndarray, na: int, width_tol: float) -> dict:
    ovs = d[:, na + 8:]
    ped = np.median(ovs, 1)
    s_ovs = float(np.median(1.4826 * np.median(np.abs(ovs - np.median(ovs, 1, keepdims=True)), 1)))
    act = d[:, :na] - ped[:, None]
    edges = block_edges(na)
    pr = profile(act, s_ovs, edges)
    ok = np.isfinite(pr["off"]) & (pr["off_err"] > 0)
    w = 1 / pr["off_err"][ok] ** 2
    mean = np.sum(w * pr["off"][ok]) / np.sum(w)
    chi2 = float(np.sum(w * (pr["off"][ok] - mean) ** 2))
    dof = int(ok.sum() - 1)
    width_ref = float(np.nanmedian(pr["width"][pr["c"] > 0.6 * na]))
    rng = float(np.nanmax(pr["off"][ok]) - np.nanmin(pr["off"][ok]))
    drift = bool(chi2_dist.sf(chi2, dof) < 1e-6 and rng > 0.1 * width_ref)
    # the columns with raised noise: blocks whose width exceeds the reference by more than width_tol (and 3 errors)
    raised = (pr["width"] > width_ref * (1 + width_tol)) & (pr["width"] - width_ref * (1 + width_tol) > 3 * pr["width_err"])
    last_raised = int(edges[1:][raised].max()) if raised.any() else 0
    # the two halves of the rows must show the same drift
    half = len(d) // 2
    p1, p2 = profile(act[:half], s_ovs, edges), profile(act[half:], s_ovs, edges)
    diff = (p1["off"] - p2["off"]) / np.hypot(p1["off_err"], p2["off_err"])
    halves_chi2 = float(np.nansum(diff ** 2) / max(np.isfinite(diff).sum(), 1))
    return {"sigma_ovs": s_ovs, "width_ref": width_ref, "drift": drift, "flat_chi2_dof": chi2 / max(dof, 1),
            "range": rng, "raised_noise_until": last_raised, "halves_chi2_dof": halves_chi2, "profile": pr, "act": act,
            "ped": ped}


def fit_model(pr: dict, start: int) -> tuple[np.ndarray, dict]:
    """The baseline curve for columns >= start: two exponentials + constant, or a spline if that fits poorly."""
    sel = (pr["c"] >= start) & np.isfinite(pr["off"]) & (pr["off_err"] > 0)
    c, y, e = pr["c"][sel], pr["off"][sel], pr["off_err"][sel]
    info = {}
    try:
        p, _ = curve_fit(EXPS, c, y, p0=[y[0] - y[-1], 50, (y[0] - y[-1]) / 2, 400, y[-1]], sigma=e,
                         bounds=([-5, 1, -5, 30, -2], [5, 300, 5, 5000, 2]), maxfev=20000)
        resid = (y - EXPS(c, *p)) / e
        chi = float(np.sum(resid ** 2) / max(len(c) - 5, 1))
        info = {"model": "2 exponentials + constant", "A1": p[0], "t1": p[1], "A2": p[2], "t2": p[3], "C": p[4], "chi2_dof": chi}
        if chi <= 3:
            return (lambda cols: EXPS(cols, *p)), info
    except (RuntimeError, ValueError):
        pass
    from scipy.interpolate import UnivariateSpline
    spl = UnivariateSpline(c, y, w=1 / e, s=len(c))
    info.update({"model": "smoothing spline (exponential fit poor)", "spline": True})
    return spl, info


def correct_image(path: Path, out: Path, cut_mode: str, width_tol: float, max_cut: int, channels: list[int]) -> dict:
    h = open_any(path)
    H = h[0].header
    hdus = [x for x in h[1:] if getattr(x, "data", None) is not None and x.data.ndim == 2]
    na = n_active(H, hdus[0].data.shape[1])
    report = {"image": path.name, "n_active": na, "channels": {}}
    results = {}
    for ch in channels:
        r = analyse_channel(hdus[ch].data.astype(float), na, width_tol)
        results[ch] = r
        report["channels"][ch] = {k: r[k] for k in ("sigma_ovs", "width_ref", "drift", "flat_chi2_dof", "range",
                                                    "raised_noise_until", "halves_chi2_dof")}
    any_drift = any(r["drift"] for r in results.values())
    report["drift"] = any_drift
    stem = path.name.split(".fits")[0]
    if not any_drift:
        H["DRIFTCOR"] = (False, "drift_correct.py: no significant along-row drift")
        h.writeto(out / f"{stem}.fits", overwrite=True)
        return report
    if cut_mode == "auto":
        cuts = []
        for i, r in results.items():
            if not r["drift"]:
                continue
            curve, _ = fit_model(r["profile"], 0)
            act = r["act"] - curve(np.arange(na, dtype=float))[None, :]
            ref = zero_peak(act[:, int(0.6 * na):].ravel(), r["sigma_ovs"])[2]
            c_ok = max_cut
            for c0 in range(0, max_cut + 1, 25):
                w = zero_peak(act[:, c0:c0 + 100].ravel(), r["sigma_ovs"])[2]
                if w <= ref * (1 + width_tol):
                    c_ok = c0
                    break
            report["channels"][i]["noise_cut"] = c_ok
            cuts.append(c_ok)
        cut = max(cuts) if cuts else 0
    else:
        cut = int(cut_mode)
    report["cut"] = cut
    # the "same in every row" check on the columns that are KEPT (the cut ones do not matter)
    for i, r in results.items():
        if r["drift"]:
            half, act = len(r["act"]) // 2, r["act"][:, cut:]
            e = block_edges(na - cut)
            p1, p2 = profile(act[:half], r["sigma_ovs"], e), profile(act[half:], r["sigma_ovs"], e)
            diff = (p1["off"] - p2["off"]) / np.hypot(p1["off_err"], p2["off_err"])
            r["halves_chi2_dof"] = float(np.nansum(diff ** 2) / max(np.isfinite(diff).sum(), 1))
            report["channels"][i]["halves_chi2_dof_kept"] = r["halves_chi2_dof"]
    warn = [i for i, r in results.items() if r["drift"] and r["halves_chi2_dof"] > 5]
    report["warning"] = (f"channels {warn}: the drift differs between the first and second half of the rows "
                         "(the correction assumes it is the same in every row)") if warn else ""
    for i, x in enumerate(hdus):
        d = x.data.astype(float)
        if i in results and results[i]["drift"]:
            curve, info = fit_model(results[i]["profile"], cut)
            cols = np.arange(na, dtype=float)
            corr = np.zeros(d.shape[1])
            corr[:na] = curve(cols)
            d = d - corr[None, :]
            report["channels"][i]["model"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in info.items()}
            for k in ("A1", "t1", "A2", "t2", "C"):
                if k in info:
                    x.header[f"DRIFT{k.upper()}"] = (float(info[k]), "drift_correct.py model")
            x.header["DRIFTSPL"] = (bool(info.get("spline", False)), "spline instead of the exponential model")
            # after: the profile of the corrected, cut image
            act2 = (d[:, :na] - results[i]["ped"][:, None])[:, cut:]
            pr2 = profile(act2, results[i]["sigma_ovs"], block_edges(na - cut))
            report["channels"][i]["after"] = {"offset_range": float(np.nanmax(pr2["off"]) - np.nanmin(pr2["off"])),
                                              "width_first_block": float(pr2["width"][0]), "width_last": float(np.nanmedian(pr2["width"][-5:]))}
        x.data = d[:, cut:].astype(np.float32)
    H["NCOLPRE"] = H.get("NCOLPRE", 0) + cut * int(H.get("NSBIN", 1) or 1)
    H["NCOL"] = H.get("NCOL", hdus[0].data.shape[1] + cut) - cut
    H["DRIFTCOR"] = (True, "drift_correct.py: along-row baseline drift corrected")
    H["DRIFTCUT"] = (cut, "first columns removed (raised noise); NCOLPRE/NCOL adjusted")
    H["DRIFTWRN"] = (bool(report["warning"]), "drift not the same in every row: correction not trustworthy")
    h.writeto(out / f"{stem}.fits", overwrite=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("images", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cut", default="200", help="columns to cut where a drift is found (default 200, calibrated); 'auto' per image (diagnostic)")
    parser.add_argument("--width-tol", type=float, default=0.05)
    parser.add_argument("--max-cut", type=int, default=400)
    parser.add_argument("--channels", type=int, nargs="*", default=[0, 1, 3], help="channels used to decide (default: the good ones of board 101)")
    parser.add_argument("--report")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for img in args.images:
        rep = correct_image(Path(img), out, args.cut, args.width_tol, args.max_cut, args.channels)
        reports.append(rep)
        line = f"{rep['image'][:60]}: " + ("drift -> cut %d columns, corrected" % rep["cut"] if rep["drift"] else "no drift, unchanged")
        if rep.get("warning"):
            line += "   WARNING: " + rep["warning"]
        print(line)
        for ch, r in rep["channels"].items():
            extra = ""
            if "model" in r:
                m = r["model"]
                extra = (f" | model: {m['model']}" + (f" A1 {m['A1']:+.2f} t1 {m['t1']:.0f} A2 {m['A2']:+.2f} t2 {m['t2']:.0f} C {m['C']:+.2f} (chi2/dof {m['chi2_dof']:.1f})" if "A1" in m else "")
                         + f" | after: offset range {r['after']['offset_range']:.3f}, width first/last {r['after']['width_first_block']:.3f}/{r['after']['width_last']:.3f}")
            print(f"   ch{ch}: drift {r['drift']} (flat chi2/dof {r['flat_chi2_dof']:.0f}, range {r['range']:.2f} ADU), width ref {r['width_ref']:.3f}, "
                  f"noise back to normal from col {r.get('noise_cut', '-')}, halves chi2/dof {r['halves_chi2_dof']:.1f}{extra}")
    if args.report:
        Path(args.report).write_text(json.dumps(reports, indent=1, default=float))


if __name__ == "__main__":
    main()
