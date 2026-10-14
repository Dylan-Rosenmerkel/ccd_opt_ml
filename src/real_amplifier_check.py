"""
real_amplifier_check.py -- can the models work WITHOUT amplifier labels?  A check on the REAL amplifiers of module 101.

Three questions, each answered with data we already have (module 101 only, as for every trace analysis):

1. GAIN.  The trace measures noise in ADU; electrons need the gain, which differs between amplifiers.  How much
   does it differ?  From the 2026-10-06 1x1 image (3000 skips) the 0, 1, 2, ... electron peaks of every channel
   give its gain (peak spacing) and its noise (peak width), without WADERs.

2. READ NOISE FROM THE TRACE, NO LABEL.  Predict each channel's image noise (ADU, 3000 skips) from its own traces
   (2026-10-06 and 2026-10-08, nominal settings): white noise only, and sqrt(a^2/N + b^2) from the Allan curves.
   Then convert to electrons with (a) the channel's own gain (needs a label) and (b) the mean gain of the OTHER
   channels (no label for this one: what an amplifier-agnostic model would do).

3. DO THE AMPLIFIERS RESPOND ALIKE TO THE SAME CHANGE?  An amplifier-agnostic model learns ONE dependence on the
   settings for all amplifiers.  From the 2026-10-12 packet tests (settings 2-5, 3 trace sets each), compare
   every feature's change with the settings between channels.

    cd src && ../.venv/bin/python real_amplifier_check.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ml.structured import allan_fit  # noqa: E402
from trace_features import ALLAN_N, scalar_names, trace_features  # noqa: E402

DATA = HERE.parent / "data"
IMAGE = DATA / "2026-10-06-PA03_PQ04_Installation_Check130K_1" / "avg_img_100x1000x3000_1x1_L2_101_20261006_132140_1.fits"
N_IMAGE = 3000


# --------------------------------------------------------------------------- #
# 1. gain and noise from the electron peaks                                    #
# --------------------------------------------------------------------------- #

def electron_peaks(pixels: np.ndarray, bin_width: float = 0.01, span: tuple = (-6, 12)):
    """
    Gain (ADU/e-) = median spacing of the electron peaks; noise (ADU) = their mean Gaussian width (fitted on each
    peak's +-0.35 gain).  ``pixels``: row-baseline-subtracted values.  Returns (gain, sigma_adu, peak positions).
    """
    from scipy.optimize import curve_fit
    edges = np.arange(*span, bin_width)
    h, e = np.histogram(pixels, edges)
    c = 0.5 * (e[1:] + e[:-1])
    smooth = np.convolve(h, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(smooth, prominence=smooth.max() * 0.02, distance=int(0.6 / bin_width))
    pos = c[peaks]
    gain = float(np.median(np.diff(pos)))
    widths = []
    for p in pos:
        m = np.abs(c - p) < 0.35 * gain
        try:
            (A, mu, s), _ = curve_fit(lambda x, A, mu, s: A * np.exp(-0.5 * ((x - mu) / s) ** 2), c[m], h[m],
                                      p0=(h[m].max(), p, 0.15), maxfev=5000)
            widths.append(abs(s))
        except RuntimeError:
            pass
    return gain, float(np.median(widths)) if widths else np.nan, pos


def image_gains():
    out = {}
    h = fits.open(IMAGE)
    for ch in range(4):
        d = h[ch + 1].data.astype(float)
        r = (d - np.median(d, axis=1, keepdims=True)).ravel()
        g, s, pos = electron_peaks(r)
        out[ch] = {"gain": g, "sigma_adu": s, "sigma_e": s / g, "n_peaks": len(pos)}
    return out


# --------------------------------------------------------------------------- #
# 2. noise predicted from the traces                                           #
# --------------------------------------------------------------------------- #

def trace_predictions():
    files = {ch: [] for ch in range(4)}
    for day in ("2026-10-06-PA03_PQ04_Installation_Check130K_1", "2026-10-08-PA03_PQ04_traces"):
        for meta in sorted((DATA / day).glob("trace_*_101_*.meta")):
            ch = int(meta.name.split("_ch")[1][0])
            files[ch].append((meta, meta.with_suffix(f".RJ45ADCDATA{ch}")))
    out = {}
    for ch, lst in files.items():
        rows = []
        for meta, raw in lst:
            try:
                f = trace_features(raw, meta=json.loads(meta.read_text()))
            except ValueError:
                continue
            cds = f.scalars["skip: CDS rms (detrended)"]
            a1, b1 = allan_fit(f.curves["noise: Allan deviation"][None], ALLAN_N, max_n=100)
            a2, b2 = allan_fit(f.curves["noise: Allan deviation, common pattern removed"][None], ALLAN_N)
            rows.append({"white": cds / np.sqrt(N_IMAGE),
                         "detrended": float(np.sqrt(a1[0] ** 2 / N_IMAGE + b1[0] ** 2)),
                         "common removed": float(np.sqrt(a2[0] ** 2 / N_IMAGE + b2[0] ** 2)),
                         "b common removed": float(b2[0]), "cds": cds,
                         "reset scatter": f.scalars.get("noise: reset-level scatter (ADU)", np.nan)})
        out[ch] = {k: float(np.median([r[k] for r in rows])) for k in rows[0]} | {"n_traces": len(rows)}
    return out


# --------------------------------------------------------------------------- #
# 3. do the channels respond alike?                                            #
# --------------------------------------------------------------------------- #

def response_similarity():
    run = DATA / "2026-10-12-PA03_PQ04_Packet_Tests"
    metas = sorted(((json.loads(p.read_text()), p) for p in run.glob("*.meta")), key=lambda x: x[0]["DATEini"])
    keys = [k for k in metas[0][0] if k[:2] in ("V1", "V2", "V3", "H1", "H2", "H3", "TG", "SW", "OG", "RG", "DG")
            or k in ("VSUB", "VDD1", "VDD2", "VDRAIN1", "VDRAIN2", "VR1", "VR2") or k.startswith("delay")]
    groups, sig = [], None
    for m, p in metas:
        if m["ACMID"] != 101:
            continue
        s = json.dumps({k: m.get(k) for k in keys})
        if s != sig:
            groups.append([])
            sig = s
        groups[-1].append((m, p))
    names = scalar_names()
    vals = {}                                        # (setting, ch) -> array (n_sets, n_features)
    for g, files in enumerate(groups, start=1):
        for m, p in files:
            if not p.name.startswith("trace"):
                continue
            ch = int(p.name.split("_ch")[1][0])
            try:
                f = trace_features(p.with_suffix(f".RJ45ADCDATA{ch}"), meta=m)
            except ValueError:
                continue
            vals.setdefault((g, ch), []).append([f.scalars.get(n, np.nan) for n in names])
    vals = {k: np.array(v) for k, v in vals.items()}
    base = 2                                         # setting 1 had another trace setup (NCOLPRE 0)
    effects = {}                                     # ch -> (n_settings, n_features): change in units of the repeat scatter
    for ch in range(4):
        rows = []
        for g in (3, 4, 5):
            a, b = vals.get((base, ch)), vals.get((g, ch))
            if a is None or b is None or len(a) < 2 or len(b) < 2:
                rows.append(np.full(len(names), np.nan))
                continue
            pooled = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2)
            rows.append((b.mean(0) - a.mean(0)) / np.where(pooled > 0, pooled, np.nan))
        effects[ch] = np.array(rows)
    out = {"correlation": {}, "sign agreement": {}}
    for i in range(4):
        for j in range(i + 1, 4):
            x, y = effects[i].ravel(), effects[j].ravel()
            ok = np.isfinite(x) & np.isfinite(y) & ((np.abs(x) > 6) | (np.abs(y) > 6))     # changes that are real in one of them
            out["correlation"][f"ch{i}-ch{j}"] = float(np.corrcoef(np.sign(x[ok]) * np.log1p(np.abs(x[ok])),
                                                                   np.sign(y[ok]) * np.log1p(np.abs(y[ok])))[0, 1]) if ok.sum() > 3 else np.nan
            out["sign agreement"][f"ch{i}-ch{j}"] = float(np.mean(np.sign(x[ok]) == np.sign(y[ok]))) if ok.sum() else np.nan
            out.setdefault("n compared", {})[f"ch{i}-ch{j}"] = int(ok.sum())
    for g_index, g in enumerate((3, 4, 5)):
        e = np.array([effects[ch][g_index] for ch in range(4)])
        real = np.abs(e) > 6
        out.setdefault("features changed", {})[f"setting {g}"] = {f"ch{ch}": int(real[ch].sum()) for ch in range(4)}
        out.setdefault("changed on all 4", {})[f"setting {g}"] = int(np.all(real, axis=0).sum())
    noise = names.index("skip: CDS rms (detrended)")
    out["CDS rms change (repeat-scatter units)"] = {f"setting {g}": {f"ch{ch}": round(float(effects[ch][k, noise]), 1) for ch in range(4)}
                                                    for k, g in enumerate((3, 4, 5))}
    return out


def main():
    print("== 1. gain and noise of every channel from the 2026-10-06 1x1 image (3000 skips)")
    img = image_gains()
    for ch, v in img.items():
        print(f"  ch{ch}: gain {v['gain']:.3f} ADU/e-, noise {v['sigma_adu']:.3f} ADU = {v['sigma_e']:.3f} e-  ({v['n_peaks']} peaks)")
    # a gain is only measured when the electron peaks are resolved (noise well below the spacing); on ch2 the noise
    # (~0.7 ADU) is as large as the spacing and the "peaks" are noise bumps (WADERs fails there too)
    good = [ch for ch in img if np.isfinite(img[ch]["gain"]) and img[ch]["sigma_adu"] < 0.3 * img[ch]["gain"]]
    for ch in img:
        if ch not in good:
            print(f"  ch{ch}: electron peaks NOT resolved (noise {img[ch]['sigma_adu']:.2f} ADU vs spacing); its gain is unknown")
    g = np.array([img[ch]["gain"] for ch in good])
    print(f"  gain spread over channels {good}: {100 * g.std() / g.mean():.1f} % (sd/mean), range {g.min():.3f}-{g.max():.3f}")

    print("\n== 2. image noise predicted from each channel's own traces (ADU, 3000 skips)")
    tr = trace_predictions()
    print(f"  {'ch':3s} {'traces':>6s} {'image':>7s} {'white':>7s} {'detrended a,b':>14s} {'common removed a,b':>19s}")
    for ch in range(4):
        t = tr[ch]
        print(f"  ch{ch} {t['n_traces']:6d} {img[ch]['sigma_adu']:7.3f} {t['white']:7.3f} {t['detrended']:14.3f} {t['common removed']:19.3f}")
    print("\n   in electrons: image | trace (common removed) / own gain [needs a label] | / mean gain of the other channels [no label]")
    for ch in good:
        others = np.mean([img[c]["gain"] for c in good if c != ch])
        own = tr[ch]["common removed"] / img[ch]["gain"]
        blind = tr[ch]["common removed"] / others
        print(f"  ch{ch}: {img[ch]['sigma_e']:.3f} e- | {own:.3f} ({100 * (own / img[ch]['sigma_e'] - 1):+.0f} %) | "
              f"{blind:.3f} ({100 * (blind / img[ch]['sigma_e'] - 1):+.0f} %);  gain error without a label {100 * (others / img[ch]['gain'] - 1):+.1f} %")

    print("\n== 3. do the channels respond alike to the same change of settings? (2026-10-12, settings 3-5 vs 2)")
    rs = response_similarity()
    for k in ("features changed", "changed on all 4", "CDS rms change (repeat-scatter units)", "sign agreement", "correlation", "n compared"):
        print(f"  {k}: {rs[k]}")
    (HERE.parent / "docs" / "ml_results" / "real_amplifier_check.json").write_text(json.dumps(
        {"image": {str(k): v for k, v in img.items()}, "traces": {str(k): v for k, v in tr.items()}, "responses": rs}, indent=1, default=float))


if __name__ == "__main__":
    main()
