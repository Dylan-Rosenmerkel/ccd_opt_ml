"""
l1_features.py -- features of a level-1 strip (src/decode_l1.py): a small image read exactly like the real one, with
every skip of every pixel kept.  They are a second input source next to the trace features (trace_features.py), so
the level-1 models can be compared with the trace models on the same packets (docs/context/2026-11-03, section 13).

The image's labels come from its pixel MEANS over all its skips (WADERs fits the 0, 1, 2 ... electron peaks of them).
A strip gives the same quantity on fewer pixels, plus what a trace gives (noise against the number of skips):

    l1: pixel noise (ADU)            width of the 0-electron peak of the pixel means over ALL skips (as the image),
                                     from a fit of the electron peaks (Poisson x Gaussian, below)
    l1: gain (ADU/e)                 spacing of the electron peaks of that fit
    l1: lambda (e/pixel)             mean charge per pixel of that fit
    l1: sigma_e (e)                  pixel noise / gain: the strip's own sigma_e
    l1: white noise per skip (ADU)   within-pixel noise of one skip (consecutive skips differenced, robust)
    l1: Allan deviation n=<N> (ADU)  within-pixel noise of N-skip block means (skips after the first SKIP_FIRST)
    l1: slow-noise excess            pixel noise / (white noise / sqrt(skips)): 1 = white noise only; > 1 = noise slower
                                     than a pixel (drifts between pixels) that averaging does not remove
    l1: transient at skip 0 (ADU)    mean of skip 0 minus the mean of the skips after SKIP_FIRST (all pixels)
    l1: transient decay (skips)      skips until the transient falls below 1/e of its skip-0 value
    l1: pixel-mean autocorrelation   lag-1 correlation of the pixel means along a row (0-electron pixels): slow drifts
    l1: row slope (ADU/100 px)       slope of the pixel means along the row (0-electron pixels)
    l1: low-frequency skip power     share of the skip-sequence power in the lowest 10 % of frequencies

Curves (log-spaced N): ``l1: Allan deviation`` (within pixel) and ``l1: image noise vs skips`` (robust spread of the
first-N-skip means of the 0-electron pixels, with the row's pedestal removed), ``l1: transient`` (mean of skips
0..49 minus the late mean), ``l1: skip spectrum``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, poisson

#: skips dropped at the start of every pixel for the within-pixel noise (the start transient; skip_noise_L1.py: 20)
SKIP_FIRST = 20
K_MAX = 6


def robust_std(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def n_values(nmax: int) -> list[int]:
    return sorted({int(round(v)) for v in np.logspace(0, np.log10(max(nmax, 1)), 20)} | {nmax})


def fit_peaks(x: np.ndarray, gain0: float = 1.3) -> dict:
    """Maximum-likelihood fit of the pixel means x (ADU, pedestal-subtracted) to sum_k Poisson(k; lam) Normal(mu + k g,
    s): the strip's own sigma (ADU), gain and lambda.  Returns NaNs if the fit fails or the peaks are not resolved."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return {"sigma": math.nan, "gain": math.nan, "lambda": math.nan}
    s0 = robust_std(x[x < np.median(x) + 0.5 * gain0])
    k = np.arange(K_MAX + 1)

    def nll(p):
        mu, ls, lg, ll = p
        s, g, lam = math.exp(ls), math.exp(lg), math.exp(ll)
        like = (poisson.pmf(k, lam)[None, :] * norm.pdf(x[:, None], mu + k[None, :] * g, s)).sum(1)
        return -np.sum(np.log(np.maximum(like, 1e-300)))

    best = None
    for g0 in (gain0 * 0.9, gain0, gain0 * 1.1):
        r = minimize(nll, [np.median(x[x < np.median(x) + 0.5 * g0]), math.log(max(s0, 1e-3)), math.log(g0), math.log(0.3)],
                     method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    mu, ls, lg, ll = best.x
    s, g, lam = math.exp(ls), math.exp(lg), math.exp(ll)
    if not (0.3 * gain0 < g < 3 * gain0) or s > 0.45 * g:                # peaks not resolved: no estimate
        return {"sigma": math.nan, "gain": math.nan, "lambda": math.nan}
    return {"sigma": s, "gain": g, "lambda": lam, "mu": mu}


def l1_features(values: np.ndarray, rows: np.ndarray, gain0: float = 1.3) -> tuple[dict, dict]:
    """(scalars, curves) of one channel's strip: values (pixels, skips) ADU, rows (pixels,) readout row."""
    v = np.asarray(values, float)
    rows = np.asarray(rows)
    npx, nskip = v.shape
    S: dict[str, float] = {}
    C: dict[str, np.ndarray] = {}
    # pixel means over ALL skips (as the image), the row's pedestal removed
    means = v.mean(1)
    for r in np.unique(rows):
        m = rows == r
        means[m] -= np.median(means[m])
    fit = fit_peaks(means, gain0)
    S["l1: pixel noise (ADU)"] = fit["sigma"]
    S["l1: gain (ADU/e)"] = fit["gain"]
    S["l1: lambda (e/pixel)"] = fit["lambda"]
    S["l1: sigma_e (e)"] = fit["sigma"] / fit["gain"] if np.isfinite(fit["gain"]) else math.nan
    # 0-electron pixels (for the drift statistics): below half a gain above the 0-electron peak
    g = fit["gain"] if np.isfinite(fit["gain"]) else gain0
    mu0 = fit.get("mu", 0.0)
    zero = means < mu0 + 0.5 * g
    # within-pixel noise (after the transient)
    w = v[:, SKIP_FIRST:] if nskip > SKIP_FIRST + 2 else v
    m_used = w.shape[1]
    S["l1: white noise per skip (ADU)"] = float(np.median([robust_std(np.diff(p) / math.sqrt(2)) for p in w]))
    ns = n_values(max(m_used // 2, 1))
    allan = []
    for n in ns:
        nb = m_used // n
        if nb < 2:
            allan.append(math.nan)
            continue
        b = w[:, :nb * n].reshape(npx, nb, n).mean(2)
        allan.append(float(np.median([robust_std(np.diff(x) / math.sqrt(2)) for x in b])) if nb > 2 else
                     robust_std((b[:, 1] - b[:, 0]) / math.sqrt(2)))
    C["l1: Allan deviation"] = np.array(allan)
    C["l1: Allan n"] = np.array(ns, float)
    for n_pick in (10, 50):
        j = int(np.argmin(np.abs(np.array(ns) - n_pick)))
        S[f"l1: Allan deviation n={n_pick} (ADU)"] = allan[j]
    # image-style noise of the first N skips (0-electron pixels, row pedestal removed)
    img = []
    nsi = n_values(nskip)
    for n in nsi:
        mn = v[:, :n].mean(1)
        for r in np.unique(rows):
            m = rows == r
            mn[m] -= np.median(mn[m])
        img.append(robust_std(mn[zero]) if zero.sum() > 20 else math.nan)
    C["l1: image noise vs skips"] = np.array(img)
    C["l1: image noise n"] = np.array(nsi, float)
    white = S["l1: white noise per skip (ADU)"]
    S["l1: slow-noise excess"] = (S["l1: pixel noise (ADU)"] / (white / math.sqrt(nskip))
                                  if np.isfinite(S["l1: pixel noise (ADU)"]) and white > 0 else math.nan)
    # start-of-pixel transient
    late = v[:, SKIP_FIRST:].mean() if nskip > SKIP_FIRST else v.mean()
    prof = v[:, :min(50, nskip)].mean(0) - late
    C["l1: transient"] = prof
    S["l1: transient at skip 0 (ADU)"] = float(prof[0])
    below = np.flatnonzero(np.abs(prof) < abs(prof[0]) / math.e)
    S["l1: transient decay (skips)"] = float(below[0]) if len(below) else float(len(prof))
    # slow drifts along the row (0-electron pixels)
    ac, sl = [], []
    for r in np.unique(rows):
        m = (rows == r) & zero
        x = means[m]
        if len(x) > 20:
            ac.append(np.corrcoef(x[:-1], x[1:])[0, 1])
            sl.append(np.polyfit(np.arange(len(x)), x, 1)[0] * 100)
    S["l1: pixel-mean autocorrelation"] = float(np.mean(ac)) if ac else math.nan
    S["l1: row slope (ADU/100 px)"] = float(np.mean(sl)) if sl else math.nan
    # skip spectrum (mean over pixels of |FFT|^2 of the skip sequence after the transient)
    if m_used >= 16:
        spec = (np.abs(np.fft.rfft(w - w.mean(1, keepdims=True), axis=1)) ** 2).mean(0)[1:]
        lo = max(1, len(spec) // 10)
        S["l1: low-frequency skip power"] = float(spec[:lo].sum() / spec.sum()) if spec.sum() > 0 else math.nan
        edges = np.unique(np.logspace(0, np.log10(len(spec)), 24).astype(int))
        C["l1: skip spectrum"] = np.array([spec[a - 1:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    else:
        S["l1: low-frequency skip power"] = math.nan
    return S, C


#: the scalar names, in order (for datasets and model inputs)
SCALARS = ["l1: pixel noise (ADU)", "l1: gain (ADU/e)", "l1: lambda (e/pixel)", "l1: sigma_e (e)",
           "l1: white noise per skip (ADU)", "l1: Allan deviation n=10 (ADU)", "l1: Allan deviation n=50 (ADU)",
           "l1: slow-noise excess", "l1: transient at skip 0 (ADU)", "l1: transient decay (skips)",
           "l1: pixel-mean autocorrelation", "l1: row slope (ADU/100 px)", "l1: low-frequency skip power"]
