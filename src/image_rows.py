"""
image_rows.py -- the dark charge of every image ROW, and its split into dark current and CIC.

WADERs gives one lambda per amplifier for the whole image.  The rows of an image did not wait equally long:
with EXPOSURE 0, a pixel collects dark current while the rows read before it are transferred, so later rows
collect more.  Row by row,

    lambda_r = CIC + DC * x_r,      x_r = (r + 1/2) / (R / 2),   r = 0 .. R-1 in readout order,

with x_r the row's relative waiting time (mean 1 over the image).  The intercept is the part every row gets
(clock-induced charge; with EXPOSURE > 0 also the exposure's dark charge), the slope DC the dark current of an
average row.  A weighted straight-line fit through the R values (weights from the variance expected at the fitted line, which
avoids the low bias of weighting by each row's own error) gives both, with their 2x2 covariance (the two
are strongly anti-correlated; docs/overleaf_ml_guide/ml_guide.tex, "Dark current and CIC").

Per row, lambda_r is fitted by maximum likelihood with WADERs' gain g and noise sigma (ADU) held fixed:

    x ~ sum_k Poisson(k; lambda_r) * Normal(mu_r + k g, sigma),    k = 0 .. 6,

over the row's active pixels (prescan and overscan columns left out, using the .meta geometry), mu_r being the
row's pedestal (median of the pixels within 3 sigma of the 0-electron peak).  The error is from the curvature of
the log-likelihood.

ASSUMPTION to check on the first real images: FITS row 0 is the first row read out.  If the fitted DC comes out
negative on every image, the order is reversed: set ``[rows] reverse = true`` in configs/processing.toml.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm, poisson

K_MAX = 6


def active_columns(meta: dict, ncol: int) -> np.ndarray:
    """Boolean mask of the (binned) columns that lie on the CCD's active area (not prescan, not overscan)."""
    prescan = int(meta.get("PRESCAN", 8))
    physcol = int(meta.get("PHYSCOl", meta.get("PHYSCOL", 6144)))
    colpre = int(meta.get("NCOLPRE", 0))
    sbin = max(1, int(meta.get("NSBIN", 1)))
    start = colpre + np.arange(ncol) * sbin
    return (start >= prescan) & (start + sbin <= prescan + physcol)


def _nll(lam, x, mu, g, s):
    k = np.arange(K_MAX + 1)
    like = (poisson.pmf(k, lam)[None, :] * norm.pdf(x[:, None], mu + k[None, :] * g, s)).sum(1)
    return -np.sum(np.log(np.maximum(like, 1e-300)))


def row_lambda(pixels: np.ndarray, gain: float, sigma: float) -> tuple[float, float]:
    """(lambda, error) of one row's pixels (ADU, any pedestal) with the gain and noise fixed."""
    x = np.asarray(pixels, float)
    x = x[np.isfinite(x)]
    if len(x) < 20 or not (gain > 0 and sigma > 0):
        return float("nan"), float("nan")
    med = np.median(x)
    core = x[np.abs(x - med) < 3 * sigma]
    mu = float(np.median(core)) if len(core) > 10 else float(med)
    x = x[(x > mu - 6 * sigma) & (x < mu + (K_MAX + 0.5) * gain)]
    res = minimize_scalar(_nll, bounds=(1e-6, 5.0), args=(x, mu, gain, sigma), method="bounded",
                          options={"xatol": 1e-7})
    lam = float(res.x)
    h = max(1e-4, 1e-2 * lam)
    curv = (_nll(lam + h, x, mu, gain, sigma) - 2 * _nll(lam, x, mu, gain, sigma) + _nll(max(lam - h, 1e-7), x, mu, gain, sigma)) / h ** 2
    err = 1 / math.sqrt(curv) if curv > 0 else float("nan")
    return lam, err


def line_fit(lam_rows: np.ndarray, err_rows: np.ndarray) -> dict:
    """Weighted straight line lambda_r = CIC + DC x_r through the rows; DC, CIC, their errors and covariance."""
    lam_rows, err_rows = np.asarray(lam_rows, float), np.asarray(err_rows, float)
    R = len(lam_rows)
    xr = (np.arange(R) + 0.5) / (R / 2)
    ok = np.isfinite(lam_rows) & np.isfinite(err_rows) & (err_rows > 0)
    if ok.sum() < 3:
        return {"n_rows_fit": int(ok.sum())}
    A = np.column_stack([xr[ok], np.ones(ok.sum())])
    y, var = lam_rows[ok], err_rows[ok] ** 2
    W = 1 / var
    for _ in range(3):
        cov = np.linalg.inv(A.T @ (A * W[:, None]))
        dc, cic = cov @ (A.T @ (W * y))
        # a row's error is dominated by counting (var ~ lambda): weights from the observed lambda_r favour rows that
        # fluctuated low and bias the line low; re-weight with the variance expected at the FITTED value
        fit = np.maximum(A @ np.array([dc, cic]), 1e-6)
        W = 1 / (var * fit / np.maximum(y, 1e-6))
    chi2 = float(np.sum(W * (lam_rows[ok] - A @ np.array([dc, cic])) ** 2))
    return {"lambda_dark": float(dc), "lambda_cic": float(cic), "lambda_dark_err": float(math.sqrt(cov[0, 0])),
            "lambda_cic_err": float(math.sqrt(cov[1, 1])), "lambda_dark_cic_cov": float(cov[0, 1]),
            "rows_chi2_dof": chi2 / max(ok.sum() - 2, 1), "n_rows_fit": int(ok.sum())}


def image_rows(fits_path, labels: list[dict], meta: dict, reverse: bool = False) -> dict[int, dict]:
    """
    Per channel (from the WADERs labels: gain_adu_per_e, sigma_adu): the per-row lambda, its errors, and the
    dark-current / CIC line fit.  ``fits_path``: the image builder's avg_*.fits (HDU ch+1 = channel ch).
    """
    from astropy.io import fits
    out = {}
    with fits.open(fits_path) as hdul:
        for lab in labels:
            ch = lab["channel"]
            g, s = lab.get("gain_adu_per_e"), lab.get("sigma_adu")
            if g is None or s is None or ch + 1 >= len(hdul) or hdul[ch + 1].data is None:
                continue
            data = np.asarray(hdul[ch + 1].data, float)
            if reverse:
                data = data[::-1]
            cols = active_columns(meta, data.shape[1])
            fits_rows = [row_lambda(row[cols], g, s) for row in data]
            lam = np.array([f[0] for f in fits_rows])
            err = np.array([f[1] for f in fits_rows])
            out[ch] = {"rows_lambda": lam.tolist(), "rows_lambda_err": err.tolist(),
                       "n_active_columns": int(cols.sum()), **line_fit(lam, err)}
    return out
