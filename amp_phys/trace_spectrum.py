"""
trace_spectrum.py -- noise of ONE trace split into physical components and propagated to the pixel noise at N skips:
a single-trace setup-noise metric.

Per trace (one channel): the per-skip CDS values of the 10 pixels (first --skip0 skips dropped, each pixel's mean
removed, then the curve repeating in every pixel of the trace (deterministic droop) removed, concatenated: ~4800 samples at the skip rate f_s = 1 / 32.6 us = 30.7 kHz; the H transfers between pixels are
~0.1 ms against 16 ms per pixel, so the series is nearly continuous).  Welch PSD (Hann, segments of --nseg skips), then:
  continuum  S_c(f) = W + K / f^alpha, fitted on the PSD with the lines masked (robust: median of 9 bins first)
  lines      bins exceeding the continuum by > --line-sigma x the scatter of an averaged periodogram bin; their excess
             power is the LINE power (discrete pickup: mains and harmonics, switching regulators, aliased clocks)
  Gaussianity excess kurtosis of the CDS (RTS / bursts)
Each component is propagated to the noise of a pixel averaged over N skips with the skipper transfer function
|H_N(f)|^2 = [sin(pi f N/f_s) / (N sin(pi f / f_s))]^2:   sigma_N^2 = sum_f S(f) |H_N(f)|^2 df  (white -> W f_s/2N).
Outputs (ADU and e- with --gain): white per skip, sigma_N total / white / 1/f / lines for N = 10, 100, 500, the line
list, and the SETUP NOISE FRACTION  SNF_N = (line + excess-over-white low-frequency power) / total at N.

    .venv/bin/python amp_phys/trace_spectrum.py --traces <npz or RJ45ADCDATA files> --label <name> --out <csv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
FS = 1 / 32.6e-6


def cds_of(path: str, skip0: int):
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        if "cds" not in z.files:
            return None
        return z["cds"].astype(float)[:, skip0:]
    from decode_trace import decode_trace
    s = decode_trace(path).skips()
    return s["cds"].reshape(len(s["pixel_ids"]), -1)[:, skip0:]


def hN(f, N):
    x = np.pi * f / FS
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.sin(N * x) / (N * np.sin(x))
    h[f == 0] = 1.0
    return h ** 2


def analyse(c, nseg, line_sigma):
    # remove the deterministic part: each pixel's mean, then the curve that repeats in every pixel of the trace
    # (AC-coupling droop / sequence-locked offset), scaled for the 1/n_pix of random noise it also removes
    c = c - c.mean(1, keepdims=True)
    c = (c - c.mean(0, keepdims=True)) * np.sqrt(c.shape[0] / (c.shape[0] - 1))
    x = c.ravel()
    f, P = welch(x, fs=FS, nperseg=nseg, window="hann", detrend="constant")
    f, P = f[1:], P[1:]
    nav = max(1, int(2 * len(x) / nseg) - 1)
    med = pd.Series(P).rolling(9, center=True, min_periods=1).median().to_numpy()
    lines = P > med * (1 + line_sigma / np.sqrt(nav))
    model = lambda f, W, K, a: W + K / f ** a
    try:
        p, _ = curve_fit(model, f[~lines], P[~lines], p0=[np.median(P[len(P) // 2:]), max(P[0] - np.median(P[len(P) // 2:]), 1e-9) * f[0], 1.0],
                         bounds=([0, 0, 0.3], [np.inf, np.inf, 2.5]), maxfev=20000)
    except Exception:
        p = [np.median(P), 0, 1]
    W, K, a = p
    Sc = model(f, *p)
    Sl = np.where(lines, np.maximum(P - Sc, 0), 0)
    df = f[1] - f[0]
    out = {"white per skip (ADU)": np.sqrt(W * FS / 2), "1/f K": K, "1/f exponent": a,
           "excess kurtosis": float(pd.Series(x).kurt()), "n lines": int(lines.sum())}
    for N in (10, 100, 500):
        H = hN(f, N)
        w = W * FS / 2 / N
        fl = np.sum(K / f ** a * H) * df
        ln = np.sum(Sl * H) * df
        out[f"sigma_{N} white"], out[f"sigma_{N} 1/f"], out[f"sigma_{N} lines"] = np.sqrt(w), np.sqrt(fl), np.sqrt(ln)
        out[f"sigma_{N} total"] = np.sqrt(w + fl + ln)
        out[f"SNF_{N}"] = (fl + ln) / (w + fl + ln)
        out[f"line share_{N}"] = ln / (w + fl + ln)
    top = np.argsort(-(Sl * df))[:3]
    out["strongest lines (Hz: ADU rms)"] = "; ".join(f"{f[i]:.0f}: {np.sqrt(Sl[i] * df):.2f}" for i in top if Sl[i] > 0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gain", type=float, default=1.39)
    ap.add_argument("--skip0", type=int, default=20)
    ap.add_argument("--nseg", type=int, default=2048)
    ap.add_argument("--line-sigma", type=float, default=6.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for t in a.traces:
        try:
            c = cds_of(t, a.skip0)
        except Exception:
            c = None
        if c is None or c.size < 1000:
            continue
        r = analyse(c, a.nseg, a.line_sigma)
        r.update({"label": a.label, "file": Path(t).name})
        for k in list(r):
            if k.startswith("sigma_"):
                r[k.replace("sigma_", "sigma_e_")] = r[k] / a.gain
        rows.append(r)
    R = pd.DataFrame(rows)
    out = Path(a.out)
    hdr = not out.exists()
    R.to_csv(out, mode="a", header=hdr, index=False)
    cols = ["white per skip (ADU)", "1/f exponent", "excess kurtosis", "sigma_e_500 total", "sigma_e_500 white",
            "sigma_e_500 1/f", "sigma_e_500 lines", "SNF_500", "line share_500"]
    print(a.label, len(R), R[cols].median().round(3).to_dict())


if __name__ == "__main__":
    main()
