"""
raw_psd.py -- the level-0 (raw-sample) noise spectrum of a trace, with the deterministic readout waveform removed:
what a "look at the PSD" of the setup should show.

The raw samples (15 MS/s) of the skip region are cut into skip cycles (from the pedestal-window flag, period P samples);
for each pixel the mean cycle (the clock-locked waveform: edges, feedthrough, reset) is subtracted from every cycle.
The residual cycles are concatenated in time order (the skips of a pixel are contiguous) -> a nearly continuous
noise record of ~2.4 M samples.  Welch PSD (Hann, --nperseg) from ~100 Hz to 7.5 MHz in ADU^2/Hz and, with the chain
gain, nV/rtHz at the video input.  Reports the band powers and the strongest narrow lines (bins above a running
median), so two setups can be compared line by line.

    .venv/bin/python amp_phys/raw_psd.py --traces <npz or RJ45ADCDATA> ... --label <name> --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
FS = 15e6
G = 0.875            # ADU per uV at the video input (chain_calibration.py)


def raw_of(path):
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        return z["values"].astype(float), (z["flags"].astype(np.int64) & 1).astype(bool)
    from decode_trace import decode_trace
    t = decode_trace(path)
    return t.values.astype(float), np.asarray(t.rd).astype(bool)


def residual_record(v, rd):
    e = np.flatnonzero(np.diff(rd.astype(int)) == 1) + 1          # pedestal-window starts = skip-cycle starts
    gaps = np.diff(e)
    P = int(np.median(gaps))
    # pixels: runs of cycles with the standard period
    breaks = np.flatnonzero(np.abs(gaps - P) > 3)
    segs, start = [], 0
    for b in list(breaks) + [len(e) - 1]:
        idx = e[start:b + 1]
        if len(idx) > 60:
            idx = idx[20:]                                          # drop the first skips (transient)
            C = np.array([v[k:k + P] for k in idx if k + P <= len(v)])
            C = C - C.mean(0, keepdims=True)                        # the clock-locked waveform of this pixel
            segs.append(C.ravel())
        start = b + 1
    return np.concatenate(segs) if segs else None, P


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--nperseg", type=int, default=2 ** 16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Ps = []
    for t in a.traces:
        try:
            v, rd = raw_of(t)
            x, P = residual_record(v, rd)
        except Exception as ex:
            print("skip", Path(t).name, ex)
            continue
        if x is None or len(x) < 4 * a.nperseg:
            continue
        f, p = welch(x, fs=FS, nperseg=a.nperseg, window="hann", detrend="constant")
        Ps.append(p)
    if not Ps:
        return
    P = np.median(Ps, 0)
    np.savez(out / f"psd_{a.label.replace(' ', '_')}.npz", f=f, P=P, n=len(Ps))
    asd = np.sqrt(P) / G * 1e3                                        # nV/rtHz at the video input
    bands = [(100, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 3e6), (3e6, 7.5e6)]
    row = {"label": a.label, "traces": len(Ps)}
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        row[f"ASD {lo:g}-{hi:g} Hz (nV/rtHz, median)"] = np.median(asd[m])
        row[f"rms {lo:g}-{hi:g} Hz (ADU)"] = np.sqrt(np.sum(P[m]) * (f[1] - f[0]))
    med = pd.Series(P).rolling(31, center=True, min_periods=1).median().to_numpy()
    ex = P / med
    pk = [i for i in np.argsort(-ex) if f[i] > 200][:12]
    row["lines (kHz: x above continuum)"] = "; ".join(f"{f[i] / 1e3:.2f}: {ex[i]:.0f}" for i in sorted(pk))
    R = pd.DataFrame([row])
    p = out / "raw_psd_summary.csv"
    R.to_csv(p, mode="a", header=not p.exists(), index=False)
    print(R.T.to_string(header=False))


if __name__ == "__main__":
    main()
