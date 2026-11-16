"""
setup_noise_N.py -- the setup's contribution to the read noise of an N-skip pixel, in electrons, from a noise spectrum.

The per-skip sigma_CDS cannot see slow pickup (the CDS suppresses f < 1 kHz as f^2), but an N-skip pixel lasts N T
(16 ms for 500 skips) and does not average it away.  The spectrum S(f) of a static record (clocks frozen; LSM psd_*.csv)
or of a clocked trace with the clock-locked waveform removed (raw_psd.residual_record: each pixel's mean skip cycle
subtracted, so f < ~1/(pixel time) is lost) is integrated through the exact filter
    |H_N(f)|^2 = 4 sin^2(pi f dt) sinc^2(f tau) * [sin(pi f N T) / (N sin(pi f T))]^2
    sigma_N^2 = int S |H_N|^2 df          (whole band, and the setup band 100 Hz - 1 kHz)
Reported in electrons (/ gain): sigma at 1 and N skips, the setup-band part at N skips, and its share of the variance.

    .venv/bin/python amp_phys/setup_noise_N.py --static <psd_*.csv> | --traces <npz|csv> ... --gain 1.39 --label <name> --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_psd import residual_record  # noqa: E402
from trace_anatomy import load  # noqa: E402

FS = 15e6


def H2(f, N, tau=10e-6, dt=13.8e-6, T=32.6e-6):
    x = np.pi * f * tau
    h = 4 * np.sin(np.pi * f * dt) ** 2 * (np.sin(x) / x) ** 2
    y = np.pi * f * T
    with np.errstate(all="ignore"):
        A = np.nan_to_num((np.sin(N * y) / (N * np.sin(y))) ** 2, nan=1.0)
    return h * A


def spectrum(path, static):
    v, fl = load(path)
    if static:
        x = v - v.mean()
    else:
        x, _ = residual_record(v, (fl & 1).astype(bool))
        if x is None:
            return None, None
    f, P = welch(x, fs=FS, nperseg=min(2 ** 18, len(x) // 4))
    return f[1:], P[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--static", nargs="+")
    g.add_argument("--traces", nargs="+")
    ap.add_argument("--gain", type=float, required=True, help="ADU per electron")
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--band", type=float, nargs=2, default=[100, 1000])
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for p in (a.static or a.traces):
        f, P = spectrum(p, bool(a.static))
        if f is None:
            continue
        df = f[1] - f[0]
        h1, hN = H2(f, 1), H2(f, a.N)
        band = (f >= a.band[0]) & (f < a.band[1])
        tot = np.sum(P * hN * (f >= a.band[0])) * df
        sb = np.sum((P * hN)[band]) * df
        rows.append({"label": a.label, "file": str(p), "source": "static" if a.static else "trace (waveform removed)",
                     "sigma 1 skip (e)": np.sqrt(np.sum(P * h1) * df) / a.gain,
                     f"sigma {a.N} skips (e)": np.sqrt(tot) / a.gain,
                     f"setup {a.band[0]:g}-{a.band[1]:g} Hz at {a.N} skips (e)": np.sqrt(sb) / a.gain,
                     f"setup share of {a.N}-skip variance": sb / tot})
    R = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    q = out / "setup_noise_N.csv"
    R.to_csv(q, mode="a", header=not q.exists(), index=False)
    print(R.drop(columns=["file"]).groupby(["label", "source"]).median().round(4).to_string())


if __name__ == "__main__":
    main()
