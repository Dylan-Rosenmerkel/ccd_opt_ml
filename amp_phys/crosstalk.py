"""
crosstalk.py -- the mutual coupling between channels: when one amplifier reads a large charge (a track), how much
appears simultaneously on the others (same pixel index, digitised at the same instant).  Per source amplifier A:
pixels with q_A > --qmin e-, target B restricted to pixels where B has no charge of its own (|B| < 4 sigma_B); robust
slope of B on A (Theil-Sen on binned medians) = crosstalk coefficient (dimensionless; 1e-4 = 1 e- per 10^4 e-).
The sign separates capacitive/inductive coupling of the video lines (sign of the signal derivative) from a shared
ground impedance (opposite sign on every channel).

    .venv/bin/python amp_phys/crosstalk.py --files <avg_img fits.bz2> ... --gain 1.324 1.304 1.248 1.256 --out <csv>
"""
import argparse
import numpy as np, pandas as pd
from astropy.io import fits
from scipy.stats import theilslopes

ap = argparse.ArgumentParser(); ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--gain", type=float, nargs=4, required=True); ap.add_argument("--qmin", type=float, default=30)
ap.add_argument("--out", required=True); a = ap.parse_args()
acc = {}
for f in a.files:
    h = fits.open(f)
    X = np.array([h[i].data.astype(float) for i in range(1, 5)])
    X = X - np.median(X, axis=(1, 2), keepdims=True)
    X -= np.median(X, axis=2, keepdims=True)
    E = X / np.array(a.gain)[:, None, None]
    sig = 1.4826 * np.median(np.abs(E), axis=(1, 2))
    for i in range(4):
        src = E[i] > a.qmin
        for j in range(4):
            if i == j:
                continue
            m = src & (np.abs(E[j]) < 4 * sig[j])
            acc.setdefault((i, j), []).append((E[i][m], E[j][m]))
rows = []
for (i, j), lst in acc.items():
    x = np.concatenate([p[0] for p in lst]); y = np.concatenate([p[1] for p in lst])
    if len(x) < 200:
        continue
    bins = np.quantile(x, np.linspace(0, 1, 21))
    xb = [np.median(x[(x >= bins[k]) & (x < bins[k + 1])]) for k in range(20)]
    yb = [np.median(y[(x >= bins[k]) & (x < bins[k + 1])]) for k in range(20)]
    s, icpt, lo, hi = theilslopes(yb, xb)
    rows.append({"source": f"ch{i}", "target": f"ch{j}", "crosstalk (1e-4)": 1e4 * s, "lo": 1e4 * lo, "hi": 1e4 * hi,
                 "n_pix": len(x)})
R = pd.DataFrame(rows); R.to_csv(a.out, index=False)
M = R.pivot(index="source", columns="target", values="crosstalk (1e-4)")
print("crosstalk, units of 1e-4 (target e- per source e-):"); print(M.round(2))
print(R.round(2).to_string(index=False))
