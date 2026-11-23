"""
cms_images.py -- does COMMON-MODE SUBTRACTION across the simultaneously read amplifiers lower the image noise?
The ACM reads its four amplifiers in parallel, so pixel (r, c) of every extension is digitised at the same instant.
For each amplifier: x_cms = x - beta * mean(other amplifiers), using only pixels where every amplifier is in its
0-electron peak (|x - 0e| < 0.6 e), beta fitted per image on half the pixels and scored on the other half.  The
0-electron peak width (robust sigma, e-) before / after = the read-noise gain.  Board 102: the three dead channels as
the reference for 102 ch3 / ch2.

    .venv/bin/python amp_phys/cms_images.py --run-dir /pool/storage/.../2026-11-10-PA03_PQ04_v3_packets \\
        --board 101 --n 40 --out amp_phys/results/v3n/cms_images.csv
"""
import argparse, glob
import numpy as np, pandas as pd
from astropy.io import fits

ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--board", default="101")
ap.add_argument("--n", type=int, default=40); ap.add_argument("--gain", type=float, nargs=4, default=[1.398, 1.384, 1.332, 1.336])
ap.add_argument("--pattern", default="avg_img_*"); ap.add_argument("--no-row-baseline", action="store_true")
ap.add_argument("--out", required=True); a = ap.parse_args()
files = sorted(glob.glob(f"{a.run_dir}/{a.pattern}_L2_{a.board}_*.fits.bz2"))
idx = np.linspace(0, len(files) - 1, min(a.n, len(files))).astype(int)
rows = []
for f in [files[i] for i in idx]:
    h = fits.open(f)
    X = np.array([h[i].data.astype(float) for i in range(1, 5)])                  # 4 x rows x cols, ADU
    X = X[:, :, 10:]                                                                # drop the first columns
    X = X - np.median(X, axis=(1, 2), keepdims=True)
    if not a.no_row_baseline:
        X -= np.median(X, axis=2, keepdims=True)                                    # per-row baseline per amplifier
    sig0 = 1.4826 * np.median(np.abs(X), axis=(1, 2))
    E = X / sig0[:, None, None]                                                     # in units of each amp's noise
    zero = np.all(np.abs(E) < 2.5, 0)                                               # every amp in its 0-e peak
    m = zero.ravel()
    half = np.zeros(m.sum(), bool); half[::2] = True
    for i in range(4):
        x = E[i].ravel()[m]
        ref = np.delete(E, i, 0).mean(0).ravel()[m]
        beta = (x[half] @ ref[half]) / (ref[half] @ ref[half])
        before = 1.4826 * np.median(np.abs(x[~half] - np.median(x[~half])))
        y = x[~half] - beta * ref[~half]
        after = 1.4826 * np.median(np.abs(y - np.median(y)))
        rows.append({"file": f.split("/")[-1], "amp": f"{a.board} ch{i}", "beta": beta, "sigma before (rel)": before,
                     "sigma after (rel)": after, "reduction %": 100 * (1 - after / before), "n_pix": int((~half).sum())})
R = pd.DataFrame(rows); R.to_csv(a.out, index=False)
print(R.groupby("amp")[["beta", "sigma before (rel)", "sigma after (rel)", "reduction %"]].median().round(3))
