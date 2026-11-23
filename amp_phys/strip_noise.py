"""
strip_noise.py -- setup and amplifier noise from ONE level-1 strip (a few-second acquisition with the image's own
clocking, all four channels recorded SIMULTANEOUSLY, every skip's value kept).  No image needed.

Per strip and channel (per-skip values, ADU; skips >= --skip0):
  sigma_1          white noise per skip: std of consecutive-skip differences / sqrt(2) (pixel charge cancels)
  sigma_N          noise of the mean of N skips: std of the difference of consecutive N-skip blocks of the SAME pixel
                   / sqrt(2) (charge cancels), N = 10, 50, 100, 250 (needs >= 2 blocks per pixel)
  floor ratio      sigma_N / (sigma_1 / sqrt(N)): 1 = white only; > 1 = slow noise (1/f, wander) adding at that N
Simultaneous, between the channels of a board (valid: one acquisition):
  rho_skip         mean correlation of the per-skip residuals (pixel mean removed) with the other channels
  rho_N            the same for the N-skip block differences: is the slow noise SHARED (setup) or own (amplifier)?
  shared fraction  at N: 1 - var(own) / var(total), own = residual after regressing on the other channels' mean
The single setup numbers: rho_N (or the shared fraction) at the operating N, and the floor ratio.

    .venv/bin/python amp_phys/strip_noise.py --files <..._L1_101_...RJ45ADCDATA0> ... --label <name> --out <csv>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from decode_l1 import read_l1  # noqa: E402

NS = (10, 50, 100, 250)


def load(path0: str, skip0: int):
    m = re.search(r"_(\d+)x(\d+)x(\d+)_1x1_L1_", path0)
    nrow, ncol, nskip = map(int, m.groups())
    V = {}
    for ch in range(4):
        p = path0[:-1] + str(ch)
        try:
            s = read_l1(p, ncol=ncol, nskip=nskip)
        except Exception:
            continue
        V[ch] = s.values.astype(float)[:, skip0:]
    if len(V) < 2:
        return None
    n = min(v.shape[0] for v in V.values())
    return {c: v[:n] for c, v in V.items()}


def block_diff(v, N):
    nb = v.shape[1] // N
    if nb < 2:
        return None
    b = v[:, :nb * N].reshape(v.shape[0], nb, N).mean(2)
    return (np.diff(b, axis=1) / np.sqrt(2)).ravel()


def shared(y, refs):
    ref = np.mean(refs, 0)
    beta = (y @ ref) / (ref @ ref)
    own = y - beta * ref
    return 1 - own.var() / y.var(), np.mean([np.corrcoef(y, r)[0, 1] for r in refs])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--files", nargs="+", required=True, help="the ...RJ45ADCDATA0 file of each strip")
    ap.add_argument("--label", required=True)
    ap.add_argument("--skip0", type=int, default=20)
    ap.add_argument("--channels", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for f in a.files:
        V = load(f, a.skip0)
        if V is None:
            continue
        chs = [c for c in a.channels if c in V]
        res = {c: V[c] - V[c].mean(1, keepdims=True) for c in chs}           # per-skip residuals
        for c in chs:
            v = V[c]
            r = {"label": a.label, "file": Path(f).name, "channel": c, "pixels": v.shape[0], "skips": v.shape[1]}
            r["sigma_1"] = np.std(np.diff(v, axis=1)) / np.sqrt(2)
            others = [o for o in chs if o != c]
            sh, rho = shared(res[c].ravel(), [res[o].ravel() for o in others])
            r["rho_skip"], r["shared_skip"] = rho, sh
            for N in NS:
                y = block_diff(v, N)
                if y is None:
                    continue
                r[f"sigma_{N}"] = y.std()
                r[f"floor ratio_{N}"] = y.std() / (r["sigma_1"] / np.sqrt(N))
                sh, rho = shared(y, [block_diff(V[o], N) for o in others])
                r[f"rho_{N}"], r[f"shared_{N}"] = rho, sh
            rows.append(r)
    R = pd.DataFrame(rows)
    out = Path(a.out)
    R.to_csv(out, mode="a", header=not out.exists(), index=False)
    cols = ["sigma_1", "sigma_50", "sigma_250", "floor ratio_50", "floor ratio_250", "rho_skip", "rho_50", "rho_250",
            "shared_250"]
    print(a.label, R.groupby("channel")[[c for c in cols if c in R]].median().round(3).to_string())


if __name__ == "__main__":
    main()
