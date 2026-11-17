"""
physics_features.py -- per packet and amplifier, the robust trace-physics features of the packet's OWN traces
(theta*_acm<board>_ch<c>.npz, the traces at the packet's setting), for test 1 of the ML prognosis:

  sigma_1 at software windows 2-9.3 us, the fitted white level S_w and 1/f level K (trap_from_trace.py; the 1/f level
  is cross-checked against static spectra to 1.01 +- 0.13), and the predicted noise of a 500-skip pixel in ADU,
      sigma_500^2 = S_w I_w(500) + K I_f(500)      (exact CDS + 500-skip average filter)
  which leaves out the slow band the clocked traces overstate (~12x on board 101).
Median over the packet's theta traces.  Output: one row per (packet, channel).

    .venv/bin/python amp_phys/physics_features.py --data processed_v3n --board 101 --channels 0 1 3 --out <csv>
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trap_from_trace import TAUS, integrals, sigmas  # noqa: E402
from trace_anatomy import load  # noqa: E402

FS = 15e6
F = np.logspace(-1, np.log10(7.5e6), 300000)
DF = np.gradient(F)


def image_integrals(N, tau, dt, T):
    x = np.pi * F * tau
    h = 4 * np.sin(np.pi * F * dt) ** 2 * (np.sin(x) / x) ** 2
    y = np.pi * F * T
    with np.errstate(all="ignore"):
        A = np.nan_to_num((np.sin(N * y) / (N * np.sin(y))) ** 2, nan=1.0)
    return np.sum(h * A * DF), np.sum(h * A / F * DF)


def one(path):
    v, fl = load(path)
    s, P, s0, n = sigmas(v, fl, 20)
    dt, T = s0 / FS, P / FS
    X = np.array([integrals(tau / FS, dt, T) for tau in TAUS])
    y = np.array([s[tau] for tau in TAUS]) ** 2
    w = 1 / y
    (Sw, K), *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    Iw, If = image_integrals(500, 10e-6, dt, T)
    out = {f"sigma_1 tau={tau / 15:.1f}us": s[tau] for tau in TAUS}
    out.update({"S_w": Sw, "K": K, "sigma_500 pred (ADU)": np.sqrt(max(Sw * Iw + max(K, 0) * If, 0)),
                "sigma_500 white only (ADU)": np.sqrt(max(Sw, 0) * Iw)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--board", type=int, default=101)
    ap.add_argument("--channels", type=int, nargs="+", default=[0, 1, 3])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for pdir in sorted(Path(a.data).glob("*/traces")):
        pk = pdir.parent.name
        for c in a.channels:
            fs = sorted(glob.glob(str(pdir / f"theta*_acm{a.board}_ch{c}.npz")))
            R = []
            for f in fs:
                try:
                    R.append(one(f))
                except Exception:
                    pass
            if R:
                m = pd.DataFrame(R).median().to_dict()
                m.update({"packet_id": pk, "board": a.board, "channel": c, "n traces": len(R)})
                rows.append(m)
        print(pk, flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
