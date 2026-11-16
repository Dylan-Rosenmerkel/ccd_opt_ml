"""
trap_from_trace.py -- 1/f noise and oxide trap density from ONE raw trace, by varying the CDS window in software.

The window length is the handle (white ~ 1/tau, 1/f ~ independent of tau; oxide_traps_tau.py varied it in hardware
over 100 strips).  A raw trace has every sample of the 10 us windows, so shorter windows are formed digitally: the
last tau of the pedestal window and the last tau of the signal window (fixed centre spacing dt = 207 samples, the
settled end of each window).  Per skip, consecutive-skip differences within a pixel; sigma_1^2(tau) fitted with
    sigma_1^2 = S_w I_w(tau) + K I_f(tau),   |H|^2 = 4 sin^2(pi f dt) sinc^2(f tau) x 2 sin^2(pi f T)
-> 1/f corner K/S_w, white ASD, and N_t (McWhorter, as oxide_traps_tau.py).  TOTAL noise (setup included): an upper
limit on the transistor's own 1/f.  --filter-fc adds the frontend roll-off (white noise is band-limited; without
it short windows under-count white noise and the fit moves it into 1/f).

    .venv/bin/python amp_phys/trap_from_trace.py --traces <npz|csv> ... --label <name> --drop 20 --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_anatomy import load, rising  # noqa: E402

FS = 15e6
Q, KB = 1.602176634e-19, 1.380649e-23
F = np.logspace(0, np.log10(7.5e6), 200000)
DF = np.gradient(F)
TAUS = (30, 45, 60, 90, 120, 140)                       # samples (2 - 9.3 us)


FILT = {"fc": None, "n": 2.0}                       # frontend roll-off (static_fit.py: ~550 kHz, order ~2)


def integrals(tau, dt, T):
    x = np.pi * F * tau
    H = 4 * np.sin(np.pi * F * dt) ** 2 * (np.sin(x) / x) ** 2 * 2 * np.sin(np.pi * F * T) ** 2
    if FILT["fc"]:                                  # white and 1/f both pass the frontend filter
        H = H / (1 + (F / FILT["fc"]) ** (2 * FILT["n"]))
    return np.sum(H * DF), np.sum(H / F * DF)


def sigmas(v, fl, drop):
    e = rising(fl, 0)
    P = int(np.median(np.diff(e)))
    b0 = ((fl >> 0) & 1).astype(np.int8)
    up, dn = np.flatnonzero(np.diff(b0) == 1) + 1, np.flatnonzero(np.diff(b0) == -1) + 1
    dn = dn[dn > up[0]]
    win = int(np.median((dn[:len(up)] - up[:len(dn)])[:400]))       # window length from the flags
    assert win >= max(TAUS), f"window {win} samples shorter than the longest software window"
    s0 = int(np.median([s for k in e[:300] for s in rising(fl[k:k + P], 1)[:1]]))
    has = np.array([k + s0 + 2 < len(fl) and ((fl[k + s0 + 2] >> 1) & 1) == 1 for k in e], bool)
    c = e[has]
    newpix = np.r_[True, np.abs(np.diff(c) - P) > 3]
    pid = np.cumsum(newpix)
    pos = np.arange(len(c)) - np.flatnonzero(newpix)[pid - 1]
    keep = (pos >= drop) & (c + s0 + win <= len(v))
    c, pid = c[keep], pid[keep]
    same = (np.diff(c) == P) | (np.abs(np.diff(c) - P) <= 3)
    same &= np.diff(pid) == 0
    cs = np.cumsum(np.r_[0, v])
    out = {}
    for tau in TAUS:
        ped = (cs[c + win] - cs[c + win - tau]) / tau
        sig = (cs[c + s0 + win] - cs[c + s0 + win - tau]) / tau
        d = np.diff(sig - ped)[same]
        out[tau] = np.std(d) / np.sqrt(2)
    return out, P, s0, int(same.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--drop", type=int, default=20)
    ap.add_argument("--gain-adu-per-uv", type=float, default=0.875)
    ap.add_argument("--wl-um2", type=float, default=345.0)
    ap.add_argument("--t", type=float, default=130.0)
    ap.add_argument("--lam-nm", type=float, default=0.1)
    ap.add_argument("--filter-fc", type=float, help="frontend roll-off corner (Hz), from static_fit.py")
    ap.add_argument("--filter-n", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    FILT.update(fc=a.filter_fc, n=a.filter_n)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in a.traces:
        v, fl = load(t)
        s, P, s0, n = sigmas(v, fl, a.drop)
        dt, T = s0 / FS, P / FS
        X = np.array([integrals(tau / FS, dt, T) for tau in TAUS])
        y = np.array([s[tau] for tau in TAUS]) ** 2
        w = 1 / y
        (Sw, K), *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
        r = {"label": a.label, "file": Path(t).name, "pairs": n,
             **{f"sigma_1 tau={tau / 15:.1f}us (ADU)": s[tau] for tau in TAUS},
             "white ASD (nV/rtHz)": np.sqrt(max(Sw, 0)) / a.gain_adu_per_uv * 1e3,
             "1/f K (ADU^2)": K, "1/f corner (kHz)": K / Sw / 1e3 if Sw > 0 else np.nan,
             "1/f share at 10 us": K * integrals(10e-6, dt, T)[1] / (Sw * integrals(10e-6, dt, T)[0] + K * integrals(10e-6, dt, T)[1]),
             "fit rms rel. residual": np.sqrt(np.mean((X @ [Sw, K] / y - 1) ** 2))}
        Kvg = max(K, 0) / (a.gain_adu_per_uv * 1e6) ** 2 / 0.85 ** 2
        Ccm = 0.454e-15 / 1e-8
        r["N_t A=0.85 C_ox (eV^-1 cm^-3)"] = Kvg * a.wl_um2 * 1e-8 * Ccm ** 2 / (Q ** 2 * KB * a.t * a.lam_nm * 1e-7) * Q
        rows.append(r)
    R = pd.DataFrame(rows)
    p = out / "trap_from_trace.csv"
    R.to_csv(p, mode="a", header=not p.exists(), index=False)
    print(R.drop(columns=["file"]).median(numeric_only=True).to_string(float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
