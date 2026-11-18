"""
telegraph.py -- random-telegraph (single-trap) noise in a raw trace: two-level steps in the per-skip CDS.

Per pixel the CDS series (first --drop skips dropped, the mean within-pixel profile removed) is smoothed with a running
mean of --w skips; a two-Gaussian mixture is fitted to the smoothed values (EM).  If two well-separated levels exist,
the step amplitude is their separation, the state sequence comes from a threshold with hysteresis, and the dwell
times in each state follow.  The amplitude is compared with one trapped charge in the gate oxide (number fluctuation):
    dV_g = q / (C_ox W L)  ->  ADU = dV_g * A * G
(W L = 345 um^2, C_ox = 0.454 fF/um^2, A = 0.85, G = 0.875 ADU/uV: 1.02 uV, 0.76 ADU); mobility fluctuation and the
trap's position (depth in the oxide, location along the channel) change the factor.

    .venv/bin/python amp_phys/telegraph.py --traces <npz|csv> ... --label <name> --drop 20 --w 15 --out <dir>
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
ONE_TRAP_ADU = 1.602e-19 / (0.454e-15 * 345) * 1e6 * 0.85 * 0.875


def cds_series(v, fl, drop):
    e = rising(fl, 0)
    P = int(np.median(np.diff(e)))
    s0 = int(np.median([s for k in e[:300] for s in rising(fl[k:k + P], 1)[:1]]))
    has = np.array([k + s0 + 2 < len(fl) and ((fl[k + s0 + 2] >> 1) & 1) == 1 for k in e], bool)
    c = e[has]
    c = c[c + P <= len(v)]
    newpix = np.r_[True, np.abs(np.diff(c) - P) > 3]
    pid = np.cumsum(newpix)
    pos = np.arange(len(c)) - np.flatnonzero(newpix)[pid - 1]
    cs = np.cumsum(np.r_[0, v])
    cds = (cs[c + s0 + 145] - cs[c + s0 + 5]) / 140 - (cs[c + 145] - cs[c + 5]) / 140
    pix = [cds[(pid == p) & (pos >= drop)] for p in np.unique(pid)]
    L = int(np.median([len(x) for x in pix]))
    full = [x[:L] - x[:L].mean() for x in pix if len(x) >= L]
    prof = np.mean(full, 0) if len(full) >= 3 else 0
    return [x - prof for x in full], P


def gmm2(x, it=200):
    m = np.percentile(x, [25, 75]).astype(float)
    s = np.array([x.std() / 2] * 2)
    w = np.array([0.5, 0.5])
    for _ in range(it):
        p = np.array([w[k] / s[k] * np.exp(-0.5 * ((x - m[k]) / s[k]) ** 2) for k in range(2)]) + 1e-300
        r = p / p.sum(0)
        w = r.mean(1)
        m = (r * x).sum(1) / r.sum(1)
        s = np.sqrt((r * (x - m[:, None]) ** 2).sum(1) / r.sum(1)) + 1e-9
    ll2 = np.log(sum(w[k] / s[k] * np.exp(-0.5 * ((x - m[k]) / s[k]) ** 2) for k in range(2)) / np.sqrt(2 * np.pi)).sum()
    ll1 = np.log(np.exp(-0.5 * ((x - x.mean()) / x.std()) ** 2) / (x.std() * np.sqrt(2 * np.pi))).sum()
    o = np.argsort(m)
    return m[o], s[o], w[o], ll2 - ll1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--drop", type=int, default=20)
    ap.add_argument("--w", type=int, default=15)
    ap.add_argument("--gain-adu-per-e", type=float, default=np.nan)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for f in a.traces:
        pix, P = cds_series(*load(f), a.drop)
        k = np.ones(a.w) / a.w
        sm = [np.convolve(x, k, "valid") for x in pix if len(x) > 3 * a.w]
        if not sm:
            continue
        allv = np.concatenate(sm)
        m, s, wts, dll = gmm2(allv)
        sep = (m[1] - m[0]) / np.sqrt(s[0] * s[1])
        # states with hysteresis (thresholds at 1/3 and 2/3 between the levels), dwell times in skips
        lo_t, hi_t = m[0] + (m[1] - m[0]) / 3, m[0] + 2 * (m[1] - m[0]) / 3
        dw = {0: [], 1: []}
        for x in sm:
            st, run = None, 0
            for val in x:
                ns = 1 if val > hi_t else (0 if val < lo_t else st)
                if ns is None:
                    continue
                if ns == st:
                    run += 1
                else:
                    if st is not None and run > 0:
                        dw[st].append(run)
                    st, run = ns, 1
        T = P / FS * 1e3
        rows.append({"label": a.label, "file": str(f), "skips": int(sum(len(x) for x in pix)),
                     "two-level log-likelihood gain": dll, "level separation (sigma)": sep,
                     "step amplitude (ADU)": m[1] - m[0], "step / one trapped charge": (m[1] - m[0]) / ONE_TRAP_ADU,
                     "step (e)": (m[1] - m[0]) / a.gain_adu_per_e,
                     "share in upper state": wts[1], "transitions": len(dw[0]) + len(dw[1]),
                     "median dwell low (ms)": np.median(dw[0]) * T if dw[0] else np.nan,
                     "median dwell high (ms)": np.median(dw[1]) * T if dw[1] else np.nan})
    R = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / f"telegraph_{a.label.replace(' ', '_')}.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.drop(columns=["file"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
