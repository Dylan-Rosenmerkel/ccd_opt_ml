"""
trace_physics2.py -- the remaining physics in a raw trace (beyond trace_anatomy.py), per skip and per pixel:

  multiplicative noise   amplitude of the per-skip pulse (bit-4 event, the largest feature of every skip cycle) relative
                         to that skip's pedestal level: its skip-to-skip jitter (consecutive differences / sqrt 2)
                         against the additive noise of the same samples -> fractional gain/supply jitter; and the
                         timing jitter of its rising edge (half-height crossing, linear interpolation), in ns
  level relaxation       the pedestal level against time within a pixel (median over pixels): a + b exp(-t/tau):
                         the slow recovery of the video chain after the horizontal transfer (AC coupling)
  CDS drift / level      the CDS against time within a pixel (median profile), tested against the baseline-slope
                         hypothesis: a level relaxing at r ADU/sample adds r * (signal-window delay) to every CDS
                         (slope 1 expected); also regressed on the level itself (a gain nonlinearity would show there)
  within-pixel coupling  the same slope from skip-to-skip fluctuations (pixel profile removed): an independent,
                         profile-free estimate of dCDS/dlevel

    .venv/bin/python amp_phys/trace_physics2.py --traces <npz|csv> ... --label <name> --drop 20 --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_anatomy import load, rising  # noqa: E402

FS = 15e6


def analyse(v, fl, drop):
    e = rising(fl, 0)
    P = int(np.median(np.diff(e)))
    s0 = int(np.median([s for k in e[:300] for s in rising(fl[k:k + P], 1)[:1]]))
    has = np.array([k + s0 + 2 < len(fl) and ((fl[k + s0 + 2] >> 1) & 1) == 1 for k in e], bool)
    c = e[has]
    c = c[c + P <= len(v)]
    newpix = np.r_[True, np.abs(np.diff(c) - P) > 3]
    pid = np.cumsum(newpix)
    pos = np.arange(len(c)) - np.flatnonzero(newpix)[pid - 1]
    C = np.array([v[k:k + P] for k in c])
    ped = C[:, 5:145].mean(1)
    cds = C[:, s0 + 5:s0 + 145].mean(1) - ped
    out = {"skip period (samples)": P}
    keep = pos >= drop
    # ---- per-skip pulse: the largest excursion of the median cycle outside the windows
    tm = np.median(C[keep], 0)
    mask = np.ones(P, bool)
    mask[:150] = False
    mask[s0:s0 + 150] = False
    pk = int(np.argmax(np.where(mask, np.abs(tm - np.median(tm[5:145])), -1)))
    sgn = np.sign(tm[pk] - np.median(tm[5:145]))
    amp = sgn * (C[:, pk] - ped)
    same = (np.diff(c) == P) | (np.abs(np.diff(c) - P) <= 3)
    same &= (np.diff(pid) == 0) & keep[1:] & keep[:-1]
    jit = np.std(np.diff(amp)[same]) / np.sqrt(2)
    add = np.std(np.diff(C[:, pk - 30])[same]) / np.sqrt(2)        # additive noise of a quiet sample, same cycles
    out["pulse position (samples into cycle)"] = pk
    out["pulse amplitude (ADU)"] = np.median(amp[keep])
    out["pulse amplitude jitter (ADU)"] = jit
    out["additive noise at one sample (ADU)"] = add
    out["multiplicative jitter (% of pulse)"] = 100 * np.sqrt(max(jit ** 2 - add ** 2, 0)) / abs(np.median(amp[keep]))
    # timing: half-height crossing on the rising side of the pulse
    lo = max(pk - 12, 0)
    base = C[:, lo]
    half = base + 0.5 * (C[:, pk] - base)
    seg = sgn * (C[:, lo:pk + 1] - half[:, None])
    tcross = []
    for row in seg:
        i = np.flatnonzero((row[:-1] < 0) & (row[1:] >= 0))
        if len(i):
            j = i[-1]
            tcross.append(lo + j + (-row[j]) / (row[j + 1] - row[j]))
        else:
            tcross.append(np.nan)
    tcross = np.array(tcross)
    dt = np.diff(tcross)[same]
    out["pulse timing jitter (ns)"] = np.nanstd(dt) / np.sqrt(2) / FS * 1e9
    out["sampling-phase jitter if asynchronous (ns)"] = 1e9 / FS / np.sqrt(12)
    # amplitude jitter after removing what follows the sampling phase (quadratic in the crossing time)
    ok = np.isfinite(tcross) & keep
    X = np.column_stack([np.ones(ok.sum()), tcross[ok], tcross[ok] ** 2])
    res = amp[ok] - X @ np.linalg.lstsq(X, amp[ok], rcond=None)[0]
    r_full = np.full(len(amp), np.nan)
    r_full[ok] = res
    jr = np.nanstd(np.diff(r_full)[same]) / np.sqrt(2)
    out["pulse amplitude jitter, phase removed (ADU)"] = jr
    out["multiplicative jitter, phase removed (% of pulse)"] = 100 * np.sqrt(max(jr ** 2 - add ** 2, 0)) / abs(np.median(amp[keep]))
    # ---- within-pixel profiles (pixels with the modal length)
    L = int(np.median(np.bincount(pid)[1:]))
    full = [p for p in np.unique(pid) if np.sum(pid == p) == L]
    if len(full) >= 3 and L > drop + 20:
        Lv = np.array([ped[pid == p] for p in full])
        Cv = np.array([cds[pid == p] for p in full])
        lv_prof = np.median(Lv - Lv.mean(1, keepdims=True), 0)[drop:]
        cd_prof = np.median(Cv - Cv.mean(1, keepdims=True), 0)[drop:]
        t = (np.arange(drop, L) * P / FS) * 1e3                       # ms
        try:
            pp, _ = curve_fit(lambda t, a, b, tau: a + b * np.exp(-t / tau), t, lv_prof,
                              p0=[lv_prof[-1], lv_prof[0] - lv_prof[-1], 3], maxfev=20000)
            out["level relaxation tau in pixel (ms)"] = pp[2]
            out["level relaxation amplitude (ADU)"] = pp[1]
        except Exception:
            out["level relaxation tau in pixel (ms)"] = np.nan
        out["CDS drift over pixel (ADU)"] = cd_prof[:20].mean() - cd_prof[-20:].mean()
        try:
            pc, _ = curve_fit(lambda t, a, b, tau: a + b * np.exp(-t / tau), t, cd_prof,
                              p0=[cd_prof[-1], cd_prof[0] - cd_prof[-1], 3], maxfev=20000)
            out["CDS drift tau (ms)"] = pc[2]
            out["CDS drift amplitude at skip 20 (ADU)"] = pc[1] * np.exp(-t[0] / pc[2])
        except Exception:
            out["CDS drift tau (ms)"] = np.nan
        # baseline-slope hypothesis: a level drifting at r ADU/sample adds r * s0 to the CDS (signal window s0 later)
        pred = np.gradient(lv_prof) / P * s0
        out["CDS drift predicted from level slope (ADU)"] = pred[:20].mean() - pred[-20:].mean()
        out["CDS profile vs slope prediction: slope"] = np.polyfit(pred, cd_prof, 1)[0]
        out["R2 of CDS profile on slope prediction"] = np.corrcoef(pred, cd_prof)[0, 1] ** 2
        sl = np.polyfit(lv_prof, cd_prof, 1)[0]
        out["dCDS/dlevel from profiles"] = sl
        out["R2 of CDS profile on level profile"] = np.corrcoef(lv_prof, cd_prof)[0, 1] ** 2
        # profile-free: skip-to-skip fluctuations after removing the profiles
        Lr = (Lv - Lv.mean(1, keepdims=True))[:, drop:] - lv_prof
        Cr = (Cv - Cv.mean(1, keepdims=True))[:, drop:] - cd_prof
        out["dCDS/dlevel from fluctuations"] = np.sum(Lr * Cr) / np.sum(Lr * Lr)
        dL = np.diff(Lr, axis=1) / P * s0                              # slope of the level fluctuations
        out["CDS on level-slope fluctuations: slope"] = np.sum(dL * Cr[:, 1:]) / np.sum(dL * dL)
        out["empty-pixel CDS offset (ADU)"] = np.median(cds[keep])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--drop", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for f in a.traces:
        try:
            r = analyse(*load(f), a.drop)
        except Exception as ex:
            print("skip", f, ex)
            continue
        r.update({"label": a.label, "file": str(f)})
        rows.append(r)
    R = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / f"trace_physics2_{a.label.replace(' ', '_')}.csv", index=False)    # one file per label (parallel-safe)
    print(R.drop(columns=["file"]).median(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
