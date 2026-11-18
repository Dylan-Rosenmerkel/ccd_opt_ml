"""
reset_pulse.py -- the reset pulse in a raw trace, measured independently of the sequencer and the frontend bandwidth.

The reset gate pulses once per pixel (twice around each horizontal transfer).  While RG is high the sense node sits at
VR plus the RG feedthrough; the pulse height relative to the settled level after it carries the reset-channel charge
(1/2 C_eff W L (RG_high - VR - V*)) and the gate-overlap feedthrough.  The pulse is only ~1 us wide, comparable to the
output's settling time, so its peak depends on the pulse width and the bandwidth (PA03: a 13-sample pulse still rising;
LSM: a sharp spike).  Instead of reading samples at fixed offsets, the pulse is FITTED:
    y(t) = b0 + b1 (t - t1) + D * g(t),   g = 1 - exp(-(t - t1)/tau)            for t1 <= t < t2
                                          g = (1 - exp(-(t2 - t1)/tau)) exp(-(t - t2)/tau)   for t >= t2
with t1, t2 (pulse edges), tau (settling) and D free.  D is the pulse height a settled output would show.
Pulses are found in the median per-pixel waveform around the horizontal-transfer flag as the largest fast excursions
(no sequencer knowledge needed).

Status (2026-11-16): the pulse FINDER is sequencer-independent (PA03 -280/+1626, LSM -268/+1626 samples from the H
flag).  The HEIGHT is not comparable across sequencers: PA03 pulses are 11-14 samples wide and the sense node is still
charging to VR when they end; LSM pulses are 2.7-5.5 samples wide, too short at 15 MS/s to separate feedthrough from
charging (a two-component fit returns equal halves; the two LSM pulses of one trace differ by 30 %).  The
sequencer-independent route to the injection slope is a DIFFERENCE of two traces on the same setup (RG_high changed),
which cancels the pulse dynamics; depth_from_trace.py then works from single traces on that setup.

    .venv/bin/python amp_phys/reset_pulse.py --traces <npz|csv> ... --label <name> --out <dir> [--plot]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clock_edges as ce  # noqa: E402
from trace_anatomy import load  # noqa: E402


PRE, POST = 450, 1850                          # window around the horizontal-transfer flag (covers both pulses)


def median_hwave(v, fl):
    e = ce.flag_edges(fl, 6)
    e = e[(e > PRE) & (e < len(v) - POST)]
    if len(e) < 10:
        return None
    return np.median(np.array([v[k - PRE:k + POST] for k in e]), 0)


def find_pulses(w, n=2, min_gap=300):
    """the n largest fast rising edges (a pulse start), largest first-difference over 3 samples"""
    d = w[3:] - w[:-3]
    idx = []
    for i in np.argsort(-d):
        if all(abs(i - j) > min_gap for j in idx) and 40 < i < len(w) - 80:
            idx.append(int(i))
        if len(idx) == n:
            break
    return sorted(idx)


def model(p, t):
    b0, b1, D, t1, t2, tau = p
    g = np.zeros_like(t, dtype=float)
    on = (t >= t1) & (t < t2)
    g[on] = 1 - np.exp(-(t[on] - t1) / tau)
    off = t >= t2
    g[off] = (1 - np.exp(-(t2 - t1) / tau)) * np.exp(-(t[off] - t2) / tau)
    return b0 + b1 * (t - t1) + D * g


def fit_pulse(w, i0, pre=25, post=45):
    t = np.arange(i0 - pre, i0 + post).astype(float)
    y = w[i0 - pre:i0 + post]
    base = np.median(w[i0 - pre:i0 - 3])
    pk = int(np.argmax(y))
    p0 = [base, 0.0, (y.max() - base) * 1.3, i0 - 1.0, t[pk] + 1.0, 3.0]
    lo = [-np.inf, -np.inf, 0, i0 - 8, i0 + 1, 0.3]
    hi = [np.inf, np.inf, np.inf, i0 + 6, i0 + 40, 40]
    p0 = list(np.clip(p0, np.array(lo, float) + 1e-6, np.array(hi, float) - 1e-6))
    r = least_squares(lambda p: model(p, t) - y, p0, bounds=(lo, hi), loss="soft_l1", f_scale=500)
    b0, b1, D, t1, t2, tau = r.x
    return {"pulse height D (ADU)": D, "pulse width (samples)": t2 - t1, "settling tau (us)": tau / 15,
            "peak (ADU)": y.max() - base, "fit rms / D": np.sqrt(np.mean(r.fun ** 2)) / max(D, 1), "pos": i0 - PRE}, (t, y, r.x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    rows, show = [], []
    for f in a.traces:
        w = median_hwave(*load(f))
        if w is None:
            continue
        for k, i0 in enumerate(find_pulses(w)):
            r, fitd = fit_pulse(w, i0)
            r.update({"label": a.label, "file": str(f), "pulse": k})
            rows.append(r)
            if len(show) < 2:
                show.append((r, fitd))
    R = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "reset_pulse.csv"
    R.to_csv(p, mode="a", header=not p.exists(), index=False)
    print(R.groupby("pulse").median(numeric_only=True).round(3).to_string())
    if a.plot and show:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(show), figsize=(5.5 * len(show), 3.4))
        for ax, (r, (t, y, px)) in zip(np.atleast_1d(axs), show):
            tt = np.linspace(t[0], t[-1], 600)
            ax.plot(t - PRE, y, ".", color="#2e6fb7", ms=4, label="median waveform")
            ax.plot(tt - PRE, model(px, tt), color="#d9822b", lw=1.5, label=f"fit: D = {r['pulse height D (ADU)']:.0f} ADU")
            ax.set_xlabel("samples after the horizontal-transfer flag")
            ax.set_title(f"{a.label}: pulse at {r['pos']}", loc="left", fontsize=9)
            ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"reset_pulse_{a.label.replace(' ', '_')}.png", dpi=140)


if __name__ == "__main__":
    main()
