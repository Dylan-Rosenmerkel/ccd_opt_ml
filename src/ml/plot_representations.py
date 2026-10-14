"""
plot_representations.py -- the "representation ladder" figure of docs/ml_models.tex, from a REAL trace.

One trace can be given to a model at four levels of detail:
  1. the raw ADC samples (~3 million per trace),
  2. the samples folded into skip cycles: one row per skip (pixels x skips rows x ~489 samples),
  3. the 12 curves of trace_features (mean skip waveform, transfer shapes, noise spectra, ...),
  4. the 85 hand-made numbers.

    python -m ml.plot_representations TRACE.RJ45ADCDATA0 OUT.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decode_trace import decode_trace
from trace_features import trace_features
from trace_regions import split_regions


def main(path: str, out: str) -> None:
    meta = json.loads(Path(path).with_suffix(".meta").read_text())
    trace = decode_trace(path)
    regions = split_regions(trace, meta)
    feats = trace_features(trace, meta=meta, regions=regions)
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.5, wspace=0.32)

    ax = fig.add_subplot(gs[0, :2])
    v = trace.values.astype(float)
    t = np.arange(v.size) / 15e6 * 1e3
    ax.plot(t, v, lw=0.2, color="#264653")
    ax.set(xlabel="time (ms)", ylabel="ADC (ADU)",
           title=f"1. raw trace: {v.size:,} samples ({v.size * 4 / 1e6:.0f} MB) for {regions.n_pixels} pixels x {len(regions.skip[1])} skips")
    inset = ax.inset_axes([0.62, 0.55, 0.36, 0.4])
    i0 = int(np.argmax(t > t[-1] * 0.4))
    inset.plot(np.arange(1500), v[i0:i0 + 1500], lw=0.6, color="#264653")
    inset.set_title("100 µs: three skip cycles", fontsize=8)
    inset.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[0, 2:])
    cyc = np.asarray(regions.skip[1], float)
    cyc = cyc - cyc[:, :regions.n_ped].mean(axis=1, keepdims=True)
    im = ax.imshow(cyc, aspect="auto", cmap="RdBu_r", vmin=-3000, vmax=3000, interpolation="nearest")
    ax.set(xlabel="sample within the skip cycle", ylabel="skip number (pixel 1)",
           title=f"2. folded: one row per skip ({cyc.shape[0]} x {cyc.shape[1]} for one pixel; "
                 f"{regions.n_pixels} pixels in the trace)")
    fig.colorbar(im, ax=ax, label="ADU (baseline removed)", fraction=0.04)

    names = ["skip: mean shape", "horizontal: mean shape", "noise: Allan deviation", "noise: CDS spectrum"]
    sub = gs[1, :3].subgridspec(2, 2, hspace=0.55, wspace=0.25)
    for k, name in enumerate(names):
        ax = fig.add_subplot(sub[k // 2, k % 2])
        c = np.asarray(feats.curves[name], float)
        if "Allan" in name:
            ax.loglog([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 160][:c.size], c, "o-", ms=3, color="#e76f51")
            ax.set_xlabel("skips averaged", fontsize=8)
        elif "spectrum" in name:
            ax.semilogy(np.linspace(0, 15, c.size), c, lw=0.8, color="#e76f51")
            ax.set_xlabel("kHz", fontsize=8)
        else:
            ax.plot(c, lw=0.9, color="#e76f51")
            ax.set_xlabel("sample", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.text(0.13, 0.505, f"3. curves: {len(feats.curves)} curves of 8-750 points ({sum(np.size(c) for c in feats.curves.values()):,} numbers); 4 shown",
             fontsize=11)

    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    pick = ["horizontal: minimum", "vertical: minimum", "fit: settling time after pulse (us)", "fit: dip recovery time (us)",
            "skip: pulse peak height", "skip: dip height", "skip: baseline at skip 1", "skip: CDS rms (detrended)",
            "fit: white noise per skip (ADU)", "fit: correlated noise floor (ADU)", "noise: reset-level scatter (ADU)",
            "fit: first-skip ringing frequency (cyc/skip)", "edge: jitter (ns)", "slow: drift (ADU/ms)"]
    lines = [f"{n[:40]:40s} {feats.scalars[n]:>10.4g}" for n in pick]
    ax.text(0, 1, f"4. {len(feats.scalars)} numbers (14 shown)\n\n" + "\n".join(lines) + "\n...", family="monospace",
            fontsize=7.6, va="top")
    fig.suptitle(f"One real trace at four levels of detail ({Path(path).name}, module 101 ch0, 2026-10-08)", fontsize=13)
    fig.savefig(out, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
