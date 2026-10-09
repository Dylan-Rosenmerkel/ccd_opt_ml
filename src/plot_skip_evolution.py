"""
plot_skip_evolution.py -- how the waveform changes from skip to skip within one pixel.

Shows what a mean template would hide.  Four panels for one pixel of a trace:

    A  residual of every skip cycle after subtracting the mean shape (rows = skip number)
    B  the value at four fixed times in the cycle, against skip number
       (the pulse peak oscillates with a period of ~3.3 skips: the sequencer cycle is
       3258 clock cycles = 488.7 samples, not a whole number of ADC samples)
    C  CDS of the first 40 skips (the first skips ring before settling)
    D  baseline (pedestal-window mean) of every skip: the droop

Also prints how much of the skip-to-skip scatter of the shape is smooth drift (a degree-4
polynomial in skip number fitted to every time sample).

Usage:  python plot_skip_evolution.py TRACE OUT.png [--pixel 6]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decode_trace import decode_trace
from trace_regions import split_regions

US = 15.0


def drift_fraction(residual: np.ndarray, columns: np.ndarray, first_skip: int = 5, degree: int = 4) -> float:
    """Fraction of the skip-to-skip variance in ``columns`` explained by a smooth polynomial in skip number."""
    k = np.arange(len(residual))
    use = k >= first_skip
    vander = np.vander((k - len(k) / 2) / (len(k) / 2), degree + 1)
    block = residual[use][:, columns]
    fitted = np.empty_like(block)
    for j in range(block.shape[1]):
        coef = np.linalg.lstsq(vander[use], block[:, j], rcond=None)[0]
        fitted[:, j] = vander[use] @ coef
    return 1 - ((block - fitted) ** 2).sum() / ((block - block.mean(axis=0)) ** 2).sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace")
    parser.add_argument("out")
    parser.add_argument("--pixel", type=int, default=6)
    args = parser.parse_args()

    r = split_regions(decode_trace(args.trace))
    cycles = r.skip[args.pixel].astype(float)
    baseline = cycles[:, :r.n_ped].mean(axis=1)
    shape = cycles - baseline[:, None]
    template = shape[5:].mean(axis=0)
    residual = shape - template
    cds = r.cds[args.pixel]
    t = np.arange(r.period) / US
    skips = np.arange(len(cycles))

    # fraction of the variance that is smooth drift, in four parts of the cycle
    parts = {"tail": (t > 25) & (t < 33), "dip": (t > 10) & (t < 12),
             "pedestal window": (t > 1) & (t < 9.5), "signal window": (t > 15) & (t < 23)}
    print("fraction of skip-to-skip variance that is smooth drift (degree-4 polynomial):")
    for name, mask in parts.items():
        print(f"  {name:16s} {drift_fraction(residual, np.flatnonzero(mask)):.2f}")
    print(f"CDS of the first 6 skips: {np.round(cds[:6], 1)}  later mean {cds[10:].mean():.1f}, scatter {cds[10:].std():.1f}")

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    a = ax[0, 0]
    image = a.imshow(residual, aspect="auto", cmap="RdBu_r", vmin=-600, vmax=600, extent=[0, r.period / US, len(cycles), 0])
    a.set(title=f"A  residual after subtracting the mean shape (pixel {args.pixel})\nrows = skip number, columns = time in cycle",
          xlabel="time in cycle (us)", ylabel="skip number")
    fig.colorbar(image, ax=a, label="ADU")

    a = ax[0, 1]
    for time_us, label, color in [(10.7, "dip minimum", "tab:red"), (25.0, "pulse peak", "tab:purple"),
                                  (27.5, "tail", "tab:orange"), (31.0, "late tail", "tab:green")]:
        a.plot(skips, shape[:, int(time_us * US)], lw=0.8, color=color, label=f"{label} ({time_us} us)")
    a.set(title="B  the same time point in every cycle, against skip number", xlabel="skip number",
          ylabel="ADU (baseline removed)")
    a.legend(fontsize=8)

    a = ax[1, 0]
    a.plot(skips[:40], cds[:40], "o-", ms=3)
    a.axhline(cds[10:].mean(), color="k", ls="--", lw=0.7)
    a.set(title="C  CDS of the first 40 skips", xlabel="skip number", ylabel="CDS = pedestal - signal (ADU)")

    a = ax[1, 1]
    a.plot(skips, baseline, lw=0.8)
    a.set(title="D  baseline of every skip: the droop", xlabel="skip number", ylabel="ADU")

    fig.tight_layout()
    fig.savefig(args.out, dpi=105)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
