"""
plot_regions.py -- figures of the three regions of a trace and of what averaging does.

Writes three PNG files into OUT_DIR, all from :mod:`trace_regions` (nothing is averaged
before the plots except where stated):

    regions_overview.png   row 2 of the trace with the vertical block, the horizontal blocks
                           and the skip regions shaded (min/max envelope of the raw signal)
    regions_templates.png  for each region one repeat after another in grey, the mean shape
                           (blue, +- 1 sigma) and the scatter between repeats
    why_mean_shape.png     A: one cycle against the mean of all cycles (zoom on the signal
                           window); B: noise of an N-skip average against 1/sqrt(N);
                           C: mean shapes of two traces and the z-score of their difference
                           (only if --other is given)

Baselines are removed before averaging: for skip cycles with the mean of the pedestal window,
for transfers with the mean of their first 30 samples.

Usage:  python plot_regions.py TRACE OUT_DIR [--other TRACE_B] [--pixel 6]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decode_trace import decode_trace
from trace_regions import TraceRegions, split_regions

US = 15.0                                        # samples per microsecond


def baseline_removed_cycles(r: TraceRegions, pixel: int) -> np.ndarray:
    """Skip cycles of one pixel with each cycle's pedestal-window mean subtracted."""
    cycles = r.skip[pixel].astype(float)
    return cycles - cycles[:, :r.n_ped].mean(axis=1, keepdims=True)


def baseline_removed_transfers(transfers: np.ndarray) -> np.ndarray:
    """Transfers with the mean of their first 30 samples subtracted."""
    return transfers - transfers[:, :30].mean(axis=1, keepdims=True)


def plot_overview(trace, r: TraceRegions, out: Path) -> None:
    """Row 2 of the trace (the first row has a partial vertical block)."""
    values = trace.values.astype(float)
    v_lo, v_hi = r.v_bounds
    first_pixel = next(k for k, (start, _) in enumerate(r.skip_bounds) if start >= v_hi)
    last_pixel = r.n_pixels - 1
    lo, hi = v_lo, r.skip_bounds[last_pixel][1]

    chunk = 400                                  # min/max envelope: 400 samples per drawn point
    segment = values[lo:hi]
    n = len(segment) // chunk
    blocks = segment[:n * chunk].reshape(n, chunk)
    t_ms = (np.arange(n) * chunk + chunk / 2) / US / 1000

    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.fill_between(t_ms, blocks.min(axis=1), blocks.max(axis=1), color="0.35", lw=0)

    def shade(a, b, color, label):
        ax.axvspan((a - lo) / US / 1000, (b - lo) / US / 1000, color=color, alpha=0.22, lw=0, label=label)

    shade(v_lo, v_hi, "tab:blue", "vertical transfers")
    for k in range(first_pixel, r.n_pixels):
        start, end = r.skip_bounds[k]
        shade(start - r.h_block, start, "tab:orange", "horizontal transfers" if k == first_pixel else None)
        shade(start, end, "tab:green", "skip region" if k == first_pixel else None)
    ax.set(xlabel="time (ms)", ylabel="ADC value (ADU), min/max envelope",
           title="One row of the trace: vertical transfers, then horizontal transfers and skip region of each pixel")
    ax.set_xlim(0, (hi - lo) / US / 1000)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_templates(r: TraceRegions, pixel: int, out: Path) -> None:
    """Mean shape and scatter of the vertical, horizontal and skip repeats."""
    sets = [(f"Vertical transfer ({len(r.v)} used, {r.v.shape[1] / US:.0f} us each)",
             baseline_removed_transfers(r.v)),
            (f"Horizontal transfer ({len(r.h[pixel])} used, {r.h[pixel].shape[1] / US:.0f} us each)",
             baseline_removed_transfers(r.h[pixel])),
            (f"Skip cycle ({len(r.skip[pixel])} cycles, {r.period / US:.1f} us each)",
             baseline_removed_cycles(r, pixel))]
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    for row, (title, repeats) in enumerate(sets):
        t = np.arange(repeats.shape[1]) / US
        mean, sigma = repeats.mean(axis=0), repeats.std(axis=0, ddof=1)
        ax = axes[row, 0]
        for repeat in repeats[:4]:
            ax.plot(t, repeat, color="0.7", lw=0.6)
        ax.plot([], [], color="0.7", lw=0.6, label="single repeats (4 shown)")
        ax.fill_between(t, mean - sigma, mean + sigma, color="tab:blue", alpha=0.3, label="mean +- 1 sigma")
        ax.plot(t, mean, color="tab:blue", lw=1.2, label=f"mean of {len(repeats)} repeats")
        ax.set(title=title + ": template", xlabel="time in repeat (us)", ylabel="ADU (baseline removed)")
        ax.legend(fontsize=8)
        ax = axes[row, 1]
        ax.semilogy(t, np.maximum(sigma, 0.1), color="tab:red", lw=0.8)
        ax.set(title="scatter between repeats", xlabel="time in repeat (us)", ylabel="sigma (ADU)")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_why_mean(r: TraceRegions, pixel: int, other: TraceRegions | None, out: Path) -> None:
    """One cycle against the mean; noise against N; optional two-trace comparison."""
    cycles = baseline_removed_cycles(r, pixel)
    t = np.arange(cycles.shape[1]) / US
    n_panels = 3 if other is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.3 * n_panels, 4.6))

    ax = axes[0]
    window = slice(int((r.sig_start / US + 0.2) * US), int(r.sig_start + r.n_ped - 5))   # flat signal window
    ax.plot(t[window], cycles[7][window], color="0.6", lw=0.8, label="one cycle")
    ax.plot(t[window], cycles[:, window].mean(axis=0), color="tab:blue", lw=1.5,
            label=f"mean of {len(cycles)} cycles")
    ax.set(title="A  signal window, zoomed: one cycle vs the mean", xlabel="time in cycle (us)",
           ylabel="ADU (baseline removed)", ylim=(-250, 250))
    ax.legend(fontsize=8)

    ax = axes[1]
    pedestal = cycles[:, :r.n_ped]
    sizes = [1, 2, 4, 5, 10, 20, 25, 50]
    noise = []
    for n in sizes:
        means = pedestal[:(len(pedestal) // n) * n].reshape(-1, n, r.n_ped).mean(axis=1)
        noise.append(means.std(axis=0, ddof=1).mean())
    ax.loglog(sizes, noise, "o-", color="tab:purple", label="noise of an N-cycle average")
    ax.loglog(sizes, noise[0] / np.sqrt(sizes), "k--", label="1/sqrt(N)")
    ax.set(title="B  averaging N cycles shrinks the noise", xlabel="number of cycles averaged N",
           ylabel="rms noise per sample (ADU)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    if other is not None:
        ax = axes[2]
        cycles_b = baseline_removed_cycles(other, pixel)
        mean_a, mean_b = cycles.mean(axis=0), cycles_b.mean(axis=0)
        err = np.sqrt(cycles.var(axis=0, ddof=1) / len(cycles) + cycles_b.var(axis=0, ddof=1) / len(cycles_b))
        z = np.abs(mean_a - mean_b) / (err + 1e-9)
        ax.plot(t, mean_a, color="tab:blue", label="trace A mean")
        ax.plot(t, mean_b, color="tab:orange", label="trace B mean")
        ax.set(xlabel="time in cycle (us)", ylabel="ADU")
        ax2 = ax.twinx()
        ax2.plot(t, z, color="tab:red", lw=0.5, alpha=0.6)
        ax2.set(yscale="log", ylabel="|z| = |difference| / error of the means")
        ax2.yaxis.label.set_color("tab:red")
        ax.set_title("C  mean shapes of two traces and where they differ\n(two amplifiers, identical parameters)")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace")
    parser.add_argument("out_dir")
    parser.add_argument("--other", help="second trace for panel C of why_mean_shape.png")
    parser.add_argument("--pixel", type=int, default=6, help="pixel used for the templates")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = decode_trace(args.trace)
    regions = split_regions(trace)
    other = split_regions(decode_trace(args.other)) if args.other else None

    plot_overview(trace, regions, out_dir / "regions_overview.png")
    plot_templates(regions, args.pixel, out_dir / "regions_templates.png")
    plot_why_mean(regions, args.pixel, other, out_dir / "why_mean_shape.png")
    print(f"wrote 3 figures to {out_dir}")


if __name__ == "__main__":
    main()
