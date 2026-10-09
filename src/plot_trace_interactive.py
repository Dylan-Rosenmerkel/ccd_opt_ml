"""
plot_trace_interactive.py -- an interactive matplotlib window for a whole decoded trace.

Opens a normal matplotlib figure (with the usual toolbar: zoom rectangle, pan, home/back/forward)
showing the ENTIRE trace of a ``.RJ45ADCDATA<ch>`` file.  Whenever the visible range changes the line is
rebuilt for the new range, so zooming from the full 200 ms down to single samples stays fast:

    * zoomed out: every pixel column of the plot is drawn as the min..max of the samples under it
      (a min/max pyramid is built once, so this is instant);
    * zoomed in: the individual samples are drawn, with markers when they are far apart.

Extras on top of the standard toolbar
    mouse wheel        zoom the time axis around the pointer (Shift + wheel: zoom the y axis)
    keys               left/right: pan, + / -: zoom, h: full trace, a: toggle automatic y scale
    check boxes        region shading (vertical / horizontal / skip), pedestal and signal windows,
                       automatic y scale (fits the visible data, ignoring rare outlier samples)
    "Full trace"       button back to the whole trace
    status bar         sample number, time and the window under the pointer

Usage
    python plot_trace_interactive.py TRACE_FILE [--start-us T0 --end-us T1]
    python plot_trace_interactive.py TRACE_FILE --save out.png [--start-us T0 --end-us T1]   # no window
"""

from __future__ import annotations

import argparse

import matplotlib

import numpy as np

from decode_trace import SAMPLE_RATE_HZ, decode_trace
from trace_regions import split_regions

US = SAMPLE_RATE_HZ / 1e6                         # samples per microsecond
_REGION_COLORS = {"vertical": "tab:blue", "horizontal": "tab:orange", "skip": "tab:green"}


def build_pyramid(values: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Min and max of blocks of 1, 2, 4, ... samples (``levels[l]`` has blocks of ``2**l`` samples)."""
    mins, maxs = [values], [values]
    while len(mins[-1]) > 1:
        lo, hi = mins[-1], maxs[-1]
        n = len(lo) // 2 * 2
        new_lo, new_hi = np.minimum(lo[:n:2], lo[1:n:2]), np.maximum(hi[:n:2], hi[1:n:2])
        if len(lo) % 2:
            new_lo, new_hi = np.append(new_lo, lo[-1]), np.append(new_hi, hi[-1])
        mins.append(new_lo)
        maxs.append(new_hi)
    return mins, maxs


class TraceViewer:
    """Figure, artists and callbacks of the interactive viewer."""

    def __init__(self, path: str):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons

        self.plt = plt
        self.trace = decode_trace(path)
        self.regions = split_regions(self.trace)
        self.values = self.trace.values.astype(np.float32)
        self.n = len(self.values)
        self.mins, self.maxs = build_pyramid(self.values)
        self.rd, self.ru = self.trace.rd, self.trace.ru
        self.name = path.split("/")[-1]

        self.fig, self.ax = plt.subplots(figsize=(14, 6.5))
        self.fig.subplots_adjust(left=0.27, right=0.98, bottom=0.12, top=0.92)
        self.ax.set_xlabel("time (us)   [sample number / 15 MHz]")
        self.ax.set_ylabel("ADC value (ADU)")
        (self.line,) = self.ax.plot([], [], color="0.1", lw=0.8, marker="", ms=3)
        self.window_patches: list = []
        self.auto_y = True
        self._updating = False

        # region shading (vertical, horizontal, skip)
        self.region_patches = []
        r = self.regions
        spans = [("skip", a, b) for a, b in r.skip_bounds] + [("horizontal", a - r.h_block, a) for a, _ in r.skip_bounds]
        if r.v_bounds:
            spans.append(("vertical",) + tuple(r.v_bounds))
        for kind, a, b in spans:
            self.region_patches.append(self.ax.axvspan(a / US, b / US, color=_REGION_COLORS[kind], alpha=0.12, lw=0))
        self.show_windows = True

        # widgets
        ax_check = self.fig.add_axes([0.005, 0.55, 0.15, 0.2])
        self.checks = CheckButtons(ax_check, ["regions", "pedestal/signal\nwindows", "auto y scale"], [True, True, True])
        self.checks.on_clicked(self.on_check)
        ax_btn = self.fig.add_axes([0.005, 0.45, 0.15, 0.06])
        self.button = Button(ax_btn, "Full trace")
        self.button.on_clicked(lambda _e: self.show(0, self.n))
        ax_leg = self.fig.add_axes([0.005, 0.15, 0.15, 0.26])
        ax_leg.axis("off")
        for k, (kind, color) in enumerate(_REGION_COLORS.items()):
            ax_leg.add_patch(self.plt.Rectangle((0, 0.8 - 0.2 * k), 0.15, 0.12, color=color, alpha=0.4))
            ax_leg.text(0.2, 0.86 - 0.2 * k, kind, va="center", fontsize=9)
        for k, (label, color) in enumerate([("pedestal window", "tab:red"), ("signal window", "tab:cyan")]):
            ax_leg.add_patch(self.plt.Rectangle((0, 0.2 - 0.2 * k), 0.15, 0.12, color=color, alpha=0.4))
            ax_leg.text(0.2, 0.26 - 0.2 * k, label, va="center", fontsize=9)

        self.ax.callbacks.connect("xlim_changed", self.on_xlim)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.ax.format_coord = self.format_coord
        self.show(0, self.n)

    # ---- data for the visible range -------------------------------------------------------------------------
    def envelope(self, i0: int, i1: int, n_cols: int):
        """x (us) and y arrays that draw samples ``[i0, i1)`` in about ``n_cols`` pixel columns."""
        span = i1 - i0
        if span <= 3 * n_cols:                                  # few samples per column: draw them all
            idx = np.arange(i0, i1)
            return idx / US, self.values[i0:i1], True
        size = span / n_cols
        level = min(len(self.mins) - 1, int(np.floor(np.log2(size))))
        block = 1 << level
        edges = (i0 + np.arange(n_cols + 1) * size / 1.0) / block
        idx = np.clip(np.floor(edges).astype(int), 0, len(self.mins[level]) - 1)
        idx = np.maximum.accumulate(idx)
        sub = slice(idx[0], idx[-1] + 1)
        starts = idx[:-1] - idx[0]
        lo = np.minimum.reduceat(self.mins[level][sub], starts)
        hi = np.maximum.reduceat(self.maxs[level][sub], starts)
        xc = (i0 + (np.arange(n_cols) + 0.5) * size) / US
        x = np.repeat(xc, 2)
        y = np.column_stack([lo, hi]).ravel()
        return x, y, False

    # ---- redrawing ---------------------------------------------------------------------------------------------------
    def on_xlim(self, _ax) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self.update()
        finally:
            self._updating = False

    def update(self) -> None:
        x0, x1 = self.ax.get_xlim()
        i0, i1 = max(0, int(np.floor(x0 * US))), min(self.n, int(np.ceil(x1 * US)))
        if i1 - i0 < 2:
            return
        n_cols = max(100, int(self.ax.get_window_extent().width))
        x, y, raw = self.envelope(i0, i1, n_cols)
        self.line.set_data(x, y)
        self.line.set_marker("." if raw and (i1 - i0) < n_cols / 4 else "")
        self.line.set_linewidth(0.8 if raw else 0.6)

        if self.auto_y:
            lo_v, hi_v = (y.min(), y.max())
            if not raw and len(y) > 400:                       # ignore the most extreme 0.5% of columns
                lows, highs = y[0::2], y[1::2]
                lo_v, hi_v = np.percentile(lows, 0.5), np.percentile(highs, 99.5)
            pad = 0.05 * (hi_v - lo_v or 1)
            self.ax.set_ylim(lo_v - pad, hi_v + pad)

        for patch in self.window_patches:
            patch.remove()
        self.window_patches = []
        if self.show_windows and (i1 - i0) < 60000:
            for mask, color in ((self.rd, "tab:red"), (self.ru, "tab:cyan")):
                change = np.diff(np.r_[0, mask[i0:i1].view(np.int8), 0])
                for a, b in zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)):
                    self.window_patches.append(
                        self.ax.axvspan((i0 + a) / US, (i0 + b) / US, color=color, alpha=0.18, lw=0))
        span_us = (x1 - x0)
        self.ax.set_title(f"{self.name}\nshowing {span_us / 1000:.3f} ms = {i1 - i0:,} of {self.n:,} samples", fontsize=10)
        self.fig.canvas.draw_idle()

    def show(self, i0: float, i1: float) -> None:
        """Set the visible sample range (also resets the y range)."""
        self.ax.set_xlim(i0 / US, i1 / US)
        self.update()

    # ---- interaction ------------------------------------------------------------------------------------------------------
    def on_check(self, label: str) -> None:
        states = dict(zip(["regions", "pedestal/signal\nwindows", "auto y scale"], self.checks.get_status()))
        for patch in self.region_patches:
            patch.set_visible(states["regions"])
        self.show_windows = states["pedestal/signal\nwindows"]
        self.auto_y = states["auto y scale"]
        self.update()

    def on_scroll(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        factor = 0.8 if event.button == "up" else 1.25
        if event.key == "shift":
            lo, hi = self.ax.get_ylim()
            self.ax.set_ylim(event.ydata - (event.ydata - lo) * factor, event.ydata + (hi - event.ydata) * factor)
            self.fig.canvas.draw_idle()
            return
        x0, x1 = self.ax.get_xlim()
        new0, new1 = event.xdata - (event.xdata - x0) * factor, event.xdata + (x1 - event.xdata) * factor
        new0, new1 = max(0, new0), min(self.n / US, new1)
        self.ax.set_xlim(new0, new1)

    def on_key(self, event) -> None:
        x0, x1 = self.ax.get_xlim()
        span = x1 - x0
        if event.key == "left":
            self.ax.set_xlim(x0 - 0.2 * span, x1 - 0.2 * span)
        elif event.key == "right":
            self.ax.set_xlim(x0 + 0.2 * span, x1 + 0.2 * span)
        elif event.key in ("+", "="):
            c = (x0 + x1) / 2
            self.ax.set_xlim(c - 0.4 * span, c + 0.4 * span)
        elif event.key == "-":
            c = (x0 + x1) / 2
            self.ax.set_xlim(c - 0.625 * span, c + 0.625 * span)
        elif event.key == "h":
            self.show(0, self.n)
        elif event.key == "a":
            self.auto_y = not self.auto_y
            self.update()

    def format_coord(self, x, y) -> str:
        i = int(round(x * US))
        if not 0 <= i < self.n:
            return f"t = {x:.2f} us"
        window = " [pedestal window]" if self.rd[i] else (" [signal window]" if self.ru[i] else "")
        return f"sample {i:,}   t = {x:.2f} us   y = {y:.0f}   (ADC at this sample: {self.values[i]:.0f}){window}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace")
    parser.add_argument("--start-us", type=float, help="start of the initial view (us)")
    parser.add_argument("--end-us", type=float, help="end of the initial view (us)")
    parser.add_argument("--save", help="write the initial view to this image and exit (no window)")
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")                               # no window needed
    viewer = TraceViewer(args.trace)
    if args.start_us is not None or args.end_us is not None:
        t0 = 0 if args.start_us is None else args.start_us
        t1 = viewer.n / US if args.end_us is None else args.end_us
        viewer.show(t0 * US, t1 * US)
    if args.save:
        viewer.fig.savefig(args.save, dpi=100)
        print(f"wrote {args.save}")
    else:
        viewer.plt.show()


if __name__ == "__main__":
    main()
