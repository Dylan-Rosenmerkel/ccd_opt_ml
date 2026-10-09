"""
trace_regions.py -- split a decoded trace into its three regions and stack the repeats.

A trace of a skipper-CCD readout consists of (see ``decode_trace.py`` for the file format)

    vertical transfers   NPBIN repeats of one vertical transfer, once per row
    horizontal transfers NSBIN repeats of one horizontal transfer, before every pixel
    skip region          NDCM skip cycles (pedestal window, signal window), per pixel

The boundaries are found from the data itself: the pedestal-window flags give the skip
cycles exactly, the gaps between skip regions give the horizontal block (and, at a row
change, the vertical block), and flag bits 6 (horizontal) and 5 (vertical) mark the start
of every individual transfer.  Nothing is averaged: every repeat is kept as its own row.

Flag-bit meaning (bit 5 = vertical transfer, bit 6 = horizontal transfer) was inferred from
where these bits toggle in real traces; it is not documented by the DAQ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from decode_trace import DecodedTrace, decode_trace

_V_BIT, _H_BIT = 5, 6


@dataclass
class TraceRegions:
    """The regions of one trace.  All waveforms are float ADU, one row per repeat."""

    n_ped: int                       # samples in one pedestal window
    sig_start: int                   # sample (within a skip cycle) where the signal window starts
    period: int                      # samples in one skip cycle
    skip: list[np.ndarray]           # per pixel: (n_skip, period), cycle starts at its pedestal window
    cds: list[np.ndarray]            # per pixel: (n_skip,) pedestal - signal from the inline L1 data
    h: list[np.ndarray]              # per pixel: (n_transfer, h_len) horizontal transfers
    v: np.ndarray | None             # (n_transfer, v_len) vertical transfers of the first full block
    h_period: float                  # samples between horizontal transfers (not an integer)
    v_period: float | None           # samples between vertical transfers
    # positions in the decoded sample stream (for plots): (start, end) sample numbers
    skip_bounds: list = field(default_factory=list)   # skip region of every pixel
    h_block: int = 0                                  # length of the horizontal block before each pixel
    v_bounds: tuple | None = None                     # the vertical block whose transfers are in ``v``

    @property
    def n_pixels(self) -> int:
        return len(self.skip)


def _rising(flags: np.ndarray, bit: int) -> np.ndarray:
    """Sample numbers where flag ``bit`` goes from 0 to 1."""
    x = ((flags >> bit) & 1).astype(np.int8)
    return np.flatnonzero(np.diff(x) == 1) + 1


def _stack(values: np.ndarray, edges: np.ndarray, length: int, last_allowed_end: int) -> np.ndarray:
    """Stack ``values[e:e+length]`` for every start ``e`` whose segment ends before ``last_allowed_end``."""
    rows = [values[e:e + length] for e in edges if e + length <= last_allowed_end]
    return np.array(rows, dtype=float) if rows else np.empty((0, length))


def split_regions(trace: DecodedTrace) -> TraceRegions:
    """Split ``trace`` into skip cycles, horizontal transfers and vertical transfers."""
    values = trace.values.astype(float)
    flags = trace.flags.astype(np.int64)

    # ---- skip regions: runs of pedestal windows separated by long gaps -------------------
    starts, ends = trace.window_slices()
    period = int(np.median(np.diff(starts)))
    n_ped = int(np.median(ends - starts))
    ru_starts, _ = trace.signal_window_slices()
    # Each pixel has one extra pedestal window, which would shift the pairing after the first
    # pixel, so the offset is measured on the first skips of the trace only.
    n_pairs = min(100, len(starts), len(ru_starts))
    sig_start = int(np.median(ru_starts[:n_pairs] - starts[:n_pairs]))
    breaks = np.flatnonzero(np.diff(starts) > 2 * period)
    first = np.r_[0, breaks + 1]
    last = np.r_[breaks, len(starts) - 1]
    skip, bounds = [], []
    for a, b in zip(first, last):
        # the last window of a pixel is the extra pedestal: not a full skip cycle, so excluded
        skip.append(np.stack([values[s:s + period] for s in starts[a:b]]))
        bounds.append((int(starts[a]), min(int(starts[b]) + period, len(values))))

    l1 = trace.skips()
    pixel_order = list(dict.fromkeys(l1["pixel"].tolist()))
    cds = [l1["cds"][l1["pixel"] == p] for p in pixel_order][:len(skip)]

    # ---- horizontal transfers: the gap between two skip regions of the same row ------------
    e_h, e_v = _rising(flags, _H_BIT), _rising(flags, _V_BIT)
    h_period = float(np.median(np.diff(e_h)[np.diff(e_h) < 5000])) if len(e_h) > 2 else float("nan")
    gap_lengths = np.array([bounds[i + 1][0] - bounds[i][1] for i in range(len(bounds) - 1)])
    h_block = int(np.median(gap_lengths)) if len(gap_lengths) else 0   # horizontal-only gap
    h_len = int(round(h_period))
    h = []
    for start, _ in bounds:
        lo, hi = start - h_block - 100, start
        edges = e_h[(e_h >= lo) & (e_h < hi)]
        h.append(_stack(values, edges, h_len, hi))

    # ---- vertical transfers: the longer gap at a row change -----------------------------------
    v, v_period, v_bounds = None, None, None
    long_gaps = np.flatnonzero(gap_lengths > 1.5 * h_block) if h_block else []
    short_v = np.diff(e_v)[np.diff(e_v) < 20000] if len(e_v) > 2 else np.empty(0)
    # On some boards (e.g. module 102, bcf with vertical = Vtransfer21) flag bit 5 does not mark
    # every vertical transfer, so no transfer period can be found; the vertical region is then
    # left out (v = None).  The vertical tools are information only anyway.
    if len(long_gaps) and len(short_v):
        v_period = float(np.median(short_v))
        i = int(long_gaps[0])
        v_lo, v_hi = bounds[i][1], bounds[i + 1][0] - h_block
        edges = e_v[(e_v >= v_lo) & (e_v < v_hi)]
        v = _stack(values, edges, int(round(v_period)), v_hi)[1:]   # first transfer follows the skip region
        v_bounds = (int(v_lo), int(v_hi))
    return TraceRegions(n_ped, sig_start, period, skip, cds, h, v, h_period, v_period,
                        bounds, h_block, v_bounds)


def load_regions(path: str) -> TraceRegions:
    """Decode ``path`` and split it into regions."""
    return split_regions(decode_trace(path))
