"""
simulate_trace.py -- write a synthetic trace in the real ACM file format, for any geometry and timing.

Only one trace geometry exists in the real data (2x5 pixels, 500 skips, 20x20 binning, reference
timing).  This module writes traces with ANY number of rows, columns, skips, binning and any
sequencer delays, so the analysis code (``decode_trace``, ``trace_regions``, ``compare_trace``,
``trace_features``) can be tested on all of them (``check_trace_analysis.py``).

The file format is the one of the real traces (see ``decode_trace.py``):

* a few leftover sample words, then the START, BUSY and header keyword words;
* raw samples: top nibble 2, flag byte (bit 0 RD = pedestal window, bit 1 RU = signal window,
  bit 6 high during the first part of every horizontal transfer, bit 7 high and bit 5 low during
  the first part of every vertical transfer, bits 2 and 4 set otherwise, as in real traces),
  20-bit signed ADC value;
* after every window, an inline L1 record (header word with the pixel number in 4 bits and the
  window length, then the 31-bit sum of the window's samples);
* a "LINE" marker word at every vertical transfer; the END keyword at the end.

Timing follows the sequencer formulas of ``trace_geometry.py`` (10 ns ticks, 15 MHz ADC): every
event is placed at its exact tick and converted to a sample number, so the skip cycle has the same
non-integer length (e.g. 488.7 samples) as in real traces.

The WAVEFORM is a rough imitation of the real one (reset pulse with a two-stage recovery and 3 MHz
ringing, summing-well dip, baseline droop along the skips, a new random level after every reset,
white noise, charge in the pixels, a slower baseline after vertical transfers).  It is meant to
exercise the code, not to model the CCD: features of simulated traces are not physics.

Usage
-----
    from simulate_trace import simulate_trace
    simulate_trace("out/trace_sim", NROW=3, NCOL=7, NDCM=50, NSBIN=1, NPBIN=5, delayInt=500)
    # writes out/trace_sim.RJ45ADCDATA0 and out/trace_sim.meta

    python simulate_trace.py OUT_STEM [--NDCM 50 --NCOL 7 ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trace_geometry import SAMPLES_PER_TICK

START, BUSY, HEADER, END = 0xD3544152, 0xC2555359, 0xE0B403E8, 0xA0454E44
LINE = 0x4C494E45
BASE_FLAGS = 0x34                    # bits 2, 4, 5 (the idle state of real traces)

#: Defaults: the trace geometry and the reference timing of reference_v5_pumping.seq.
DEFAULTS = dict(NROW=2, NCOL=5, NDCM=500, NPBIN=20, NSBIN=20, EXPOSURE=0,
                delayOGH=3, delayOGL=48, delayRGL=27, delayRGH=800, delayInt=1000,
                delaySWH=30, delaySWL=350, delayDGL=27, delayDGH=47,
                delay_Ha=1800, delay_Hb=1800, delay_Hc=1800, delay_Va=6000, delay_Vb=6000,
                SW_1_H=1.0, SW_1_L=-6.0, RG_1_H=9.0, RG_1_L=3.0)


def _events(p: dict) -> list[tuple]:
    """
    The readout as a list of (kind, start tick, end tick, info), in time order.

    kinds: "idle", "vtransfer", "htransfer", "front" (OG + reset before a window), "ped", "sw",
    "sig", "drain".  The pedestal/signal windows carry the pixel number and skip index.
    """
    t = 0.0
    out = []

    def add(kind, ticks, **info):
        nonlocal t
        out.append((kind, t, t + ticks, info))
        t += ticks

    front = p["delayOGH"] + p["delayOGL"] + p["delayRGL"] + p["delayRGH"]
    t_h = p["delayRGL"] + 3 * p["delay_Ha"] + 3 * p["delay_Hb"] + p["delay_Hc"]
    t_v = p["delayRGL"] + 3 * p["delay_Va"] + 3 * p["delay_Vb"]
    add("idle", 2000)
    pixel = 0
    for row in range(p["NROW"]):
        for _ in range(p["NPBIN"]):
            add("vtransfer", t_v, first_part=3 * p["delay_Va"])
        for col in range(p["NCOL"]):
            for _ in range(p["NSBIN"]):
                add("htransfer", t_h, first_part=3 * p["delay_Ha"])
            for k in range(p["NDCM"]):
                add("front", front, pixel=pixel, skip=k)
                add("ped", p["delayInt"], pixel=pixel, skip=k)
                add("sw", p["delaySWH"] + p["delaySWL"], pixel=pixel, skip=k)
                add("sig", p["delayInt"], pixel=pixel, skip=k)
            add("front", front, pixel=pixel, skip=p["NDCM"])
            add("ped", p["delayInt"], pixel=pixel, skip=p["NDCM"])        # the extra pedestal
            add("drain", p["delaySWH"] + p["delayDGL"] + p["delayDGH"], pixel=pixel)
            pixel += 1
    add("idle", 2000)
    return out


def simulate_trace(stem: str | Path, seed: int = 0, charge_per_pixel: float = 5.0,
                   charged_pixels: tuple = (), channel: int = 0, **params) -> Path:
    """
    Write ``<stem>.RJ45ADCDATA<channel>`` and ``<stem>.meta``; return the data file path.

    ``params`` override ``DEFAULTS`` (geometry and sequencer delays).  ``charge_per_pixel`` is the
    mean charge (electrons) of every pixel; pixels in ``charged_pixels`` get 15,000 electrons.
    """
    p = {**DEFAULTS, **params}
    rng = np.random.default_rng(seed)
    events = _events(p)
    n_samples = int(np.floor(events[-1][2] * SAMPLES_PER_TICK)) + 1
    values = np.zeros(n_samples)
    flags = np.full(n_samples, BASE_FLAGS, dtype=np.int64)

    def span(a, b):
        return int(np.floor(a * SAMPLES_PER_TICK)), int(np.floor(b * SAMPLES_PER_TICK))

    gain, n_ped = 1.34, p["delayInt"] * SAMPLES_PER_TICK
    charge = {}                                               # electrons per pixel
    baseline0 = -20000.0
    level = baseline0
    after_vertical = 0.0                                      # extra offset after vertical transfers
    l1 = []                                                   # (sample index after which to insert, header, sum)
    lines = []
    for kind, a, b, info in events:
        i0, i1 = span(a, b)
        n = i1 - i0
        if n <= 0:
            continue
        tt = np.arange(n, dtype=float)
        if kind == "vtransfer":
            first = int(info["first_part"] * SAMPLES_PER_TICK)
            flags[i0:i0 + first] = (BASE_FLAGS & ~0x20) | 0x80 | 0x08     # bit 7 high, bit 5 low
            values[i0:i1] = level - 12000 * np.exp(-tt / 300) + 9000 * (tt > first) * np.exp(-(tt - first) / 400)
            after_vertical = -12000.0
            lines.append(i0)
        elif kind == "htransfer":
            first = int(info["first_part"] * SAMPLES_PER_TICK)
            flags[i0:i0 + first] = BASE_FLAGS | 0x40 | 0x08               # bit 6 high
            third = max(1, n // 3)
            values[i0:i1] = level - 50000 * np.exp(-tt / (0.3 * third)) * (tt < third) \
                + 35000 * (tt >= third) * np.exp(-(tt - third) / (0.2 * third))
        elif kind == "front":
            k = info["skip"]
            droop = 15000 * (1 - np.exp(-k / 236.0))
            after_vertical *= 0.996
            level = baseline0 + droop + after_vertical + rng.normal(0, 150)   # new level after the reset
            pulse_at = int((p["delayOGH"] + p["delayOGL"] + p["delayRGL"]) * SAMPLES_PER_TICK)
            rise = np.clip((tt - pulse_at) / 5.0, 0, 1)
            decay = np.where(tt > pulse_at + 5, 25000 * np.exp(-(tt - pulse_at - 5) / 1.5)
                             + 21000 * np.exp(-(tt - pulse_at - 5) / 13.0)
                             + 600 * np.exp(-(tt - pulse_at - 5) / 5.0) * np.sin(2 * np.pi * 0.2 * (tt - pulse_at - 5)), 0)
            values[i0:i1] = level + 46000 * rise * (tt <= pulse_at + 5) + decay
        elif kind == "ped":
            flags[i0:i1] |= 0x01
            values[i0:i1] = level + 10 * np.exp(-tt / 20)
        elif kind == "sw":
            pix = info["pixel"]
            if pix not in charge:
                charge[pix] = 15000 if pix in charged_pixels else rng.poisson(charge_per_pixel)
            values[i0:i1] = level - 13000 * np.exp(-tt / 4.5) - charge[pix] * gain * (1 - np.exp(-tt / 2))
            sig_level = level - charge[pix] * gain * (1 - 0.0001 * info["skip"])   # a little charge loss
        elif kind == "sig":
            flags[i0:i1] |= 0x02
            values[i0:i1] = sig_level - 150 * np.exp(-tt / 8) + 0.002 * (level - baseline0)
        elif kind in ("drain", "idle"):
            values[i0:i1] = level
        if kind in ("ped", "sig"):
            noise = rng.normal(0, 53, n)
            values[i0:i1] += noise
            window = np.round(values[i0:i1]).astype(np.int64)
            header = (0x5 << 28) | ((((0x80 if kind == "sig" else 0) | (info["pixel"] & 0xF)) & 0xFF) << 20) | (n & 0xFFFFF)
            l1.append((i1, header, int(window.sum())))
        else:
            values[i0:i1] += rng.normal(0, 53, n)

    adc = np.clip(np.round(values), -2**19, 2**19 - 1).astype(np.int64)
    raw = ((0x2 << 28) | ((flags & 0xFF) << 20) | (adc & 0xFFFFF)).astype(np.uint32)

    # interleave: raw samples, with L1 records after their windows and LINE markers
    inserts = sorted([(i, [h, s & 0x7FFFFFFF]) for i, h, s in l1] + [(i, [LINE]) for i in lines], key=lambda x: x[0])
    pieces, previous = [], 0
    for position, words in inserts:
        pieces.append(raw[previous:position]); pieces.append(np.array(words, dtype=np.uint32)); previous = position
    pieces.append(raw[previous:])
    body = np.concatenate(pieces)
    leftover = raw[:5]
    words = np.concatenate([leftover, np.array([START, BUSY, HEADER], dtype=np.uint32), body,
                            np.array([END], dtype=np.uint32)])

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    path = stem.parent / f"{stem.name}.RJ45ADCDATA{channel}"
    words.astype(">u4").tofile(path)
    meta = {"SIMULATED": True, "SEQNAME": "reference_v5_pumping.seq", "ROUTINE": "AcquireImage",
            "binname": stem.name, **{k: v for k, v in p.items()},
            **{f"FILTER{i}": (0 if i == channel else 3) for i in range(4)},
            **{f"ch{i}": f"CCD_{'ABCD'[i]}" for i in range(4)}}
    stem.with_suffix(".meta").write_text(json.dumps(meta, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic trace in the ACM format.")
    parser.add_argument("stem")
    parser.add_argument("--seed", type=int, default=0)
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    args = vars(parser.parse_args())
    stem, seed = args.pop("stem"), args.pop("seed")
    print(simulate_trace(stem, seed=seed, **args))


if __name__ == "__main__":
    main()
