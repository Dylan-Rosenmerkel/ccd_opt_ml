"""
trace_features.py -- every feature of one raw trace, computed by one function.

This merges the two earlier feature sets:

* the 39 **comparison tools** of ``compare_trace.py`` (11 fits, 19 skip tools, 5 horizontal,
  4 vertical), computed per replicate (pixel, or vertical transfer) and summarised here as the
  median over the usable replicates;
* the **advanced features** of ``notebooks/advanced_trace_features.ipynb`` (CDS level, noise
  character, discrete noise, edges, slow structure).

``compare_trace.py`` and the advanced-features notebook are left unchanged for reference; this
module imports the comparison tools from ``compare_trace.py`` so both always agree.

Usage
-----
    from trace_features import trace_features, FEATURES
    result = trace_features("trace_...ch0_...RJ45ADCDATA0")          # or a DecodedTrace
    result.scalars["noise: reset-level scatter (ADU)"]                 # one number per feature
    result.spread["fit: dip recovery time (us)"]                       # pixel-to-pixel spread
    result.curves["noise: Allan deviation"]                            # fixed-length vectors
    FEATURES["edge: jitter (ns)"].description                          # what it is

    python trace_features.py TRACE [--meta TRACE.meta]                 # print all features

Conventions
-----------
* Names start with their group: ``fit:``, ``skip:``, ``horizontal:``, ``vertical:`` (comparison
  tools) and ``level:``, ``noise:``, ``discrete:``, ``edge:``, ``slow:`` (advanced features), ``start:``
  (the start-of-pixel transient, 2026-10-21) and ``clock:`` (the V-clock blocks and the first H block of a row).
* Every feature has a ``tier``: 0 = the original set, 1 and 2 = added 2026-10-21 in
  ``trace_features_extra.py`` (docs/trace_feature_candidates.md); ``names_by_tier`` lists them.
* ``scalars`` holds one number per feature.  For per-pixel features it is the median over the
  usable pixels (no row-start pixels, no charged pixels), and ``spread`` holds their robust
  pixel-to-pixel spread (1.4826 x median absolute deviation).  A feature that cannot be computed
  for this trace (e.g. charge loss without a charged pixel, vertical tools when the vertical
  transfers are not found) is NaN.
* ``curves`` holds vectors (mean shapes, autocorrelation, spectra, ...), always of the same length
  for the same trace geometry.
* The features describe the trace; they are candidate inputs for a model, not calibrated physical
  quantities.  Every feature is documented in ``FEATURES`` and in ``docs/trace_features.tex``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from compare_trace import detrend_degree, fit_exponential, measure, row_start_pixels, skip_counts, usable_pixels
from decode_trace import SAMPLE_RATE_HZ, DecodedTrace, decode_trace
from trace_regions import TraceRegions, split_regions

REFERENCE_NYQUIST = SAMPLE_RATE_HZ / 488.7 / 2   # half the usual skip rate (15.35 kHz)
TRUE_PERIOD = 488.7            # reference skip cycle (3258 ticks x 0.15); traces use their own (regions.period_exact)
FIRST_SKIP = 10                # default for 500 skips; traces use trace_sizes(...)["first_skip"]


# --------------------------------------------------------------------------- #
# The feature registry                                                         #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Feature:
    """Documentation of one feature."""
    group: str          # name prefix
    unit: str
    description: str    # what is computed
    link: str = ""      # why a model might use it
    kind: str = "scalar"   # "scalar" or "curve"
    origin: str = "advanced"   # "comparison" (compare_trace.py), "advanced" or "extra" (trace_features_extra.py)
    tier: int = 0          # 0 = the original 85 + 14; 1, 2 = added 2026-10-21 (docs/trace_feature_candidates.md)


def _f(group, unit, description, link="", kind="scalar", origin="advanced", tier=0):
    return Feature(group, unit, description, link, kind, origin, tier)


_C = "comparison"
FEATURES: dict[str, Feature] = {
    # ---------------------------------------------------------------- comparison tools: fits
    "fit: settling time after pulse (us)": _f("fit", "µs", "Time constant of an exponential fitted to the mean skip shape after the reset-pulse maximum.", "amplifier recovery speed; reset settings", origin=_C),
    "fit: settling amplitude after pulse": _f("fit", "ADU", "Amplitude of that exponential.", "reset feedthrough size", origin=_C),
    "fit: dip recovery time (us)": _f("fit", "µs", "Time constant of an exponential fitted from the minimum of the summing-well dip to the start of the signal window.", "settling before the signal window; SW timing", origin=_C),
    "fit: dip recovery amplitude": _f("fit", "ADU", "Amplitude of that exponential.", "SW feedthrough size", origin=_C),
    "fit: baseline droop time constant (skips)": _f("fit", "skips", "Time constant of an exponential fitted to the per-skip baseline (pedestal-window mean) along the 500 skips of a pixel.", "reset leakage, output-stage operating point", origin=_C),
    "fit: baseline droop amplitude": _f("fit", "ADU", "Amplitude of that exponential.", "reset leakage", origin=_C),
    "fit: first-skip ringing frequency (cyc/skip)": _f("fit", "cycles/skip", "Frequency of a damped oscillation plus a constant offset, fitted to the CDS of skips 1-19 (skip 0 left out: own feature; median of later skips removed). Searched in 0.02-0.5 cycles/skip, refined by a parabola. (Before 2026-10-13 the model had no offset and returned 0.01 for every trace.)", "transient after the horizontal transfers", origin=_C),
    "fit: first-skip ringing decay (skips)": _f("fit", "skips", "Decay constant of that oscillation.", "transient after the horizontal transfers", origin=_C),
    "fit: first-skip ringing amplitude": _f("fit", "ADU", "Amplitude of that oscillation (envelope extrapolated to skip 0).", "transient after the horizontal transfers", origin=_C),
    "fit: white noise per skip (ADU)": _f("fit", "ADU", "White term σ_w of the model σ(N)² = σ_w²/N + σ_f² fitted to the noise of N-skip averages (N = 1, 2, 5, 10, 25, 50) of the detrended CDS.", "read noise per skip", origin=_C),
    "fit: correlated noise floor (ADU)": _f("fit", "ADU", "Floor term σ_f of that model (often 0: the CDS noise averages down faster than white).", "noise that does not average down", origin=_C),
    # ---------------------------------------------------------------- comparison tools: skip
    "skip: mean shape": _f("skip", "ADU", "Mean waveform of one skip cycle (each skip's baseline removed; skips 0-4 excluded), block-averaged to at most 500 points.", "the complete readout waveform", "curve", _C),
    "skip: baseline droop curve": _f("skip", "ADU", "Per-skip baseline along the pixel minus its value at skip 1.", "reset leakage", "curve", _C),
    "skip: dip height": _f("skip", "ADU", "Minimum of the mean shape between the pedestal and signal windows (summing-well clock feedthrough).", "SW coupling × amplification", origin=_C),
    "skip: signal-window level": _f("skip", "ADU", "Mean of the mean shape inside the signal window (minus 10 samples at each end).", "charge + offset", origin=_C),
    "skip: pulse peak height": _f("skip", "ADU", "Maximum of the mean shape after the signal window (reset pulse).", "RG coupling × amplification", origin=_C),
    "skip: end-of-cycle tail level": _f("skip", "ADU", "Mean of the last 30 samples of the mean shape (before the next pedestal window).", "residual of the reset recovery", origin=_C),
    "skip: baseline at skip 1": _f("skip", "ADU", "Pedestal-window mean of skip 1.", "absolute output level", origin=_C),
    "skip: droop total": _f("skip", "ADU", "Baseline of the last skip minus that of skip 1.", "reset leakage", origin=_C),
    "skip: droop half-time (skips)": _f("skip", "skips", "Skip at which the baseline has moved half of the droop total.", "reset leakage", origin=_C),
    "skip: CDS mean (skips>=10)": _f("skip", "ADU", "Mean CDS (pedestal − signal) of skips 10-499.", "charge in the pixel + offset", origin=_C),
    "skip: CDS rms (detrended)": _f("skip", "ADU", "Standard deviation of the CDS of skips 10-499 after removing a degree-4 polynomial.", "read noise per skip", origin=_C),
    "skip: first-skip CDS excess": _f("skip", "ADU", "CDS of skip 0 minus the median CDS of later skips.", "first-skip transient", origin=_C),
    "skip: skips 1-4 CDS ringing": _f("skip", "ADU", "RMS of the CDS of skips 1-4 around the median of later skips.", "first-skip transient", origin=_C),
    "skip: CDS autocorrelation at lag 2": _f("skip", "", "Autocorrelation of the detrended CDS between skips two apart.", "how the noise averages down", origin=_C),
    "skip: noise of 10-skip average": _f("skip", "ADU", "Standard deviation of 10-skip block means of the detrended CDS.", "noise after averaging", origin=_C),
    "skip: noise of 50-skip average": _f("skip", "ADU", "Standard deviation of 50-skip block means of the detrended CDS.", "noise after averaging", origin=_C),
    "skip: noise power 0.1-1 MHz": _f("skip", "log10 ADU²", "Power, 0.1-1 MHz, of the pedestal-window samples after removing the mean shape (each skip's window mean removed).", "noise spectrum inside a window", origin=_C),
    "skip: noise power 1-3 MHz": _f("skip", "log10 ADU²", "Same, 1-3 MHz.", "noise spectrum inside a window", origin=_C),
    "skip: noise power 3-7.5 MHz": _f("skip", "log10 ADU²", "Same, 3-7.5 MHz.", "noise spectrum inside a window", origin=_C),
    # ---------------------------------------------------------------- comparison tools: transfers
    "horizontal: mean shape": _f("horizontal", "ADU", "Mean waveform of the horizontal transfers before a pixel (first 30 samples set to 0).", "H-clock feedthrough", "curve", _C),
    "horizontal: minimum": _f("horizontal", "ADU", "Minimum of that mean shape.", "H-clock coupling", origin=_C),
    "horizontal: maximum": _f("horizontal", "ADU", "Maximum of that mean shape.", "H-clock coupling", origin=_C),
    "horizontal: end level": _f("horizontal", "ADU", "Mean of the last 50 samples of that mean shape.", "level left after the transfer", origin=_C),
    "horizontal: scatter between transfers": _f("horizontal", "ADU", "Median over samples of the standard deviation between the transfers of one pixel.", "transfer-to-transfer variation", origin=_C),
    "vertical: mean shape": _f("vertical", "ADU", "Mean waveform of one vertical transfer (first 30 samples set to 0). Information only: consecutive vertical transfers are not exchangeable.", "V-clock feedthrough", "curve", _C),
    "vertical: minimum": _f("vertical", "ADU", "Minimum of a vertical-transfer waveform (median over transfers).", "V-clock coupling", origin=_C),
    "vertical: maximum": _f("vertical", "ADU", "Maximum of a vertical-transfer waveform.", "V-clock coupling", origin=_C),
    "vertical: end level": _f("vertical", "ADU", "Mean of the last 50 samples of a vertical-transfer waveform.", "level left after the transfer", origin=_C),
    # ---------------------------------------------------------------- A. CDS level
    "level: common CDS rise (ADU)": _f("level", "ADU", "The median CDS of the usable pixels is a pattern shared by all pixels; its mean over the last 10 % of the skips minus its mean over the 10 % after the first skips (skips 450-499 minus 10-59 for 500 skips).", "baseline leaking into the CDS; output-stage operating point"),
    "level: pixel level spread (ADU)": _f("level", "ADU", "Each pixel's mean CDS minus the common pattern ('relative level'); robust spread (1.4826 MAD) over the usable pixels.", "charge collected by the 400 CCD pixels behind each trace pixel (λ-related)"),
    "level: pixel level max (ADU)": _f("level", "ADU", "Largest relative level among the usable pixels.", "charge, hot pixels"),
    "level: charge-loss slope (%)": _f("level", "%", "For pixels (row-start pixels included) with relative level > 1000 ADU: slope of CDS − common along the skips, × number of skips / level (median over such pixels). NaN if none.", "charge lost or changed between skips; SW/OG settings"),
    "level: charged-pixel skewness": _f("level", "", "Skewness of the CDS − common of those charged pixels. NaN if none.", "charge loss / level-dependent response"),
    "level: common CDS pattern": _f("level", "ADU", "The common pattern itself, skips 0-499.", "", "curve"),
    # ---------------------------------------------------------------- B. noise character
    "noise: CDS autocorrelation at lag 1": _f("noise", "", "Autocorrelation of the detrended CDS (skips 10-499) at lag 1.", "how the noise averages down"),
    "noise: Allan deviation n=50 (ADU)": _f("noise", "ADU", "Allan deviation (scatter between consecutive 50-skip means / √2) of the detrended CDS.", "noise after averaging; σe-like"),
    "noise: Allan slope n=10-100": _f("noise", "", "Log-log slope of the Allan deviation between n = 10 and n = 100 (white noise: −0.5).", "colour of the noise at large n"),
    "noise: CDS power <1 kHz / >10 kHz": _f("noise", "", "Ratio of the mean spectral density of the detrended CDS sequence (sampled at the skip rate) in its lowest 6.5 % to that in its top 35 % (below 1 kHz and above 10 kHz at the usual skip rate of 30.7 kHz).", "1/f vs anticorrelated noise"),
    "noise: reset-level scatter (ADU)": _f("noise", "ADU", "Skip-to-skip standard deviation of each cycle's pedestal-window level after removing the slow droop (polynomial across skips at each sample position). Each reset sets a new random level; the CDS cancels it.", "sense node (kTC) and reset gate; related to the gain"),
    "noise: within-cycle power 0.1-0.3 MHz": _f("noise", "fraction", "Fraction of the noise power above 0.1 MHz that lies in 0.1-0.3 MHz, from the pedestal-window start to the signal-window end (no reset inside), each cycle's own mean removed.", "which part of the chain makes the noise the CDS keeps"),
    "noise: within-cycle power 0.3-1 MHz": _f("noise", "fraction", "Same, 0.3-1 MHz.", ""),
    "noise: within-cycle power 1-3 MHz": _f("noise", "fraction", "Same, 1-3 MHz.", ""),
    "noise: within-cycle power 3-7.5 MHz": _f("noise", "fraction", "Same, 3-7.5 MHz.", ""),
    "noise: window-length ratio 75/150": _f("noise", "", "CDS recomputed from the raw samples with half-length windows at the start of the real windows; ratio of its detrended noise to that with the full windows (75/150 samples for the usual windows; white noise: 1.41).", "how the noise behaves within the integration; delayInt"),
    "noise: late/early window ratio": _f("noise", "", "Noise of half-length windows ending at the end of the real windows divided by that of half-length windows starting at their start.", "settling at the window start"),
    "noise: pedestal-window settling (ADU)": _f("noise", "ADU", "Mean waveform inside the pedestal window (window mean removed): last 10 samples minus first 10.", "window not flat: timing sensitivity"),
    "noise: signal-window settling (ADU)": _f("noise", "ADU", "Same for the signal window.", "signal still settling after the SW clock; SW/OG delays"),
    "noise: baseline coupling (1e-3)": _f("noise", "×10⁻³", "Slope of the CDS against the skip's baseline level over skips 10-499 of a pixel (the baseline droops by thousands of ADU).", "common-mode rejection of the output stage; VDD, VDRAIN, VR"),
    "noise: noise trend along skips (ADU/100 skips)": _f("noise", "ADU/100 skips", "Slope of the standard deviation of 50-skip blocks of the detrended CDS along the pixel.", "noise depending on the amplifier operating point"),
    "noise: CDS autocorrelation": _f("noise", "", "Autocorrelation of the detrended CDS, lags 1-50.", "", "curve"),
    "noise: Allan deviation": _f("noise", "ADU", "Allan deviation of the detrended CDS for n = 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 160. Points with n > N/5 are biased low: the degree-4 detrending absorbs most of the difference between so few block means (n = 160 of 490 skips: about 0.4 of the white-noise value).", "", "curve"),
    "noise: CDS spectrum": _f("noise", "ADU²/Hz", "Spectral density of the detrended CDS sequence, 0-15 kHz.", "", "curve"),
    "noise: Allan deviation, common pattern removed": _f("noise", "ADU", "Overlapping Allan deviation (all windows of n skips; n as above) of each pixel's CDS minus the mean CDS of all usable pixels at the same skip, times sqrt(P/(P-1)) for P pixels. The pattern along the skips that every pixel shares is not noise for an image (it shifts every pixel alike), but no polynomial is removed: slow noise that differs from pixel to pixel (a noise floor) stays in, as it does in an image.", "", "curve"),
    "noise: CDS spectrum, log frequency": _f("noise", "ADU²/Hz", "Spectral density of the same series (common pattern removed, not detrended), averaged in 40 logarithmically spaced frequency bins from the lowest non-zero frequency to the Nyquist frequency (empty bins interpolated). The low frequencies, where a noise floor shows, get as many points as every other decade.", "", "curve"),
    "noise: within-cycle spectrum": _f("noise", "arb.", "Power spectrum of the within-cycle noise (see the band fractions), 0-7.5 MHz.", "", "curve"),
    "noise: noise vs window length": _f("noise", "ADU", "Detrended CDS noise for window lengths of 1/15, 2/15, 3/15, 1/3, 1/2, 2/3, 5/6 and 1 times the window (10 ... 150 samples for the usual windows).", "", "curve"),
    # ---------------------------------------------------------------- C. discrete noise
    "discrete: largest baseline step (t)": _f("discrete", "", "Largest difference between the 20-skip means before and after a skip of the detrended per-skip baseline, in units of its standard error.", "telegraph noise in the output transistor"),
    "discrete: largest CDS step (t)": _f("discrete", "", "Same for the detrended CDS (chance level for 490 skips: about 4).", "telegraph noise"),
    "discrete: CDS excess kurtosis": _f("discrete", "", "Excess kurtosis of the detrended CDS.", "heavy tails: glitches, bursts"),
    "discrete: CDS outlier fraction": _f("discrete", "", "Fraction of skips more than 5 robust σ from the median of the detrended CDS.", "glitches, bursts, charge arriving during readout"),
    "discrete: CDS spectrum lines": _f("discrete", "count", "Number of peaks more than 5× above the local median in the CDS spectrum.", "periodic pickup"),
    "discrete: within-cycle spectrum lines": _f("discrete", "count", "Same in the within-cycle spectrum.", "periodic pickup"),
    "discrete: ADC low-bit chi2/dof": _f("discrete", "", "χ² per degree of freedom of the histogram of (sample value mod 64) inside the pedestal windows against a flat distribution.", "ADC differential non-linearity (channel health)"),
    # ---------------------------------------------------------------- D. edges
    "edge: jitter (ns)": _f("edge", "ns", "Scatter of the reset-edge times (half-height crossing, linear interpolation) around a straight line in skip number.", "timing jitter; matters where windows sit on slopes"),
    "edge: period offset (1e-3 samples)": _f("edge", "×10⁻³ samples", "Slope of that line minus 488.7. Small differences between amplifiers come from the edge-time estimate drifting with the baseline droop, not from the clock.", "edge-shape change along the pixel"),
    "edge: rise time (ns)": _f("edge", "ns", "10-90 % rise time of the reset pulse rebuilt by equivalent-time sampling (samples placed at their exact time relative to the edge line, 0.1-sample bins; crossings interpolated).", "video-chain bandwidth"),
    "edge: recovery tau1 (us)": _f("edge", "µs", "Faster time constant of two exponentials fitted to the equivalent-time waveform after the pulse peak.", "amplifier response"),
    "edge: recovery tau2 (us)": _f("edge", "µs", "Slower time constant of that fit.", "amplifier response"),
    "edge: recovery amplitude ratio": _f("edge", "", "Amplitude of the fast exponential divided by that of the slow one.", "amplifier response"),
    "edge: ringing frequency (MHz)": _f("edge", "MHz", "Strongest frequency in the residual of that fit over the first 25 samples after the peak (zero-padded spectrum, peak refined by a parabola).", "amplifier stability/bandwidth"),
    "edge: ringing rms (ADU)": _f("edge", "ADU", "RMS of the residual of that fit over the first 25 samples after the peak.", "amplifier stability"),
    "edge: dip per volt SW (ADU/V)": _f("edge", "ADU/V", "Dip height divided by the summing-well swing SW_1_H − SW_1_L (from the .meta; NaN without it). Assumes the channel uses SW_1.", "coupling × amplification (gain-related)"),
    "edge: pulse per volt RG (ADU/V)": _f("edge", "ADU/V", "Reset-pulse height above the signal level divided by RG_1_H − RG_1_L.", "coupling × amplification (gain-related)"),
    "edge: pulse/dip ratio": _f("edge", "", "Reset-pulse height divided by dip height; the common amplification cancels.", "relative coupling"),
    "edge: cycle variation in signal window (ADU)": _f("edge", "ADU", "Median, inside the signal window, of the standard deviation across skips at each sample position (each skip's pedestal level subtracted).", "noise that enters the CDS"),
    "edge: cycle variation at dip (ADU)": _f("edge", "ADU", "Maximum of that profile around the dip.", "timing (phase walk, jitter) × edge steepness"),
    "edge: cycle variation at reset (ADU)": _f("edge", "ADU", "Maximum of that profile around the reset pulse.", "timing × edge steepness"),
    "edge: cycle variation after reset (ADU)": _f("edge", "ADU", "Median of that profile after the reset (sample 420 to the end of the cycle).", "reset level scatter seen within the cycle"),
    "edge: equivalent-time pulse": _f("edge", "ADU", "The reset pulse at 0.1-sample resolution, from 15 samples before to 60 after the edge.", "", "curve"),
    "edge: variation along the cycle": _f("edge", "ADU", "The standard deviation across skips at each of the 489 positions of the cycle.", "", "curve"),
    # ---------------------------------------------------------------- E. slow structure
    "slow: row-start recovery time (skips)": _f("slow", "skips", "Time constant of an exponential fitted to the baseline of the second row-start pixel minus that of the next pixel, skip by skip.", "electronics recovering from the vertical clocking"),
    "slow: row-start recovery amplitude (ADU)": _f("slow", "ADU", "Amplitude of that exponential.", "vertical clock coupling"),
    "slow: drift (ADU/ms)": _f("slow", "ADU/ms", "Slope of the baseline at the middle skip (250 of 500) of successive usable pixels against time in the trace.", "slow drift (thermal, supplies)"),
    "slow: H-transfer peak-to-peak (ADU)": _f("slow", "ADU", "Peak-to-peak amplitude of a horizontal transfer, mean over transfers and usable pixels.", "H-clock coupling"),
    "slow: H-transfer first-to-last change (ADU)": _f("slow", "ADU", "Peak-to-peak of the last complete transfer minus that of the first.", "change along the transfers (charge in the serial register?)"),
    "slow: PC1 variance fraction": _f("slow", "fraction", "Fraction of the skip-to-skip variation (residual cycles of the usable pixels) in the first principal component (in the example trace: the sub-sample phase walk).", "structure of the skip-to-skip variation"),
    "slow: PC2 variance fraction": _f("slow", "fraction", "Same, second component (in the example: the level shift set by each reset).", ""),
    "slow: PC3 variance fraction": _f("slow", "fraction", "Same, third component.", ""),
}


# --------------------------------------------------------------------------- #
# The result                                                                   #
# --------------------------------------------------------------------------- #

# the tier-1 and tier-2 features of 2026-10-21 (computed in trace_features_extra.py)
from trace_features_extra import EXTRA as _EXTRA, extra_features  # noqa: E402

for _name, _group, _unit, _desc, _link, _kind, _tier in _EXTRA:
    FEATURES[_name] = _f(_group, _unit, _desc, _link, _kind, "extra", _tier)


def names_by_tier(tiers=(0, 1, 2), kind: str = "scalar") -> list[str]:
    """Feature names of the given tiers and kind, in registry order."""
    return [name for name, f in FEATURES.items() if f.tier in tiers and f.kind == kind]


@dataclass
class TraceFeatures:
    """All features of one trace (see the module docstring)."""
    scalars: dict[str, float] = field(default_factory=dict)
    spread: dict[str, float] = field(default_factory=dict)
    curves: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #

def detrend(x: np.ndarray, degree: int = 4) -> np.ndarray:
    """Remove a polynomial in the skip index (degree 4 from 200 skips, lower for short pixels; never a
    running mean: it cancels block averages)."""
    if len(x) < 2:
        return x - x.mean()
    degree = min(degree, detrend_degree(len(x)))
    k = np.arange(len(x))
    return x - np.polyval(np.polyfit(k, x, degree), k)


def residual_cycles(cycles: np.ndarray, first_skip: int = FIRST_SKIP, degree: int = 4) -> np.ndarray:
    """Every cycle minus a polynomial across skips at each sample position (mean waveform and droop removed)."""
    Y = cycles[first_skip:]
    if len(Y) < 2:
        return Y - Y.mean(0)
    degree = min(degree, detrend_degree(len(Y)))
    k = np.arange(len(Y))
    X = np.vander((k - k.mean()) / k.std(), degree + 1)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ coef


def robust_spread(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) > 1 else float("nan")


def acf(x: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Autocorrelation at lags 1..max_lag (NaN where the series is too short)."""
    x = x - x.mean()
    norm = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / norm if k < len(x) and norm > 0 else np.nan
                     for k in range(1, max_lag + 1)])


ALLAN_N = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 160])


def allan_deviation(x: np.ndarray, ns=ALLAN_N) -> np.ndarray:
    out = []
    for n in ns:
        m = len(x) // n
        means = x[:m * n].reshape(m, n).mean(axis=1)
        out.append(np.sqrt(0.5 * np.mean(np.diff(means) ** 2)) if m > 2 else np.nan)
    return np.array(out)


def allan_deviation_overlapping(x: np.ndarray, ns=ALLAN_N) -> np.ndarray:
    """
    Overlapping Allan deviation: for block length n, sqrt(1/2 mean((m_{i+n} - m_i)^2)) over EVERY start i, where
    m_i is the mean of x[i:i+n].  Same expectation as the disjoint-block version, much less scatter for long
    blocks.  NaN where fewer than 3 disjoint blocks fit (as allan_deviation).
    """
    x = np.asarray(x, float)
    c = np.r_[0.0, np.cumsum(x)]
    out = []
    for n in ns:
        if len(x) // n <= 2:
            out.append(np.nan)
            continue
        means = (c[n:] - c[:-n]) / n
        d = means[n:] - means[:-n]
        out.append(np.sqrt(0.5 * np.mean(d ** 2)))
    return np.array(out)


def log_frequency_bins(f: np.ndarray, S: np.ndarray, n_bins: int = 40) -> np.ndarray:
    """
    A spectrum averaged in ``n_bins`` logarithmically spaced bins between the lowest non-zero frequency and the
    highest; bins without an FFT frequency (at the low end) are interpolated in log frequency.
    """
    f, S = np.asarray(f, float), np.asarray(S, float)
    keep = f > 0
    f, S = f[keep], S[keep]
    edges = np.geomspace(f[0], f[-1] * (1 + 1e-9), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    which = np.digitize(f, edges) - 1
    out = np.array([S[which == k].mean() if np.any(which == k) else np.nan for k in range(n_bins)])
    ok = np.isfinite(out)
    if ok.sum() >= 2 and not ok.all():
        out[~ok] = np.exp(np.interp(np.log(centers[~ok]), np.log(centers[ok]), np.log(np.maximum(out[ok], 1e-300))))
    return out


def spectrum(series_list: list[np.ndarray], rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean one-sided spectral density of equal-length series (Hann window)."""
    n = min(len(s) for s in series_list)
    window = np.hanning(n)
    S = np.mean([np.abs(np.fft.rfft(s[:n] * window)) ** 2 for s in series_list], axis=0)
    return np.fft.rfftfreq(n, 1 / rate), S * 2 / (rate * (window ** 2).sum())


def largest_step(x: np.ndarray, window: int = 20) -> float:
    """Largest |mean after − mean before| over all skips, in units of its standard error."""
    if window < 3 or len(x) < 2 * window + 1:
        return float("nan")
    c = np.r_[0, np.cumsum(x)]; c2 = np.r_[0, np.cumsum(x * x)]
    best = 0.0
    for k in range(window, len(x) - window):
        s1, s2 = c[k] - c[k - window], c[k + window] - c[k]
        q1, q2 = c2[k] - c2[k - window], c2[k + window] - c2[k]
        v1 = (q1 - s1 * s1 / window) / (window - 1); v2 = (q2 - s2 * s2 / window) / (window - 1)
        best = max(best, abs(s2 - s1) / window / np.sqrt((v1 + v2) / window))
    return float(best)


def count_lines(S: np.ndarray, factor: float = 5, half_width: int = 5) -> int:
    n = 0
    for i in range(half_width, len(S) - half_width):
        local = np.median(np.r_[S[i - half_width:i], S[i + 1:i + half_width + 1]])
        n += bool(S[i] > factor * local and S[i] == S[i - 1:i + 2].max())
    return n


def cycle_starts(trace: DecodedTrace, period: int) -> list[np.ndarray]:
    """Absolute first sample of every skip cycle, per pixel (same grouping as split_regions)."""
    starts, _ = trace.window_slices()
    breaks = np.flatnonzero(np.diff(starts) > 2 * period)
    return [starts[a:b] for a, b in zip(np.r_[0, breaks + 1], np.r_[breaks, len(starts) - 1])]


def edge_times(cycles: np.ndarray, starts: np.ndarray, lo: int, hi: int, before: slice) -> np.ndarray:
    """Absolute time (samples) of the reset edge in every skip: half-height crossing, interpolated."""
    out = []
    for k, y in enumerate(cycles):
        seg = y[lo:hi]; base = y[before].mean(); i = int(np.argmax(seg)); level = (base + seg[i]) / 2
        below = np.flatnonzero(seg[:i + 1] < level)
        if len(below) == 0 or below[-1] + 1 >= len(seg):
            out.append(np.nan); continue
        j = below[-1]
        out.append(starts[k] + lo + j + (level - seg[j]) / (seg[j + 1] - seg[j]))
    return np.array(out)


def fit_two_exponentials(t, y, tau_grid):
    """
    A1 exp(-t/τ1) + A2 exp(-t/τ2) + c with τ1 < τ2 (amplitudes by least squares).

    Coarse search on ``tau_grid``, then a fine 21 x 21 search between the neighbouring grid points
    of the best pair, so the time constants are not quantized to the grid.
    """
    def solve(t1, t2):
        X = np.column_stack([np.exp(-t / t1), np.exp(-t / t2), np.ones_like(t)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(np.sum((X @ coef - y) ** 2)), t1, t2, coef, X @ coef

    best = min((solve(t1, t2) for i, t1 in enumerate(tau_grid) for t2 in tau_grid[i + 1:]), key=lambda b: b[0])
    i1, i2 = int(np.argmin(abs(tau_grid - best[1]))), int(np.argmin(abs(tau_grid - best[2])))
    fine1 = np.geomspace(tau_grid[max(i1 - 1, 0)], tau_grid[min(i1 + 1, len(tau_grid) - 1)], 21)
    fine2 = np.geomspace(tau_grid[max(i2 - 1, 0)], tau_grid[min(i2 + 1, len(tau_grid) - 1)], 21)
    best = min((solve(t1, t2) for t1 in fine1 for t2 in fine2 if t1 < t2), key=lambda b: b[0])

    # last step: a parabola through the error at 3 points (in log tau) for each time constant
    t1, t2 = best[1], best[2]
    for which, step in ((0, fine1[1] / fine1[0]), (1, fine2[1] / fine2[0])):
        trial = [(t1 * f, t2) if which == 0 else (t1, t2 * f) for f in (1 / step, 1.0, step)]
        e = [solve(*pair)[0] for pair in trial]
        denom = e[0] - 2 * e[1] + e[2]
        if denom > 0:
            shift = float(np.clip(0.5 * (e[0] - e[2]) / denom, -1, 1)) * np.log(step)
            t1, t2 = (t1 * np.exp(shift), t2) if which == 0 else (t1, t2 * np.exp(shift))
    refined = solve(t1, t2)
    return refined if refined[0] <= best[0] * (1 + 1e-9) else best


def crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """First x where y rises through ``level`` (linear interpolation between points)."""
    i = int(np.flatnonzero(y >= level)[0])
    return float(x[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])) if i > 0 else float(x[0])


def peak_frequency(f: np.ndarray, power: np.ndarray) -> float:
    """Frequency of the highest spectral peak (excluding 0), refined by a parabola through 3 bins."""
    i = int(np.argmax(power[1:])) + 1
    if 1 <= i < len(power) - 1:
        a, b, c = np.log(power[i - 1:i + 2] + 1e-30)
        shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        return float(f[i] + shift * (f[1] - f[0]))
    return float(f[i])


# --------------------------------------------------------------------------- #
# The function                                                                 #
# --------------------------------------------------------------------------- #

def trace_features(trace: DecodedTrace | str | Path, meta: dict | None = None,
                   regions: TraceRegions | None = None) -> TraceFeatures:
    """
    Compute every feature of one trace, for any geometry and clock timing.

    ``trace``: a ``DecodedTrace`` or the path of a ``.RJ45ADCDATA<ch>`` file.  ``meta``: the
    trace's ``.meta`` dict (used for the row starts, the expected timing and the per-volt
    feedthrough); when a path is given and ``meta`` is None, the ``.meta`` next to it is read if
    it exists.  Raises ``ValueError`` for a trace with quality problems (lost data).

    All counts adapt to the trace (see ``trace_sizes``); for the usual 2x5 pixels x 500 skips with
    150-sample windows they equal the values used before.  A feature that cannot be computed for
    this trace (too few skips, pixels, transfers, ...) is NaN; if a whole group fails, its
    features are NaN and the reason is in ``info["errors"]``.
    """
    if not isinstance(trace, DecodedTrace):
        path = Path(trace)
        if meta is None and path.with_suffix(".meta").exists():
            meta = json.loads(path.with_suffix(".meta").read_text())
        trace = decode_trace(path)
    problems = trace.quality_problems()
    if problems:
        raise ValueError(f"trace has quality problems: {problems}")
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)       # empty slices etc. give NaN on purpose
        return _trace_features(trace, meta, regions)


def _trace_features(trace: DecodedTrace, meta: dict | None, regions: TraceRegions | None) -> TraceFeatures:
    """The body of :func:`trace_features` (expected numpy warnings are silenced by the caller)."""
    r = regions or split_regions(trace, meta)
    good = usable_pixels(r)
    row_starts = row_start_pixels(r)
    z = trace_sizes(r)
    n_ped, sig, P, sig_end, first = r.n_ped, r.sig_start, r.period, z["sig_end"], z["first_skip"]
    expected_period = r.geometry.cycle if r.geometry.timing_known else r.period_exact
    skip_rate = SAMPLE_RATE_HZ / r.period_exact
    result = TraceFeatures(info={"pixels_used": good, "row_start_pixels": row_starts, "n_pixels": r.n_pixels,
                                 "n_skips": r.n_skips, "sizes": z, "timing": {}, "errors": {},
                                 "region_warnings": list(r.warnings)})
    S, D, Cv = result.scalars, result.spread, result.curves
    nan = float("nan")

    def per_pixel(name, values):
        values = np.asarray(values, float)
        S[name] = float(np.nanmedian(values)) if np.isfinite(values).any() else nan
        D[name] = robust_spread(values)

    def run(group, function):
        """Run one group; on failure its features become NaN and the reason is recorded."""
        try:
            function()
        except Exception as err:                       # noqa: BLE001 - one group must not stop the trace
            result.info["errors"][group] = f"{type(err).__name__}: {err}"
        for name, f in FEATURES.items():
            if f.group == group and f.kind == "scalar":
                S.setdefault(name, nan)

    # ---- comparison tools (compare_trace.py), median over replicates ---------------------
    def comparison():
        replicates = measure(r, good)
        result.info["timing"] = replicates.timing
        for name, values in replicates.tools.items():
            if np.ndim(values[0]) == 0:
                values = np.asarray(values, float)
                S[name] = float(np.nanmedian(values)) if np.isfinite(values).any() else nan
                D[name] = robust_spread(values)
            else:
                lengths = {len(v) for v in values}
                if len(lengths) == 1:
                    Cv[name] = np.nanmedian(np.asarray(values, float), axis=0)
    try:
        comparison()
    except Exception as err:                            # noqa: BLE001
        result.info["errors"]["comparison tools"] = f"{type(err).__name__}: {err}"
    for name, f in FEATURES.items():
        if f.origin == "comparison" and f.kind == "scalar":
            S.setdefault(name, nan)

    # ---- shared quantities -------------------------------------------------------------------
    C = [np.asarray(r.cds[p], float) for p in range(r.n_pixels)]
    n = r.n_skips
    Cm = np.array([c[:n] for c in C])                    # pixels x skips (equal lengths)
    cyc = {p: r.skip[p].astype(float) for p in range(r.n_pixels)}
    det = {p: detrend(Cm[p, first:]) for p in good}
    res = {p: residual_cycles(cyc[p], first) for p in good}
    starts = {p: np.asarray(r.ped_starts[p][:len(cyc[p])]) for p in range(r.n_pixels)}

    # ---- A. CDS level ------------------------------------------------------------------------
    def level():
        common = np.median(Cm[good], axis=0)
        Cv["level: common CDS pattern"] = common
        k = z["tenth"]
        S["level: common CDS rise (ADU)"] = float(common[n - k:].mean() - common[first:first + k].mean()) \
            if n - k >= first + k else nan
        rel = {p: float((Cm[p, first:] - common[first:]).mean()) for p in range(r.n_pixels)}
        S["level: pixel level spread (ADU)"] = robust_spread([rel[p] for p in good])
        S["level: pixel level max (ADU)"] = float(max(rel[p] for p in good))
        charged = [p for p in range(r.n_pixels) if rel[p] > 1000]
        loss, skews = [], []
        for p in charged:
            d = Cm[p, first:] - common[first:]
            if len(d) >= 5:
                slope = np.polyfit(np.arange(len(d)), d, 1)[0]
                loss.append(100 * slope * len(d) / d.mean())
                skews.append(((d - d.mean()) ** 3).mean() / d.std() ** 3)
        S["level: charge-loss slope (%)"] = float(np.median(loss)) if loss else nan
        S["level: charged-pixel skewness"] = float(np.median(skews)) if skews else nan
        result.info["charged_pixels"] = charged
    result.info["charged_pixels"] = []
    run("level", level)

    # ---- B. noise character -------------------------------------------------------------------
    def noise():
        acfs = np.array([acf(det[p], 50) for p in good])
        per_pixel("noise: CDS autocorrelation at lag 1", acfs[:, 0])
        Cv["noise: CDS autocorrelation"] = np.nanmean(acfs, axis=0)
        allan = np.array([allan_deviation(det[p]) for p in good])
        Cv["noise: Allan deviation"] = np.nanmean(allan, axis=0) if np.isfinite(allan).any() else allan[0]
        i50, i10, i100 = (list(ALLAN_N).index(k) for k in (50, 10, 100))
        per_pixel("noise: Allan deviation n=50 (ADU)", allan[:, i50])
        per_pixel("noise: Allan slope n=10-100", np.log(allan[:, i100] / allan[:, i10]) / np.log(10))
        # not detrended: only the pattern every pixel shares is removed (it is not noise for an image)
        P = len(good)
        shared = Cm[good, first:].mean(0) if P > 1 else 0.0
        scale = np.sqrt(P / (P - 1)) if P > 1 else 1.0
        raw = {p: scale * (Cm[p, first:] - shared - (Cm[p, first:] - shared).mean()) for p in good}
        allan_raw = np.array([allan_deviation_overlapping(raw[p]) for p in good])
        Cv["noise: Allan deviation, common pattern removed"] = np.nanmean(allan_raw, axis=0) if np.isfinite(allan_raw).any() else allan_raw[0]
        if len(raw[good[0]]) >= 8:
            f_raw, S_raw = spectrum(list(raw.values()), skip_rate)
            Cv["noise: CDS spectrum, log frequency"] = log_frequency_bins(f_raw, S_raw)
        if len(det[good[0]]) >= 8:
            f_cds, S_cds = spectrum(list(det.values()), skip_rate)
            Cv["noise: CDS spectrum"] = S_cds
            # bands as shares of the spectrum's range, so they exist for any skip rate:
            # exactly < 1 kHz and > 10 kHz at the usual skip rate (30.69 kHz)
            nyquist = skip_rate / 2
            low = S_cds[(f_cds > 0) & (f_cds < nyquist * 1e3 / REFERENCE_NYQUIST)]
            high = S_cds[f_cds > nyquist * 10e3 / REFERENCE_NYQUIST]
            S["noise: CDS power <1 kHz / >10 kHz"] = float(low.mean() / high.mean()) if len(low) and len(high) else nan
            S["discrete: CDS spectrum lines"] = float(count_lines(S_cds[1:]))
        per_pixel("noise: reset-level scatter (ADU)", [res[p][:, :n_ped].mean(1).std() for p in good])
        segment = sig_end
        inside = [res[p][:, :segment] - res[p][:, :segment].mean(1, keepdims=True) for p in good]
        S_in = np.mean([np.mean(np.abs(np.fft.rfft(x * np.hanning(segment), axis=1)) ** 2, axis=0) for x in inside], axis=0)
        f_in = np.fft.rfftfreq(segment, 1 / SAMPLE_RATE_HZ)
        Cv["noise: within-cycle spectrum"] = S_in
        total = S_in[f_in >= 0.1e6].sum()
        for a, b, label in ((0.1e6, 0.3e6, "0.1-0.3"), (0.3e6, 1e6, "0.3-1"), (1e6, 3e6, "1-3"), (3e6, 7.6e6, "3-7.5")):
            S[f"noise: within-cycle power {label} MHz"] = float(S_in[(f_in >= a) & (f_in < b)].sum() / total) if total > 0 else nan
        S["discrete: within-cycle spectrum lines"] = float(count_lines(S_in[1:]))

        def reintegrated(p, length, offset):
            Y = cyc[p]
            c = Y[:, offset:offset + length].mean(1) - Y[:, sig + offset:sig + offset + length].mean(1)
            return detrend(c[first:]).std()
        lengths = z["window_lengths"]
        noise_len = np.array([[reintegrated(p, L, 0) for L in lengths] for p in good])
        Cv["noise: noise vs window length"] = noise_len.mean(0)
        half = z["half_window"]
        per_pixel("noise: window-length ratio 75/150", [reintegrated(p, half, 0) / reintegrated(p, n_ped, 0) for p in good])
        per_pixel("noise: late/early window ratio", [reintegrated(p, half, n_ped - half) / reintegrated(p, half, 0) for p in good])
        k = z["settle_samples"]
        for name, start in (("pedestal", 0), ("signal", sig)):
            shape = np.mean([(w - w.mean(1, keepdims=True)).mean(0)
                             for w in (cyc[p][first:, start:start + n_ped] for p in good)], axis=0)
            S[f"noise: {name}-window settling (ADU)"] = float(shape[-k:].mean() - shape[:k].mean())
        per_pixel("noise: baseline coupling (1e-3)",
                  [1e3 * np.polyfit(cyc[p][first:, :n_ped].mean(1), Cm[p, first:], 1)[0] if n - first >= 3 else nan for p in good])
        block = z["noise_block"]
        trends = []
        for p in good:
            blocks = [det[p][i:i + block].std() for i in range(0, len(det[p]) - block + 1, block)]
            trends.append(100 * np.polyfit(np.arange(len(blocks)) * block, blocks, 1)[0] if len(blocks) >= 3 else nan)
        per_pixel("noise: noise trend along skips (ADU/100 skips)", trends)
    run("noise", noise)

    # ---- C. discrete noise --------------------------------------------------------------------
    def discrete():
        w = z["step_window"]
        per_pixel("discrete: largest baseline step (t)",
                  [largest_step(detrend(cyc[p][first:, :n_ped].mean(1)), w) if w >= 3 else nan for p in good])
        per_pixel("discrete: largest CDS step (t)", [largest_step(det[p], w) if w >= 3 else nan for p in good])
        per_pixel("discrete: CDS excess kurtosis",
                  [((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3 if len(x) > 3 and x.var() > 0 else nan for x in det.values()])
        outliers = []
        for x in det.values():
            med = np.median(x); rob = 1.4826 * np.median(np.abs(x - med))
            outliers.append(np.mean(np.abs(x - med) > 5 * rob) if rob > 0 else nan)
        per_pixel("discrete: CDS outlier fraction", outliers)
        samples = np.concatenate([cyc[p][:, :n_ped].ravel() for p in good]).astype(np.int64)
        counts = np.bincount(np.mod(samples, 64), minlength=64); expected = len(samples) / 64
        S["discrete: ADC low-bit chi2/dof"] = float(((counts - expected) ** 2 / expected).sum() / 63)
    run("discrete", discrete)

    # ---- D. edges -----------------------------------------------------------------------------
    def edges():
        mean_shape = np.mean([(cyc[p][z["template_from"]:] - cyc[p][z["template_from"]:, :n_ped].mean(1, keepdims=True)).mean(0)
                              for p in good], axis=0)
        if P - sig_end < 5:
            raise ValueError("no reset pulse inside the cycle")
        peak_idx = sig_end + int(np.argmax(mean_shape[sig_end:]))
        lo, hi = max(sig_end - 7, peak_idx - 30), min(P, peak_idx + 46)
        before = slice(max(sig, lo - 10), lo + 6)
        result.info["sizes"]["pulse_peak"] = peak_idx
        jitters, periods, fits = [], [], {}
        for p in good:
            t = edge_times(cyc[p], starts[p], lo, hi, before)
            k = np.arange(len(t)); ok = np.isfinite(t) & (k >= min(5, len(t) // 10))
            if ok.sum() < 5:
                continue
            fit = np.polyfit(k[ok], t[ok], 1); fits[p] = fit
            jitters.append((t[ok] - np.polyval(fit, k[ok])).std()); periods.append(fit[0])
        per_pixel("edge: jitter (ns)", np.array(jitters) / SAMPLE_RATE_HZ * 1e9 if jitters else [nan])
        per_pixel("edge: period offset (1e-3 samples)", 1e3 * (np.array(periods) - expected_period) if periods else [nan])

        # equivalent-time pulse from all usable pixels (later skips, where the droop is slow)
        taus, vals = [], []
        for p, fit in fits.items():
            for k in range(z["ets_from"], len(cyc[p])):
                tau = starts[p][k] + np.arange(P) - np.polyval(fit, k)
                sel = (tau > -15) & (tau < 60)
                taus.append(tau[sel]); vals.append((cyc[p][k] - cyc[p][k, :n_ped].mean())[sel])
        if taus:
            taus, vals = np.concatenate(taus), np.concatenate(vals)
            bins = np.arange(-15, 60.05, 0.1); idx = np.digitize(taus, bins)
            sums = np.bincount(idx, weights=vals, minlength=len(bins) + 1)[1:len(bins)]
            counts = np.bincount(idx, minlength=len(bins) + 1)[1:len(bins)]
            ets = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
            centers = bins[:-1] + 0.05
            Cv["edge: equivalent-time pulse"] = ets
            peak = np.nanmax(ets); ok = np.isfinite(ets)
            rising = ok & (centers <= centers[np.nanargmax(ets)])
            S["edge: rise time (ns)"] = float((crossing(centers[rising], ets[rising], 0.9 * peak)
                                               - crossing(centers[rising], ets[rising], 0.1 * peak)) / SAMPLE_RATE_HZ * 1e9)
            peak_at = centers[np.nanargmax(ets)]
            sel = (centers > peak_at + 0.5) & ok
            t_after, y_after = centers[sel] - peak_at, ets[sel]
            if len(t_after) >= 20:
                _, tau1, tau2, coef, model = fit_two_exponentials(t_after, y_after, np.geomspace(0.5, 200, 40))
                S["edge: recovery tau1 (us)"] = float(tau1 / SAMPLE_RATE_HZ * 1e6)
                S["edge: recovery tau2 (us)"] = float(tau2 / SAMPLE_RATE_HZ * 1e6)
                S["edge: recovery amplitude ratio"] = float(coef[0] / coef[1]) if coef[1] else nan
                early = t_after < 25
                residual = (y_after - model)[early]
                if len(residual) >= 20:
                    spec = np.abs(np.fft.rfft(residual * np.hanning(len(residual)), n=4 * len(residual))) ** 2
                    f_res = np.fft.rfftfreq(4 * len(residual), 0.1 / SAMPLE_RATE_HZ)
                    S["edge: ringing frequency (MHz)"] = peak_frequency(f_res, spec) / 1e6
                    S["edge: ringing rms (ADU)"] = float(residual.std())

        dips, pulses = [], []
        m = min(20, max(1, n_ped // 4))
        for p in good:
            shape = (cyc[p][first:] - cyc[p][first:, :n_ped].mean(1, keepdims=True)).mean(0)
            dips.append(shape[n_ped:sig].min() if sig > n_ped else nan)
            pulses.append(shape[sig_end:].max() - shape[sig_end - m:sig_end].mean())
        dips, pulses = np.array(dips), np.array(pulses)
        if meta and all(k in meta for k in ("SW_1_H", "SW_1_L", "RG_1_H", "RG_1_L")):
            per_pixel("edge: dip per volt SW (ADU/V)", dips / (meta["SW_1_H"] - meta["SW_1_L"]))
            per_pixel("edge: pulse per volt RG (ADU/V)", pulses / (meta["RG_1_H"] - meta["RG_1_L"]))
        per_pixel("edge: pulse/dip ratio", pulses / dips)

        profiles = [(Y - Y[:, :n_ped].mean(1, keepdims=True)).std(0) for Y in (cyc[p][first:] for p in good)]
        profile = np.mean(profiles, axis=0)
        Cv["edge: variation along the cycle"] = profile
        S["edge: cycle variation in signal window (ADU)"] = float(np.median(profile[sig:sig_end]))
        S["edge: cycle variation at dip (ADU)"] = float(profile[n_ped:sig].max()) if sig > n_ped else nan
        S["edge: cycle variation at reset (ADU)"] = float(profile[max(sig_end, peak_idx - 17):min(P, peak_idx + 23)].max())
        after = profile[peak_idx + 46:]
        S["edge: cycle variation after reset (ADU)"] = float(np.median(after)) if len(after) else nan
    run("edge", edges)

    # ---- E. slow structure --------------------------------------------------------------------
    def slow():
        # row-start recovery: a row-start pixel (preferably not the first) followed by a pixel of the same row
        candidates = [p for p in row_starts if p + 1 < r.n_pixels and p + 1 not in row_starts]
        candidates = [p for p in candidates if p > 0] + [p for p in candidates if p == 0]
        if candidates and n >= 10:
            rs = candidates[0]
            diff = cyc[rs][:n, :n_ped].mean(1) - cyc[rs + 1][:n, :n_ped].mean(1)
            tau, amp, _ = fit_exponential(np.arange(len(diff), dtype=float), diff, 2, max(2000, 4 * n))
            S["slow: row-start recovery time (skips)"] = float(tau)
            S["slow: row-start recovery amplitude (ADU)"] = float(amp)
        mid = n // 2
        if len(good) >= 2:
            points = np.array([[starts[p][mid] / SAMPLE_RATE_HZ * 1e3, cyc[p][mid, :n_ped].mean()] for p in good])
            S["slow: drift (ADU/ms)"] = float(np.polyfit(points[:, 0], points[:, 1], 1)[0])
        amps = []
        for p in good:
            H = r.h[p].astype(float)
            if H.ndim == 2 and len(H):
                H = H - H[:, :min(30, max(1, H.shape[1] // 10))].mean(1, keepdims=True)
                amps.append(H.max(1) - H.min(1))
        if amps:
            S["slow: H-transfer peak-to-peak (ADU)"] = float(np.mean([a.mean() for a in amps]))
            changes = [a[-1] - a[0] for a in amps if len(a) >= 2]
            S["slow: H-transfer first-to-last change (ADU)"] = float(np.mean(changes)) if changes else nan
        R = np.vstack(list(res.values()))
        if len(R) >= 3:
            sv = np.linalg.svd(R - R.mean(0), compute_uv=False)
            frac = sv ** 2 / np.sum(sv ** 2)
            for i in range(min(3, len(frac))):
                S[f"slow: PC{i + 1} variance fraction"] = float(frac[i])
    run("slow", slow)

    # ---- F. tier-1/2 features of 2026-10-21 (trace_features_extra.py) -------------------------
    def extra():
        ctx = {"Cm": Cm, "cyc": cyc, "res": res, "det": det, "good": good, "row_starts": row_starts, "n_ped": n_ped,
               "sig": sig, "sig_end": sig_end, "first": first, "n": n, "skip_rate": skip_rate, "sample_rate": SAMPLE_RATE_HZ,
               "starts": starts, "meta": meta, "scalars": dict(S), "allan_n": ALLAN_N, "common": np.median(Cm[good], axis=0),
               "regions": r, "trace": trace, "curves": dict(Cv)}
        scalars, pixel_values, curves = extra_features(ctx)
        for name, err in scalars.pop("_errors", {}).items():
            result.info["errors"][f"extra: {name}"] = err
        S.update(scalars)
        for name, values in pixel_values.items():
            per_pixel(name, values)
        Cv.update(curves)
    try:
        extra()
    except Exception as err:                            # noqa: BLE001
        result.info["errors"]["extra"] = f"{type(err).__name__}: {err}"

    # keep the registry order and make sure every scalar exists
    result.scalars = {name: S.get(name, nan) for name in scalar_names()}
    return result


def trace_sizes(r: TraceRegions) -> dict:
    """
    The counts used by the features, derived from the trace (values for 500 skips and
    150-sample windows in brackets).
    """
    n = r.n_skips
    counts = skip_counts(n)
    n_ped = r.n_ped
    fractions = np.array([10, 20, 30, 50, 75, 100, 125, 150]) / 150
    lengths = sorted({max(1, int(round(f * n_ped))) for f in fractions})
    return {
        "first_skip": counts["noise_from"],                      # skips left out of noise features [10]
        "template_from": counts["template_from"],                # skips left out of mean shapes [5]
        "tenth": max(1, n // 10),                                # 10 % of the skips [50]
        "noise_block": 50 if n - counts["noise_from"] >= 250 else max(2, (n - counts["noise_from"]) // 5),   # [50]
        "step_window": min(20, (n - counts["noise_from"]) // 10),   # [20]
        "ets_from": n // 5,                                      # equivalent-time pulse from skip [100]
        "window_lengths": lengths,                               # [10 ... 150]
        "half_window": max(1, n_ped // 2),                       # [75]
        "settle_samples": min(10, max(1, n_ped // 10)),          # [10]
        "sig_end": min(r.sig_start + n_ped, r.period),           # [357]
    }


def scalar_names() -> list[str]:
    """Names of every scalar feature, in registry order."""
    return [name for name, f in FEATURES.items() if f.kind == "scalar"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print every feature of one trace.")
    parser.add_argument("trace", help=".RJ45ADCDATA<ch> file")
    args = parser.parse_args()
    result = trace_features(args.trace)
    for name in scalar_names():
        print(f"{name:52s} {result.scalars.get(name, float('nan')):14.6g}   {FEATURES[name].unit}")
    print(f"\ncurves: {', '.join(f'{k} ({len(v)})' for k, v in result.curves.items())}")


if __name__ == "__main__":
    main()
