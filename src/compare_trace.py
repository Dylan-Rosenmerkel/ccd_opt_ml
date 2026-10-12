"""
compare_trace.py -- can the data tell two traces apart, and which tools tell them apart best?

Given two trace files A and B, the script

    1. decodes both and splits them into regions (``trace_regions.py``): the skip cycles,
       the horizontal transfers and the vertical transfers;
    2. turns every region into REPLICATES (one pixel, or one transfer, = one replicate) and
       measures a set of "tools" on each replicate: mean waveform shapes, edge heights,
       baseline droop, first-skip behaviour, noise against number of skips averaged, noise
       power in frequency bands, and PARAMETRIC FITS (settling and droop time constants,
       damped ringing of the first skips, white-plus-floor noise model, ...);
    3. for every tool asks whether A and B differ by more than the replicate-to-replicate
       scatter, with a PERMUTATION TEST (shuffle the A/B labels of the replicates many times
       and count how often the shuffled difference is as large as the real one);
    4. runs the same test as two CONTROLS on trace A alone: even- against odd-numbered
       replicates, and the first half against the second half (row 1 against row 2).  These
       must come out "not distinguishable"; if one does not, the tools also see differences
       inside one trace and the verdict is declared inconclusive;
    5. prints which tools distinguish the traces, ranked, and an overall verdict.

How to read the numbers
-----------------------
    z   size of the difference in units of its statistical error (Welch z; for a curve the
        root-mean-square z over the curve).  Larger = more different.
    p   permutation p-value: probability of a difference this large if A and B were the same.
    A tool "distinguishes" the traces if p is below ``alpha / number_of_tools``
    (Bonferroni correction, default alpha = 0.01).

Replicates: skip region and horizontal block: one pixel each (typically 10 per trace).
Vertical block: one transfer each (typically 18).  Consecutive vertical transfers drift along
the block and are NOT interchangeable, so these tools are printed for information only and
are excluded from the verdict.

Usage
-----
    python compare_trace.py TRACE_A TRACE_B [--perms 10000] [--alpha 0.01] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from decode_trace import decode_trace
from trace_regions import TraceRegions, split_regions

SAMPLE_RATE_HZ = 15e6
_MAX_CURVE_POINTS = 500          # long curves are block-averaged down to this many points


# --------------------------------------------------------------------------- #
# 1. Tools: measurements on one replicate                                      #
# --------------------------------------------------------------------------- #

def _binned(curve: np.ndarray) -> np.ndarray:
    """Block-average a curve to at most ``_MAX_CURVE_POINTS`` points (keeps tests fast)."""
    step = int(np.ceil(len(curve) / _MAX_CURVE_POINTS))
    n = len(curve) // step * step
    return curve[:n].reshape(-1, step).mean(axis=1)


def detrend_degree(n: int) -> int:
    """Polynomial degree for detrending n skips: 4 for 200 or more skips, lower for short pixels."""
    return int(min(4, max(0, n // 50)))


def _poly_detrended(values: np.ndarray, degree: int = 4) -> np.ndarray:
    """
    ``values`` minus a low-order polynomial in the index (removes the slow drift).

    A running mean must NOT be used here: subtracting a short running mean also cancels the
    average over blocks of the same length, which would make the noise of N-skip averages
    look far too small.  A degree-4 polynomial only removes variations slower than about a
    quarter of the data, so averages over up to ~50 skips keep their noise.
    """
    if len(values) < 2:
        return values - values.mean()
    degree = min(degree, detrend_degree(len(values)))
    x = (np.arange(len(values)) - len(values) / 2) / (len(values) / 2)
    return values - np.polyval(np.polyfit(x, values, degree), x)


# --------------------------------------------------------------------------- #
# 1b. Parametric fits (numpy only)                                             #
# --------------------------------------------------------------------------- #

def fit_exponential(x: np.ndarray, y: np.ndarray, tau_min: float, tau_max: float) -> tuple[float, float, float]:
    """
    Fit ``y = offset + amp * exp(-x / tau)``; return ``(tau, amp, offset)``.

    ``amp`` and ``offset`` are linear parameters, so for every trial ``tau`` on a log grid
    they are solved exactly by least squares; the best ``tau`` is then refined on a finer
    grid.  Returns NaNs if the data are unusable.
    """
    if len(y) < 5 or not np.isfinite(y).all():
        return float("nan"), float("nan"), float("nan")

    def solve(tau):
        basis = np.column_stack([np.exp(-x / tau), np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        return float(((basis @ coef - y) ** 2).sum()), coef

    grid = np.geomspace(tau_min, tau_max, 60)
    best = grid[int(np.argmin([solve(t)[0] for t in grid]))]
    fine = np.geomspace(best / 1.25, best * 1.25, 41)
    sse = np.array([solve(t)[0] for t in fine])
    i = int(np.argmin(sse))
    if 0 < i < len(fine) - 1:                    # parabola through the minimum: removes the grid quantization
        log_tau = np.log(fine[i - 1:i + 2])
        a, b, _ = np.polyfit(log_tau, sse[i - 1:i + 2], 2)
        best = float(np.exp(-b / (2 * a))) if a > 0 else float(fine[i])
    else:
        best = float(fine[i])
    _, coef = solve(best)
    return float(best), float(coef[0]), float(coef[1])


def fit_damped_ringing(y: np.ndarray) -> tuple[float, float, float]:
    """
    Fit ``y_k = exp(-k / tau) * (a cos(2 pi f k) + b sin(2 pi f k))`` to the first skips.

    ``f`` (cycles per skip, 0 = no oscillation) and ``tau`` (skips) are searched on a grid; ``a`` and ``b`` are
    solved by least squares.  Returns ``(frequency, tau, amplitude)``.
    """
    k = np.arange(len(y), dtype=float)
    best = (np.inf, float("nan"), float("nan"), float("nan"))
    for freq in np.linspace(0.0, 0.5, 51):          # f = 0 is a plain decay
        for tau in np.geomspace(1.0, 30.0, 25):
            envelope = np.exp(-k / tau)
            basis = np.column_stack([envelope * np.cos(2 * np.pi * freq * k),
                                     envelope * np.sin(2 * np.pi * freq * k)])
            coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
            sse = float(((basis @ coef - y) ** 2).sum())
            if sse < best[0]:
                best = (sse, freq, tau, float(np.hypot(*coef)))
    return best[1], best[2], best[3]


def fit_noise_model(block_sizes: list[int], noise: list[float]) -> tuple[float, float]:
    """
    Fit ``noise(N)^2 = white^2 / N + floor^2`` to the noise of N-skip averages.

    Linear in ``white^2`` and ``floor^2`` (both kept non-negative).  Returns ``(white, floor)``
    in ADU: the single-skip white noise and the level that averaging cannot remove.
    """
    n = np.array(block_sizes, dtype=float)
    basis = np.column_stack([1 / n, np.ones_like(n)])
    coef, *_ = np.linalg.lstsq(basis, np.array(noise) ** 2, rcond=None)
    coef = np.maximum(coef, 0)
    return float(np.sqrt(coef[0])), float(np.sqrt(coef[1]))


def skip_counts(n_skips: int) -> dict[str, int]:
    """
    How many skips the skip tools leave out or use, for a pixel with ``n_skips`` skips.

    For the usual 500 skips: template from skip 5, noise from skip 10, ringing fit on the first 20
    skips, blocks of 10 and 50 skips.  For shorter pixels everything scales down, and tools that
    need more skips than there are return NaN.
    """
    return {"template_from": min(5, n_skips // 10), "noise_from": min(10, n_skips // 10),
            "ringing": min(20, n_skips // 2)}


def skip_tools(r: TraceRegions, pixel: int) -> dict[str, np.ndarray | float]:
    """All skip-region tools for one pixel (any number of skips, window length and cycle length)."""
    nan = float("nan")
    cycles = r.skip[pixel]
    n_skip = len(cycles)
    counts = skip_counts(n_skip)
    t0, n0 = counts["template_from"], counts["noise_from"]
    n_ped, sig, P = r.n_ped, r.sig_start, r.period
    sig_end = min(sig + n_ped, P)
    base = cycles[:, :n_ped].mean(axis=1)                          # baseline of every skip
    shape = cycles - base[:, None]                                  # baseline removed
    template = shape[t0:].mean(axis=0)                              # mean shape, first skips excluded
    cds = r.cds[pixel]
    later = cds[n0:]
    resid = _poly_detrended(later) if len(later) > 1 else later - later.mean()

    if n_skip > 1:
        droop_total = base[-1] - base[min(1, n_skip - 1)]
        reached = np.flatnonzero((base[1:] - base[1]) * np.sign(droop_total) >= 0.5 * abs(droop_total))
        half_skip = float(reached[0]) if len(reached) and droop_total != 0 else nan
    else:
        droop_total, half_skip = nan, nan

    margin = min(10, n_ped // 8)
    sig_slice = slice(sig + margin, sig_end - margin)
    window = shape[t0:, :n_ped] - template[:n_ped]                  # noise left in the pedestal window
    window = window - window.mean(axis=1, keepdims=True)
    power = (np.abs(np.fft.rfft(window, axis=1)) ** 2).mean(axis=0) / n_ped
    freq = np.fft.rfftfreq(n_ped, 1 / SAMPLE_RATE_HZ)

    def band(lo, hi):
        selected = power[(freq >= lo) & (freq < hi)]
        return float(np.log10(selected.sum())) if len(selected) and selected.sum() > 0 else nan

    def block_noise(n):                                             # noise of an n-skip average
        m = len(resid) // n * n
        if m // n < 3:                                              # fewer than 3 blocks: undefined
            return nan
        return float(resid[:m].reshape(-1, n).mean(axis=1).std(ddof=1))

    if len(resid) > 2:
        centred = resid - resid.mean()
        acf2 = float(np.dot(centred[:-2], centred[2:]) / np.dot(centred, centred))   # lag 2
    else:
        acf2 = nan

    # parametric fits ---------------------------------------------------------------------
    us = 1 / 15.0
    after = template[sig_end:]
    if len(after) >= 5:
        i_peak = sig_end + int(np.argmax(after))
        tail_tau, tail_amp, _ = fit_exponential((np.arange(len(template) - i_peak)) * us, template[i_peak:], 0.05, 20)
    else:
        tail_tau = tail_amp = nan
    if sig > n_ped + 5:
        i_dip = n_ped + int(np.argmin(template[n_ped:sig]))
        dip_tau, dip_amp, _ = fit_exponential((np.arange(sig - i_dip)) * us, template[i_dip:sig], 0.02, 10)
    else:
        dip_tau = dip_amp = nan
    if n_skip > 6:
        droop_tau, droop_amp, _ = fit_exponential(np.arange(1, len(base), dtype=float), base[1:],
                                                  min(5, max(1, n_skip // 100)), max(5000, 10 * n_skip))
    else:
        droop_tau = droop_amp = nan
    n_ring = counts["ringing"]
    if n_ring >= 5 and len(later):
        ring_freq, ring_tau, ring_amp = fit_damped_ringing(cds[:n_ring] - np.median(later))
    else:
        ring_freq = ring_tau = ring_amp = nan
    sizes = [n for n in (1, 2, 5, 10, 25, 50) if len(resid) // n >= 3]
    if len(sizes) >= 2:
        noise_white, noise_floor = fit_noise_model(sizes, [block_noise(n) for n in sizes])
    else:
        noise_white = noise_floor = nan

    tail_n = min(30, max(1, (P - sig_end) // 4))
    return {
        "fit: settling time after pulse (us)": tail_tau,
        "fit: settling amplitude after pulse": tail_amp,
        "fit: dip recovery time (us)": dip_tau,
        "fit: dip recovery amplitude": dip_amp,
        "fit: baseline droop time constant (skips)": droop_tau,
        "fit: baseline droop amplitude": droop_amp,
        "fit: first-skip ringing frequency (cyc/skip)": ring_freq,
        "fit: first-skip ringing decay (skips)": ring_tau,
        "fit: first-skip ringing amplitude": ring_amp,
        "fit: white noise per skip (ADU)": noise_white,
        "fit: correlated noise floor (ADU)": noise_floor,
        "skip: mean shape": _binned(template),
        "skip: baseline droop curve": _binned(base[1:] - base[1]) if n_skip > 1 else np.full(1, nan),
        "skip: dip height": float(template[n_ped:sig].min()) if sig > n_ped else nan,
        "skip: signal-window level": float(template[sig_slice].mean()) if sig_slice.stop > sig_slice.start else nan,
        "skip: pulse peak height": float(after.max()) if len(after) else nan,
        "skip: end-of-cycle tail level": float(template[-tail_n:].mean()),
        "skip: baseline at skip 1": float(base[min(1, n_skip - 1)]),
        "skip: droop total": float(droop_total),
        "skip: droop half-time (skips)": half_skip,
        "skip: CDS mean (skips>=10)": float(later.mean()) if len(later) else nan,
        "skip: CDS rms (detrended)": float(resid.std(ddof=1)) if len(resid) > 1 else nan,
        "skip: first-skip CDS excess": float(cds[0] - np.median(later)) if len(later) else nan,
        "skip: skips 1-4 CDS ringing": float(np.sqrt(np.mean((cds[1:5] - np.median(later)) ** 2))) if len(cds) > 1 and len(later) else nan,
        "skip: CDS autocorrelation at lag 2": acf2,
        "skip: noise of 10-skip average": block_noise(10),
        "skip: noise of 50-skip average": block_noise(50),
        "skip: noise power 0.1-1 MHz": band(0.1e6, 1e6),
        "skip: noise power 1-3 MHz": band(1e6, 3e6),
        "skip: noise power 3-7.5 MHz": band(3e6, 7.51e6),
    }


def transfer_tools(label: str, transfers: np.ndarray) -> dict[str, np.ndarray | float]:
    """
    Tools for repeated transfers of one block (rows = transfers), baseline removed.

    For the horizontal block ``transfers`` holds all transfers of one pixel (their mean is the
    replicate).  For the vertical block it holds ONE transfer (a transfer is the replicate),
    so the scatter between transfers is not available there.
    """
    length = transfers.shape[1]
    base_n, end_n = min(30, max(1, length // 10)), min(50, max(1, length // 10))
    shape = transfers - transfers[:, :base_n].mean(axis=1, keepdims=True)
    template = shape.mean(axis=0)
    tools = {
        f"{label}: mean shape": _binned(template),
        f"{label}: minimum": float(template.min()),
        f"{label}: maximum": float(template.max()),
        f"{label}: end level": float(template[-end_n:].mean()),
    }
    if len(transfers) > 1:
        tools[f"{label}: scatter between transfers"] = float(np.median(shape.std(axis=0, ddof=1)))
    return tools


@dataclass
class Replicates:
    """Per tool: array of shape (n_replicates,) or (n_replicates, n_points)."""
    tools: dict[str, np.ndarray]
    timing: dict[str, float]


def row_start_pixels(r: TraceRegions) -> list[int]:
    """Pixels read first in a row (pixel 0 and every pixel after vertical transfers); see trace_regions."""
    return list(r.row_starts)


def usable_pixels(r: TraceRegions) -> list[int]:
    """
    Pixels suitable for comparing waveform shapes (if every pixel starts a row, as with one
    column, the row-start pixels are kept rather than leaving none).

    Excludes the row-start pixels (the electronics are still settling after the vertical
    transfers: their baseline differs strongly) and pixels holding a large charge (CDS mean
    far from the median, judged with the median absolute deviation, which the outliers
    themselves cannot distort).
    """
    candidates = [p for p in range(r.n_pixels) if p not in row_start_pixels(r) and len(r.cds[p])]
    if not candidates:                                  # e.g. one column: every pixel starts a row
        candidates = [p for p in range(r.n_pixels) if len(r.cds[p])]
    n0 = skip_counts(r.n_skips)["noise_from"]
    cds_mean = np.array([np.mean(r.cds[p][n0:]) for p in candidates])
    median = np.median(cds_mean)
    mad = 1.4826 * np.median(np.abs(cds_mean - median))
    return [p for p, x in zip(candidates, cds_mean) if abs(x - median) < 8 * max(mad, 5.0)]


def measure(r: TraceRegions, pixels: list[int] | None = None) -> Replicates:
    """Measure every tool on every replicate of one trace (``pixels``: subset to use, default all)."""
    collected: dict[str, list] = {}

    def add(tools):
        for name, value in tools.items():
            collected.setdefault(name, []).append(value)

    for pixel in (range(r.n_pixels) if pixels is None else pixels):
        add(skip_tools(r, pixel))
        if r.h[pixel].size and len(r.h[pixel]):                # pixels without a horizontal transfer: no tools
            add(transfer_tools("horizontal", r.h[pixel]))
    if r.v is not None:                                 # vertical: one replicate per transfer
        for transfer in range(len(r.v)):
            add(transfer_tools("vertical", r.v[transfer:transfer + 1]))
    tools = {name: np.array(values) for name, values in collected.items()}
    timing = {"skip cycle (samples)": r.period, "horizontal transfer (samples)": r.h_period,
              "vertical transfer (samples)": r.v_period if r.v_period else float("nan")}
    return Replicates(tools, timing)


# --------------------------------------------------------------------------- #
# 2. The permutation test                                                      #
# --------------------------------------------------------------------------- #

def _welch_statistic(x: np.ndarray, in_a: np.ndarray) -> np.ndarray:
    """
    Welch z of the A/B difference for many labelings at once.

    ``x``: (n, L) replicates; ``in_a``: (n_labelings, n) 0/1 matrix, row = which replicates are in A.
    Returns one number per labeling: |z| for a single point, RMS z over L points for a curve.
    """
    in_b = 1 - in_a
    na, nb = in_a.sum(axis=1, keepdims=True), in_b.sum(axis=1, keepdims=True)
    mean_a, mean_b = in_a @ x / na, in_b @ x / nb
    var_a = np.maximum(in_a @ x ** 2 / na - mean_a ** 2, 0) * na / (na - 1)
    var_b = np.maximum(in_b @ x ** 2 / nb - mean_b ** 2, 0) * nb / (nb - 1)
    err = np.sqrt(var_a / na + var_b / nb)
    scale = 1e-9 + 1e-6 * np.abs(x).mean()
    z = (mean_a - mean_b) / (err + scale)
    return np.sqrt((z ** 2).mean(axis=1))


MAX_EXACT = 50000      # enumerate ALL relabelings when there are at most this many (exact p-values)


def n_labelings(n_a: int, n_b: int) -> int:
    """Number of distinct ways to split ``n_a + n_b`` replicates into groups of ``n_a`` and ``n_b``."""
    return math.comb(n_a + n_b, n_a)


def smallest_possible_p(n_a: int, n_b: int, n_perm: int) -> float:
    """
    Smallest p-value the permutation test can report for groups of ``n_a`` and ``n_b`` replicates.

    With few replicates every relabeling is enumerated: the real labeling (and, for equal groups, its
    mirror image, which has the same |z|) always counts, so p cannot go below 1 or 2 over the number of
    labelings.  With many replicates random relabelings are used and the floor is 1/(n_perm + 1).
    """
    total = n_labelings(n_a, n_b)
    if total <= max(n_perm, MAX_EXACT):
        return (2 if n_a == n_b else 1) / total
    return 1 / (n_perm + 1)


def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator):
    """Return ``(z_observed, p_value)`` for replicate sets ``a`` and ``b`` (axis 0 = replicates)."""
    x = np.concatenate([a, b]).astype(float)
    x = x.reshape(len(x), -1)
    n, na = len(x), len(a)
    observed = np.zeros((1, n))
    observed[0, :na] = 1
    z_obs = _welch_statistic(x, observed)[0]

    n_combos = n_labelings(na, n - na)
    if n_combos <= max(n_perm, MAX_EXACT):               # few replicates: enumerate all labelings exactly
        labelings = np.zeros((n_combos, n))
        for i, chosen in enumerate(itertools.combinations(range(n), na)):
            labelings[i, list(chosen)] = 1
    else:
        labelings = np.zeros((n_perm, n))
        for i in range(n_perm):
            labelings[i, rng.permutation(n)[:na]] = 1
    z_perm = np.concatenate([_welch_statistic(x, labelings[i:i + 500]) for i in range(0, len(labelings), 500)])
    # relative tolerance: the observed labeling is recomputed inside the batch, and rounding must not
    # make it fall just below z_obs (that gave p = 0 for tools with almost no scatter)
    at_least = z_perm >= z_obs * (1 - 1e-9) - 1e-12
    if n_combos > max(n_perm, MAX_EXACT):
        p = (1 + np.sum(at_least)) / (1 + len(z_perm))
    else:
        p = max(np.sum(at_least), 1) / len(z_perm)             # the observed labeling is one of them
    return float(z_obs), float(p)


def run_tests(tools_a: dict, tools_b: dict, n_perm: int, seed: int = 0):
    """Test every tool that both traces provide.  Returns a list of dicts."""
    rng = np.random.default_rng(seed)
    results = []
    for name in tools_a:
        if name not in tools_b:
            continue
        a, b = tools_a[name], tools_b[name]
        if a.ndim == 2:                                  # curves: compare on the common length
            length = min(a.shape[1], b.shape[1])
            a, b = a[:, :length], b[:, :length]
        keep_a, keep_b = np.isfinite(a.reshape(len(a), -1)).all(axis=1), np.isfinite(b.reshape(len(b), -1)).all(axis=1)
        a, b = a[keep_a], b[keep_b]
        if len(a) < 3 or len(b) < 3:
            continue
        z, p = permutation_test(a, b, n_perm, rng)
        results.append({"tool": name, "region": name.split(":")[0], "n_a": len(a), "n_b": len(b), "z": z, "p": p})
    return results


def split_even_odd(tools: dict) -> tuple[dict, dict]:
    """Control: even-numbered against odd-numbered replicates of ONE trace."""
    return ({k: v[0::2] for k, v in tools.items()}, {k: v[1::2] for k, v in tools.items()})


def split_halves(tools: dict) -> tuple[dict, dict]:
    """Control: first half against second half of the replicates (e.g. row 1 against row 2)."""
    return ({k: v[:len(v) // 2] for k, v in tools.items()}, {k: v[len(v) // 2:] for k, v in tools.items()})


# --------------------------------------------------------------------------- #
# 3. Report                                                                    #
# --------------------------------------------------------------------------- #

def in_verdict(result: dict) -> bool:
    """Vertical-block tools are informational: consecutive transfers are not exchangeable repeats
    (the control showed the first and second half of the block differ), so they cannot be
    shuffled and are kept out of the verdict and the multiple-test correction."""
    return result["region"] != "vertical"


def format_p(p: float, floor: float) -> str:
    """p-value as text; values at the floor are marked, since they only say 'as small as the test can go'."""
    return f"{p:.1e}*" if p <= 1.01 * floor else f"{p:.3g}"


def report(name_a, name_b, results, controls, timing_a, timing_b, alpha, n_perm) -> bool:
    """Print the full report; return True if the traces are distinguishable."""
    used = [r for r in results if in_verdict(r)]
    threshold = alpha / len(used)
    print(f"\nA = {name_a}\nB = {name_b}")
    print(f"{len(used)} tools used for the verdict (+{len(results) - len(used)} informational); "
          f"a tool distinguishes the traces if p < {threshold:.2g} (Bonferroni, alpha = {alpha})\n")
    floor = smallest_possible_p(used[0]["n_a"], used[0]["n_b"], n_perm)
    print(f"smallest p this test can report: {floor:.2g} "
          f"({n_labelings(used[0]['n_a'], used[0]['n_b']):,} distinct labelings of {used[0]['n_a']} + {used[0]['n_b']} replicates)\n")
    if floor >= threshold:
        print(f"WARNING: the smallest possible p ({floor:.2g}) is not below the threshold: no tool can be flagged. "
              "Use more replicates (or --perms larger when replicates are many).\n")

    print("TIMING (exact, no statistics)")
    for key in timing_a:
        same = abs(timing_a[key] - timing_b[key]) < 0.5 or np.isnan(timing_a[key])
        print(f"  {key:34s} A={timing_a[key]:9.2f}  B={timing_b[key]:9.2f}  {'same' if same else 'DIFFERENT'}")

    print("\nRESULTS (ranked)")
    print(f"  {'tool':46s} {'z':>9s} {'p':>9s}  verdict   (* = at the smallest p the test can report)")
    ranked = sorted(results, key=lambda r: (r["p"], -r["z"]))
    for res in ranked:
        if not in_verdict(res):
            verdict = "(information only)"
        else:
            verdict = "DISTINGUISHES" if res["p"] < threshold else ("marginal" if res["p"] < 0.05 else "-")
        print(f"  {res['tool']:46s} {res['z']:9.2f} {format_p(res['p'], floor):>9s}  {verdict}")

    winners = [r for r in ranked if in_verdict(r) and r["p"] < threshold]
    print("\nSUMMARY BY REGION")
    for region in ("skip", "fit", "horizontal"):
        rows = [r for r in used if r["region"] == region]
        hits = [r for r in rows if r["p"] < threshold]
        print(f"  {region:11s}: {len(hits)} of {len(rows)} tools distinguish the traces")

    controls_ok = True
    for label, control in controls.items():
        control = [c for c in control if in_verdict(c)]
        c_below = [c for c in control if c["p"] < 0.05]
        c_bonf = [c for c in control if c["p"] < alpha / len(control)]
        ok = len(c_bonf) == 0
        controls_ok &= ok
        print(f"\nCONTROL, trace A split {label} (expect 'same')")
        print(f"  {len(c_below)} of {len(control)} tools below p=0.05 (about {0.05 * len(control):.1f} expected by chance); "
              f"{len(c_bonf)} below the corrected threshold; smallest p = {min(c['p'] for c in control):.3g}")
        print(f"  -> {'PASSED' if ok else 'FAILED: ' + ', '.join(c['tool'] for c in c_bonf)}")

    print("\nVERDICT")
    if not controls_ok:
        print("  Inconclusive: a control failed, so the tools also see differences inside one trace")
        return False
    if winners:
        print("  The traces ARE distinguishable. Best tools: " + ", ".join(r["tool"] for r in winners[:5]))
        return True
    print("  The traces are NOT distinguishable at this sensitivity")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace_a")
    parser.add_argument("trace_b")
    parser.add_argument("--perms", type=int, default=10000, help="random relabelings per test")
    parser.add_argument("--alpha", type=float, default=0.01, help="overall false-alarm probability")
    parser.add_argument("--csv", help="write the ranked results to this CSV file")
    parser.add_argument("--clean-pixels", action="store_true",
                        help="exclude row-start pixels and charge-contaminated pixels from the tools")
    parser.add_argument("--allow-damaged", action="store_true",
                        help="compare even if a trace fails the quality check (not recommended)")
    args = parser.parse_args()

    regions = []
    for path in (args.trace_a, args.trace_b):
        decoded = decode_trace(path)
        problems = decoded.quality_problems()
        if problems and not args.allow_damaged:
            raise SystemExit(f"{path} looks damaged: {'; '.join(problems)}\n"
                             "(use --allow-damaged to compare anyway)")
        meta_path = Path(path).with_suffix(".meta")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        regions.append(split_regions(decoded, meta))          # .meta: row starts and timing cross-check
    ra, rb = regions
    pick = (lambda r: usable_pixels(r)) if args.clean_pixels else (lambda r: None)
    meas_a, meas_b = measure(ra, pick(ra)), measure(rb, pick(rb))

    results = run_tests(meas_a.tools, meas_b.tools, args.perms)
    controls = {"into even/odd replicates": run_tests(*split_even_odd(meas_a.tools), n_perm=args.perms, seed=1),
                "into first/second half (row 1 / row 2)": run_tests(*split_halves(meas_a.tools), n_perm=args.perms, seed=2)}
    distinguishable = report(args.trace_a.split("/")[-1], args.trace_b.split("/")[-1], results, controls,
                             meas_a.timing, meas_b.timing, args.alpha, args.perms)

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["tool", "region", "n_a", "n_b", "z", "p"])
            writer.writeheader()
            writer.writerows(sorted(results, key=lambda r: r["p"]))
        print(f"\nwrote {args.csv}")
    raise SystemExit(0 if distinguishable else 1)


if __name__ == "__main__":
    main()
