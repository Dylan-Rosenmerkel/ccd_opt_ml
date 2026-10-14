"""
plot_feature_figures.py -- the figures of the trace-features notes, drawn at document size.

Every figure is made for a page 6.9 in wide (full width) or 3.4 in (half width), with 8-9 pt text,
so that it stays readable when included at its natural width in LaTeX.  All figures come from the
real 2026-10-08 traces of module 101 (the example trace: channel 0, first repeat), computed with
the same helpers as trace_features.py.

    python plot_feature_figures.py OUT_DIR            (from src/)

Writes one PNG per figure (names as used in docs/overleaf_trace_features*/).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_trace import fit_exponential, row_start_pixels, usable_pixels
from decode_trace import decode_trace
from trace_features import (ALLAN_N, FEATURES, acf, allan_deviation, count_lines, detrend, edge_times,
                            fit_two_exponentials, largest_step, residual_cycles, scalar_names, spectrum,
                            trace_features, trace_sizes)
from trace_regions import split_regions

PROJECT = Path(__file__).resolve().parent.parent
SESSION = PROJECT / "data" / "2026-10-08-PA03_PQ04_traces"
EXAMPLE = SESSION / "trace_2x5x500_20x20_ch0_101_20261008_185447_12.RJ45ADCDATA0"
FULL, HALF = 6.9, 3.4
RATE = 15e6
GROUP_COLORS = {"fit": "#e76f51", "skip": "#264653", "horizontal": "#2a9d8f", "vertical": "#8ab17d", "level": "#e9c46a",
                "noise": "#6d597a", "discrete": "#b56576", "edge": "#355070", "slow": "#a0a0a0"}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})


class Example:
    """The example trace and the quantities shared by the figures (same as in trace_features)."""

    def __init__(self, path: Path):
        self.meta = json.loads(path.with_suffix(".meta").read_text())
        self.trace = decode_trace(path)
        self.r = r = split_regions(self.trace, self.meta)
        self.good = usable_pixels(r)
        self.row_starts = row_start_pixels(r)
        self.z = z = trace_sizes(r)
        self.n_ped, self.sig, self.P, self.sig_end = r.n_ped, r.sig_start, r.period, z["sig_end"]
        self.first, self.n = z["first_skip"], r.n_skips
        self.Cm = np.array([np.asarray(r.cds[p], float)[:self.n] for p in range(r.n_pixels)])
        self.cyc = {p: r.skip[p].astype(float) for p in range(r.n_pixels)}
        self.det = {p: detrend(self.Cm[p, self.first:]) for p in self.good}
        self.res = {p: residual_cycles(self.cyc[p], self.first) for p in self.good}
        self.starts = {p: np.asarray(r.ped_starts[p][:len(self.cyc[p])]) for p in range(r.n_pixels)}
        self.features = trace_features(self.trace, meta=self.meta, regions=r)
        self.skip_rate = RATE / r.period_exact

    def shade_windows(self, ax, scale=1.0):
        ax.axvspan(0, self.n_ped * scale, color="#2a9d8f", alpha=0.12, lw=0)
        ax.axvspan(self.sig * scale, self.sig_end * scale, color="#e76f51", alpha=0.12, lw=0)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- A. CDS level

def level_figures(e: Example, out: Path) -> None:
    common = np.median(e.Cm[e.good], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6))
    k = np.arange(e.n)
    for p in range(e.r.n_pixels):
        m = len(e.Cm[p]) // 10 * 10
        blocks = e.Cm[p, :m].reshape(-1, 10).mean(1)
        axes[0].plot(np.arange(len(blocks)) * 10 + 5, blocks, ":" if p in e.row_starts else "-", lw=0.9, label=f"pixel {p}")
    axes[0].set(xlabel="skip", ylabel="CDS, 10-skip mean (ADU)", title="CDS of every pixel along its skips")
    axes[0].legend(ncol=2, fontsize=5.5, loc="lower right")
    axes[1].plot(k, common, lw=0.8, color="k")
    axes[1].set(xlabel="skip", ylabel="ADU", title="common pattern (median of the usable pixels)")
    axes[1].set_ylim(np.percentile(common[e.first:], 0.5) - 10, np.percentile(common[e.first:], 99.5) + 10)
    fig.tight_layout()
    save(fig, out, "level_common_pattern.png")

    rel, err = [], []
    for p in range(e.r.n_pixels):
        d = e.Cm[p, e.first:] - common[e.first:]
        rel.append(d.mean())
        m = len(d) // 10 * 10
        err.append(d[:m].reshape(-1, 10).mean(1).std(ddof=1) / np.sqrt(m // 10))
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.errorbar(range(e.r.n_pixels), rel, err, fmt="o", ms=3, capsize=2)
    for p in e.row_starts:
        ax.annotate("row start", (p, rel[p]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="pixel", ylabel="mean CDS − common (ADU)", title="relative level of each pixel")
    fig.tight_layout()
    save(fig, out, "level_pixel_levels.png")

    # a charged pixel from the other channel-0 repeats
    for path in sorted(SESSION.glob("trace_*_ch0_101_*.RJ45ADCDATA0")):
        try:
            other = Example(path)
        except ValueError:
            continue
        com = np.median(other.Cm[other.good], axis=0)
        charged = [p for p in range(other.r.n_pixels) if (other.Cm[p, other.first:] - com[other.first:]).mean() > 1000]
        if charged:
            p = charged[0]
            d = other.Cm[p, other.first:] - com[other.first:]
            kk = np.arange(len(d))
            fit = np.polyfit(kk, d, 1)
            fig, ax = plt.subplots(figsize=(HALF, 2.4))
            ax.plot(kk + other.first, d, lw=0.4, color="0.5")
            ax.plot(kk + other.first, np.polyval(fit, kk), color="#e76f51",
                    label=f"slope: {100 * fit[0] * len(d) / d.mean():+.1f} % over the pixel")
            ax.set(xlabel="skip", ylabel="CDS − common (ADU)", title=f"charged pixel ({path.name[-14:-13]}{path.stem[-2:]}, pixel {p})")
            ax.legend()
            fig.tight_layout()
            save(fig, out, "level_charge_loss.png")
            break


# --------------------------------------------------------------------------- B. noise character

def noise_figures(e: Example, out: Path) -> None:
    acfs = np.array([acf(e.det[p], 50) for p in e.good])
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    lags = np.arange(1, 51)
    ax.bar(lags, np.nanmean(acfs, 0), width=0.8, color="#6d597a")
    band = 2 / np.sqrt(len(next(iter(e.det.values()))))
    ax.axhspan(-band, band, color="0.85", zorder=0)
    ax.set(xlabel="lag (skips)", ylabel="autocorrelation", title="autocorrelation of the detrended CDS")
    fig.tight_layout()
    save(fig, out, "noise_autocorrelation.png")

    allan = np.nanmean([allan_deviation(e.det[p]) for p in e.good], axis=0)
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.loglog(ALLAN_N, allan, "o-", ms=3, label="measured")
    ax.loglog(ALLAN_N, allan[0] / np.sqrt(ALLAN_N), "--", color="0.5", label="white noise from n = 1")
    ax.set(xlabel="skips averaged n", ylabel="Allan deviation (ADU)", title="Allan deviation")
    ax.legend()
    fig.tight_layout()
    save(fig, out, "noise_allan.png")

    f, S = spectrum(list(e.det.values()), e.skip_rate)
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    ax.semilogy(f[1:] / 1e3, S[1:], lw=0.7)
    ax.set(xlabel="frequency (kHz)", ylabel="ADU²/Hz", title="spectrum of the detrended CDS")
    fig.tight_layout()
    save(fig, out, "noise_cds_spectrum.png")

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    reset = [e.res[p][:, :e.n_ped].mean(1).std() for p in e.good]
    cds_rms = [e.det[p].std(ddof=1) for p in e.good]
    axes[0].plot(e.good, reset, "o", label="reset-level scatter")
    ax2 = axes[0].twinx()
    ax2.plot(e.good, cds_rms, "s", color="#e76f51", label="CDS noise per skip")
    axes[0].set(xlabel="pixel", ylabel="reset-level scatter (ADU)", title="per usable pixel")
    ax2.set_ylabel("CDS noise per skip (ADU)", color="#e76f51")
    seg = e.sig_end
    inside = [e.res[p][:, :seg] - e.res[p][:, :seg].mean(1, keepdims=True) for p in e.good]
    S_in = np.mean([np.mean(np.abs(np.fft.rfft(x * np.hanning(seg), axis=1)) ** 2, axis=0) for x in inside], axis=0)
    f_in = np.fft.rfftfreq(seg, 1 / RATE)
    axes[1].loglog(f_in[1:] / 1e6, S_in[1:], lw=0.8)
    for edge in (0.1, 0.3, 1, 3):
        axes[1].axvline(edge, color="0.6", lw=0.6, ls=":")
    axes[1].set(xlabel="frequency (MHz)", ylabel="power (arb.)", title="spectrum inside a cycle (band edges dotted)")
    fig.tight_layout()
    save(fig, out, "noise_reset_and_bands.png")

    def reintegrated(p, length, offset):
        Y = e.cyc[p]
        c = Y[:, offset:offset + length].mean(1) - Y[:, e.sig + offset:e.sig + offset + length].mean(1)
        return detrend(c[e.first:]).std()
    L = np.array(e.z["window_lengths"])
    start = np.array([[reintegrated(p, l, 0) for l in L] for p in e.good]).mean(0)
    end = np.array([[reintegrated(p, l, e.n_ped - l) for l in L] for p in e.good]).mean(0)
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    ax.loglog(L, start, "o-", ms=3, label="windows starting at the flag")
    ax.loglog(L, end, "s-", ms=3, label="windows ending at the flag")
    ax.loglog(L, start[-1] * np.sqrt(L[-1] / L), "--", color="0.5", label=r"white: $\propto 1/\sqrt{L}$")
    ax.set(xlabel="window length L (samples)", ylabel="CDS noise (ADU)", title="noise against window length")
    ax.legend()
    fig.tight_layout()
    save(fig, out, "noise_window_length.png")

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3))
    for ax, (name, s0) in zip(axes, (("pedestal", 0), ("signal", e.sig))):
        shape = np.mean([(w - w.mean(1, keepdims=True)).mean(0) for w in (e.cyc[p][e.first:, s0:s0 + e.n_ped] for p in e.good)], axis=0)
        ax.plot(shape, lw=0.9)
        ax.set(xlabel="sample in the window", ylabel="ADU (window mean removed)", title=f"{name} window: mean waveform")
    fig.tight_layout()
    save(fig, out, "noise_window_settling.png")

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    for p in e.good:
        b = e.cyc[p][e.first:, :e.n_ped].mean(1)
        axes[0].plot(b - b.mean(), e.Cm[p, e.first:] - e.Cm[p, e.first:].mean(), ".", ms=1, alpha=0.4)
    axes[0].set(xlabel="baseline of the skip − mean (ADU)", ylabel="CDS − mean (ADU)", title="CDS against the skip's baseline")
    block = e.z["noise_block"]
    for p in e.good:
        blocks = [e.det[p][i:i + block].std() for i in range(0, len(e.det[p]) - block + 1, block)]
        axes[1].plot(np.arange(len(blocks)) * block + block / 2 + e.first, blocks, "o-", ms=2, lw=0.6)
    axes[1].set(xlabel="skip", ylabel=f"CDS noise in {block}-skip blocks (ADU)", title="noise along the pixel")
    fig.tight_layout()
    save(fig, out, "noise_baseline_coupling.png")


# --------------------------------------------------------------------------- C. discrete noise

def discrete_figures(e: Example, out: Path) -> None:
    p = e.good[0]
    w = e.z["step_window"]
    base = detrend(e.cyc[p][e.first:, :e.n_ped].mean(1))
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 3.6), gridspec_kw={"width_ratios": [3, 1]})
    for row, (label, x) in enumerate((("baseline", base), ("CDS", e.det[p]))):
        k = np.arange(len(x)) + e.first
        axes[row, 0].plot(k, x, lw=0.5)
        steps = [abs(x[i:i + w].mean() - x[i - w:i].mean()) for i in range(w, len(x) - w)]
        i = int(np.argmax(steps)) + w
        axes[row, 0].axvline(i + e.first, color="#e76f51", ls="--", lw=0.8)
        axes[row, 0].set(ylabel=f"{label} (ADU)", title=f"detrended {label}, pixel {p}: largest step t = {largest_step(x, w):.1f}")
        axes[row, 1].hist(x, 40, orientation="horizontal", density=True, color="0.6")
        yy = np.linspace(x.min(), x.max(), 100)
        axes[row, 1].plot(np.exp(-0.5 * (yy / x.std()) ** 2) / (x.std() * np.sqrt(2 * np.pi)), yy, color="#e76f51")
        axes[row, 1].set(xlabel="density")
    axes[1, 0].set_xlabel("skip")
    fig.tight_layout()
    save(fig, out, "discrete_steps.png")

    seg = e.sig_end
    inside = [e.res[q][:, :seg] - e.res[q][:, :seg].mean(1, keepdims=True) for q in e.good]
    S_in = np.mean([np.mean(np.abs(np.fft.rfft(x * np.hanning(seg), axis=1)) ** 2, axis=0) for x in inside], axis=0)
    f_in = np.fft.rfftfreq(seg, 1 / RATE)
    f, S = spectrum(list(e.det.values()), e.skip_rate)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    for ax, (x, y, unit, title) in zip(axes, ((f_in / 1e6, S_in, "MHz", "within-cycle spectrum"), (f / 1e3, S, "kHz", "CDS spectrum"))):
        local = np.array([np.median(np.r_[y[max(i - 5, 0):i], y[i + 1:i + 6]]) for i in range(len(y))])
        ax.semilogy(x[1:], y[1:], lw=0.7, label="spectrum")
        ax.semilogy(x[1:], 5 * local[1:], lw=0.7, ls="--", color="#e76f51", label="5 × local median (line threshold)")
        ax.set(xlabel=unit, title=f"{title}: {count_lines(y[1:])} lines")
    axes[0].legend()
    fig.tight_layout()
    save(fig, out, "discrete_lines.png")

    samples = np.concatenate([e.cyc[q][:, :e.n_ped].ravel() for q in e.good]).astype(np.int64)
    counts = np.bincount(np.mod(samples, 64), minlength=64)
    expected = len(samples) / 64
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    ax.bar(range(64), counts / expected - 1, color="#b56576")
    chi2 = ((counts - expected) ** 2 / expected).sum()
    ax.set(xlabel="ADC sample mod 64", ylabel="relative excess", title=f"ADC low bits: $\\chi^2$/dof = {chi2 / 63:.2f}")
    fig.tight_layout()
    save(fig, out, "discrete_adc_codes.png")


# --------------------------------------------------------------------------- D. edges

def edge_figures(e: Example, out: Path) -> None:
    n_ped, sig_end, P = e.n_ped, e.sig_end, e.P
    mean_shape = np.mean([(e.cyc[p][e.z["template_from"]:] - e.cyc[p][e.z["template_from"]:, :n_ped].mean(1, keepdims=True)).mean(0)
                          for p in e.good], axis=0)
    peak_idx = sig_end + int(np.argmax(mean_shape[sig_end:]))
    lo, hi = max(sig_end - 7, peak_idx - 30), min(P, peak_idx + 46)
    before = slice(max(e.sig, lo - 10), lo + 6)
    p = e.good[0]
    t = edge_times(e.cyc[p], e.starts[p], lo, hi, before)
    k = np.arange(len(t))
    ok = np.isfinite(t) & (k >= 5)
    fit = np.polyfit(k[ok], t[ok], 1)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    axes[0].plot(k, t - e.starts[p], ".", ms=1.5)
    axes[0].set(xlabel="skip", ylabel="edge position in the cycle (samples)", title="reset edge inside its cycle")
    axes[1].plot(k[ok], (t[ok] - np.polyval(fit, k[ok])) / RATE * 1e9, ".", ms=1.5)
    axes[1].set(xlabel="skip", ylabel="edge time − line (ns)",
                title=f"jitter: {(t[ok] - np.polyval(fit, k[ok])).std() / RATE * 1e9:.2f} ns rms")
    fig.tight_layout()
    save(fig, out, "edge_jitter.png")

    fits = {}
    for q in e.good:
        tq = edge_times(e.cyc[q], e.starts[q], lo, hi, before)
        kq = np.arange(len(tq)); okq = np.isfinite(tq) & (kq >= 5)
        if okq.sum() >= 5:
            fits[q] = np.polyfit(kq[okq], tq[okq], 1)
    taus, vals = [], []
    for q, fq in fits.items():
        for kk in range(e.z["ets_from"], len(e.cyc[q])):
            tau = e.starts[q][kk] + np.arange(P) - np.polyval(fq, kk)
            sel = (tau > -15) & (tau < 60)
            taus.append(tau[sel]); vals.append((e.cyc[q][kk] - e.cyc[q][kk, :n_ped].mean())[sel])
    taus, vals = np.concatenate(taus), np.concatenate(vals)
    bins = np.arange(-15, 60.05, 0.1)
    idx = np.digitize(taus, bins)
    sums = np.bincount(idx, weights=vals, minlength=len(bins) + 1)[1:len(bins)]
    cnt = np.bincount(idx, minlength=len(bins) + 1)[1:len(bins)]
    ets = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    centers = bins[:-1] + 0.05
    fig, ax = plt.subplots(figsize=(FULL * 0.62, 2.5))
    sub = np.random.default_rng(0).choice(len(taus), min(20000, len(taus)), replace=False)
    ax.plot(taus[sub], vals[sub], ".", ms=0.5, color="0.75", label="samples at their exact time")
    ax.plot(centers, ets, color="#e76f51", lw=1, label="equivalent-time average (0.1-sample bins)")
    whole = [np.nanmean(vals[(taus >= j - 0.5) & (taus < j + 0.5)]) for j in range(-15, 60)]
    ax.plot(range(-15, 60), whole, "o", ms=2, color="k", label="whole-sample average")
    ax.set(xlabel="samples from the reset edge", ylabel="ADU", title="reset pulse by equivalent-time sampling")
    ax.legend(loc="center right")
    fig.tight_layout()
    save(fig, out, "edge_equivalent_time.png")

    peak_at = centers[np.nanargmax(ets)]
    sel = (centers > peak_at + 0.5) & np.isfinite(ets)
    ta, ya = centers[sel] - peak_at, ets[sel]
    _, tau1, tau2, coef, model = fit_two_exponentials(ta, ya, np.geomspace(0.5, 200, 40))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    axes[0].plot(ta, ya, lw=0.8, label="after the peak")
    axes[0].plot(ta, model, "--", color="#e76f51",
                 label=f"two exponentials: {tau1 / RATE * 1e6:.3f}, {tau2 / RATE * 1e6:.3f} µs")
    axes[0].set(xlabel="samples after the peak", ylabel="ADU", title="recovery after the reset pulse")
    axes[0].legend()
    early = ta < 25
    axes[1].plot(ta[early], (ya - model)[early], lw=0.9)
    axes[1].set(xlabel="samples after the peak", ylabel="residual (ADU)", title="residual: ringing")
    fig.tight_layout()
    save(fig, out, "edge_recovery_ringing.png")

    profile = np.mean([(Y - Y[:, :n_ped].mean(1, keepdims=True)).std(0) for Y in (e.cyc[q][e.first:] for q in e.good)], axis=0)
    fig, ax = plt.subplots(figsize=(FULL, 2.3))
    ax.semilogy(profile, lw=0.8)
    e.shade_windows(ax)
    ax.set(xlabel="sample in the cycle", ylabel="std across skips (ADU)",
           title="variation along the cycle (green: pedestal window, red: signal window)")
    fig.tight_layout()
    save(fig, out, "edge_cycle_variation.png")


# --------------------------------------------------------------------------- E. slow structure

def slow_figures(e: Example, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    for q in range(e.r.n_pixels):
        b = e.cyc[q][:, :e.n_ped].mean(1)
        axes[0].plot(e.starts[q] / RATE * 1e3, b, lw=0.6, color="#e76f51" if q in e.row_starts else "#264653")
    axes[0].set(xlabel="time in the trace (ms)", ylabel="baseline (ADU)", title="baseline of every skip (red: row starts)")
    cands = [q for q in e.row_starts if q + 1 < e.r.n_pixels and q + 1 not in e.row_starts]
    cands = [q for q in cands if q > 0] + [q for q in cands if q == 0]
    rs = cands[0]
    diff = e.cyc[rs][:e.n, :e.n_ped].mean(1) - e.cyc[rs + 1][:e.n, :e.n_ped].mean(1)
    tau, amp, off = fit_exponential(np.arange(len(diff), dtype=float), diff, 2, max(2000, 4 * e.n))
    axes[1].plot(diff, lw=0.6, color="0.5")
    axes[1].plot(off + amp * np.exp(-np.arange(len(diff)) / tau), color="#e76f51", label=f"τ = {tau:.0f} skips")
    axes[1].set(xlabel="skip", ylabel="ADU", title=f"row-start pixel {rs} minus pixel {rs + 1}")
    axes[1].legend()
    fig.tight_layout()
    save(fig, out, "slow_row_start.png")

    mid = e.n // 2
    pts = np.array([[e.starts[q][mid] / RATE * 1e3, e.cyc[q][mid, :e.n_ped].mean()] for q in e.good])
    fit = np.polyfit(pts[:, 0], pts[:, 1], 1)
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=3)
    ax.plot(pts[:, 0], np.polyval(fit, pts[:, 0]), color="#e76f51", label=f"{fit[0]:.2f} ADU/ms")
    ax.set(xlabel="time (ms)", ylabel="baseline at the middle skip (ADU)", title="drift")
    ax.legend()
    fig.tight_layout()
    save(fig, out, "slow_drift.png")

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4))
    q = e.good[0]
    H = e.r.h[q].astype(float)
    H = H - H[:, :min(30, max(1, H.shape[1] // 10))].mean(1, keepdims=True)
    for i in range(0, len(H), 4):
        axes[0].plot(H[i], lw=0.6, color=plt.cm.viridis(i / len(H)))
    axes[0].set(xlabel="sample", ylabel="ADU", title=f"every 4th horizontal transfer before pixel {q}")
    amps = [Hp.max(1) - Hp.min(1) for Hp in ((e.r.h[g].astype(float) - e.r.h[g].astype(float)[:, :30].mean(1, keepdims=True))
                                           for g in e.good)]
    axes[1].plot(np.mean(amps, 0), "o-", ms=3)
    axes[1].set(xlabel="transfer number", ylabel="peak-to-peak (ADU)", title="peak-to-peak of each transfer (mean of pixels)")
    fig.tight_layout()
    save(fig, out, "slow_h_transfers.png")

    R = np.vstack(list(e.res.values()))
    U, sv, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
    frac = sv ** 2 / np.sum(sv ** 2)
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.3), gridspec_kw={"width_ratios": [1, 2, 1.6]})
    axes[0].bar(range(1, 9), frac[:8], color="#a0a0a0")
    axes[0].set_xticks(range(1, 9))
    axes[0].set(xlabel="component", ylabel="variance fraction", title="explained variance")
    for i in range(3):
        axes[1].plot(Vt[i], lw=0.7, label=f"PC{i + 1}")
    e.shade_windows(axes[1])
    axes[1].set(xlabel="sample in the cycle", title="component shapes")
    axes[1].legend()
    n_first = len(e.res[e.good[0]])
    axes[2].plot(np.arange(100) + e.first, U[:100, 0] * sv[0], "o-", ms=1.5, lw=0.5)
    axes[2].set(xlabel="skip", title="PC1 score, first 100 skips")
    fig.tight_layout()
    save(fig, out, "slow_principal_components.png")


# --------------------------------------------------------------------------- curves and the session

def curve_figures(e: Example, out: Path) -> None:
    curves = e.features.curves
    names = list(curves)
    xlabels = {"skip: mean shape": "sample", "skip: baseline droop curve": "skip", "horizontal: mean shape": "sample",
               "vertical: mean shape": "sample", "level: common CDS pattern": "skip", "noise: CDS autocorrelation": "lag (skips)",
               "noise: Allan deviation": "skips averaged", "noise: CDS spectrum": "kHz", "noise: within-cycle spectrum": "MHz",
               "noise: noise vs window length": "window length (samples)", "edge: equivalent-time pulse": "samples from the edge",
               "edge: variation along the cycle": "sample in the cycle",
               "noise: Allan deviation, common pattern removed": "skips averaged",
               "noise: CDS spectrum, log frequency": "log-frequency bin (lowest frequency to Nyquist)"}
    half = (len(names) + 1) // 2
    for part, chunk in (("a", names[:half]), ("b", names[half:])):
        rows = (len(chunk) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(FULL, 2.15 * rows))
        for ax, name in zip(axes.ravel(), chunk):
            c = np.asarray(curves[name], float)
            if name in ("noise: Allan deviation", "noise: Allan deviation, common pattern removed"):
                ax.loglog(ALLAN_N[:len(c)], c, "o-", ms=2.5)
                ax.loglog(ALLAN_N, c[0] / np.sqrt(ALLAN_N), "k:", lw=0.7)       # white noise
            elif name == "noise: CDS spectrum, log frequency":
                ax.semilogy(c, "o-", ms=2, lw=0.7)
            elif name == "noise: CDS spectrum":
                ax.semilogy(np.linspace(0, e.skip_rate / 2e3, len(c)), c, lw=0.7)
            elif name == "noise: within-cycle spectrum":
                ax.loglog(np.fft.rfftfreq(e.sig_end, 1 / RATE)[1:len(c)] / 1e6, c[1:], lw=0.7)
            elif name == "noise: noise vs window length":
                ax.loglog(e.z["window_lengths"][:len(c)], c, "o-", ms=2.5)
            elif name == "noise: CDS autocorrelation":
                ax.bar(np.arange(1, len(c) + 1), c, width=0.8)
            elif name == "edge: equivalent-time pulse":
                ax.plot(np.arange(-15, 60, 0.1)[:len(c)], c, lw=0.8)
            elif name == "edge: variation along the cycle":
                ax.semilogy(c, lw=0.7)
            else:
                ax.plot(c, lw=0.7)
            ax.set(title=name, xlabel=xlabels.get(name, ""))
        for ax in axes.ravel()[len(chunk):]:
            ax.set_visible(False)
        fig.tight_layout()
        save(fig, out, f"all_curves_{part}.png")


def session_figures(out: Path) -> dict:
    """Separation of the 4 amplifiers (6 pairs) and feature correlation over the healthy 2026-10-08 traces."""
    names = scalar_names()
    rows = {}
    for ch in range(4):
        for path in sorted(SESSION.glob(f"trace_*_ch{ch}_101_*.RJ45ADCDATA{ch}")):
            try:
                rows[(ch, path.name)] = trace_features(path).scalars
            except ValueError:
                pass
    by_ch = {ch: np.array([[s[n] for n in names] for (c, _), s in rows.items() if c == ch]) for ch in range(4)}
    sep = np.full((len(names), 6), np.nan)
    for k, (a, b) in enumerate(itertools.combinations(range(4), 2)):
        A, B = by_ch[a], by_ch[b]
        with np.errstate(invalid="ignore", divide="ignore"):
            pooled = np.sqrt((np.nanvar(A, 0, ddof=1) * (len(A) - 1) + np.nanvar(B, 0, ddof=1) * (len(B) - 1)) / (len(A) + len(B) - 2))
            sep[:, k] = np.where(pooled > 0, np.abs(np.nanmean(A, 0) - np.nanmean(B, 0)) / pooled, np.nan)
    med = np.nanmedian(sep, axis=1)

    # separation: two columns of horizontal bars, readable at full page
    order = [i for i in np.argsort(-np.nan_to_num(med, nan=-1))]
    halves = (order[:43], order[43:])
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 8.6))
    top = np.nanmax(med) * 2
    for ax, part in zip(axes, halves):
        part = part[::-1]
        vals = [med[i] if np.isfinite(med[i]) else 0 for i in part]
        ax.barh(range(len(part)), np.maximum(vals, 1e-2), color=[GROUP_COLORS[FEATURES[names[i]].group] for i in part])
        ax.set_yticks(range(len(part)))
        ax.set_yticklabels([names[i] for i in part], fontsize=5.6)
        ax.set_xscale("log")
        ax.axvline(3, color="k", ls="--", lw=0.7)
        ax.set_xlabel("median separation over the 6 amplifier pairs")
        ax.set_ylim(-0.6, len(part) - 0.4)
        ax.set_xlim(1e-2, top)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    fig.legend(handles, list(GROUP_COLORS), ncol=9, loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, out, "feature_separation.png")

    defined = [n for n in names if all(np.isfinite(s[n]) for s in rows.values())]
    X = np.array([[s[n] for n in defined] for s in rows.values()])
    keep = X.std(0) > 0
    defined = [n for n, k in zip(defined, keep) if k]
    R = np.corrcoef(X[:, keep].T)
    groups = [FEATURES[n].group for n in defined]
    fig, ax = plt.subplots(figsize=(FULL, 6.2))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    # one labelled band per feature group instead of 85 unreadable names
    bounds = [0] + [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]] + [len(groups)]
    centers, labels = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        for x in (a - 0.5,):
            ax.axhline(x, color="k", lw=0.4)
            ax.axvline(x, color="k", lw=0.4)
        centers.append((a + b - 1) / 2)
        labels.append(f"{groups[a]} ({b - a})")
        ax.add_patch(plt.Rectangle((-4.5, a - 0.5), 3, b - a, color=GROUP_COLORS[groups[a]], clip_on=False))
        ax.add_patch(plt.Rectangle((a - 0.5, len(groups) + 1.5), b - a, 3, color=GROUP_COLORS[groups[a]], clip_on=False))
    ax.set_yticks(centers)
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="y", pad=24, length=0)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.tick_params(axis="x", pad=26, length=0)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="correlation over the 19 traces")
    fig.tight_layout()
    save(fig, out, "feature_correlation.png")
    return {"n_traces": len(rows), "separation": {n: med[i] for i, n in enumerate(names)}}


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    e = Example(EXAMPLE)
    for function in (level_figures, noise_figures, discrete_figures, edge_figures, slow_figures, curve_figures):
        function(e, out)
        print("done:", function.__name__, flush=True)
    session_figures(out)
    print("done: session_figures", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "figures"))
