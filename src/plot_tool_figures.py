"""
plot_tool_figures.py -- summary figures for the comparison tools of ``compare_trace.py``.

Two groups of figures, written as PNG files into OUT_DIR:

1. ``tools_annotated.png``  what the tools measure.  One healthy trace, one pixel: the skip-cycle
   template with the dip / pulse / signal-window / tail markers, the three exponential fits, the
   baseline droop with its fit, the first-skip CDS with the ringing fit, the noise-against-N model,
   the noise power bands, the horizontal block markers, and the per-pixel values of one tool for
   several runs (replicate scatter against run-to-run scatter).

2. ``compare_*.png``  how much stronger the distinguishability is for two DIFFERENT amplifiers
   (channel 0 against channel 3) than for REPEATS of channel 0 with identical parameters.  Uses
   every healthy channel-0 and channel-3 trace of a session folder, with row-start pixels and
   charge-contaminated pixels excluded (``compare_trace.usable_pixels``):

       compare_z_by_tool.png   z of every tool for ch0-ch0 pairs and ch0-ch3 pairs, and the size of the
                               ch0-ch3 difference in units of the run-to-run spread of ch0
       compare_verdicts.png    how many tools are flagged for each pair; and the waveform deviations
       compare_waveforms.png   mean cycle shapes and deviations from the ch0 reference

Usage:
    python plot_tool_figures.py SESSION_DIR OUT_DIR [--perms 10000]
SESSION_DIR holds trace_*_ch0_101_*.RJ45ADCDATA0 and trace_*_ch3_101_*.RJ45ADCDATA3 files.
"""

from __future__ import annotations

import argparse
import glob
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import compare_trace as ct
from decode_trace import decode_trace
from trace_regions import TraceRegions, split_regions

US = 15.0                                         # samples per microsecond
RED, GREEN, BLUE, ORANGE, GREY = "tab:red", "tab:green", "tab:blue", "tab:orange", "0.6"


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def healthy_traces(pattern: str) -> list[tuple[str, TraceRegions]]:
    """Decode every file matching ``pattern``; keep only traces that pass the quality check."""
    out = []
    for path in sorted(glob.glob(pattern)):
        trace = decode_trace(path)
        if trace.quality_problems():
            print(f"skipping damaged trace {Path(path).name}")
            continue
        out.append((Path(path).name, split_regions(trace)))
    return out


def pixel_template(r: TraceRegions, pixel: int) -> np.ndarray:
    """Mean baseline-removed cycle of one pixel (skips 5 onwards)."""
    cycles = r.skip[pixel].astype(float)
    return (cycles - cycles[:, :r.n_ped].mean(axis=1, keepdims=True))[5:].mean(axis=0)


# --------------------------------------------------------------------------- #
# Figure 1: what the tools measure                                             #
# --------------------------------------------------------------------------- #

def plot_annotated(r: TraceRegions, others: list[TraceRegions], pixel: int, out: Path) -> None:
    """Nine-panel figure that annotates what each tool measures."""
    cycles = r.skip[pixel].astype(float)
    base = cycles[:, :r.n_ped].mean(axis=1)
    shape = cycles - base[:, None]
    template = shape[5:].mean(axis=0)
    cds = r.cds[pixel]
    later = cds[10:]
    t = np.arange(r.period) / US
    n_ped, j_sig = r.n_ped, r.sig_start

    i_dip = n_ped + int(np.argmin(template[n_ped:j_sig]))
    i_peak = j_sig + n_ped + int(np.argmax(template[j_sig + n_ped:]))
    tau_dip, amp_dip, off_dip = ct.fit_exponential(np.arange(j_sig - i_dip) / US, template[i_dip:j_sig], 0.02, 10)
    tau_set, amp_set, off_set = ct.fit_exponential(np.arange(len(template) - i_peak) / US, template[i_peak:], 0.05, 20)
    # the values printed in the figure must agree with what compare_trace measures
    tools = ct.skip_tools(r, pixel)
    assert abs(tools["fit: dip recovery time (us)"] - tau_dip) < 1e-9
    assert abs(tools["fit: settling time after pulse (us)"] - tau_set) < 1e-9

    fig, ax = plt.subplots(3, 3, figsize=(18, 14))

    # A: the cycle with windows and level markers -------------------------------------------------
    a = ax[0, 0]
    a.plot(t, template, color="0.2", lw=1)
    a.axvspan(0, n_ped / US, color=RED, alpha=0.15, label="pedestal window")
    a.axvspan(j_sig / US, (j_sig + n_ped) / US, color=GREEN, alpha=0.15, label="signal window")
    a.plot(i_dip / US, template[i_dip], "v", color=BLUE, ms=9, label="dip height (min)")
    a.plot(i_peak / US, template[i_peak], "^", color=ORANGE, ms=9, label="pulse peak height (max)")
    sig = slice(j_sig + 10, j_sig + n_ped - 10)
    a.hlines(template[sig].mean(), t[sig][0], t[sig][-1], color=GREEN, lw=3, label="signal-window level")
    a.hlines(template[-30:].mean(), t[-30], t[-1], color="tab:purple", lw=3, label="end-of-cycle tail level")
    a.set(title="A  skip-cycle template (mean of 495 cycles)", xlabel="time in cycle (us)", ylabel="ADU (baseline removed)")
    a.legend(fontsize=8, loc="upper left")

    # B: dip recovery fit ---------------------------------------------------------------------------------
    a = ax[0, 1]
    x = np.arange(j_sig - i_dip) / US
    a.plot(x, template[i_dip:j_sig], "o", ms=3, color=GREY, label="template")
    a.plot(x, off_dip + amp_dip * np.exp(-x / tau_dip), color=BLUE, lw=2,
           label=f"c + A exp(-x/tau)\ntau = {tau_dip:.2f} us, A = {amp_dip:.0f} ADU")
    a.set(title="B  fit: dip recovery (time and amplitude)", xlabel="time after dip minimum (us)", ylabel="ADU")
    a.legend(fontsize=9)

    # C: settling fit -------------------------------------------------------------------------------------
    a = ax[0, 2]
    x = np.arange(len(template) - i_peak) / US
    a.plot(x, template[i_peak:], "o", ms=3, color=GREY, label="template")
    a.plot(x, off_set + amp_set * np.exp(-x / tau_set), color=ORANGE, lw=2,
           label=f"c + A exp(-x/tau)\ntau = {tau_set:.2f} us, A = {amp_set:.0f} ADU")
    a.set(title="C  fit: settling after the pulse (time and amplitude)", xlabel="time after pulse peak (us)", ylabel="ADU")
    a.legend(fontsize=9)

    # D: baseline droop ------------------------------------------------------------------------------------
    a = ax[1, 0]
    k = np.arange(1, len(base))
    tau_d, amp_d, off_d = ct.fit_exponential(k.astype(float), base[1:], 5, 5000)
    a.plot(k, base[1:], ".", ms=3, color=GREY, label="baseline of each skip")
    a.plot(k, off_d + amp_d * np.exp(-k / tau_d), color=BLUE, lw=2, label=f"fit: tau = {tau_d:.0f} skips")
    total = base[-1] - base[1]
    a.annotate("", xy=(len(base) - 1, base[-1]), xytext=(len(base) - 1, base[1]),
               arrowprops=dict(arrowstyle="<->", color=RED))
    a.text(len(base) * 0.72, (base[1] + base[-1]) / 2, f"droop total\n{total:.0f} ADU", color=RED)
    half = np.flatnonzero((base[1:] - base[1]) * np.sign(total) >= 0.5 * abs(total))[0]
    a.axvline(half + 1, color="tab:green", ls="--", label=f"droop half-time: {half} skips")
    a.plot(1, base[1], "o", color=ORANGE, ms=8, label="baseline at skip 1")
    a.set(title="D  baseline droop: skip 1, total, half-time, fit", xlabel="skip number", ylabel="pedestal-window mean (ADU)")
    a.legend(fontsize=8, loc="lower right")

    # E: first skips: CDS, excess, ringing fit ---------------------------------------------------------------
    a = ax[1, 1]
    n_show = 25
    freq, tau_r, amp_r = ct.fit_damped_ringing(cds[:20] - np.median(later))
    kk = np.arange(20)
    envelope = np.exp(-kk / tau_r)
    basis = np.column_stack([envelope * np.cos(2 * np.pi * freq * kk), envelope * np.sin(2 * np.pi * freq * kk)])
    coef = np.linalg.lstsq(basis, cds[:20] - np.median(later), rcond=None)[0]
    a.plot(np.arange(n_show), cds[:n_show], "o-", ms=4, color=GREY, label="CDS of each skip")
    a.plot(kk, np.median(later) + basis @ coef, color=BLUE, lw=2,
           label=f"ringing fit: f = {freq:.2f}, decay = {tau_r:.1f} skips")
    a.axhline(later.mean(), color=GREEN, ls="--", label=f"CDS mean (skips >= 10): {later.mean():.0f}")
    a.annotate("", xy=(0, cds[0]), xytext=(0, np.median(later)), arrowprops=dict(arrowstyle="<->", color=RED))
    a.text(1.2, (cds[0] + np.median(later)) / 2, "first-skip\nCDS excess", color=RED)
    a.set(title="E  first skips: excess, ringing fit, CDS mean", xlabel="skip number", ylabel="CDS = pedestal - signal (ADU)")
    a.legend(fontsize=8)

    # F: noise against N ------------------------------------------------------------------------------------------
    a = ax[1, 2]
    resid = ct._poly_detrended(later)
    sizes = [1, 2, 5, 10, 25, 50]
    noise = [resid[:len(resid) // n * n].reshape(-1, n).mean(axis=1).std(ddof=1) for n in sizes]
    white, floor = ct.fit_noise_model(sizes, noise)
    grid = np.geomspace(1, 60, 100)
    a.loglog(sizes, noise, "o", color="tab:purple", label="noise of an N-skip average")
    a.loglog(grid, np.sqrt(white ** 2 / grid + floor ** 2), color=BLUE, lw=2,
             label=f"fit: white = {white:.1f} ADU, floor = {floor:.2f} ADU")
    a.loglog(grid, noise[0] / np.sqrt(grid), "k--", lw=0.8, label="1/sqrt(N)")
    a.set(title="F  noise against skips averaged (white + floor model)", xlabel="N skips averaged",
          ylabel="rms of the average (ADU)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3, which="both")

    # G: noise power bands ---------------------------------------------------------------------------------------------
    a = ax[2, 0]
    window = shape[5:, :n_ped] - template[:n_ped]
    window = window - window.mean(axis=1, keepdims=True)
    power = (np.abs(np.fft.rfft(window, axis=1)) ** 2).mean(axis=0) / n_ped
    f_mhz = np.fft.rfftfreq(n_ped, 1 / (US * 1e6)) / 1e6
    a.semilogy(f_mhz[1:], power[1:], color="0.2")
    for lo, hi, color in [(0.1, 1, BLUE), (1, 3, GREEN), (3, 7.51, ORANGE)]:
        a.axvspan(lo, hi, color=color, alpha=0.2, label=f"noise power {lo:g}-{min(hi, 7.5):g} MHz")
    a.set(title="G  noise power of the pedestal window, three bands", xlabel="frequency (MHz)", ylabel="power per sample (ADU^2)")
    a.legend(fontsize=8)

    # H: horizontal block --------------------------------------------------------------------------------------------------
    a = ax[2, 1]
    h = r.h[pixel].astype(float)
    h = (h - h[:, :30].mean(axis=1, keepdims=True)).mean(axis=0)
    th = np.arange(len(h)) / US
    a.plot(th, h, color="0.2", lw=1)
    a.plot(th[np.argmin(h)], h.min(), "v", color=BLUE, ms=9, label="horizontal minimum")
    a.plot(th[np.argmax(h)], h.max(), "^", color=ORANGE, ms=9, label="horizontal maximum")
    a.hlines(h[-50:].mean(), th[-50], th[-1], color="tab:purple", lw=3, label="end level")
    a.set(title="H  horizontal transfer (mean of 19): minimum, maximum, end level", xlabel="time in transfer (us)",
          ylabel="ADU (baseline removed)")
    a.legend(fontsize=8)

    # I: replicates against runs -------------------------------------------------------------------------------------------------
    a = ax[2, 2]
    for run, reg in enumerate([r] + others):
        values = [ct.skip_tools(reg, p)["skip: pulse peak height"] for p in ct.usable_pixels(reg)]
        a.plot(np.full(len(values), run) + np.linspace(-0.15, 0.15, len(values)), values, "o", ms=5, color=f"C{run}")
        a.hlines(np.mean(values), run - 0.25, run + 0.25, color="k", lw=2)
    a.set(title="I  one tool, per pixel (dots) and per run (bars)\nrun-to-run shifts exceed pixel scatter",
          xlabel="repeat run (identical parameters)", ylabel="pulse peak height (ADU)", xticks=range(1 + len(others)))
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure group 2: ch0-ch3 against ch0 repeats                                  #
# --------------------------------------------------------------------------- #

def run_value(tools: dict, name: str) -> np.ndarray:
    """Per-run value of a tool: mean over the pixels (a curve stays a curve)."""
    return np.asarray(tools[name]).mean(axis=0).ravel()


def plot_comparisons(ch0: list, ch3: list, out_dir: Path, n_perm: int) -> None:
    """Statistics and the three comparison figures."""
    meas0 = [ct.measure(r, ct.usable_pixels(r)) for _, r in ch0]
    meas3 = [ct.measure(r, ct.usable_pixels(r)) for _, r in ch3]
    names = [n for n in meas0[0].tools if not n.startswith("vertical")]
    threshold = 0.01 / len(names)

    def pair_results(a, b, seed):
        res = [x for x in ct.run_tests(a.tools, b.tools, n_perm, seed) if x["region"] != "vertical"]
        return {x["tool"]: x for x in res}

    same = [pair_results(meas0[i], meas0[j], 10 * i + j) for i, j in itertools.combinations(range(len(meas0)), 2)]
    diff = [pair_results(a, b, 100 + 10 * i + j) for i, a in enumerate(meas0) for j, b in enumerate(meas3)]
    print(f"ch0 repeats: {len(meas0)}, ch3 repeats: {len(meas3)}; pairs: {len(same)} same-amplifier, {len(diff)} cross-amplifier")

    # ---- ratio: ch0-ch3 difference / run-to-run spread of ch0 ------------------------------------------------------------
    ratio = {}
    for n in names:
        runs0 = np.array([run_value(m.tools, n) for m in meas0])
        runs3 = np.array([run_value(m.tools, n) for m in meas3])
        spread = np.sqrt(np.mean(runs0.var(axis=0, ddof=1)))
        delta = np.sqrt(np.mean((runs0.mean(axis=0) - runs3.mean(axis=0)) ** 2))
        if spread > 1e-12 and np.isfinite(delta):
            ratio[n] = delta / spread

    # ---- figure: z by tool and ratio -----------------------------------------------------------------------------------------
    order = sorted(names, key=lambda n: np.median([d[n]["z"] for d in diff if n in d]))
    fig, ax = plt.subplots(1, 2, figsize=(17, 10), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    for row, n in enumerate(order):
        z_same = [max(d[n]["z"], 0.01) for d in same if n in d]
        z_diff = [max(d[n]["z"], 0.01) for d in diff if n in d]
        a.plot(z_same, np.full(len(z_same), row) - 0.12, "o", color=GREY, ms=5, alpha=0.8,
               label="ch0 vs ch0 repeats (same amplifier)" if row == 0 else None)
        a.plot(z_diff, np.full(len(z_diff), row) + 0.12, "o", color=RED, ms=4, alpha=0.5,
               label="ch0 vs ch3 (different amplifier)" if row == 0 else None)
    a.set_xscale("log")
    a.set_yticks(range(len(order)))
    a.set_yticklabels(order, fontsize=8)
    a.axvline(3, color="k", ls=":", lw=0.8)
    a.set(title="z of every tool: repeats of one amplifier against two different amplifiers\n"
                "(z = size of the difference in units of the pixel-level error)", xlabel="z (log scale)")
    a.legend(loc="lower right", fontsize=9)
    a.grid(alpha=0.3, axis="x")

    a = ax[1]
    keys = sorted(ratio, key=lambda n: ratio[n])
    colors = [RED if ratio[n] >= 3 else GREY for n in keys]
    a.barh(range(len(keys)), [ratio[n] for n in keys], color=colors)
    a.set_yticks(range(len(keys)))
    a.set_yticklabels(keys, fontsize=8)
    a.set_xscale("log")
    a.axvline(1, color="k", ls="-", lw=0.8)
    a.axvline(3, color="k", ls="--", lw=0.8)
    a.text(1.05, 0.2, "1 = typical\nrepeat noise", fontsize=8)
    a.text(3.2, 0.2, "3 = detectable", fontsize=8)
    a.set(title="size of the ch0-ch3 difference in units of the\nrun-to-run spread of ch0 repeats (the honest noise floor)",
          xlabel="|ch0 - ch3| / run-to-run spread (log scale)")
    a.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_z_by_tool.png", dpi=100)
    plt.close(fig)

    # ---- figure: number of tools flagged per pair --------------------------------------------------------------------------------
    flags_same = [sum(d[n]["p"] < threshold for n in d) for d in same]
    flags_diff = [sum(d[n]["p"] < threshold for n in d) for d in diff]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    rng = np.random.default_rng(0)
    a.plot(rng.uniform(-0.12, 0.12, len(flags_same)), flags_same, "o", color=GREY, ms=8, label=f"ch0 vs ch0 ({len(same)} pairs)")
    a.plot(1 + rng.uniform(-0.12, 0.12, len(flags_diff)), flags_diff, "o", color=RED, ms=8, alpha=0.7, label=f"ch0 vs ch3 ({len(diff)} pairs)")
    a.set(xlim=(-0.6, 1.6), ylim=(-1, len(names) + 1), xticks=[0, 1], xticklabels=["same amplifier\n(identical parameters)", "different amplifiers\n(identical parameters)"],
          ylabel=f"number of the {len(names)} tools flagged as different", title="pixel-level test (as in compare_trace.py):\nnumber of tools flagged per pair")
    a.axhline(len(names), color="k", ls=":", lw=0.8)
    a.legend(fontsize=9, loc="center left")
    a = ax[1]
    # flags against the run-to-run spread of ch0: a tool is flagged for a pair of runs if their
    # difference exceeds 3 standard deviations of the expected difference of two repeats
    spread = {n: np.sqrt(np.mean(np.array([run_value(m.tools, n) for m in meas0]).var(axis=0, ddof=1))) for n in names}

    def honest_flags(a_tools, b_tools):
        count = 0
        for n in names:
            if spread[n] <= 1e-12:
                continue
            d = np.sqrt(np.mean((run_value(a_tools, n) - run_value(b_tools, n)) ** 2))
            count += d > 3 * np.sqrt(2) * spread[n]
        return count

    honest_same = [honest_flags(meas0[i].tools, meas0[j].tools) for i, j in itertools.combinations(range(len(meas0)), 2)]
    honest_diff = [honest_flags(x.tools, y.tools) for x in meas0 for y in meas3]
    a.plot(rng.uniform(-0.12, 0.12, len(honest_same)), honest_same, "o", color=GREY, ms=8, label=f"ch0 vs ch0 ({len(same)} pairs)")
    a.plot(1 + rng.uniform(-0.12, 0.12, len(honest_diff)), honest_diff, "o", color=RED, ms=8, alpha=0.7, label=f"ch0 vs ch3 ({len(diff)} pairs)")
    a.set(xlim=(-0.6, 1.6), ylim=(-1, len(names) + 1), xticks=[0, 1],
          xticklabels=["same amplifier", "different amplifiers"],
          ylabel=f"number of the {len(names)} tools flagged",
          title="same count, judged against the run-to-run spread\n(flag if the difference exceeds 3 sigma of two repeats)")
    a.axhline(len(names), color="k", ls=":", lw=0.8)
    a.legend(fontsize=9, loc="center left")
    print("flags against the run-to-run spread: same amplifier", honest_same, "; different amplifiers", sorted(honest_diff))
    fig.tight_layout()
    fig.savefig(out_dir / "compare_verdicts.png", dpi=100)
    plt.close(fig)

    # ---- figure: waveforms --------------------------------------------------------------------------------------------------------
    def run_template(r):
        return np.mean([pixel_template(r, p) for p in ct.usable_pixels(r)], axis=0)

    t0 = np.array([run_template(r) for _, r in ch0])
    t3 = np.array([run_template(r) for _, r in ch3])
    reference = t0.mean(axis=0)
    t = np.arange(t0.shape[1]) / US
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    a = ax[0]
    for i, tpl in enumerate(t0):
        a.plot(t, tpl, color=GREY, lw=1, label="ch0 runs" if i == 0 else None)
    for i, tpl in enumerate(t3):
        a.plot(t, tpl, color=RED, lw=1, label="ch3 runs" if i == 0 else None)
    a.set(title="A  mean cycle shape of every run", xlabel="time in cycle (us)", ylabel="ADU (baseline removed)")
    a.legend()
    a = ax[1]
    zoom = (t > 23.8) & (t < 27)
    for tpl in t0:
        a.plot(t[zoom], tpl[zoom], color=GREY, lw=1.2)
    for tpl in t3:
        a.plot(t[zoom], tpl[zoom], color=RED, lw=1.2)
    a.set(title="B  zoom on the pulse: repeats of ch0 (grey) against ch3 (red)", xlabel="time in cycle (us)")
    a = ax[2]
    floor = 1e-1
    for i, tpl in enumerate(t0):
        a.semilogy(t, np.maximum(np.abs(tpl - reference), floor), color=GREY, lw=0.8,
                   label="|ch0 run - ch0 mean|" if i == 0 else None)
    for i, tpl in enumerate(t3):
        a.semilogy(t, np.maximum(np.abs(tpl - reference), floor), color=RED, lw=0.8,
                   label="|ch3 run - ch0 mean|" if i == 0 else None)
    a.set(title="C  deviation from the ch0 mean shape", xlabel="time in cycle (us)", ylabel="ADU")
    a.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "compare_waveforms.png", dpi=100)
    plt.close(fig)

    low = sorted((kv for kv in ratio.items() if kv[1] < 3), key=lambda kv: kv[1])
    print("tools below the 3x detectability line:", "; ".join(f"{k.split(': ')[1]} {v:.1f}" for k, v in low))
    print("tools with no run-to-run variation (excluded from the ratio):", [n for n in names if n not in ratio])
    big = sorted(ratio.items(), key=lambda kv: -kv[1])[:6]
    print("largest ch0-ch3 / run-to-run ratios:", ", ".join(f"{k.split(': ')[1]} {v:.0f}" for k, v in big))
    print(f"tools flagged per pair: same amplifier {flags_same}; different amplifiers {sorted(flags_diff)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--perms", type=int, default=10000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ch0 = healthy_traces(f"{args.session_dir}/trace_*_ch0_101_*.RJ45ADCDATA0")
    ch3 = healthy_traces(f"{args.session_dir}/trace_*_ch3_101_*.RJ45ADCDATA3")
    plot_annotated(ch0[0][1], [r for _, r in ch0[1:]], pixel=6, out=out_dir / "tools_annotated.png")
    plot_comparisons(ch0, ch3, out_dir, args.perms)
    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
