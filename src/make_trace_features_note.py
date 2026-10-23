"""
make_trace_features_note.py -- write docs/overleaf_trace_features/trace_features.tex (and its zip) from the FEATURES
registry of trace_features.py (141 scalars, 18 curves since 2026-10-21), the tiers (lvl1/feature_study/evidence_tiers.json,
MODEL_TIERS of lvl1/train_lvl1.py), the 2026-10-08 separation results and the figures.

Since 2026-10-22 it adds the start: and clock: groups, an implementation/evidence tier column, the origin of every new
feature and a tier section, and redraws the curve, separation and correlation figures for all features
(src/plot_feature_figures.py draws the per-group figures).  The old 85-feature copy docs/trace_features.tex was removed.

    cd src && ../.venv/bin/python plot_feature_figures.py ../docs/overleaf_trace_features/figures   # optional
    ../.venv/bin/python make_trace_features_note.py

The figures are drawn at document size (6.9 in = the text width, or 3.4 in = half of it), so they
are included at their natural width and their text stays readable.
"""
import itertools, re, sys
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from trace_features import ALLAN_N, FEATURES, scalar_names, trace_features
import json

UNICODE = {
    "µ": r"$\mu$", "σ": r"$\sigma$", "λ": r"$\lambda$", "τ": r"$\tau$", "θ": r"$\theta$", "χ": r"$\chi$",
    "Δ": r"$\Delta$", "²": r"$^{2}$", "³": r"$^{3}$", "⁻³": r"$^{-3}$", "×": r"$\times$", "−": r"$-$",
    "√": r"$\surd$", "≥": r"$\geq$", "≤": r"$\leq$", "±": r"$\pm$", "→": r"$\to$", "≈": r"$\approx$",
    "–": "--", "—": "---", "·": r"$\cdot$", "’": "'", "₁": r"$_1$", "₂": r"$_2$",
}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"$\sim$", "^": r"\textasciicircum{}",
           "<": r"\textless{}", ">": r"\textgreater{}"}


def tex(text: str) -> str:
    """Escape plain text for LaTeX (special characters first, then unicode symbols)."""
    out = "".join(SPECIAL.get(c, c) for c in text)
    for u in sorted(UNICODE, key=len, reverse=True):          # "⁻³" before "³"
        out = out.replace(u, UNICODE[u])
    leftover = {c for c in out if ord(c) > 127}
    if leftover:
        raise ValueError(f"unconverted characters {leftover} in {text!r}")
    return out


# --------------------------------------------------------------------------- results on 10-08
SESSION = PROJECT / "data" / "2026-10-08-PA03_PQ04_traces"
session = {}
for ch in range(4):
    for path in sorted(SESSION.glob(f"trace_*_ch{ch}_101_*.RJ45ADCDATA{ch}")):
        try:
            session[(ch, path.name)] = trace_features(path).scalars
        except ValueError:
            pass
names = scalar_names()
by_ch = {ch: np.array([[s[n] for n in names] for (c, _), s in session.items() if c == ch]) for ch in range(4)}
sep = np.full((len(names), 6), np.nan)
for k, (a, b) in enumerate(itertools.combinations(range(4), 2)):
    A, B = by_ch[a], by_ch[b]
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.sqrt((np.nanvar(A, 0, ddof=1) * (len(A) - 1) + np.nanvar(B, 0, ddof=1) * (len(B) - 1)) / (len(A) + len(B) - 2))
        sep[:, k] = np.where(pooled > 0, np.abs(np.nanmean(A, 0) - np.nanmean(B, 0)) / pooled, np.nan)
n_sep = np.sum(sep >= 3, axis=1)
med = np.nanmedian(sep, axis=1)
defined = [n for n in names if all(np.isfinite(s[n]) for s in session.values())]
X = np.array([[s[n] for n in defined] for s in session.values()])
R = np.corrcoef(X.T)
pairs = sorted(((defined[i], defined[j], R[i, j]) for i in range(len(defined)) for j in range(i + 1, len(defined))
                if abs(R[i, j]) > 0.98), key=lambda p: -abs(p[2]))
n_traces = len(session)

# --------------------------------------------------------------------------- tiers
TIERS = json.loads((PROJECT / "lvl1" / "feature_study" / "evidence_tiers.json").read_text())
EVIDENCE = {n: t for t in ("A1", "A2", "B", "C") for n in TIERS.get(t, [])}
PROTECTED = list(TIERS.get("P", []))
sys.path.insert(0, str(PROJECT / "lvl1"))
import train_lvl1  # noqa: E402  (MODEL_TIERS: the per-model default feature sets)
MODEL_TIERS, DEFAULT_MODEL_TIERS = train_lvl1.MODEL_TIERS, train_lvl1.DEFAULT_MODEL_TIERS


def unit(text: str) -> str:
    """A unit for a narrow column: may break after a slash."""
    return tex(text).replace("/", r"/\allowbreak{}")


def tier_cell(n: str) -> str:
    """Implementation tier / evidence tier, and P for the protected estimators."""
    return f"{FEATURES[n].tier}/{EVIDENCE.get(n, '--')}" + (r" P" if n in PROTECTED else "")


# where each new feature came from (docs/literature/signal_analysis_features.md, consolidated table, by the "#N" in the
# description; the unnumbered ones from docs/trace_feature_candidates.md and the user's trace inspection)
_TABLE = {}
for _line in (PROJECT / "docs" / "literature" / "signal_analysis_features.md").read_text().splitlines():
    _m = re.match(r"\| (\d+) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \|", _line)
    if _m:
        _TABLE[int(_m[1])] = _m[6].strip()
FIRST_SKIP = "first-skip offset study (this project)"
ORIGIN_BY_NAME = {
    "start: slope-explained fraction": FIRST_SKIP, "start: transient amplitude (ADU)": FIRST_SKIP,
    "start: first-skip excess per volt H (ADU/V)": FIRST_SKIP, "start: mean transient": FIRST_SKIP,
    "start: row-start/inner first-skip ratio": FIRST_SKIP, "start: reset jump skip 0 to 1 (ADU)": FIRST_SKIP,
    "start: ringing frequency (kHz)": FIRST_SKIP, "level: slope-corrected CDS rise (ADU)": FIRST_SKIP,
    "discrete: null-CDS kurtosis": "GW null streams", "noise: null-CDS autocorrelation at lag 1": "GW null streams",
    "noise: Hadamard deviation": "time-frequency metrology",
    "noise: CDS variance share of the 3 largest waveform modes": "cardiology (KLT of heartbeats), machine diagnostics",
    "noise: waveform modes for 90 % of the CDS variance": "cardiology (KLT of heartbeats), machine diagnostics",
    "discrete: line phase coherence, absolute time": "gravitational waves, power systems",
    "noise: structure-function excess at 9 blocks (ADU)": "astronomy (structure functions)",
    "noise: DFA exponent": "cardiology", "noise: CDS spectral slope": "audio", "noise: CDS spectral centroid": "audio",
    "noise: sample entropy": "nonlinear dynamics", "noise: permutation entropy (m=4)": "nonlinear dynamics",
    "noise: red-noise beta (50 skips)": "astronomy",
}
TRACE_LOOK = "inspection of the raw trace (this project)"


def origin(n: str) -> str:
    if n in ORIGIN_BY_NAME:
        return ORIGIN_BY_NAME[n]
    if FEATURES[n].group == "clock" or "pre-pulse" in n:
        return TRACE_LOOK
    nums = re.findall(r"#(\d+)", FEATURES[n].description)
    vals = [_TABLE.get(int(k), "") for k in nums]
    vals = [v for v in vals if v and v not in ("—", "-")]
    if not vals:
        raise ValueError(f"no origin for {n!r}")
    return vals[0]


NEW = [n for n, f in FEATURES.items() if f.tier > 0]
NEW_ORIGIN = {n: origin(n) for n in NEW}

# --------------------------------------------------------------------------- figures
import shutil
FIGDIR = PROJECT / "docs" / "overleaf_trace_features" / "figures"   # written by plot_feature_figures.py
OUT = PROJECT / "docs" / "overleaf_trace_features"
EX = "Example trace: module 101, channel 0, 2026-10-08, first repeat."
FIGS = {   # group -> [(source png, file name, caption, width)]
    "level": [
        ("level_common_pattern.png", "level_common_pattern.png", "Left: CDS of every pixel along its 500 skips (10-skip average; dotted: row-start pixels). Right: the common pattern (median of the usable pixels), which rises by about 29 ADU along the skips. " + EX, 1.0),
        ("level_pixel_levels.png", "level_pixel_levels.png", "Relative CDS level of each pixel (mean CDS minus the common pattern) with its ±1σ uncertainty from 10-skip blocks (about ±0.2 ADU). Pixel 0 is a row-start pixel. " + EX, 0.5),
        ("level_charge_loss.png", "level_charge_loss.png", "The charged pixel of the fourth channel-0 repeat (about 18,400 ADU): CDS minus the common pattern along its skips, with a straight-line fit (−2.8 % over the pixel).", 1.0),
    ],
    "noise": [
        ("noise_autocorrelation.png", "noise_autocorrelation.png", "Autocorrelation of the detrended CDS, lags 1–50 (mean of the usable pixels); grey band: ±2/√N expected for white noise. Lags 2 and 3 are about −0.2. " + EX, 1.0),
        ("noise_allan.png", "noise_allan.png", "Allan deviation of the detrended CDS (mean of the usable pixels); dashed: white noise scaled from n = 1. " + EX, 0.5),
        ("noise_cds_spectrum.png", "noise_cds_spectrum.png", "Spectral density of the detrended CDS sequence (sampled at the skip rate, 0–15 kHz); note the peak near 5 kHz. " + EX, 1.0),
        ("noise_reset_and_bands.png", "noise_reset_and_bands.png", "Left: reset-level scatter (about 146 ADU) and CDS noise per skip (about 5.6 ADU) for each usable pixel. Right: noise spectrum inside a cycle (pedestal-window start to signal-window end, each cycle's own mean removed). " + EX, 1.0),
        ("noise_window_length.png", "noise_window_length.png", "CDS noise per skip recomputed with shorter windows (circles: windows starting at the flag; squares: windows ending at the flag), against 1/√L white noise (dashed). " + EX, 0.5),
        ("noise_window_settling.png", "noise_window_settling.png", "Mean waveform inside the pedestal window (left) and the signal window (right), window mean removed. The signal window starts about 150 ADU low and settles within about 30 samples (2 µs). " + EX, 1.0),
        ("noise_baseline_coupling.png", "noise_baseline_coupling.png", "Left: CDS against the baseline level of each skip (usable pixels); the slope is the baseline coupling (about 2.1×10⁻³). Right: CDS noise in 50-skip blocks along the pixel. " + EX, 1.0),
    ],
    "discrete": [
        ("discrete_steps.png", "discrete_steps.png", "Detrended baseline (top) and CDS (bottom) of one pixel, with the position of the largest step (dashed), and their histograms with a Gaussian of the same variance. No telegraph noise is seen. " + EX, 1.0),
        ("discrete_lines.png", "discrete_lines.png", "Line search in the within-cycle spectrum (left, MHz) and the CDS spectrum (right, kHz); no peak exceeds 5 times the local median in this trace. " + EX, 1.0),
        ("discrete_adc_codes.png", "discrete_adc_codes.png", "Relative excess of each value of (ADC sample mod 64) inside the pedestal windows; χ² = 58 for 63 degrees of freedom (flat). " + EX, 1.0),
    ],
    "edge": [
        ("edge_jitter.png", "edge_jitter.png", "Left: position of the reset edge inside its own cycle; it saws by a third of a sample because the cycle is 488.7 samples long but starts on whole samples. Right: edge time minus the straight-line fit against the skip number (jitter about 1.7 ns rms). " + EX, 1.0),
        ("edge_equivalent_time.png", "edge_equivalent_time.png", "The reset pulse rebuilt by equivalent-time sampling (red, 0.1-sample bins) compared with the plain average of whole samples (dots); grey: individual samples at their exact time. " + EX, 0.62),
        ("edge_recovery_ringing.png", "edge_recovery_ringing.png", "Left: equivalent-time waveform after the pulse peak with a two-exponential fit. Right: the residual, a decaying oscillation at about 3 MHz (ringing). " + EX, 1.0),
        ("edge_cycle_variation.png", "edge_cycle_variation.png", "Standard deviation across skips at every position of the cycle (each skip's pedestal level removed); shaded: pedestal and signal windows. Small and flat inside the windows, large at the dip and the reset pulse, a few hundred ADU after the reset. " + EX, 1.0),
    ],
    "slow": [
        ("slow_row_start.png", "slow_row_start.png", "Left: baseline of every skip through the trace (red: row-start pixels). Right: baseline of the second row-start pixel minus that of the next pixel, with an exponential fit (about 230 skips). " + EX, 1.0),
        ("slow_drift.png", "slow_drift.png", "Baseline at skip 250 of successive usable pixels against time in the trace; the slope is the drift feature. " + EX, 0.5),
        ("slow_h_transfers.png", "slow_h_transfers.png", "Left: every 4th horizontal transfer before a pixel. Right: peak-to-peak amplitude of each transfer (mean of the usable pixels). " + EX, 1.0),
        ("slow_principal_components.png", "slow_principal_components.png", "Principal components of the skip-to-skip variation: explained variance (left), the first three component shapes (centre; shaded: windows) and the PC1 score along the skips (right). PC1 (82 %) is the sub-sample phase walk at the edges; PC2 is the level shift set by each reset. " + EX, 1.0),
    ],
    "curves": [
        ("all_curves_a.png", "all_curves_a.png", "The 18 curves returned by trace_features, part 1 of 2. " + EX, 1.0),
        ("all_curves_b.png", "all_curves_b.png", "The 18 curves returned by trace_features, part 2 of 2. " + EX, 1.0),
    ],
    "redundancy": [
        ("feature_correlation.png", "feature_correlation.png", "Correlation between the features across the 19 healthy traces of the 2026-10-08 session (features that are NaN in any trace, or constant, are left out), grouped by feature group, in registry order within a group (the bands show the groups and how many features each has). The strongest individual pairs are listed in the table below.", 1.0),
    ],
    "separation": [
        ("feature_separation.png", "feature_separation.png", "Median separation over the 6 channel pairs (difference of the means divided by the pooled repeat scatter), coloured by group; dashed line: 3.", 1.0),
    ],
}


#: Figures drawn half the text width wide (3.4 in) or 0.62 of it; all others are full width (6.9 in).
NATURAL_WIDTH = {name: 0.5 for name in ("level_pixel_levels.png", "level_charge_loss.png", "noise_autocorrelation.png",
                                         "noise_allan.png", "noise_cds_spectrum.png", "noise_window_length.png",
                                         "discrete_adc_codes.png", "slow_drift.png")}
NATURAL_WIDTH["edge_equivalent_time.png"] = 0.62


def figures(group: str) -> str:
    blocks = []
    for src, name, caption, width in FIGS.get(group, []):
        if (FIGDIR / src).resolve() != (OUT / "figures" / name).resolve():
            shutil.copyfile(FIGDIR / src, OUT / "figures" / name)
        place = "p" if name in ("feature_separation.png", "feature_correlation.png") else "htbp"
        width = NATURAL_WIDTH.get(name, 1.0)              # the width the figure was drawn for
        blocks.append(f"\\begin{{figure}}[{place}]\n\\centering\n"
                      f"\\includegraphics[width={width}\\textwidth,height=0.78\\textheight,keepaspectratio]{{{name}}}\n"
                      f"\\caption{{{tex(caption)}}}\n\\label{{fig:{name[:-4]}}}\n\\end{{figure}}")
    return "\n\n".join(blocks) + ("\n\n\\clearpage\n" if blocks else "")


(OUT / "figures").mkdir(parents=True, exist_ok=True)


def draw_figures() -> None:
    """The curve figures (18 curves) and the separation and correlation figures (all scalars), at document size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_feature_figures as pf                      # rcParams, the example trace, save()
    e = pf.Example(pf.EXAMPLE)
    curves = e.features.curves
    order = [n for n, f in FEATURES.items() if f.kind == "curve" and n in curves]
    xlabels = {"skip: mean shape": "sample", "skip: baseline droop curve": "skip", "horizontal: mean shape": "sample",
               "vertical: mean shape": "sample", "level: common CDS pattern": "skip", "noise: CDS autocorrelation": "lag (skips)",
               "noise: Allan deviation": "skips averaged", "noise: CDS spectrum": "kHz", "noise: within-cycle spectrum": "MHz",
               "noise: noise vs window length": "window length (samples)", "edge: equivalent-time pulse": "samples from the edge",
               "edge: variation along the cycle": "sample in the cycle", "noise: Allan deviation, common pattern removed": "skips averaged",
               "noise: CDS spectrum, log frequency": "log-frequency bin (lowest frequency to Nyquist)",
               "start: mean transient": "skip", "noise: Hadamard deviation": "skips averaged",
               "noise: modified Allan deviation": "skips averaged", "noise: within-cycle structure function": "lag (15-sample blocks)"}
    loglog_n = ("noise: Allan deviation", "noise: Allan deviation, common pattern removed", "noise: Hadamard deviation",
                "noise: modified Allan deviation")
    half = (len(order) + 1) // 2
    for part, chunk in (("a", order[:half]), ("b", order[half:])):
        rows = (len(chunk) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(pf.FULL, 1.8 * rows))
        for ax, name in zip(axes.ravel(), chunk):
            c = np.asarray(curves[name], float)
            if name in loglog_n:
                ns = np.asarray(ALLAN_N[:len(c)], float)
                ok = np.isfinite(c) & (c > 0)
                ax.loglog(ns[ok], c[ok], "o-", ms=2.5)
                if ok.any():
                    ax.loglog(ns, c[ok][0] * np.sqrt(ns[ok][0] / ns), "k:", lw=0.7)      # white noise
            elif name == "noise: CDS spectrum, log frequency":
                ax.semilogy(c, "o-", ms=2, lw=0.7)
            elif name == "noise: CDS spectrum":
                ax.semilogy(np.linspace(0, e.skip_rate / 2e3, len(c)), c, lw=0.7)
            elif name == "noise: within-cycle spectrum":
                ax.loglog(np.fft.rfftfreq(e.sig_end, 1 / pf.RATE)[1:len(c)] / 1e6, c[1:], lw=0.7)
            elif name == "noise: noise vs window length":
                ax.loglog(e.z["window_lengths"][:len(c)], c, "o-", ms=2.5)
            elif name == "noise: CDS autocorrelation":
                ax.bar(np.arange(1, len(c) + 1), c, width=0.8)
            elif name == "edge: equivalent-time pulse":
                ax.plot(np.arange(-15, 60, 0.1)[:len(c)], c, lw=0.8)
            elif name == "edge: variation along the cycle":
                ax.semilogy(c, lw=0.7)
            elif name in ("start: mean transient", "noise: within-cycle structure function"):
                ax.plot(np.arange(len(c)) + (1 if "structure" in name else 0), c, "o-", ms=2.5, lw=0.8)
            else:
                ax.plot(c, lw=0.7)
            ax.set(title=name, xlabel=xlabels.get(name, ""))
        for ax in axes.ravel()[len(chunk):]:
            ax.set_visible(False)
        fig.tight_layout()
        pf.save(fig, OUT / "figures", f"all_curves_{part}.png")

    colors = dict(pf.GROUP_COLORS, start="#f4a261", clock="#023e8a")
    # separation: two columns of horizontal bars
    order_s = list(np.argsort(-np.nan_to_num(med, nan=-1)))
    h = (len(order_s) + 1) // 2
    fig, axes = plt.subplots(1, 2, figsize=(pf.FULL, 9.0))
    top = np.nanmax(med) * 2
    for ax, part in zip(axes, (order_s[:h], order_s[h:])):
        part = part[::-1]
        vals = [med[i] if np.isfinite(med[i]) else 0 for i in part]
        ax.barh(range(len(part)), np.maximum(vals, 1e-2), color=[colors[FEATURES[names[i]].group] for i in part])
        ax.set_yticks(range(len(part)))
        ax.set_yticklabels([names[i] for i in part], fontsize=4.3)
        ax.set_xscale("log")
        ax.axvline(3, color="k", ls="--", lw=0.7)
        ax.set_xlabel("median separation over the 6 amplifier pairs")
        ax.set_ylim(-0.6, len(part) - 0.4)
        ax.set_xlim(1e-2, top)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, list(colors), ncol=11, loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=6)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    pf.save(fig, OUT / "figures", "feature_separation.png")

    keep = X.std(0) > 0
    group_order = ["fit", "skip", "horizontal", "vertical", "level", "noise", "discrete", "edge", "slow", "start", "clock"]
    idx = sorted(np.flatnonzero(keep), key=lambda i: (group_order.index(FEATURES[defined[i]].group), i))   # by group
    dn = [defined[i] for i in idx]
    Rk = np.corrcoef(X[:, idx].T)
    groups = [FEATURES[n].group for n in dn]
    fig, ax = plt.subplots(figsize=(pf.FULL, 6.4))
    im = ax.imshow(Rk, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    bounds = [0] + [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]] + [len(groups)]
    centers, labels = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        ax.axhline(a - 0.5, color="k", lw=0.4)
        ax.axvline(a - 0.5, color="k", lw=0.4)
        centers.append((a + b - 1) / 2)
        labels.append(f"{groups[a]} ({b - a})")
        ax.add_patch(plt.Rectangle((-6.5, a - 0.5), 4, b - a, color=colors[groups[a]], clip_on=False))
        ax.add_patch(plt.Rectangle((a - 0.5, len(groups) + 1.5), b - a, 4, color=colors[groups[a]], clip_on=False))
    ax.set_yticks(centers)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.tick_params(axis="y", pad=30, length=0)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, fontsize=5.5, rotation=60, ha="right")
    ax.tick_params(axis="x", pad=32, length=0)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label=f"correlation over the {n_traces} traces")
    fig.tight_layout()
    pf.save(fig, OUT / "figures", "feature_correlation.png")


draw_figures()

# --------------------------------------------------------------------------- the document
GROUP_TEXT = {
    "fit": ("Comparison tools: parametric fits",
            "Exponentials are fitted by variable projection: the amplitude and offset are solved by least squares for every "
            "time constant on a logarithmic grid, and the best time constant is refined on a finer grid. The damped ringing and "
            "the noise model are fitted the same way. Computed per usable pixel; the feature is the median over pixels."),
    "skip": ("Comparison tools: skip-cycle tools",
             "Computed per usable pixel from the 500 skip cycles (one row of 489 samples per skip) and from the CDS sequence; "
             "the feature is the median over pixels."),
    "horizontal": ("Comparison tools: horizontal transfers",
                   "From the 19 complete horizontal transfers before each pixel, baseline (first 30 samples) removed; median over pixels."),
    "vertical": ("Comparison tools: vertical transfers",
                 "From the vertical transfers of the first row change; one replicate per transfer (median over transfers). "
                 "Consecutive vertical transfers are not exchangeable, so these were information only in the comparison test. "
                 "NaN when the vertical transfers are not found (module 102, Vtransfer21)."),
    "level": ("Advanced features: CDS level",
              "The CDS level of a pixel is its charge plus an offset from the electronics. With 20$\\times$20 binning each trace "
              "pixel collects the charge of 400 CCD pixels. Two features were added on 2026-10-21 (tier 1 and 2)."),
    "noise": ("Advanced features: noise character",
              "These describe how the noise behaves (correlation, averaging, spectrum, window length, settling, coupling), "
              "not only its size. The 2026-10-21 additions (tier 1 and 2, after the original 15) add the null CDS (a CDS between "
              "the two halves of the pedestal window, where no charge moves), a jump-robust sigma, the 1/f and Lorentzian "
              "corners, drift-insensitive Allan variants, the waveform-mode noise budget and several noise-shape statistics."),
    "discrete": ("Advanced features: discrete and transient noise",
                 "Steps, heavy tails, interference lines and ADC code non-uniformity. The 2026-10-21 additions detect telegraph "
                 "steps inside single windows, split the CDS variance into a smooth part and jumps, give calibrated jump counts, "
                 "and find pickup lines with a stated false-alarm rate (multitaper F-test)."),
    "edge": ("Advanced features: edges",
             "From the reset pulse and the summing-well dip: timing, sub-sample shape, recovery, coupling per volt, and the "
             "skip-to-skip variation along the cycle. Added 2026-10-21: the sharp dip about 120--130 ns before the reset pulse, "
             "measured on the equivalent-time pulse."),
    "slow": ("Advanced features: slow structure", "Row-start recovery, drift across the trace, horizontal-transfer amplitude and principal components."),
    "start": ("Start of the pixel (added 2026-10-21)",
              "The first skips of every pixel read a decaying baseline: the CDS of skip $k$ is offset by the baseline slope inside "
              "the windows times the separation of the window centres (about $-207$ samples), so the first-skip offset is a "
              "baseline slope, not charge. Its size follows the horizontal-clock swing (H-clock feedthrough). Computed per usable "
              "pixel from its first skips: the slope regression uses skips 0--19, the mean transient skips 0--29; the window slopes are "
              "compared with their median over skips 20--59 and the CDS with its median from skip 10 on. Per-pixel features "
              "are the median over the usable pixels."),
    "clock": ("Clocking regions (added 2026-10-21)",
              "From the vertical-transfer block before the second row (its first transfer, which follows the skip region, left "
              "out), the horizontal transfers before each pixel, and the block the trace starts in. The trace starts at a random "
              "point inside its first vertical block (14--19 of 20 transfers recorded on 2026-10-12), so the slope of that block "
              "is stored (tier 2) but not repeatable."),
}


def feature_table(group: str) -> str:
    rows = [(n, f) for n, f in FEATURES.items() if f.group == group and f.kind == "scalar"]
    lines = [r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.235\textwidth}>{\raggedright\arraybackslash}p{0.075\textwidth}"
             r">{\raggedright\arraybackslash}p{0.335\textwidth}>{\raggedright\arraybackslash}p{0.15\textwidth}"
             r">{\raggedright\arraybackslash}p{0.07\textwidth}}",
             r"\toprule Feature & Unit & What is computed & Possible link & Tier \\ \midrule \endhead",
             r"\bottomrule \endfoot"]
    for n, f in rows:
        lines.append(rf"\texttt{{{tex(n)}}} & {unit(f.unit)} & {tex(f.description)} & {tex(f.link)} & {tier_cell(n)} \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def curve_table() -> str:
    lines = [r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.29\textwidth}>{\raggedright\arraybackslash}p{0.09\textwidth}"
             r">{\raggedright\arraybackslash}p{0.50\textwidth}}",
             r"\toprule Curve & Unit & Content \\ \midrule \endhead", r"\bottomrule \endfoot"]
    for n, f in FEATURES.items():
        if f.kind == "curve":
            lines.append(rf"\texttt{{{tex(n)}}} & {tex(f.unit)} & {tex(f.description)}"
                         + (f" (Tier {f.tier}.)" if f.tier else "") + r" \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def separation_table() -> str:
    order = np.argsort(-np.nan_to_num(med, nan=-1))
    lines = [r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.55\textwidth}rr}",
             r"\toprule Feature & Pairs $\geq 3$ & Median separation \\ \midrule \endhead", r"\bottomrule \endfoot"]
    for i in order:
        value = "--" if not np.isfinite(med[i]) else f"{med[i]:.1f}"
        lines.append(rf"\texttt{{{tex(names[i])}}} & {n_sep[i]} & {value} \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def redundancy_table(top=30) -> str:
    lines = [r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.40\textwidth}>{\raggedright\arraybackslash}p{0.40\textwidth}r}",
             r"\toprule Feature & Feature & $r$ \\ \midrule \endhead", r"\bottomrule \endfoot"]
    for a, b, r_ in pairs[:top]:
        lines.append(rf"\texttt{{{tex(a)}}} & \texttt{{{tex(b)}}} & {r_:+.3f} \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


counts = {g: sum(1 for f in FEATURES.values() if f.group == g and f.kind == "scalar") for g in GROUP_TEXT}
n_scalar = sum(counts.values())
n_curve = sum(1 for f in FEATURES.values() if f.kind == "curve")
n_all6 = int(np.sum(n_sep == 6)); n_none = int(np.sum(n_sep == 0))
assert n_scalar == len(scalar_names()), (n_scalar, len(scalar_names()))
GROUPS = list(GROUP_TEXT)
by_tier = {g: [sum(1 for f in FEATURES.values() if f.group == g and f.kind == "scalar" and f.tier == t) for t in (0, 1, 2)] for g in GROUPS}
curves_by_tier = [sum(1 for f in FEATURES.values() if f.kind == "curve" and f.tier == t) for t in (0, 1, 2)]
ev_counts = {t: {i: sum(1 for n in TIERS.get(t, []) if FEATURES[n].tier == i) for i in (0, 1, 2)} for t in ("A1", "A2", "B", "C", "P")}
assert sum(len(TIERS[t]) for t in ("A1", "A2", "B", "C")) == n_scalar
NEW_SCALAR = [n for n in NEW if FEATURES[n].kind == "scalar"]
NEW_CURVE = [n for n in NEW if FEATURES[n].kind == "curve"]


def origin_table() -> str:
    lines = [r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.40\textwidth}c>{\raggedright\arraybackslash}p{0.42\textwidth}}",
             r"\toprule Feature (2026-10-21 addition) & Tier & Where the idea came from \\ \midrule \endhead", r"\bottomrule \endfoot"]
    for n in NEW:
        lines.append(rf"\texttt{{{tex(n)}}} & {FEATURES[n].tier} & {tex(NEW_ORIGIN[n])} \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def model_tier_rows() -> str:
    rows = [rf"\code{{{tex(m)}}} & \code{{{tex(v)}}} \\" for m, v in MODEL_TIERS.items()]
    rows.append(rf"all other models & \code{{{tex(DEFAULT_MODEL_TIERS)}}} \\")
    return "\n".join(rows)

doc = r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=0.8in]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{graphicx}
\usepackage{placeins}
\usepackage{hyperref}
\graphicspath{{figures/}{overleaf_trace_features/figures/}}

\title{Trace Features for Machine Learning:\\ the Complete List (\texttt{trace\_features.py})}
\author{ML-assisted optimization of CCD operating parameters --- working note}
\date{9 October 2026; updated 22 October 2026}
\newcommand{\code}[1]{\texttt{#1}}
% long feature names and paths in typewriter type: allow line breaks after underscores and slashes
\let\oldunderscore\_
\renewcommand{\_}{\oldunderscore\allowbreak}
\setlength{\emergencystretch}{3em}

\begin{document}
\maketitle

\begin{abstract}
This note lists every feature computed from one raw skipper-CCD trace by the function \code{trace\_features} in
\code{src/trace\_features.py}: """ + str(n_scalar) + r""" numbers and """ + str(n_curve) + r""" curves. The original 85 numbers and 14 curves
(implementation tier 0) merge the 39 comparison tools of \code{compare\_trace.py} with the advanced features developed in
\code{notebooks/advanced\_trace\_features.ipynb}; """ + str(len(NEW_SCALAR)) + r""" numbers and """ + str(len(NEW_CURVE)) + r""" curves
(tiers 1 and 2, \code{src/trace\_features\_extra.py}) were added on 2026-10-21 from a literature search across 21 fields, the
study of the first-skip offset, and inspection of the raw trace. The features are candidate inputs for a model that relates traces and
operating parameters to the read noise $\sigma_e$, the dark current $\lambda$ and the gain of calibrated images. They describe
the trace; they are not calibrated physical quantities. For each feature the note gives what is computed and why a model
might find it informative, its implementation tier and its evidence tier (Section~\ref{sec:tiers}). The last sections show how
well each feature separates four amplifiers compared with its repeat-to-repeat scatter on the 2026-10-08 traces, and which
features are nearly redundant. Exact definitions are in the technical note (\code{docs/overleaf\_trace\_features\_technical});
the results of the new features on real traces are in \code{docs/overleaf\_feature\_tiers\_2026-10-21}.
\end{abstract}

\tableofcontents

\section{The trace and the conventions}

\paragraph{The trace.} A trace is an ordinary \code{AcquireImage} readout of a small image (2 rows $\times$ 5 columns,
20$\times$20 binning, 500 skips, no exposure, no cuts) with one channel streaming every ADC sample at 15\,MHz. It contains, in time
order, the vertical transfers of a row, then for each pixel the horizontal transfers and the 500 skip cycles. One skip cycle
lasts 488.7 samples (3258 sequencer ticks; cut to 489 whole samples). Inside a cycle:
\begin{center}
\begin{tabular}{ll}
\toprule
Samples & Content \\
\midrule
0--150 & pedestal window (RD flag) \\
150--207 & summing-well clock: the \emph{dip} \\
207--357 & signal window (RU flag) \\
357--$\sim$420 & output gate and reset: the \emph{reset pulse} (peak near 374) \\
$\sim$420--489 & reset settling \\
\bottomrule
\end{tabular}
\end{center}
The CDS of a skip is the mean of the pedestal window minus the mean of the signal window (from the firmware's inline sums).

\paragraph{Any geometry and timing.} The function works for any trace geometry (rows, columns, number of
skips, binning) and any clock timing. Pixels are found from the order of the pedestal and signal windows (a pixel ends with
its extra pedestal window; the 4-bit pixel number in the inline records repeats after 16 pixels and is not used), the skip
cycle from the spacing of the pedestal windows, horizontal and vertical transfers from the sequencer's clock flags (bits 6
and 7), and row starts from \code{NCOL} in the \code{.meta} (or from the vertical transfers without it). The \code{.meta}
geometry (\code{trace\_geometry.py}) is used to cross-check what is found. Every count used by a feature scales with the
trace (for example the first 10\,\% of the skips instead of skips 10--59, half the window length instead of 75 samples, the
middle skip instead of skip 250); for the usual traces (2$\times$5 pixels, 500 skips, 150-sample windows) the counts are
the values given in the tables. A feature that cannot exist for a trace (e.g.\ the noise of 50-skip averages with 100
skips, the row-start recovery with one column) is NaN. This was tested on 25 synthetic geometries and timings
(\code{simulate\_trace.py}, \code{check\_trace\_analysis.py}) and on the real traces.

\paragraph{Pixels used.} Features are computed on the \emph{usable} pixels: all pixels except the row-start pixels (the first
pixel of each row, still recovering from the vertical transfers) and pixels holding a large charge (CDS mean beyond 8 robust
standard deviations). For per-pixel features the reported value is the median over the usable pixels, and the robust spread
over pixels ($1.4826\times$ median absolute deviation) is reported alongside.

\paragraph{Pre-processing used by many features.}
\begin{itemize}
  \item \emph{Detrended CDS}: the CDS of skips 10--499 minus a degree-4 polynomial in the skip index. This removes the slow change
        along the pixel without touching the skip-to-skip noise (a running mean must not be used: it cancels block averages).
  \item \emph{Residual cycles}: every cycle minus a degree-4 polynomial across skips at the same sample position. This removes the
        mean waveform and its slow change, leaving only what varies from skip to skip.
  \item \emph{Common pattern}: the median CDS of the usable pixels, skip by skip. It is the same in every pixel and therefore not charge.
  \item \emph{Edge times}: the time at which the reset pulse crosses half its height, interpolated between samples; fitted with a
        straight line against the skip number.
  \item \emph{Equivalent-time sampling}: because the cycle is 488.7 samples long but always starts on a whole sample, the sampling
        phase walks by about a third of a sample every skip. Placing each sample at its exact time relative to the edge line and
        averaging in 0.1-sample bins rebuilds the pulse at sub-sample resolution.
\end{itemize}

\paragraph{Missing values.} A feature is NaN when it cannot exist for a trace: the charge-loss features without a charged pixel,
the vertical tools when the vertical transfers are not found (module 102), the per-volt feedthrough without the \code{.meta}
file. A trace with lost data (the decoder's quality check) is refused.

\paragraph{Usage.}
\begin{verbatim}
from trace_features import trace_features, FEATURES
result = trace_features("trace_...ch0_...RJ45ADCDATA0")
result.scalars      # one number per feature
result.spread       # pixel-to-pixel spread (per-pixel features)
result.curves       # fixed-length vectors
FEATURES[name]      # unit, description, possible link, tier
\end{verbatim}
\code{process\_packet.py} calls the function for every trace of a packet and writes, per packet, board and channel, the mean
of each feature over the 3 trace sets at $\theta$ (\code{trace.*}) and at the reference (\code{ref.*}), their scatter between sets
(\code{trace\_sd.*}, \code{ref\_sd.*}) and the difference (\code{diff.*}). The curves are stored in each trace's \code{.npz}.
The earlier files (\code{compare\_trace.py}, the comparison and advanced-features notebooks) are unchanged.

\begin{center}
\begin{tabular}{llrrrr}
\toprule
Origin & Group & Tier 0 & Tier 1 & Tier 2 & Numbers \\
\midrule
""" + "\n".join(rf"{'comparison' if g in ('fit', 'skip', 'horizontal', 'vertical') else ('advanced' if g not in ('start', 'clock') else 'new (2026-10-21)')} & \code{{{g}}} & {by_tier[g][0]} & {by_tier[g][1]} & {by_tier[g][2]} & {counts[g]} \\" for g in GROUP_TEXT) + r"""
\midrule
Total & & """ + " & ".join(str(sum(by_tier[g][i] for g in GROUPS)) for i in range(3)) + " & " + str(n_scalar) + r""" \\
Curves & & """ + " & ".join(str(c) for c in curves_by_tier) + " & " + str(n_curve) + r""" \\
\bottomrule
\end{tabular}
\end{center}
In the feature tables the last column gives the implementation tier and the evidence tier (\emph{e.g.}\ \code{1/A1}), and
\code{P} marks the protected estimators (Section~\ref{sec:tiers}).

\section{Tiers}
\label{sec:tiers}

All features are computed and stored for every trace; the tiers only decide which features a model sees.

\paragraph{Implementation tier} (\code{FEATURES[name].tier}): 0 = the original 85 numbers and 14 curves; 1 = additions of
2026-10-21 to implement first (physics-motivated, piloted on real traces); 2 = candidate additions. Section~\ref{sec:origin}
lists where each addition came from.

{\raggedright\paragraph{Evidence tier} (\code{lvl1/feature\_study.py} on the 59 usable board-101 traces of 2026-10-12, stored in
\code{lvl1/feature\_study/evidence\_tiers.json}). It uses no image labels: settings $F$ (ANOVA over the 4 changed settings), amplifier
$F$, repeat share (within-cell variance over total), missing share and redundancy clusters ($|\rho_\text{Spearman}|>0.95$).\par}
\begin{itemize}
  \item \textbf{A} (core): settings $F$ min $>10$ or amplifier $F>10$, repeat share $<0.5$, at most 20\,\% missing, and the most
        informative member of its cluster; split by informativeness into two halves, \textbf{A1} and \textbf{A2}.
  \item \textbf{B} (supporting): informative but redundant, or moderately informative ($F$ 4--10).
  \item \textbf{C} (store only): not informative on these data, unstable, mostly missing or constant.
  \item \textbf{P} (protected, besides the evidence tier): direct physical estimators of $\sigma_e$ and the gain, given to every
        model, because the label-free tiers can demote them (on these traces the amplifier dominates every correlation).
\end{itemize}
\begin{center}
\begin{tabular}{lrrrr}
\toprule
Evidence tier & tier 0 & tier 1 & tier 2 & total \\
\midrule
""" + "\n".join(rf"{t} & {ev_counts[t][0]} & {ev_counts[t][1]} & {ev_counts[t][2]} & {len(TIERS.get(t, []))} \\" for t in ("A1", "A2", "B", "C", "P")) + r"""
\bottomrule
\end{tabular}
\end{center}
{\raggedright The protected estimators: """ + ", ".join(rf"\code{{{tex(n)}}}" for n in PROTECTED) + r""". No direct estimator of $\lambda$
exists in a trace yet.\par} The evidence tiers are provisional: these four settings did not change the noise, so noise features were
judged by amplifier separation alone; they are to be rerun on the first scan session.

\paragraph{Learned tiers} (\code{lvl1/feature\_importance.py}, needs labels): L1 = stability selection and permutation importance
both beat row-shuffled shadow features by more than 1.5$\times$; L2 = a candidate; L3 = not shown to matter. They are diagnostics
(which features to look at physically); they do not choose any model's features.

\paragraph{Model feature sets} (\code{MODEL\_TIERS} in \code{lvl1/train\_lvl1.py}, used by \code{--feature-tiers auto}):
\begin{center}
\begin{tabular}{ll}
\toprule
Model & Default feature set \\
\midrule
""" + model_tier_rows() + r"""
\bottomrule
\end{tabular}
\end{center}
They come from a synthetic tolerance test and the synthetic feature ladder (the plain Gaussian process collapses beyond 15--25
features). \code{--tier-ablation} and \code{--feature-ladder} train every model on several sets; on real data the ladder decides.

\section{The 2026-10-21 additions: where they came from}
\label{sec:origin}

{\raggedright """ + str(len(NEW_SCALAR)) + r""" numbers and """ + str(len(NEW_CURVE)) + r""" curves were added from a literature search for hand-computable
features across 21 fields (\code{docs/literature/signal\_analysis\_features.md}; candidate list and pilots:
\code{docs/trace\_feature\_candidates.md}), the study of the first-skip offset, and a look at the raw trace (the clocking regions
and the dip before the reset pulse). Not implemented: hidden-Markov dwell times, step rise times, jitter decomposition, spectral
flux and cepstrum; heart-rate-variability measures (exact functions of the lag-1 autocorrelation).\par}

""" + origin_table() + r"""

"""

for group, (title, intro) in GROUP_TEXT.items():
    doc += rf"""\section{{{title}}}
{intro}

{feature_table(group)}

{figures(group)}
"""

doc += r"""\section{Key definitions}

\paragraph{Allan deviation.} For $n$ skips, with $\bar{x}_j$ the means of consecutive blocks of $n$ detrended CDS values,
\begin{equation}
\sigma_A(n) = \sqrt{\tfrac{1}{2}\,\langle(\bar{x}_{j+1}-\bar{x}_j)^2\rangle}.
\end{equation}
White noise gives $\sigma_A\propto n^{-1/2}$; anticorrelated skips fall faster, $1/f$ noise flattens.

\paragraph{Reset-level scatter.} With $r_{k,i}$ the residual cycles (skip $k$, sample $i$), the level set by reset $k$ is
$\ell_k=\frac{1}{150}\sum_{i<150} r_{k,i}$; the feature is the standard deviation of $\ell_k$ over skips. The CDS cancels it
because both windows of one cycle share $\ell_k$ (in the example trace: about 146\,ADU against a CDS noise of 5.6\,ADU).

\paragraph{Within-cycle band powers.} From sample 0 to the end of the signal window (no reset inside), each cycle's own mean is
removed and the Hann-windowed power spectrum is averaged over skips and pixels; the features are the fractions of the power
above 0.1\,MHz in four bands.

\paragraph{Window-length ratio.} The CDS is recomputed from the raw samples with windows of $L$ samples,
$c_k(L)=\frac{1}{L}\sum_{i<L}y_{k,i}-\frac{1}{L}\sum_{i<L}y_{k,207+i}$, and its detrended standard deviation $\sigma(L)$ is
computed; the feature is $\sigma(75)/\sigma(150)$ (white noise: $\sqrt{2}$).

\paragraph{Feedthrough per volt.} Dip height divided by $\mathrm{SW\_1\_H}-\mathrm{SW\_1\_L}$, and reset-pulse height divided by
$\mathrm{RG\_1\_H}-\mathrm{RG\_1\_L}$; their ratio cancels the common amplification. The channel-to-amplifier wiring (SW\_1/RG\_1
for channel 0) is an assumption to confirm from the bcf.

\paragraph{Null CDS.} With the residual cycles $r_{k,i}$ and $h$ half the pedestal window, $\nu_k=\frac{1}{h}\sum_{i<h}r_{k,i}-
\frac{1}{h}\sum_{h\le i<2h}r_{k,i}$: a CDS between the two halves of the pedestal window, where no charge moves. Its standard
deviation, kurtosis and lag-1 autocorrelation are features; telegraph switching in the output stage makes it heavy-tailed.

\paragraph{Jump-robust sigma and jump share.} With $e_k$ the detrended CDS ($N$ skips), the median realized variance
$\mathrm{MedRV}=\frac{\pi}{6-4\sqrt3+\pi}\frac{N}{N-2}\sum_k \mathrm{med}(|e_{k-1}|,|e_k|,|e_{k+1}|)^2$ (Andersen, Dobrev and
Schaumburg 2012) ignores isolated jumps; the jump-robust sigma is $\sqrt{\mathrm{MedRV}/N}$ and the jump share
$1-\mathrm{MedRV}/\sum_k e_k^2$ (clipped at 0).

\paragraph{In-window steps.} Every pedestal and signal window is cut into 10-sample blocks; the best single step between block
means is expressed in units of its standard error (the block-mean noise from the within-block scatter, median over all
windows). A window with $t>6$ has a step; the step rate is the percentage of such windows and the step amplitude the median
$|\text{step}|$.

\paragraph{Hadamard deviation.} With $\bar{x}_j$ the means of consecutive blocks of $n$ CDS values (common pixel pattern
removed, not detrended),
$\sigma_H(n)=\sqrt{\tfrac16\langle(\bar{x}_{j+2}-2\bar{x}_{j+1}+\bar{x}_j)^2\rangle}$: like the Allan deviation, but a linear drift
cancels.

\paragraph{Noise budget.} The residual cycles of all usable pixels are decomposed into waveform modes (singular value
decomposition, $R-\bar R=U\,\mathrm{diag}(s)\,V^T$). With $w$ the CDS weights ($+1/150$ on the pedestal window, $-1/150$ on the
signal window), mode $m$ contributes $s_m^2 (V_m\cdot w)^2$ to the CDS variance; the features are the share of the three modes
with the most waveform variance and the number of modes (in that order) needed for 90\,\%.

\paragraph{Matched-filter transient amplitude.} With $T_k$ the mean over the usable pixels of the first 30 CDS values minus the median CDS from skip 10 on
(the curve \code{start: mean transient}), each pixel's amplitude is $\langle c, T\rangle/\langle T,T\rangle\cdot\max_k|T_k|$, the
radar matched filter: less noisy than the first-skip excess alone.

\paragraph{Pre-pulse dip.} On the equivalent-time pulse (0.1-sample bins relative to the edge), the minimum between 6 and 0
samples before the edge minus the mean 15 to 8 samples before; divided by the pulse height above the same baseline for the ratio.

\section{Curves}
""" + curve_table() + "\n\n" + figures("curves") + r"""

\section{Which features separate amplifiers better than repeats}

On the 2026-10-08 session (module 101, """ + str(n_traces) + r""" healthy traces: 4 channels $\times$ 5 repeats, minus one
damaged trace), every feature was computed and, for each of the 6 pairs of channels,
\begin{equation}
\text{separation} = \frac{|\overline{f}_A-\overline{f}_B|}{s_\text{pooled}},
\end{equation}
where $s_\text{pooled}$ is the pooled standard deviation of the repeats. A separation of at least 3 means the feature tells two
amplifiers apart well beyond its repeat scatter: the precondition for seeing a change of the operating parameters. Features with a
zero repeat scatter would give an undefined separation (shown as --); a pair in which both amplifiers have zero scatter is left
out of the median (e.g.\ the in-window step rate, which is exactly 0 on every amplifier but ch2).
""" + f"{n_all6} features separate all 6 pairs and {n_none} separate none." + r"""
A low separation does not make a feature useless: a feature that is the same on every amplifier (e.g.\ the baseline coupling) is
a clean parameter signal if it moves with $\theta$. Only a scan with varied parameters can show which features respond to $\theta$
and to the image labels.

""" + figures("separation") + "\n" + separation_table() + r"""

\section{Nearly redundant features}

Pairs of features whose values are correlated with $|r|>0.98$ across the same """ + str(n_traces) + r""" traces (""" + str(len(pairs)) + r""" pairs;
the 30 strongest are listed). With 19 traces from 4 amplifiers, such correlations mostly mean that both features follow the same
amplifier difference; they suggest redundancy, but only a parameter scan shows whether they also move together with $\theta$.
Some pairs are redundant by construction (e.g.\ the signal-window level and the CDS mean; the dip height and the dip per volt at a
fixed swing).

""" + figures("redundancy") + "\n" + redundancy_table() + r"""

\section{Where it lives}
\begin{center}
\begin{tabular}{ll}
\toprule
Item & File \\
\midrule
Feature function and registry & \code{src/trace\_features.py} \\
The 2026-10-21 additions & \code{src/trace\_features\_extra.py} (registry \code{EXTRA}) \\
Tests of the additions & \code{tests/test\_trace\_features\_new.py} \\
Evidence tiers & \code{lvl1/feature\_study.py} $\to$ \code{lvl1/feature\_study/} \\
Learned tiers & \code{lvl1/feature\_importance.py} \\
Feature sets in training & \code{lvl1/train\_lvl1.py} (\code{--feature-tiers}, \code{--feature-ladder}) \\
Walkthrough of the additions & \code{lvl1/new\_trace\_features.ipynb} \\
Results of the additions & \code{docs/overleaf\_feature\_tiers\_2026-10-21} \\
Exact definitions & \code{docs/overleaf\_trace\_features\_technical} \\
Comparison tools (unchanged) & \code{src/compare\_trace.py} \\
Use in the pipeline & \code{src/process\_packet.py} (\code{process\_trace}) \\
Walkthrough of the merged set & \code{notebooks/trace\_features\_walkthrough.ipynb} \\
Development of the advanced features & \code{notebooks/advanced\_trace\_features.ipynb} \\
Description of the comparison tools & \code{docs/trace\_fit\_tools.tex} \\
\bottomrule
\end{tabular}
\end{center}

\end{document}
"""

# basic consistency checks (no LaTeX installed to compile)
assert doc.count("{") == doc.count("}") + doc.count("\\{") - doc.count("\\}"), "brace mismatch"
for env in set(re.findall(r"\\begin\{(\w+)\}", doc)):
    assert doc.count(f"\\begin{{{env}}}") == doc.count(f"\\end{{{env}}}"), env
assert not {c for c in doc if ord(c) > 127}, {c for c in doc if ord(c) > 127}
(OUT / "trace_features.tex").write_text(doc)
import zipfile
with zipfile.ZipFile(PROJECT / "docs" / "overleaf_trace_features.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(OUT / "trace_features.tex", "trace_features.tex")
    for f in sorted((OUT / "figures").glob("*.png")):
        z.write(f, f"figures/{f.name}")
print("written", len(doc), "chars;", n_scalar, "features,", n_curve, "curves,", len(pairs), "redundant pairs")
