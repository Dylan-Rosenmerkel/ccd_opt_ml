"""
make_trace_features_note.py -- write docs/trace_features.tex (and its Overleaf copy) from the FEATURES
registry of trace_features.py, the 2026-10-08 separation results and the figures of
plot_feature_figures.py.

    cd src && ../.venv/bin/python plot_feature_figures.py ../docs/overleaf_trace_features/figures
    ../.venv/bin/python make_trace_features_note.py

The figures are drawn at document size (6.9 in = the text width, or 3.4 in = half of it), so they
are included at their natural width and their text stays readable.
"""
import itertools, re, sys
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from trace_features import FEATURES, scalar_names, trace_features

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
        ("all_curves_a.png", "all_curves_a.png", "The 14 curves returned by trace_features, part 1 of 2. " + EX, 1.0),
        ("all_curves_b.png", "all_curves_b.png", "The 14 curves returned by trace_features, part 2 of 2. " + EX, 1.0),
    ],
    "redundancy": [
        ("feature_correlation.png", "feature_correlation.png", "Correlation between the 85 features across the 19 healthy traces of the 2026-10-08 session, ordered as in the registry and grouped (the bands show the feature groups and how many features each has). The strongest individual pairs are listed in the table below.", 1.0),
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
                      f"\\includegraphics[width={width}\\textwidth,height=0.86\\textheight,keepaspectratio]{{{name}}}\n"
                      f"\\caption{{{tex(caption)}}}\n\\label{{fig:{name[:-4]}}}\n\\end{{figure}}")
    return "\n\n".join(blocks) + ("\n\n\\FloatBarrier\n" if blocks else "")


(OUT / "figures").mkdir(parents=True, exist_ok=True)

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
              "pixel collects the charge of 400 CCD pixels."),
    "noise": ("Advanced features: noise character",
              "These describe how the noise behaves (correlation, averaging, spectrum, window length, settling, coupling), "
              "not only its size."),
    "discrete": ("Advanced features: discrete and transient noise", "Steps, heavy tails, interference lines and ADC code non-uniformity."),
    "edge": ("Advanced features: edges",
             "From the reset pulse and the summing-well dip: timing, sub-sample shape, recovery, coupling per volt, and the "
             "skip-to-skip variation along the cycle."),
    "slow": ("Advanced features: slow structure", "Row-start recovery, drift across the trace, horizontal-transfer amplitude and principal components."),
}


def feature_table(group: str) -> str:
    rows = [(n, f) for n, f in FEATURES.items() if f.group == group and f.kind == "scalar"]
    lines = [r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.26\textwidth}>{\raggedright\arraybackslash}p{0.095\textwidth}"
             r">{\raggedright\arraybackslash}p{0.36\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}}",
             r"\toprule Feature & Unit & What is computed & Possible link \\ \midrule \endhead",
             r"\bottomrule \endfoot"]
    for n, f in rows:
        lines.append(rf"\texttt{{{tex(n)}}} & {tex(f.unit)} & {tex(f.description)} & {tex(f.link)} \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def curve_table() -> str:
    lines = [r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.29\textwidth}>{\raggedright\arraybackslash}p{0.09\textwidth}"
             r">{\raggedright\arraybackslash}p{0.50\textwidth}}",
             r"\toprule Curve & Unit & Content \\ \midrule \endhead", r"\bottomrule \endfoot"]
    for n, f in FEATURES.items():
        if f.kind == "curve":
            lines.append(rf"\texttt{{{tex(n)}}} & {tex(f.unit)} & {tex(f.description)} \\")
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
\date{9 October 2026}
\newcommand{\code}[1]{\texttt{#1}}

\begin{document}
\maketitle

\begin{abstract}
This note lists every feature computed from one raw skipper-CCD trace by the function \code{trace\_features} in
\code{src/trace\_features.py}: """ + str(n_scalar) + r""" numbers and """ + str(n_curve) + r""" curves. They merge the 39
comparison tools of \code{compare\_trace.py} with the advanced features developed in
\code{notebooks/advanced\_trace\_features.ipynb}. The features are candidate inputs for a model that relates traces and
operating parameters to the read noise $\sigma_e$, the dark current $\lambda$ and the gain of calibrated images. They describe
the trace; they are not calibrated physical quantities. For each feature the note gives what is computed and why a model
might find it informative. The last sections show how well each feature separates four amplifiers compared with its
repeat-to-repeat scatter on the 2026-10-08 traces, and which features are nearly redundant.
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
FEATURES[name]      # unit, description, possible link
\end{verbatim}
\code{process\_packet.py} calls the function for every trace of a packet and writes, per packet, board and channel, the mean
of each feature over the 3 trace sets at $\theta$ (\code{trace.*}) and at the reference (\code{ref.*}), their scatter between sets
(\code{trace\_sd.*}, \code{ref\_sd.*}) and the difference (\code{diff.*}). The curves are stored in each trace's \code{.npz}.
The earlier files (\code{compare\_trace.py}, the comparison and advanced-features notebooks) are unchanged.

\begin{center}
\begin{tabular}{llr}
\toprule
Origin & Group & Numbers \\
\midrule
""" + "\n".join(rf"{'comparison' if g in ('fit', 'skip', 'horizontal', 'vertical') else 'advanced'} & \code{{{g}}} & {counts[g]} \\" for g in GROUP_TEXT) + r"""
\midrule
Total & & """ + str(n_scalar) + r""" \\
\bottomrule
\end{tabular}
\end{center}

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
zero repeat scatter would give an undefined separation (shown as --).
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
(PROJECT / "docs" / "trace_features.tex").write_text(doc)
(OUT / "trace_features.tex").write_text(doc)
shutil.make_archive(str(PROJECT / "docs" / "overleaf_trace_features"), "zip", root_dir=OUT)
print("written", len(doc), "chars;", n_scalar, "features,", n_curve, "curves,", len(pairs), "redundant pairs")
