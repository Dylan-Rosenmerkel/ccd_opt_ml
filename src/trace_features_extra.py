"""
trace_features_extra.py -- the tier-1 and tier-2 trace features of docs/trace_feature_candidates.md (2026-10-21).

Called by ``trace_features._trace_features`` (which registers ``EXTRA`` in ``trace_features.FEATURES`` with their
tier), on the same shared arrays.  Numbers in brackets (#n) refer to the candidate table of
docs/literature/signal_analysis_features.md, Section 2.

Groups:
* ``start:`` (new) -- the start-of-pixel transient.  Pilot result: the first-skip CDS offset is the CDS reading the
  slope of a decaying baseline (CDS_k - late ~ L * in-window slope, L ~ -(window-centre separation)); skip 0 holds an
  exponential (tau ~ 11-12.5 us) whose size follows the horizontal-clock swing.
* ``noise:`` / ``discrete:`` additions -- telegraph switching inside one window (null CDS, in-window steps),
  jump-robust noise (finance), noise shape (1/f corner, Hadamard, DFA, entropies, ...), pickup lines (multitaper
  F-test and their phase), and the skip-cycle noise budget (which waveform modes reach the CDS).
* ``level:`` additions -- the slope-corrected CDS rise, the coupling of each pixel to the common CDS pattern.
* ``edge:`` additions (2026-10-21) -- the dip just before the per-cycle pulse, from the equivalent-time pulse.
* ``clock:`` (new, 2026-10-21/23) -- the vertical-clock block before row 2 (level slope, peak-to-peak), the H/V
  peak-to-peak ratio, the settling at the start of each row's first H burst, and the (stored only) slope of the block
  the trace starts in; all but the level slope work at the 1x1 binning of the matched traces.

Every function is numpy/scipy only and returns NaN where the trace is too short.
"""

from __future__ import annotations

import math

import numpy as np

# (name, group, unit, description, link, kind, tier)
EXTRA = [
    # ------------------------------------------------------------------ start of pixel (tier 1)
    ("start: CDS-slope lever arm (samples)", "start", "samples",
     "Slope of the regression of CDS_k - median(CDS of later skips) on the in-window baseline slope excess Delta_k - late (Delta = "
     "mean of the pedestal- and signal-window slopes; late = median of skips 20-59), skips 0-19, pooled over the usable pixels (#2).",
     "about -(window-centre separation) if the first-skip pattern is pure baseline slope; a departure means a charge or gain effect", "scalar", 1),
    ("start: slope-explained fraction", "start", "", "R^2 of that regression (#3).", "how much of the first-skip pattern is baseline slope", "scalar", 1),
    ("start: in-cycle slope at skip 0 (ADU/us)", "start", "ADU/µs", "Delta_0 - late, in ADU per microsecond (#1).", "derivative of the start transient", "scalar", 1),
    ("start: transient time constant (us)", "start", "µs",
     "Time constant of A exp(-t/tau) + c fitted to the skip-0 waveform minus the late mean waveform (each skip minus its pedestal "
     "mean), on the pedestal and signal windows only (#4).", "recovery of the video chain / output stage after the H-clock burst", "scalar", 1),
    ("start: transient amplitude (ADU)", "start", "ADU", "Amplitude A of that fit at the start of the cycle (#5).", "size of the H-clock disturbance", "scalar", 1),
    ("start: first-skip excess per volt H (ADU/V)", "start", "ADU/V", "skip: first-skip CDS excess divided by the H1_A clock swing (H1_A_H - H1_A_L, .meta) (#6).",
     "H-clock coupling per volt, comparable across settings", "scalar", 1),
    ("start: mean transient", "start", "ADU", "Mean over the usable pixels of CDS_k - median of later skips, k = 0-29 (curve).", "shape of the start transient", "curve", 1),
    ("level: slope-corrected CDS rise (ADU)", "level", "ADU",
     "level: common CDS rise minus lever arm x (Delta over the last 10 % of the skips - Delta over the first 10 % after the noise start) (#13).",
     "part of the CDS rise NOT explained by the baseline slope: a real gain or charge effect", "scalar", 1),
    # ------------------------------------------------------------------ telegraph switching / non-Gaussian (tier 1)
    ("noise: null-CDS noise (ADU)", "noise", "ADU",
     "Std over skips of mean(first half) - mean(second half) of the pedestal window of the residual cycles (#15).",
     "noise without a charge transfer (white: about sqrt(2) x the CDS noise)", "scalar", 1),
    ("discrete: null-CDS kurtosis", "discrete", "", "Excess kurtosis of the null CDS (#16).", "non-Gaussian noise in the output stage, separate from the transfer (RTS)", "scalar", 1),
    ("discrete: in-window step rate (%)", "discrete", "%",
     "Percentage of windows (pedestal and signal, skips from the noise start) whose best single step in 10-sample block means has t > 6 (#18).",
     "random-telegraph switching rate x detection efficiency", "scalar", 1),
    ("discrete: in-window step amplitude (ADU)", "discrete", "ADU", "Median |step| of the detected in-window steps (#19).", "RTS amplitude x gain (trap size)", "scalar", 1),
    ("noise: jump-robust CDS sigma (ADU)", "noise", "ADU", "sqrt(MedRV / N) of the detrended CDS (Andersen-Dobrev-Schaumburg median realized variance) (#25).",
     "continuous noise without jumps", "scalar", 1),
    ("discrete: jump share of CDS variance", "discrete", "", "1 - MedRV / RV of the detrended CDS (#26).", "fraction of the CDS variance in jumps", "scalar", 1),
    # ------------------------------------------------------------------ noise shape (tier 1)
    ("noise: 1/f corner frequency (kHz)", "noise", "kHz",
     "f_c = (A/W)^(1/gamma) of S(f) = W + A f^-gamma fitted (in log space) to noise: CDS spectrum, log frequency, clipped to "
     "[half the lowest, twice the highest] frequency of the band (#36).", "where 1/f noise takes over from white noise", "scalar", 1),
    ("noise: 1/f exponent", "noise", "", "gamma of that fit (#36).", "type of the low-frequency noise", "scalar", 1),
    ("noise: Hadamard deviation n=50 (ADU)", "noise", "ADU",
     "Hadamard deviation (third difference of disjoint n-skip means / sqrt 6) of the CDS with only the common pixel pattern removed, not detrended (#38).",
     "averaging noise, insensitive to linear drift and without the polynomial bias of the detrended Allan", "scalar", 1),
    ("noise: Hadamard deviation", "noise", "ADU", "The Hadamard deviation for the Allan n (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 160; curve); NaN where n is too large for the skips (n = 160 at 490 skips).", "", "curve", 1),
    ("noise: CDS variance share of the 3 largest waveform modes", "noise", "",
     "PCA of the residual skip cycles; the fraction of the CDS variance carried by the 3 modes with the most waveform variance (exploratory report, idea 1).",
     "do the big clock / reset fluctuations reach the CDS? (pilot: no)", "scalar", 1),
    ("noise: waveform modes for 90 % of the CDS variance", "noise", "", "Number of modes (by waveform variance) needed to carry 90 % of the CDS variance.", "how concentrated the CDS noise is", "scalar", 1),
    # ------------------------------------------------------------------ pickup lines (tier 1)
    ("discrete: line count (F-test, p<0.001)", "discrete", "",
     "Frequencies where the multitaper Thomson F statistic (NW = 3, K = 5) of the detrended CDS exceeds F(2, 2K-2) at p = 0.001 in at least half of the usable pixels (#32).",
     "pickup lines with a stated false-alarm rate", "scalar", 1),
    ("discrete: strongest line frequency (kHz)", "discrete", "kHz", "Frequency of the largest median F (#33).", "identifies the pickup source", "scalar", 1),
    ("discrete: strongest line amplitude (ADU)", "discrete", "ADU", "Median over pixels of the multitaper line amplitude 2|mu| at that frequency (#33).", "pickup size", "scalar", 1),
    ("discrete: line phase coherence, pixel clock", "discrete", "",
     "|mean_p exp(i phi_p)| with phi_p the line phase referred to each pixel's own first skip (#34).", "1 = the line is made by the readout sequence (clock-locked)", "scalar", 1),
    ("discrete: line phase coherence, absolute time", "discrete", "",
     "The same with the phase referred to absolute time in the trace.", "1 = an external source running continuously", "scalar", 1),
    # ------------------------------------------------------------------ start of pixel (tier 2)
    ("start: row-start/inner first-skip ratio", "start", "", "Median first-skip excess of the row-start pixels / that of the usable pixels (#7).",
     "memory of the clocking before the pixel (prescan, vertical transfers)", "scalar", 2),
    ("start: reset jump skip 0 to 1 (ADU)", "start", "ADU", "Pedestal mean of skip 1 minus that of skip 0 (#8).", "first reset after the transfers; operating point of the sense node", "scalar", 2),
    ("start: mode frequency (kHz)", "start", "kHz", "Frequency of the strongest oscillating pole of a matrix-pencil fit (order 4) of the mean transient, skips 0-29 (#9).",
     "the start ringing in physical units", "scalar", 2),
    ("start: mode damping (us)", "start", "µs", "Damping time of that pole (#9).", "", "scalar", 2),
    ("start: fast decay (us)", "start", "µs", "Decay time of the non-oscillating pole with the largest amplitude among those decaying within 10 skips (#9).", "", "scalar", 2),
    ("start: transient amplitude, matched filter (ADU)", "start", "ADU", "Projection of each pixel's CDS of skips 0-29 minus its median CDS from skip 10 on, onto the trace's mean transient, times its peak (#10).",
     "per-pixel transient size with less noise than the first-skip excess", "scalar", 2),
    ("start: ringing frequency (kHz)", "start", "kHz", "fit: first-skip ringing frequency (cyc/skip) x the skip rate (#11).", "comparable across delayInt settings", "scalar", 2),
    # ------------------------------------------------------------------ telegraph / non-Gaussian (tier 2)
    ("noise: null-CDS autocorrelation at lag 1", "noise", "", "Lag-1 autocorrelation of the null CDS along the skips (#17).", "slow change of the in-window slope", "scalar", 2),
    ("discrete: jump count (Lee-Mykland, 1 %)", "discrete", "",
     "Number of skips whose |e_k| / local bipower sigma (20 skips) exceeds the 1 % Gumbel critical value of Lee & Mykland, with c = sqrt(2/pi) (#27). With a 20-skip window pure Gaussian noise still gives ~0.26 per series: a relative count.", "jump count (relative)", "scalar", 2),
    ("discrete: STA/LTA maximum", "discrete", "", "max_k mean(e^2, k..k+4) / mean(e^2, k-50..k-1) of the detrended CDS (#28).", "localized bursts (seismology trigger)", "scalar", 2),
    ("discrete: crest factor", "discrete", "", "max |e| / std(e) of the detrended CDS (#29).", "largest excursion", "scalar", 2),
    ("noise: volatility clustering (Ljung-Box of e^2, 10 lags)", "noise", "", "Ljung-Box Q of the squared detrended CDS, 10 lags (#30).", "bursty noise regimes (ARCH)", "scalar", 2),
    ("discrete: whitened CDS kurtosis", "discrete", "", "Excess kurtosis of the detrended CDS after dividing its FFT by the square root of its smoothed spectrum (#31).",
     "non-Gaussianity without the colour of 1/f noise and pickup", "scalar", 2),
    ("discrete: max within-window spectral kurtosis", "discrete", "", "max over f of the spectral kurtosis of the pedestal windows of the residual cycles (#45).",
     "frequency band of impulsive noise", "scalar", 2),
    ("noise: Lorentzian corner (kHz)", "noise", "kHz", "f_L of S(f) = W + B / (1 + (f/f_L)^2) fitted to the log-frequency CDS spectrum (#37).", "RTS rate (Machlup)", "scalar", 2),
    ("noise: within-cycle structure function", "noise", "ADU", "sqrt<(m_{j+L} - m_j)^2> of 15-sample block means of the pedestal windows, L = 1-9 (curve) (#23).",
     "time scale of the noise inside the integration window", "curve", 2),
    ("noise: structure-function excess at 9 blocks (ADU)", "noise", "ADU", "sqrt(max(D(9)^2 - D(1)^2, 0)) of that curve (#24).", "slow in-window noise (RTS with long dwell)", "scalar", 2),
    # ------------------------------------------------------------------ noise shape (tier 2)
    ("noise: modified Allan deviation", "noise", "ADU", "Modified Allan deviation of the detrended CDS, n as the Allan deviation (curve) (#39).", "separates anticorrelated from white noise", "curve", 2),
    ("noise: lag-1 autocorrelation of 10-skip means", "noise", "", "Lag-1 autocorrelation of disjoint 10-skip means of the detrended CDS (#40).", "noise type at the 10-skip scale", "scalar", 2),
    ("noise: DFA exponent", "noise", "", "Detrended-fluctuation-analysis exponent of the detrended CDS (scales 4-48 skips).", "long-range correlation (cardiology); mostly new information (R^2 0.15)", "scalar", 2),
    ("noise: CDS spectral slope", "noise", "", "Slope of log power against log frequency of the detrended CDS.", "noise colour; new information (R^2 0.40)", "scalar", 2),
    ("noise: CDS spectral centroid", "noise", "", "Mean frequency (fraction of the Nyquist) of the CDS power spectrum.", "separates amplifiers strongly (F 63)", "scalar", 2),
    ("noise: sample entropy", "noise", "", "Sample entropy (m = 2, r = 0.2 sd) of the detrended CDS.", "regularity (nonlinear dynamics)", "scalar", 2),
    ("noise: permutation entropy (m=4)", "noise", "", "Normalized permutation entropy, order 4, of the detrended CDS.", "ordinal structure", "scalar", 2),
    ("noise: red-noise beta (50 skips)", "noise", "", "Std of 50-skip bin means / (std / sqrt 50) of the detrended CDS (astronomy beta factor).", "excess noise at the 50-skip scale", "scalar", 2),
    # ------------------------------------------------------------------ lines, levels (tier 2)
    ("discrete: line frequency, whole trace (kHz)", "discrete", "kHz", "Peak of the Lomb-Scargle periodogram of the detrended CDS of all usable pixels in absolute time, 0.2-15 kHz (#35).",
     "fine frequency resolution across the trace", "scalar", 2),
    ("level: common-pattern amplitude spread", "level", "", "Robust spread over the usable pixels of a_p in CDS_p,k - mean ~ a_p (common_k - mean) (#41).",
     "whether the common CDS pattern couples differently to different pixels", "scalar", 2),
    ("noise: baseline-CDS residual correlation", "noise", "", "Correlation of the detrended per-skip pedestal level with the detrended CDS (#42).", "common supply or reset disturbance", "scalar", 2),
    # ------------------------------------------------------------------ clocking regions (user idea, 2026-10-21)
    ("clock: V-block level slope (ADU/transfer)", "clock", "ADU/transfer",
     "Slope of the median level per vertical transfer against the transfer number, in the V block before the second row "
     "(regions.v; its first transfer, which follows the skip region, is left out).",
     "relaxation of the output level during the V clocking; steeper with the V rails low (pilot settings F 45)", "scalar", 1),
    ("clock: V-transfer peak-to-peak (ADU)", "clock", "ADU", "Median over the vertical transfers of that block (regions.v) of the "
     "peak-to-peak of one transfer; at 1x1 binning the block is the single V transfer before row 2.",
     "V-clock feedthrough amplitude (pilot settings F 544)", "scalar", 1),
    ("clock: H/V peak-to-peak ratio", "clock", "", "Median peak-to-peak of every H transfer of the pixels that do not start a row "
     "(1 per pixel at 1x1 binning), divided by the V-transfer peak-to-peak.",
     "H-clock relative to V-clock feedthrough (pilot settings F 879)", "scalar", 1),
    ("clock: first H block of a row, excess (ADU)", "clock", "ADU",
     "In the H block of a row-start pixel (not pixel 0; at least 10 transfers: the prescan burst, NSBIN (NCOLPRE + 1)), the "
     "mean peak-to-peak of transfers 0-4 minus the median peak-to-peak of the second half of the same block; mean over "
     "those row starts. Works at any binning that has a prescan burst (since 2026-10-23; before: against the other pixels' "
     "blocks, which needs NSBIN >= 5).",
     "memory of the V clocking in the first H transfers after coming off a row (pilot settings F 172)", "scalar", 1),
    # ------------------------------------------------------------------ the dip before the per-cycle pulse (user question, 2026-10-21)
    ("edge: pre-pulse dip depth (ADU)", "edge", "ADU",
     "Minimum of the equivalent-time pulse between 6 and 0 samples before the edge, minus its mean 15-8 samples before "
     "(a sharp dip ~120-130 ns before the pulse, ~1.5 samples wide, so single cycles show it as one low sample).",
     "feedthrough of a clock edge just before the pulse (SW/RG/OG, not yet identified); pilot: amplifier-specific, "
     "repeat share 0.001", "scalar", 1),
    ("edge: pre-pulse dip / pulse height", "edge", "", "That depth divided by the height of the equivalent-time pulse above the same baseline.",
     "coupling of that edge relative to the pulse; the common amplification cancels", "scalar", 1),
    ("clock: leading V-block level slope (ADU/transfer)", "clock", "ADU/transfer",
     "Slope of the median level per vertical transfer over the last 12 transfers of the V block that the trace starts in "
     "(before pixel 0), aligned at the block end.",
     "recovery from what preceded the trace; NOT repeatable while the trace starts at a random point in this block "
     "(pilot: 14-19 of 20 transfers recorded, repeat share 0.3)", "scalar", 2),
]

MPENCIL_ORDER = 4
LATE = (20, 60)
FIRST_SKIPS = 20          # skips used for the slope regression
TRANSIENT_SKIPS = 30      # skips in the mean transient


# --------------------------------------------------------------------------- generic helpers

def slopes(W: np.ndarray) -> np.ndarray:
    """Least-squares slope (per sample) of every row of W."""
    t = np.arange(W.shape[1], dtype=float)
    t -= t.mean()
    return (W - W.mean(1, keepdims=True)) @ t / (t @ t)


def kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (0 for a Gaussian); NaN for fewer than 4 values or zero variance."""
    x = np.asarray(x, float)
    return float(((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3) if len(x) > 3 and x.var() > 0 else float("nan")


def med_rv(e: np.ndarray) -> float:
    """Median realized variance (Andersen, Dobrev & Schaumburg 2012), scaled to the sum of squares."""
    a = np.abs(np.asarray(e, float))
    if len(a) < 5:
        return float("nan")
    m = np.median(np.vstack([a[:-2], a[1:-1], a[2:]]), axis=0)
    n = len(a)
    return float(math.pi / (6 - 4 * math.sqrt(3) + math.pi) * n / (n - 2) * np.sum(m ** 2))


def lee_mykland_count(e: np.ndarray, K: int = 20, alpha: float = 0.01) -> float:
    """Number of skips flagged as jumps by the Lee & Mykland (2008) test with a K-skip bipower volatility (a relative
    count; see the comment inside)."""
    e = np.asarray(e, float)
    n = len(e)
    if n < 3 * K:
        return float("nan")
    a = np.abs(e)
    bp = a[1:] * a[:-1]
    c = np.r_[0.0, np.cumsum(bp)]
    count = 0
    # Lee & Mykland (2008): the bipower estimate below is (2/pi) sigma^2, so C_n and S_n carry c = sqrt(2/pi).  Even so,
    # with a local window of only K = 20 skips the rate on pure Gaussian noise is ~0.26 per series, not 1 %: a relative count.
    cc = math.sqrt(2 / math.pi)
    cn = (math.sqrt(2 * math.log(n)) / cc
          - (math.log(math.pi) + math.log(math.log(n))) / (2 * cc * math.sqrt(2 * math.log(n))))
    sn = 1 / (cc * math.sqrt(2 * math.log(n)))
    beta = -math.log(-math.log(1 - alpha))
    for k in range(K, n):
        sigma2 = (c[k - 1] - c[k - K + 1]) / (K - 2)          # bipower over the K previous skips
        if sigma2 <= 0:
            continue
        L = a[k] / math.sqrt(sigma2)
        count += (L - cn) / sn > beta
    return float(count)


def sta_lta(x: np.ndarray, ns: int = 5, nl: int = 50) -> float:
    """Maximum over the series of the short-term (ns) over long-term (nl, preceding) average of the squared values
    (seismology trigger)."""
    e = np.asarray(x, float) ** 2
    if len(e) < nl + ns + 1:
        return float("nan")
    c = np.r_[0.0, np.cumsum(e)]
    k = np.arange(nl, len(e) - ns)
    lta = (c[k] - c[k - nl]) / nl
    return float(np.max(((c[k + ns] - c[k]) / ns) / np.maximum(lta, 1e-300)))


def ljung_box_sq(x: np.ndarray, h: int = 10) -> float:
    """Ljung-Box Q of the squared values over h lags: volatility clustering."""
    e = np.asarray(x, float) ** 2
    e = e - e.mean()
    n, d = len(e), np.dot(e, e)
    if n <= h + 1 or d <= 0:
        return float("nan")
    return float(n * (n + 2) * sum((np.dot(e[:-k], e[k:]) / d) ** 2 / (n - k) for k in range(1, h + 1)))


def whitened_kurtosis(x: np.ndarray, smooth: int = 9) -> float:
    """Kurtosis after dividing the spectrum by its smoothed amplitude: non-Gaussianity without the colour of the
    noise."""
    x = np.asarray(x, float) - np.mean(x)
    if len(x) < 32:
        return float("nan")
    X = np.fft.rfft(x)
    P = np.abs(X) ** 2
    k = np.ones(smooth) / smooth
    Ps = np.convolve(np.r_[P[smooth:0:-1], P, P[-2:-smooth - 2:-1]], k, mode="same")[smooth:smooth + len(P)]
    w = np.fft.irfft(X / np.sqrt(np.maximum(Ps, 1e-300)), n=len(x))
    return kurtosis(w)


def spectral_kurtosis_max(W: np.ndarray) -> float:
    """Largest spectral kurtosis over frequency of the windows (rows of W, Hann-tapered): the band of impulsive
    noise."""
    if W.shape[0] < 10:
        return float("nan")
    X = np.abs(np.fft.rfft(W * np.hanning(W.shape[1]), axis=1)) ** 2
    sk = (X ** 2).mean(0) / np.maximum(X.mean(0) ** 2, 1e-300) - 2
    return float(np.max(sk[1:]))


def structure_function(W: np.ndarray, block: int = 15, max_lag: int = 9) -> np.ndarray:
    """sqrt of the mean squared difference of block means L blocks apart, L = 1..max_lag, inside every window (rows
    of W)."""
    m = W.shape[1] // block
    if m < 2:
        return np.full(max_lag, np.nan)
    B = W[:, :m * block].reshape(len(W), m, block).mean(2)
    return np.array([np.sqrt(np.mean((B[:, L:] - B[:, :-L]) ** 2)) if L < m else np.nan for L in range(1, max_lag + 1)])


def hadamard_deviation(x: np.ndarray, ns) -> np.ndarray:
    """Hadamard deviation (third difference of n-sample means) for every n; NaN where fewer than 4 means exist."""
    out = []
    for n in ns:
        m = len(x) // n
        if m < 4:
            out.append(np.nan)
            continue
        mu = x[:m * n].reshape(m, n).mean(1)
        out.append(np.sqrt(np.mean((mu[2:] - 2 * mu[1:-1] + mu[:-2]) ** 2) / 6))
    return np.array(out)


def modified_allan(x: np.ndarray, ns) -> np.ndarray:
    """Modified Allan deviation of a series (frequency-type data) for every n; NaN where it cannot be formed."""
    x = np.asarray(x, float)
    c = np.r_[0.0, np.cumsum(x)]
    out = []
    for n in ns:
        if len(x) < 3 * n + 1 or len(x) // n < 3:
            out.append(np.nan)
            continue
        a = (c[n:] - c[:-n]) / n                 # moving means of length n
        d = a[n:] - a[:-n]                        # differences n apart
        cd = np.r_[0.0, np.cumsum(d)]
        inner = (cd[n:] - cd[:-n]) / n            # averaged over n consecutive starts
        out.append(np.sqrt(0.5 * np.mean(inner ** 2)))
    return np.array(out)


def dfa_alpha(x: np.ndarray, scales=(4, 6, 8, 12, 16, 24, 32, 48)) -> float:
    """Detrended-fluctuation-analysis exponent (log-log slope of the fluctuation against the scale)."""
    y = np.cumsum(np.asarray(x, float) - np.mean(x))
    F, used = [], []
    for s in scales:
        m = len(y) // s
        if m < 4:
            continue
        seg = y[:m * s].reshape(m, s)
        t = np.arange(s)
        coef = np.polyfit(t, seg.T, 1)
        fit = np.outer(coef[0], t) + coef[1][:, None]
        F.append(np.sqrt(np.mean((seg - fit) ** 2)))
        used.append(s)
    return float(np.polyfit(np.log(used), np.log(F), 1)[0]) if len(used) >= 3 else float("nan")


def psd(x: np.ndarray) -> np.ndarray:
    """Periodogram of the Hann-tapered, mean-removed series, without the zero-frequency bin."""
    return (np.abs(np.fft.rfft((x - np.mean(x)) * np.hanning(len(x)))) ** 2)[1:]


def spectral_slope(x: np.ndarray) -> float:
    """Slope of log power against log frequency index: the colour of the noise."""
    S = psd(x)
    f = np.arange(1, len(S) + 1)
    ok = S > 0
    return float(np.polyfit(np.log(f[ok]), np.log(S[ok]), 1)[0]) if ok.sum() >= 3 else float("nan")


def spectral_centroid(x: np.ndarray) -> float:
    """Power-weighted mean frequency of the periodogram, as a fraction of the Nyquist frequency."""
    S = psd(x)
    f = np.arange(1, len(S) + 1) / len(S)
    return float((f * S).sum() / S.sum()) if S.sum() > 0 else float("nan")


def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Sample entropy (template length m, tolerance r standard deviations); NaN for short or constant series."""
    x = np.asarray(x, float)
    if len(x) < 20 or x.std() == 0:
        return float("nan")
    x = (x - x.mean()) / x.std()
    n = len(x)

    def count(mm):
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[: n - m]
        d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
        return (np.sum(d <= r) - len(emb)) / 2
    B, A = count(m), count(m + 1)
    return float(-np.log(A / B)) if A > 0 and B > 0 else float("nan")


def permutation_entropy(x: np.ndarray, m: int = 4) -> float:
    """Normalized permutation entropy of order m (1 = all ordinal patterns equally likely)."""
    x = np.asarray(x, float)
    if len(x) < 10 * math.factorial(m):
        return float("nan")
    pats = np.argsort(np.lib.stride_tricks.sliding_window_view(x, m), axis=1)
    codes = (pats * (m ** np.arange(m))).sum(1)
    _, c = np.unique(codes, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(m)))


def red_noise_beta(x: np.ndarray, n: int = 50) -> float:
    """Scatter of n-sample block means over the white-noise expectation (astronomy beta factor)."""
    x = np.asarray(x, float)
    m = len(x) // n
    if m < 3:
        return float("nan")
    b = x[:m * n].reshape(m, n).mean(1)
    return float(b.std(ddof=1) / (x.std(ddof=1) / math.sqrt(n)))


def lag1_of_block_means(x: np.ndarray, n: int = 10) -> float:
    """Lag-1 autocorrelation of disjoint n-sample block means."""
    m = len(x) // n
    if m < 5:
        return float("nan")
    b = x[:m * n].reshape(m, n).mean(1)
    b = b - b.mean()
    return float(np.dot(b[:-1], b[1:]) / np.dot(b, b)) if np.dot(b, b) > 0 else float("nan")


def fit_one_over_f(f: np.ndarray, S: np.ndarray) -> tuple[float, float]:
    """Fit S = W + A f^-gamma in log space; returns (corner frequency in units of f, gamma)."""
    from scipy.optimize import curve_fit
    ok = np.isfinite(S) & (S > 0) & (f > 0)
    f, S = f[ok], S[ok]
    if len(f) < 6:
        return float("nan"), float("nan")
    model = lambda lf, lw, la, g: np.log(np.exp(lw) + np.exp(la) * np.exp(-g * lf))
    W0 = np.median(S[len(S) // 2:])
    try:
        p, _ = curve_fit(model, np.log(f), np.log(S), p0=[np.log(W0), np.log(W0 * f[0]), 1.0],
                         bounds=([-np.inf, -np.inf, 0.2], [np.inf, np.inf, 3.0]), maxfev=5000)
    except Exception:                                # noqa: BLE001
        return float("nan"), float("nan")
    lw, la, g = p
    fc = np.exp((la - lw) / g)
    # a corner outside the measured band means "no 1/f noise in the band" (below) or "1/f everywhere" (above):
    # clip to half the lowest / twice the highest frequency so that it stays a usable number
    return float(np.clip(fc, f[0] / 2, f[-1] * 2)), float(g)


def fit_lorentzian(f: np.ndarray, S: np.ndarray) -> float:
    """Corner frequency f_L of S = W + B / (1 + (f/f_L)^2) fitted in log space; NaN if the fit fails."""
    from scipy.optimize import curve_fit
    ok = np.isfinite(S) & (S > 0) & (f > 0)
    f, S = f[ok], S[ok]
    if len(f) < 6:
        return float("nan")
    model = lambda ff, lw, lb, lfl: np.log(np.exp(lw) + np.exp(lb) / (1 + (ff / np.exp(lfl)) ** 2))
    W0 = np.median(S[len(S) // 2:])
    try:
        p, _ = curve_fit(model, f, np.log(S), p0=[np.log(W0), np.log(max(S[0] - W0, W0 * 0.1)), np.log(np.median(f))],
                         bounds=([-np.inf, -np.inf, np.log(f[0] / 10)], [np.inf, np.inf, np.log(f[-1] * 10)]), maxfev=5000)
    except Exception:                                # noqa: BLE001
        return float("nan")
    return float(np.exp(p[2]))


def multitaper_lines(series: list[np.ndarray], rate: float, NW: float = 3.0, K: int = 5, p: float = 0.001):
    """Thomson F test per series: (freqs, F (series x freqs), mu (complex, series x freqs), F critical)."""
    from scipy.signal.windows import dpss
    from scipy.stats import f as fdist
    n = min(len(s) for s in series)
    tapers = dpss(n, NW, K)                               # K x n, unit energy
    U0 = tapers.sum(1)                                    # K
    Fs, mus = [], []
    for s in series:
        x = np.asarray(s[:n], float) - np.mean(s[:n])
        Y = np.fft.rfft(tapers * x, axis=1)               # K x nf
        mu = (U0[:, None] * Y).sum(0) / (U0 ** 2).sum()
        resid = np.sum(np.abs(Y - mu[None, :] * U0[:, None]) ** 2, axis=0)
        Fs.append((K - 1) * np.abs(mu) ** 2 * (U0 ** 2).sum() / np.maximum(resid, 1e-300))
        mus.append(mu)
    freqs = np.fft.rfftfreq(n, 1 / rate)
    return freqs, np.array(Fs), np.array(mus), float(fdist.ppf(1 - p, 2, 2 * K - 2))


def window_steps(W: np.ndarray, block: int = 10, threshold: float = 6.0) -> tuple[float, float]:
    """
    In-window step detection: for every window (row of W), the best single step between 10-sample block means, in
    units of its standard error (block-mean noise from the within-block scatter, median over all windows).
    Returns (percentage of windows with t > threshold, median |step| of those windows in ADU; NaN if none).
    """
    m = W.shape[1] // block
    if m < 4 or len(W) == 0:
        return float("nan"), float("nan")
    Bl = W[:, :m * block].reshape(len(W), m, block)
    b = Bl.mean(2)
    sb = np.sqrt(np.median(Bl.var(axis=2, ddof=1)) / block)
    if sb <= 0:
        return float("nan"), float("nan")
    best_t, best_d = np.zeros(len(b)), np.zeros(len(b))
    c = np.cumsum(b, axis=1)
    tot = c[:, -1]
    for j in range(1, m):
        before = c[:, j - 1] / j
        after = (tot - c[:, j - 1]) / (m - j)
        t = np.abs(after - before) / (sb * np.sqrt(1 / j + 1 / (m - j)))
        better = t > best_t
        best_t[better], best_d[better] = t[better], (after - before)[better]
    flagged = best_t > threshold
    return float(100 * flagged.mean()), (float(np.median(np.abs(best_d[flagged]))) if flagged.any() else float("nan"))


def matrix_pencil(y: np.ndarray, order: int = MPENCIL_ORDER) -> tuple[np.ndarray, np.ndarray]:
    """Poles z and amplitudes c of y_k ~ sum c_i z_i^k (matrix-pencil method; amplitudes by least squares)."""
    y = np.asarray(y, float)
    N = len(y)
    L = N // 2
    if N < 2 * order + 2:
        return np.array([]), np.array([])
    Y = np.array([y[i:i + L + 1] for i in range(N - L)])
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    V = Vt[:order].T
    V1, V2 = V[:-1], V[1:]
    z = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    Z = z[None, :] ** np.arange(N)[:, None]
    c, *_ = np.linalg.lstsq(Z, y.astype(complex), rcond=None)
    return z, c


# --------------------------------------------------------------------------- the computation

def extra_features(ctx: dict) -> tuple[dict, dict, dict]:
    """
    ctx: the shared arrays of trace_features._trace_features: Cm (pixels x skips CDS), cyc {p: skips x samples},
    res {p: residual cycles}, det {p: detrended CDS}, good, row_starts, n_ped, sig, sig_end, first, n, skip_rate,
    sample_rate, starts {p: absolute cycle starts}, meta, scalars (the existing ones, read only), allan_n, common,
    regions (TraceRegions), trace (DecodedTrace), curves (the existing curves, read only).
    Returns (scalars, per_pixel {name: values}, curves).
    """
    S, PP, C = {}, {}, {}
    Cm, cyc, res, det = ctx["Cm"], ctx["cyc"], ctx["res"], ctx["det"]
    good, row_starts = ctx["good"], ctx["row_starts"]
    n_ped, sig, first, n = ctx["n_ped"], ctx["sig"], ctx["first"], ctx["n"]
    fs, skip_rate, meta = ctx["sample_rate"], ctx["skip_rate"], ctx["meta"] or {}
    nan = float("nan")
    errors = {}

    def block(name, fn):
        try:
            fn()
        except Exception as err:                                  # noqa: BLE001 - one block must not stop the rest
            errors[name] = f"{type(err).__name__}: {err}"

    # ---------------- start of pixel
    late = slice(min(LATE[0], n // 5), min(LATE[1], n))
    k_first = min(FIRST_SKIPS, n)

    def window_slopes(p):
        Y = cyc[p]
        return 0.5 * (slopes(Y[:, :n_ped]) + slopes(Y[:, sig:sig + n_ped]))

    def start():
        xs, ys, x0, trans, E0, jumps = [], [], [], [], {}, []
        for p in range(len(Cm)):
            delta = window_slopes(p)
            cds = Cm[p]
            ref = np.median(cds[first:])
            E0[p] = cds[0] - ref
            if p in good:
                xs.append(delta[:k_first] - np.median(delta[late]))
                ys.append(cds[:k_first] - ref)
                x0.append((delta[0] - np.median(delta[late])) * fs / 1e6)
                trans.append(cds[:TRANSIENT_SKIPS] - ref)
                Y = cyc[p]
                jumps.append(Y[1, :n_ped].mean() - Y[0, :n_ped].mean())
        X, Yv = np.concatenate(xs), np.concatenate(ys)
        A = np.column_stack([X, np.ones_like(X)])
        coef, *_ = np.linalg.lstsq(A, Yv, rcond=None)
        resid = Yv - A @ coef
        S["start: CDS-slope lever arm (samples)"] = float(coef[0])
        S["start: slope-explained fraction"] = float(1 - resid.var() / Yv.var()) if Yv.var() > 0 else nan
        PP["start: in-cycle slope at skip 0 (ADU/us)"] = x0
        T = np.mean(trans, axis=0)
        C["start: mean transient"] = T
        # transient inside skip 0: skip-0 waveform minus the late mean waveform, each minus its pedestal mean
        taus, amps = [], []
        t_all = np.arange(cyc[good[0]].shape[1]) / fs * 1e6
        sel = np.r_[np.arange(0, n_ped), np.arange(sig, min(sig + n_ped, cyc[good[0]].shape[1]))]
        from compare_trace import fit_exponential
        for p in good:
            Y = cyc[p]
            Yb = Y - Y[:, :n_ped].mean(1, keepdims=True)
            d0 = Yb[0] - Yb[late].mean(0)
            tau, amp, _ = fit_exponential(t_all[sel], d0[sel], 0.5, 200.0)
            taus.append(tau)
            amps.append(amp)
        PP["start: transient time constant (us)"] = taus
        PP["start: transient amplitude (ADU)"] = amps
        excess = ctx["scalars"].get("skip: first-skip CDS excess", nan)
        if "H1_A_H" in meta and "H1_A_L" in meta and meta["H1_A_H"] != meta["H1_A_L"]:
            S["start: first-skip excess per volt H (ADU/V)"] = float(excess / (meta["H1_A_H"] - meta["H1_A_L"]))
        # slope-corrected CDS rise
        tenth = max(1, n // 10)
        dmed = np.median([window_slopes(p) for p in good], axis=0)
        rise = ctx["scalars"].get("level: common CDS rise (ADU)", nan)
        if n - tenth >= first + tenth:
            S["level: slope-corrected CDS rise (ADU)"] = float(rise - coef[0] * (dmed[n - tenth:].mean() - dmed[first:first + tenth].mean()))
        # tier 2
        inner = np.median([E0[p] for p in good])
        rs = [E0[p] for p in row_starts if p < len(Cm)]
        S["start: row-start/inner first-skip ratio"] = float(np.median(rs) / inner) if rs and inner else nan
        PP["start: reset jump skip 0 to 1 (ADU)"] = jumps
        z, c = matrix_pencil(T)
        if len(z):
            # the oscillating pole pair and the real pole with the largest amplitude (decaying poles only)
            osc = [(q, abs(a)) for q, a in zip(z, c) if q.imag > 1e-6 and 0 < abs(q) < 1]
            # real poles that decay within 10 skips (slower ones are the CDS trend, not the transient)
            real = [(q.real, abs(a)) for q, a in zip(z, c) if abs(q.imag) <= 1e-6 and np.exp(-1 / 10) > q.real > 0]
            if osc:
                q = max(osc, key=lambda qa: qa[1])[0]
                S["start: mode frequency (kHz)"] = float(np.angle(q) / (2 * np.pi) * skip_rate / 1e3)
                S["start: mode damping (us)"] = float(-1 / (np.log(abs(q)) * skip_rate) * 1e6)
            if real:
                q = max(real, key=lambda qa: qa[1])[0]
                S["start: fast decay (us)"] = float(-1 / (np.log(q) * skip_rate) * 1e6)
        TT = float(np.dot(T, T))
        if TT > 0:
            PP["start: transient amplitude, matched filter (ADU)"] = [float(np.dot(tr, T) / TT * np.max(np.abs(T))) for tr in trans]
        rf = ctx["scalars"].get("fit: first-skip ringing frequency (cyc/skip)", nan)
        S["start: ringing frequency (kHz)"] = float(rf * skip_rate / 1e3)
    block("start", start)

    # ---------------- telegraph switching inside one window, jump-robust noise
    def switching():
        half = n_ped // 2
        nulls = {p: res[p][:, :half].mean(1) - res[p][:, half:2 * half].mean(1) for p in good}
        PP["noise: null-CDS noise (ADU)"] = [v.std() for v in nulls.values()]
        PP["discrete: null-CDS kurtosis"] = [kurtosis(v) for v in nulls.values()]
        PP["noise: null-CDS autocorrelation at lag 1"] = [float(np.corrcoef(v[:-1], v[1:])[0, 1]) for v in nulls.values()]
        rates, amps = [], []
        for p in good:
            rate, amp = window_steps(np.vstack([res[p][:, :n_ped], res[p][:, sig:sig + n_ped]]))
            rates.append(rate)
            amps.append(amp)
        PP["discrete: in-window step rate (%)"] = rates
        PP["discrete: in-window step amplitude (ADU)"] = amps
        PP["noise: jump-robust CDS sigma (ADU)"] = [math.sqrt(med_rv(e) / len(e)) for e in det.values()]
        PP["discrete: jump share of CDS variance"] = [max(0.0, 1 - med_rv(e) / np.sum(e ** 2)) for e in det.values()]
        PP["discrete: jump count (Lee-Mykland, 1 %)"] = [lee_mykland_count(e) for e in det.values()]
        PP["discrete: STA/LTA maximum"] = [sta_lta(e) for e in det.values()]
        PP["discrete: crest factor"] = [float(np.abs(e).max() / e.std()) for e in det.values()]
        PP["noise: volatility clustering (Ljung-Box of e^2, 10 lags)"] = [ljung_box_sq(e) for e in det.values()]
        PP["discrete: whitened CDS kurtosis"] = [whitened_kurtosis(e) for e in det.values()]
        PP["discrete: max within-window spectral kurtosis"] = [spectral_kurtosis_max(res[p][:, :n_ped]) for p in good]
        sf = np.nanmean([structure_function(res[p][:, :n_ped]) for p in good], axis=0)
        C["noise: within-cycle structure function"] = sf
        S["noise: structure-function excess at 9 blocks (ADU)"] = float(np.sqrt(max(sf[-1] ** 2 - sf[0] ** 2, 0))) if np.isfinite(sf[[0, -1]]).all() else nan
    block("switching", switching)

    # ---------------- noise shape
    def shape():
        ns = ctx["allan_n"]
        P = len(good)
        shared = Cm[good, first:].mean(0) if P > 1 else 0.0
        scale = np.sqrt(P / (P - 1)) if P > 1 else 1.0
        raw = {p: scale * (Cm[p, first:] - shared - (Cm[p, first:] - shared).mean()) for p in good}
        had = np.array([hadamard_deviation(raw[p], ns) for p in good])
        C["noise: Hadamard deviation"] = np.nanmean(had, axis=0)
        i50 = list(ns).index(50)
        PP["noise: Hadamard deviation n=50 (ADU)"] = had[:, i50]
        C["noise: modified Allan deviation"] = np.nanmean([modified_allan(det[p], ns) for p in good], axis=0)
        from trace_features import log_frequency_bins, spectrum
        f_raw, S_raw = spectrum(list(raw.values()), skip_rate)
        binned = log_frequency_bins(f_raw, S_raw)
        fpos = f_raw[f_raw > 0]
        edges = np.geomspace(fpos[0], fpos[-1] * (1 + 1e-9), len(binned) + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
        fc, g = fit_one_over_f(centers, binned)
        S["noise: 1/f corner frequency (kHz)"] = fc / 1e3
        S["noise: 1/f exponent"] = g
        S["noise: Lorentzian corner (kHz)"] = fit_lorentzian(centers, binned) / 1e3
        PP["noise: lag-1 autocorrelation of 10-skip means"] = [lag1_of_block_means(e) for e in det.values()]
        PP["noise: DFA exponent"] = [dfa_alpha(e) for e in det.values()]
        PP["noise: CDS spectral slope"] = [spectral_slope(e) for e in det.values()]
        PP["noise: CDS spectral centroid"] = [spectral_centroid(e) for e in det.values()]
        PP["noise: sample entropy"] = [sample_entropy(e) for e in det.values()]
        PP["noise: permutation entropy (m=4)"] = [permutation_entropy(e) for e in det.values()]
        PP["noise: red-noise beta (50 skips)"] = [red_noise_beta(e) for e in det.values()]
        # noise budget: which waveform modes carry the CDS variance
        R = np.vstack([res[p] for p in good])
        w = np.zeros(R.shape[1])
        w[:n_ped] = 1 / n_ped
        if sig + n_ped > R.shape[1]:
            raise ValueError("signal window runs past the cycle: the CDS weights would not balance")
        w[sig:sig + n_ped] = -1 / n_ped
        R0 = R - R.mean(0)
        U, sv, Vt = np.linalg.svd(R0, full_matrices=False)
        contrib = (sv ** 2) * (Vt @ w) ** 2
        tot = contrib.sum()
        if tot > 0:
            S["noise: CDS variance share of the 3 largest waveform modes"] = float(contrib[:3].sum() / tot)
            S["noise: waveform modes for 90 % of the CDS variance"] = float(np.searchsorted(np.cumsum(contrib) / tot, 0.9) + 1)
    block("shape", shape)

    # ---------------- pickup lines
    def lines():
        series = [det[p] for p in good]
        freqs, F, mu, Fcrit = multitaper_lines(series, skip_rate)
        lo = freqs > 3.0 / len(series[0]) * skip_rate          # leave out the lowest bins (taper bandwidth)
        frac = (F > Fcrit).mean(0)
        S["discrete: line count (F-test, p<0.001)"] = float(np.sum((frac >= 0.5) & lo))
        Fmed = np.median(F, axis=0)
        Fmed[~lo] = 0
        i = int(np.argmax(Fmed))
        f0 = freqs[i]
        S["discrete: strongest line frequency (kHz)"] = float(f0 / 1e3)
        S["discrete: strongest line amplitude (ADU)"] = float(np.median(2 * np.abs(mu[:, i])))
        phi = np.angle(mu[:, i])
        t0 = np.array([ctx["starts"][p][first] / fs for p in good])
        S["discrete: line phase coherence, pixel clock"] = float(np.abs(np.mean(np.exp(1j * phi))))
        S["discrete: line phase coherence, absolute time"] = float(np.abs(np.mean(np.exp(1j * (phi - 2 * np.pi * f0 * t0)))))
        # whole trace, absolute time
        from scipy.signal import lombscargle
        t = np.concatenate([(np.asarray(ctx["starts"][p][first:first + len(det[p])]) / fs) for p in good])
        y = np.concatenate([det[p][:len(ctx["starts"][p][first:first + len(det[p])])] for p in good])
        grid = np.arange(200.0, min(15000.0, skip_rate / 2), 5.0)
        pw = lombscargle(t, y - y.mean(), 2 * np.pi * grid)
        S["discrete: line frequency, whole trace (kHz)"] = float(grid[int(np.argmax(pw))] / 1e3)
    block("lines", lines)

    # ---------------- levels
    def levels():
        common = ctx["common"][first:] - ctx["common"][first:].mean()
        cc = np.dot(common, common)
        a = [float(np.dot(Cm[p, first:] - Cm[p, first:].mean(), common) / cc) for p in good] if cc > 0 else []
        from trace_features import detrend, robust_spread
        if a:
            S["level: common-pattern amplitude spread"] = robust_spread(np.asarray(a))
        PP["noise: baseline-CDS residual correlation"] = [
            float(np.corrcoef(detrend(cyc[p][first:n, :n_ped].mean(1)), det[p])[0, 1]) for p in good]
    block("levels", levels)

    # ---------------- the dip before the per-cycle pulse (from the equivalent-time pulse of trace_features)
    def prepulse():
        e = ctx.get("curves", {}).get("edge: equivalent-time pulse")
        if e is None:
            return
        from trace_features import ETS_BINS
        e = np.asarray(e, float)
        c = ETS_BINS[:-1] + 0.05                                    # the bin centres of that curve (samples from the edge)
        base = np.nanmean(e[(c > -15) & (c < -8)])
        win = np.flatnonzero((c > -6) & (c < 0) & np.isfinite(e))
        if not len(win) or not np.isfinite(base):
            return
        depth = float(np.nanmin(e[win]) - base)
        S["edge: pre-pulse dip depth (ADU)"] = depth
        peak = float(np.nanmax(e) - base)
        if peak > 0:
            S["edge: pre-pulse dip / pulse height"] = depth / peak
    block("prepulse", prepulse)

    # ---------------- clocking regions (V blocks, first H block of a row)
    def clock():
        r = ctx.get("regions")
        if r is None:
            return
        def level_slope(V):
            med = np.median(V, axis=1)
            return float(np.polyfit(np.arange(len(med)), med, 1)[0]) if len(med) >= 3 else nan
        pp_v = nan
        # regions.v falls back to the leading block (the trace starts inside it at a random point) when no later
        # pixel starts a row; that block is not repeatable, so the V features need a row start after pixel 0
        later_row = any(p > 0 for p in r.row_starts)
        # every clock feature but the level slope works at 1x1 binning (one V transfer per row, one H transfer per pixel,
        # plus the prescan burst at each row start), the binning of the matched traces (docs/packet_design.md section 6)
        if later_row and r.v is not None and r.v.ndim == 2 and len(r.v) >= 1:
            if len(r.v) >= 3:                                  # needs several V transfers per row (NPBIN >= 4)
                S["clock: V-block level slope (ADU/transfer)"] = level_slope(r.v)
            pp_v = float(np.median(np.ptp(r.v, axis=1)))
            S["clock: V-transfer peak-to-peak (ADU)"] = pp_v
        inner = [p for p in range(len(r.h)) if p not in r.row_starts and r.h[p].ndim == 2 and len(r.h[p]) >= 1]
        if inner and np.isfinite(pp_v) and pp_v > 0:
            S["clock: H/V peak-to-peak ratio"] = float(np.median(np.concatenate([np.ptp(r.h[p], axis=1) for p in inner])) / pp_v)
        ex = []
        for p in r.row_starts:
            if p > 0 and r.h[p].ndim == 2 and len(r.h[p]) >= 10:
                pp = np.ptp(r.h[p], axis=1)
                ex.append(float(np.mean(pp[:5]) - np.median(pp[len(pp) // 2:])))
        if ex:
            S["clock: first H block of a row, excess (ADU)"] = float(np.mean(ex))
        trace = ctx.get("trace")
        if trace is not None and r.v_period and np.isfinite(r.v_period) and r.ped_starts:
            from trace_regions import _rising
            flags = np.asarray(trace.flags)
            e_v = _rising(flags, 7)
            if len(e_v) < 2:
                e_v = _rising(flags, 5)
            e_h = _rising(flags, 6)
            end = int(e_h[0]) if len(e_h) else int(r.ped_starts[0][0])
            lead = e_v[e_v < end]
            period = int(round(r.v_period))
            values = np.asarray(trace.values, float)
            V = np.array([values[e:e + period] for e in lead if e + period <= end])
            if len(V) >= 13:
                S["clock: leading V-block level slope (ADU/transfer)"] = level_slope(V[-12:])
    block("clock", clock)

    S["_errors"] = errors
    return S, PP, C
