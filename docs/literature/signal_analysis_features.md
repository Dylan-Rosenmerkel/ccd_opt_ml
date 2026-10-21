# Hand-computable signal-analysis features from other fields, for the skipper-CCD readout traces

*Written 2026-10-21 by a literature agent. Input: `src/trace_features.py` (the `FEATURES` registry: 85 scalars + 14 curves),
`src/trace_regions.py`, `src/decode_trace.py`, `src/compare_trace.py` (the skip tools, including the first-skip ringing fit),
`docs/literature/landscape_exploratory.md` (the exploratory analyst's report), and the feature table
`docs/slides/data/oct12_trace_features.csv` (59 traces = 5 settings x 3 repeats x 4 amplifiers, board 101, 2026-10-12). No code in the
repository was changed. Four small read-only pilots were run on the real 2026-10-12 traces (Section 0.2). Their scripts are in the
session scratchpad, not the repository.*

**Scope.** These are interpretable, hand-computable numbers and diagnostic plots, not black-box models. Each one is checked
against what `trace_features.py` already computes, so the report does not re-propose existing features. Ideas already in
`landscape_exploratory.md` are credited and only extended: the beat-space KLT, bipower/MedRV, spectral kurtosis, STA/LTA,
coda-wave stretching, GMWM, the two-clock row-start model and Westgard rules.

**Labels.**
* **(checked)**: I found the bibliographic record (title, authors, venue, year) during this work with a web search. "Record only"
  means I did not read the full text.
* **[unverified]**: the reference is cited from memory, or its details could not be confirmed.
* **[pilot]**: I tested the idea on real data in this session.
* **[speculative]**: an interpretation that has not been tested.
* "Existing" names the feature in `FEATURES` that already covers the idea.

---

## 0. Summary

### 0.1 Headline findings

1. **The first-skip offset is the CDS acting as a differentiator of a baseline transient** [pilot, all 59 traces]:
   * For skips 0-19 of every usable pixel, `CDS_k − median` equals −L · (in-window baseline slope of skip k − late slope).
     L is 211-221 samples (the window centres are 207 samples apart), with R² = 0.95-0.98 on ch0, ch1 and ch3.
   * The same identity predicts the slow 29-ADU "common CDS rise" along the 500 skips to within 1 ADU.
   * Inside skip 0, the transient is an exponential with **τ ≈ 11-12.5 µs** (all amplifiers, all settings) and amplitude
     **560-790 ADU**.
   * The amplitude, and with it the first-skip excess, **scales with the H-clock swing**: "all H low rails" (swing +0.5 V,
     +18-22 %) raises the excess by +22-24 % on all four amplifiers.
   * With 185 prescan columns, **row-start pixels have only 0.41-0.48 of the excess** of inner pixels (0.93 with NCOLPRE = 0).
     The transient therefore remembers the clocking that precedes the pixel.
   * The excess is the same on ch2 as on ch0, so ch2's noise source plays no part in it.
   * Section 3 develops these results.
2. **ch2's excess noise exists inside a single integration window, with no charge transfer involved** [pilot]:
   * A "null CDS" (first half minus second half of the same pedestal window) has kurtosis 11-15 on ch2 against −0.4 to 0 on
     the others.
   * A single-step detector applied to 10-sample block means inside each 10-µs window finds steps in **6.3 %** (reference) and
     **13 %** (V low rails) of ch2's windows, and in **0 of 7,840** windows of ch0.
   * The steps are ±symmetric (238 up / 255 down) and mostly 40-60 ADU (reference) or 40-110 ADU (V low rails).
   * This is the signature of random telegraph switching in the output stage or video chain. It supports the exploratory
     report's guess, and moves it from a CDS-level inference to direct evidence in the raw samples.
3. **Several popular cross-field statistics add nothing here** [pilot]:
   * permutation entropy (0.995-0.999 on every amplifier);
   * DFA α (≈ 0.5 on every amplifier after the existing degree-4 detrending);
   * Poincaré SD1/SD2 and RMSSD, which are exact functions of the variance and the lag-1 autocorrelation (Brennan et al. 2001
     make the same point for HRV);
   * spectral flatness (0.47-0.55, weak).
   They are listed at low priority so nobody spends time on them.

### 0.2 Pilots run (read-only, `.venv/bin/python`, the project's `load_regions`, `usable_pixels` and `residual_cycles`)

| Script (scratchpad) | Data | What it measured |
|---|---|---|
| `first_skip_pilot.py` | all 59 traces of `oct12_trace_features.csv` | per pixel: first-skip excess E0, its spread, row-start E0; regression of CDS excess on in-window slope (skips 0-19); exponential fit of the skip-0 difference waveform; reset-level jump from skip 0 to skip 1 |
| inline snippet | ch0 reference trace `…185604_32` | slope vs CDS over skips 1-19; the slow CDS rise vs the change of in-window slope (50-skip blocks) |
| `structure_pilot.py` | 6 traces (ch0-ch3 reference, ch2 and ch0 V low rails) | within-cycle structure function of 15-sample block means; null CDS (half-window differences) |
| `rts_steps_pilot.py` | ch0 reference, ch2 reference, ch2 V low rails | single-step detection in 10-sample block means inside every pedestal and signal window: rate, amplitude histogram, up/down balance |
| `cheap_features_pilot.py` | reference and V-low-rails traces, 4 amplifiers | permutation entropy, DFA α, spectral flatness, crest factor, RMSSD/SD, STA/LTA maximum, Ljung-Box on e² |

All scripts are under
`/private/tmp/claude-501/-Users-dylanrosenmerkel-Documents-Software-ccd-opt-ml/0face8dd-3b86-44b8-b894-2b5f02a97511/scratchpad/`.

### 0.3 Ranked top 10 (details in Section 5)

1. `start: CDS-slope lever arm (samples)` and `start: slope-explained fraction` (the differentiator identity).
2. `start: transient time constant (us)` and `start: transient amplitude (ADU)`.
3. `start: first-skip excess per volt H (ADU/V)`.
4. `discrete: in-window step rate (%)` and `discrete: in-window step amplitude (ADU)` (RTS in raw samples).
5. `noise: null-CDS noise (ADU)` and `discrete: null-CDS kurtosis`.
6. `noise: jump-robust CDS sigma (ADU)` and `discrete: jump share of CDS variance` (MedRV; from the exploratory report, never
   implemented).
7. `level: slope-corrected CDS rise (ADU)`.
8. `discrete: strongest line frequency (kHz)`, its amplitude, and `discrete: line phase coherence` (multitaper F-test with
   tracking across pixels).
9. `noise: within-cycle structure function` (curve) and its large-lag excess.
10. `noise: 1/f corner frequency (kHz)` from a white + 1/f fit, and `start: row-start/inner first-skip ratio`.

---

## 1. Field by field

For each method:
* **Def**: the definition, kept short;
* **Ours**: the analogous quantity and region of our trace;
* **Reveals**: the CCD or output-stage effect it could reveal;
* **Status**: existing (the feature name) or new;
* **Cost**: compute cost and robustness at 500 skips × ~8 usable pixels.

### 1.1 Cardiology: ECG beats and heart-rate variability (HRV)

The ECG is a strongly periodic signal. Cardiology measures how each beat deviates from the template and how the beat
interval varies.

* **Beat template subtraction and beat-to-beat KLT.** Covered by the exploratory report (Laguna et al. 1999, verified there).
  `skip: mean shape`, `slow: PC1-3 variance fraction` and `edge: variation along the cycle` already implement the template
  and the residual PCA. Not re-proposed.
* **SDNN, RMSSD.**
  * Def: SDNN = std of the intervals; RMSSD = sqrt(mean(Δx²)) of successive differences [1].
  * Ours: the detrended CDS series e_k.
  * Status: **equivalent to existing features.** RMSSD/SD = sqrt(2(1 − ρ₁)), with ρ₁ = `noise: CDS autocorrelation at lag 1`.
    Pilot: 1.30-1.35 on ch0/1/3 and 1.44-1.51 on ch2, exactly as ρ₁ predicts.
  * Low priority.
* **Poincaré (return) plot, SD1/SD2.**
  * Def: scatter of e_{k+1} against e_k. SD1 = std((e_{k+1} − e_k)/√2), SD2 = std((e_{k+1} + e_k)/√2).
  * Brennan, Palaniswami & Kamen [2] showed that SD1 and SD2 are linear functions of the autocovariance. The numbers are
    redundant (SD1² = σ²(1 − ρ₁), SD2² = σ²(1 + ρ₁)), but **the picture is not**.
  * For a two-level RTS with dwell times near one skip, the cloud gets satellite clusters offset by ±Δ along the
    anti-diagonal: a step followed by its reversal.
  * Use as a diagnostic plot (Section 4, D3). Low priority as a feature.
* **Detrended fluctuation analysis (DFA).**
  * Def: integrate the series; detrend it linearly in windows of size s; α = slope of log F(s) against log s [3].
  * Ours: the CDS.
  * Pilot: α = 0.48-0.54 on every amplifier, i.e. white, after the existing degree-4 detrending. Redundant with
    `noise: Allan slope n=10-100`. Low.
  * Without the polynomial detrending (DFA removes trends itself), α would add something only in combination with the
    common-pattern removal. Already covered by `noise: Allan deviation, common pattern removed`.
* **Sample entropy (SampEn).**
  * Def: −ln(A/B), where B is the number of template matches of length m within tolerance r and A the number of length m+1 [4].
  * Ours: CDS.
  * Reveals: regularity, which is low for Gaussian white noise. It would rise for a two-state process, but only weakly
    because of the ~6-ADU Gaussian background.
  * Cost O(N²) = 2.5 × 10⁵ comparisons per pixel: fine. Low priority (the RTS-specific tools below are sharper).

### 1.2 Neuroscience: EEG band power, event-related averaging, spike sorting, phase-amplitude coupling

* **Band power and relative band power.**
  * Def: integrated PSD in fixed bands divided by the total.
  * Status: **existing.** `noise: within-cycle power 0.1-0.3 / 0.3-1 / 1-3 / 3-7.5 MHz`, `skip: noise power …`,
    `noise: CDS power <1 kHz / >10 kHz`.
* **Event-related (triggered) averaging.**
  * Def: average many epochs aligned on an event; the noise drops as 1/√n.
  * Ours: (i) raw samples around each detected in-window RTS step (a step-triggered average), which gives the step's rise
    shape, i.e. the bandwidth of the stage after the switching element; (ii) the CDS around row starts. (ii) already exists as
    `slow: row-start recovery`.
  * (i) is new: `edge: RTS step rise time (ns)`. Medium; needs the step detector of 1.3.
* **Spike sorting** (detect, align, extract features, cluster) [Quiroga et al. 2004, *Neural Comput.* 16:1661, unverified].
  * Ours: cluster the detected in-window steps of ch2 by amplitude and sign. One cluster pair (±Δ) means one trap; several
    amplitude clusters mean several traps or a multi-level RTS.
  * Pilot: the ch2 reference trace has a narrow 40-60 ADU population. In the V-low-rails trace, steps spread over 40-110 ADU,
    which suggests more than one trap, or a larger one, at that setting [speculative].
  * Diagnostic (D4), not a feature.
* **Phase-amplitude coupling (PAC), the modulation index of Tort et al.** [5] (checked).
  * Def: bin the phase φ(t) of a slow band. The amplitude distribution over the phase bins is P(j), and
    MI = (log N − H(P))/log N.
  * Ours: φ = phase of the pickup line (2.2 or 5 kHz) along the CDS sequence. "Amplitude" = |e_k| or the in-window step
    indicator of ch2.
  * Reveals: whether ch2's switching rate is modulated by the pickup. A trap's capture/emission rates depend exponentially on
    the gate bias, so a bias ripple would modulate the RTS rate [speculative].
  * Variant: φ = position along the baseline droop. That variant is partly covered by
    `noise: noise trend along skips (ADU/100 skips)`.
  * Cost: trivial. ~5,000 skips per trace give ~500 per phase bin with 10 bins. Medium.

### 1.3 Electrophysiology: single-channel (patch-clamp) recordings (added field)

A single ion channel switches between closed and open: a textbook two-level RTS in white noise, sampled at high rate. This
field has the most mature hand tools for exactly the process suspected in ch2.

* **Idealization by half-amplitude threshold, or a hidden Markov model (HMM).**
  * Def: (a) threshold the low-pass-filtered current at half the open level, giving an open/closed sequence. (b) Better at low
    SNR: fit a 2-state HMM with Gaussian emissions (Baum-Welch) and run Viterbi. Chung et al. [7] (checked, record only)
    introduced HMM processing for currents buried in noise.
  * Ours: 10-sample block means of the residual cycles *within* the windows (5,000 skips × 2 windows × 15 blocks per trace).
    Across the dip and reset regions the level is unobservable, so treat those as gaps.
  * Reveals: RTS amplitude ΔI, mean dwell times τ_up and τ_down, and the occupancy ratio. These are the trap parameters of
    Kirton & Uren [24].
  * New. Medium-high.
  * Cost: a 2-state HMM forward-backward on ~10⁵ points is milliseconds in numpy. Robust only when Δ/σ_block ≳ 3. For ch2
    reference, Δ ≈ 50 ADU and σ_block ≈ 8 ADU, so it qualifies.
* **Log-binned dwell-time histograms (Sigworth & Sine 1987)** [6] (checked; open full text at PMC1330104).
  * Def: histogram of log(dwell time) with the square root of the counts on the vertical axis. Each exponential component
    becomes a peak of fixed width at log τ_i, so mixtures of time constants are visible by eye.
  * Ours: dwell times from the idealization above.
  * Reveals: one trap (one peak per state) or several, and τ_c and τ_e directly.
  * Diagnostic (D4) plus the feature `noise: RTS mean dwell time (us)`.
* **Single-step changepoint inside a window** (a simplified idealization) [pilot].
  * Def: per window, block means b_1..b_15 (10 samples each). For every split j, t_j = Δ_j / (σ_b √(1/j + 1/(m − j))). Keep
    max|t_j| and Δ at that split. σ_b is the robust σ of the first differences of the block means divided by √2.
  * Results: rate of |t| > 6 = 0 % (ch0), 6.3 % (ch2 reference), 13.2 % (ch2 V low rails). Up/down counts are balanced.
  * Proposed features: `discrete: in-window step rate (%)`, `discrete: in-window step amplitude (ADU)` (median |Δ|).
  * New. **High.** Vectorized: about 1 s per trace.

### 1.4 Seismology

* **STA/LTA trigger** [8] (Allen 1978, checked via citing papers). The exploratory report proposed it (13c).
  * Def: the ratio of the short-term mean to the long-term mean of a characteristic function (here e_k²). The LTA window
    precedes the STA window.
  * Pilot: the maximum of STA(5 skips)/LTA(50 skips) per pixel is 3.6-4.8 on ch0/1/3 and **9.9-35 on ch2**.
  * Unlike kurtosis, it localizes the burst in the skip sequence. Feature `discrete: STA/LTA maximum`. New. Medium.
* **Coda-wave interferometry (stretching).** Exploratory idea 7. Not repeated.
* **Spectral ratios (H/V style).**
  * Def: the ratio of two spectra of the same excitation, which cancels the source and leaves the transfer function.
  * Ours: the CDS spectrum of a trace divided by that of the packet's reference trace, same amplifier.
  * It turns the 40-bin `noise: CDS spectrum, log frequency` curve into a "what changed" curve with the stationary pickup
    cancelled.
  * Not a scalar. Medium as a derived curve for Level-1 inputs.
* **Coda Q / decay of the energy envelope** [Aki & Chouet 1975, *JGR* 80:3322, unverified].
  * Def: fit ln(A(t)·t) ∝ −πft/Q.
  * Ours: the squared envelope of the first-skip ringing (skips 1-19), giving a quality factor Q = πf_ring·τ_ring of the
    7.2-kHz resonance (Section 3).
  * Equivalent to the existing `fit: first-skip ringing decay (skips)` × frequency. Low (a re-expression).
* **Ambient-noise cross-correlation** [10] (Bensen et al. 2007, checked).
  * Not applicable between amplifiers: one amplifier per trace (exploratory, 13d).
  * Applicable *within* a trace between two signals recorded simultaneously: the per-skip baseline (pedestal level, mostly kTC
    reset noise plus droop) and the CDS.
  * Their cross-correlation at lags 0 and ±1 tells whether the CDS noise shares a source with the reset level, e.g. a supply
    disturbance moving both.
  * Lag 0 is close to `noise: baseline coupling (1e-3)`, but that feature is a regression on the slow droop. New: the
    correlation of the *detrended* baseline and CDS residuals, `noise: baseline-CDS residual correlation`. Medium-low.

### 1.5 Audio and speech

* **Spectral flatness** (Wiener entropy).
  * Def: geometric mean / arithmetic mean of the PSD.
  * Pilot: on the CDS spectrum, 0.47-0.54 (ch0/1/3) and 0.55 (ch2). For white noise with a Hann window, expect ≈ e^−γ ≈ 0.56.
    Weak and redundant with the line count. Low.
* **Spectral centroid and rolloff.**
  * Def: Σ f S / Σ S; the frequency below which 85 % of the power lies.
  * Ours: the within-cycle spectrum. A single-number summary of the four `noise: within-cycle power` bands. Low.
* **Zero-crossing rate.**
  * Def: the fraction of sign changes of e_k.
  * For a Gaussian series it is arccos(ρ₁)/π, so it is redundant with ρ₁. Low.
* **Onset detection: spectral flux, high-frequency content** [13] (Bello et al. 2005, checked).
  * Def: flux = Σ_f max(0, |X_t(f)| − |X_{t−1}(f)|) between short frames.
  * Ours: frames = consecutive skip cycles, X = FFT of the residual cycle.
  * A per-skip "something new happened" score, sensitive to glitches and to charge arriving during readout. More general than
    the existing `discrete: largest CDS step (t)`, which sees only CDS level steps.
  * New: `discrete: max spectral flux (z)`. Medium-low.
* **Cepstrum and MFCC** [12] (Davis & Mermelstein 1980, checked).
  * Def: IFFT(log|FFT|).
  * Use: periodic structure in a spectrum (harmonic families) shows as a single quefrency peak.
  * Ours: the cepstrum of the CDS spectrum. A pickup with harmonics (e.g. 2.2, 4.4, 6.6 kHz) gives one peak at 1/2.2 kHz.
    The cepstrum of the within-cycle spectrum would show residual skip-rate harmonics.
  * MFCCs themselves (a mel scale) have no physical meaning here. Reject them.
  * New: `discrete: cepstral peak (ms)`. Low-medium.
* **Teager-Kaiser energy operator** [14] (Kaiser 1990, checked via citations).
  * Def: Ψ[x_n] = x_n² − x_{n−1}x_{n+1}, an instantaneous energy that emphasizes sharp transients.
  * Ours: raw samples at the reset edge and the SW dip.
  * Overlaps the existing `edge:` features. Low.

### 1.6 Finance: high-frequency econometrics

* **Realized variance, bipower variation, MedRV; jump share** [15, 16] (both checked).
  * Def: RV = Σ r²; BV = (π/2) Σ|r_i||r_{i−1}|; MedRV = π/(6 − 4√3 + π) · N/(N−2) · Σ med(|r_{i−1}|, |r_i|, |r_{i+1}|)².
    Jump share = 1 − MedRV/RV.
  * Ours: e_k (the "returns" are the per-skip CDS deviations).
  * Piloted in the exploratory report: ch2 has a 16-41 % jump share; the others have none. **Not yet in `FEATURES`.**
  * Proposed: `noise: jump-robust CDS sigma (ADU)`, `discrete: jump share of CDS variance`. High; 3 lines of numpy.
* **Lee-Mykland jump test** [17] (checked).
  * Def: L_k = r_k / σ̂_k, with σ̂_k the local bipower volatility. The maximum over k is compared with the Gumbel distribution
    of the maximum of N Gaussians.
  * Ours: gives **calibrated** jump counts per pixel: `discrete: jump count (Lee-Mykland, 1 %)`.
  * Better than the fixed 5σ cut of `discrete: CDS outlier fraction`, because the local volatility adapts to the noise trend
    along the skips. Medium.
* **Volatility clustering: Ljung-Box / Engle LM test on e²** (exploratory 15d).
  * Pilot: LB(e², 10 lags) = 8-13 on ch0/1/3, near its χ²₁₀ expectation of 10; 12-24 on ch2. Modest.
  * `noise: squared-residual autocorrelation (LB10)`. Low-medium.
* **Hurst exponent (R/S).** Same information as DFA and the Allan slope. Low.
* **Roll's spread / negative lag-1 autocovariance** (exploratory 15b). Existing: `noise: CDS autocorrelation at lag 1`.
* **Variance-ratio test (Lo-MacKinlay)** [unverified].
  * Def: VR(q) = Var(q-sum)/(q·Var(1)).
  * This is the Allan ratio under another name: `skip: noise of 10-skip average` / (`skip: CDS rms` /√10) is VR(10)^{1/2}.
    Existing in effect.

### 1.7 Machine-vibration diagnostics

* **Spectral kurtosis (SK) and the kurtogram** [18] (Antoni 2007 fast kurtogram, checked; the 2006 SK paper details are
  [unverified]). The exploratory report proposed it (23a).
  * Def: SK(f) = ⟨|X(t,f)|⁴⟩/⟨|X(t,f)|²⟩² − 2 over STFT frames.
  * Our extension: compute it on **the residual cycles, with frame = one window** (150 samples), across all skips. Bands
    where ch2's noise is impulsive show SK > 0; for Gaussian noise SK = 0.
  * Our pilot shows the null CDS is heavy-tailed. SK will tell whether the steps also carry a band-limited component (an
    oscillation) or are pure steps (power at low f only).
  * New: `discrete: max within-window spectral kurtosis` and the band where it occurs. Medium.
* **Envelope analysis** [19] (Randall & Antoni 2011, checked; open copy on HAL).
  * Def: band-pass around a resonance; envelope |Hilbert|; spectrum of the envelope; look for repetition frequencies.
  * Ours: band-pass the raw trace around the 0.1-0.3 MHz component seen inside the windows (the structure function's ~6.7-µs
    periodicity, 1.9 below). The envelope spectrum shows whether its amplitude is modulated at the pickup frequencies.
  * Diagnostic. Medium-low.
* **Cepstrum for periodic families** [20] (Randall's history of cepstrum analysis, MSSP; year 2016/2017 [unverified]). See 1.5.
* **Crest factor.**
  * Def: max|x|/rms.
  * Pilot on e_k: 3.0-3.5 on ch0/1/3, 5.6-7.4 on ch2.
  * Partly redundant with `discrete: CDS excess kurtosis` and `discrete: CDS outlier fraction`, but it reads directly as
    "largest excursion in σ". Low-medium.
* **Order tracking and synchronous averaging.**
  * Def: resample to the shaft angle, then average.
  * Ours: **existing.** The cycles are cut at the skip period (`period_exact`), and the equivalent-time pulse
    (`edge: equivalent-time pulse`) is order tracking at sub-sample resolution.

### 1.8 Radar and sonar

* **Matched filter** [Turin 1960, *IRE Trans. IT* 6:311, unverified].
  * Def: correlate with a known template; optimal SNR in white noise.
  * Ours: (a) **the CDS itself is a matched filter for a step** between two windows (Hopkinson & Lumb [45], record not found).
    (b) A matched filter for the *first-skip transient template* (the mean of the CDS of skips 0-29 over all pixels) gives
    each pixel's transient amplitude with all 30 skips, not just skip 0. Its noise is ≈ σ_CDS/‖template‖, about 30 % lower
    than E0 alone.
  * New: `start: transient amplitude, matched filter (ADU)`. Medium.
* **CFAR detection** [21] (Rohling 1983, OS-CFAR, checked).
  * Def: threshold = α × the k-th order statistic of the neighbouring cells, so the false-alarm rate stays fixed when the
    noise level changes.
  * Ours: (a) line detection in the CDS spectrum. Existing `discrete: CDS spectrum lines` already uses a local-median rule
    (a CA/OS-CFAR variant); make the threshold give a stated false-alarm probability rather than "5×". (b) Glitch detection in
    the per-cycle residual energy: cell = one skip cycle, reference = ±20 cycles.
  * New: `discrete: CFAR cycle-glitch count`. Low-medium.

### 1.9 Gravitational-wave detector characterization

* **PSD whitening and glitch classification** [22] (LIGO-Virgo 2020 guide, checked), Gravity Spy [23] (checked).
  * Def: divide the data's FFT by sqrt(PSD), which makes stationary noise white; glitches then stand out in Q-transform
    images and are classified by morphology.
  * Ours: whiten the CDS sequence by its own Welch PSD, then apply kurtosis and the outlier fraction to the *whitened* series.
    The existing `discrete:` features act on the coloured series, where 1/f and pickup inflate them.
  * New: `discrete: whitened CDS kurtosis`. Medium.
* **Spectral line tracking.**
  * Def: track each narrow line's frequency, amplitude and phase over time; build line lists; identify the coupling path.
  * Ours: at the trace level, the pickup lines (~2.2 and ~5 kHz in one example).
    * Their *frequencies* identify the source (mains harmonic, switching regulator, fan).
    * Their *phase relative to the readout clock* identifies whether they are clock-generated: phase-locked to each pixel's
      first skip means the readout makes them; random phase means an external source. This uses the absolute sample times
      `ped_starts`.
  * Existing: only the count (`discrete: CDS spectrum lines`).
  * New: `discrete: strongest line frequency (kHz)`, `discrete: strongest line amplitude (ADU)`,
    `discrete: line phase coherence` (|mean of e^{iφ_p}| over pixels, with φ_p the line phase at each pixel's start). High.
* **Null streams** (combinations of channels with the signal cancelled).
  * Ours: the **null CDS** of 1.3 (half-window differences: same charge, no transfer). New. High; see Section 0.1.

### 1.10 Time and frequency metrology (we already use the Allan deviation)

Reference: Riley & Howe, *Handbook of Frequency Stability Analysis*, NIST SP 1065 [25] (checked; open PDF at nvlpubs.nist.gov).

* **Modified Allan deviation** (MDEV) and **time deviation** (TDEV = τ·MDEV/√3).
  * Def: average n adjacent samples *before* differencing (a second averaging stage).
  * MDEV separates white phase-like from flicker phase-like noise, which the ordinary Allan deviation cannot: both give
    slope −1 in ADEV.
  * Ours: the analogue of "phase" is the cumulative sum of e_k. In practice: MDEV of e_k separates a white CDS from an
    MA(1)-anticorrelated one (ch2, ρ₁ < 0).
  * New curve. Medium.
* **Hadamard deviation.**
  * Def: the third difference of n-skip means, HVAR = ⟨(m_{i+2} − 2m_{i+1} + m_i)²⟩/6.
  * It is **insensitive to a linear drift**, so it can be applied to the raw CDS without the degree-4 polynomial. The existing
    notes warn that the polynomial biases the Allan points at n > N/5 low (0.4 of white at n = 160).
  * New: `noise: Hadamard deviation n=50 (ADU)` and curve. **Medium-high**, because it removes a known bias.
* **Noise-type identification from the lag-1 autocorrelation** (Riley & Greenhall [26]; described with expected values in
  arXiv:1604.05076, checked; the original 2004 paper is [unverified]).
  * Def: for n-skip means, ρ₁(n) is about −1/2 for white "phase", 0 for white "frequency" and +1/3 for flicker "frequency".
    The power-law exponent α at each averaging factor n follows from ρ₁ via a small table.
  * Ours: compute ρ₁ of the n-skip block means for n = 1, 2, 5, 10, 25. This gives the noise *type* at each time scale,
    which is more robust than the slope of 3-5 Allan points.
  * New: `noise: noise-type exponent at n=10` (−2 … +2). Medium.
* **Total / Theo deviation** (bias-reduced at long τ). Low: our long-τ end is short anyway.
* GMWM composite fits: exploratory idea 8. Not repeated.

### 1.11 Electronics and semiconductor noise (1/f, RTS)

* **RTS physics** (Kirton & Uren 1989 [24], checked, record only).
  * A single oxide trap near the channel of a small MOSFET switches the drain current between two levels. Capture and
    emission times are exponential and bias- and temperature-dependent. The spectrum is a Lorentzian
    S(f) = 4ΔI² / ((τ_c + τ_e)((1/τ_c + 1/τ_e)² + (2πf)²)) (Machlup 1954 [27], [unverified]: record not found).
  * Ours: the in-window step amplitude (1.3) is ΔI × gain. τ_c and τ_e come from the dwell-time histograms.
* **RTS and CDS in image-sensor source followers** (Wang, Rao, Mierop & Theuwissen, IEDM 2006 [28], checked; full text hosted
  by Teledyne DALSA; Goiffon et al., IISW 2011 [29], checked).
  * Wang et al. found that source-follower 1/f noise in pinned-photodiode CMOS pixels comes from very few traps and appears as
    RTS. They also describe how CDS responds: the CDS noise is largest when the trap occupancy is the same during the two
    samples.
  * Goiffon et al. note that fluctuations slower than the inter-sample time have no effect on the read noise.
  * Direct relevance: for ch2, the CDS variance from an RTS depends on τ_RTS relative to the 207-sample (13.8-µs) window
    separation. Our pilot gives a direct handle on that: the step rate per 10-µs window.
* **1/f + white fit and corner frequency.**
  * Def: S(f) = W + A/f^γ. The corner f_c = (A/W)^{1/γ}. Hooge's empirical law expresses A through α_H/N
    (Hooge 1969 [30], title wording [unverified]).
  * Ours: fit the existing 40-bin `noise: CDS spectrum, log frequency` curve with a few lines of `scipy.optimize.curve_fit`.
  * The corner frequency is the single most standard number of an amplifier's noise; the existing features only give band
    ratios.
  * New: `noise: 1/f corner frequency (kHz)`, `noise: 1/f exponent`. High-medium.
* **Lorentzian + white fit** for ch2 (an RTS with one dominant trap).
  * Gives the corner f_L = (1/τ_c + 1/τ_e)/2π, comparable to the dwell times.
  * New: `noise: Lorentzian corner (kHz)`. Medium (ch2-specific).
* **Current-noise spectroscopy / bias dependence.** The equivalent here is the dependence of the RTS rate and amplitude on
  VDD, VR and VDRAIN across packets. That is Level-1/2 work, made possible by the features above.

### 1.12 Power systems

* **Harmonic analysis** (IEC 61000-4-7 [unverified, standard]).
  * Def: amplitudes of the fundamental and its harmonics in synchronized windows; THD = sqrt(Σ_{h≥2} A_h²)/A_1.
  * Ours: for the pickup lines, check whether 2.2 and 5 kHz are harmonics of a common fundamental, and give a THD-like
    "harmonic share" of the line power.
  * New, folded into the line features (1.9). Low as a separate feature.
* **Transient detection with wavelets** (Santoso et al., IEEE Trans. Power Delivery 1996 [31]; OSTI record checked, DOI
  10.1109/61.489353; the author list was not confirmed in the record).
  * Def: dyadic wavelet detail coefficients localize sudden events in time and scale.
  * Ours: the detail coefficients of the per-skip CDS (scales 1-4 skips) localize steps and the start transient. This
    overlaps STA/LTA and the step detectors. Low.
* **Prony analysis / matrix pencil for electromechanical modes** [32] (Hua & Sarkar 1990, checked).
  * Def: fit y_k = Σ_i A_i z_i^k with complex poles z_i = e^{(−1/τ_i + 2πif_i)Δ}, using the SVD of a Hankel matrix. Damping
    and frequency of several modes come out at once, with no grid search.
  * Ours: **the start-of-pixel transient**, i.e. the per-skip in-window slope sequence (or the CDS) of skips 0-30.
    * Section 3 shows that the transient has at least two modes: a fast exponential (τ ≈ 12 µs) and the 7.2-kHz ringing
      (τ ≈ 2.5 skips).
    * The existing grid fit (`fit: first-skip ringing …`) handles one mode and leaves skip 0 out.
    * Matrix pencil on the mean over pixels of CDS_k − median, k = 0..29, with model order 3-4, gives all modes in physical
      units.
  * New: `start: mode frequencies (kHz)`, `start: mode damping (us)`. **Medium-high.** About 20 lines of numpy SVD.

### 1.13 Astronomy: photometric time series

* **RMS against bin size and the red-noise β factor** (Pont, Zucker & Queloz 2006 [33], checked).
  * Def: β = σ_N / (σ_1/√N · sqrt(M/(M − 1))) for bins of N points (M bins).
  * Ours: β at N = 50 = `skip: noise of 50-skip average` × √50 / `skip: CDS rms (detrended)`. **Derivable from existing
    features.** Add only as a named ratio if the Level-1 models need it. Low.
* **Lomb-Scargle periodogram** [34] (VanderPlas 2018, arXiv:1703.09824, checked; ApJS volume [unverified]).
  * Def: least-squares sinusoid fit at each frequency; works for uneven sampling.
  * Ours: **concatenate all pixels of a trace in absolute time** (the skips of different pixels are separated by transfer
    gaps, so the sampling is uneven).
    * This gives a periodogram of the whole 0.2-s trace with ~10× finer frequency resolution (5 Hz instead of ~60 Hz).
    * It separates pickup lines that are coherent across pixels (external) from those that reset at each pixel.
  * New, folded into the line features (1.9). Medium-high.
* **SysRem / PCA detrending of many light curves** [35] (Tamuz, Mazeh & Zucker 2005, checked).
  * Def: rank-1 (or rank-k) factorization of the star × time matrix, removing systematics common to many stars.
  * Ours: the pixel × skip CDS matrix (8 × 490).
    * Rank 1 with a common amplitude is the existing `level: common CDS pattern`.
    * SysRem allows a **pixel-specific amplitude** of the common pattern. If the amplitudes differ (e.g. row-start pixels, or
      pixels with charge), the common pattern has a charge- or position-dependent coupling.
  * New: `level: common-pattern amplitude spread` (robust spread of the per-pixel coefficients) and
    `level: second systematic fraction`. Medium.
* **Structure function** [36] (Simonetti, Cordes & Heeschen 1985, checked via citing papers).
  * Def: D(τ) = ⟨(x(t + τ) − x(t))²⟩. A white process gives a constant; a process correlated over τ_c rises and saturates at
    τ_c; an RTS rises roughly as 1 − e^{−2τ/τ_d}.
  * Ours: **15-sample block means of the residual cycles, at lags inside the windows and across them** [pilot]:

    | Amplifier (trace) | sqrt D at lag 1, 3, 6, 9 blocks (1-9 µs) (ADU) | across windows (ADU) |
    |---|---|---|
    | ch0 reference | 11.8, 15.9, 15.0, 12.3 | 14.8 |
    | ch1 reference | 10.8, 14.1, 13.6, 11.8 | 13.5 |
    | ch3 reference | 11.0, 14.9, 14.6, 13.0 | 14.5 |
    | ch2 reference | 14.7, 22.9, 25.7, 25.8 | 29.0 |
    | ch2 V low rails | 15.9, 27.3, 34.4, 38.7 | 42.6 |

    * The good amplifiers show a rise and fall with a maximum at ~3 blocks (3 µs): an oscillatory component with a period of
      ~6.7 µs (~150 kHz), consistent with the 0.1-0.3 MHz band of the within-cycle spectrum.
    * ch2's excess over ch0 (subtracted in quadrature) grows over the whole window: 8.8 → 22.7 ADU (reference), 10.4 →
      36.6 ADU (V low rails). That is the signature of a switching process with dwell times longer than ~10 µs.
  * New: curve `noise: within-cycle structure function`, scalar `noise: structure-function excess at 9 us (ADU)`. High-medium.

### 1.14 Oceanography and climate

* **Multitaper spectrum and Thomson's harmonic F-test** [37] (Thomson 1982, checked).
  * Def: average K eigenspectra computed with Slepian tapers. The F-statistic at each frequency tests a sinusoid of
    unknown amplitude against the local continuum.
  * Ours: **the right tool for the pickup lines in the CDS.** It is a statistical test with a stated false-alarm probability,
    unlike the "5× local median" rule of `discrete: CDS spectrum lines`.
    * With 490 skips and time-bandwidth NW = 3-4, the resolution is ~2NW/N × the skip rate ≈ 0.4-0.5 kHz.
    * `scipy.signal.windows.dpss` provides the tapers.
  * New: `discrete: line count (F-test, p<0.001)` and the strongest line's frequency and amplitude. High.
* **EOF / PCA** [38] (Hannachi, Jolliffe & Stephenson 2007, checked).
  * Existing as `slow: PC1-3 variance fraction` (residual cycles). The new use is SysRem-like on the pixel × skip CDS matrix
    (1.13).
* **Wavelet power and coherence** [39] (Torrence & Compo 1998, checked).
  * Def: continuous Morlet transform with red-noise significance contours; coherence between two series.
  * Ours: the scalogram of the per-skip CDS concatenated over pixels shows when pickup lines appear and vanish, and where
    bursts sit. Coherence between the per-skip baseline and the CDS, as a function of frequency, localizes the
    `noise: baseline coupling` in frequency.
  * Diagnostic (D7). Medium-low as features.

### 1.15 Structural health monitoring (SHM)

* **Modal parameter estimation: log decrement, half-power bandwidth, Prony / matrix pencil** (Farrar & Worden 2012 book [40],
  checked).
  * Def: damping ratio ζ = δ/sqrt(4π² + δ²), with δ = ln(x_k/x_{k+1}) between successive peaks.
  * Ours: the first-skip ringing has f ≈ 0.235 cycles/skip ≈ 7.2 kHz and τ ≈ 2.3-2.9 skips ≈ 75-95 µs, so
    ζ = 1/(2πf τ) ≈ 0.23-0.29. That is a strongly damped resonance.
  * Use the matrix pencil (1.12). The re-expression `start: ringing Q` is low priority.
* **Novelty index** (Mahalanobis distance of the feature vector from a baseline population) [Worden, Manson & Fieller 2000,
  *JSV*, unverified].
  * Ours: the distance of each trace's feature vector from the reference-group cloud of the same amplifier. A one-number
    "is this amplifier in a new state" monitor, complementary to the exploratory report's Westgard rules.
  * Not a trace feature (it needs a population). Medium for monitoring.

### 1.16 NMR and spectroscopy

* **Lineshape fitting (Lorentzian / Gaussian / Voigt).**
  * Ours: fitting the pickup-line profile in the multitaper spectrum separates a pure tone (instrumental width) from a
    wandering frequency (broader line, e.g. a switching regulator whose frequency drifts with load). Low-medium.
* **Baseline correction by asymmetric least squares / Whittaker smoother** [41] (Eilers 2003, checked; Eilers & Boelens 2005
  AsLS, unpublished report, checked via the pybaselines documentation).
  * Ours: a Whittaker-smoother detrend of the baseline droop (one λ) instead of the degree-4 polynomial. It follows the
    exponential droop better, without the polynomial's end effects (the stated n > N/5 bias). Low-medium (a processing change,
    not a feature).

### 1.17 Chromatography

* **Peak detection and deconvolution with the exponentially modified Gaussian (EMG)** [42] (Grushka 1972, checked via
  citations).
  * Def: a Gaussian (σ) convolved with a one-sided exponential (τ). The moments give σ² + τ² and the skew 2τ³.
  * Ours: the **reset pulse** in `edge: equivalent-time pulse`. One EMG fit gives a bandwidth-like σ and a tail τ together.
    The existing features give a rise time and a two-exponential recovery separately. Also the SW dip.
  * New: `edge: reset-pulse EMG sigma (ns)`, `edge: reset-pulse EMG tau (ns)`. Low-medium (overlaps existing edges).
* **Savitzky-Golay derivative filters** [Savitzky & Golay 1964, *Anal. Chem.* 36:1627, unverified].
  * Def: local polynomial least squares, which gives smoothed derivatives.
  * Ours: the in-window baseline slope of Section 3 is a degree-1 Savitzky-Golay derivative over one window. Use it as the
    implementation of the slope features.

### 1.18 Telecom and serial links

* **Eye diagram.**
  * Def: overlay of all symbol periods. Eye height and width give the margin.
  * Ours: overlay all skip cycles as a 2-D histogram (position in cycle × ADU, log colour). Existing in numbers as
    `edge: variation along the cycle`.
  * The *picture* shows multi-level bands: for ch2, two bands inside the windows where the RTS lives. Diagnostic D9.
* **Jitter decomposition RJ/DJ, dual-Dirac** [43] (Keysight white paper 5989-3206, checked).
  * Def: the TIE histogram is modelled as a Gaussian (RJ, σ) convolved with two Diracs separated by DJ(δδ), fitted to the
    tails. The TIE spectrum separates periodic jitter (PJ) at discrete frequencies.
  * Ours: the reset-edge time residuals already computed for `edge: jitter (ns)` (total only).
  * New:
    * `edge: random jitter (ns)` from the tail fit;
    * `edge: deterministic jitter (ns)` = DJ(δδ);
    * `edge: periodic jitter frequency (kHz)`. If the edge-time residuals carry the 2.2/5-kHz pickup, the pickup modulates
      timing (a clock or supply path), not only amplitude.
  * Medium.
* **SNR / EVM.** Not meaningful without a known symbol constellation. The electron-peak resolution (WADERs) plays this role.
  Reject.

### 1.19 Nonlinear dynamics and complexity

* **Permutation entropy** [44] (Bandt & Pompe 2002, checked).
  * Def: Shannon entropy of the ordinal patterns of order m, normalized by log m!.
  * Pilot: order 3 gives 0.995-0.999 on every amplifier, including ch2. Ordinal patterns ignore amplitudes, so rare large
    jumps barely change them. **Reject** for this purpose.
* **Recurrence quantification analysis (RQA)** [46] (Marwan et al. 2007, checked).
  * Def: recurrence matrix R_ij = Θ(ε − |x_i − x_j|). Determinism (DET) = the fraction of recurrent points on diagonal lines;
    laminarity (LAM) = the fraction on vertical lines.
  * Ours: on the *block-mean level sequence inside windows* (not the CDS), an RTS gives a checkerboard of laminar blocks, so
    LAM is high.
  * Diagnostic D11. Cost O(n²) with n ~ 10⁴ blocks, so subsample. Low as a feature: the HMM gives the same information
    quantitatively.
* **Sample entropy.** See 1.1.

### 1.20 Statistics: higher-order spectra, stationarity, changepoints

* **Bispectrum / bicoherence and Hinich's Gaussianity and linearity tests** [47] (Hinich 1982, checked).
  * Def: B(f1, f2) = E[X(f1)X(f2)X*(f1 + f2)]; bicoherence = |B|²/(E|X1X2|² E|X3|²). Hinich's statistic tests B = 0
    (Gaussianity) and constant bicoherence (linearity).
  * Ours: the CDS sequence (490 skips per pixel, ~8 pixels: use segments of 64 skips and average). Two uses:
    * (a) A Gaussianity test that is not dominated by one outlier, as kurtosis is.
    * (b) **A curious coincidence worth testing [speculative]:** in one example the pickup lines are at ~2.2 and ~5 kHz, and
      the first-skip ringing is at 0.235 × 30.69 = **7.2 kHz ≈ 2.2 + 5 kHz**. If the ringing were a mixing product, the
      bicoherence at (2.2, 5) kHz would be large. More likely it is a coincidence: the ringing is identical on all four
      amplifiers and in all settings.
  * New: `noise: Hinich Gaussianity statistic`. Low-medium.
* **Stationarity tests (KPSS, ADF).** Our series are detrended by construction. Low.
* **Changepoint detection (PELT)** [48] (Killick, Fearnhead & Eckley 2012, checked).
  * Def: exact penalized segmentation in linear time.
  * Ours: segment the in-window block-mean sequence of ch2 (the multi-step generalization of the single-step pilot), or
    segment the per-skip CDS into level regimes (bursts).
  * New: `discrete: changepoint count per 1000 skips` (with a BIC penalty). Medium (the single-step version is the cheap
    first step).
* **Cyclostationarity** [49] (Gardner 1991, checked; the exploratory report adds Antoni 2009).
  * Existing in spirit (cycles cut at the skip period). Exploratory idea 1 covers the rest.

### 1.21 CCD / CMOS image-sensor engineering (directly related fields)

* **Skipper-CCD readout** (Tiffenberg et al. 2017 [50], Cancelo et al. 2021 LTA [51], Chierchie et al. 2021 smart readout
  [52], all checked).
  * They establish the σ/√N averaging that the `noise:` features test.
  * I did **not** find a paper that characterizes the first-skip / start-of-pixel transient (a search for "first skip"
    discarding returned nothing specific). The analysis in Section 3 appears to be new for this project, and possibly in
    general [unverified absence].
* **Deferred charge / serial CTI** (Snyder et al. 2021, *JATIS* 7(4):048002 [53]; Snyder & Roodman, arXiv:2001.03223 [54],
  checked).
  * In LSST ITL CCDs, excess deferred charge is modelled as trapping plus a **signal-dependent electronic offset** that
    appears in the next pixels.
  * Ours: the analogue is the dependence of pixel p+1's first-skip transient on pixel p's charge. Section 3.5 proposes the
    measurement. It needs traces with charge (none in 2026-10-12).
* **CDS transfer function** (Hopkinson & Lumb 1982 [45], record not found; cited in arXiv:1511.05649, checked).
  * The CDS is a band-pass whose response follows from the two windows.
  * The baseline-slope identity of Section 3 is its low-frequency limit: H(f) ≈ 2πif·L for f ≪ 1/(window length).

---

## 2. Consolidated candidate feature table

Proposed names follow the `group: name (unit)` style. **`start:` is a proposed new group** for the start-of-pixel transient.
The rows could instead go under `skip:` and `fit:` if a new group is unwelcome.

Notation:
* e_k = detrended CDS of skips ≥ 10 (`det[p]` in `trace_features`);
* R = residual cycles (`res[p]`);
* W_k = the samples of window w (pedestal or signal) of skip k;
* slope(·) = least-squares slope against the sample index (vectorized: `W @ tc / (tc @ tc)`, with `tc = t − t.mean()`);
* "late" = median over skips 20-59;
* Δ = slope(ped) + slope(sig), averaged: 0.5 · (slope(ped) + slope(sig)).

Priority: **H** = implement first, **M** = worthwhile, **L** = low or redundant.

| # | Proposed feature | Definition | Region | Physical meaning | Field | New / existing | Prio | Implementation notes |
|---|---|---|---|---|---|---|---|---|
| 1 | `start: in-cycle slope at skip 0 (ADU/us)` | 0.5·(slope(ped₀) + slope(sig₀)) − late, × 15 | raw samples, skip 0 | derivative of the baseline transient at the start of the pixel | SG filters (chromatography), CDS theory | new | H | vectorized over skips; per pixel → median and spread |
| 2 | `start: CDS-slope lever arm (samples)` | slope of (CDS_k − median(CDS_{≥10})) regressed on (Δ_k − late), skips 0-19, pooled over usable pixels | CDS + raw windows, first 20 skips | ≈ −(window-centre separation) if the start transient is pure baseline slope; deviations = a charge or gain component | CDS transfer function | new | H | pilot: −211 to −221 (ch0/1/3), −157 to −204 (ch2, attenuated by noise in x) |
| 3 | `start: slope-explained fraction` | R² of #2 | same | how much of the first-skip pattern is baseline slope | — | new | H | pilot 0.95-0.98 (ch0/1/3), 0.62-0.94 (ch2) |
| 4 | `start: transient time constant (us)` | τ of A·e^{−t/τ} + c fitted to D₀(t) = (skip 0 − its pedestal mean) − (mean of skips 20-59, each minus its pedestal mean), on the pedestal and signal windows only | raw samples, skip 0 | recovery time of the video chain or output stage after the H-clock burst | Prony/SHM; exponential settling | new | H | 1-D grid in τ + linear LS (as `fit_exponential`). Pilot 11.1-12.4 µs |
| 5 | `start: transient amplitude (ADU)` | A of #4 (at the start of the pedestal window) | same | size of the H-clock-induced disturbance | — | new | H | pilot 560-790 ADU |
| 6 | `start: first-skip excess per volt H (ADU/V)` | `skip: first-skip CDS excess` / (H1_A_H − H1_A_L) from the .meta | CDS skip 0 + meta | H-clock coupling × amplification, like `edge: dip per volt SW` | — | new | H | needs meta; pilot: the excess scales with the H swing (+22-24 % for +18-22 % swing) |
| 7 | `start: row-start/inner first-skip ratio` | median E0 of the row-start pixels / median E0 of the inner pixels | CDS skip 0, row-start pixels | memory of the clocking before the pixel (prescan, vertical transfers) | — | new | M-H | pilot 0.93 (NCOLPRE 0), 0.41-0.48 (NCOLPRE 185) |
| 8 | `start: reset jump skip 0→1 (ADU)` | pedestal mean of skip 1 − that of skip 0 | per-skip baseline | first reset after the transfers; operating point of the sense node | — | new | M | pilot 1,500-2,300 ADU, channel-dependent; `skip: baseline droop curve` starts at skip 1 and misses it |
| 9 | `start: mode frequencies (kHz)` and `start: mode damping (us)` | matrix-pencil poles (order 3-4) of the mean over pixels of CDS_k − median, k = 0..29, converted with the skip rate | CDS, first 30 skips | all modes of the transient (fast exponential, 7.2-kHz resonance), in physical units | power systems (Prony), SHM | new (extends `fit: first-skip ringing …`) | M-H | numpy SVD of a Hankel matrix; ~20 lines |
| 10 | `start: transient amplitude, matched filter (ADU)` | ⟨CDS_{0..29} − median, T⟩/⟨T, T⟩ · ‖T‖∞, with T = the trace's mean transient template | CDS, first 30 skips | per-pixel transient size with ~30 % less noise than E0 | radar (matched filter) | new | M | use for the pixel-to-pixel and charge-dependence studies |
| 11 | `start: ringing frequency (kHz)` | `fit: first-skip ringing frequency (cyc/skip)` × skip rate | — | the resonance is (probably) fixed in time units: test by changing the skip period | — | re-expression of existing | M | one line; makes settings with different `delayInt` comparable |
| 12 | `start: charge memory (ADU/kADU)` | slope of E0(p+1) against the relative level of pixel p, over pixels with charge | CDS skip 0, consecutive pixels | signal-dependent offset after a charged pixel (deferred-charge analogue) | astronomy CCDs (LSST deferred charge) | new | M (needs data with charge) | NaN without charged pixels, like `level: charge-loss slope` |
| 13 | `level: slope-corrected CDS rise (ADU)` | `level: common CDS rise` + L·(Δ̄_late − Δ̄_early), with L from #2 | CDS + raw windows, all skips | part of the CDS rise *not* explained by baseline slope: a genuine gain or charge effect | — | new | H | pilot: rise 28.9 ADU, slope prediction 28.2 ADU → residual 0.7 ADU on ch0 reference |
| 14 | `noise: in-window drift vs baseline (1/ms)` | slope of Δ_k (ADU/sample) against the pedestal level, over skips 10-499 | raw windows + baseline | an AC-coupling-like relaxation of the output level (pilot: ≈ 1/(7 ms)) [speculative] | electronics | new | M | relates to `noise: baseline coupling`; one polyfit |
| 15 | `noise: null-CDS noise (ADU)` | std over skips of mean(first half of the pedestal window) − mean(second half), on R | raw pedestal window | noise without a charge transfer; for white noise expect ≈ √2 × the CDS noise (two 75-sample halves instead of two 150-sample windows) | GW null streams; A-A tests | new | H | pilot 7.0-8.0 (ch0/1/3) vs 15.8-24.6 (ch2) |
| 16 | `discrete: null-CDS kurtosis` | excess kurtosis of #15 | same | non-Gaussian noise not tied to the charge transfer (RTS in the output stage) | — | new | H | pilot −0.4…0 vs 10-15 (ch2) |
| 17 | `noise: null-CDS autocorrelation at lag 1` | ρ₁ of #15 along the skips | same | slow change of the in-window slope (phase walk, droop) | — | new | L-M | pilot +0.23…+0.44 (good amps), ≈ 0 (ch2) |
| 18 | `discrete: in-window step rate (%)` | fraction of windows (pedestal and signal, skips ≥ 10) whose best single-step t (10-sample blocks) exceeds 6 | raw windows | RTS switching rate × detection efficiency | electrophysiology (idealization), changepoints | new | H | pilot 0 % (ch0), 6.3 % (ch2 reference), 13.2 % (ch2 V low) |
| 19 | `discrete: in-window step amplitude (ADU)` | median |Δ| of the detected steps | raw windows | RTS amplitude × gain (trap size) | electrophysiology, semiconductor noise | new | H | pilot median 52 ADU (reference), 69 ADU (V low) |
| 20 | `discrete: step up/down balance` | (n_up − n_down)/(n_up + n_down) | same | ~0 for a stationary two-level RTS; ≠ 0 for one-sided glitches (charge arrival, cosmic) | — | new | L | pilot −0.03, +0.01 |
| 21 | `noise: RTS mean dwell time (us)` | mean dwell in each state from a 2-state Gaussian HMM (Viterbi) on the in-window block means, gaps at the dip and reset | raw windows | trap capture and emission times | electrophysiology (HMM, Sigworth-Sine) | new | M | numpy forward-backward; only meaningful where #18 > ~1 % |
| 22 | `edge: RTS step rise time (ns)` | 10-90 % rise of the step-triggered average of raw samples | raw windows around detected steps | bandwidth between the switching element and the ADC (locates the RTS: output transistor vs later stage) | neuroscience (event-related averaging) | new | M | needs #18; sub-sample alignment like the equivalent-time pulse |
| 23 | `noise: within-cycle structure function` (curve) | sqrt⟨(m_{j+L} − m_j)²⟩ of 15-sample block means of R, L = 1..9 inside the windows, plus pedestal-to-signal pairs | raw windows | time scale of the noise inside the integration (white, oscillation, switching) | astronomy (structure functions) | new | M-H | pilot table in 1.13 |
| 24 | `noise: structure-function excess at 9 us (ADU)` | sqrt(D(9 blocks) − D(1 block)) | same | slow (≥ µs) in-window noise; RTS with long dwell | — | new | M-H | pilot ≈ 0 (good amps: D falls), 22.7 and 36.6 (ch2 excess over ch0) |
| 25 | `noise: jump-robust CDS sigma (ADU)` | sqrt(MedRV/N) of e_k | CDS | continuous noise without jumps | finance | new (piloted by the exploratory report) | H | 3 lines |
| 26 | `discrete: jump share of CDS variance` | 1 − MedRV/RV | CDS | fraction of the CDS variance in jumps | finance | new (exploratory) | H | pilot 16-41 % on ch2, ≈ 0 elsewhere |
| 27 | `discrete: jump count (Lee-Mykland, 1 %)` | number of skips with |e_k|/σ̂_BV,k beyond the 1 % Gumbel critical value | CDS | calibrated jump count with a local volatility | finance | new | M | rolling bipower σ̂ over 20 skips |
| 28 | `discrete: STA/LTA maximum` | max_k mean(e², k..k+4) / mean(e², k−50..k−1) | CDS | localized bursts | seismology | new (exploratory 13c) | M | pilot 3.6-4.8 vs 9.9-35 (ch2) |
| 29 | `discrete: crest factor` | max|e|/std(e) | CDS | largest excursion | machine diagnostics | new | L-M | pilot 3.0-3.5 vs 5.6-7.4 |
| 30 | `noise: squared-residual autocorrelation (LB10)` | Ljung-Box Q on e², 10 lags | CDS | volatility clustering (bursty regimes) | finance (ARCH) | new | L-M | pilot 8-13 vs 12-24 |
| 31 | `discrete: whitened CDS kurtosis` | kurtosis of e_k after dividing its FFT by sqrt(Welch PSD) | CDS | non-Gaussianity without the colour of 1/f and pickup | gravitational waves | new | M | Welch PSD from `spectrum()` |
| 32 | `discrete: line count (F-test, p<0.001)` | number of frequencies with Thomson F > F_crit(2, 2K−2) | CDS | pickup lines with a stated false-alarm rate | oceanography/climate (multitaper) | new (refines `discrete: CDS spectrum lines`) | H | `scipy.signal.windows.dpss`, NW = 3, K = 5 |
| 33 | `discrete: strongest line frequency (kHz)` and `… amplitude (ADU)` | argmax F; amplitude from the multitaper regression | CDS | identifies the pickup source; tracks it across packets | gravitational waves (line tracking) | new | H | report the top 2 lines |
| 34 | `discrete: line phase coherence` | |mean_p e^{iφ_p}|, with φ_p the phase of the strongest line at each pixel's first skip | CDS, absolute times (`ped_starts`) | 1 = generated by the readout sequence (clock-locked); ~0 = external | gravitational waves, power systems | new | H | needs the line frequency from #33; φ_p by least squares on each pixel |
| 35 | `discrete: line frequency, whole trace (kHz)` | Lomb-Scargle peak of the CDS of all pixels in absolute time | CDS, all pixels | 5-Hz resolution; separates nearby lines | astronomy (Lomb-Scargle) | new | M | `scipy.signal.lombscargle` |
| 36 | `noise: 1/f corner frequency (kHz)` and `noise: 1/f exponent` | fit S(f) = W + A f^{−γ} to `noise: CDS spectrum, log frequency`; f_c = (A/W)^{1/γ} | CDS | standard amplifier noise number; 1/f from traps | semiconductor noise | new | H-M | `curve_fit` in log space; 40 bins |
| 37 | `noise: Lorentzian corner (kHz)` | fit W + B/(1 + (f/f_L)²) (ch2-type spectra) | CDS | RTS rate (1/τ_c + 1/τ_e)/2π | semiconductor noise (Machlup) | new | M | compare with #21 |
| 38 | `noise: Hadamard deviation n=50 (ADU)` and curve | third difference of n-skip means of the *undetrended* CDS (skips ≥ 10) | CDS | drift-insensitive averaging noise, no polynomial bias | time-frequency metrology | new | M-H | same block-mean code as `allan_deviation` |
| 39 | `noise: modified Allan deviation` (curve) | MDEV of e_k | CDS | separates anticorrelated (MA-type) from white | time-frequency metrology | new | M | cumulative-sum implementation |
| 40 | `noise: noise-type exponent at n=10` | α from ρ₁ of 10-skip means (Riley-Greenhall) | CDS | noise type at a given time scale | time-frequency metrology | new | M | small lookup table |
| 41 | `level: common-pattern amplitude spread` | robust spread of the per-pixel coefficients a_p in CDS_p,k ≈ c_p + a_p·common_k | CDS, pixel × skip | whether the common pattern couples differently to different pixels (position or charge) | astronomy (SysRem) | new | M | per-pixel LS; row-start pixels reported separately |
| 42 | `noise: baseline-CDS residual correlation` | correlation of the detrended per-skip pedestal level with e_k at lag 0 (and ±1) | baseline + CDS | a common supply or reset disturbance in both | seismology (noise cross-correlation) | new | M-L | complements `noise: baseline coupling` |
| 43 | `edge: random jitter (ns)` and `edge: deterministic jitter (ns)` | dual-Dirac tail fit to the histogram of reset-edge time residuals | edge times | separates clean random jitter from bounded or periodic timing modulation | telecom | new (splits `edge: jitter`) | M | Q-scale fit of the two tails |
| 44 | `edge: periodic jitter frequency (kHz)` | strongest line in the spectrum of the edge-time residuals | edge times | the pickup modulates timing (clock or supply path) | telecom | new | M | reuse the line finder of #32 |
| 45 | `discrete: max within-window spectral kurtosis` | max_f SK(f) over 150-sample frames of R | raw windows | frequency band of the impulsive noise | machine diagnostics | new (exploratory 23a) | M | rfft of the framed windows |
| 46 | `noise: RTS rate modulation index` | Tort MI of the step indicator (#18) against the phase of the strongest line | raw windows + CDS phase | bias modulation of the trap rates by the pickup [speculative] | neuroscience (PAC) | new | L-M | needs #18 and #33 |
| 47 | `edge: reset-pulse EMG sigma (ns)` and `… tau (ns)` | EMG fit to `edge: equivalent-time pulse` | reset edge | bandwidth and tail in one fit | chromatography | new (overlaps edges) | L | `scipy.special.erfc` model |
| 48 | `discrete: max spectral flux (z)` | max over skips of the robust z-score of Σ max(0, |X_k| − |X_{k−1}|) of the residual cycles | raw cycles | any new spectral event in one skip | audio onset detection | new | L-M | rfft of R, already partly computed |
| 49 | `discrete: cepstral peak (ms)` | quefrency of the largest cepstral peak of the CDS spectrum (excluding < 2 bins) | CDS | harmonic families of pickup | audio, machine diagnostics | new | L | — |
| 50 | Poincaré SD1/SD2, RMSSD, zero-crossing rate, DFA α, permutation entropy, spectral flatness, red-noise β | as in Section 1 | CDS | — | cardiology, audio, nonlinear dynamics, astronomy | **redundant or uninformative** (pilot) | L | do not implement as features; keep the Poincaré *plot* |

---

## 3. The first-skip offset (start-of-pixel transient)

### 3.1 What we already measure

| Feature | What it is | Pilot or CSV values (board 101, 2026-10-12) |
|---|---|---|
| `skip: first-skip CDS excess` (E0) | CDS of skip 0 − median CDS of skips ≥ 10 | 244-349 ADU; channel order ch1 ≈ ch3 > ch0 ≈ ch2; "all H low rails" +45-55 |
| `skip: skips 1-4 CDS ringing` | rms of CDS of skips 1-4 around the median | 69-96 ADU |
| `fit: first-skip ringing frequency / decay / amplitude` | damped cosine + offset fitted to skips 1-19 (skip 0 excluded) | 0.21-0.28 cycles/skip (mostly 0.23-0.24); 2.0-4.9 skips (one outlier of 28); 200-300 ADU |
| `level: common CDS pattern` | median CDS of the usable pixels at each skip (curve) | contains the transient as its first ~10 points |
| `spread[...]` of the above | pixel-to-pixel robust spread | E0 spread 3-12 ADU (ch0/1/3), 7-35 ADU (ch2) |

The existing features describe the transient *as a CDS pattern in skip units*. They do not say what physically produces it,
they mix time and skip units, and they leave skip 0 out of the ringing model.

### 3.2 Pilot result 1: the transient is a baseline slope seen through the CDS

The CDS is a difference of two window means whose centres are 207 samples (13.8 µs) apart. If the baseline B(t) under the
windows has a slope dB/dt, the CDS picks up −207 · dB/dt (in ADU per sample) on top of the charge signal. The CDS is a
first-difference (differentiating) filter at low frequencies.

**Test.** For each skip k = 0..19 of every usable pixel:
* x_k = the mean of the least-squares slopes of the raw samples inside the pedestal and signal windows of skip k, minus their
  median over skips 20-59;
* y_k = CDS_k − median(CDS_{≥10}).

Regress y on x, pooling the pixels of a trace.

**Result** (mean over the 3 repeats of each setting; ch0/1/3 shown, ch2 below):

| Setting | ch0: E0 / lever L / R² | ch1 | ch3 |
|---|---|---|---|
| reference (NCOLPRE 0) | 266 / −211 / 0.97 | 297 / −220 / 0.97 | 294 / −217 / 0.96 |
| VDD −20.5 V | 253 / −211 / 0.97 | 285 / −217 / 0.97 | 282 / −216 / 0.95 |
| V low rails + TG | 250 / −211 / 0.96 | 285 / −218 / 0.97 | 277 / −213 / 0.96 |
| all H low rails | 311 / −216 / 0.97 | 348 / −221 / 0.98 | 341 / −218 / 0.97 |
| H3_A_L only | 244 / −213 / 0.97 | 282 / −219 / 0.97 | 277 / −215 / 0.96 |

* For ch2, R² is 0.62-0.94 and L = −157 to −204. Its large in-window noise biases the regression toward zero
  (errors-in-variables). The transient itself (E0 = 253-309) is the same as on ch0.
* **The lever arm is 211-221 samples**, close to the window-centre separation (207). The small excess is expected: the
  transient is curved on the scale of the cycle, and the mean of two window slopes is not exactly the slope between the window
  centres. On skips 1-19 alone (one ch0 trace) the fitted lever is −278, with R² 0.97.
* **The same identity explains the slow CDS rise along the 500 skips.** In 50-skip blocks of the ch0 reference trace:
  * the in-window slope falls from 0.33 to 0.20 ADU/sample;
  * −210 × its change predicts a CDS rise of 9.1, 14.4, 17.5, 20.4, 22.2, 25.0, 26.6, 27.6, 28.2 ADU;
  * the measured rise is 9.0, 13.5, 17.8, 20.8, 22.9, 25.1, 26.9, 28.4, 28.9 ADU.
  * `level: common CDS rise` (~29 ADU) is therefore almost entirely baseline slope (0.7 ADU left over). The same physics
    sits behind `noise: baseline coupling (1e-3)`.
* Inside each cycle the level drifts by 0.2-0.33 ADU/sample (100-160 ADU per cycle), more than the reset-to-reset change of
  the pedestal level (10-70 ADU per skip). Each reset pulls the level back part of the way. The droop is the slow change of
  this sawtooth.

**Consequence.** The first-skip offset, the skips 1-9 "ringing" and the slow CDS rise are one phenomenon: the time derivative
of the output level, sampled by the CDS. They are not charge or gain effects. A charge-dependent part, if there is one, has to
be looked for in the **residual** of this identity (features #2, #3 and #13).

### 3.3 Pilot result 2: shape and time constant inside skip 0

D₀(t) = (skip 0 − its pedestal mean) − (mean of skips 20-59, each minus its pedestal mean), averaged over the usable pixels:
* +150 ADU at the start of the pedestal window;
* falling smoothly to about −300 ADU just before the reset, with no change of shape at the SW dip;
* then **+2,000 ADU after the reset**: the first reset re-establishes a level about 2,000 ADU higher (feature #8, the reset
  jump skip 0→1 = 1,500-2,300 ADU).

A·e^{−t/τ} + c fitted over the two windows:

| Setting | τ (µs), mean of 4 amplifiers | A (ADU) |
|---|---|---|
| reference | 12.0 | 630 |
| VDD −20.5 V | 11.7 | 609 |
| V low rails + TG | 11.4 | 605 |
| all H low rails | 12.2 | **744** |
| H3_A_L only | 11.9 | 606 |

Per amplifier, τ is 11.1-12.4 µs, and every setting shows it. It is about 3-4 % shorter on ch3 than on ch0 and ch1, a
reproducible amplifier difference.

### 3.4 Pilot result 3: what the amplitude depends on

* **H-clock swing.** "All H low rails" lowers every H low rail by 0.5 V: swing 2.8 → 3.3 V for H1/H3 (+18 %), 2.3 → 2.8 V for
  H2 (+22 %). E0 rises by +22-24 % on all four amplifiers (e.g. ch0 250 → 311, ch1 285 → 348), and A by +23 %.
  * Lowering H3_A_L alone (one phase, one side) changes nothing beyond the scatter.
  * The transient is therefore **capacitive feedthrough of the horizontal clocks**, decaying with τ ≈ 12 µs after the last
    horizontal transfer. This fits its being in place before the first pedestal window.
  * A per-volt coupling (feature #6) makes it comparable to the existing `edge: dip per volt SW` and `edge: pulse per volt RG`.
* **The amplifier** (via its chain). The channel order ch1 ≈ ch3 > ch0 ≈ ch2 (by ~30-45 ADU) is stable across all settings: a
  per-channel coupling fingerprint. **ch2's excess noise does not touch it**: E0 is the same as on ch0, and τ and A match.
  * Since τ is nearly the same on all four channels and in all settings, it is probably set by a board-level element shared by
    the chain design (e.g. a preamp or AC-coupling time constant, or the clock drivers' return path) rather than by the CCD's
    output transistor [speculative].
* **The clocking history before the pixel.**
  * Row-start pixels have E0/E0_inner = 0.93 when NCOLPRE = 0.
  * They have **0.41-0.48 when NCOLPRE = 185**, when 185 × 20 prescan transfers precede the row-start pixel.
  * The transient is not a fixed response to "the last 20 transfers". It depends on how long the H clocks ran, e.g. a coupling
    path that saturates or charges up during long clock bursts [speculative].
  * Pixel 0 (the first pixel after the start of the trace) in the reference group shows E0 = 210-276, below pixel 5 (also a
    row start): what precedes the trace matters too.
* **VDD, the V rails and the H3 rail** change E0 by at most ~15 ADU against the reference. That is confounded with NCOLPRE
  (0 vs 185) and with the time of day, so it is not separable in these data.
* **Pixel-to-pixel variation.** For ch0/1/3, the spread of E0 over the inner pixels is 3-12 ADU, i.e. the single-skip noise
  (σ_CDS ≈ 5.7-6.9 ADU). **E0 is deterministic from pixel to pixel to within the read noise.** Its spread is a noise
  measurement, not a physical variation. The matched-filter amplitude (#10) uses all 30 skips and lowers that noise.
* **The previous pixel's charge.** Untestable here: no pixel in the 59 traces has charge (`level: pixel level max` ≤ 58 ADU).
* **NDCMPRE.** Untestable here: all 60 trace .meta files have NDCMPRE = 0 (also on 2026-10-06 and 2026-10-08).

### 3.5 Proposed characterization and tests

1. **Report the transient in time units, and from the raw samples** (features #1, #4, #5, #8).
   * τ (µs), A (ADU) and the slope at skip 0 (ADU/µs) are physical parameters.
   * They do not change meaning when `delayInt`, the windows or the skip period change. E0 does: it depends on the window
     separation through L.
2. **Keep the differentiator identity as a standing check** (#2, #3).
   * L should stay ≈ (window-centre separation) + a few samples. If the windows change, L must follow.
   * A drop in R² on a good amplifier means a non-baseline (charge or gain) component has appeared in the first skips.
3. **Fit all modes at once** with a matrix pencil (#9) on the mean transient, skips 0-29.
   * Convert to kHz and µs. The current grid fit sees one damped cosine in skips 1-19.
   * Pilot reading: a fast decay (~12 µs) and a resonance at ≈ 7.2 kHz (0.235 × 30.69 kHz) with τ ≈ 75-95 µs (ζ ≈ 0.25).
   * **Test whether the resonance is fixed in time or in skips** by comparing traces with different skip periods (e.g. a
     `delayInt` scan). A fixed-kHz resonance points to an electrical LC or feedback loop; a fixed cycles/skip value points to
     a sequencer-locked effect.
4. **NDCMPRE prediction (testable).** If the NDCMPRE pre-skips are clocked like ordinary skips, the transient simply advances:
   E0(NDCMPRE = m) ≈ CDS_m − median of today's traces.
   * Concretely, from the ch0 reference: ≈ −50 ADU for m = 1, ≈ −125 ADU for m = 2, ≈ −50 ADU for m = 3, ≈ −5 ADU for m = 4.
   * The first recorded skip would then be *below* the later level, not above it.
   * A clear deviation would show that the pre-skips differ from recorded skips.
   * One packet with NDCMPRE ∈ {0, 1, 2, 5, 10} settles it.
5. **Charge dependence (deferred-charge analogue, cf. [53, 54]).**
   * Take traces with charge: an LED pulse, a non-zero EXPOSURE, or a hot column in the trace region.
   * Regress the *residual* of the slope identity in pixel p+1's skip 0 on pixel p's charge (#12), and separately on its own
     charge.
   * Because the baseline part is removed first, a signal-dependent electronic offset or a first-transfer charge effect is
     isolated.
6. **History dependence.** Compare row-start and inner pixels (#7) across NCOLPRE and NCOLCLR.
   * The ratio 0.93 → 0.45 suggests that a model with the number of preceding transfers and the idle time would describe the
     transient's amplitude, like the exploratory report's two-clock model of the row-start drift.
7. **As an output-stage diagnostic.** τ, A/V_H, the reset jump 0→1 and the in-window drift against the baseline level (#14)
   characterize the chain's DC and low-frequency response with a known stimulus (the H-clock burst), in every trace, for free.
   Track them in the reference trace of every packet (Westgard-style, exploratory 12d) as a monitor of the video chain.

---

## 4. Hand-analysis diagnostic plots and procedures (not features)

| # | Plot / procedure | How | What it would show |
|---|---|---|---|
| D1 | **Differentiator plot** | CDS_k − median against the in-window slope excess, skips 0-19, all pixels, coloured by k; reference line slope −207 | Whether the start transient is pure baseline slope (all points on one line) and which skips depart from it; ch2 as a cloud around the same line [pilot] |
| D2 | **Skip-0 difference waveform** | D₀(t) of 3.3 overlaid for the 5 settings, with the exponential fits | τ unchanged, amplitude following the H swing; where in the cycle the transient sits relative to the windows |
| D3 | **Poincaré / return plot** of e_k against e_{k+1} | per amplifier, 2-D histogram | Round cloud for ch0/1/3; for ch2, off-diagonal satellite clusters at (±Δ, ∓Δ) if the RTS reverses within one skip. SD1/SD2 add nothing beyond ρ₁, but the clusters are visible only in the picture |
| D4 | **RTS step-amplitude histogram and log-binned dwell-time histogram** (Sigworth-Sine) | signed step amplitudes from the in-window detector; dwell times from HMM idealization; sqrt(counts) against log(dwell) | Number of traps (amplitude clusters), τ_c and τ_e (peaks), whether they change with VDD, VR or the V rails. Pilot: ±40-60 ADU in reference, broader 40-110 ADU in V low rails |
| D5 | **Within-cycle structure function** per amplifier | 1.13 | White + ~150-kHz oscillation on good amplifiers; a growing switching term on ch2 |
| D6 | **Spectral-kurtosis map** (kurtogram) of the residual windows | SK(f) against frame length (50, 75, 150 samples), ch2 vs ch0 | Whether ch2's impulsive noise is broadband steps (SK at low f) or a band-limited oscillation |
| D7 | **Wavelet scalogram** of the CDS in absolute time (all pixels concatenated, gaps marked) | Morlet, red-noise significance (Torrence & Compo) | When pickup lines appear or vanish within the 0.2 s; burst epochs on ch2; whether lines are continuous across pixel boundaries |
| D8 | **Phase-folded CDS** at the strongest line frequency | fold e_k by absolute time modulo the line period; separately by time since the pixel's first skip | Folding by absolute time sharpens an external line; folding by time-since-pixel-start sharpens a clock-generated one. Decides #34 by eye |
| D9 | **Eye diagram** of the cycles | 2-D histogram of (position in cycle, ADU) over all skips, log colour; zoom on the windows using 10-sample block means | For ch2: two or more bands inside the windows (RTS levels); for all: jitter at the edges, the first-skip cycles as outliers |
| D10 | **Step-triggered average** | average raw samples ±30 samples around detected RTS steps (separately up and down) | Step shape → bandwidth after the switching element; asymmetry between up and down; whether the step is followed by overshoot or settling |
| D11 | **Recurrence plot** of the in-window block-mean sequence (a subsample) | R_ij = Θ(ε − |m_i − m_j|) | Laminar blocks (RTS states) on ch2; homogeneous texture on good amplifiers |
| D12 | **Bicoherence map** of the CDS | segment-averaged bicoherence, 64-skip segments | Tests the speculative 2.2 + 5 → 7.2 kHz coincidence; otherwise a Gaussianity check |
| D13 | **Matrix-pencil pole plot** of the start transient | poles in the complex plane, all traces, coloured by setting | Whether the modes stay put (circuit) or move with settings (operating point) |
| D14 | **Row-start vs inner transient overlay** | mean CDS_k − median, k = 0..29, of row-start and inner pixels, by NCOLPRE | The 0.93 → 0.45 amplitude change; whether the shape (τ, ringing) changes too |
| D15 | **Novelty chart** | Mahalanobis distance of each trace's new start-transient features from the reference-trace population, in time order | Video-chain drift between packets, with a known stimulus (Westgard-like rules on top) |

---

## 5. Ranked top 10 to implement first (in `src/trace_features.py`)

1. **`start: CDS-slope lever arm (samples)` + `start: slope-explained fraction`** (#2, #3).
   * The pilot shows that they turn the first-skip excess, the "ringing" and the slow CDS rise into one physically interpreted
     quantity.
   * The R² is a standing check that will flag any genuine charge or gain effect in the first skips.
   * Cheap: vectorized slopes, one regression.
2. **`start: transient time constant (us)` + `start: transient amplitude (ADU)` + `start: in-cycle slope at skip 0 (ADU/us)`**
   (#1, #4, #5).
   * Physical units, independent of the window settings, and reproducible (τ = 11.1-12.4 µs on every trace).
   * They characterize the video chain with a stimulus that is free in every trace.
3. **`start: first-skip excess per volt H (ADU/V)`** (#6). The pilot shows that the excess follows the H swing on all four
   amplifiers. A per-volt coupling fits the existing `edge: … per volt …` features and helps attribute H-rail changes.
4. **`discrete: in-window step rate (%)` + `discrete: in-window step amplitude (ADU)`** (#18, #19).
   * The clearest RTS evidence so far: 0 of 7,840 windows on ch0 against 6-13 % on ch2, with symmetric ±40-110 ADU steps.
   * Lets Level 1 and the hardware team track ch2's trap(s) against VDD, VR and VDRAIN.
5. **`noise: null-CDS noise (ADU)` + `discrete: null-CDS kurtosis`** (#15, #16).
   * Separates noise that needs a charge transfer from noise that does not; ch2's does not.
   * Two lines on the existing residual cycles.
6. **`noise: jump-robust CDS sigma (ADU)` + `discrete: jump share of CDS variance`** (#25, #26).
   * Already validated by the exploratory pilot, never added.
   * The jump-robust σ is the right noise number for ch2-like amplifiers, and the jump share is a clean "damaged" indicator.
7. **`level: slope-corrected CDS rise (ADU)`** (#13).
   * Removes the baseline-slope part (≈ 28 of 29 ADU on ch0) from `level: common CDS rise`, so what is left can be read as a
     gain or charge effect.
   * Reuses the lever arm from item 1.
8. **Line features: `discrete: line count (F-test)`, `discrete: strongest line frequency (kHz)` and amplitude,
   `discrete: line phase coherence`** (#32-34).
   * Turns "there are some lines" into "which line, how strong, and is it made by our own clocking".
   * That is the question that decides whether a fix belongs in the sequencer or in the grounding and supplies.
9. **`noise: within-cycle structure function` (curve) + `noise: structure-function excess at 9 us (ADU)`** (#23, #24).
   * Shows the time scale of the noise *inside* the integration windows, which no current feature resolves. This is directly
     relevant to choosing `delayInt` and the window lengths.
10. **`noise: 1/f corner frequency (kHz)` (+ exponent)** (#36), and **`start: row-start/inner first-skip ratio`** (#7).
    * The corner frequency is the standard amplifier-noise summary and is missing.
    * The row-start ratio is a one-line number that captures the history dependence of the transient (0.93 vs 0.45 with
      NCOLPRE).

**Next in line:** the matrix-pencil modes of the transient (#9; replaces the grid ringing fit and adds kHz units), the
Hadamard deviation (#38; removes the polynomial bias at large n), the RTS dwell time by HMM (#21), STA/LTA (#28) and the
RJ/DJ jitter split (#43).

**Not worth implementing** (pilot or algebra): permutation entropy, DFA α, Poincaré SD1/SD2, RMSSD, zero-crossing rate,
spectral flatness and the red-noise β. They are redundant with ρ₁, the variance and the Allan features, or they do not
separate the amplifiers.

---

## 6. References

Status: "(checked)" = bibliographic record found during this work; "(record only)" = no full text read; "[unverified]" = from
memory or details not confirmed. References already verified in `landscape_exploratory.md` are cited there and not repeated
(Laguna 1999 KLT; Zhang, Mykland & Aït-Sahalia 2005; Ané & Geman 2000; Antoni 2009; Sens-Schönfelder & Wegler 2006; Westgard
1981; Yoon 2015 FAST; Yeh 2018 matrix profile; Smaragdis & Brown 2003).

1. Task Force of the ESC and NASPE (1996). Heart rate variability: standards of measurement, physiological interpretation and
   clinical use. *Circulation* 93(5):1043-1065. PubMed 8598068. (checked, record only)
2. Brennan, M., Palaniswami, M., Kamen, P. (2001). Do existing measures of Poincaré plot geometry reflect nonlinear features of
   heart rate variability? *IEEE Trans. Biomed. Eng.* 48(11):1342-1347. doi:10.1109/10.959330. (checked, record only)
3. Peng, C.-K., Havlin, S., Stanley, H. E., Goldberger, A. L. (1995). Quantification of scaling exponents and crossover phenomena
   in nonstationary heartbeat time series. *Chaos* 5:82-87. (checked, record only)
4. Richman, J. S., Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy.
   *Am. J. Physiol. Heart Circ. Physiol.* 278:H2039-H2049. doi:10.1152/ajpheart.2000.278.6.H2039. (checked)
5. Tort, A. B. L., Komorowski, R., Eichenbaum, H., Kopell, N. (2010). Measuring phase-amplitude coupling between neuronal
   oscillations of different frequencies. *J. Neurophysiol.* 104(2):1195-1210. doi:10.1152/jn.00106.2010, open at PMC2941206.
   (checked)
6. Sigworth, F. J., Sine, S. M. (1987). Data transformations for improved display and fitting of single-channel dwell time
   histograms. *Biophys. J.* 52(6):1047-1054. doi:10.1016/S0006-3495(87)83298-8, open at PMC1330104. (checked)
7. Chung, S. H., Moore, J. B., Xia, L., Premkumar, L. S., Gage, P. W. (1990). Characterization of single channel currents using
   digital signal processing techniques based on hidden Markov models. *Phil. Trans. R. Soc. B* 329:265-285.
   doi:10.1098/rstb.1990.0170. (checked, record only)
8. Allen, R. (1978). Automatic earthquake recognition and timing from single traces. *Bull. Seismol. Soc. Am.* 68:1521-1532.
   (checked via citing papers)
9. *(unused number)*
10. Bensen, G. D., Ritzwoller, M. H., et al. (2007). Processing seismic ambient noise data to obtain reliable broad-band surface
    wave dispersion measurements. *Geophys. J. Int.* 169(3):1239-1260. doi:10.1111/j.1365-246X.2007.03374.x. (checked)
11. *(unused number)*
12. Davis, S. B., Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in
    continuously spoken sentences. *IEEE Trans. ASSP* 28(4):357-366. (checked)
13. Bello, J. P., Daudet, L., Abdallah, S., Duxbury, C., Davies, M., Sandler, M. B. (2005). A tutorial on onset detection in
    music signals. *IEEE Trans. Speech Audio Process.* 13(5):1035-1047. doi:10.1109/TSA.2005.851998. (checked)
14. Kaiser, J. F. (1990). On a simple algorithm to calculate the "energy" of a signal. *Proc. ICASSP-90*, vol. 1, pp. 381-384.
    (checked via citations)
15. Barndorff-Nielsen, O. E., Shephard, N. (2004). Power and bipower variation with stochastic volatility and jumps.
    *J. Financial Econometrics* 2(1):1-37. doi:10.1093/jjfinec/nbh001. (checked)
16. Andersen, T. G., Dobrev, D., Schaumburg, E. (2012). Jump-robust volatility estimation using nearest neighbor truncation.
    *J. Econometrics* 169(1):75-93 (NBER WP 15533). (checked)
17. Lee, S. S., Mykland, P. A. (2008). Jumps in financial markets: a new nonparametric test and jump dynamics. *Rev. Financial
    Studies* 21:2535-. doi:10.1093/rfs/hhm056. (checked)
18. Antoni, J. (2007). Fast computation of the kurtogram for the detection of transient faults. *Mech. Syst. Signal Process.*
    21(1):108-124. (checked). Antoni (2006), the spectral kurtosis paper in *MSSP* 20: [unverified details].
19. Randall, R. B., Antoni, J. (2011). Rolling element bearing diagnostics: a tutorial. *Mech. Syst. Signal Process.*
    25(2):485-520. doi:10.1016/j.ymssp.2010.07.017; open at HAL hal-01018742. (checked)
20. Randall, R. B. A history of cepstrum analysis and its application to mechanical problems. *Mech. Syst. Signal Process.*
    (2016 online / 2017) [year and volume unverified].
21. Rohling, H. (1983). Radar CFAR thresholding in clutter and multiple target situations. *IEEE Trans. Aerosp. Electron. Syst.*
    AES-19:608-621. (checked)
22. LIGO Scientific Collaboration and Virgo Collaboration (2020). A guide to LIGO-Virgo detector noise and extraction of
    transient gravitational-wave signals. *Class. Quantum Grav.* 37:055002. arXiv:1908.11170. (checked)
23. Zevin, M., et al. (2017). Gravity Spy: integrating Advanced LIGO detector characterization, machine learning, and citizen
    science. *Class. Quantum Grav.* 34:064003. arXiv:1611.04596. (checked)
24. Kirton, M. J., Uren, M. J. (1989). Noise in solid-state microstructures: a new perspective on individual defects, interface
    states and low-frequency (1/f) noise. *Adv. Phys.* 38(4):367-468. (checked, record only)
25. Riley, W. J. (with Howe, D. A.) (2008). *Handbook of Frequency Stability Analysis*. NIST Special Publication 1065.
    doi:10.6028/NIST.SP.1065; open PDF at nvlpubs.nist.gov. (checked)
26. Riley, W. J., Greenhall, C. A. (2004). Power law noise identification using the lag 1 autocorrelation [venue unverified].
    Its method and expected ρ₁ values are described in arXiv:1604.05076 ("On temporal correlations in high-resolution
    frequency counting"), which cites the NIST handbook pp. 45-46. (secondary source checked)
27. Machlup, S. (1954). Noise in semiconductors: spectrum of a two-parameter random signal. *J. Appl. Phys.* 25 [unverified:
    record not found].
28. Wang, X., Rao, P. R., Mierop, A., Theuwissen, A. J. P. (2006). Random telegraph signal in CMOS image sensor pixels.
    *IEEE IEDM 2006*, pp. 115-118. (checked; full text hosted by Teledyne DALSA)
29. Goiffon, V., et al. (2011). RTS in CMOS image sensors (paper R27). *International Image Sensor Workshop 2011*, imagesensors.org.
    (checked existence; title not recorded exactly)
30. Hooge, F. N. (1969). 1/f noise is no surface effect. *Phys. Lett. A* 29 [title wording and pages unverified].
31. "Power quality assessment via wavelet transform analysis" (1996). *IEEE Trans. Power Delivery*. doi:10.1109/61.489353, OSTI
    244830. (record checked; authors usually given as Santoso, Powers, Grady & Hofmann [unverified])
32. Hua, Y., Sarkar, T. K. (1990). Matrix pencil method for estimating parameters of exponentially damped/undamped sinusoids in
    noise. *IEEE Trans. ASSP* 38(5):814-824. doi:10.1109/29.56027. (checked)
33. Pont, F., Zucker, S., Queloz, D. (2006). The effect of red noise on planetary transit detection. *MNRAS* 373:231-242.
    arXiv:astro-ph/0608597. (checked)
34. VanderPlas, J. T. (2018). Understanding the Lomb-Scargle periodogram. *ApJS* [vol. 236, 16, unverified]. arXiv:1703.09824.
    (checked)
35. Tamuz, O., Mazeh, T., Zucker, S. (2005). Correcting systematic effects in a large set of photometric light curves.
    *MNRAS* 356(4):1466-1470. arXiv:astro-ph/0502056. (checked)
36. Simonetti, J. H., Cordes, J. M., Heeschen, D. S. (1985). *ApJ* 296:46 (structure functions of radio-source flicker).
    (checked via citing papers; title not confirmed)
37. Thomson, D. J. (1982). Spectrum estimation and harmonic analysis. *Proc. IEEE* 70:1055-1096. doi:10.1109/PROC.1982.12433.
    (checked)
38. Hannachi, A., Jolliffe, I. T., Stephenson, D. B. (2007). Empirical orthogonal functions and related techniques in
    atmospheric science: a review. *Int. J. Climatol.* 27(9):1119-1152. doi:10.1002/joc.1499. (checked)
39. Torrence, C., Compo, G. P. (1998). A practical guide to wavelet analysis. *Bull. Amer. Meteor. Soc.* 79(1):61-78. (checked;
    the authors host a copy)
40. Farrar, C. R., Worden, K. (2012). *Structural Health Monitoring: A Machine Learning Perspective*. Wiley. (checked)
41. Eilers, P. H. C. (2003). A perfect smoother. *Anal. Chem.* 75(14):3631-3636. (checked). Eilers, P., Boelens, H. (2005).
    Baseline correction with asymmetric least squares smoothing. Leiden University Medical Centre report, unpublished.
    (checked via the pybaselines documentation)
42. Grushka, E. (1972). Characterization of exponentially modified Gaussian peaks in chromatography. *Anal. Chem.* 44:1733-1738.
    (checked via citations)
43. Keysight Technologies. Jitter analysis: the dual-Dirac model, RJ/DJ, and Q-scale. White paper 5989-3206. (checked)
44. Bandt, C., Pompe, B. (2002). Permutation entropy: a natural complexity measure for time series. *Phys. Rev. Lett.*
    88:174102. (checked)
45. Hopkinson, G. R., Lumb, D. H. (1982). Noise reduction techniques for CCD image sensors. *J. Phys. E* 15 [record not found;
    cited in arXiv:1511.05649, "Optimal CCD readout by digital correlated double sampling", checked existence].
46. Marwan, N., Romano, M. C., Thiel, M., Kurths, J. (2007). Recurrence plots for the analysis of complex systems. *Phys. Rep.*
    438(5-6):237-329. doi:10.1016/j.physrep.2006.11.001. (checked)
47. Hinich, M. J. (1982). Testing for Gaussianity and linearity of a stationary time series. *J. Time Series Anal.*
    3(3):169-176. doi:10.1111/j.1467-9892.1982.tb00339.x. (checked)
48. Killick, R., Fearnhead, P., Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost.
    *J. Amer. Statist. Assoc.* 107(500):1590-1598. arXiv:1101.1438. (checked)
49. Gardner, W. A. (1991). Exploitation of spectral redundancy in cyclostationary signals. *IEEE Signal Process. Mag.* 8:14-36.
    (checked)
50. Tiffenberg, J., et al. (2017). Single-electron and single-photon sensitivity with a silicon Skipper CCD. *Phys. Rev. Lett.*
    119:131802. arXiv:1706.00028. (checked)
51. Cancelo, G., et al. (2021). Low Threshold Acquisition controller for Skipper CCDs. *J. Astron. Telesc. Instrum. Syst.* 7(1).
    arXiv:2004.07599. (checked)
52. Chierchie, F., et al. (2021). Smart readout of nondestructive image sensors with single photon-electron sensitivity.
    *Phys. Rev. Lett.* 127:241101. arXiv:2111.09905. (checked)
53. Snyder, A., Longley, E., Lage, C., Digel, S., Neal, H., Utsumi, Y., Roodman, A. (2021). Characterization and correction of
    serial deferred charge in LSST camera ITL CCDs. *J. Astron. Telesc. Instrum. Syst.* 7(4):048002. (checked)
54. Snyder, A., Roodman, A. (2020). Investigation of deferred charge effects in LSST ITL sensors. arXiv:2001.03223 (also
    *JATIS* 5(4):041509). (checked)

Also cited inline and **[unverified]**: Quiroga, Nadasdy & Ben-Shaul 2004 (spike sorting); Aki & Chouet 1975 (coda Q); Turin
1960 (matched filters); Lo & MacKinlay (variance ratio); Worden, Manson & Fieller 2000 (SHM outlier analysis; search did not
find the record); Savitzky & Golay 1964; IEC 61000-4-7.
