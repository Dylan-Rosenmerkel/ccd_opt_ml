# Trace features: everything that could be added to `src/trace_features.py`

2026-10-21. A summary of every candidate trace feature found so far, with the evidence we have for each, and a plan for
adding them. Sources:

- [`docs/literature/signal_analysis_features.md`](literature/signal_analysis_features.md): 21 fields, 49 candidates
  (numbers `#1`–`#49` below refer to its Section 2 table), 15 diagnostic plots.
- [`docs/literature/landscape_exploratory.md`](literature/landscape_exploratory.md): the exploratory analyst's ideas
  (skip-cycle mode decomposition, finance jump statistics, censored labels).
- A settings-separation test run today on all 59 real traces of the 2026-10-12 packet tests (Section 2).

The current set is 85 numbers and 14 curves (`trace_features.FEATURES`).

---

## 1. What makes a feature worth adding

A feature earns its place if it carries **information about the labels** (σe, λ, gain). Separating the settings, or
the amplifiers, is a proxy for that, and each proxy matters in its own way:

| A feature that… | helps because… |
|---|---|
| separates **settings** | the model can learn how the labels change with a setting, even where the settings vector alone is ambiguous, e.g. what a voltage change actually did at the output stage |
| separates **amplifiers** | no model is told which amplifier it sees; the trace must say it (noise level, ch2's switching) |
| tracks the **CCD's state** (temperature, drift, the start transient) | it explains label changes that the settings cannot |

So a feature should **not** be rejected because it fails one of these tests. It can be rejected if it is:

- **redundant**: a deterministic function of numbers we already have;
- **unstable**: it varies between repeats as much as between settings, and does not separate amplifiers either;
- **impossible** with our data, e.g. it needs charged pixels.

**More features are not free.** With about 120 labelled examples, each extra input is another chance for a model to fit
noise. The models differ in how much this hurts:

- Bayesian ridge, boosted trees and the SAAS GP (now used for λ) largely switch off inputs that do not help;
- the plain GP and the neural networks are more exposed.

**The plan, then, is to compute and store every cheap candidate, and choose per model.** Features get a tier, models
can include or exclude tiers, and after session A the new validation decides which tiers stay. That validation has
paired comparisons with bootstrap intervals (`lvl1/train_lvl1.py`); the SAAS relevance scores show which inputs the
models actually use. Computing a feature costs milliseconds. Leaving it out loses information we cannot get back
without reprocessing. Processing always recomputes features from the raw files, so a feature can be added later, but
then every packet must be reprocessed.

---

## 2. Today's test: separation of settings, separation of amplifiers, redundancy

**How it was measured:**

- **Data:** all 59 usable traces of 2026-10-12; 5 settings × 3 repeats × 4 amplifiers, one damaged ch3 trace left out.
- **Statistics:** each one is computed on the detrended CDS (skips ≥ 10) of every usable pixel, and its median over the
  pixels is taken.
- **Settings F:** one-way ANOVA over the 4 changed settings, per amplifier (ch0, ch1, ch3), using only settings taken
  with NCOLPRE 185, so the reference's different NCOLPRE cannot fake a separation. F above about 4 means p < 0.05.
- **Amplifier F:** the same ANOVA over the 4 amplifiers, within each setting.
- **Redundancy:** R² of a rank regression on the four existing CDS-noise numbers.

Script: `docs/slides/feature_sep_test.py`.

| statistic | from | settings F (min / median over amps) | amplifier F (median) | R² from existing noise numbers | verdict |
|---|---|---|---|---|---|
| H-transfer minimum (existing) | — | 441 448 / 467 702 | 23 057 | — | separates both |
| V-transfer minimum (existing) | — | 65 390 / 182 970 | 1 323 | — | separates both |
| CDS rms (existing) | — | 0.2 / 1.2 | 690 | — | amplifiers only |
| Allan deviation n=50 (existing) | — | 0.1 / 0.9 | 34 | — | amplifiers only |
| lag-1 autocorrelation (existing) | — | 0.7 / 0.9 | 52 | — | amplifiers only |
| excess kurtosis (existing) | — | 0.2 / 0.9 | 28 | — | amplifiers only |
| DFA exponent | cardiology | 0.6 / 0.7 | 2.8 | **0.15** | mostly new information, weak so far |
| Ljung-Box of squared CDS (volatility clustering) | finance | 0.1 / 0.5 | 2.5 | **0.16** | mostly new information, weak so far |
| spectral slope | audio | 0.3 / 1.2 | 23 | 0.40 | new information, separates amplifiers |
| sample entropy | nonlinear dynamics | 0.1 / 2.1 | 34 | 0.44 | new information, separates amplifiers |
| red-noise β (50 skips) | astronomy | 0.1 / 0.8 | 7.0 | 0.44 | new information |
| STA/LTA maximum | seismology | 0.9 / 1.6 | 14 | 0.59 | partly new |
| permutation entropy (m = 3 / 4) | nonlinear dynamics | 0.3 / 0.8, 0.9 / 0.9 | 5.9, 12.5 | 0.63, 0.74 | partly redundant (lag-1) |
| zero-crossing rate | audio | 0.2 / 0.3 | 14 | 0.66 | partly redundant (lag-1) |
| spectral centroid | audio | 0.5 / 1.4 | **63** | 0.67 | partly redundant, separates amplifiers strongly |
| jump share (bipower) | finance | 0.2 / 1.4 | 36 | 0.72 | partly redundant, separates amplifiers |
| spectral flatness | audio | 0.5 / 2.2 | 19 | 0.77 | mostly redundant |
| crest factor | machine diagnostics | 0.6 / 1.5 | 56 | 0.77 | mostly redundant (kurtosis) |
| RMSSD / SD (Poincaré) | cardiology | 0.7 / 0.7 | 54 | **0.99** | **redundant**: equals √(2(1 − ρ₁)) |

**What this shows.**

1. **No CDS-noise statistic separates these 4 settings, the existing ones included.** That is because these settings
   (VDD, V rails, H rails, H3) did not change the noise, as found earlier; the waveform features show that they did
   change the electronics. So this is no evidence against any noise statistic. The real test is session A, where
   delayInt changes the noise directly.
2. **Many of the earlier "rejected" statistics do separate the amplifiers** (F 13–63) on the full data. The earlier
   verdict rested on 2 traces per setting.
3. **Only RMSSD/SD is truly redundant.** It is an exact function of the lag-1 autocorrelation, and so are the Poincaré
   SD1/SD2. The DFA exponent, volatility clustering, spectral slope, sample entropy and red-noise β carry mostly new
   information: R² ≤ 0.44 from the existing noise numbers.

---

## 3. The candidates, by tier

**Tier 1** = implement first: physically motivated, piloted on real traces, cheap. **Tier 2** = add as candidates:
cheap and plausibly informative; the validation decides after session A. **Tier 3** = do not add as features, for the
reason given.

### 3.1 Start of pixel: a new `start:` group (the first-skip offset)

The pilot result (signal report, Section 3) is that the first-skip offset is the CDS **reading the slope of a decaying
baseline**. On ch0, ch1 and ch3:

- the offset equals −L × the in-window slope, with L ≈ 211–221 samples (the window centres are 207 apart) and
  R² 0.95–0.98;
- skip 0 contains an exponential with τ ≈ 11–12.5 µs and amplitude 560–790 ADU;
- the size follows the horizontal-clock swing. In our own feature it rises ×1.17 when the H low rails are lowered, or
  ×1.23 against settings with the same NCOLPRE.

| tier | feature | definition (short) | why |
|---|---|---|---|
| 1 | `start: CDS-slope lever arm (samples)` (#2) | regression of the first-skip CDS excess on the in-window slope, skips 0–19 | a value near −207 means pure baseline slope; a departure would mean a charge or gain effect |
| 1 | `start: slope-explained fraction` (#3) | R² of that regression | a standing check of the explanation |
| 1 | `start: in-cycle slope at skip 0 (ADU/µs)` (#1) | slope inside the windows of skip 0 | the transient's derivative |
| 1 | `start: transient time constant (µs)` (#4) | exponential fit inside skip 0 | a property of the output stage, independent of the window settings |
| 1 | `start: transient amplitude (ADU)` (#5) | the amplitude of that fit | the size of the H-clock disturbance |
| 1 | `start: first-skip excess per volt H (ADU/V)` (#6) | the excess ÷ the H-clock swing from the .meta | coupling per volt, comparable across settings |
| 1 | `level: slope-corrected CDS rise (ADU)` (#13) | the existing CDS rise minus its slope part | what is left is a real gain or charge effect (0.7 of 28.9 ADU on one trace) |
| 2 | `start: row-start/inner first-skip ratio` (#7) | row-start pixels ÷ inner pixels | memory of the clocking before the pixel: 0.93 with no prescan, 0.41–0.48 with 185 prescan columns |
| 2 | `start: reset jump skip 0→1 (ADU)` (#8) | the pedestal of skip 1 minus that of skip 0 | the first reset; the existing droop curve starts at skip 1 and misses it |
| 2 | `start: mode frequencies (kHz)` and `start: mode damping (µs)` (#9) | matrix-pencil poles of the first 30 skips | all the transient's modes in physical units, including the ~7.2 kHz ringing |
| 2 | `start: transient amplitude, matched filter (ADU)` (#10) | projection on the trace's mean transient | per-pixel size with ~30 % less noise |
| 2 | `start: ringing frequency (kHz)` (#11) | the existing ringing frequency × the skip rate | comparable across delayInt settings |
| 3 | `start: charge memory` (#12) | offset after a charged pixel | needs charged pixels; keep it for data with charge |

**Two open questions, testable cheaply on the DAQ.**

- **NDCMPRE.** The report predicts that the first kept skip would read about −50 ADU at NDCMPRE = 1, and about −125 ADU
  at NDCMPRE = 2. This bears on the trace-settings decision.
- **Previous-pixel charge.** It needs a trace with charged pixels.

### 3.2 Telegraph switching and non-Gaussian noise (ch2)

The pilots found that ch2's excess noise is inside one 10-µs window, with no charge transfer involved: kurtosis 11–15
there against about 0 elsewhere. Steps are found in 6–13 % of ch2's windows and in none of ch0's 7,840. They are equally
up and down, ±40–110 ADU.

| tier | feature | definition (short) | why |
|---|---|---|---|
| 1 | `noise: null-CDS noise (ADU)` (#15) | first half minus second half of the pedestal window | noise with no charge transfer |
| 1 | `discrete: null-CDS kurtosis` (#16) | its excess kurtosis | output-stage switching, separate from the transfer |
| 1 | `discrete: in-window step rate (%)` (#18) | fraction of windows with a significant step | the switching rate |
| 1 | `discrete: in-window step amplitude (ADU)` (#19) | the median step size | the trap size × gain |
| 1 | `noise: jump-robust CDS sigma (ADU)` (#25) | √(MedRV/N) | noise without the jumps |
| 1 | `discrete: jump share of CDS variance` (#26) | 1 − MedRV/RV | the fraction of the variance in jumps |
| 2 | `discrete: jump count (Lee–Mykland)` (#27) | jumps against a local volatility | a calibrated count |
| 2 | `discrete: STA/LTA maximum` (#28) | the largest short/long energy ratio | localized bursts |
| 2 | `discrete: crest factor` (#29) | max\|e\| / sd | partly redundant with kurtosis but cheap |
| 2 | `noise: squared-residual autocorrelation (LB10)` (#30) | volatility clustering | mostly new information (R² 0.16) |
| 2 | `discrete: whitened CDS kurtosis` (#31) | kurtosis after whitening by the PSD | non-Gaussianity without the colour of the noise |
| 2 | `noise: RTS mean dwell time (µs)` (#21) | a 2-state HMM on the in-window block means | the trap capture and emission times |
| 2 | `noise: Lorentzian corner (kHz)` (#37) | the RTS corner in the spectrum | a cross-check of the dwell time |
| 2 | `discrete: max within-window spectral kurtosis` (#45) | the kurtogram maximum | the band of the impulsive noise |
| 3 | `discrete: step up/down balance` (#20) | (up − down)/(up + down) | about 0 everywhere; keep as a check, not a feature |
| 3 | `edge: RTS step rise time (ns)` (#22), `noise: RTS rate modulation index` (#46) | — | only meaningful where steps exist; a diagnostic for ch2 |

### 3.3 Noise shape and time scales

| tier | feature | definition (short) | why |
|---|---|---|---|
| 1 | `noise: 1/f corner frequency (kHz)` and `noise: 1/f exponent` (#36) | fit W + A f^−γ to the existing log-frequency spectrum | the standard amplifier noise numbers; delayInt should move them |
| 1 | `noise: Hadamard deviation n=50 (ADU)` and curve (#38) | the third difference of n-skip means, not detrended | averaging noise that is insensitive to drift, without the polynomial bias of the detrended Allan |
| 2 | `noise: within-cycle structure function` (curve) and `… excess at 9 µs` (#23, #24) | how the noise inside the windows grows with the time lag | the time scale of the in-window noise; switching on ch2 |
| 2 | `noise: modified Allan deviation` (curve) (#39) | MDEV | separates anticorrelated from white noise |
| 2 | `noise: noise-type exponent at n=10` (#40) | from ρ₁ of 10-skip means | the noise type at a given time scale |
| 2 | `noise: DFA exponent` | detrended fluctuation analysis | mostly new information (R² 0.15) |
| 2 | `noise: spectral slope` | log-log slope of the CDS spectrum | new information (R² 0.40), separates amplifiers |
| 2 | `noise: sample entropy`, `noise: permutation entropy (m=4)` | regularity measures | new or partly new information, separate amplifiers |
| 2 | `noise: red-noise β (50 skips)` | the rms of bin means ÷ white expectation | new information (R² 0.44) |
| 2 | `noise: spectral centroid` | the mean frequency of the CDS spectrum | separates amplifiers strongly (F 63) |
| 2 | `noise: null-CDS autocorrelation at lag 1` (#17) | ρ₁ of the null CDS | a slow change of the in-window slope |
| 3 | RMSSD/SD, Poincaré SD1/SD2, zero-crossing rate, spectral flatness | — | redundant with the lag-1 autocorrelation or kurtosis (R² 0.66–0.99); keep the Poincaré *plot* |

### 3.4 Pickup lines

| tier | feature | definition (short) | why |
|---|---|---|---|
| 1 | `discrete: line count (F-test, p < 0.001)` (#32) | multitaper Thomson F-test | pickup lines with a stated false-alarm rate (refines the existing line count) |
| 1 | `discrete: strongest line frequency (kHz)` and `… amplitude (ADU)` (#33) | from the F-test | identifies the source; can be tracked across packets |
| 1 | `discrete: line phase coherence` (#34) | phase of the line at each pixel's first skip | 1 means our own clocking makes it; 0 means it is external |
| 2 | `discrete: line frequency, whole trace (kHz)` (#35) | Lomb–Scargle over all pixels in absolute time | 5-Hz resolution |
| 3 | `discrete: cepstral peak` (#49) | — | low value |

### 3.5 Levels, edges, cross-structure

| tier | feature | definition (short) | why |
|---|---|---|---|
| 2 | `level: common-pattern amplitude spread` (#41) | per-pixel coupling to the common CDS pattern | whether the pattern couples differently to different pixels |
| 2 | `noise: baseline–CDS residual correlation` (#42) | correlation of the baseline and CDS residuals | a common supply or reset disturbance |
| 2 | `edge: random / deterministic jitter (ns)` (#43) | dual-Dirac fit to the edge-time residuals | splits the existing jitter |
| 2 | `edge: periodic jitter frequency (kHz)` (#44) | line in the edge-time residuals | pickup that modulates the timing |
| 2 | `noise: in-window drift vs baseline (1/ms)` (#14) | in-window slope against the pedestal level | an AC-coupling-like relaxation (speculative) |
| 2 | skip-cycle **noise budget** (exploratory idea 1) | the fraction of the CDS variance carried by each principal mode of the residual cycles | which waveform fluctuations actually reach the CDS: the large clock and reset modes do not |
| 3 | `edge: reset-pulse EMG fit` (#47), `discrete: max spectral flux` (#48) | — | overlap existing edge features, or low value |

---

## 4. Diagnostic plots (by hand, not features)

The most useful, from the signal report's Section 4 (D1–D15):

- **D1 differentiator plot:** first-skip CDS against the in-window slope. It shows whether the start transient is pure
  baseline slope.
- **D2 skip-0 difference waveforms** for each setting.
- **D3 Poincaré / return plot:** ch2's off-diagonal clusters would mean switching within one skip.
- **D4 RTS step-amplitude and dwell-time histograms:** the number of traps and their time constants.
- **D8 phase-folded CDS** at the strongest line: an external source or our own clocking.
- **D9 eye diagram** of the cycles: RTS levels inside the windows.
- **D15 novelty chart** of the reference traces over time: video-chain drift between packets.

---

## 5. Proposed implementation

1. **Tier attribute.** Add a `tier` field (1, 2, or 0 = existing) to every `FEATURES` entry, and a `start:` group.
   `src/ml/real_inputs.py` exports all features. `lvl1/train_lvl1.py` gets `--feature-tiers 0 1 2` (default: 0 and 1),
   so a model can be run with or without the candidates. Curves stay as they are; new curves are added to the curve set.
2. **Tier 1: about 22 numbers.** Start of pixel (7), telegraph switching (6), noise shape (3: the 1/f corner and
   exponent, Hadamard), pickup lines (4), the noise budget. Each gets a unit test on a synthetic trace with a known
   answer: a known exponential, a known telegraph signal, a known sine. They are checked on the 2026-10-12 traces
   against the pilot numbers. About 2–3 days.
3. **Tier 2: about 25 numbers,** in the same pass or a second one. About 1–2 days.
4. **Decide after session A.** Run the models with tiers {0}, {0, 1} and {0, 1, 2}, with paired comparisons against
   the current features. Read the SAAS relevance scores. Repeat the settings-separation test on session A, where
   delayInt changes the noise. Keep a tier if it helps or costs nothing; demote a feature that only adds variance.
5. **Reprocess every packet** after the change (processing recomputes from the raw files).

The two DAQ questions (NDCMPRE, previous-pixel charge) are worth one short trace test each, before the trace settings
are fixed.
