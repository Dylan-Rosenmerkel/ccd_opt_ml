# Ideas from unrelated fields for the skipper-CCD noise project: an exploratory analysis

*Written 2026-10-21 by the exploratory analyst of a three-agent team. Input: the catalogue
`docs/literature/ml_landscape_catalogue.md` ("catalogue"), the earlier review `ml_models_literature_review.md`, the model table in
`docs/model_summary/make_model_summary.py`, `lvl1/README.md`, the session notes up to `docs/context/2026-10-19_session_context.md`, and the
real-image checks in `lvl1/real_image_checks/`. A separate grounded analyst covers the directly applicable items. This report looks
for ideas from **fields that seem unrelated**, and from the gaps the catalogue left open. Speculation is allowed, but every idea is
labelled, and the physics is checked against what we know about board 101 / PA03.*

**Labels used throughout**

* **[pilot]**: I ran a quick read-only test on real data during this analysis (scripts kept outside the repository; Section 2 says
  how to repeat them). Pilot numbers are first looks, not results.
* **[speculative]**: a hypothesis that needs a test. Nothing has been measured yet.
* **P** = plausibility (1-5): will it work here, physically and statistically, with our data sizes? **N** = novelty (1-5): how
  unusual it would be in CCD or detector work (5 = I know of no one doing it).
* References: "(verified)" means I confirmed title, authors, year and venue during this analysis (arXiv API or a publisher,
  repository or index page). "(catalogue)" means it is in the catalogue's reference list, verified there. **[unverified]** means from
  memory only.

---

## 0. Summary

**Top 5 (plausibility × value), each with a first experiment that needs no new hardware time:**

| Rank | Idea | Source field | First experiment | Helps with |
|---|---|---|---|---|
| 1 | **Beat-space KLT noise budget**: treat the ~5,000 skip cycles of a trace like heartbeats; decompose their residual waveforms into modes; each mode's contribution to the CDS variance is exact | ECG analysis (KLT of ST-T complexes), machine diagnostics (cyclostationarity) | Already piloted on the 2026-10-12 traces (Section 2.2). Next: run on session-A traces and check that the per-mode budget follows `delayInt` | σe physics, ch2 |
| 2 | **Jump-robust noise estimators and the "bid–ask bounce"** from high-frequency finance (bipower and MedRV variance, Roll's autocovariance) | Finance | Already piloted (Section 2.1): ch2 has 16-41 % of its CDS variance in heavy-tailed jumps and negative lag-1 autocorrelation. The other amplifiers have neither. Next: add the 3 numbers as trace features and use MedRV in model E's set weighting | ch2, damaged sets |
| 3 | **Event clocks and impulse-response models for the row-start drift** ("which clock does the drift run on?") | Finance (transaction clock), sports science (Banister fitness–fatigue), batteries (distribution of relaxation times) | A two-clock model fits the published skip-noise table (Section 2.3). Next: refit it to the full column profiles of the skip-noise images, with a relaxation-time spectrum | drift |
| 4 | **Check plots, batch correction and interleaving** for small effects under drift | Plant breeding (augmented designs), genomics (ComBat), web search (interleaving), clinical crossover trials | Re-analyse session A with its 3 nominal packets as checks. Plan session B with ABBA pairs inside packets | small σe effects, slow drift |
| 5 | **Failed fits as censored labels** (Tobit / survival analysis) instead of dropping them | Survival analysis, econometrics | Rehearsal data: make ch2 "unfitted" a censored σe > c label and compare model bias at high noise with and without it | ch2, Level 2/3 bias |

**The most surprising transfer:** high-frequency finance. Its tools separate "diffusive" volatility from jumps and detect the
negative autocorrelation that a price bouncing between bid and ask produces. Applied to the per-skip CDS series, they separate the
four amplifiers cleanly. ch0, ch1 and ch3 look Gaussian and slightly positively correlated (a low-frequency component). ch2 is
heavy-tailed, slightly *negatively* correlated from skip to skip, and its squared values cluster. That pattern is what a two-level
(telegraph) process with dwell times of the order of a skip cycle would produce **[speculative interpretation]**. A runner-up is
the event-clock fit of the row-start drift. Three numbers from the existing skip-noise table are predicted to within 0.02 ADU by
a model in which the disturbance decays per *column* (≈ 175 columns) and is averaged within each pixel over ≈ 1,000 skips (≈ 33 ms).

---

## 1. Our problem, seen from far away

To match ideas across fields, here is what our problem looks like in abstract terms (the "Y" in "their X is our Y"):

| Abstract shape | Our concrete case | Where it bites |
|---|---|---|
| **A strongly periodic (cyclostationary) waveform** with thousands of repetitions | A skip cycle of 488.7 samples (32.6 µs at 15 MHz): pedestal window [0, 150), signal window [207, 357), then reset and clocks. 500 skips × 10 pixels per trace, plus pixel and row periods | Level-1 features, noise physics |
| **A noise level estimated from a short high-frequency record** | σ of the CDS from ~490 skips per pixel. The Allan curve is extrapolated to N = 500 | trace physics (model A) |
| **A state disturbed by a burst that recovers by *events* rather than by time** | The row-start dip: −0.9 to −1.0 ADU, recovering over ~300-600 *columns* whatever the skip count | drift diagnosis |
| **One unit that behaves differently and intermittently** | ch2 (CCD_A): CDS noise 16-37 ADU in the 10-12 traces against 5.7 for ch0, varying between days. Its electron peaks are usually unresolved | data loss, model range |
| **Labels missing exactly where the outcome is bad** | WADERs cannot fit ch2 (peaks unresolved). Level 2/3 will meet more such settings | biased training set |
| **Small effects, slow drift, few runs** | Tested voltage changes barely moved the noise. The CCD state drifts between packets (temperature-like) | Level-1 labels have little variance |
| **A setting leaves a "fingerprint" pattern across many measured features** | The 85 features separate 5 settings perfectly (59/59, nearest centroid). VDD, H rails and V rails each move their own feature subset | drift attribution |
| **Expensive, safety-limited sequential experiments** | Packets of ~10 min, `max_step` voltage limits, hardware windows | Level 2/3 |

Two **physical facts** constrain several ideas, and I checked them before using them:

* **Traces are one amplifier at a time.** Each trace acquisition reads one amplifier, selected by `FILTER` (docstring of
  `docs/slides/oct12_features.py`). Cross-amplifier coherence (EEG-style connectivity, seismic interferometry, DeepClean-style witness
  channels) therefore **cannot** be computed from traces. It needs simultaneous data: the image overscan (all 4 amplifiers are read
  together), or the raw saved skips (`mb_L1`, 15 pixels per channel).
* **The row-start dip follows columns, not time.** At 100 and 2000 skips the dip at column 100 is −0.41 and −0.20 ADU, although
  column 100 is reached 20× later at 2000 skips (`2026-10-16_PA03_PQ04_16rows.md`). Exposure (0/1/10 s) and prescan length
  (2800/5300) do not change it. BOOLCLR 0 makes it ~25 % smaller but longer. It is absent on board 102's CCD-less channels and on
  module 106.

---

## 2. Three pilots run during this analysis [pilot]

All three are read-only. Pilots 2.1 and 2.2 use the 59 board-101 traces of 2026-10-12 (`data/2026-10-12-PA03_PQ04_Packet_Tests`, 5
settings × 3 repeats × 4 amplifiers, NCOLPRE 0 for the reference group and 185 for the others), decoded with the project's own
`decode_trace`, `split_regions` and `usable_pixels`. Skips < 10 are dropped, and each pixel's CDS is detrended with a degree-4
polynomial, as in `trace_features`. Pilot 2.3 uses only numbers already published in `lvl1/real_image_checks/`.

### 2.1 Finance's jump-robust volatility estimators on the per-skip CDS

**Analogy.** In finance, a price observed at high frequency is "efficient price + microstructure noise + rare jumps". Realized
variance (RV, the mean of squared returns) counts everything. Bipower variation, $BV = \tfrac{\pi}{2}\,\overline{|r_i||r_{i-1}|}$
[Barndorff-Nielsen & Shephard 2004], and the median estimator MedRV [Andersen, Dobrev & Schaumburg 2012, **unverified**] ignore
isolated jumps, so RV/MedRV > 1 measures the jump share. A price bouncing between bid and ask gives a *negative* lag-1
autocovariance, from which Roll (1984) [**unverified**] estimated the spread. Our detrended CDS per skip, $e_i$, plays the role of the
returns: each skip is an independent look at the same charge.

**Result** (median over the usable pixels of each amplifier × setting; 16-24 pixels each):

| | RV/MedRV | jump share of variance | excess kurtosis | lag-1 autocorr. of e | lag-1 autocorr. of e² | jump-robust σ (ADU) |
|---|---|---|---|---|---|---|
| ch0, ch1, ch3 (all 5 settings) | 0.995-1.012 | ≈ 0 | −0.16 to +0.11 | **+0.10 to +0.15** | −0.02 to +0.03 | 5.7-6.7 |
| ch2 (all 5 settings) | **1.09-1.31** | **16-41 %** | **2.4-20.6** | **−0.04 to −0.13** | **+0.02 to +0.09** | 12.5-31 |

**Reading:**

* For the three good amplifiers, the CDS per skip is Gaussian. Its small positive skip-to-skip correlation is the low-frequency
  (1/f-like) part that the Allan curve sees.
* ch2 differs in *kind*, not only in level:
  * heavy tails (jumps or bursts);
  * volatility clustering (squared residuals correlated);
  * negative skip-to-skip correlation, so a fluctuation in one skip tends to reverse in the next.
* Even its jump-robust σ is still 2-5× that of ch0, so ch2 has an elevated continuous component *and* a jump component.
* The pattern fits random telegraph switching between two levels, with dwell times of the order of the 32.6 µs skip cycle. The
  CDS sees a step when a switch falls between the pedestal and the signal window, and the step often reverses one skip later
  **[speculative; an HMM fit on the pedestal-window levels would test it]**. Random telegraph signals from single traps in small
  output transistors are a known noise mechanism in sensors in general. I did not search the CCD-specific literature on it.

### 2.2 Beat-space Karhunen–Loève (PCA) of the skip cycles: which waveform fluctuations does the CDS keep?

**Analogy.** ECG analysis aligns thousands of beats, subtracts the average beat, and decomposes the beat-to-beat residuals with the
Karhunen–Loève transform. A few coefficients describe the shape changes, and their beat series reveal alternans [Laguna et al.
1999]. Here a "beat" is one skip cycle. The CDS is the linear functional $w$ (+1/150 over the pedestal window, −1/150 over the
signal window). With $C$ the covariance of the cycle residuals, the CDS variance splits exactly into $\sum_k \lambda_k (w\cdot\phi_k)^2$
over eigen-modes $\phi_k$. This is a noise budget by waveform mode.

**Result** (after per-sample degree-4 detrending along the skips):

* **The method checks out.** $w\cdot x$ reproduces the inline CDS exactly (correlation 1.000, same σ: 5.74 vs 5.74 ADU for ch0),
  so the windows are right.
* **Per-sample noise is almost the same on all amplifiers**: 152-164 ADU rms inside the windows, ch2 only ~5 % higher. Outside the
  windows the per-sample variance is ~14× larger (clock edges and reset), and none of it reaches the CDS.
* **The CDS keeps a tiny, low-frequency part of the waveform noise:**
  * ch0, ch1, ch3: the modes that carry the CDS variance are low-variance modes (ranks 47-104 of 489, each ≤ 0.01 % of the
    waveform variance), Gaussian (kurtosis ≈ 0), with spectral centroids of 0.7-3 MHz. In ch1 one mode carries 30-68 % of the CDS
    variance; in ch0 it is spread over several.
  * **ch2's CDS variance sits in a few *large* waveform modes** (ranks 5-24, 0.03-0.14 % of the waveform variance each), with
    excess kurtosis 0.6-3.7. Two to three modes carry 87-97 % of its CDS variance in the VDD, V-rail and H3 groups (71 % in the
    reference group). Their shape is a
    level that differs between the pedestal and the signal window (a random step between the windows), with spectral power at the
    lowest frequencies.
* The excess noise of ch2 is therefore a *structured, low-rank, non-Gaussian* fluctuation, consistent with Section 2.1, and not
  broadband amplifier noise.

**Why this matters beyond ch2:**

* The budget says *which part of the cycle* a σe change comes from. That is exactly what a timing scan (`delayInt`, the
  windows) changes.
* The large-variance modes (clock edges, reset) can change a lot with a setting without moving σe, which may be why the hand-made
  features separate settings perfectly while the noise barely moves. A model fed with 85 features mostly sees CDS-irrelevant
  variance. The per-mode budget gives features that are CDS-relevant by construction.

### 2.3 The row-start drift on two clocks (fit to the published table) [speculative]

The skip-noise run table (board 101 ch0, offsets at image columns 10 / 100 / 400 / 800) gives:

| skips N | col 10 | col 100 | col 400 | col 800 | ratio col100/col10 |
|---|---|---|---|---|---|
| 100 | −0.69 | −0.41 | −0.20 | −0.00 | 0.59 |
| 1000 | −0.44 | −0.27 | −0.13 | −0.06 | 0.61 |
| 2000 | −0.33 | −0.20 | −0.12 | −0.08 | 0.61 |
| (500, other day) | −0.24 | −0.26 | −0.08 | −0.02 | 1.08 |

The shape in columns (ratio ≈ 0.6 from column 10 to 100) is the same at 100, 1000 and 2000 skips: the recovery runs on a **column
clock**. The amplitude, however, falls with N. A model with **two clocks** fits both:

$$\text{offset}(c,N) = -A\,a^{c}\;\frac{M}{N}\left(1-e^{-N/M}\right),$$

which reads: a disturbance whose size decays by a factor $a$ per column ($a^{90}\approx 0.6$, i.e. an e-folding length of ≈ 175
columns) relaxes *within* each pixel over ≈ M skips, and the pixel value averages that relaxation over its N skips. With
M ≈ 1,000 skips (≈ 33 ms) and A fixed by N = 100, the model predicts −0.46 and −0.31 ADU at column 10 for N = 1000 and 2000. The
measured values are −0.44 and −0.33.

Caveats:
* three amplitudes fitted with two free numbers;
* the 500-skip row (another day) does not fit;
* the tail at column 800 *grows* with N, which points to a third, time-like component or a constant offset.

Treat it as a hypothesis generator, not a finding. Its prediction is testable:
* in raw skips (`mb_L1`) or in traces with NCOL ≥ 10, the per-skip CDS of the first pixels of a row should relax along the skips
  with τ ≈ 1,000 skips, and late pixels should not;
* the effect is ~0.2-0.3 ADU over a pixel, so many traces must be averaged;
* it must be read as a *difference between pixel positions*, because the common CDS pattern along the skips (~30 ADU,
  `level: common CDS rise`) is dominated by the coupling of the ~16,000 ADU baseline droop.

**How to repeat the pilots:**
* 2.1: per pixel, $e$ = detrended inline CDS of skips ≥ 10; RV = mean(e²), MedRV = π/(6−4√3+π) · mean(median(|e_{i−1}|,|e_i|,|e_{i+1}|)²).
* 2.2: per pixel, the stacked cycles `regions.skip[p][10:]`, detrended per sample position along the skips; eigen-decompose their
  pooled covariance; project $w$.
* Both take a minute on the 59 traces.

---

## 3. Field by field: analogies, methods, ratings and cheap tests

Column "Where": **D** data taking, **Pr** processing / features, **L1** Level-1 model, **U** uncertainty, **M** drift monitoring,
**L2/L3** optimization, **Dx** diagnosis of hardware effects (drift, ch2).

### B11 Batteries and electrochemistry

| # | Their X ↔ our Y | Transferable method | How it would work here | Where | P | N | Cheap first test |
|---|---|---|---|---|---|---|---|
| 11a | Voltage relaxation after a current pulse ↔ baseline recovery after each row's clock burst (and the baseline droop along the skips) | **Distribution of relaxation times (DRT)**: a regularized inverse Laplace transform of a relaxation curve into a spectrum of time constants [Ciucci & Chen 2015, verified] | Offset vs column (from `image_drift` block fits) and offset vs skip. Peaks that stay put across settings mark physical elements; peaks that move with BOOLCLR or NDCM identify their cause | Dx, Pr | 4 | 5 | DRT of the column profiles of images 127-135 and the skip-noise run, in both the column and the time domain. Which domain gives N-independent peaks? (Section 4, idea 3) |
| 11b | ΔQ(V) = difference of discharge curves between cycle 100 and cycle 10, whose log-variance predicted cycle life [Severson 2019, catalogue] ↔ difference between the new-settings curves and the same packet's reference curves | log-variance (and other moments) of the 14 Δ-curves as a handful of scalar features | Uses the reference trace in every packet as the "cycle 10". Drift common to both cancels | L1 | 4 | 3 | Add log var(Δ Allan curve) and log var(Δ spectrum) to the features model on the synthetic sets. Does `gp` improve at 13/41 packets? |
| 11c | Galvanostatic pulse tests (designed excitation) ↔ designed clock excitation | Vary the burst before each row on purpose (NCOLPRE 0/50/185/1000, BOOLCLR) in *traces* and fit an impulse response | Gives the row-start transfer function in a 1-s trace instead of a 5-min image | D, Dx | 4 | 3 | 4 traces per NCOLPRE value: first-pixel excess noise vs burst length (we have 0 → 1.1-1.4×, 185 → 1.8-2.6×) |

### B12 Medicine and biosignals

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 12a | Beats ↔ skip cycles | **KLT of beat residuals; alternans detection** [Laguna 1999, verified] | Section 2.2. Per-mode CDS budget, and the mode coefficient series along the skips (beat spectrum) | Pr, L1, Dx | 5 | 4 | Done [pilot]. Next: session-A traces |
| 12b | Template subtraction (fetal ECG under maternal ECG) ↔ remove the deterministic cycle waveform before spectral features | Subtract the jitter-aligned mean cycle, then compute the PSD of the residual | Residual PSD is the noise the CDS filters (see the filter-function idea in the gaps). Cleaner than spectra containing the clock feedthrough | Pr | 4 | 3 | Compare the residual PSD's low-frequency power with σ_CDS across the 59 traces |
| 12c | Crossover trials (each patient gets A and B in random order) ↔ settings compared within a packet | AB/BA (ABBA) ordering of settings inside one packet or session | Cancels linear drift between A and B. The variance of an effect estimate drops from "packet-to-packet" to "within-packet" | D, L1 | 5 | 3 | Simulate with the toy world's drift: effect σ for sequential vs ABBA ordering |
| 12d | Westgard multirule QC on control samples [Westgard 1981, verified] ↔ the nominal reference trace in every packet | Rules 1-3s, 2-2s, R-4s, 4-1s, 10-x on a few reference features (σ_CDS, baseline, edge time) | A clear, auditable "stop and recalibrate" rule for Level 3 | M, L3 | 5 | 3 | Apply to the 10-12 references and to session A when taken |
| 12e | Censored survival times ↔ failed WADERs fits | Tobit / survival likelihood (Section 4, idea 5) | | L1, U | 4 | 4 | see idea 5 |

### B13 Seismology

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 13a | Repeating unknown earthquakes in continuous data ↔ repeating unknown disturbances (pickup bursts, ch2 events) in raw traces | **FAST** similarity search [Yoon et al. 2015, verified] or the **matrix profile** [Yeh et al. 2018, verified via arXiv] | Self-join of cycle-residual subsequences: motifs (repeating shapes) and discords (one-off glitches) without templates | Dx, Pr | 3 | 5 | Matrix profile (e.g. the `stumpy` package [unverified name/version]) on the concatenated residual of one ch2 trace vs one ch0 trace |
| 13b | Coda-wave stretching to measure 0.1 % velocity changes from noise correlations [Sens-Schönfelder & Wegler 2006, verified] ↔ tiny changes in the amplifier's speed between packets | **Stretching / sub-sample registration** of the whole mean cycle waveform against the reference | A one-number "thermometer" of the video chain: the time-stretch factor of the waveform, measured coherently over the entire cycle | M, Dx | 4 | 5 | Section 4, idea 7 |
| 13c | STA/LTA triggers ↔ glitches and steps along the skips | Short-term / long-term average ratio on the CDS or on its squared residual | Simple, unsupervised event flags. Complements the `discrete:` features | Pr | 4 | 2 | Run on ch2 vs ch0. Do the triggers coincide with the jump share of 2.1? |
| 13d | Ambient-noise interferometry between stations ↔ cross-coupling between amplifiers | Cross-correlation and coherence of simultaneous noise | **Not from traces** (one amplifier per trace). Use the image overscan or the `mb_L1` raw skips, which are simultaneous | Dx | 3 | 3 | Overscan pixel-to-pixel correlation matrix of the 4 amplifiers in images 130/131. Does ch2's excess correlate with anything? |

### B14 Speech and audio

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 14a | Polyphonic music = additive notes with fixed spectra ↔ noise = additive independent sources with fixed PSD shapes | **Non-negative matrix factorization** of spectrograms [Smaragdis & Brown 2003, verified] | Matrix: (pixel or skip block) × frequency of the cycle-residual power. NMF gives a few source spectra and their activations. A ch2-only source, if any, appears as its own component | Dx, Pr | 4 | 4 | NMF with k = 3-5 on the pooled 10-12 residual spectrograms of all amplifiers |
| 14b | Speaker embeddings + PLDA scoring ("same speaker?") ↔ "same amplifier *state* as yesterday?" | Small embedding (or the 85 features) + a within/between-class covariance model | A drift monitor that knows normal trace-to-trace scatter. Not a Level-1 input (no amplifier labels there) | M | 3 | 3 | Within-amplifier vs between-session distances of reference traces, once sessions exist |

### B15 Finance and econometrics

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 15a | Jumps in prices ↔ glitches and steps in CDS | **Bipower variation, MedRV** [Barndorff-Nielsen & Shephard 2004, verified] | Jump-robust σ per pixel and set. Jump share as a feature. Set weights in model E from MedRV | Pr, L1 | 5 | 4 | Done [pilot] |
| 15b | Bid–ask bounce: negative return autocovariance [Roll 1984, unverified] ↔ telegraph switching | Negative lag-1 autocovariance of the CDS minus the good-amplifier baseline → a switching amplitude | A cheap RTS indicator per trace | Dx | 3 | 5 | ch2 shows −0.04 to −0.13 vs +0.10 to +0.15 elsewhere [pilot] |
| 15c | **Transaction clock**: returns are closer to Gaussian when measured per trade than per second [Ané & Geman 2000, verified; Clark 1973 via the same record] ↔ the drift recovers per column, not per second | Find the "subordinator": which event count (columns, H transfers, resets, skips, seconds) makes the recovery a stationary, single-shape process | Formal version of Section 2.3 | Dx | 4 | 5 | Section 4, idea 3 |
| 15d | Volatility clustering (ARCH/GARCH) ↔ bursts in ch2 | Engle LM test, GARCH(1,1) or a 2-state HMM on the CDS residual | Turns "ch2 is noisy" into "ch2 has regimes", with switching rates | Dx | 3 | 3 | acf(e²) is already +0.02 to +0.09 for ch2 [pilot]. Fit the HMM |
| 15e | Synthetic control [Abadie et al. 2010, verified] ↔ separate CCD changes from environment drift | Build the drift of a reference feature from "donor" series that settings do not touch (board-102 CCD-less channels, overscan level, temperature log) | Subtract the environmental part before attributing a change to the settings | M, L1 | 3 | 4 | Once nominal packets repeat: regress the reference features on donors over time |
| 15f | Realized-variance "signature plot" and the two-scale estimator [Zhang, Mykland & Aït-Sahalia 2005, verified] ↔ Allan curve with a white "microstructure" term | Estimator that separates white noise from the slow part using two averaging scales | A model-light alternative to the a²/N + b² fit | Pr | 3 | 3 | Compare its floor estimate with `fit: correlated noise floor` on the 59 traces |

### B16 Climate, weather and hydrology

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 16a | Detection and attribution: observed change = Σ β_k × model fingerprint_k + internal variability [Allen & Stott 2003, verified] ↔ change of the 85 features between packets | **Optimal fingerprinting** (generalized least squares with the repeat-scatter covariance) | "Which knob does today's drift look like?" Section 4, idea 6 | M, Dx | 4 | 5 | 10-12 fingerprints (VDD, H rails, V rails) vs the drift between session-A nominal packets |
| 16b | Analog ensemble: past forecasts most similar to today's, with their verified outcomes, give a calibrated ensemble [Delle Monache et al. 2013, verified] ↔ nearest past traces with their labels | k nearest training packets in a metric on repeat-standardized features; the label spread of the analogs is the predictive distribution | Uncertainty without a parametric model. Weighting recent analogs follows drift | L1, U | 4 | 3 | Synthetic sets: CRPS and coverage of the analog distribution vs the GP at 13/41 packets |
| 16c | Ungauged basins ↔ amplifiers without labels (ch2, a new module) | Regionalization: transfer from labelled units by attribute similarity | Sharpened by idea 5 (censored labels) | L1 | 3 | 3 | See idea 5 |

### B17 Materials and self-driving labs

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 17a | Autonomous phase mapping: find phase boundaries with few samples (CAMEO; physics-aware phase mapping [Kusne 2021, verified]) ↔ boundaries of regimes in settings space (fit fails, peaks unresolved, ch2 blows up, CTI appears) | Active learning for **level-set estimation** [Gotovos et al. 2013, verified] | A GP classifier or regressor whose acquisition targets the boundary, not the optimum. It maps the safe region for Level 3 | L3 | 4 | 3 | Toy world: packets needed to locate the "σe = 0.3 e⁻" contour of `delayInt` × VDD with LSE vs a grid |

### B18 Drug discovery and chemistry

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 18a | QSAR applicability domain (leverage, Williams plot, distance to training set) ↔ "flag test packets whose traces are unusual" (an open item, 2026-10-19 §5) | Leverage $h = x^\top(X^\top X)^{-1}x$ on a few principal components of the features. Widen the error bar when $h > 3p/n$ | Exactly the missing flag for packet 35 (floor-dominated trace physics) | U | 5 | 3 | Synthetic 41-packet set: does the leverage of packet syn_00035 stand out, and does widening fix its coverage? |
| 18b | Pairwise difference regression, PADRE [Tynes et al. 2021, verified] ↔ few labels, many possible pairs | Train on (x_i, x_j, x_i − x_j) → y_i − y_j and average over training anchors | n² training pairs. Packet offsets partly cancel. A critique argues it reduces to weighted averaging [Góes 2021, Sci. Rep., record seen] | L1 | 3 | 4 | Wrap `ridge` and `trees` with PADRE on the synthetic sets |
| 18c | Δ-learning: learn the correction to a cheap physics estimate [Ramakrishnan et al. 2015, verified] | — | **Already what model A does** (trace physics + learned correction). It confirms the design and needs no action | — | — | 1 | — |

### B19 Proteins

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 19a | **Global epistasis**: fitness = g(Σ additive mutation effects), g monotone and nonlinear [Otwinowski et al. 2018, verified] ↔ σe = g(Σ_j f_j(setting_j)) | Single-index / GAM with a learned monotone link (I-splines in the paper) | Few parameters. Explains apparent interactions and saturation. Readable for Level 2 | L2, L1 | 4 | 4 | Section 4, idea 9 |
| 19b | Direct coupling analysis (co-variation in sequence alignments) ↔ which of the 85 features vary *directly* together | Graphical lasso on the unlabelled-trace features | Groups features by physical origin; feature selection without labels | Pr | 3 | 3 | Sparse inverse covariance of the 59 traces' features within settings groups |

### B20 Robotics and control

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 20a | **Inertial-sensor (gyro) noise identification from the Allan variance**: white, flicker, random walk, Gauss–Markov terms fitted jointly, with uncertainties (GMWM) [Guerrier et al. 2013, verified] ↔ our a²/N + b² Allan fit | Composite noise model fitted to the wavelet (≈ Allan) variance with an efficient weighting matrix | Replaces the 2-term fit in trace physics with a model whose terms are physical (incl. anticorrelation and Gauss–Markov drift) and which has error bars | Pr, L1, U | 5 | 4 | Section 4, idea 8 |
| 20b | Iterative learning control: learn a feedforward for a repeated cycle from past errors [Arimoto et al. 1984, unverified] ↔ the row start repeats every row | Per-row feedforward correction in hardware (a DAC offset or clock shaping during the first columns) | Cancels the dip at the source instead of cutting 200 columns | D | 2 | 5 | Only after the drift's cause is known. Ask the front-end expert whether any per-row-programmable level exists |

### B21 Recommender systems and online experiments

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 21a | **Interleaving**: both rankers shown to the same user, far more sensitive than A/B tests [Chapelle et al. 2012, verified] ↔ comparing settings under drift | Interleave settings A/B *within* a packet (alternate trace sets, or alternate images in a session) | Within-packet comparisons make small σe effects measurable. Merged into idea 4 | D | 4 | 4 | idea 4 |
| 21b | Missing-not-at-random ratings ↔ labels missing where noise is high | Explicit missingness model | Merged into idea 5 | L1 | 4 | 3 | idea 5 |
| 21c | Amplifier × setting matrix factorization (catalogue) | Reference-only (needs amplifier IDs, which Level 1 forbids); fine at Level 2 | | L2 | 3 | 2 | — |

### B22 Agriculture

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 22a | **Augmented designs**: new lines unreplicated, check varieties repeated in every block, block effects estimated from the checks [Federer 1956, record only] ↔ planned packets + nominal packets at the start, middle and end of each session | Mixed model with block (time-window) effects estimated from the checks; neighbour adjustment along the run order | The formal analysis for the packet plan that already contains checks | D, L1 | 5 | 4 | idea 4 |
| 22b | Sparse sensor placement by QR pivoting on a learned basis [Manohar et al. 2018, verified] ↔ which few features or waveform samples a Level-3 monitor needs | QR-pivoted selection on the feature (or cycle-sample) PCA basis | A fast, small monitor (e.g. 8 features) computed on every trace | M, L3 | 3 | 4 | Reconstruction error of all 85 features from k selected ones on the 59 traces |

### B23 Predictive maintenance and machine diagnostics

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 23a | **Spectral kurtosis / kurtogram**, which finds the band where a bearing signal is impulsive [Antoni 2006, verified] ↔ where in frequency ch2's noise is non-Gaussian | Kurtosis of STFT coefficients per band, on the cycle residual or on the CDS series | Shows whether ch2's bursts are broadband steps or a band-limited oscillation | Dx | 4 | 5 | STFT of the 10-12 ch2 vs ch0 residuals. SK per band |
| 23b | **Cyclostationarity** of rotating machines: cyclic power, spectral correlation at the shaft frequency [Antoni 2009, verified] ↔ periods of skip, pixel and row | Cyclic variance profile and spectral correlation at α = skip rate, pixel rate, row rate | The row-start dip is cyclostationarity at the row frequency. Clock-modulated noise appears as cyclic correlation | Pr, Dx | 5 | 4 | Merged into idea 1 |
| 23c | Health index + remaining useful life ↔ slow degradation of an amplifier (does ch2 get worse over weeks?) | A trended health index (e.g. jump-robust σ and jump share of the nominal trace) with changepoints | Separates "slowly ageing" from "intermittent" | M | 3 | 2 | Collect over sessions; nothing to test yet |

### B24 Ecology

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 24a | **Taylor's law**: variance = a·mean^b; b > 1 means aggregation [Taylor 1961, verified] ↔ is dark charge Poisson, or clustered (hot columns, CIC hot spots)? | Dispersion of 1e⁻ counts across column blocks and rows; a negative-binomial λ | The 10-12 binned images already showed bin-to-bin variance dominated by uneven dark charge. If 1×1 images are overdispersed too, the Poisson λ errors are too small | U, Pr | 4 | 4 | Index of dispersion of 1e⁻ counts per 25-column block in images 130/131 (after the drift cut), compared with Poisson |
| 24b | **Occupancy with imperfect detection** [MacKenzie et al. 2002, verified]: repeat surveys separate "absent" from "present but missed" ↔ an intermittent source seen in some of a packet's 10 trace sets | Occupancy likelihood: P(source active in packet) × P(detected in a set) | A principled alternative to hard flags for damaged sets (model E's job), and a per-packet probability that ch2's burst mode was on | Pr, U | 3 | 5 | On the synthetic intermittent-pickup world: compare occupancy-weighted sets with model E |

### B25 NLP and LLMs

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 25a | Agents that read logs and propose experiments ↔ the drift investigation itself | An LLM with read access to the context documents and image headers, proposing *discriminating* tests and checking them against the record | This is how the drift work of 2026-10-16 was in fact done. The transferable lesson is to keep a structured hypothesis log (claim, test, result) so that retracted claims (FILTER 85, the transient) do not resurface | Dx | 3 | 2 | Convert the drift notes into a hypothesis table |

### B26 Computer vision

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 26a | Single-image noise-level estimation from the smallest eigenvalues of patch covariances [Liu, Tanaka & Okutomi 2013, unverified] ↔ white floor of the cycle residual | Smallest eigenvalues of the beat-space covariance (2.2) | A structure-free white-noise estimate per trace | Pr | 4 | 3 | Read it off the pilot 2.2 eigenvalues |
| 26b | Push-broom destriping ↔ column pattern of the row-start dip | Variational destriping | The current cut + slow-remainder subtraction already works (χ²/ndf 0.65-1.15). Low added value | Pr | 3 | 2 | — |

### B27 Sports science

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 27a | **Fitness–fatigue impulse-response model**: performance = response to a train of training "impulses", two first-order systems with different time constants [Calvert, Banister et al. 1976, verified] ↔ baseline = response to the train of clock events since the row start | Linear impulse response driven by the event sequence (fast H transfers of the clear and prescan, then one H transfer per pixel), with event-indexed decay | Predicts the *trace* row-start disturbance (short rows, NCOL 5-10) from the *image* recovery, and the dependence on BOOLCLR, NCOLPRE and NCOL | Dx | 4 | 5 | idea 3 |

### B28 Gaming

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 28a | Route planning ↔ ordering a batch of packets | Travelling-salesman ordering under ramp and settling costs, constrained to keep checks spread and the order randomized with respect to drift | Saves ramp time in Levels 2/3 without confounding settings with time | L3 | 4 | 3 | Session C (11 joint packets): total |Δvoltage| of the plan order vs a TSP order |

### B29 Hyperparameter optimization

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| 29a | Learning-curve extrapolation with prior-fitted networks (LC-PFN) [Adriaensen 2023, catalogue] ↔ **the Allan curve is a learning curve** (error vs number of averages), extrapolated to N = 500 | An "Allan-PFN": a small transformer pretrained on simulated Allan curves of composite noise (white, 1/f, random walk, Gauss–Markov, steps) | Bayesian extrapolation with calibrated uncertainty, including the n > N/5 bias of short records | Pr, U | 3 | 5 | Simpler first: GMWM (20a). Build the PFN only if GMWM's error bars are poorly calibrated |

### Fields outside the catalogue (added)

| # | Their X ↔ our Y | Method | How | Where | P | N | First test |
|---|---|---|---|---|---|---|---|
| X1 | Psychophysics: thresholds found with few binary trials by a Bayesian staircase, **QUEST** [Watson & Pelli 1983, verified] ↔ the voltage at which an amplifier "fails" (peaks unresolved, bias collapse) | Bayesian adaptive search for a 1-D threshold of a binary outcome | Level-3 "cliff finding" per knob with few packets | L3 | 4 | 5 | idea 10 |
| X2 | Genomics batch effects, **ComBat** [Johnson, Li & Rabinovic 2007, verified] ↔ session or cool-down offsets in the trace features | Empirical-Bayes location/scale adjustment per batch, preserving the setting effects | Harmonizes features across sessions before Level 1, robust at small batch size | Pr, L1 | 4 | 4 | idea 4 |
| X3 | Econometric limited-dependent variables (Tobit [Tobin 1958, unverified]) | Censored likelihood | idea 5 | L1 | 4 | 4 | idea 5 |

### The catalogue's unsearched gaps

| # | Gap | Idea | Where | P | N | First test / verdict |
|---|---|---|---|---|---|---|
| G1 | **Optimal transport** | (i) **1-D OT = quantile functions**: compare the zero-electron pixel distribution of each column block with the row end. The quantile difference gives shift, width and *shape* of the row-start disturbance without Gaussian fits. (ii) OT domain adaptation [Courty et al. 2015, verified] to align feature clouds across cool-downs. (iii) *Rejected for correction*: climate-style "quantile mapping" would shrink real pixel-level noise | Dx, M | (i) 4, (ii) 2 | 3 | (i) Quantile curves of blocks 0-25 … 400-852 in image 131 |
| G2 | **Bayesian quadrature / probabilistic numerics** | **Filter-function prediction**: σ²_CDS(T, N) = ∫ S(f) \|H_{T}(f)\|² \|G_N(f)\|² df, with S(f) the residual noise PSD from the trace (12b), H_T the CDS window response and G_N the skip-averaging comb. A GP over S(f) and Bayesian quadrature give the integral *with* uncertainty. This predicts σ for any `delayInt` or window length **without labels**. Physics-first; close to the qubit filter-function idea in the catalogue's B6 | Pr, L1, L2 | 4 | 3 | Check self-consistency on existing traces: the predicted vs measured `noise vs window length` curve (8 window lengths) |
| G3 | **HIVE-COTE** (time-series classification ensembles) | Classifies series. Our classification is already solved (59/59 nearest centroid), and the regression needs are covered by MiniRocket (prior review) | — | 1 | 1 | Rejected (Section 5) |
| G4 | **Quantum ML** | No quantum data and no advantage at our scale | — | 1 | — | Rejected |
| G5 | **Neuromorphic / spiking networks** | Event-driven on-FPGA glitch detection is conceivable for Level 3, but STA/LTA (13c) does the same in 10 lines | — | 1 | 4 | Rejected for now |

---

## 4. Big ideas

### Idea 1. A noise budget from beat-space modes (ECG KLT + machine cyclostationarity) [pilot]

**The idea.** Treat each trace as ~5,000 aligned "beats" (skip cycles):
1. subtract the mean cycle, after a sub-sample alignment if edge jitter matters;
2. eigen-decompose the residual covariance;
3. attribute the CDS variance exactly to modes: $\sigma^2_{CDS}=\sum_k\lambda_k(w\cdot\phi_k)^2$;
4. for each important mode, follow its coefficient along the skips (Allan deviation, kurtosis, beat spectrum).

Extend it to the other periods: cyclic variance and spectral correlation at the pixel rate and the row rate (Antoni's
cyclostationary toolkit).

**Why it might work here.**
* It is exact linear algebra on data we already have.
* It answers the question the 85 features cannot: *which* waveform fluctuations reach σe. The pilot shows that the dominant
  waveform variance (clock edges, reset; ~14× the in-window variance per sample) does not reach the CDS at all, while for ch2 a
  handful of large, heavy-tailed modes carry most of the CDS variance.
* Because σe is a linear functional of the waveform, the budget also predicts how σe changes when the windows change. A
  different $w$ for a different `delayInt` window gives a new budget from the *same* covariance, as long as the noise process
  itself does not change with the timing. This is a label-free prior for session A.

**What it would take.**
* A function in the feature code that returns, per trace:
  * the top ~5 CDS-carrying modes (shape, CDS share, kurtosis, autocorrelation);
  * the white floor (smallest eigenvalues);
  * the cyclic variance profile.
* About 100 lines.
* For the predicted-σ use, the CDS weight vector for other window settings (simple), and a check that the in-window residual
  statistics do not depend on the window placement.

**Quick dead-end test.**
* On session-A traces: does $w_{T}^\top C\, w_{T}$, computed from the *nominal* trace's covariance for each scanned window
  length T, predict the measured CDS σ of the scanned traces within the trace-to-trace scatter?
* If it is off by more than ~10 %, the noise process depends on the timing itself (e.g. settling), and the budget stays a
  diagnostic only.
* For ch2: if its top modes are the same shape in every session, the cause is stable hardware; if they wander, it is pickup.

**P 5 · N 4 · Value: high** (σe physics, a model-A prior for timing scans, ch2 diagnosis).

### Idea 2. Market-microstructure statistics on the CDS series (finance) [pilot]

**The idea.** Add three cheap numbers per pixel and per set:
* jump-robust σ (MedRV or BV);
* jump share (1 − MedRV/RV);
* the sign and size of the lag-1 autocovariance relative to the good-amplifier baseline (a Roll-style "bounce" amplitude).

Use MedRV instead of RV where a robust σ is wanted. Fit a 2-state HMM to the pedestal-window level sequence when the jump share is high.

**Why it might work here.**
* The pilot separates ch2 from the others on every one of these numbers, in all 5 settings.
* The robust σ solves part of the "damaged sets" problem without a learned network. In the synthetic packet syn_00035, half the
  sets were damaged and the mean Allan curve's floor was inflated. A MedRV-based Allan curve would be largely immune to *sparse*
  glitches (not to dense pickup).
* For the image, the relevant σ is the one *including* the jumps (the image averages everything), so both numbers are needed:
  the robust σ describes the "clean" amplifier, and the jump share is what the image will additionally see.

**What it would take.** About 30 lines in `trace_features`, plus 3 features; a MedRV-based Allan variant for trace physics; an HMM
(`hmmlearn`) on ch2 traces for the telegraph hypothesis.

**Quick dead-end test.**
* The interpretation (telegraph switching) predicts that raw pedestal-window *levels* show discrete steps of a fixed size, with
  exponential dwell times.
* If the 2-state HMM is not clearly better than a 1-state Gaussian with the same variance (BIC), the bursts are something else
  (e.g. pickup), and only the robust-σ part of the idea survives.
* In the image: the ch2 overscan pixel distribution should then show a non-Gaussian, possibly bimodal, component.

**P 5 · N 4 · Value: high** (ch2, robustness). **Helps explain ch2 directly.**

### Idea 3. Which clock does the row-start drift run on? (finance transaction clock, sports impulse-response, battery DRT) [speculative + fit]

**The idea.** Model the baseline at pixel $c$ of a row as a linear response to the *sequence of clock events* since the row
start:
* the burst (register clear, prescan skip);
* one H transfer and N reset/skip cycles per pixel.

Each event kicks the state, and relaxation is indexed by an event count, by time, or both. The question "which clock?" becomes a
model comparison: refit the same impulse response with decay indexed by (a) seconds, (b) columns, (c) skips, (d) combinations. The
right clock makes the profiles of all skip counts collapse onto one curve, as returns become near-Gaussian on the transaction clock.
A distribution-of-relaxation-times deconvolution in the winning clock then gives the spectrum of time constants, whose peaks can be
followed across BOOLCLR, NCOLPRE and NDCM.

**Why it might work here.**
* The data already say "columns, not time" for the shape. Section 2.3 shows that a two-clock model with an intra-pixel relaxation of
  ≈ 1,000 skips also explains the falling amplitude with N.
* An event-clocked process points to *charge or state that is moved per transfer* (a trap population, a coupling capacitor
  re-charged per H pulse, an AC-coupled stage restored per pixel) rather than a thermal or RC time constant. That is a concrete
  question for the board-101 front-end expert.
* The same model must also explain the *traces*' first-pixel noise excess (1.1-1.4× at NCOLPRE 0, 1.8-2.6× at 185). That is a free
  cross-check between images and traces.

**What it would take.**
* The full column profiles of the skip-noise images (100-5000 skips) and of images 127-135, from `image_drift`'s block fits.
* A small fitting script with 3-4 clock hypotheses and DRT (Tikhonov, or the Bayesian version) in each.
* One new measurement would be decisive and costs minutes: a 16-row image at NDCM 100 with NCOL extended, so that more columns are
  read, plus raw skips (`mb_L1`) of row-start pixels to see the intra-pixel relaxation directly.

**Quick dead-end test.**
* If no single clock (or pair) collapses the 100/1000/2000/5000-skip profiles within their errors, the process is not a linear
  event-driven relaxation (e.g. it saturates), and the idea reduces to empirical correction, which already works.
* If (b) wins cleanly with *no* intra-pixel term, the N-dependence of the amplitude must come from elsewhere (e.g. the overscan
  reference) and should be checked against row-end references.

**P 4 · N 5 · Value: high for diagnosis** (the correction is already in place; this is about finding and removing the cause).
**Targets the drift directly.**

### Idea 4. Measuring small effects under drift: check plots, batch correction, interleaving (agriculture, genomics, web search, clinical trials)

**The idea.** Three borrowed designs and one analysis:
* **Augmented design** (plant breeding). The nominal packets at the start, middle and end of a session are "check plots". Fit
  block (time-window) effects from them, with neighbour adjustment along the run order, and analyse the other packets relative to
  the local block level.
* **ComBat** (genomics) harmonizes feature location and scale per session or cool-down with empirical-Bayes shrinkage, which is
  stable for small batches. The setting effects are protected as covariates.
* **Interleaving and crossover** (web search, clinical trials). For a planned comparison, alternate A and B *within* a packet or
  session (ABBA), so that linear drift cancels exactly.

**Why it might work here.**
* The tested voltage changes moved σe little, and the CCD drifts between packets. When the effect is small relative to the drift,
  the *design* matters more than the model.
* The packet plan already contains checks, so this is mostly an analysis choice.
* Interleaving made web-search comparisons far more sensitive than A/B tests, for exactly this reason: each unit sees both
  conditions.

**What it would take.**
* A mixed model (`statsmodels` MixedLM or `bambi`) with a block term per session window, estimated from the checks.
* ComBat (`pycombat` / `neuroCombat` [unverified package names]) on the features.
* For interleaving, the packet script would take two settings and alternate trace sets (A, B, B, A). That changes data taking and
  is for the user to decide.

**Quick dead-end test.**
* Toy world with the simulator's drift: compare the variance of an estimated setting effect for (a) the current plan, (b) the plan
  analysed with check-based block effects, (c) an ABBA plan.
* If (b) and (c) do not reduce the variance by at least ~30 %, the drift is too small to matter and the plain plan is fine.
* On real data: if the three nominal packets of session A agree within the label errors, there is nothing for the checks to
  correct yet.

**P 5 · N 4 · Value: high** (Level-1 sensitivity; it matters most exactly when "noise barely moved").

### Idea 5. Failed fits are censored labels, not missing ones (survival analysis, Tobit)

**The idea.** When WADERs cannot resolve the electron peaks (ch2, or any amplifier at a bad setting), the label is not "unknown":
it is "σe > c", where c ≈ 0.3-0.35 e⁻ for 500 skips is the noise above which the 0/1 e⁻ peaks merge. Its σ in ADU (overscan
noise) is even *measured*, and only the gain is missing; the gain of the other amplifiers is 1.25-1.34 ADU/e⁻ with a small spread.
Two ways to use this:
* a censored (Tobit-type) likelihood for σe, P(σe > c | x), inside the GPs (for example through an expectation-propagation or
  variational likelihood);
* a σe label with an inflated error from σ_ADU / gain prior.

**Why it might work here.**
* Dropping those rows biases every model toward optimistic predictions in exactly the regions Level 2/3 must avoid. That is the
  "missing not at random" problem of recommender ratings and of survival data.
* ch2 is an amplifier with a different noise mechanism (Section 2); including it widens the range the model has seen.

**What it would take.** A censored-Gaussian likelihood in the λ/σe GPs (or a two-part model: a classifier for "fit fails" plus
the regressor). A rule in `process_packet` that records *why* a label is missing (peaks unresolved vs. other failure), and the
overscan σ_ADU of every amplifier.

**Quick dead-end test.**
* Rehearsal / toy world: generate packets where the high-noise amplifier is "unfitted". Compare predicted σe on held-out
  high-noise rows for drop vs censor.
* If censoring does not reduce the bias at the high end, and the high end never matters for Level 2 (it only needs to *avoid*
  it), a "fit fails" classifier alone is enough.

**P 4 · N 4 · Value: high for Level 2/3, medium for Level 1.** **Uses ch2 instead of discarding it.**

### Idea 6. Optimal fingerprinting: which knob does the drift look like? (climate detection and attribution)

**The idea.** The 10-12 analysis already measured "fingerprints": the pattern of changes across the 85 features caused by VDD
(amplifier speed only), H low rails (H-transfer shape and baseline), and V rails/TG (V-transfer minimum), each with a repeat-scatter
covariance. Any observed change, such as the drift between two nominal packets or between sessions, can be regressed on these
fingerprints with generalized least squares, weighted by the inverse repeat covariance, as climate scientists attribute warming to
forcings. This gives a scaling factor per candidate cause with an interval, plus a residual test ("unexplained by any known knob").

**Why it might work here.** The slow drift is "temperature-like" but not measured. If it projects onto the VDD fingerprint
(amplifier speed) it is likely the output transistor's operating point; if onto the H-rail fingerprint, the clock drivers. The
residual consistency test is a principled "new effect" alarm for Level 3.

**What it would take.**
* Fingerprints for more knobs (the session-B one-group-at-a-time packets are exactly that).
* The repeat covariance from the 3-10 trace sets per packet, shrunk (Ledoit–Wolf) because 85 features exceed the repeats.
* About 50 lines.

**Quick dead-end test.** Fingerprints that are nearly collinear (e.g. VDD and RG both acting through the amplifier speed) cannot be
separated. If the condition number of the fingerprint matrix is large, report groups of knobs, not knobs. If the drift between the
session-A nominal packets is within the repeat scatter, there is nothing to attribute yet.

**P 4 · N 5 · Value: medium-high** (drift diagnosis, Level-3 alarm).

### Idea 7. A waveform-stretch thermometer (seismic coda-wave interferometry)

**The idea.** Seismologists detect 0.1 % velocity changes by measuring how much a whole waveform is *stretched* in time against a
reference, using every sample coherently rather than one feature. Our mean skip cycle (and the H- and V-transfer waveforms) is a
very high signal-to-noise template. Fit, per trace, a stretch factor ε and a gain factor, $x(t)\approx g\,x_\text{ref}((1+\varepsilon)t)$,
with sub-sample interpolation (the edge code already does equivalent-time sampling), separately for the reset edge region and the
settling regions.

**Why it might work here.**
* Amplifier speed changes are what VDD did (+0.009 µs settling, +3 ns rise time), and what temperature would do.
* The current edge features use a few samples each. A whole-waveform stretch uses hundreds and should be several times more
  precise. That turns the slow "temperature-like" drift between packets into a measured covariate for Level 1 (packet state) and
  for time-varying BO at Level 2.

**What it would take.** A least-squares registration (two parameters per region), about 40 lines.

**Quick dead-end test.**
* On the 10-12 traces: the stretch between the reference and VDD −20.5 V groups should be clearly non-zero, and the scatter
  between repeats should be several times smaller than with the existing `fit: settling time` feature.
* If the repeat scatter is not smaller, nothing is gained over the existing features.

**P 4 · N 5 · Value: medium** (drift covariate, monitoring).

### Idea 8. Inertial-sensor noise identification for the Allan curve (gyro calibration: GMWM)

**The idea.** Navigation engineers identify gyro noise by fitting a *sum* of standard processes (white, quantization, flicker,
random walk, first-order Gauss–Markov) to the Allan (or Haar wavelet) variance. GMWM does this as a proper estimator, with a
weighting matrix from the variance of the wavelet-variance estimates, consistency results, and uncertainties. Our trace physics fits
only $a^2/N + b^2$, and the notes say the floor is "often 0: the CDS noise averages down faster than white". That is the signature of
an anticorrelated (or high-pass) component, which a composite model can represent (e.g. white + MA(1) or Gauss–Markov terms), and
which the two-term fit forces into a zero floor.

**Why it might work here.** The image σe at N = 500 is an extrapolation of the trace's Allan curve beyond the point where it is
reliable (n > N/5 is biased low). A composite model with honest uncertainties improves both the central value and model A's
error bar.

**What it would take.**
* A Python implementation (the reference GMWM code is in R [unverified package status]), or a small weighted least-squares fit
  of a composite model with the Allan-variance covariance from simulation.
* Then swap it into `trace_physics` and compare.

**Quick dead-end test.**
* On the synthetic worlds (known truth) and the real traces: does the composite fit improve the trace_physics → image σe
  prediction (rank and error)?
* If the two-term fit is already within the label error on synthetic data, the gain will only show on real data (where the
  anticorrelation lives), so judge it on session A.

**P 5 · N 4 · Value: medium-high** (model A's core term).

### Idea 9. Global epistasis as a Level-2 surrogate (protein fitness landscapes)

**The idea.** Mutational-scanning data are explained surprisingly well by "fitness = g(sum of additive effects)", with g a learned
monotone nonlinearity. The nonlinearity alone creates apparent interactions and saturation. Applied here:
$\log\sigma_e = g\big(\sum_j f_j(\theta_j)\big)$, where each $f_j$ is a 1-D spline in one setting and g is monotone (I-splines). It
is a single-index GAM.

**Why it might work here.** With 13-41 packets in ~61 dimensions, full GPs are under-determined, and SAAS helped λ only. The
one-at-a-time sessions (A: timing, B: one voltage group at a time) give exactly the data that identify the $f_j$, and session C's
joint changes test whether interactions beyond g are needed. For Level 2 the model is readable ("VDD matters through the common
nonlinearity"), monotone where physics says so, and cheap to optimize.

**What it would take.** A small PyTorch or `pygam` model with a monotone output link, with uncertainty by bootstrap or a Laplace
approximation, on the settings only (the Level-2 setting) and on settings + trace physics.

**Quick dead-end test.** On the synthetic worlds (whose truth is close to multiplicative, e.g. σ₁ ∝ √(1000/delayInt)), the
single-index model should match the GP at 13 packets. On session C, if the residuals show clear pairwise structure, local
interactions dominate and a GP with an additive-plus-interaction kernel is the better surrogate.

**P 4 · N 4 · Value: medium-high for Level 2.**

### Idea 10. Cliff finding with Bayesian staircases (psychophysics QUEST) and level-set estimation (autonomous materials labs)

**The idea.** Some Level-3 questions are not "where is the optimum" but "where does it break", for example:
* the VDD at which the output stage loses its operating point;
* the clock rail at which charge transfer fails;
* the noise at which electron peaks merge.

Psychophysics finds such thresholds with 20-40 binary trials using QUEST: a Bayesian posterior over the threshold, each trial
placed at its most probable value, and a known psychometric slope. In several dimensions, GP level-set estimation (Gotovos et al.),
as used in autonomous phase mapping, does the same.

**Why it might work here.** The outcome is binary and cheap to judge from a *trace* (peaks resolved? baseline in range? jump share
exploding?), so it can run at trace cost, not image cost. The safety rules (`max_step`, windows) translate into a prior that
starts the staircase from the safe side.

**What it would take.** A QUEST implementation (e.g. PsychoPy's; [unverified API]) or a 1-D Bernoulli-GP, with a trace-based
"failure" classifier defined first. Hardware time, and the user's decision on which cliffs are worth mapping.

**Quick dead-end test.** In the toy world, compare packets-to-locate-threshold for QUEST vs a coarse grid. If the trace-based
failure flag is not reliable (disagrees with images), the method has nothing to work with.

**P 4 · N 5 · Value: medium** (Level 3 only; not urgent).

### Idea 11. Is the dark charge Poisson? (ecology's Taylor's law)

**The idea.** Ecologists test whether counts are random (variance = mean) or clustered (variance ∝ mean^b, b > 1). Apply this to
1-electron counts across column blocks and rows of the 1×1 images (after the drift cut). If they are overdispersed (hot columns,
clustered CIC), a negative-binomial λ model gives honest λ errors. The current Poisson errors would be too small, which would show
up as `physics_joint`'s λ calibration factor > 1 on real data.

**Why it might work here.** The 10-12 binned images already showed bin-to-bin variance dominated by uneven dark charge. On the
1×1 images the question is open and decides whether λ labels need wider errors.

**Quick test and dead-end criterion.** Index of dispersion per 25-column block in images 130/131 against its Poisson sampling
distribution. If it is consistent with 1, nothing changes.

**P 4 · N 4 · Value: medium** (λ labels and their errors).

---

## 5. Considered and rejected (or demoted)

| Idea | Source | Why rejected |
|---|---|---|
| **Cross-amplifier coherence or ICA from traces** (EEG connectivity, seismic interferometry, DeepClean witnesses) | B12, B13, B4 | Traces read one amplifier at a time (`FILTER`), so no simultaneous data. Kept only for image overscan or raw skips (13d) |
| **"Portfolio" optimal skip weights** (minimum-variance weights $\Sigma^{-1}\mathbf 1$ with shrinkage covariance) | B15 | Physically right, but optimal readout filters for skipper CCDs already exist (Chierchie et al., Fermilab-PUB-20-319 [record seen]; the MAS-CCD readout work, arXiv:2503.07930, verified). Images also average skips before we see them. Not novel and not usable without raw skips |
| **Quantile mapping to correct the row-start columns** | B16 bias correction | It maps each column's pixel distribution onto the row-end distribution, which would *shrink* the real extra noise of the first columns. The values would look clean while being wrong. Cutting columns is honest. 1-D OT is kept for *diagnosis* only (G1) |
| **Model-free RL, AlphaZero/MCTS scan planning** | B28 | Needs a simulator of the CCD response, which we lack. BO plus a TSP ordering (28a) captures the planning value |
| **HIVE-COTE** | gap | A classification ensemble. Our classification is trivial (59/59), and its regression cousins are covered by MiniRocket in the prior review |
| **Quantum ML** | gap | No quantum data, no advantage, no tooling fit |
| **Spiking / neuromorphic networks** | gap | Online glitch detection is 10 lines of STA/LTA; there is no power or latency constraint that needs neuromorphic hardware |
| **Speaker embeddings as Level-1 inputs** | B14 | Would leak amplifier identity, which Level 1 forbids. Kept only as a drift monitor (14b) |
| **Large self-supervised trace models** (wav2vec-style) | B14, B19 | Model D already does in-domain pretraining. The pilot shows the CDS-relevant variance is a tiny fraction of the waveform variance, so reconstruction objectives mostly learn clock edges. Better: self-supervision on the beat-space residual (idea 1) |
| **Hot-hand / runs tests for glitch streaks** | B27 | Subsumed by the HMM and volatility statistics of idea 2 |
| **Destriping (remote sensing)** | B26 | The existing cut + slow-remainder subtraction gives χ²/ndf 0.65-1.15. Nothing left to gain |
| **PADRE as a headline method** | B18 | Demoted to a table entry: gains are reported for random forests on large chemistry sets, a critique argues it is equivalent to weighted averaging, and our GPs already pool across rows |
| **Iterative learning control in hardware** | B20 | Kept at P 2: only meaningful after the drift's cause is known, and it needs a per-row programmable level that may not exist |

---

## 6. Ranked top 5 with first experiments

Ranking by plausibility × value for the project as it stands (Level 1 about to see real packets, drift corrected but unexplained,
ch2 discarded).

1. **Beat-space KLT noise budget (idea 1).** *First experiment:*
   * put the pilot into a function;
   * on the 59 traces, write per trace the top-3 CDS-carrying modes (share, kurtosis, lag-1 autocorrelation), the white floor, and
     the in-window/out-of-window variance ratio;
   * when session A arrives, test whether the nominal trace's covariance predicts the CDS σ at the other `delayInt` values.

   *Drift / ch2:* identifies the ch2 modes; the row-rate cyclic variance on longer traces (NCOL ≥ 10) measures the row-start
   disturbance in traces.
2. **Jump-robust estimators and the bounce statistic (idea 2).** *First experiment:*
   * add MedRV σ, jump share and lag-1 autocovariance to `trace_features` as 3 features;
   * fit a 2-state HMM to ch2's pedestal-window levels (BIC vs 1 state);
   * on the synthetic 41-packet set, recompute packet syn_00035's Allan curve with MedRV and see whether its trace-physics error
     drops.

   *ch2:* direct (telegraph hypothesis). *Drift:* none.
3. **Event-clock model of the row-start drift (idea 3).** *First experiment:*
   * fit the 3-4 clock hypotheses to the full column profiles of the skip-noise images (100, 500, 1000, 2000, 5000 skips) and
     images 127-135, with the block fits `image_drift` already makes;
   * if the two-clock form holds, take one 16-row image at 100 skips with NCOL widened and saved raw skips of the first pixels,
     to see the intra-pixel relaxation directly.

   *Drift:* direct; it turns "follows columns" into a mechanism class to put to the front-end expert.
4. **Check plots + ComBat + interleaving (idea 4).** *First experiment:*
   * a toy-world simulation of the three analysis/design variants (current, check-adjusted, ABBA), measuring the variance of a
     `delayInt` effect estimate;
   * then apply the check-adjusted analysis to session A as soon as it exists.

   *Drift:* separates slow drift from setting effects, and gives it a measured size per session.
5. **Censored labels for failed fits (idea 5).** *First experiment:*
   * in the rehearsal or toy world, mark the high-noise amplifier's labels as censored (σe > 0.3 e⁻) instead of dropped;
   * compare bias and coverage on held-out high-noise rows;
   * in `process_packet`, record the reason a label is missing and the overscan σ_ADU for every amplifier (no model change).

   *ch2:* brings it back into training with honest uncertainty.

Close behind: **GMWM for the Allan curve** (idea 8: high plausibility, core term of model A, but its value shows only on real
data), **optimal fingerprinting** (idea 6: needs session-B fingerprints first) and the **waveform-stretch thermometer** (idea 7:
cheap, but only useful once drift between packets is shown to matter).

**Which ideas also bear on the hardware puzzles:**

| Puzzle | Most relevant ideas |
|---|---|
| Row-start drift | 3 (event clock, DRT), 1 (row-rate cyclostationarity in longer traces), G1 (1-D OT profiles: shape of the disturbance beyond a Gaussian shift), 11c (designed burst-length traces), 6 (attribution, if the dip size drifts between sessions) |
| Noisy ch2 | 2 (heavy tails, negative autocorrelation, volatility clustering: telegraph-like) [pilot], 1 (ch2's CDS noise in a few large, non-Gaussian modes) [pilot], 23a (spectral kurtosis: in which band), 14a (NMF: is there a ch2-only source), 13d (is it shared with other amplifiers, from overscan), 5 (keep it as censored data) |
| Slow, temperature-like drift | 7 (stretch thermometer), 4 (checks, ComBat), 6 (fingerprint attribution), 12d (Westgard rules for Level 3) |

---

## 7. References

Verified during this analysis unless marked. Keys "(catalogue)" are in `ml_landscape_catalogue.md`'s reference list.

* **[Adriaensen 2023]** (catalogue). LC-PFN learning-curve extrapolation. arXiv:2310.20447.
* **[Allen & Stott 2003]** Allen, Stott. Estimating signal amplitudes in optimal fingerprinting, part I: theory. *Climate Dynamics*
  21:477–491 (2003). (verified: ORA/STFC records)
* **[Andersen, Dobrev & Schaumburg 2012]** Jump-robust volatility estimation using nearest neighbor truncation (MedRV). *J.
  Econometrics* 169 (2012). **[unverified]**
* **[Ané & Geman 2000]** Ané, Geman. Order flow, transaction clock, and normality of asset returns. *J. Finance* 55(5):2259–2284
  (2000). doi:10.1111/0022-1082.00286 (verified: RePEc). Builds on Clark (1973), a subordinated process with volume as the clock,
  *Econometrica* (cited via the same record).
* **[Antoni 2006]** Antoni. The spectral kurtosis: a useful tool for characterising non-stationary signals. *Mech. Syst. Signal
  Process.* 20(2):282–307 (2006). (verified: citation records)
* **[Antoni 2009]** Antoni. Cyclostationarity by examples. *Mech. Syst. Signal Process.* 23:987–1036 (2009).
  doi:10.1016/j.ymssp.2008.10.010 (verified)
* **[Arimoto et al. 1984]** Arimoto, Kawamura, Miyazaki. Bettering operation of robots by learning. *J. Robotic Systems* (1984).
  **[unverified]**
* **[Barndorff-Nielsen & Shephard 2004]** Power and bipower variation with stochastic volatility and jumps. *J. Financial
  Econometrics* 2(1):1–37 (2004). doi:10.1093/jjfinec/nbh001 (verified)
* **[Calvert, Banister et al. 1976]** Calvert, Banister, Savage, Bach. A systems model of the effects of training on physical
  performance. *IEEE Trans. Syst. Man Cybern.* SMC-6(2):94–102 (1976). (verified: secondary records; author list per the search
  result)
* **[Chapelle et al. 2012]** Chapelle, Joachims, Radlinski, Yue. Large-scale validation and analysis of interleaved search
  evaluation. *ACM TOIS* 30(1):6 (2012). doi:10.1145/2094072.2094078 (verified)
* **[Chierchie et al. 2020]** Optimal-filter readout of Skipper CCDs with charge-transfer and feedthrough modelling,
  Fermilab-PUB-20-319; journal version "Detailed modeling of the video signal and optimal readout of charge-coupled devices",
  doi:10.1002/cta.2784. (record seen; exact title of the Fermilab report **[unverified]**)
* **[Ciucci & Chen 2015]** Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: a
  Bayesian and hierarchical Bayesian approach. *Electrochim. Acta* 167:439–454 (2015). (verified: repository listing; details of the
  method **[unverified]**). Related tool: DRTtools (Wan et al., *Electrochim. Acta* 184:483–499, 2015).
* **[Courty et al. 2015]** Courty, Flamary, Tuia, Rakotomamonjy. Optimal transport for domain adaptation. arXiv:1507.00504 (verified)
* **[Delle Monache et al. 2013]** Probabilistic weather prediction with an analog ensemble. *Mon. Weather Rev.* 141:3498–3516
  (2013). doi:10.1175/MWR-D-12-00281.1 (verified)
* **[Federer 1956]** Federer. Augmented (or hoonuiaku) designs. *Hawaiian Planters' Record* (1956). (bibliographic record only)
* **[Góes 2021]** Góes. Pairwise difference regressions are just weighted averages. *Sci. Rep.* (2021). (record seen in search)
* **[Gotovos et al. 2013]** Gotovos, Casati, Hitz, Krause. Active learning for level set estimation. IJCAI 2013, pp. 1344–1350.
  (verified)
* **[Guerrier et al. 2013]** Guerrier, Skaloud, Stebler, Victoria-Feser. Wavelet-variance-based estimation for composite stochastic
  processes. *JASA* 108(503):1021–1030 (2013). doi:10.1080/01621459.2013.799920 (verified)
* **[Johnson, Li & Rabinovic 2007]** Adjusting batch effects in microarray expression data using empirical Bayes methods
  (ComBat). *Biostatistics* 8(1):118–127 (2007). doi:10.1093/biostatistics/kxj037 (verified)
* **[Kusne 2021]** Kusne et al. Physics in the machine: integrating physical knowledge in autonomous phase-mapping. arXiv:2111.07478
  (verified). The original CAMEO paper (Kusne et al., *Nat. Commun.* 2020) **[unverified]**.
* **[Laguna et al. 1999]** Laguna, Moody, García, Goldberger, Mark. Analysis of the ST-T complex of the electrocardiogram using the
  Karhunen-Loève transform: adaptive monitoring and alternans detection. *Med. Biol. Eng. Comput.* 37(2):175–189 (1999).
  doi:10.1007/BF02513285 (verified)
* **[Lin et al. 2025]** Lin et al. Optimizing charge-coupled device readout enabled by the floating-gate amplifier. arXiv:2503.07930
  (verified)
* **[Liu, Tanaka & Okutomi 2013]** Single-image noise level estimation for blind denoising. *IEEE Trans. Image Process.* (2013).
  **[unverified]**
* **[MacKenzie et al. 2002]** Estimating site occupancy rates when detection probabilities are less than one. *Ecology*
  83(8):2248–2255 (2002). (verified)
* **[Manohar et al. 2018]** Manohar, Brunton, Kutz. Data-driven sparse sensor placement for reconstruction. *IEEE Control Syst.
  Mag.* 38:63–86 (2018). arXiv:1701.07569 (verified)
* **[Otwinowski et al. 2018]** Otwinowski, McCandlish, Plotkin. Inferring the shape of global epistasis. *PNAS*
  115(32):E7550–E7558 (2018). doi:10.1073/pnas.1804015115 (verified)
* **[Ramakrishnan et al. 2015]** Big data meets quantum chemistry approximations: the Δ-machine learning approach. arXiv:1503.04987
  (verified)
* **[Roll 1984]** Roll. A simple implicit measure of the effective bid-ask spread in an efficient market. *J. Finance* 39(4)
  (1984). **[unverified]**
* **[Sens-Schönfelder & Wegler 2006]** Passive image interferometry and seasonal variations of seismic velocities at Merapi
  Volcano, Indonesia. *Geophys. Res. Lett.* 33 (2006). doi:10.1029/2006GL027797 (verified; whether this paper itself uses the
  stretching method was not confirmed, but stretching (dt/t) is the standard technique of the field)
* **[Severson 2019]** (catalogue).
* **[Smaragdis & Brown 2003]** Non-negative matrix factorization for polyphonic music transcription. IEEE WASPAA 2003, pp. 177–180.
  (verified: MERL TR2003-139)
* **[Taylor 1961]** Taylor. Aggregation, variance and the mean. *Nature* 189:732–735 (1961). doi:10.1038/189732a0 (verified)
* **[Tobin 1958]** Tobin. Estimation of relationships for limited dependent variables. *Econometrica* 26 (1958). **[unverified]**
* **[Tynes et al. 2021]** Tynes et al. Pairwise difference regression: a machine learning meta-algorithm for improved prediction and
  uncertainty quantification in chemical search. *J. Chem. Inf. Model.* (2021). (verified: ACS figshare / OSTI)
* **[Watson & Pelli 1983]** QUEST: a Bayesian adaptive psychometric method. *Percept. Psychophys.* 33(2):113–120 (1983). (verified)
* **[Westgard et al. 1981]** Westgard, Barry, Hunt, Groth. A multi-rule Shewhart chart for quality control in clinical chemistry.
  *Clin. Chem.* 27:493–501 (1981). (verified: secondary citations)
* **[Yeh et al. 2018]** Towards a near universal time series data mining tool: introducing the matrix profile. arXiv:1811.03064
  (verified; the original Matrix Profile I paper, ICDM 2016, **[unverified]**)
* **[Yoon et al. 2015]** Yoon, O'Reilly, Bergen, Beroza. Earthquake detection through computationally efficient similarity search.
  *Sci. Adv.* 1(11):e1501057 (2015). doi:10.1126/sciadv.1501057 (verified)
* **[Zhang, Mykland & Aït-Sahalia 2005]** A tale of two time scales: determining integrated volatility with noisy high-frequency
  data. *JASA* 100(472):1394–1411 (2005). doi:10.1198/016214505000000169 (verified)

Project sources used: `lvl1/real_image_checks/2026-10-16_PA03_PQ04_16rows.md` (drift tables and acquisition tests),
`docs/context/2026-10-12_session_context.md` (fingerprints, ch2 noise), `docs/context/2026-10-16_session_context.md`,
`docs/context/2026-10-19_session_context.md`, `src/trace_features.py` (feature definitions), `src/trace_regions.py` (cycle
structure), `docs/slides/data/oct12_trace_features.csv`.
