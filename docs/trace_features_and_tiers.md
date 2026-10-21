# New trace features, the tiered structure, and the plan for the first data

2026-10-21. This document gathers in one place:

- the 56 trace features and 4 curves added today (49 from the literature search, 5 in the `clock:` group, 2 for the
  pre-pulse dip);
- how features are now organised in tiers;
- what the tests showed;
- the packet plan ("the scan");
- what to do with the tiers once the first real packets exist.

It builds on [`trace_feature_candidates.md`](trace_feature_candidates.md) (the candidate list),
[`literature/signal_analysis_features.md`](literature/signal_analysis_features.md) (where the ideas come from) and
[`../lvl1/feature_study/README.md`](../lvl1/feature_study/README.md) (the full test results).

---

## 1. Summary

- **Every trace now gives 141 numbers and 18 curves**, where it gave 85 and 14. The new ones are in
  `src/trace_features_extra.py`; 16 tests cover them in `tests/test_trace_features_new.py`; a trace takes about 1.2 s.
  Processing (`process_packet.py`) writes them into `model_inputs/` without any further change; this was checked on
  simulated packets.
- **Every feature has two labels:**
  - an *implementation tier* (0 / 1 / 2): where it came from;
  - an *evidence tier* (A / B / C): what the data say about it.
- **Each model gets the evidence tiers it can cope with.** A synthetic test showed how many uninformative inputs each
  model tolerates; the plain Gaussian process tolerates almost none.
- **The most interesting results:**
  1. The new **start-of-pixel features** separate the *settings*, which almost nothing about the noise does on the
     present data. The best one, a matched-filter amplitude of the start transient, has a settings F of 264 and a
     repeat scatter of only 2 % of its variance. The start transient is the CDS reading a decaying baseline; it is
     horizontal-clock feedthrough and grows about 21 % when the H low rails are lowered.
  2. The new **telegraph-switching features single out ch2**: steps in 8.2 % of its integration windows against 0 % on
     the other amplifiers, null-CDS kurtosis 5.2 against about 0, and 27 % of its CDS variance in jumps.
  3. The **noise budget** of the skip cycle: the three largest waveform fluctuations carry 95 % of the waveform's
     variation but under 1 % of the CDS noise. The CDS noise is spread over about 100 small modes.
  4. **The plain Gaussian process collapses** when 20 uninformative features are added: σe error +51 percentage
     points, which is about "guess the average". On real data it must get only the core features.
  5. **Nothing about the noise separates the four voltage settings tested on 2026-10-12**, because those settings did
     not change the noise. The noise features can only be judged on session A, where delayInt changes the noise.

---

## 2. The new features

There are 49 scalars and 4 curves, in six families. The evidence tier and the key numbers come from the 59 real traces
of 2026-10-12 (§4).

**Settings F** = how strongly the feature separates the four changed settings, and **amplifier F** = how strongly it
separates the four amplifiers. Both are ANOVA F values, against the scatter between repeats; F > 4 is significant
(p < 0.05). **Repeat share** = the fraction of the feature's variance that is just repeat-to-repeat scatter.

### 2.1 Start of pixel (`start:`, a new group)

The first skips of every pixel carry a transient, about +290 ADU at skip 0 on board 101. It is the CDS reading the
slope of a decaying baseline (lever arm ≈ −212 samples, close to the 207-sample spacing of the window centres;
R² 0.97). Inside skip 0 it is an exponential with τ ≈ 12 µs, and it scales with the horizontal-clock swing.

| feature | impl. | evidence | settings F | amplifier F | repeat share |
|---|---|---|---|---|---|
| start: transient amplitude, matched filter (ADU) | 2 | **A** | **264** | 70 | 0.02 |
| start: transient amplitude (ADU) | 1 | B (same cluster as the one above) | 171 | 42 | 0.04 |
| start: in-cycle slope at skip 0 (ADU/µs) | 1 | **A** | 58 | 19 | 0.07 |
| start: first-skip excess per volt H (ADU/V) | 1 | **A** | 12 | 87 | 0.04 |
| start: slope-explained fraction | 1 | A | 2.4 | 12 | 0.17 |
| start: CDS-slope lever arm (samples) | 1 | A | 1.0 | 14 | 0.20 |
| start: reset jump skip 0 to 1 (ADU) | 2 | A | 0.7 | 26 | 0.14 |
| start: row-start/inner first-skip ratio | 2 | B | 4.1 | 1.7 | 0.02 |
| level: slope-corrected CDS rise (ADU) | 1 | B | 0.1 | 6.8 | 0.27 |
| start: transient time constant (µs) | 1 | C (unstable) | 1.9 | 2.1 | 0.58 |
| start: ringing frequency, mode frequency, mode damping, fast decay | 2 | C (unstable, or mostly NaN) | <1 | <1 | ≥ 1 |
| curve: start: mean transient (skips 0–29) | 1 | — | | | |

### 2.2 Telegraph switching inside one window

ch2's excess noise sits inside single 10 µs integration windows, with no charge transfer involved. Steps of ±40–110 ADU
occur equally up and down: the signature of random telegraph switching in the output stage.

| feature | impl. | evidence | amplifier F | ch0 / ch1 / **ch2** / ch3 |
|---|---|---|---|---|
| noise: null-CDS noise (ADU): half window minus half window, no transfer | 1 | A | 225 | 7.8 / 7.1 / **20.4** / 8.0 |
| discrete: null-CDS kurtosis | 1 | A | 49 | −0.3 / −0.2 / **5.2** / −0.1 |
| discrete: in-window step rate (%) | 1 | A | 73 | 0 / 0 / **8.2** / 0 |
| discrete: in-window step amplitude (ADU) | 1 | C (NaN where there are no steps) | 186 | — / — / ~52 / — |
| noise: null-CDS autocorrelation at lag 1 | 2 | B | 164 | |
| noise: structure-function excess at 9 blocks (ADU) | 2 | A | 327 | |
| discrete: max within-window spectral kurtosis | 2 | B | 5.7 | |
| curve: noise: within-cycle structure function | 2 | — | | |

### 2.3 Jump-robust noise (from high-frequency finance)

| feature | impl. | evidence | amplifier F | notes |
|---|---|---|---|---|
| discrete: jump share of CDS variance (1 − MedRV/RV) | 1 | A | 32 | ch2 0.27, the others ≤ 0.005 |
| noise: jump-robust CDS sigma (ADU) | 1 | B (redundant with the CDS rms) | 255 | |
| discrete: jump count (Lee–Mykland, 1 %) | 2 | A | 33 | |
| discrete: crest factor | 2 | A | 56 | |
| discrete: whitened CDS kurtosis | 2 | A | 23 | |
| discrete: STA/LTA maximum (seismology) | 2 | A | 14 | |
| noise: volatility clustering (Ljung–Box of e²) | 2 | C | 2.5 | |

### 2.4 Noise shape

| feature | impl. | evidence | amplifier F | notes |
|---|---|---|---|---|
| noise: Hadamard deviation n=50 (ADU) + curve | 1 | A | 41 | drift-insensitive averaging noise |
| noise: sample entropy | 2 | A | 37 | |
| noise: CDS spectral centroid | 2 | A | 63 | |
| noise: CDS spectral slope | 2 | A | 23 | |
| noise: permutation entropy (m=4) | 2 | A | 13 | |
| noise: red-noise β (50 skips) | 2 | B | 7.0 | |
| noise: lag-1 autocorrelation of 10-skip means | 2 | B | 7.4 | |
| noise: DFA exponent | 2 | C | 2.8 | carries mostly new information but is unstable here |
| noise: 1/f corner frequency, 1/f exponent, Lorentzian corner | 1, 1, 2 | C | 1.5–2.8 | there is no 1/f noise in the band of a 0.2 s trace; delayInt should change that |
| curve: noise: modified Allan deviation | 2 | — | | |

### 2.5 Pickup lines

| feature | impl. | evidence | notes |
|---|---|---|---|
| discrete: strongest line amplitude (ADU) | 1 | A | |
| discrete: line count (multitaper F-test, p < 0.001), strongest line frequency, line phase coherence (pixel clock / absolute time), line frequency over the whole trace | 1, 2 | C | **unstable between repeats** (repeat share 0.4–1.1): the pickup changes from trace to trace |

### 2.6 Noise budget and levels

| feature | impl. | evidence | notes |
|---|---|---|---|
| noise: waveform modes for 90 % of the CDS variance | 1 | **A** (amplifier F 700) | ch0 ≈ 106 modes |
| noise: CDS variance share of the 3 largest waveform modes | 1 | C | about 1 % everywhere: the big clock and reset fluctuations do not reach the CDS |
| level: common-pattern amplitude spread | 2 | B | |
| noise: baseline–CDS residual correlation | 2 | C | |

**Not implemented** (in the candidate list, left out for now):

- RTS dwell times from a hidden Markov model;
- RTS step rise time;
- jitter decomposition into random and deterministic parts;
- in-window drift against the baseline;
- the exponentially-modified-Gaussian pulse fit, spectral flux and cepstrum.

---

### 2.7 Clocking regions (`clock:`, from the user's look at the trace)

| feature | tier | evidence | settings F | what it shows |
|---|---|---|---|---|
| clock: H/V peak-to-peak ratio | 1 | A1 | 879 | H- relative to V-clock feedthrough |
| clock: V-transfer peak-to-peak (ADU) | 1 | A1 | 544 | V-clock feedthrough; larger with the V rails low |
| clock: first H block of a row, excess (ADU) | 1 | A1 | 172 | memory of the V clocking in the first H transfers; largest with the H rails low |
| clock: V-block level slope (ADU/transfer) | 1 | A1 | 45 | output-level relaxation in the V block before row 2; steeper with the V rails low |
| clock: leading V-block level slope (ADU/transfer) | 2 | B | 1.5 | the climb in the block the trace starts in: real, but not repeatable (random start point) |

Details and figure: `lvl1/feature_study/README.md` and `lvl1/feature_study/clock_pilot/`.

### 2.8 The dip before the per-cycle pulse

`edge: pre-pulse dip depth (ADU)` (tier 1, A1) and `edge: pre-pulse dip / pulse height` (tier 1, B): a sharp dip
120–130 ns before the pulse edge, −505 to −990 ADU by amplifier, repeat share 0.001, unchanged by these settings. Its
clock edge (SW, RG or OG) is to be identified with the session-B RG/SW packets. Pilot: `lvl1/feature_study/prepulse_pilot/`.

**Caveat on the tiers** (found while adding these): the label-free evidence tiers put the most direct σe estimator
(`skip: CDS rms (detrended)`) and the reset-level scatter in B. See `lvl1/feature_study/README.md` for the proposed
guards.

## 3. The tiered structure of the data and the learning

Every trace is reduced to features once, in processing, and **all of them are stored**. What changes is which features
each model *sees*. Three layers decide that:

```
raw trace ──► 141 scalars + 18 curves, all stored (process_packet.py → dataset.csv, model_inputs/)
                 │
   layer 1  implementation tier   0 / 1 / 2        where the feature came from (fixed)
                 │
   layer 2  evidence tier          P + A / B / C    P: protected physical estimators (fixed by physics);
                                                    A/B/C: what the data say, without labels (revised as data arrive)
                 │
   layer 3  model feature set      per model        which evidence tiers a model can cope with
```

**Layer 1: implementation tier.** It is stored as `FEATURES[name].tier`; `trace_features.names_by_tier(...)` lists the
features of each tier.

| tier | what | count |
|---|---|---|
| 0 | the original features | 85 scalars + 14 curves |
| 1 | implement-first additions (physics-motivated, piloted; incl. `clock:` and the pre-pulse dip) | 29 scalars + 2 curves |
| 2 | candidate additions | 27 scalars + 2 curves |

**Layer 2: evidence tier.** It comes from `lvl1/feature_study.py` and is stored in `lvl1/feature_study/evidence_tiers.json`.

| tier | rule | original | new | total |
|---|---|---|---|---|
| **P**, protected (guard a) | direct physical estimators of the labels, chosen by physics, given to **every** model: σe — CDS rms (detrended), white noise per skip, jump-robust sigma, null-CDS noise, Allan deviation n=50; gain — reset-level scatter (kTC), pulse per volt RG, dip per volt SW. No direct λ estimator exists yet | 6 | 2 | **8** (also in A/B) |
| **A**, core | informative (settings F min > 10 or amplifier F > 10), repeat share < 0.5, at most 20 % NaN, and the most informative member of its redundancy cluster (\|ρ\| > 0.95) | 46 | 27 | **73** |
| **B**, supporting | informative but redundant, or moderately informative (F 4–10) | 22 | 11 | **33** |
| **C**, store only | not informative on the data so far, unstable, mostly NaN, or constant | 17 | 18 | **35** |

**Why P exists.** The evidence tiers reward *separating* the traces, not *predicting* σe, λ or gain, and on the 59
traces the amplifier dominates every correlation, so "redundant" means "identifies the same amplifier". Without P the
most direct σe estimator (`skip: CDS rms (detrended)`) and the kTC noise (`noise: reset-level scatter`) sit in B, and a
tier-A model would not see them. In a synthetic test (lvl1/feature_study/README.md) these two features alone took the
plain GP from 57.5 % to 9.6 % (σe) and from 31 % to 0.9 % (gain). Two further guards were prototyped (clusters within each
amplifier; no repeat-share penalty) and are **not adopted**: they helped less and enlarge tier A.

**Balanced sizes (2026-10-21).** A was split by informativeness into **A1** (the more informative half) and **A2**,
so the four tiers hold similar numbers: A1 37, A2 36, B 33, C 35.

The evidence tiers are *provisional*. They rest on 59 traces at 5 settings whose voltage changes did not move the
noise, so the noise features have been judged by amplifier separation alone. They will be revised on real packets (§6).

**Layer 3: model feature sets.** These come from the synthetic tolerance test (`lvl1/feature_tolerance.py`). The test
added 20, 50 or 100 uninformative columns to 3 synthetic 41-packet datasets and measured the change in error, in
percentage points:

| model | σe at +20 / +50 / +100 | λ at +20 / +50 / +100 | suggested feature set |
|---|---|---|---|
| gp (plain) | **+51 / +51 / +51** | +21 / +21 / +21 | **P only** (the ladder: collapse beyond 15–25 features); `gp_saas`, `gp_dim`: P + A1 until their tolerance is tested |
| nn | +2 / +7 / +15 | +3 / +5 / +8 | P + A |
| ridge | +2 / +4 / +5 | +6 / +8 / +14 | P + A |
| cnn | 0 / +1 / +4 | +1 / +3 / +5 | P + A + B |
| physics | 0 / 0 / 0 | 0 / +2.5 / +7 | P + A + B for σe and gain |
| physics_joint | 0 / 0 / 0 | +1 / +2 / +3 | P + A + B |
| sets | 0 / 0 / 0 | +2.5 / +3.5 / +4 | P + A + B |
| trees | 0 / 0 / 0 | 0 / 0 / 0 | everything |

These sets are wired in (`MODEL_TIERS` in `lvl1/train_lvl1.py`): `--feature-tiers auto` uses them, `--feature-tiers
P+A+B` (or any combination of P, A1, A2, A, B, C, impl0, impl1, impl2) gives every model the same set, `--tier-ablation`
trains every model with several sets, and `--feature-ladder` feeds every model more and more features (§6.2). They are
**starting points**; the ladder on real data decides.

---

## 4. The tests behind the tiers (the 59 real traces of 2026-10-12)

**Data.** Board 101: 5 settings (reference, VDD −20.5 V, V low rails + TG, all H low rails, H3_A_L only) × 3 repeats ×
4 amplifiers. One damaged ch3 trace is left out (as processing would), leaving 59 traces. The settings separation uses
only the 4 changed settings, because the reference was taken with a different NCOLPRE.

**What separates the settings.** The waveform features do so enormously: the horizontal- and vertical-transfer minima
have F ≈ 10⁵, since each changed clock shows up in the part of the trace it drives. The new start-transient features
do too (F 12–264). **No** noise statistic does, old or new, because these four settings did not change the noise.

**What separates the amplifiers.** Most noise statistics do, old and new. The telegraph features pick out ch2 cleanly.

**What is redundant.** RMSSD/SD and the Poincaré SD1/SD2 are exact functions of the lag-1 autocorrelation; they were
not implemented. Among the implemented features, the redundancy clusters merge, for example:
- the two transient amplitudes;
- the jump-robust sigma with the CDS rms.

**What is unstable.** These vary between repeats almost as much as overall:
- the pickup-line features: the pickup itself changes;
- the start-transient modes and time constant;
- the 1/f corners.

A short [notebook](../lvl1/new_trace_features.ipynb) walks through each new family on real traces, with plots:
- the differentiator plot of the first-skip offset;
- the null-CDS distributions of ch0 and ch2;
- a ch2 window with a telegraph step;
- the Allan, Hadamard and modified Allan curves;
- the multitaper F spectrum;
- the noise budget;
- every feature placed by its settings F against its amplifier F;
- the tolerance results.

---

## 5. The scan: the 43-packet plan (`lvl1/packet_plan/`, `lvl1/make_packet_plan.py`)

| session | packets | what changes | why it matters for the features |
|---|---|---|---|
| A | 14 (~3.8 h) | timing only. delayInt in a **shuffled** order (1500, 300, 3000, 500, 2000, 700), with the delay_H and delayRGH packets between them; nominal packets at the start, middle and end; delayInt 300 **repeated** late in the session | **delayInt changes the noise directly.** This is the first data where the noise features can separate settings, and where every feature can be compared with the image labels |
| B | 18 (~4.9 h) | one voltage group at a time, one max_step either way; a group's two signs **far apart in time**; delayInt 3000 **repeated** in the middle | voltage effects; the replicate measures repeat scatter and drift away from nominal |
| C | 11 | joint changes (Latin hypercube) and a final nominal | kept as the **final, untouched test** (`train_lvl1.py --holdout-session C`) |

**Why the order matters.** A monotone delayInt scan would mix any slow drift during the session with the strongest
knob. Shuffling, and nominal packets in the middle, keep the two apart.

**Why the replicates matter.** The repeat share in §3 needs repeats at *non-nominal* settings too. The two replicates
give exactly that, and also a second measure of drift.

Two short DAQ trace tests are still worth doing before the trace settings are fixed:
- NDCMPRE = 1 and 2: the start-transient analysis predicts the first kept skip at about −50 and −125 ADU;
- a trace with charged pixels: does the previous pixel's charge change the next pixel's start transient?

---

## 6. Plan for the first real data

### 6.1 Finalize the tiers (without letting the labels choose the features)

After session A, rerun `lvl1/feature_study.py` on the real packets. The settings F over the delayInt packets is the
first test where noise features have something to separate. The evidence tiers stay **label-free**.

Label relevance (how strongly a feature follows σe, λ and gain within an amplifier) and the learned tiers (§6.4) are
computed and **reported beside** the tiers, as diagnostics: they say which features to look at physically. They do not
move features between tiers on their own. A feature joins P only with a stated physical mechanism (as the 8 did), not on
a number alone.

**Why not rank the features by the labels.** Ranking ~141 features by how well each follows the labels, on ~30 packets:
- is noisy, so the top of the ranking is partly luck (the winner's curse);
- makes the cross-validated error of the top features optimistic unless the ranking is redone inside every fold;
- favours features that matter for the settings scanned so far (session A: timing), and misses those that matter only
  for later settings or only together with other features;
- lets the labels' own errors (WADERs fits) steer the feature choice.
The synthetic guard test showed a label-driven set *can* nearly reach the true-feature result, but there the ranking
came from sibling datasets of a known world, the optimistic case. On real data it is at most one clearly marked extra set
after session B, ranked inside each fold, and confirmed on C.

### 6.2 Feed every model more and more features (the feature ladder, `--feature-ladder`)

Every model is trained on a nested series of feature sets, on the same setting-grouped folds, and its error is read
against the number of features. **The orders never use the labels**; the labels only score a few fixed, declared sets.
The orders can be frozen before the first labels are seen (pre-registration): the cover order uses the features only.

| order | steps | what it asks |
|---|---|---|
| `tiers` | none (settings only) → P → P+A1 → P+A → P+A+B → all | does each evidence tier help this model? |
| `cover` | P → 15 → 25 → 40 → 60 → 90 → all | the next feature is always the one least explained by those already taken (column-pivoted QR of the standardized features, after P): covers the feature space evenly, without favouring amplifier separation |
| `random` | P → the same sizes, 3 random orders | the control: if `tiers` or `cover` do not beat random orders, the order does not matter for this model |
| `groups` | P + one physics group at a time (start, clock, edge, noise, discrete, …) | which family adds what (not nested) |

**Read-out per model and target:** error % against the number of features (table and `feature_ladder.png`), with
bootstrap intervals; paired differences against the same model on P. **Suggested set:** the smallest step within one
standard error of the model's best step (the one-standard-error rule: do not chase noise).

**Choosing without fooling ourselves:**
- choose on sessions A + B only, confirm once on session C (`--holdout-session C`);
- report every step of every order, not just the winner (Cawley & Talbot 2010);
- the number of steps is fixed in advance (about 6 per order).

**Which models.** All except the baselines and `trace_physics`: gp, gp_dim, gp_saas, nn, ridge (expected to be
sensitive), trees, sets, physics_joint (expected to be robust), the λ GP of physics, and the scalar block of cnn.

    .venv/bin/python lvl1/train_lvl1.py --inputs <model_inputs> --feature-ladder tiers cover random [groups] \
        [--ladder-steps 15 25 40 60 90] [--ladder-random 3] --holdout-session C

**Synthetic test of the ladder** (`lvl1/feature_study/ladder_synthetic/`: td41_s0/1/2 with the 120 other real features
added as uninformative real-named columns; `--quick`; error %, mean of 3, σe / λ / gain):

| model | settings only | P (8) | P+A1 (44) | all (137) | shape of the curve |
|---|---|---|---|---|---|
| gp | 57.5 / 21.5 / 31.3 | **9.6 / 22.0 / 0.9** | collapsed | collapsed | best at 8–15 features, collapses beyond 15–25 in every order |
| nn | 57.5 / 19.3 / 31.4 | **12.9 / 18.6 / 6.3** | 17.2 / 20.5 / 8.0 | 26.6 / 23.5 / 13.8 | gets steadily worse with every step |
| ridge | 57.4 / 17.2 / 31.3 | 12.2 / 17.0 / 6.3 | **10.5 / 18.5 / 1.3** | 14.9 / 21.7 / 4.7 | best around 25–45 features, worse beyond 60 |
| trees | 57.0 / 21.0 / 31.2 | 9.3 / 21.0 / 1.3 | 10.0 / 21.0 / 1.3 | 9.9 / 20.5 / 1.4 | flat |
| physics_joint | 36.0 / 15.7 / 31.3 | **9.3 / 5.0 / 1.0** | 8.8 / 5.8 / 1.0 | 9.0 / 7.5 / 1.7 | flat for σe and gain; λ worse with more features |
| sets | 32.0 / 20.7 / 31.3 | **5.8 / 9.3 / 1.0** | 6.2 / 12.6 / 1.0 | 6.6 / 15.6 / 1.7 | flat for σe and gain; λ clearly worse with more features |

What it shows:
- **P carries almost everything here**, because two of its features (CDS rms, reset-level scatter) are informative in
  the synthetic world; on real data A and B features will carry more than in this all-junk test.
- **gp must stay small** (now `MODEL_TIERS`: P only). Even the "robust" models lose on λ as features are added
  (sets 9.3 → 15.6), so their default A+B is to be checked on real data.
- **The cover order did not beat the random control** here, as expected: without labels no order can know which of
  these columns carry information. On real data, the comparison of `tiers`, `cover` and `random` is the point.
- No order uses the labels; the ladder only measures how each model responds to more features.

### 6.3 What has been built

1. **Done:** `--feature-tiers` (`all`, `none`, `auto` = `MODEL_TIERS`, or tier labels such as `P+A`, `A1`, `impl0`).
2. **Done:** `--tier-ablation [SET ...]` (default `none P P+A1 P+A P+A+B all impl0`), each model paired against its own
   `--ablation-ref` run.
3. **Done (2026-10-21):** guard (a), the protected tier P, in `lvl1/feature_study.py` (`PROTECTED`) and in every
   default set; `--feature-ladder` with the four label-free orders, the one-standard-error suggestion and the plot;
   unit tests in `tests/test_lvl1_validation.py`.
4. **Not started:** label relevance as a *reported* column of `lvl1/feature_study.py` (§6.1).
5. **Not started:** the tolerance test of `gp_saas` and `gp_dim`.

### 6.4 Learned tiers: let trained models say which features matter (`lvl1/feature_importance.py`)

The evidence tiers need no labels. With labels, trained models can rank the features too:

- **stability selection:** how often an elastic net, refitted on packet-bootstrap resamples, keeps the feature;
- **permutation importance:** the increase of the held-out error of gradient-boosted trees when the feature is
  shuffled.

At ~40 packets a plain ranking does not work: on synthetic data, real and pure-noise features had the same
importances. Every feature is therefore compared with **shadow features**, row-shuffled copies of every column
(Boruta). The learned tiers:

- **L1:** both methods beat the shadows' 95th percentile by more than 1.5 ×, in most of 5 rounds;
- **L2:** a candidate (one method, or both narrowly);
- **L3:** not shown to matter.

**Synthetic check** (td41_s0/1/2, 20 pure-noise columns added): no noise column reached L1, but 5–6 of the 20 reached
L2. So L1 can be used as a feature set and L2 only as a list to look at. The 1.5 margin was chosen on these runs and then
confirmed on two unseen datasets (td100_s0, td13_s0): again no noise column in L1. Details: `lvl1/feature_study/README.md`, step 3.

**Use on real data:** after session A, beside §6.1, as a **diagnostic**: features in L1 but not in tier A, and tier-A
features in L3, are the ones to look at physically. Learned importance does not choose any model's features (§6.1).

---

## 7. Files

| file | what |
|---|---|
| `src/trace_features_extra.py` | the 56 new features and 4 curves (registry `EXTRA`, `extra_features`) |
| `src/trace_features.py` | the `tier` attribute and `names_by_tier()`; calls the new block |
| `tests/test_trace_features_new.py` | 16 tests: synthetic signals with known answers, plus the pilot numbers on one real trace |
| `docs/slides/oct12_features.py` → `docs/slides/data/oct12_trace_features.csv` | all 141 features of the 59 real traces |
| `lvl1/feature_study.py` → `lvl1/feature_study/` | separation, stability and redundancy per feature; evidence tiers; results README |
| `lvl1/feature_tolerance.py` → `lvl1/feature_study/tolerance_*.csv` | the synthetic tolerance test |
| `lvl1/feature_importance.py` → `feature_importance.csv` | learned tiers L1/L2/L3 (stability selection + permutation vs shadow features) |
| `lvl1/make_feature_notebook.py` → `lvl1/new_trace_features.ipynb` | the walkthrough notebook |
| `docs/trace_feature_candidates.md` | the candidate list and the reasons |
| `lvl1/packet_plan/PLAN.md` | the 43-packet scan |
