# Feature study: how informative, stable and redundant is every trace feature?

`lvl1/feature_study.py`, on the 59 usable board-101 traces of the 2026-10-12 packet tests: 5 settings × 3 repeats ×
4 amplifiers, one damaged ch3 trace left out. It covers all 141 scalar features: the 85 original ones, and 56 added on
2026-10-21 in `src/trace_features_extra.py` (49 from the literature search, 5 in the `clock:` group, 2 for the
pre-pulse dip). The data come from `docs/slides/oct12_features.py` and are in
`docs/slides/data/oct12_trace_features.csv`. Outputs: [`feature_table.csv`](feature_table.csv) (every number per
feature) and [`evidence_tiers.json`](evidence_tiers.json).

## Measures

| measure | meaning |
|---|---|
| settings F | ANOVA F over the 4 changed settings (all NCOLPRE 185), per amplifier ch0/1/3; min and median. F > ~4 means p < 0.05 |
| amplifier F | ANOVA F over the 4 amplifiers within each setting; median |
| repeat share | within-(setting, amplifier) variance ÷ total variance: 0 = perfectly repeatable, about 1 = pure repeat noise |
| cluster | features with \|Spearman ρ\| > 0.95 across the 59 traces form one cluster; its most informative member represents it |

**Evidence tiers.** These are for the models, and are not the implementation tiers 0/1/2:

- **A, core:** informative (settings F min > 10, or amplifier F median > 10), repeat share < 0.5, at most 20 % NaN, and
  the representative of its cluster.
- **B, supporting:** informative but redundant, or moderately informative (F 4–10).
- **C, store only:** not informative on this data, mostly NaN, or constant.

**Caveat.** These four settings did not change the noise. Noise features can therefore show only amplifier separation
here. Revisit the tiers on session A, where delayInt changes the noise.

## Counts

| implementation tier | A1 | A2 | B | C |
|---|---|---|---|---|
| 0 (original 85) | 27 | 19 | 22 | 17 |
| 1 (new, 29) | 8 | 8 | 4 | 9 |
| 2 (new, 27) | 2 | 9 | 7 | 9 |
| total | 37 | 36 | 33 | 35 |

A is split by informativeness into A1 (top half) and A2. So 27 of the 56 new features reach tier A, and 22 of the
original 85 are redundant or only moderately informative (B).

## The new features (2026-10-21)

**The start-of-pixel group is the strongest addition.** It separates settings, which almost no noise feature does here,
and the amplifiers too:

| feature | settings F min | amplifier F | repeat share | tier |
|---|---|---|---|---|
| start: transient amplitude, matched filter | 264 | 70 | 0.02 | A |
| start: transient amplitude | 171 | 42 | 0.04 | B (same cluster as the one above) |
| start: in-cycle slope at skip 0 | 58 | 19 | 0.07 | A |
| start: first-skip excess per volt H | 12 | 87 | 0.04 | A |
| start: CDS-slope lever arm | 1.0 | 14 | 0.20 | A |
| start: slope-explained fraction | 2.4 | 12 | 0.17 | A |

The transient amplitude rises from about 615 ADU to 741 ADU (+21 %) only when all H low rails are lowered: the
H-clock feedthrough of the pilot.

**Telegraph switching and non-Gaussian noise single out ch2.** Medians per amplifier:

| | ch0 | ch1 | ch2 | ch3 |
|---|---|---|---|---|
| discrete: in-window step rate (%) | 0 | 0 | **8.2** | 0 |
| discrete: null-CDS kurtosis | −0.34 | −0.20 | **5.2** | −0.11 |
| discrete: jump share of CDS variance | 0.001 | 0.005 | **0.27** | 0.005 |
| noise: null-CDS noise (ADU) | 7.8 | 7.1 | **20.4** | 8.0 |

These, and the other new noise measures in tier A, separate the amplifiers (F 12–700) but not these four settings:
- the null-CDS noise and its lag-1 autocorrelation;
- the structure-function excess and the number of waveform modes for 90 % of the CDS variance;
- the Hadamard deviation, sample entropy, spectral slope and centroid;
- the crest factor, whitened kurtosis, Lee–Mykland jump count, STA/LTA and permutation entropy.

**Unstable for now (tier C).** These vary between repeats almost as much as overall (repeat share 0.6–1.3):
- the pickup-line features (count, frequency, phase coherence), because the pickup itself changes from trace to trace;
- the start-transient modes (frequency, damping, fast decay; fast decay is 80 % NaN);
- the transient time constant;
- the 1/f and Lorentzian corners and exponent, and volatility clustering.

`discrete: in-window step amplitude` is NaN wherever there are no steps (58 %), so it is C by the NaN rule, although
it separates the amplifiers well (F 186).

## The clocking regions (`clock:`, added 2026-10-21 evening)

From the user's look at the trace: the first V-clock block climbs, the second is flat; the H and V transfers differ;
the first H block after coming off a row looks different. Pilot: [`clock_pilot/`](clock_pilot/).

| feature | settings F min | amplifier F | repeat share | tier | per setting (median) |
|---|---|---|---|---|---|
| clock: H/V peak-to-peak ratio | 879 | 1093 | 0.00 | A1 | 2.7 with V rails low, 3.3 with H rails low, else 3.0–3.1 |
| clock: V-transfer peak-to-peak (ADU) | 544 | 1240 | 0.00 | A1 | 46.1 k with V rails low, else 41.8–42.3 k |
| clock: first H block of a row, excess (ADU) | 172 | 173 | 0.01 | A1 | 3.0 k with H rails low, else 1.9–2.2 k |
| clock: V-block level slope (ADU/transfer) | 45 | 65 | 0.03 | A1 | −173 with V rails low, else −113 to −123 |
| clock: leading V-block level slope (ADU/transfer) | 1.5 | 8.4 | 0.30 | B | not repeatable |

None is in a redundancy cluster (closest existing |ρ| 0.71–0.84). The leading V block (the one the trace starts in)
does climb by 300–1,200 ADU per transfer, but the trace starts at a random point in it (14–19 of 20 transfers
recorded) and what came before is not controlled, so it is stored as tier 2 only. Dropped: the H − V level difference
(flips sign with NCOLPRE) and the dip at the second H transfer (|ρ| 0.98 with `fit: baseline droop amplitude`).

## The dip before the per-cycle pulse (`edge: pre-pulse dip …`, added 2026-10-21 evening)

Pilot: [`prepulse_pilot/`](prepulse_pilot/). From the equivalent-time pulse: a sharp dip 120–130 ns before the pulse
edge, about 1.5 samples wide. Depth ch0 −886, ch1 −505, ch2 −990, ch3 −601 ADU; repeat share 0.001; amplifier F 5851;
settings F 0.2 (no setting touched RG/SW/OG/DG). `edge: pre-pulse dip depth (ADU)` is A1; the dip/pulse ratio is B
(same cluster). The clock edge behind it is not identified yet: the session-B RG_H, RG_L and SW_L packets will show it.

## Caveat: the label-free tiers can demote the best noise estimators

The evidence tiers reward separation of settings and amplifiers, and choose cluster representatives by that. On these
59 traces the amplifier dominates every correlation, so "redundant" means "identifies the same amplifier":

- `skip: CDS rms (detrended)`, the most direct σe estimator, is in B; its cluster (with `fit: white noise per skip`,
  `noise: jump-robust CDS sigma`) is represented by `noise: waveform modes for 90 % of the CDS variance`;
- `noise: reset-level scatter (ADU)` (kTC noise; the top learned feature on synthetic data) is in B (amplifier F 7);
- `fit: dip recovery time` is in B, represented by `noise: within-cycle power 0.3-1 MHz`.

A model limited to tier A does not see them.

**Guard (a), adopted 2026-10-21:** `PROTECTED` in `feature_study.py`, written as tier `P` in `evidence_tiers.json` and
included in every model's default set (`MODEL_TIERS` in `lvl1/train_lvl1.py`): CDS rms (detrended), white noise per
skip, jump-robust sigma, null-CDS noise, Allan deviation n=50 (σe); reset-level scatter, pulse per volt RG, dip per
volt SW (gain).

**Prototyped, not adopted** (scratchpad test: synthetic td41_s0/1/2 plus the 120 other real features as uninformative
real-named columns; error %, mean of 3, σe / λ / gain):

| set | gp | nn | ridge |
|---|---|---|---|
| A (old default) | 57.5 / 29.2 / 31.3 (collapsed) | 32.9 / 22.4 / 17.6 | 16.8 / 19.9 / 4.4 |
| guards a–c (P + tiers with within-amplifier clusters, no repeat penalty) | collapsed | 21.7 / 21.9 / 10.2 | 14.7 / 23.2 / 3.3 |
| P alone | 9.6 / 22.0 / 0.9 | 12.9 / 18.6 / 6.3 | 12.2 / 17.0 / 6.3 |
| P + label-driven set (from other datasets) | 9.0 / 14.5 / 0.9 | 10.2 / 16.9 / 2.6 | 9.5 / 14.0 / 0.9 |
| the 17 true synthetic features | 6.9 / 8.0 / 0.9 | 10.1 / 17.1 / 1.5 | 9.5 / 12.4 / 0.9 |

Guards (b) within-amplifier clusters and (c) no repeat penalty helped less and enlarge tier A; (d) label-driven selection
is deliberately not used to choose features (docs/trace_features_and_tiers.md §6.1): instead every model is fed the
features in label-free orders (`--feature-ladder`, §6.2).

## What next

- Step 2: the synthetic tolerance test, i.e. how many extra columns each model can take.
- After session A: rerun this study on real packets, where the noise changes, then choose the feature sets per model
  with the setting-grouped CV and the session-C hold-out.

## Step 2: how many extra feature columns can each model take? (`lvl1/feature_tolerance.py`, synthetic)

**Setup.** Three synthetic 41-packet datasets; the toy world has 17 trace features. 20, 50 or 100 extra columns were
added to both the own-trace and the reference features, of three kinds:

- noise: independent Gaussian columns;
- copies: near-copies of existing features;
- real: real 2026-10-12 feature values drawn at random, i.e. realistic distributions and outliers, but no information.

All models were run with `--quick`. Files: [`tolerance_results.csv`](tolerance_results.csv),
[`tolerance_summary.csv`](tolerance_summary.csv).

**Error with no extra columns** (σe / λ / gain %):

| model | σe / λ / gain % |
|---|---|
| gp | 6.9 / 8.0 / 0.9 |
| physics | 9.1 / 7.4 / 0.9 |
| physics_joint | 9.1 / 4.6 / 0.9 |
| sets | 5.7 / 10.9 / 0.9 |
| ridge | 9.5 / 12.4 / 0.9 |
| trees | 9.7 / 21.0 / 1.3 |
| nn | 10.0 / 17.0 / 1.5 |
| cnn | 9.5 / 14.5 / 3.3 |

**Change in error with extra columns** (percentage points, mean of 3 datasets; "noise" and "real" columns behave almost
the same):

| model | σe at +20 / +50 / +100 | λ at +20 / +50 / +100 | gain at +100 | verdict |
|---|---|---|---|---|
| gp (plain ARD, maximum likelihood) | **+51 / +51 / +51** | **+21 / +21 / +21** | **+30** | collapses to about "guess the average" with only 20 uninformative columns |
| nn | +2 / +7 / +15 | +3 / +5 / +8 | +11 | degrades steadily |
| ridge | +2 / +4 / +5 | +6 / +8 / +14 | +3 | degrades, mostly for λ |
| cnn | 0 / +1 / +4 | +1 / +3 / +5 | +1 | mild (it sees the extras only as scalars) |
| physics (λ by a SAAS GP) | 0 / 0 / 0 | 0 / +2.5 / +7 | 0 | σe and gain immune (they come from the trace physics); λ degrades at 100 |
| physics_joint | 0 / 0 / 0 | +1 / +2 / +3 | 0 | robust |
| sets | 0 / 0 / 0 | +2.5 / +3.5 / +4 | 0 | robust |
| trees | 0 / 0 / 0 | 0 / 0 / 0 | 0 | immune |

Near-copies hurt much less than uninformative columns, except for the λ of the plain gp (+6 to +11) and of physics
(+3 to +5).

**What this means for the real data.**

*(The recommendations below are from 2026-10-21; the current defaults, after the protected tier P and the feature
ladder, are in `MODEL_TIERS` of `lvl1/train_lvl1.py`: gp and gp_saas P, gp_dim P+A1, nn/ridge P+A, trees all, the rest
P+A+B.)*

- **The plain `gp` must not get all features.** On synthetic data it sees 17 + 17 trace features; on real data it would
  see 141 + 141 and, by this test, collapse. Give it evidence tier A only, or use the prior-regularized variants
  (`gp_saas`, `gp_dim`) and test their tolerance the same way.
- **`nn` and `ridge`:** tier A, perhaps A + B; not tier C.
- **`trees`, `sets`, `physics_joint`, `cnn`:** can take A + B, and trees even everything.
- **`physics`:** σe and gain are safe. Its λ GP should get tier A only, or switch to the joint λ, which is more robust.

The per-model choice is to be confirmed on real data after session A, with the setting-grouped CV and the session-C
hold-out. **Not done yet:** the `--feature-tiers` option in `train_lvl1.py`, and the tolerance test of `gp_saas` and
`gp_dim`.

## Step 3: can a trained model learn the tiers? (`lvl1/feature_importance.py`, synthetic check)

**Method.** Two independent learned importances per target (log σe, log λ, log gain), always with packet-grouped
resampling:

- stability selection: how often an elastic net, refitted on packet-bootstrap resamples, keeps the feature;
- permutation importance: the increase of the held-out error of gradient-boosted trees when the feature is shuffled.

Ranking features against each other does not work at 41 packets: real and pure-noise features had the same importances.
So every feature is compared with **shadow features** (a row-shuffled copy of every column; Boruta). It must beat the
shadows' 95th percentile in most of 5 rounds with fresh shadows.

**Learned tiers.**

- **L1:** both methods confirm the feature, each by more than 1.5 × the shadows' 95th percentile.
- **L2, candidate:** one method only, or both but narrowly.
- **L3:** not shown to matter.

**Check on synthetic data.** Three 41-packet datasets (td41_s0/1/2), with 20 pure-noise columns added (the "current"
row is the 2026-10-22 rerun with packet-grouped folds for the elastic net's alpha):

| rule | noise columns in L1 | noise columns in L2 | real features in L1 |
|---|---|---|---|
| L1 = both methods win (no margin) | 0 / 0 / 2 | 6 / 6 / 3 | 5 / 4 / 8 |
| L1 = both win by > 1.5 × (current) | **0 / 0 / 0** | 6 / 5 / 5 | 5 / 4 / 8 |

The features in L1 in all three datasets: reset-level scatter, detrended CDS rms, dip recovery time, horizontal minimum or maximum.

**What this means.**

- L2 is not trustworthy: 25–30 % of the noise columns reach it. Use it as a list to look at.
- The 1.5 margin was chosen on these three runs. On two unseen datasets: td100_s0 has 0 of 20 noise columns in L1
  (3 in L2) and 3 real features in L1; **td13_s0 (13 packets) has 1 of 20 in L1** (6 in L2) and 6 real features in L1.
  So L1 is reliable at 41–100 packets, but at the size of session A (~14 packets) about 1 noise column in 20 can reach it.
- The learned tiers are diagnostics only (docs/trace_features_and_tiers.md §6.1): features in L1 but not in tier A, and
  tier-A features in L3, are the ones to look at physically; they never choose a model's features.
