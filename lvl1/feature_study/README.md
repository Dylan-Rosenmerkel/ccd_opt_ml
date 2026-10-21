# Feature study: how informative, stable and redundant is every trace feature?

`lvl1/feature_study.py`, on the 59 usable board-101 traces of the 2026-10-12 packet tests: 5 settings × 3 repeats ×
4 amplifiers, one damaged ch3 trace left out. It covers all 134 scalar features: the 85 original ones, and 49 added on
2026-10-21 in `src/trace_features_extra.py`. The data come from `docs/slides/oct12_features.py` and are in
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

| implementation tier | A | B | C |
|---|---|---|---|
| 0 (original 85) | 46 | 22 | 17 |
| 1 (new) | 11 | 3 | 9 |
| 2 (new) | 11 | 6 | 9 |

So 22 of the 49 new features reach tier A, and 22 of the original 85 are redundant or only moderately informative (B).

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

## What next

- Step 2: the synthetic tolerance test, i.e. how many extra columns each model can take.
- After session A: rerun this study on real packets, where the noise changes, then choose the feature sets per model
  with the setting-grouped CV and the session-C hold-out.
