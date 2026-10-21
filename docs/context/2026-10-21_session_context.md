# Session context, 2026-10-21

Continues `2026-10-19_session_context.md` (it also covers 10-20: SAAS defaults, slide deck, model summary PDF).
**If the session reset: read section 3 ("Status") first; it says exactly where the work stopped.**

## 1. Done earlier today (all pushed)

- **Literature, three agents:**
  - `docs/literature/ml_landscape_catalogue.md` (116 techniques, 29 fields);
  - `landscape_grounded.md` (top 5: packet-plan order, H1 validation, settings+reference baseline, label audit,
    novelty-aware error bars);
  - `landscape_exploratory.md` (skip-cycle mode decomposition, finance jump statistics, two-clock model of the
    row-start drift, censored labels).
- **Packet plan** (`lvl1/make_packet_plan.py`), now 43 packets:
  - session A shuffled (delayInt 1500, 300, 3000, 500, 2000, 700, with delay_H and delayRGH between them);
  - session B's two signs far apart in time;
  - two non-nominal replicates (delayInt 300 late in A, delayInt 3000 in mid B);
  - README updated. The slides still show the old 41 packets.
- **H1 validation** (agent, commit a16bbb9; `lvl1/train_lvl1.py`, `src/ml/evaluate.py`,
  `tests/test_lvl1_validation.py`, 12 tests):
  - packet-bootstrap CIs, paired comparison vs physics (`--paired-ref`), CRPS, worst-amplifier coverage;
  - CV grouped by setting (default; `--group-by packet` for the old behaviour);
  - `--holdout-session C`, `--loao`.
- **Signal-analysis literature** (`docs/literature/signal_analysis_features.md`): 21 fields, 49 candidates. Findings:
  - the first-skip offset = the CDS reading a decaying baseline slope (lever arm ≈ −211…−221 samples, R² 0.95–0.98);
  - skip 0 contains an exponential with τ ≈ 11–12.5 µs that scales with the H-clock swing (×1.17 in our feature,
    ×1.23 against settings with the same NCOLPRE);
  - ch2 shows telegraph steps inside single windows.
- **`docs/trace_feature_candidates.md`:** every candidate in tiers, plus the settings-separation test on 59 real traces
  (`docs/slides/feature_sep_test.py`, data in `docs/slides/data/feature_sep_test.json`):
  - no CDS-noise statistic separates the 4 voltage settings (they did not change the noise);
  - many "rejected" statistics separate the amplifiers;
  - only RMSSD/SD is redundant.

## 2. The current task (user, 2026-10-21)

"Implement tier 1 and tier 2 new features and curves and give some results on how they perform, also include a new
notebook for them, then start with these steps on the new full set of trace features."

**The steps:**

- (1) Tier ALL features, existing and new, using the real traces: redundancy clusters, repeat stability, physics prior.
- (2) Synthetic tolerance test: add 0/20/50/100 extra columns (noise, near-copies, shuffled real values) and see how
  each model degrades.
- (3) The actual choice of feature set per model waits for real labels (sessions A+B by setting-grouped CV, session C
  held out).

The session may run out: document every step here and push.

**Plan:**

1. In `src/trace_features.py`:
   - add the tier 1 and tier 2 features of `docs/trace_feature_candidates.md` §3;
   - add a `tier` attribute to `Feature` (0 = existing);
   - add a new `start:` group;
   - add new curves.
2. Unit tests: `tests/test_trace_features_new.py`, on synthetic signals with known answers.
3. Run on the 59 traces of 2026-10-12, and compare with the pilot numbers in the signal report: settings F, amplifier F,
   redundancy.
4. A notebook on the new features (generator script, as `lvl1/make_walkthrough.py`).
5. Step (1) tiering on all features; step (2) tolerance test.
6. Keep `process_packet.py` / `real_inputs.py` working (new features flow through automatically); check that the
   models train.

## 3. Status (update after each step)

- [x] 1 features implemented: `src/trace_features_extra.py` (EXTRA registry + extra_features(ctx)), registered in
  `trace_features.FEATURES` with `tier` (0 = original 85 scalars + 14 curves; tier 1: 23 scalars incl. 2 curves...;
  `names_by_tier()`), called as block F of `_trace_features`. 85 + 23 + 26 scalars = 134; curves 14 + 4 (start: mean
  transient, noise: Hadamard deviation, noise: modified Allan deviation, noise: within-cycle structure function).
  1.2 s per trace. Fixes made: 1/f corner clipped to the band; matrix-pencil poles chosen by amplitude, "fast decay"
  only among poles decaying within 10 skips. Not implemented (documented as such): RTS HMM dwell time (#21), RTS step
  rise time (#22), jitter decomposition (#43-44), in-window drift vs baseline (#14), EMG, spectral flux, cepstrum.
- [x] 2 tests: `tests/test_trace_features_new.py`, 16 tests pass (synthetic signals with known answers + the pilot
  numbers on one real ch0 trace: lever arm -212, R^2 0.97, tau 12.9 us, amplitude 614 ADU, row-start ratio 0.92).
- [x] 3 results on real traces: `docs/slides/oct12_features.py` rerun (59 x 134 features in
  `docs/slides/data/oct12_trace_features.csv`); `lvl1/feature_study.py` -> `lvl1/feature_study/` (feature_table.csv,
  evidence_tiers.json, README.md with the results). Evidence tiers A/B/C for ALL features (= step 5a done too):
  original 85: 46 A / 22 B / 17 C; new tier 1: 11/3/9; new tier 2: 11/6/9. Highlights: start transient amplitude
  (matched filter) settings F 264, amp F 70, repeat share 0.02; ch2 step rate 8.2 % vs 0, null kurtosis 5.2, jump share
  0.27; pickup-line and start-mode features unstable (repeat share 0.6-1.3).
- [x] 4 notebook: `lvl1/make_feature_notebook.py` -> `lvl1/new_trace_features.ipynb`, 8 sections, executed, all cells ok.
- [x] 5a tiering of all features (done with step 3, `lvl1/feature_study.py`)
- [x] 5b synthetic tolerance test (rerun after the stop, completed): results in lvl1/feature_study/README.md
  ("Step 2") and tolerance_*.csv. Plain gp collapses with +20 uninformative columns (sigma_e +51 pp); nn and ridge
  degrade steadily; trees immune; physics/physics_joint/sets sigma_e immune, lambda degrades mildly. Consequence: the
  plain gp must get tier A only on real data (134+134 features would break it).
- [x] 6 pipeline check: process_packet on 2 rehearsal packets -> model_inputs features.npz X_trace (8, 134), 13 start:
  features, curves (8, 18, 256) incl. Hadamard and mean transient. OK.
- [x] Document `docs/trace_features_and_tiers.md` (user request): new features, the 3-layer tier structure
  (implementation 0/1/2, evidence A/B/C, model feature sets), test results, the 43-packet scan, and the plan for the
  first data: (1) finalize tiers after session A (add label relevance), (2) per-model tier ablation S0/SA/SAB/SABC/
  S-impl0 with paired comparisons, chosen on A+B, confirmed once on C.
- NEXT STEPS (not started): `--feature-tiers` option in train_lvl1.py (per-model default
  feature sets: gp/nn/ridge = evidence tier A; trees/sets/physics_joint/cnn = A+B; physics lambda GP = A); tolerance
  test of gp_saas / gp_dim; re-run process_packet on the rehearsal to confirm the new features flow into model_inputs.

## 4. Next request (user, 2026-10-21, later)

1. Update the context and push (this section).
2. Build `--feature-tiers` in `lvl1/train_lvl1.py`.
3. The training script should automatically try the different tiers and compare performance (tier ablation).
4. Can tier A be broken down further, so that all tiers have a similar number of features?
5. Can an ML model help decide which feature goes in which tier (via the weights or importances of trained models)?
6. Standing preference: "when I tell you to write up results I want you to write up an overleaf latex doc with the
   results, not md" (saved in memory).

Status of this request: see section 5 below (updated as the work goes).

## 5. Status of the section-4 request

- [x] 1 context updated and pushed.
- [x] 2+3 `--feature-tiers` (all / none / auto / tier labels such as A, A+B, A1, impl0) and `--tier-ablation [SET ...]`
  (every model x feature set; paired comparison of each against its own `--ablation-ref` run, default `all`; "Tier
  ablation" section in the report) in `lvl1/train_lvl1.py`; `MODEL_TIERS` holds the per-model defaults for `auto`.
  Tested on synthetic td41_s0 (commit 2f433d5).
- [x] 4 tier A split by informativeness into A1/A2: A1 34, A2 34, B 31, C 35 (`lvl1/feature_study.py`,
  `evidence_tiers.json`).
- [x] 5 `lvl1/feature_importance.py` (commit c28faca): stability selection (elastic net, packet bootstrap) +
  permutation importance (HistGradientBoosting, grouped folds), each against Boruta shadow features over 5 rounds.
  Learned tiers: L1 = both methods beat the shadows' 95th percentile by > 1.5x; L2 = candidate; L3 = not shown.
  Synthetic td41_s0/1/2 with 20 noise columns: 0 noise columns in L1 (with the margin rule; 2 in one run without it),
  but 5-6 of 20 in L2, so L2 is not a feature set. Consistent L1: reset-level scatter, detrended CDS rms, dip recovery
  time, horizontal min/max. Write-up in lvl1/feature_study/README.md, step 3.
- [x] 1.5 margin confirmed on unseen td100_s0 (0/20 noise in L1, 4 in L2, 3 real in L1) and td13_s0 (0 in L1, 8 in L2,
  5 real in L1).
- [x] Importance results added to `docs/trace_features_and_tiers.md` (6.3 updated, new 6.4).
- Figures shown to the user today (scratchpad, not in the repo): top 16 new features (settings F vs amplifier F), start
  transient (lever arm + H-rail dependence), ch2 telegraph, jump share, noise budget, tolerance curves.

## 6. User idea (2026-10-21, late): clocking-region features (pilot done, not implemented)

User: the first V-clock block has a clear upward slope, the second is flat; the ADU difference between H and V
clocking; the first H/V clocks after coming off a row look different. Pilot on the 59 traces:
`lvl1/feature_study/clock_pilot/` (pilot.py, clock_pilot.csv, fig_clock_regions.png). Findings:
- V block 1 starts the trace at a random point (14-19 of 20 transfers recorded) and climbs +300 to +1,200 ADU per
  transfer (recovery from before the trace, tau ~ms) but NOT repeatable (settings F ~2, repeat share 0.3-0.9) -> tier C
  unless the acquisition start is synchronized.
- V block 2 slope: -90 to -230 ADU/transfer, repeat share 0.03, settings F min 45 (steeper with V rails low); max
  |rho| with existing features 0.71 -> new, candidate A.
- V transfer peak-to-peak (V feedthrough amplitude): settings F min 544, repeat 0.00, |rho| 0.84 -> new, candidate A.
- H/V peak-to-peak ratio: settings F min 879, repeat 0.00.
- H-V level difference: settings F 194 among the NCOLPRE-185 settings, but flips sign with NCOLPRE (reference +39 k).
- First H block of row 2, peak-to-peak excess over the other blocks (transfers 0-4): settings F min 172, repeat 0.01,
  largest with H rails low; |rho| 0.84 with the droop features.
- Redundant: H transfer-1 dip (|rho| 0.98 with fit: baseline droop amplitude / slow: H-transfer first-to-last change).
DONE (user said go ahead): `clock:` group in src/trace_features_extra.py (4 tier 1, all evidence A1; leading-block slope
tier 2, evidence B); ctx gets regions + trace; test_clock_regions; oct12 CSV regenerated (59 x 144); feature_study rerun:
139 scalars, A1 36 / A2 36 / B 32 / C 35. LaTeX doc docs/overleaf_feature_tiers_2026-10-21/ updated (section 3.5, 13
pages, no warnings) and re-zipped; README and trace_features_and_tiers.md updated.
