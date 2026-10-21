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

- [ ] 1 features implemented
- [ ] 2 tests
- [ ] 3 results on real traces
- [ ] 4 notebook
- [ ] 5a tiering of all features
- [ ] 5b synthetic tolerance test
- [ ] 6 pipeline check
