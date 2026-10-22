# Session context, 2026-10-21

Continues `2026-10-19_session_context.md` (it also covers 10-20: SAAS defaults, slide deck, model summary PDF).
The session ran past midnight; the last items (section 10) are from 2026-10-22.
**If the session reset: read section 0 first.** Sections 1-10 are the history, in order.

## 0. Where things stand (end of the session, 2026-10-22)

**Trace features** (`src/trace_features.py` + `src/trace_features_extra.py`): 141 scalars + 18 curves per trace,
1.2 s per trace; 85 original (implementation tier 0), 29 tier 1, 27 tier 2. New today: start-of-pixel transient,
telegraph/jump, noise shape, pickup lines, noise budget (from the literature search); the `clock:` group (user's idea:
second V block slope, V peak-to-peak, H/V ratio, first-H-block excess; leading V-block slope stored only); the pre-pulse
dip (user's question: depth and dip/pulse ratio). 18 + 15 unit tests pass (`tests/`).

**Tiers** (`lvl1/feature_study.py` -> `lvl1/feature_study/evidence_tiers.json`, from the 59 traces of 2026-10-12):
- P, protected (guard a): 8 direct sigma_e/gain estimators, given to every model;
- A1 37 / A2 36 / B 33 / C 35 evidence tiers (label-free: separation, repeatability, redundancy);
- learned tiers L1/L2/L3 (`lvl1/feature_importance.py`): diagnostics only, they never choose features.
Guards b-d were prototyped (scratchpad) and NOT adopted; the labels never order the features.

**Training** (`lvl1/train_lvl1.py`): `--feature-tiers auto` uses MODEL_TIERS: gp P, gp_saas P, gp_dim P+A1, nn/ridge P+A,
trees all, everything else P+A+B. `--tier-ablation [SET ...]`. `--feature-ladder [tiers cover random groups]` feeds every
model more and more features in label-free orders and suggests the smallest step within 1 SE of the best.

**Synthetic ladder** (`lvl1/feature_study/ladder_synthetic/`, 8 models): plain gp collapses beyond 15-25 features;
gp_dim flat up to 137; gp_saas loses lambda with more features; nn degrades every step; ridge best ~25-45; trees,
physics_joint, sets flat for sigma_e/gain, lose on lambda.

**Write-up:** `docs/overleaf_feature_tiers_2026-10-21/` (LaTeX, 16 pages, compiles without warnings; zip
`docs/overleaf_feature_tiers_2026-10-21.zip`). Markdown: `docs/trace_features_and_tiers.md`, `lvl1/feature_study/README.md`,
`lvl1/README.md`.

**Open / next:**
1. Session A data: run process_packet, rerun `lvl1/feature_study.py` on the real packets (label-free), freeze the ladder
   orders BEFORE looking at labels, then `train_lvl1.py --feature-ladder tiers cover random groups` on A (+B later);
   confirm once on session C (`--holdout-session C`).
2. Label relevance as a reported-only column of `feature_study.py` (not built).
3. Session B RG_H/RG_L/SW_L packets: identify which clock edge makes the pre-pulse dip.
4. The trace-acquisition start is random inside the first V block; if it can be synchronized, the leading V-block slope
   becomes usable.
5. Open from before: the trace-settings decision (BOOLCLR/NDCMPRE); the slides still show 41 packets.

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

## 6. User idea (2026-10-21, late): clocking-region features (pilot, then implemented)

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

## 7. User question: the dip just before the per-cycle (reset) pulse (pilot, then implemented)

Never measured as a number before: it sits inside the `edge: equivalent-time pulse` curve (from -15 samples) and
inside the window of `edge: cycle variation at reset`, but no scalar measures it. Pilot on the 59 traces
(`lvl1/feature_study/prepulse_pilot/`: pilot.py from the equivalent-time pulse, predip.csv, fig_prepulse_dip.png):
- a sharp dip ~120-130 ns before the pulse's half-height crossing, ~70-100 ns wide;
- depth per amplifier: ch0 -886, ch1 -505, ch2 -990, ch3 -601 ADU; repeat share 0.001; amplifier F 5851;
  settings F 0.2 (none of the 2026-10-12 settings touched RG/SW/OG/DG);
- not a fixed fraction of the pulse (dip/pulse 0.011-0.021 by amplifier; no correlation with the pulse height within
  an amplifier); |rho| 0.95 with amplifier-identifying features, which on these data is unavoidable.
- origin (reference_v5.seq, ReadoutCDS): right after the signal window SW goes high and RG low together, 3 units later
  SW and OG go low, then OG high, then RG high; the flags do not record these clocks, so the edge is not identified.
  Session B's RG_H/RG_L/SW_L packets (L1_17, L1_18, L1_19, L1_28, L1_29) will show which rail it follows.
DONE: both added (extra_features `prepulse` block, from the ETS curve; ctx gets `curves`); test_prepulse_dip; 141 scalars;
tiers A1 37 / A2 36 / B 33 / C 35; LaTeX section 3.6 + tier caveat + proposed guards (14 pages), md docs updated.

## 8. User concern: do the label-free tiers bias us against the best noise predictors? YES, found evidence
`skip: CDS rms (detrended)` is B (cluster rep = waveform modes, chosen by amplifier separation); reset-level scatter B;
dip recovery time B. With --feature-tiers auto, gp/nn/ridge (tier A) would not see them. Proposed guards, waiting for
the user's go-ahead: (a) protected physical-estimator set always included; (b) clusters within amplifier; (c) no
repeat-share penalty until compared with the labels' repeat scatter; (d) label relevance + learned L1 on all features
override the evidence tier once labels exist. OUTCOME: only (a) adopted (section 9).
Guards PROTOTYPED IN THE SCRATCHPAD ONLY (user: "implement the guards on a scratchpad and test them, then I make the
call whether to change fully"): `<scratchpad>/guards/guarded_study.py` (guards a-c + label_relevance) and
`build_and_run.py` (test). Repo code unchanged. Results:
- real 59 traces: 21 features change tier; guarded sizes A1 41 / A2 41 / B 24 / C 35; protected set P = 8 estimators.
- synthetic test: td41_s0/1/2 + the 120 other real features as uninformative real-named columns (values drawn at
  random); D = synthetic features that are learned-L1 in >= 2 of the OTHER importance runs or have within-amplifier
  label relevance > 0.5 on the other datasets. Error %, mean of 3 (sigma_e / lambda / gain):
  gp: A 57.5/29.2/31.3 (collapsed, = mean); P 9.6/22.0/0.9; D 8.9/10.9/0.9; P+D 9.0/14.5/0.9; TRUE 6.9/8.0/0.9
  nn: A 32.9/22.4/17.6; gA+P 21.7/21.9/10.2; P+D 10.2/16.9/2.6; TRUE 10.1/17.1/1.5
  ridge: A 16.8/19.9/4.4; gA+P 14.7/23.2/3.3; P+D 9.5/14.0/0.9; TRUE 9.5/12.4/0.9
  Guards a-c alone help nn/ridge on sigma_e and gain but cannot save the plain gp (tier A still holds ~80 columns);
  the label-driven set (d) nearly reaches the true-feature result. Waiting for the user's decision.

## 9. Guard (a) adopted, feature ladder built (user: "adopt guard a and build the feature ladder", labels must not order the features)
- `PROTECTED` (8 estimators) in lvl1/feature_study.py -> tier "P" in evidence_tiers.json; MODEL_TIERS: gp P, gp_dim/gp_saas
  P+A1, nn/ridge P+A, trees all, default P+A+B. Guards b-d NOT adopted (labels never order the features).
- `--feature-ladder [tiers cover random groups]` in lvl1/train_lvl1.py (cover_order = column-pivoted QR after P,
  label-free; random = control; groups = P + one family); one-SE suggestion; feature_ladder.png; tests in
  tests/test_lvl1_validation.py (TestFeatureLadder).
- Synthetic ladder (lvl1/feature_study/ladder_synthetic/): gp best 8-15 features, collapses beyond 15-25 (-> gp default P);
  nn worse with every step; ridge best ~25-45; trees flat; physics_joint/sets flat for sigma_e/gain, lose on lambda.
  Cover order did not beat random (expected with all-junk extras).
- Docs: trace_features_and_tiers.md (sec 3 tiers with P, 6.1-6.3 rewritten, ladder results), lvl1/README.md,
  lvl1/feature_study/README.md, LaTeX (16 pages: P, sec 7.1 guards, 7.2 ladder + figure, plan rewritten), zip rebuilt.

## 10. Ladder of gp_dim and gp_saas (user, 2026-10-22: "run the ladder on gp_saas and gp_dim too")

Same synthetic test (td41_s0/1/2 + 120 uninformative real-named columns, `--quick`), error % mean of 3
(sigma_e / lambda / gain):
- gp_dim: settings only 57.0/21.3/31.2; P 9.0/22.2/1.1; P+A1 8.4/22.1/1.4; all 9.4/19.1/1.6 -> flat everywhere, no
  collapse (dimension-scaled prior works); default stays P+A1.
- gp_saas: P 8.9/19.4/0.9; P+A1 9.0/28.1/1.0; all 10.6/29.5/0.9 -> sigma_e and gain nearly flat, lambda worse with every
  step; default changed to P. gp_saas took ~2-4 min per set (NUTS); gp_dim seconds.
Results merged into lvl1/feature_study/ladder_synthetic/ladder_all.csv and the figure (8 models); docs and LaTeX updated.

## 11. RESTART HERE (2026-10-22, stopped for a session-limit reset)

**User request in progress:** "why haven't the trace features and trace features technical latex been updated based on
yesterday. Is there a technical document for the ML models; if so update it, if not write it (all relevant details).
Make sure everything done so far is accurate and validated."

**Done (this commit):**
- `docs/overleaf_trace_features/trace_features.tex` regenerated for 141 scalars / 18 curves with tier column, tiers
  section, origins of new features (generator now `src/make_trace_features_note.py`, the updated copy; old
  `docs/trace_features.tex` removed; figure height 0.78 textheight and \clearpage after figure blocks). 38 pages, 0 warnings,
  zip rebuilt by the generator.
- `docs/overleaf_trace_features_technical/trace_features_technical.tex`: exact definitions of all new features, ctx,
  tiers (23 pages). EDITED AFTER ITS LAST COMPILE (Lee-Mykland constants, clock fallback, noise-budget guard, path):
  RECOMPILE + re-zip `docs/overleaf_trace_features_technical.zip`.
- NEW `docs/overleaf_ml_models_technical/ml_models_technical.tex` (+ zip): technical reference of every model, data flow,
  validation, selection, synthetic world (14 pages). EDITED AFTER ITS LAST COMPILE (packet state, grouped ElasticNetCV,
  log curves, fixed-issues list): RECOMPILE + re-zip.
- `docs/overleaf_ml_models/ml_models.tex` updated to 2026-10-22 (new "Changes since 14 October", section 18, plan; 45
  pages); EDITED AFTER COMPILE (Data bullet): RECOMPILE + re-zip `docs/overleaf_ml_models.zip`.
- `docs/overleaf_feature_tiers_2026-10-21/`: validation agent fixed many numbers (56 not 49 features, noise-budget per
  channel, +250-290, DFA/entropies implemented, etc.); EDITED AFTER COMPILE (Lee-Mykland line): RECOMPILE + re-zip.
- Validation audit (agent): tests 33/33 pass; oct12 CSV / feature_table / evidence_tiers reproduce exactly; pipeline
  end-to-end OK (rehearsal model_inputs X_trace (8,141)); train_lvl1 auto/ablation/ladder OK. It FIXED in train_lvl1.py:
  mean/trace_physics keep plain names under --feature-tiers auto (vs-mean ratios were silently lost); --paired-ref resolves
  to "physics [P+A+B]" etc. (paired comparison was empty under auto); groups ladder gets no "suggested set".
- Code fixes by me (2026-10-22), tests pass, tiers unchanged (A1 37/A2 36/B 33/C 35/P 8):
  * Lee-Mykland constants now carry c = sqrt(2/pi) (was 1.6 false jumps per Gaussian series, now ~0.26; still relative);
    oct12 CSV + feature_table regenerated; jump count median ch0/1/3 = 0, ch2 = 8.5; its tier A2, amp F 31.
  * clock: V features NaN when no row start after pixel 0 (regions.v would be the non-repeatable leading block);
  * noise-budget raises (-> NaN) if sig + n_ped > cycle; Hadamard and matched-filter registry descriptions corrected;
  * packet_state (train_lvl1.py) is amplifier-centred like StructuredPrep.state_vector;
  * feature_importance.py ElasticNetCV uses packet-grouped folds;
  * real_inputs._log_names also logs Hadamard and structure-function curves;
  * process_trace docstring (141/18), src/trace_features.py doc path, plot_feature_figures.py colours for start/clock and
    no hard-coded 43.

**Interrupted (rerun after restart):**
1. Importance re-check with grouped CV: for d in td41_s0 td41_s1 td41_s2 td100_s0 td13_s0:
   `.venv/bin/python -W ignore lvl1/feature_importance.py --inputs <scratchpad>/gpcmp/$d/model_inputs --out <scratchpad>/imp2/$d --add-noise 20`
   (scratchpad = /private/tmp/claude-501/-Users-dylanrosenmerkel-Documents-Software-ccd-opt-ml/0face8dd-3b86-44b8-b894-2b5f02a97511/scratchpad;
   if the scratchpad is gone, rebuild synthetic sets with the gpcmp scripts / make_test_data). Check "check:" lines: is
   L1 still free of noise columns? Update lvl1/feature_study/README.md step 3 and the docs if numbers change.
2. Full-model rerun with the fixed packet state:
   `.venv/bin/python -W ignore lvl1/train_lvl1.py --inputs lvl1/test_data/model_inputs --models mean settings trace_physics physics physics_joint physics_hier sets ridge trees gp nn cnn pretrained deep_kernel gp_dim --n-boot 1000 --out <scratchpad>/mltech/full100_v2`
   (~10 min). Then update Table tab:full100 in ml_models_technical.tex and the "Testing without real data" table in
   lvl1/README.md (still the 2026-10-15 numbers: physics 5.5/4.9; the pre-fix 2026-10-22 run gave physics 5.3/3.9/0.9).
3. Recompile (tectonic: <scratchpad>/tectonic/tectonic, or reinstall) and re-zip the four LaTeX docs flagged above; send
   the PDFs to the user with a summary of the validation findings.

**Open decisions for the user (report them):**
- The clock: features are NaN with the planned packet trace geometry (daq_scripts/manual_packet.sh: NPBIN=1, NSBIN=1,
  NCOLPRE=100); they need >= 3 V and >= 5 H transfers (2026-10-12 traces were 20x20). Go back to 20x20 trace binning,
  or adapt the features.
- Still-open behaviour changes: sets fits lambda with the weak-prior GP (not SAAS); --quick does not shorten sets;
  Allan NaN filled with the global median in _physics_X; NN early stopping validates on rows not packets; per-volt H/SW
  features always use the side-A (H1_A / SW_1) swing.
- Lambda inputs: log CCD temperature and time since clear/erase per packet; per-target feature sets.

## 12. Resumed after the reset (2026-10-22): the interrupted items are done

1. Importance re-check with packet-grouped ElasticNetCV (scratchpad imp2/): (noise in L1, noise in L2, real in L1)
   td41_s0 (0, 6, 5), td41_s1 (0, 5, 4), td41_s2 (0, 5, 8), td100_s0 (0, 3, 3), td13_s0 (1, 6, 6). So 1 of 100 noise
   columns reached L1, on the 13-packet set (the size of session A). Same real L1 features as before (CDS rms,
   reset-level scatter in 5/5; dip recovery 4/5). All docs that said "no noise column in L1" corrected (feature_tiers
   LaTeX + figure, ml_models.tex, ml_models_technical.tex, trace_features_and_tiers.md, feature_study/README.md,
   feature_importance.py comment). Learned tiers stay diagnostics only.
2. Full-model rerun on lvl1/test_data with the fixed (amplifier-centred) packet state: every number identical to the
   pre-fix run (all synthetic packets have 4 amplifiers, so centring shifts the state by a constant). lvl1/README.md
   results table updated (physics 5.3 / 3.9 / 0.9; calibrated 68 % coverage 0.67-0.69 for every model).
3. All LaTeX docs recompiled with 0 warnings and re-zipped: trace_features 38 p, trace_features_technical 23 p,
   ml_models 45 p, ml_models_technical 14 p, feature_tiers_2026-10-21 16 p.
Open decisions for the user: unchanged (section 11).
