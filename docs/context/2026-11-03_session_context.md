# Session context, 2026-11-03

Continues `2026-10-21_session_context.md` (sections 0-16; 16 = the board-reset fixes of 2026-11-02) and
`data_taking_scripts_handoff.md`. **If the session reset: read section 0 first.**

## 0. Where things stand

User request (2026-11-03): look at the new data in `data/`, process the packets, and do the first training tests
("the data isn't perfect, work with what we have; cleaner data next stage"); follow the plans already made
(lvl1/README.md, lvl1/packet_plan/PLAN.md, the 10-21 context section 0 "Open / next").

The agreed protocol (lvl1/README.md, "The prospective test"):
1. label-free feature tiers on the real traces, frozen before any labels are used;
2. choose the model(s) on the setting-grouped CV of sessions A + B (L1_01..L1_32);
3. freeze that choice (date + commit) in this document;
4. only then score session C (L1_33..L1_43) once with `--holdout-session C`.
Session C stays sealed: its labels are not looked at and it is not in the A+B inputs (`processed_AB/`).

## 1. The new data (taken 2026-11-02 15:20 to 2026-11-03 05:10 local, user)

* `data/2026-11-02-PA03_PQ04_Packet_Test/`: an aborted start (one L1_01 ref1 trace); not used.
* `.../Packet_Test2/`: L1_01 (2 images; the manifest takes the one after the last trace, 17:20).
* `.../Packet_Test3/`: L1_02.
* `.../Packet_Test4/`: L1_03 .. L1_43 (41 board-101 images, 40 of board 102: L1_18's board-102 image is missing,
  and its board-102 traces are partial; module 102 is not used for trace analysis anyway).
* `data/packet_records/`: the 43 records; `hdu_end_abtest/`: a separate A/B trace test of 2026-11-02 14:34 (not a
  packet; not used).
* Records: `flush_between = "none"` (the user ran without flushes between trace sets), 5 ref + 10 sets x 4 channels.
* **Settings check** (scratchpad `data1103/check_settings.py`, every board-101 .meta against the plan): every packet's
  reference traces are at nominal (no leak from the previous packet: the reset works), every requested change arrived
  in the traces and the image. VSUB reads back the potentiometer step (40 -> 40.19, 50 -> 50.17, 41.26 -> 41.51 ...),
  0.2-0.25 V above the request.
* `src/make_manual_manifest.py`: the biases now have the readback tolerance 0.3 V (was 0.15 V: the six VSUB packets
  got status "check" for the step); all 43 manifests `ok` (`manifests/`).

## 2. Processing (`src/process_packet.py`, output `processed/`)

* L1_01 alone: 3.3 min; 116/120 traces usable (the 4 unusable are board 102), board 101: all 60 usable.
* Labels L1_01 board 101 (run_WADERs): ch0 sigma 0.152 e, gain 1.344, lambda 0.344 e/pix; ch1 0.153 / 1.324 / 0.393;
  ch2 0.175 / 1.258 / 0.603; ch3 skipped by run_WADERs ("unresolved outlier CCD") although its histogram has clear
  0-4 e peaks. lambda is ~10x the 2026-10-16 images (0.02-0.03): the CCD is in a high-dark-charge state.
* `configs/processing.toml`: `run_waders_args` now `--all-ext` (fit ch3 too) and `--overwrite` (reprocessing reuses
  the WADERs folder). ch3 then: sigma 0.36 e, gain 1.284, lambda 0.42, chi2/ndf 3.1 -> flagged by our check ("the fit
  model does not describe the image"), so it stays out of training unless `--include-flagged`.
* The other 42 packets: 4 parallel workers (scratchpad `data1103/run_all.sh`, logs `proc_<id>.log`, `progress.txt`).
  L1_06/08/09/14/16 first crashed on damaged traces (`decode_trace.check_l1` on a trace with no complete CDS window;
  `trace_features` on pixels of unequal skip counts). Fixed: such traces are marked unusable (`no complete pedestal and
  signal window`, `features failed: ...`); rerun, all 43 processed. The manifests' paths are relative to `src/`: run
  `build_dataset` / `build_model_inputs` from `src/`.
* Board-101 usable traces (of 60 per packet): 51-60 in A and B; C: 58-60 except **L1_37 20, L1_40 20, L1_41 19**
  (data loss). Total 2370 / 2580.
* `processed_AB/` (scratchpad `data1103/make_subset.py AB A B`): symlinks to L1_01..L1_32 + their dataset and
  model_inputs. Session C is not in it.
* **A+B labels** (board 101, scratchpad `label_summary.py`): 119 of 128 ok (9 flagged chi2/ndf > 3; ch3 fits in 29 of
  32 packets with --all-ext). Medians: sigma_e ch0 0.152, ch1 0.155, ch2 0.175, ch3 0.183 e; gain 1.338 / 1.316 /
  1.258 / 1.267; lambda 0.28 / 0.35 / 0.67 / 0.52 e/pix (range up to 2.7 on ch2). Nominal repeatability (L1_01, 07, 14,
  32; sd of log): sigma_e 1.4-2.1 % (quoted 0.8-0.9 %), gain 0.5-0.8 % (0.3 %), **lambda 8-14 % (quoted 2 %)**: lambda
  drifts down over the night (ch0 0.344, 0.354, 0.331, 0.260), so lambda's label noise is set by the CCD state, not by
  the fit.

## 3. Label-free tiers on the real packets (`lvl1/feature_study.py --packets`)

New mode (the 10-12 mode reproduces the committed tiers exactly): rows = board-101 traces at the packets' own
settings, sessions A + B; the settings comparison uses PACKET means (trace sets are minutes apart and hide the drift):
settings F = variance of the packet means / pooled variance within the replicated settings (nominal x4 in A+B,
delayInt 300 x2, delayInt 3000 x2; 5 degrees of freedom), per amplifier, min and median; repeat share = that
within-setting variance / total; amplifier F within each packet over its trace sets; clusters on packet means. Same
tier rule as before. Results: 32 packets, 27 settings, 1223 traces: **A1 38 / A2 38 / B 19 / C 46 / P 8**
(frozen in git, `lvl1/feature_study/real_AB/`, commit b7d25ca, before any training). Versus the 10-12 tiers only 64 of
141 features keep their tier (the 10-12 settings did not change the noise; session A does). The protected estimators
are given to every model (guard a) whatever their tier (white noise per skip B, CDS rms A2, Allan n=50 C, reset
scatter B, dip per volt SW B, pulse per volt RG A1, null-CDS A1, jump-robust B). 17 of the 46 C are NaN in > 20 %.

## 4. First training (A+B, setting-grouped CV, C sealed)

Run 1: `train_lvl1.py --inputs processed_AB/model_inputs --board 101 --feature-tiers auto --tier-file
lvl1/feature_study/real_AB/evidence_tiers.json --n-boot 1000 --out processed_AB/lvl1_results/2026-11-03_auto`
(log: scratchpad `data1103/train_auto.log`). 119 examples, 32 packets, 27 setting groups (merged L1_01+07+14+32,
L1_04+13, L1_06+23). Error % (x mean); label noise at nominal ~2 / ~10 / ~0.6 %:

| model | sigma_e | lambda | gain |
|---|---|---|---|
| mean | 22.4 | 45.0 | 2.8 |
| settings | 23.2 (1.04x) | 43.7 (0.97x) | 2.8 |
| trace_physics | 68.7 (3.07x) | = mean | = mean |
| physics [P+A+B] | 27.8 (1.24x) | 20.6 (0.46x) | 1.5 (0.53x) |
| physics_joint | = physics | 33.8 | = physics |
| physics_hier (labels) | 27.5 | 34.9 | 1.2 |
| sets | 20.4 (0.91x) | 20.6 | 1.5 |
| ridge [P+A] | 29.6 | 34.8 | 2.1 |
| **trees [all]** | **16.2 (0.72x)** | 21.3 (0.47x) | **1.1 (0.40x)** |
| gp [P] | 25.4 | 35.6 | 1.7 |
| nn [P+A] | 18.0 (0.80x) | 24.0 | 1.1 |
| cnn | 19.7 | 25.3 | 1.1 |
| pretrained | 18.8 | 28.2 | 1.3 |
| deep_kernel | 18.5 | 40.2 | 1.4 |

Paired vs physics (95 % CI excludes 0): sigma_e better: trees -11.6, nn -9.8, pretrained -9.0, deep_kernel -9.3,
sets -7.3; lambda worse: physics_joint, physics_hier, gp, pretrained, deep_kernel (trees +0.7 and sets "same" are
level with physics); gain better: trees, nn, cnn.

**Why the physics family fails sigma_e here (diagnosis):** log(estimate / image sigma_adu) over the 119 rows: white
noise per skip / sqrt(NDCM): mean +0.136, sd 0.175 (per channel 0.12-0.22; within-channel corr with log sigma_e
0.43-0.61); CDS rms the same; Allan n=50 x sqrt(50/N): -0.03, sd 0.18; trace_physics (Allan a^2/n + b^2 fit, x mean
gain): +0.62, sd 0.30, within-channel corr -0.08..0.27. The Allan floor extrapolated to n = 500 is unstable on real
traces (L1_04 vs L1_13, same setting: x1.45 vs x1.17). Physics findings: the trace is ~14 % noisier than the image at
nominal; delay_Ha 3600 and delayRGH 400 raise the IMAGE sigma_e by ~45 % with no change in the trace's white noise;
delayInt 500 raises the image sigma_e by 50 % but the trace's white noise by 9 %; delayInt 3000 raises the trace's
white noise (5.6 -> 6.4 ADU) while the image sigma_e falls slightly (0.152 -> 0.142). Not yet understood (candidates:
the image pixels see clocking the 2 x 5 trace pixels do not; WADERs' sigma_e affected by the 10x higher lambda).

Run 2 (feature ladder; models settings physics physics_joint sets ridge trees gp gp_dim nn cnn pretrained deep_kernel;
orders tiers cover random; n-boot 300; out processed_AB/lvl1_results/2026-11-03_ladder; log train_ladder.log): running.

## 5. User questions (2026-11-03): why does the learned correction not fix the physics term; try the adjustment

* physics predicts log sigma_e = trace term - learned gain + learned correction (GP on settings, packet state and
  features). The correction can only learn what is a repeatable function of its inputs; the Allan-floor term jumped
  randomly between packets (x1.45 vs x1.17 at the same setting), which no input explains.
* **`PhysicsStructured(noise_source="white")`** (src/ml/structured.py): trace term = (white noise per skip)^2 / N,
  `X["white"]` from the physics file before any feature selection (train_lvl1 `_physics_X`; it is the physics term, not
  an input). New models (extra): `physics_white`, `trace_physics_white`; `NO_FEATURES` replaces the hard-coded
  ("mean", "trace_physics"). On the (drift-corrected) A+B labels: physics sigma_e 27.8 -> **17.9 %** (paired -9.9 pp
  [-18.5, -1.8], P 0.99); trace_physics 68.7 -> 22.7 %; lambda and gain unchanged. Synthetic: identical (no white
  feature there, falls back to the Allan term).

## 6. FOUND: the drift correction corrupted the session-A labels (data processing problem, fixed)

* Image sigma_e in session B (voltages) is steady (ch0 ~0.150); in session A (timing) it jumped incoherently between
  amplifiers (delay_Ha 900: ch0 0.694, ch1 0.342), and the replicate pair L1_06 / L1_23 (delayInt 3000) disagreed by
  60 % while their traces agreed.
* run_WADERs on the RAW vs the drift-corrected images (scratchpad `data1103/rawwaders/`), sigma_e ch0..ch3:
  L1_14 raw .148/.155/.168/.179, corr the same; L1_06 raw .145/.159/.186/.195, corr .348/.227/.184/.258;
  L1_23 raw .144/.144/.180/.181 (= L1_06 raw: the replicates agree); L1_03 raw .238/.221/.374/.247, corr .694/.342/.404/
  .348; L1_02 ch1 .183 -> .278; L1_05 ch1 .179 -> .267; L1_09 ch3 .236 -> .416; L1_08 the same. Raw chi2/ndf 0.8-1.3 at
  nominal: this night's images have little drift. Real effects that remain on the raw images: delayInt 500 (.232),
  delay_Ha 3600 (.227), delay_Ha 900 (.238) raise sigma_e.
* **`[drift] enabled = false`** (configs/processing.toml, with the evidence). `process_packet.py --reuse-traces`: keeps
  features.json and redoes only images + labels (~1 min per packet). Relabelling all 43 (scratchpad `relabel.sh`,
  6 workers, then processed/ and processed_AB/ rebuilt). The ladder run (section 4) was stopped: its labels were the
  corrupted ones. Everything is rerun after the relabel.

## 7. Clean labels (drift correction off) and the rerun

* Relabel of all 43 (traces reused): A+B labels 127 / 128 ok (ch3 in all 32). Nominal repeatability sigma_e 1.1-2.3 %,
  gain 0.5-0.7 %, lambda 10-14 % (the drift is in the raw images too: ch0 0.307, 0.328, 0.309, 0.240). The 5 "exit 1"
  of the relabel are the parallel dataset rebuild racing (labels written; the final rebuild ran after all workers).
* Rerun (`processed_AB/lvl1_results/2026-11-03_auto_raw`, log `train_auto_raw.log`), error % (x mean):

| model | sigma_e | lambda | gain |
|---|---|---|---|
| mean | 17.4 | 47.8 | 2.7 |
| settings | 17.9 | 46.4 | 2.7 |
| trace_physics / _white | 73.3 / 20.8 | = mean | = mean |
| physics | 30.6 | 37.0 | 1.2 |
| **physics_white** | 12.6 (0.73x) | 37.0 | 1.2 |
| physics_joint / hier | 30.6 / 31.2 | 37.4 / 43.9 | 1.2 / 0.9 |
| sets | 15.7 | 37.0 | 1.2 |
| ridge [P+A] | 103 (unstable) | 213 | 3.6 |
| trees [all] | 13.2 | **28.8 (0.60x)** | **0.8 (0.30x)** |
| gp [P] | 17.0 | 40.2 | 1.7 |
| nn [P+A] | 13.0 | 33.0 | 0.9 |
| cnn | 16.9 | 32.4 | 0.9 |
| **pretrained** | **11.8 (0.68x)** | 35.6 | 1.2 |
| deep_kernel | 12.7 | 43.9 | 1.2 |

  Paired vs physics, sigma_e (95 % CI excludes 0): physics_white -18.0, pretrained -18.8, deep_kernel -17.9, nn -17.6,
  trees -17.4, sets -14.9, cnn -13.7, gp -13.6. lambda: no model's interval excludes 0 except ridge (worse) and the
  baselines (worse); intervals are wide (physics 23.5-46.8 %): a few packets with high lambda (up to 1.2-2.6 e/pix)
  dominate. gain: trees -0.4 *. Ladder rerun on these labels: `2026-11-03_ladder_raw`, log `train_ladder_raw.log`.
* Pending user decisions: v2 packet plan (proposals in the chat of 2026-11-03: joint multi-level design, wider
  ranges, fixed pre-image routine + state logging (temperature, time since clear), nominal every 6-8 packets, fewer
  trace sets 3+5 to fit ~45 % more packets, automatic retake of damaged trace sets); drift correction only when it
  improves the image's fit (run WADERs on both, keep the better chi2).

## 8. Drift check, safeguards, v2 plan, CIC (2026-11-03, later)

* **DAQ edits merged** into daq_scripts/manual_packet.sh (from the other agent's unpushed branch, by its diff): traces
  per board, TRACE_CHANNELS override, no external-clock line. **User's flush scheme:** flush before / after the
  reference traces, 2 settling traces after the change, flush before / after the image, 2 s pause after every trace
  and the image.
* **Safeguards** (manual_packet.sh, CHECK_TRACES=1): waits for each trace phase (each board writes a marker into
  MARK_DIR when the queue reaches it), checks every board-101 trace (src/check_trace_files.py: missing, unfinished,
  decoder quality check), retakes damaged traces under the next set number (2 rounds), restarts the packet from the
  reset as <id>_r1 when more than 5 are damaged (record status "restarted", skipped by make_manual_manifest.py).
  Tested with a stub cdaq (scratchpad data1103/stubtest): 2 damaged -> retaken, complete; 6 damaged -> restart, r1
  complete. On the 2026-11-02 data ~1 packet in 10 would restart. Caveats: the clients must be on the DAQ machine;
  the repeat queue may run between phases.
* **32-row image** `configs/image_params_32rows.toml` (copy to cdaq image_params/tc1/img32.toml); IMAGE_TOML env.
* **v2 plan** `lvl1/make_packet_plan_v2.py` -> `lvl1/packet_plan_v2/` (76 packets, ~21 h: session D 61 = 48 Sobol
  joint points over 12 knob groups + 10 nominal + 3 replicates; session E 15 = 12 + 3 nominal, prospective test;
  3 + 5 trace sets; all dry-run OK).
* **Drift check: OFF for these data (final).** The per-channel rule (corrected only with less noise and a fit no
  worse, `fit_choice`, test) chose the corrected copy for 60 / 128 labels, but the subtraction raises lambda by 7-10 %
  on EVERY channel (L1_14 ch0 0.309 -> 0.331; the 200-column cut alone changes nothing), so mixing made lambda's
  nominal repeatability worse (ch0 13.8 -> 18.1 %). Any mix of corrected and uncorrected images is inconsistent;
  if used, apply to the whole dataset. `cut_columns` (image_drift.py) keeps both copies on the same pixels when on.
  Relabelled once more with it off: identical to section 7 (127 / 128 ok).
* **CIC as a target** (scratchpad-built `processed_AB/model_inputs_cic`: Y[:,1] = log lambda_cic;
  `lvl1_results/2026-11-03_cic`): mean 41.9, settings 41.7, physics_white 28.1, sets 28.1, ridge 45.7, **trees 26.1
  (0.62x)**, gp 35.7, nn 31.5, cnn 30.0, pretrained 32.3, deep_kernel 34.9 %. But the CIC LABEL repeats at nominal
  only to 8-25 % (ch2 0.40-0.69; quoted 3 %), about its whole spread over the packets on ch2/ch3 (18-24 %): with 16
  rows the dark-current / CIC split is unreliable (slope and intercept trade off; rows chi2/dof median 4.4). Total
  lambda (2 % quoted, 10-14 % repeat) is the better target now; 32 rows should help the split.

## 9. How lambda is labelled, and the serial / array split test (not adopted)

* sigma_e, gain and total lambda: run_WADERs (unchanged). The dark-current / CIC split is ours (src/image_rows.py,
  per-row lambda with WADERs' gain and sigma fixed, straight line over rows): its "CIC" intercept is everything common
  to all rows, i.e. also serial-register charge and charge collected before the readout. Evidence: the OVERSCAN (empty
  serial-register cells) holds 70-85 % of the active lambda (median ch0 0.20 vs 0.28); lambda rises ~x1.5 along the
  row (serial-register wait); L1_18 (queue stalled 38 min before its image, no flush) active 0.67, overscan 0.22.
* `image_rows.overscan_lambda` now records `label.lambda_overscan` (and `lambda_active`) for every label (relabelled;
  recorded only, no model uses it).
* Test (scratchpad data1103/components.py; `lvl1_results/2026-11-03_serial`, `_ratio`): log lambda = log overscan +
  log(lambda / overscan), each part predicted, summed, vs the direct lambda prediction, packet bootstrap:
  trees 28.8 (direct) vs 30.8 % (sum) [+2.0, CI -0.5..+6.4]; physics_white 37.0 vs 33.7 [-3.3, -11.7..+6.1]; nn, cnn
  ~0; no model significantly better -> NOT adopted. The serial part is predictable (physics/sets 20.5 % vs mean 39.8 %);
  the array ratio is not (best ~29-32 % vs mean 34.5 %): lambda's unpredictable part is in the array (CCD state,
  time since the last flush), which the v2 procedure (flush right before the image, fixed timeline, more nominal
  packets, 32 rows) addresses. The parts' nominal repeatability (overscan 14-37 %, ratio 19-31 %) is worse than the
  total's (10-14 %).

## 10. Feature ladder on the clean A+B labels (`processed_AB/lvl1_results/2026-11-03_ladder_raw`, figure feature_ladder.png)

Orders tiers / cover / random (3), 13 models, same setting-grouped folds. Error % at each step (tiers order):

| model | sigma_e best (set) | lambda best (set) | gain best (set) |
|---|---|---|---|
| physics_white | 12.3 (P+A); 13.9 with none | 30.7 (P); 27.7 (cover 25) | 1.1 (P) |
| sets | 15.3 (P+A1), flat | = physics | = physics |
| trees | 12.1 (P+A1) | 28.8 (all) | 0.8 (P+A..all) |
| gp | 13.8 (P+A), else ~17 | 38.6 (P+A1) | 1.7 (P) |
| **gp_dim** | **10.5 (P+A1)** | 34.1 (all) | **0.8 (P)** |
| nn | 12.8 (all) | 28.9 (all) | 0.8 (P+A+B) |
| cnn | 16.6 (P+A+B) | 29.4 (all) | 0.8 (P+A+B) |
| **pretrained** | 11.7 (P+A+B) | **27.0 (all)** | 1.0 (P+A) |
| deep_kernel | 12.4-13.5, flat | 38.1 (all) | 1.1 |
| ridge | 22.0 (none), unstable with any trace features (33-103 %) | blows up | 1.3 (P+A1) |

* With 32 packets the one-SE suggestions are mostly the smallest sets ("none"/"P"): the intervals are wide, so the
  ladder cannot yet separate feature-set sizes; differences of 1-3 points are not significant.
* Same pattern as on synthetic data: plain gp gains nothing beyond P-ish; gp_dim takes many features without
  collapsing; nn/cnn/pretrained improve up to all features; ridge breaks with trace features (keep it at P or drop it);
  cover order is not better than random.
* physics (Allan term) stays at 30-35 % sigma_e for every set: the trace term, not the features, was its problem.
* Candidate frozen choice (to agree with the user before session C is scored): sigma_e gp_dim [P+A1] (10.5) with
  physics_white [P+A] (12.3) as the physics reference; lambda pretrained [all] (27.0) / trees [all] (28.8); gain
  gp_dim [P] or trees [P+A] (0.8).

## 11. Do the traces predict anything beyond the amplifier? (labelled reference without traces)

New extra models (labelled references, `USES_LABELS`): `settings_amp` (GP on settings + amplifier one-hot, no trace)
and `settings_amp_trees` (the same with the trees learner). A+B CV: sigma_e 14.6 / 15.7, lambda 41.6 / 36.1, gain
1.7 / **0.8** %. Paired (same folds, packet bootstrap 2000), best trace models vs these:
* gain: trees [P+A] / gp_dim [P] 0.8 vs settings_amp_trees 0.8: +0.0 -> for the gain the traces act as an amplifier
  fingerprint (label-free models identify the amplifier), nothing more;
* sigma_e: gp_dim [P+A1] 10.5 vs 14.6 / 15.7: -4.1 [-6.7, -1.8] *, -5.2 [-9.4, -1.1] *; physics_white, trees,
  pretrained, nn -2 to -4 points, intervals touching 0 -> a modest, real gain from the traces;
* lambda: pretrained / trees / nn [all] 27-29 vs 41.6 / 36.1: -7 to -15 points, all * -> the traces clearly help.
Caveat: the best feature set per model was chosen on this same CV (optimistic); session C is the honest test.

## 12. FROZEN CHOICE (2026-11-03, before session C was looked at), user approved

| target | main model | alternative |
|---|---|---|
| sigma_e | gp_dim [P+A1] | physics_white [P+A] (physics reference) |
| lambda | pretrained [all] | trees [all] |
| gain | gp_dim [P] | trees [P+A] |

Tiers: `lvl1/feature_study/real_AB/evidence_tiers.json` (b7d25ca). Labels: drift subtraction off, run_WADERs --all-ext.
Then session C is scored once (`--holdout-session C`), plus every model at each feature set (best set per model and
target from the ladder, section 10) for reference.

## 13. Next packet design: traces vs level-1 vs both (user, 2026-11-03)

User: "first and foremost see if the trace can predict everything, then see if the lvl1 data can predict everything,
then see what both of them can predict together; model the new packet designs after this."

* **Level 1** (acmpy: filter level per channel, 0 raw / 1 per-skip CDS sums / 2 pixel averages / 3 off; FitsMaker
  decodes all-L1 0x55 and all-L2 0xAA; `daq.take_image(level=1, ...)`): every skip's pedestal and signal sums of every
  pixel, all 4 channels at once, ~8 kB per pixel per channel at 500 skips. Layout from
  `comm_scripts/utils/skip_noise_L1.py` (read only): LINE per row, per pixel 4*NSKIP + 2 + 13 words (skip headers/sums,
  one extra pedestal, 13-word trailer with the 62-bit sum of (signal - pedestal), EPIX). Checked on our 2026-10-06
  partial 1x1 L1 row (100x1000x3000, file cut after 80 pixels): 80/80 pixels pass every layout check; per-skip sd
  5.37 ADU (traces: 5.6), 2980-skip pixel means 0.102 ADU (white expectation 0.098). The 20x20-binned L1 images have
  another layout (not decoded).
* **Design rule:** every packet holds the same three kinds of input, so the three questions use the same packets,
  labels and folds: (a) traces (raw, as now), (b) a level-1 STRIP read exactly like the image (image's prescan
  NCOLPRE 5300, register clear BOOLCLR 1, NDCM 500, 1x1, EXPOSURE 0; NROW x NCOL small, e.g. 2 x 1000 = 16 MB per
  channel, size to be confirmed on the DAQ), each at the reference and at the new settings, (c) the image.
  Order: reference traces, reference strip, change, new traces, new strip, flush, image (traces before the strip, so
  the trace inputs are not affected by it).
* **Analysis:** three input sources per model (`trace`, `l1`, `both`), same folds; report all three; the prospective
  block is scored for each.
* Implementation steps: (1) src/decode_l1.py reader + tests; (2) strip features (image-style noise at the image's
  skips, within-pixel Allan curve, their gap, transient, spectrum, direct sigma/gain/lambda from the pixel means);
  (3) simulated L1 strips for tests; (4) manual_packet.sh strip steps + checks; (5) make_manual_manifest finds them;
  (6) process_packet features -> dataset `l1.*`, `l1ref.*`; (7) real_inputs families per source; (8) train_lvl1
  `--source trace|l1|both`; (9) v2 plan with strips, rehearsal end to end. DAQ test of strip sizes pending (user).

## 14. Session C scored (frozen choice, commit 25384b9) and the report

* Bug found and fixed while scoring: under an explicit `--feature-tiers <set>` the prospective test and LOAO refit
  every model with ALL features (run keys carry no set); now each refit uses its run's set (`run_spec`).
* Data rule added (label-free): `trace_ok` = at least one usable trace at the packet's own settings. L1_37/40/41 lost all
  40 new traces; with them in, physics_white sigma_e on C was 27.5 % (51 % on those 3 alone). Results with them kept in
  `processed/lvl1_results/2026-11-03_holdoutC_with_damaged/`; main results `2026-11-03_holdoutC/` (8 packets, 32 rows;
  `summary.csv`).
* Session C vs settings + amplifier (trees), paired packet bootstrap: lambda physics [P+A] 5.3 vs 23.2 % (-17.9
  [-25.4, -10.3] *); frozen pretrained [all] 19.7 (-3.6, n.s.); gain physics [P+A1] 0.6 vs 1.4 (*), trees [P+A] 1.1 (*),
  frozen gp_dim [P] 0.9 (n.s.); sigma_e physics_white [P+A] 7.3 vs 10.2 (-2.9 [-9.5, +2.9]), nn [P+A1] 5.9 (n.s.),
  frozen gp_dim [P+A1] 13.5 (worse). Frozen choice: sigma_e FAILED, lambda marginal, gain passed. Lesson: CV on
  one-at-a-time packets measures extrapolation (favours flexible models); C (joint changes) measures interpolation, where
  the physics-structured model generalises best.
* Report: `docs/overleaf_first_data_2026-11-03/first_data.tex` (8 pages, 0 warnings; zip
  `docs/overleaf_first_data_2026-11-03.zip`), figures by `lvl1/plot_first_data.py` (6 figures), tables generated from
  the result files.
* Level-1 implementation done (section 13 steps 1-2, 4-8): decode_l1.py + tests, l1_features.py + test, manual_packet.sh
  strips (+ check, retake), manifest (strips never the image), process_packet (l1.* / l1ref.*), real_inputs
  (model_inputs, model_inputs_l1, model_inputs_both on the same rows), train_lvl1 (strip features kept by every set;
  strip white noise for physics_white). Plumbing-tested with synthetic strips. Still to do: v2 plan text with strips,
  rehearsal with simulated strips, the DAQ strip-size test (user).

## 15. Every model explained (2026-11-04)

* Correction: the physics model's lambda and gain were shown at P+A / P+A1, sets picked after looking at session C.
  At the sets the A+B CV chose (lambda all, gain P): lambda 5.3 % (-17.9 [-25.8, -10.1] * vs settings_amp_trees),
  gain 0.9 % (-0.5 [-1.3, +0.3], n.s.). Significant gain winners on C: ridge [P+A1] 0.8, deep_kernel [P+A1] 0.7, trees
  [P+A+B] 1.1. fig1/fig2 (`PHYS` in plot_first_data.py) and the report text fixed.
* `lvl1/explain_models.py`: every model refitted once on A+B (as the session C fit) at its CV-chosen set per target,
  captured at its final predict; native parameters (GP 1/l, SAAS relevance, ridge coefficients, tree split gains,
  deep-kernel relevance, physics terms) and shuffle importance (input groups and single inputs; across all rows,
  within amplifier, session C loss). Output `processed/lvl1_results/2026-11-03_explain/explain.json` (+ run.log).
* `lvl1/report_models.py`: `models_performance.tex`, `models/<model>.tex`, fig7 (all models CV vs C), fig8 (reliance
  heatmap), fig9 (physics sigma_e term by term). Report now 34 pages: section 4 "Every model at a glance", appendix A
  "The models in detail" (`models_detail.tex`: exact math of every model).
* Findings: the gain comes from recognising the amplifier (reference-trace noise features move it 3.5 %, within an
  amplifier 0.5 %); lambda (physics family) = amplifier recognition from the reference trace (C loss 46 points) + own
  trace (18 points, also within amplifier); the packet state moves lambda 37 % but gives nothing on C; one feature
  (ref edge jitter) moves lambda 58 % (fragile). sigma_e: physics_white rests on the measured white noise (trace term
  alone 8.8 % on C, + learned gain 7.6, + correction 7.3); flexible models use timing proxies (row-start recovery,
  skips 1-4 ringing) and fail on C. settings_amp_trees uses 5 inputs (delayInt + 4 amplifier columns): the 10-row
  leaf minimum cannot split on knobs changed in 1-2 packets.

## 16. Board 102 as a test set (2026-11-04, user request; trained on 101 only)

* `lvl1/test_new_module.py`: every model fitted on board-101 A+B (as for session C), predicts board-102 rows
  (`processed/lvl1_results/2026-11-04_board102/results.json`). Board 102 rows get own packet ids (packet state per board).
* Board 102 sat at NOMINAL settings all night (its theta never changes: the packets changed acm101 only), so it tests
  only a new amplifier's level, not the settings dependence. Its trace white noise is constant (sd 0-1 %) while its
  image sigma_e labels scatter 9-13 % at nominal (101: 1-3 %).
* Channels: ch0, ch1 dead (no charge, flat). ch2, ch3 show charge but their column profiles ramp 0 -> 2.4 ADU along
  the row (not like 101); ch2's profile is the same in every packet, ch3's changes (up to 9 ADU) and its labels are
  noisier; the 2026-10-12 note says only ch2 showed charge -> ch2 taken as the working one (user does not remember).
* ch2 (41 rows): sigma_e best ~10.8 % (mean, ridge, gp) = label scatter; physics_white -30 % biased (image/trace-white
  ratio 1.06 on 102 vs 0.82 on 101). lambda 1.11 vs 101's 0.27-0.61: most models ~-90 % (predict ~0.4); gp_dim 10.3 %,
  deep_kernel 13.6 %. gain: +2 to +8 % biased (settings_amp_trees 2.0, gp_dim 2.4). After one image (first nominal
  offset removed) every model is at the label scatter: sigma_e ~11 %, lambda ~8 %, gain 0.3 %.

## 17. Amplifier fingerprints and leave-one-amplifier-out (2026-11-04)

* LOAO (`processed_AB/lvl1_results/2026-11-04_loao/`, run.sh; each model at its CV-chosen set; 4 amps x 5 packet
  folds). New-amplifier error (mean baseline: sigma_e 18.7, lambda 54.6, gain 3.5): sigma_e physics_white 14.0, sets
  15.3, flexible models 17-23; lambda every model 52-80 (no better than the mean), after removing the amplifier's
  offset (shape) physics 27.6, nn 28.9, pretrained 29.6 vs mean 37.2; gain 3.2-5.4, shape ~0.8-1.0 = mean 0.8.
* `lvl1/amp_fingerprint.py` -> `lvl1/feature_study/real_AB/amp_fingerprint.csv` (eta2 = between-amplifier variance
  share, single-feature amplifier accuracy) and exclusion lists eta90 (52 columns: 46 ref, 6 own) / eta50 (93).
  Reference-trace features are almost all fingerprints (taken at nominal); own-trace noise features are not (white
  noise eta2 0.09). `train_lvl1.py --exclude-features FILE` (excluded columns set to NaN -> dropped).
* Without fingerprints (`processed/lvl1_results/2026-11-04_no_fingerprint_eta90|eta50/`; CV, session C, LOAO for
  eta90): session C gets worse where the wins came from amplifier recognition: lambda physics 5.3 -> 15.2 / 17.6, gain
  physics 0.9 -> 2.8 / 1.8, nn lambda 13.5 -> 17.3; still better than settings_amp_trees (23.2) for lambda (cnn 13.8 and
  deep_kernel 15.4 significant). sigma_e physics_white unaffected (7.3 -> 6.9 / 7.0). New amplifier: modest gains for
  the curve models (lambda cnn 64 -> 52, pretrained 61 -> 53; sigma_e pretrained 18.9 -> 14.4, deep_kernel 19.4 -> 15.9)
  but lambda and gain levels still no better than the mean.

## 18. Without board 101 ch2 (EXT3) and the overview deck (2026-11-04)

* ch2 (WADERs EXT3): per-row lambda fit chi2/ndf 22 (others 3-4), lambda 0.61 (others 0.27-0.48), sigma_e repeat 3.1 %.
* `train_lvl1.py --exclude-amps 101:2`; predictions.csv now also holds the leave-one-amplifier-out predictions
  (`loao <model>: log <t>`). Runs (each model at its CV-chosen set; CV, session C, LOAO): `processed/lvl1_results/
  2026-11-04_all_amps`, `..._no_ch2`, `..._no_ch2_no_fingerprint` (run.sh in each). `lvl1/compare_runs.py` scores
  them on the same rows (ch0/1/3) -> `2026-11-04_no_ch2/compare.json`.
* Result (ch0/1/3): sigma_e and gain change within the noise (physics_white C 6.7 -> 6.2 -> 5.9 %). lambda on a new
  amplifier improves: physics 70 -> 49 -> 46 %, nn 55 -> 47 -> 40 % (mean 52 %): first time below the average.
  ch2's odd lambda level pulled every model (mean's C lambda 41.6 -> 34.8 %). Caveat: LOAO without ch2 trains on 2
  amplifiers.
* Overview deck (41 slides, pedagogical, with math, flow charts, results up to here):
  https://claude.ai/artifact/KrzAfKm1qzchrDqXSdwCwG ; equations and new figures by `docs/slides/make_deck_2026_11_04.py`
  into `docs/slides/deck_2026-11-04/`.

## 19. Learning curve (2026-11-04)

* `lvl1/learning_curve.py`: random subsets of 8/16/24/32 A+B packets (5 draws each; 32 = all), without 101 ch2 and
  without the eta90 fingerprints, scored on session C (`processed/lvl1_results/2026-11-04_learning_curve/`).
* physics_white: sigma_e 9.7 / 7.9 / 7.5 / 5.9 %, lambda 19.0 / 12.8 / 12.1 / 10.4 %; cnn lambda 22.6 -> 12.7 %;
  gain flattens near 1.0-1.2 % (cnn, nn). sigma_e and lambda still improving at 32 packets: more packets help.

## 20. Traces only (no settings) and the overview PDF (2026-11-04)

* `train_lvl1.py --no-settings`: theta never enters any model (also not the pre-training). Run
  `processed/lvl1_results/2026-11-04_no_ch2_no_fingerprint_no_settings/` (without ch2, fingerprints and settings).
  vs the same with settings (ch0/1/3): nothing gets worse; C: cnn sigma_e 10.0 -> 7.0, pretrained sigma_e 15.7 -> 5.2,
  physics gain 2.0 -> 0.7, cnn lambda 13.2 -> 11.8; new amplifier: cnn lambda 44.4 -> 39.7, deep_kernel 53.2 -> 42.0.
  The user wants traces-only models (they learn from the trace, not the parameters); settings were an input only by
  the 2026-10 design. Proposed: make --no-settings the default for trace models (baselines keep theta).
* Overview as a Beamer PDF (44 frames, real LaTeX math, TikZ flow charts): `docs/overleaf_overview_2026-11-04/`
  (overview.tex, no_settings.tex, figures/; zip `docs/overleaf_overview_2026-11-04.zip`).

## 21. CNN encoder inside the physics model; assessment (2026-11-04)

* `train_lvl1.py` models `physics_cnn` (physics_white + the pre-trained encoder's embedding of own and reference curves
  and their difference as extra GP inputs) and `physics_cnn_only` (embedding only). Run
  `processed/lvl1_results/2026-11-04_physics_cnn/` (no ch2, no fingerprints, no settings). No gain: sigma_e C 6.1 vs
  6.3, lambda C 10.4 vs 13.7 (n.s.), gain 0.7 vs 1.3 (significantly worse); embedding alone worse for lambda (26.9).
* Settings check (paired bootstrap): dropping theta is neutral or better for physics and the networks, worse for
  gp / gp_dim on new amplifiers. Document corrected.
* `docs/overleaf_overview_2026-11-04/level1_summary.tex` (11 pages) now has the CNN subsection and section 8, an
  assessment: what works, what does not, why (information, not models), prognosis, decisions for v2.

## 22. Predicting the CHANGE between nominal and a new setting (2026-11-04)

* User's reframing: from two traces (reference at nominal, own at the new setting) predict the change of sigma_e,
  lambda, gain. `lvl1/change_test.py`: target log y(new) - log y(nominal), nominal interpolated in time between the
  nominal packets of the same amplifier; inputs diff.* (own - reference, P+A tiers); board 101 without ch2; results in
  `processed/lvl1_results/2026-11-04_change_test/results.json`.
* New amplifier (true change RMS: sigma_e 18.5 %, lambda 36.8 %, gain 0.9 %): sigma_e change trees 12.6 % (direction of
  clear changes right 94 %); lambda nn 30.9 % (direction 88 %); gain at the label noise (0.9 %; direction 64-82 %).
  Settings-change GP about as good (sigma_e 13.5, lambda 33.6). Only 84 A+B rows.

## 23. Change prediction follow-ups (2026-11-04)

* `lvl1/change_test2.py`: label noise of a change (nominal to nominal): sigma_e 1.9 %, lambda 11.9 %, gain 0.8 %;
  median true change sigma_e 3.9 %, lambda 8.5 %, gain 0.4 % -> lambda and gain changes mostly within label noise.
  VDD / VR packets: the same knob change gave opposite gain changes (L1_15 +1.2 %, L1_31 -2 %), common to the 4
  amplifiers of a packet (60 % of the variance): a shared per-packet effect, not VDD / VR; gain follows delayInt in C.
  Lasso / PLS blow up on outliers; trees, GP, SAAS best; physics difference model best median sigma_e error (3.8 %).
* `lvl1/change_pairs.py`: all pairs of packets of one amplifier (both labels measured), new-amplifier test.
* change_pairs (new amplifier, direction of clear changes): all pairs -- settings GP sigma_e 68 % / lambda 66 % / gain
  80 %, trace trees 50 / 75 / 82 % (trace GP blew up); pairs within 1 h (405 pairs, 54 test) -- trace trees lambda 88 %
  (25 cases), sigma_e 59 %; settings lambda 76 %. Within a packet (change_test) sigma_e 94 %, lambda 88 %. Fixed: the
  time conversion assumed nanoseconds (pandas gives microseconds); change_test's interpolation is unaffected (scale-free).

## 24. Change models (2026-11-04): Siamese, direction losses, difference curves

* `lvl1/change_models.py` (results `processed/lvl1_results/2026-11-04_change_models/`). New amplifier, within-packet
  changes (+ cross-packet pairs within 1 h for training). Best size is still "feature difference: trees" (sigma_e
  12.6 %, dir 94 %) and settings GP (lambda 33.6 %). Siamese (shared encoder) on features or pre-trained curve
  embedding: no better; direction-only loss gets lambda direction 94 % (features) but meaningless sizes. Difference
  curves (PCA 10): trees sigma_e 14.5 % dir 94 %, lambda 36.2 % dir 76 %; intercept-free ridge unstable on new amps but
  best on C (sigma_e 6.9 %). 84 rows: the fancier models overfit; v2 data needed.
* Difference-curve correlation map (difference_curve_correlation.png): d sigma_e tracks the change of the mean skip
  shape (|r| 0.74 at point 61), within-cycle structure function and spectrum, noise vs window length; d lambda tracks
  the within-cycle spectrum (0.60), CDS spectrum (log f), mean skip shape, CDS autocorrelation. Candidates for
  hand-made difference features.
* `lvl1/change_handpicked.py`: 6 bands per target of the difference curves, chosen in each training fold, + the change of
  the white noise. GP: new amplifier sigma_e 14.5 % (dir 94 %), lambda 35.3 % (dir 82 %); session C sigma_e 6.6 %
  (best trace-based; settings GP 7.5, no change 11.9), lambda 24.1 %. Huber blows up on new amplifiers. Most often
  picked: sigma_e noise-vs-window-length (short windows), within-cycle spectrum; lambda mean skip shape (point 111),
  within-cycle spectrum, equivalent-time pulse; gain the variation along the cycle.

## 25. v2 training plan document; neighbouring-packet check (2026-11-04)

* `docs/overleaf_v2_training_plan_2026-11-04/v2_training_plan.tex` (10 pages; zip next to it): questions Q1-Q5, what
  changes from v1 and why, the v2 packet and its data products, pre-night checklist (strip test, pull main, settings on
  every working board, voltage windows, temperature, ch2), processing, tasks A (absolute, three sources), B (change
  between two traces: within-packet strip labels, image labels, neighbouring pairs; models and losses), C (transfer),
  D (diagnostics), expected outcomes and decisions, risks, commands.
* `change_pairs.py --max-step K --block-folds` (contiguous time folds keep neighbours together). Neighbours (n+-1, 81
  test pairs): trace trees lambda direction 91 % (44 clear), trace GP sigma_e 75 %, gain 81 %; settings GP lambda
  82 %, sigma_e 53 %; sizes ~ no change (lambda RMS dominated by L1_18). n+-2: lambda 86 %, sigma_e 71 %.

## 26. Four-way scorecard and ranking test (2026-11-04)

* `lvl1/scorecard.py` (results `processed/lvl1_results/2026-11-04_scorecard/`, copy of the report in
  `docs/scorecard_2026-11-04/`; `--rescore` re-scores the saved predictions). Board 101 ch0/1/3, 40 packets with traces,
  5 contiguous time blocks; trained amplifiers vs new amplifier (LOAO x blocks); absolute vs relative (pairs within a
  packet, n -> n+1, n -> n+2; training pairs pooled, both orders); trace models never see settings; each model vs its
  cell's settings-only reference (settings + amplifier GP when amplifiers are trained, settings GP for a new one),
  packet bootstrap 95 %. Ranking (5): per amplifier, shortlist 5 / 10 of its ~35 settings, regret vs the true best.
* CORRECTION: 96 % of the clear within-packet sigma_e changes go UP (nominal is near the best setting; lambda 76 %),
  so "always up" scores 96 %: the earlier "sigma_e direction 94 % within a packet" (change_test.py, trained in one
  order) was mostly this base rate. With both orders in training: trace trees 76 % (trained) / 65 % (new amp) vs
  settings 33-45 %. Neighbouring-packet pairs are balanced (~50 % up) and are the fair direction test.
* (1) absolute, trained amps: trace models beat settings + amplifier for lambda (28-29 vs 37 %) and gain (0.9-1.0 vs
  1.4 %, but the amplifier mean is 1.0); sigma_e same (10-11 vs 10.7 %).
* (2) absolute, new amp: nothing reliably beats settings GP (15.7 / 49 / 3.5 %); physics sigma_e 14.7 (n.s.), cnn
  lambda 41 % (better), deep kernel lambda 90 % (worse).
* (3)/(4) relative, n -> n+1 (balanced): trained amps trace trees 78 / 89 / 86 % (sigma_e / lambda / gain) vs settings
  + amp GP 72 / 79 / 78; new amp trace trees 72 / 82 / 89 vs settings GP 72 / 82 / 69; absolute pretrained cnn
  differenced 92 / 89 / 83 (trained), 86 / 80 / 72 (new). None significant at 95 % (about 50 clear changes each).
  Sizes: sigma_e 10-13 % vs 15 % for "no change"; lambda ~ no change.
* (5) ranking: noisy (3 amps x 35 settings). Lambda: trace-change trees put the true best in the top 10 for all 3
  amplifiers (trained and new amp), top-5 regret 3 / 9 % vs 20 % random; settings-change GP also 3 / 3. Sigma_e: spreads
  among the good settings are small (random top-5 regret 4 %); best models ~2 %. Combined sigma_e + lambda, new amp:
  absolute trace trees regret 0.3 %, settings GP 0.0 %, random 12.5 %.

## 27. v2 plan revised after the scorecard (2026-11-04)

* `daq_scripts/manual_packet.sh`: new order: [reference settings] -> flush -> reference strip -> reference traces ->
  flush -> change -> settling -> new traces (N_SETS) -> new strip -> flush -> image -> standalone -> post-image
  traces (`<id>_post<set>`, N_POST_SETS) -> flush -> reset. `REF_CHANGES_FILE` = reference at those settings (own
  settling traces `<id>_settleref`, then a reset to nominal before the new settings). `SCREEN=1` = trace-only packet.
  Record v2 gains ref_changes, screen, trace_prefix_post, n_post_sets.
* `src/make_manual_manifest.py`: `post_traces` (settings checked against the new traces), `reference`
  (nominal / settings), status `screen` for screens. `src/process_packet.py`: kind "post", exported as `post.*`.
* `daq_scripts/run_plan.sh`: passes `<id>_ref.txt`, SCREEN=1 with the plan's screen sets, N_POST_SETS; stops at a
  shortlist packet whose changes file is missing.
* `lvl1/make_packet_plan_v2.py` (D and E unchanged): N_SETS 3 + N_POST_SETS 2; 21 D packets with the reference at
  the previous packet's settings; optional session F (L2_77..L2_136, 5.4 h): nominal, 48 screens, 5 SHORTLIST (filled
  by lvl1/funnel_shortlist.py, not yet written) + 5 random CONTROL screen settings in random order, nominal.
* `docs/overleaf_v2_training_plan_2026-11-04/` (14 pages): base-rate correction, new rows in the changes table, new
  sequence, scorecard section (8) with the v1 results, session F section (9: objective = Pareto front then
  dlog sigma_e + dlog lambda; gain enters via sigma_e in electrons), outcomes, risks, commands.
* scorecard ranking: hybrid (sigma_e from absolute trace trees, lambda from change trees) did not beat absolute trace
  trees on sigma_e + lambda; the 3 amplifiers share packets, so ranking results are ~1 trial.

## 28. Weighted joint objective; session F part of the plan (2026-11-04)

* `lvl1/shortlist.py`: J(w) = w dlog sigma_e + (1-w) dlog lambda (w = sigma_e weight, 0..1); rules "weighted" and
  "pareto" (predicted Pareto front first). `lvl1/scorecard.py --weights ...` (default 1 .75 .5 .25 0): ranking now per
  weight and rule, settings scored as the mean over the 3 amplifiers (a setting applies to the board).
  First night (regret of the best of 5 shortlisted, %): random 2.9 / 3.1 / 6.3 / 8.0 / 21 for w = 1 .. 0; trace change
  trees (new amp) 2.8 / 3.2 / 6.1 / 6.5 / 0.0; absolute trace trees 2.9 / 0 / 0 / 1.2 / 16. sigma_e-only shortlisting
  gains little (good settings within ~3 %); lambda-heavy gains most. Pareto-first loses at the extremes -> default
  weighted. Each column ~ one trial.
* `lvl1/funnel_shortlist.py`: trains change trees (within-packet + n+1/n+2 pairs, both orders; reference label =
  interpolated nominal or the packet at the reference settings) and absolute trees (f(new) - f(ref)); reads screens'
  features.json; board mean over amplifiers; `--weight w ...` share the shortlist slots in turn; skips screens that are
  random controls; writes SHORTLIST changes files + shortlist.json; `--evaluate` on labelled screens (first-night last
  block: best caught for w = 1, .5, 0, but 5 of 7 shortlisted: a code check only).
* Session F is now part of the plan (plan.json "funnel": weights [1, 0.5, 0], rule weighted); PLAN.md has the shortlist
  step; run_plan.sh's stop message names it. Plan document (15 pages): section 9 objective J(w), weight table, scores
  per weight.

## 29. ch2 in the relative models; combined change models; every kind of change (2026-11-04)

* `scorecard.py --channels --abs-models --rel-models`; run with ch0-3 in `processed/lvl1_results/2026-11-04_scorecard_with_ch2/`;
  `lvl1/compare_amp_sets.py` -> `docs/scorecard_2026-11-04/ch2_comparison.md`. ch2 as a new amplifier: absolute
  models are off by -54..-100 % in lambda (its offset), change models by -7..0 %; change trees n->n+1 direction ch2
  76 / 90 / 82 % (sigma_e / lambda / gain) vs others 68 / 89 / 92. ch2 in training: change models unchanged within
  the intervals (shared-encoder net worse in lambda size, +22..29 %); absolute models' new-amp lambda worse (physics
  +24, trees +11, pretrained +5 %, significant). -> ch2 usable for change models, not for absolute ones.
* `lvl1/relative_extra.py` -> `docs/scorecard_2026-11-04/relative_extra.md`: combined trees on [A, B, B-A] (both
  orders) best change model on a new amplifier, between packets 75 / 89 / 88 % vs change trees 70 / 88 / 86 (n.s.);
  hand-picked bands add nothing; [A, B-A] without B worse on new amps (lambda size 53 %). Within a packet [A, B, B-A]
  gets sigma_e 90 % (it can recognise the nominal trace: partly the base rate).
* By kind of change (new amp): timing changes work (nominal vs timing lambda 96-100 %, sigma_e 64-88 %; timing vs
  timing sigma_e 80-88 %, lambda 88-90 %); output stage: gain 100 % (9-19 clear), sigma_e/lambda changes too small;
  clock rails, VSUB: almost no clear changes in v1 (within one max_step); two different kinds (e.g. delayInt vs VR,
  48 pairs): lambda 90 %, gain 100 % (5 clear), sigma_e only 44 % (17 clear, small changes ~5-10 %).
* `docs/context/figures/subtracted_traces_example.png`: raw B - A (clock pulses out of phase, noise invisible) vs the
  difference curves the models use (L1_04 delayInt, L1_16 VR). `process_packet.py --packets-dir` now also processes
  status "screen".

* Figures: docs/context/figures/scorecard_absolute.png, scorecard_relative.png, subtracted_traces_example.png.

## 30. Funnel: two rankings side by side, sorting pairwise comparisons (2026-11-04)

* User: the funnel must sort ~100 traces by comparing two at a time (an ML comparator + a sorting algorithm); run
  WITH-reference and WITHOUT-reference rankings side by side on the same screens (screens keep 1 reference set).
* `lvl1/shortlist.py`: SORTS mean change (least squares of the differences), wins (round robin), Bradley-Terry (MM fit
  of P(i beats j) = p_i / (p_i + p_j)); applied to the comparisons of J(w).
* `lvl1/rank_methods.py` (first night, 35 settings as screens, pairs compared by models trained without both blocks;
  change model = trees on [A, B, B-A] trained on within-packet + all drift-corrected pairs): Spearman with the true
  setting effect, new amplifier, w = 1 / 0.5 / 0: with reference change model 0.77 / 0.51 / 0.53; one-trace difference
  0.74 / 0.56 / 0.72; without reference one trace 0.73 / 0.56 / 0.69; screen vs screen mean change 0.69 / 0.66 / 0.74,
  wins 0.75 / 0.77 / 0.83, Bradley-Terry 0.76 / 0.76 / 0.83 (trained amps similar). -> without reference >= with;
  direction-only sorts best where lambda counts. Default sort: bradley-terry.
* `lvl1/funnel_shortlist.py` rewritten: change model [A, B, B-A] on within + random drift-corrected pairs; one-trace
  model -> setting effect; rankings "with reference" and "without reference" (+ --sort); slots go round weights x
  rankings; shortlist.json has both full rankings and scores.
* Plan: SHORTLIST 6 (2 rankings x 3 weights) + CONTROLS 5 (controls unchanged); F 61 packets 5.7 h, total 137 / 27.1 h;
  plan.json funnel {weights [1, .5, 0], sort bradley-terry, methods}. manual_packet.sh accepts N_REF_SETS=0 (screens
  without references possible; not the default). Plan document section 9 updated.
* Running: lvl1/pairs_extra.py (random drift-corrected pairs vs neighbours for training; shared references).

## 31. Funnel simplified: no references, lambda first, noise attached (2026-11-04)

* User decisions: (1) no one-trace ("Way 1") model in the ranking; (2) first round ranks on lambda (DC) only, the
  predicted noise is only attached to each pick afterwards; (3) screens WITHOUT reference traces: the question is
  "from ~100 traces, can a shortlist of the best settings be predicted?".
* `lvl1/funnel_shortlist.py`: METHODS = without reference only (screen vs screen, Bradley-Terry default); weight
  default 0 (lambda); noise model = trees on own features + amplifier one-hot -> log sigma_e (CV ~11 %, 14 % on the
  mock screens), attached per amplifier in the output, the changes-file comments and shortlist.json; --pairs random
  (default) | neighbours.
* `lvl1/pairs_extra.py` (fast trees now): direction of single pairs, new amp, [A, B, B-A]: neighbours sigma_e 80 /
  lambda 90 / gain 89 %, random drift-corrected pairs 84 / 85 / 72 %. `rank_methods.py --pairs neighbours`: Bradley-Terry
  lambda ranking 0.81 (new) / 0.73 (trained) vs random 0.83 / 0.78 -> random pairs the default for the funnel.
* Plan: 100 screens, SCREEN_REF_SETS 0, SHORTLIST 5 + CONTROLS 5; F 112 packets 6.9 h; total 188 packets 28.3 h;
  plan.json funnel {weights [0], sort bradley-terry, pairs random, methods [without reference]}. Plan document
  section 9 rewritten. The flow chart in the chat (48 screens, references) is now out of date.
