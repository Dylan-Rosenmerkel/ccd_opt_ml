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
