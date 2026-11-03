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
tier rule as before. Results: (pending).
