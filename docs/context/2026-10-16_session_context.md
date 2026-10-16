# Session context: 2026-10-16

Continuation of `2026-10-15_session_context.md` (learning guide, level 1 built, model tutorials, A+B as a labelled
reference).

## 1. Requests

* "What do we need to work on now since we are waiting for new data?" → I proposed (1) a level-1 rehearsal with a
  simulated image of the real geometry, (2) a level-1 packet plan, (3) a first-packet checklist, (4) optional level-2 prep.
* User: **work on 1 and 2**, document everything properly, and **read `~/Documents/Software/comm_scripts`** (the user's new
  folder) and include it in the day's context documents. (comm_scripts was only READ; nothing outside ccd_opt_ml was
  changed.)

## 2. comm_scripts (`~/Documents/Software/comm_scripts`, the user's repository, commit 3ad4026)

Purpose: run panaSKImg (WADERs) on skipper-CCD images **without hand-written JSONs**. Its own hand-over notes are
`comm_scripts/docs/CONTEXT_2026-10-16.md` (read first when working on it); its rules: show proposed edits and get approval
first, do not modify pysimdamicm unless asked (one local patch exists: `x_max_plot`), never change how "normal" images are
fitted without saying so, regression-test autoconfig changes against ~32 images (146 JSONs).

Layout and tools:
* `bin/run_WADERs INPUT [-o DIR] [-j N] [--no-clusters] [--tune] [--gain-correction] [--autoconfig-args="..."]`:
  image or list → per-CCD JSONs (`gain_dc_split.py --per-ext`, via `autoconfig_panaSKImg.py`) → panaSKImg per CCD in
  parallel → `WADERs_analysis_<name>/` with `summary.txt`, **`fit_quality.csv`** (per CCD: chi2/ndf, gain, noise [e-],
  DC [e-/pix], errors scaled by sqrt(chi2/ndf), DC bias vs lam01, gain_held), `json/<image>/final/`, `root/`, `logs/`.
  Finds the conda env `damic` itself (`--env`, `$WADERS_ENV`).
* `src/autoconfig_panaSKImg.py`: measures each CCD from the raw pixels: **geometry from the header**
  (`ccd_cols = PHYSCOL - NCOLPRE + PRESCAN`, unbinned), dead channels (`is_dead_channel`: < `--dead-frac` 2e-4 of
  pixels clearly charged + same noise as overscan + no offset; `--keep-dead` disables), gain (comb), noise, DC per row,
  pedestal method (overscan median by default), fits only the Poisson-like peaks (`--peaks auto`, usually 0e-/1e-: unbiased
  DC; more peaks bias DC by +5-25 %), `--tune`, a fallback for unresolved CCDs (10000-skip images). Reference JSON built
  in (`id_row_start` 2, i.e. the first 2 rows are left out).
* `src/gain_dc_split.py` (gain from an all-peaks fit where more precise, DC from 0e-/1e-), `src/fit_quality.py`,
  `src/gain_vs_charge.py` (g(n): gain falls ~3 % by ~1000 e-/pixel; `--gain-correction` fixes Fe-55 K-alpha 0.9835 →
  0.9977 of true over 82 CCDs and Am-241 lines), `utils/` (DC-fit PDF tiling vs skips, splitting cluster files).
* Findings recorded there that matter here: the **skip-noise run** (`~/Documents/JH1_Testing/2026-10-14-PA03_PQ04_Skip_Noise/`,
  boards 101/102, 1 → 10000 skips): 5000 skips gave the best noise (EXT1 0.066 e-); the 10000-skip image was ~3x noisier
  (conditions changed, 4.6 h readout); ACM 102 mostly dead channels (EXT3, partly EXT2/EXT4 usable) — consistent with not
  using module 102 for trace analysis. Gains of module 101 from the 10-06 image in its notes: EXT1 1.3353, EXT2 1.3096,
  EXT3 1.2502, EXT4 1.2605 (`--gain` example).
* Open items there (not ours): PATH line in `~/.bashrc` still needs `/bin`; g(n) coverage ~80 %; panaSKImg issues
  (SetParameters 6-of-7, `.fits.bz2` names) noted, not fixed.

**How ccd_opt_ml now uses it:** `process_packet.py` can call `run_WADERs` (`[waders] runner = "run_WADERs"`) and reads
the labels from `fit_quality.csv` (`labels_from_run_waders`; flags: CCD not fitted, error > 50 % of value,
chi2/ndf > 3). Since the rehearsal (section 3) this is the DEFAULT in `configs/processing.toml`, with
`--no-clusters --autoconfig-args=--keep-dead`.

## 3. Task 1: level-1 rehearsal with the planned image geometry

New: `lvl1/simulate_manual_run.py` (hand-taken packets with records, traces via `simulate_trace.py` — new `noise_sd`
argument — and 16 x 1000 x 500 images, NCOLPRE 5300, 852 active columns, pedestal ~31 ADU, a few large events, a known
truth: gains 1.31/1.30/1.28/1.25, sigma_1 5.8/5.8/23/5.9 e- at delayInt 1000 ∝ sqrt(1000/delayInt), floor
0.08 (delayInt/1000)^0.3, DC 0.010 exp((VSUB-45)/25), CIC 0.008 exp(-2 ΔH), λ_r = CIC + DC x_r), `lvl1/check_rehearsal.py`
(labels vs truth, pulls), `lvl1/rehearse_lvl1.py` (the whole chain for both WADERs routes → `REHEARSAL.md`).

Found and fixed / decided:
* **The configured WADERs JSON (`img_9.json`) does not fit the planned image** (100 rows, 854 columns, skips 2 rows).
  With the geometry overridden (ccd_rows 16, ccd_cols 852, id_row_start 0) panaSKImg still read, over 6 packets (17 labels):
  sigma_e -8 %, λ -12 %, gain +11 % (median). **run_WADERs**: sigma_e +1.6 %, λ +4 %, gain 0 % (18 labels; RMS 3.5 / 10 / 3 %).
  → `run_WADERs` is now the default route.
* **run_WADERs' dead-channel test dropped a working channel** on a 16-row image with no exposure (too few clearly charged
  pixels: 8.4e-5 < 2e-4). → `--autoconfig-args=--keep-dead` in our config. (Must be passed with `=`, else its argparse
  takes `--keep-dead` as its own option.) Worth telling the user as a comm_scripts observation; not changed there.
* sigma_e error bars from run_WADERs are sometimes too small (pulls up to ~5-7 on single images) while the median is
  within 2 %: the labels' quoted errors should not be trusted below the ~3 % level.
* **Dark current / CIC from 16 rows at λ ≈ 0.02-0.03 e-/pixel: only ~20-40 % each** (pulls ~1, so the errors are honest).
  The joint DC/CIC model can learn little from such images; total λ is fine.
* The noisy amplifier (ch2, sigma_e 0.6-1.5 e-) is correctly not fitted / flagged by both routes.
* Per-row λ now runs for both routes and takes the active columns from the FITS header (with the .meta on top).

## 4. Task 2: level-1 packet plan

* `daq_scripts/manual_packet.sh`: `CHANGES_FILE=<file>` (one NAME=VALUE or acm101:NAME=VALUE per line, # comments);
  **timing parameters** are now known names (reference_v5_pumping.seq values), checked against the limits.toml timing
  windows and required to be whole cycles; the record stores `changes_file`. Tested: bad window, fractional cycles and
  TG HIGH<LOW are refused.
* `lvl1/make_packet_plan.py` → `lvl1/packet_plan/` (`PLAN.md`, `plan.json`, `L1_NN.txt`): **41 packets, ~11.3 h**:
  * A (13, ~4 h): timing only — nominal ×3, delayInt 300/500/700/1500/2000/3000, delay_H 900/3600, delayRGH 400/1600;
    can be taken now;
  * B (~17): one voltage group at a time (VDD, VR, RG_H, RG_L, SW_L, H lows, V lows, TG, VSUB), one max_step either way
    (refused by the rules: SW_L +0.5 V, V lows −0.5 V);
  * C (11): joint Latin hypercube in the same box + a final nominal.
  Rules: every configuration passes ccd_limits; every voltage within ONE max_step of nominal (no ramps needed: manual
  packets return to nominal by re-reading the .bcf); changes on board 101 only (`acm101:`); both stages' pins move
  together. B and C use the PLACEHOLDER voltage windows (reviewed = false) and are provisional. All 41 files pass
  manual_packet.sh's own dry-run checks.

## 5. Open (for the user)

* Confirm the voltage windows (then `reviewed = true`); sessions B and C wait for it.
* comm_scripts observations to pass on (their decision): dead-channel test on small images; sigma_e error bars small.
* Copy `~/packet_records` with the data after each session.
* The first real packet decides: cdaq file naming with the PACKET_ID prefix, run_WADERs on the real 16-row image, row
  order (DC > 0).

## 6. First real 16-row images (user: "use these images to check how WADERs performs and if they will be sufficient")

Data: `data/2026-10-16-PA03_PQ04_16rows` — 16 x 1000 x 500, 1x1, NCOLPRE 5300, **EXPOSURE 10 s** (not 0), all at
nominal, ~6 min apart (4:57 readout each); board 101: 3 images (its first image of the series is missing; run_ACM101.dat
lists 3), board 102: 4. No traces. `var_*` files hold negative values (unit unclear; not used). Check script:
`lvl1/check_real_images.py`; report: `lvl1/real_image_checks/2026-10-16_PA03_PQ04_16rows.md`.

Findings:
* **Row order confirmed**: λ rises from the first to the last rows on every channel (ch0 0.077 → 0.170 e-/pix) → FITS
  row 0 = first read; `[rows] reverse = false` is right. DC and CIC (intercept, incl. the 10 s exposure) are each
  ~0.06-0.09 e-/pix, formally ±0.005, but the DC repeats only to ~20 % (e.g. ch0 0.066/0.063/0.091).
* **Along-row baseline trend**: the active region starts 0.33-0.45 ADU below the row's overscan pedestal and recovers
  over the row (to −0.05), the same in every image and in ch0/1/3 (smaller in ch2). Probably the baseline recovering after
  the fast skip of the 5300 prescan columns. A per-row pedestal leaves it in: the 0e- peak is 15-25 % wider than the
  overscan noise, chi2/ndf 2.4-5.1, and run_WADERs' sigma_e is ~12 % high (0.190 vs 0.166 e- for ch0) and scatters 4-6 %
  between repeats (quoted 1.7 %); EXT4's DC disagreed with the 0e-/1e- counts by −0.3 to −0.4.
* **Removing the trend** (per column, median of the 0e- pixels over the 16 rows, smoothed over 25 columns, overscan
  untouched) before run_WADERs: chi2/ndf 1.2-2.1 (one 3.5), sigma_e repeat scatter ch1 0.7 %, ch3 0.5 %, ch0 2.9 %;
  gain 0.1-1.3 %; λ 1.6-5.9 %; DC bias ~0. Labels: σe ch0 0.166, ch1 0.171, ch3 0.214, ch2 0.251 e-; gain ch0 ~1.34,
  ch1 1.33, ch3 1.276; λ 0.13-0.22 e-/pix (10 s exposure + readout).
* **ch2** (CCD_A): peaks resolved in 1 of 3 images only → ~3 usable amplifiers per packet (100 packets ≈ 300 examples).
* **Dead-channel test**: without --keep-dead, run_WADERs kept ch0/1/3 in the raw images but dropped two working channels
  of one image after the trend correction (it relies on muon pixels > ~5 e-, rare in 16 rows) → keep --keep-dead.
  Board 102: EXT1/EXT4 correctly dead, EXT3 flooded (λ ~1.7) — not usable, as known.

Verdict given to the user: the images are sufficient for training (label precision after the trend correction:
σe ≲ 1-3 %, gain ≲ 1 %, λ 2-6 %; session A changes σe by tens of %), **provided the along-row trend is removed before
WADERs** (otherwise σe labels carry a ~12 % offset and 4-6 % scatter that no trace measures). Proposed (not yet done,
awaiting the user): a `baseline_trend` step in process_packet before WADERs and the per-row fit; tell comm_scripts.

## 7. User: "is there anything in the header/bcf/config that tells you something is wrong?"

Yes (details in `lvl1/real_image_checks/2026-10-16_PA03_PQ04_16rows.md`, last section):
* **H3_A_L = 6.2 V** in the loaded configuration of BOTH boards (bcf hashes 478f… / f008…; 478f… is a 10-12 test
  configuration) — the nominal .bcf files on the DAQ probably still hold a 10-12 change; our nominal (limits.toml,
  manual_packet.sh, packet plan) says 6.7 V. To confirm with the user.
* **Start-of-pixel transient on board 101** (from `mb_L1_*.pkl.bz2` = raw skips of 15 pixels per channel, shape 15 x 500;
  `mb_L0` empty): skip 0 ≈ +290 ADU, ringing over ~5 skips; adds 0.35-0.53 ADU to each 500-skip average; board 102
  has none. Same size as the along-row trend; 3000-skip 10-06 image shows the trend ~5x weaker (1/NDCM dilution). Likely
  cause of the trend (not proven: the 15 pixels' row position is unknown). Fix candidate at acquisition: **NDCMPRE**
  (sequencer `FakeCDS` = clocking without sampling before the real skips; currently 0 in images and traces; limits
  0-10). Suggested test: one image with NDCMPRE = 10, one with H3_A_L = 6.7.
* EXPOSURE 10 s; board 101 missing its first image (run_ACM101.dat); sequencer differs from 10-12 only in geometry and
  pocket pumping (PP_NCYCLES 50000, PP_SHIFT 3); readbacks/firmware normal; pixel [0,0] bad.
* Single-skip noise from the saved skips (skips 10-499): board 101 ch0 5.8, ch1 5.8-6.0, ch3 6.4-6.8, ch2 6.8-9.0 ADU;
  board 102 4.5-5.1 ADU.

## 8. User: "is this caused by the image or by the WADERs scripts? are my new scripts helpful?"

Ran the old hand-JSON panaSKImg route on the same 3 board-101 images (raw and trend-removed). Answer given:
* **Caused by the images** (measured on the pixels without fits; raw skips show the transient; both WADERs routes read σe
  ~12 % high on raw images and agree after trend removal; scales with 1/NDCM).
* **comm_scripts/run_WADERs is clearly helpful**: hand JSON locked the gain at ~2x (2e- peak as 1e-) in 4 of 18 fits with no
  flag (σe halved, λ ~10x low); run_WADERs 0 of 18; header geometry; diagnostics (χ²/ndf, scaled errors, DC bias vs 0e-/1e-
  counts) flagged the problem; rehearsal 0-4 % vs 8-12 %. Limits: dead-channel test on 16 rows, quoted errors < repeat
  scatter on raw images, reference id_row_start 2, cannot fix an along-row drift (needs trend removal or NDCMPRE).

## 9. Follow-up images 130 / 131 (user fixed H3_A_L in the .bcf files to 6.7; 131 also NDCMPRE = 5)

* H3_A_L not the cause of the drift or transient. NDCMPRE = 5 removes the start-of-pixel transient (keep it, also for
  traces). **The along-row drift is unchanged → the transient is NOT its cause (my earlier "likely cause" was wrong).**
* Remaining candidate: analog FILTER (10-06: 85 = code 1, no drift; 10-12/10-16: 170 = code 2). Suggested test image with
  the 10-06 filter setting. Software trend removal still works (χ²/ndf 1.2-2.1; 131 best 1.16-1.66).
* ch3's raw DC bias vs 0e-/1e- counts: −0.3..−0.43 at H3_A_L 6.2, −0.09/−0.02 at 6.7.

## 10. User: what does the drift removal do, and how does it affect images without the effect?

Explained `column_trend` (per-row overscan pedestal, zero-electron pixels within ±0.6 ADU, per-column median over rows,
25-column running median, overscan untouched, one curve subtracted from every row). Tested on drift-free images: 6
simulated (labels vs truth unchanged: σe +1.6 → +1.5 %, λ RMS 10.0 → 8.6 %, gain 0.0 → 0.2 %, χ²/ndf 0.92 → 0.96;
removes 0.02-0.04 ADU rms) and real 10-06 (0.02-0.07 ADU rms; up to 0.5 ADU only in columns 0-2). To fix before adopting:
noise-scaled window, first columns, minimum zero-electron pixels per column.

## 11. User: what causes the drift and can it be fixed? (skip-noise run, ~/Documents/JH1_Testing/2026-10-14-PA03_PQ04_Skip_Noise)

* **Correction of my earlier claims**: "10-06 has no drift" and "10-06 had FILTER 85" were WRONG (FILTER 85 came from a
  different 10-06 file, the L1 image 161012_2; the 10-06 avg image 132140_1 has FILTER 170 and an unresolved zero peak,
  σ_ovs ~1.8 ADU, so it cannot show the drift). No evidence about the filter.
* Skip-noise run (all FILTER 170, H3_A_L 6.2, NCOLPRE 5300/5800, BOOLCLR 1): drift present since at least 10-14 at 100,
  500, 1000, 2000, 5000 skips (where the zero peak is resolved), 0.2–0.7 ADU; recovers over ~300–800 COLUMNS regardless of
  skips (so per pixel read, not per second); same in first and last rows; board 102 channels WITHOUT a CCD show none →
  not generic electronics/clocking; tied to board 101 + PA03. 5800-image dips at its own start (weak evidence for a
  row-start effect).
* Proposed tests: BOOLCLR 0; NCOLPRE 5000; (board swap if practical). Software trend removal remains the working fix.

## 12. Survey of ~/Documents/JH1_Testing (user: "see if these data help"; "don't pay too much attention to DR03")

`lvl1/drift_survey.py` (new) over all run folders; table `lvl1/real_image_checks/drift_survey_JH1_Testing_2026-10-16.csv`.
* PA03 board 101 ch0 had the dip already on 2026-09-22/24 (500 skips, NCOLPRE 2800, EXPOSURE 1): −0.1..−0.25 ADU at the
  first columns → present since September; at the readout start wherever that is (col 2800 then, 5300 now) → row-start
  effect, not a CCD location.
* AMT module 106 (other setup, same recipe NCOLPRE 2800, BOOLCLR 1, 500 skips): no dip (a flat +0.2 ADU active–overscan
  step instead, different effect). With board 102's CCD-less channels: the recipe/electronics alone don't make it; it
  belongs to PA03 + board 101.
* Size: Sept −0.1..−0.25 (EXP 1, NCOLPRE 2800), Oct −0.3..−0.7 (EXP 10, NCOLPRE 5300); shrinks with skips but not 1/NDCM.
* Proposed tests: EXPOSURE 0/1; NCOLPRE 2800; BOOLCLR 0.

## 13. User: "I am concerned that this software drift fix is unphysical"

Tests (5 board-101 images, two-Gaussian fits per column block): the gain (1e- − 0e- spacing) is constant along the row →
the drift is a rigid baseline shift (so a baseline subtraction is legitimate in principle). But: the shift at the row start
is −0.9..−1.0 ADU (my ±0.6 ADU window clipped it to −0.45: under-corrected), −0.1..−0.15 ADU even at the row end, and
the noise is +30 % in the first 25 columns (+12 % to col 100) — not removable by an offset. **Cutting the first 200 columns**
without any subtraction: χ²/ndf 0.8–1.3, σe 0.163/0.167/0.206 e- with 1.0–2.6 % repeat scatter (best); cut + trend
0.156/0.161/0.200 (residual slow offset ~4 % on σe). λ depends on the columns used (later columns higher: serial-register
wait). Recommendation: acquisition fix if found; else cut 200 columns (physical), keep the cut fixed for all packets.

## 14. Drift tool + trace consequences (user: write something that detects and corrects the drift as physically as possible; how does it affect the packet / trace parameters?) — user is taking the test images meanwhile

* `lvl1/drift_correct.py` (standalone, not in process_packet): Gaussian zero-peak fits per column block → detect (p<1e-6,
  range > 0.1 width) → FIXED cut 200 columns (calibrated on 5 images: width excess +15-18 % cols 0-24, +5-10 % to ~175,
  within ±3 % from 200) → subtract the slow remainder (2 exponentials + constant, spline fallback) → check row halves on
  kept columns (warning/DRIFTWRN) → header keywords, NCOLPRE/NCOL adjusted. On 5 images: χ²/ndf 0.65-1.15, σe 0.155/
  0.161/0.200 (0.6-2 %). Drift-free sims untouched; 127 flagged.
* Traces (10-12, 20x20): first pixel's skip noise 1.1-1.4x (NCOLPRE 0) vs 1.8-2.6x (NCOLPRE 185); second 1.0 vs 1.04-1.18 →
  disturbance grows with the dumped prescan (strong support for the dump hypothesis). trace_features drops pixel 0 only.
* Trace recommendations (pending the tests): NDCMPRE 5 in traces if images use it; NCOL 10 with the first two pixels per row
  left out; keep the trace's own dump small; match BOOLCLR if it matters.

## 15. Acquisition test images 132-135 (user: "check them, some may not have been taken correctly")

* Taken: 132 EXP 0, 133 EXP 1, 134 NCOLPRE 2800 (EXP 1), 135 BOOLCLR 0 (EXP 1); all NDCMPRE 0 (5 was suggested; harmless);
  132-134 without .meta/raw locally (header has the settings); board 102 images 51/53 missing/aborted (boards out of step);
  **134 has no overscan** (NCOLPRE 2800 with NCOL 1000 is a window inside the CCD — my test suggestion omitted widening NCOL).
* Compared against each row's end: exposure 0/1/10 → same drift; NCOLPRE 2800 vs 5300 → same drift (dumped-charge
  explanation unlikely, or saturating); BOOLCLR 0 → dip ~25 % smaller but noise raised to col ~400 (not a fix).
* Remaining picture: row-start disturbance of PA03 + board 101 output chain after the fast horizontal clocking before each
  row; recovers per pixel read; a question for the board-101 front-end expert.
* drift_correct.py: now works without overscan (row-end reference); finds and treats the drift in all tests.
* Basis for the user's decision: no setting removes it → cut 200 + slow remainder (drift_correct.py); keep BOOLCLR 1.
