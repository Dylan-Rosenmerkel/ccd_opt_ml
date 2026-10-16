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
