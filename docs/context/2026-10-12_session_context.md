# Session context: 2026-10-12

Continuation of `2026-10-09_session_context.md` (which also covers the GitHub setup and the
Overleaf export). Scripts: `data_taking_scripts_handoff.md`; trace features: `docs/trace_features.tex`.

## 1. Questions answered (no code changes)

* **How the fake DAQ works.** `src/fake_daq.py` is our own stand-in, not the group's `ldaq/fakedaq.py`
  and not acmpy code. It imitates the acmpy calls the packet uses: names and behaviour were read
  from the acmpy source with `git show`, but the DAQ machine runs commit `e750760`, which is not
  local. `FakeController` runs the exact queued text with `exec(text, {"daq": fake})`. The fake
  boards refuse what would hang the real hardware: trace, flush or erase outside standalone mode,
  an image without leader mode or without `acq_level`, and any e-purge. They write files with the
  real naming, copying real data. Not tested by it: hardware, timing, the cdaq queue and MQTT. The
  group's `fakedaq.py` only sleeps and publishes MQTT (queue testing).
* **Is there one script that gives all features of any trace?** Yes: `src/trace_features.py`
  (`python src/trace_features.py TRACE`, or `trace_features(path)`): 85 numbers, pixel spread and
  12 curves.
* **Does `compare_trace.py` use all features?** No, only the original 39 tools (35 in the
  verdict). Its permutation test needs per-pixel replicates. About half of the new features are
  per pixel; the rest describe the whole trace and need repeated traces to be judged. Extending it
  was offered, not done.
* **Why are the vertical tools not in the verdict?** A trace has essentially one usable
  vertical block (once per row, not per pixel). Its ~18 transfers drift along the block, so they
  are not interchangeable replicates, and the self-control (first half vs second half of one
  trace) failed. They remain information only in that test.
* **Can the vertical features still be used by a model?** Yes. They are computed for every trace
  and are in `dataset.csv`; a model uses the scatter between repeated trace sets as their noise. They
  are the trace's direct record of the vertical clocks (V rails, `delay_V*`, TG).

## 2. Trace analysis for any geometry and timing (commit `53f11e9`)

* **Problem**: the analysis assumed the single real geometry (2×5 pixels, 500 skips, 20×20,
  reference timing):
  * the 4-bit pixel number wraps after 16 pixels;
  * many counts were fixed;
  * transfer periods were assumed below 5000/20000 samples;
  * row starts came from gap lengths;
  * vertical transfers came from flag bit 5, which does not work on module 102.
* **Fix**:
  * `decode_trace.py`: pixels from the order of the window records;
  * `trace_regions.py`: rewritten. Pixels from the window order, the exact cycle length,
    horizontal transfers = flag bit 6, vertical transfers = bit 7, glitches shorter than 10 samples
    ignored, row starts from `NCOL` or from the vertical edges;
  * new `trace_geometry.py`: the `.meta` cross-check and fallback;
  * `compare_trace.py` / `trace_features.py`: every count scales with the trace, a feature that
    cannot exist is NaN, and a failing group no longer stops the trace.
* **Tests**: `simulate_trace.py` (synthetic traces in the real format, any geometry and timing) and
  `check_trace_analysis.py`. All 25 configurations pass, with and without the `.meta`: 1–2000
  skips, 1–21 pixels, 1×1–20×20 binning, windows of 30–450 samples, fast and slow clocks, late
  reset, charged pixel. On the real traces only the vertical tools changed (aligned on bit 7;
  module 102 now has them), and "cycle variation after reset" changed by less than 0.2 %.
* **Notebooks** re-run with updated text: the comparison `_v2`, `process_packet_walkthrough`
  (now shows the 85 features) and `trace_features_walkthrough` (new section 8: synthetic geometries).
  `take_packet_walkthrough` was not affected; `advanced_trace_features` and the old comparison copy
  are kept as records.
* Committed and pushed to `github.com/Dylan-Rosenmerkel/ccd_opt_ml` (`53f11e9`). The Overleaf
  folder and its zips are not in the repository.

## 3. First look at `data/2026-10-12-PA03_PQ04_Packet_Tests`

**The first data with varied operating parameters.** The folder has 378 files (3.9 GB):

* 60 traces (board 101 only, 2×5×500, 20×20 binning);
* 8 images (50×130×500, 20×20 binning, `EXPOSURE` 10 s, `NCOLPRE` 185, `BOOLCLR` 1, boards 101 and
  102 read together);
* `.meta` files of one pocket-pumping image (60×300×500, 1×1, `EXPOSURE` 100 s, `PP_SHIFT` 3,
  `PP_NCYCLES` 5000). It has no data file and no end time: it was still running, or never
  finished, when the folder was copied.

There are no packet manifests or receipts, so the run was apparently taken with the group's
scripts. Each setting is 3 trace sets about 33 s apart (consistent with a flush between sets), then
an image about 3 min later. There are no separate reference traces per setting.

### Settings (from the `.meta` files; times UTC)

| Group | Time | Change from the reference (VDD −19, V lows 7.9/7.7, H lows 6.7/6.5, TG 7.4) |
|---|---|---|
| 1 | 18:56–19:03 | reference |
| 2 | 19:15–19:19 | VDD −20.5 V (read back −20.524) |
| 3 | 20:40–20:43 | (VDD back to −19 V from here on) V low rails −0.5 V (7.4/7.2); TG 7.0. **On board 101 TG_B_L stayed 7.4 while TG_B_H = 7.0 (high below low)**; board 102 has 7.0/7.0 |
| 4 | 20:59–21:02 | all H low rails −0.5 V (6.2/6.0) |
| 5 | 21:09–21:15 | only H3_A_L −0.5 V (6.2); traces only, then the unfinished pocket-pumping image |

### Problems found

* **The traces inherited the image's prescan.** The 12 traces of group 1 have `NCOLPRE = 0`
  (0.8 s, 12 MB). All 48 later traces have `NCOLPRE = 185` (1.7 s, 68 MB): each row first skips
  3700 columns, about 3720 horizontal transfers before the row-start pixels. `take_trace.py` does
  not set `NCOLPRE`, so it keeps the previous image's value. **Group 1 is therefore not directly
  comparable with groups 2–5.** Groups 2–5 are comparable with each other. Our packet script sets
  all cuts explicitly for this reason.
* One trace lost data (`trace_..._ch3_101_20261012_211002_91`, group 5); the quality check refused it.
* The TG_B setting on board 101 in group 3 has its high level below its low level.
* Board 102's images: channels 0, 1 and 3 contain no charge and no overscan step (flat, 0.2 ADU
  noise), and only channel 2 shows charge. This is consistent with module 102 not being usable.

### Traces: the features respond to the parameter changes

Feature means over the 3 trace sets (± scatter between sets), channel 0 (CCD_D):

| Feature | 1 ref* | 2 VDD | 3 V/TG | 4 H low | 5 H3_A_L |
|---|---|---|---|---|---|
| horizontal: minimum (ADU) | −91,410 | −93,060 | −91,380 | **−106,900** | −98,980 |
| H-transfer peak-to-peak (ADU) | 125,700 | 127,500 | 125,700 | **136,400** | 131,600 |
| vertical: minimum (ADU) | −22,450 | −22,900 | **−28,370** | −22,500 | −22,470 |
| baseline at skip 1 (ADU) | −19,400 | −19,730 | −19,320 | **−22,100** | −20,500 |
| settling time after pulse (µs) | 0.6293 | **0.6369** | 0.6287 | 0.6264 | 0.6290 |
| dip recovery time (µs) | 0.3156 | **0.3250** | 0.3155 | 0.3155 | 0.3174 |
| CDS noise per skip (ADU) | 5.71 | 5.78 | 5.67 | 5.78 | 5.75 |

(* group 1 was taken with a different trace setup; see above.) Scatter between sets: about 10–30 ADU
for the waveform levels and about 0.001 µs for the time constants.

* **Horizontal low rails** (group 4) change the horizontal-transfer waveform (minimum about 15,000 ADU
  deeper, peak-to-peak +10,000 ADU), and the baseline level (about −2,700 ADU). Changing only H3_A_L (group 5)
  gives about half of that. Both are visible on all 4 channels.
* **Vertical low rails / TG** (group 3) move the vertical-transfer minimum (−6,000 ADU) on all
  channels. On **channel 2 (CCD_A)** the noise rises strongly: CDS noise per skip 22 → 37 ADU,
  reset-level scatter 146 → 156 ADU, while channel 0 is unchanged.
* **VDD −1.5 V** (group 2) slows the amplifier response on every channel: settling time
  +0.008 µs, dip recovery time +0.01 µs (about 10× the scatter between sets), pulse peak +480 ADU.
  These are the amplifier-speed features we expected to see an output-stage bias.
* Counted over the 85 features (separation ≥ 6 with 3 vs 3 trace sets): every pair of groups differs
  in 11–29 features per channel. Part of that can be drift over the 1–2 hours between groups:
  group 3 came 85 min after group 2.
* Channel 2 (CCD_A) is about 4× noisier than channel 0 at every setting (CDS noise 16–37 against
  5.7 ADU).
* The noise of 50-skip averages and the baseline coupling differ between group 1 and the rest. That
  is consistent with the different trace setup (`NCOLPRE`), not with θ.

### Images (board 101, preliminary, no WADERs yet)

The last 7 of the 130 columns are overscan. The charge per bin (20×20 = 400 CCD pixels) grows from
the first to the last row: it is collected during the readout. Median charge is 27–44 ADU per bin,
roughly 20–35 e⁻ with the gain of about 1.3 ADU/e⁻ from the 2026-10-06 1×1 image.

| Channel | Overscan noise, robust (ADU), groups 1/2/3/4 | Overscan level change vs group 1 |
|---|---|---|
| ch0 (D) | 0.58 / 0.66 / 0.43 / 0.42 | group 4: −1.3 ADU |
| ch1 (C) | 1.81 / 1.62 / 1.78 / 1.91 | group 4: −1.9 ADU |
| ch2 (A) | 1.98 / 1.56 / **6.35** / 1.42 | group 3: **+10.9 ADU**; group 4: −1.3 ADU |
| ch3 (B) | 1.50 / 1.52 / **3.80** / 1.62 | group 3: **+4.4 ADU**; group 4: −1.2 ADU |

* The **group 3 change (V/TG, with TG_B inverted on board 101) raises the overscan level and noise
  only on the A and B amplifiers** (ch2, ch3). The trace of ch2 shows the same thing (CDS noise
  37 ADU): the trace and the image agree.
* **Group 4 (H low −0.5 V) lowers the overscan level on all channels** by 1.2–1.9 ADU, matching the
  baseline shift seen in the traces.
* **WADERs labels are not yet possible for these images.** With about 20–35 e⁻ per bin, the
  dark-current fit of `img_9.json` (made for mostly 0- and 1-electron pixels) does not apply, and a
  photon-transfer estimate gives 6–12 ADU/e⁻: the bin-to-bin variance is dominated by uneven dark
  charge between columns, not by Poisson statistics, so it is not the gain. For labels: σe from the
  overscan (with the gain from an unbinned image), and λ from an unbinned image or a binning that
  keeps the charge low.

## 4. Suggestions for the next data

1. Set every cut in the trace command (`NCOLPRE=0`, `NROWPRE`, `NCOLPOST`, `NROWPOST`,
   `NDCMPRE`, `BOOLCLR=0`), or use `take_packet.py`, which does. Otherwise traces taken after an
   image inherit its prescan.
2. Take reference traces in every packet (θ_ref just before θ), so drift between groups separates
   from the effect of θ.
3. Check that every rail keeps high ≥ low (the TG_B case); `ccd_limits.py` enforces this.
4. For image labels, an image setup that WADERs can fit (unbinned, or low charge per pixel), plus a
   gain image per voltage set.
5. Repeat the reference setting at the end of the run to measure the drift over the session.

## 5. Recommended dark-current image (about 5 minutes, 1×1)

Per-row time from `take_packet.formula_acquisition_seconds` (reference timing):

| Parameter | Value |
|---|---|
| `NROW` × `NCOL` × `NDCM` | **35 × 1000 × 200** |
| `NPBIN`, `NSBIN` | 1, 1 |
| `NCOLPRE` | 5300 (as the 2026-10-06 1×1 image: 854 active + 146 overscan columns) |
| `NROWPRE`, `NCOLPOST`, `NROWPOST`, `NDCMPRE` | 0 |
| `BOOLCLR`, `EXPOSURE` | 1, 10 s |
| Readout | 8.11 s per row → **≈ 4.9 min** (34 rows 4.8 min, 36 rows 5.0 min) |

* About 30,000 active pixels per channel. With today's dark rate (4.4–6.7 × 10⁻⁴ e⁻/pix/s) the
  pixels collect 0.005–0.15 e⁻ along the readout (mean about 0.05–0.1), i.e. about 2,000–3,000
  one-electron pixels: enough for the WADERs dark-current fit.
* With 200 skips σ ≈ 0.23 e⁻ on channel 0 (well separated 0/1 e⁻ peaks); channel 2 (CCD_A) is too
  noisy (≈ 0.9 e⁻).
* WADERs: `img_9.json` with `ccd_rows` 35 and `ccd_cols` 854. λ grows along the rows (charge
  collected during the readout), so a per-row dark-current fit is better than one λ.
* Alternatives: `NDCM` 300 → 25 rows (σ ≈ 0.18 e⁻, fewer pixels); `NDCM` 100 → 59 rows (more pixels,
  σ ≈ 0.3 e⁻).

## 6. Comparative analysis notebook

`notebooks/packet_tests_2026-10-12.ipynb` (figures in `docs/context/figures/oct12_*.png`):

* **VDD returned to −19 V after group 2.** Comparisons of group 2 with 3–5 contain the VDD change
  back; 4 vs 5 is the cleanest single comparison (four H low rails).
* Features with separation ≥ 6 per comparison: 11–29 per channel.
* H low rails (4 vs 5): horizontal minimum +7,000–7,900 ADU (separation ≈ 430), H peak-to-peak
  −4,800 to −6,000 ADU, baseline at skip 1 +1,400–1,600 ADU, the same on all 4 channels.
* V lows / TG (2 vs 3): vertical minimum −5,200 to −5,500 ADU on all channels (separation ≈ 330).
* VDD: settling time and dip-recovery time 0.008–0.012 µs longer at −20.5 V, on every channel.
* Channel 2 noise in group 3: trace CDS 25 → 37 ADU, image overscan noise 1.6 → 6.3 ADU.

## 7. Open

* Process these images with a suitable WADERs configuration, or define the overscan-based σe label.
* Take the recommended 1×1 image (section 5) and fit it with WADERs.
* Decide whether the Overleaf export belongs in the repository.
