# Notebooks

Step-by-step walkthroughs of the code in `../src`. Every notebook is saved **with its outputs**, so
it can be read without running it. Nothing here touches the DAQ hardware.

| Notebook | Topic | Status |
|---|---|---|
| `trace_comparison_walkthrough_v2.ipynb` | Comparing two traces (39 tools, permutation test) | current version |
| `trace_comparison_walkthrough.ipynb` | Same, older copy edited by hand | kept for reference |
| `take_packet_walkthrough.ipynb` | Taking a data packet (`take_packet.py`, `scan_packets.py`) | current |
| `process_packet_walkthrough.ipynb` | Processing a packet (`process_packet.py`, WADERs) | current |
| `advanced_trace_features.ipynb` | Development of the advanced trace features | kept for reference; superseded by the next one |
| `trace_features_walkthrough.ipynb` | All trace features in one function (`trace_features.py`) | current |
| `packet_tests_2026-10-12.ipynb` | First run with varied parameters: traces and images compared across 5 settings | analysis |

A suggested reading order: comparison → take packet → process packet → trace features.

---

## `trace_comparison_walkthrough_v2.ipynb`: can we tell two traces apart?

Uses `decode_trace.py`, `trace_regions.py` and `compare_trace.py` on module 101 traces.

1. Decode a raw trace (`.RJ45ADCDATA<ch>`) and check it.
2. Cut it into regions: skip cycles, horizontal transfers, vertical transfers.
3. Look at one pixel's skips; compute the summary tools and the parametric fits.
4. Treat pixels as replicates and compare trace A with trace B (z-scores and p-values per tool).
5. Explain the permutation test, then run the controls (even vs odd pixels, first vs second half).
6. Print the full report, then compare real repeats of channel 0 against channel 0 vs channel 3.

The region splitter it uses works for any trace geometry and timing (re-run on 2026-10-12).

**Result:** different amplifiers are clearly separated. Repeats of one amplifier also look
"different" in the pixel-level test; judged against the run-to-run spread, they are not.

`trace_comparison_walkthrough.ipynb` is an older copy of the same notebook that was edited by hand
(window in cell 5, `PIXEL = 0`, kernel `damic`). It lacks the p-value panels and the repeats section.
`_v2` includes those two edits.

## `take_packet_walkthrough.ipynb`: taking a data packet

Uses `take_packet.py`, `daq_commands.py`, `daq_control.py`, `ccd_limits.py`, `fake_daq.py`,
`scan_packets.py` and `configs/daq_setup.toml`, `packet_example.toml`, `limits.toml`,
`scan_example.toml`.

1. How commands reach a board through the cdaq queue, and how this follows the group's
   `take_image.py` / `take_trace.py`.
2. The DAQ setup: boards, pool, leader, the acmpy command texts, standalone vs master/slave.
3. The packet file: θ per board, `[image]` size, `[trace]` parameters.
4. Safety checks and voltage ramps.
5. The 26 planned steps and the exact text each board runs:
   3 flushes → 3 reference trace sets → θ → flush → 3 trace sets → image.
6. The receipt frame around every step.
7. A rehearsal of two packets on **fake boards**: the clock is set once, a damaged trace is retaken.
8. The manifest, then four deliberate mistakes the fake boards catch.
9. A scan dry run, and the checklist before running on the DAQ machine.

**Writes:** `../rehearsal/work/` (manifests, receipts, fake data files). The fake boards copy real
files from `../data/`; every file they write is marked `"FAKE": true`.

## `process_packet_walkthrough.ipynb`: from a packet to training data

Uses `process_packet.py`, `configs/processing.toml` and WADERs (`panaSKImg`, run through
`conda run -n damic`).

1. Read the manifest and find the files (DAQ paths are mapped to local ones).
2. Decode the traces. Save them losslessly as `.npz`, check that the saved arrays match, and compute
   the features.
3. Combine the 3 repeated trace sets (mean and scatter), then take θ − reference.
4. Decompress the FITS image and run WADERs.
5. Turn the WADERs output into labels (σe = MESIG/MEGAIN, λ = MEDC/MEGAIN), with flags for fits stuck at
   a limit or unconstrained.
6. Build `dataset.csv`: one row per (packet, board, channel).

**Needs:** the rehearsal packet from `take_packet_walkthrough.ipynb` (it is created if missing).
**Writes:** `../rehearsal/processed/`.

Step 2 uses the general region splitter (any geometry, `.meta` cross-check) and `trace_features`:
85 numbers and 12 curves per trace.

## `advanced_trace_features.ipynb`: brainstormed features, one by one (reference)

Every advanced feature computed with inline code on the module 101 channel 0 trace from 2026-10-08,
with a plot and the reason a model might use it:

* **A. CDS level:** common pattern, pixel spread, charge loss.
* **B. Noise character:** autocorrelation, Allan deviation, CDS spectrum, reset-level scatter and
  band powers, noise against window length, settling, baseline coupling.
* **C. Discrete noise:** steps, outliers, interference lines, ADC codes.
* **D. Edges:** jitter, sub-sample pulse, recovery and ringing, feedthrough per volt, variation along
  the cycle.
* **E. Slow structure:** row-start recovery, drift, horizontal transfers, principal components.
* **F.** Channel 0 vs channel 3 separation.

These features now live in `src/trace_features.py`, so this notebook is kept as the development record.
Its inline code assumes the usual trace geometry (500 skips, 150-sample windows); the general version
is `trace_features.py`.

## `trace_features_walkthrough.ipynb`: all features in one function

Uses `src/trace_features.py`: `trace_features(trace)` returns 85 numbers (the 39 comparison tools
plus the advanced features), their pixel spread, and 12 curves. The `FEATURES` registry describes
each one.

1. The registry: groups and counts.
2. Every feature of one trace, with units and pixel spread.
3. The 12 curves.
4. When features are missing (NaN): charged pixel, module 102, no `.meta`, damaged trace.
5. Redundancy: feature pairs correlated with |r| > 0.98 across 19 traces.
6. Separation of all 6 channel pairs vs repeat scatter.
7. How the features enter `dataset.csv` through `process_packet.py`.
8. Any trace geometry and timing: synthetic traces (`simulate_trace.py`) with 1–500 skips, up to
   21 pixels, 1×1 to 20×20 binning and different clock delays, all decoded correctly. The full test
   is `python src/check_trace_analysis.py`.

The full list of features is in `../docs/trace_features.tex`.

## `packet_tests_2026-10-12.ipynb`: what the parameter changes did

Data: `../data/2026-10-12-PA03_PQ04_Packet_Tests/` (5 settings: reference, VDD −20.5, V low rails +
TG, H low rails, H3_A_L only; 3 trace sets each on board 101, plus an image for the first 4).

1. The settings, as a table and a picture; which comparisons are clean (VDD went back to −19 V after
   setting 2).
2. Trace setup and quality (inherited `NCOLPRE = 185`; one damaged trace).
3. What a "separation" means, with one example feature (each dot one trace set).
4. Each parameter's fingerprint: the features that change on all channels, and a count per kind of
   feature.
5. Waveforms of the clean comparisons, with the difference shaded.
6. Trace noise and image noise per channel and setting.
7. Could an ML model be trained on this? Leave-one-out test, feature-space picture, and what is missing.
8. Appendix: all 85 features × 4 comparisons.

Saves its key figures to `../docs/context/figures/oct12_*.png`.

---

## Running the notebooks

* **Kernel:** a Python 3 environment with numpy and matplotlib (the notebooks are labelled `damic`).
  `process_packet_walkthrough.ipynb` also needs astropy for one image plot (optional) and runs
  WADERs through the conda environment `damic`.
* **Working directory:** this folder. Each notebook adds `../src` to the Python path itself.
* **Data:** `../data/2026-10-08-PA03_PQ04_traces/`,
  `../data/2026-10-06-PA03_PQ04_Installation_Check130K_1/` and (for the packet-test analysis)
  `../data/2026-10-12-PA03_PQ04_Packet_Tests/`. Only module 101 is used for the trace
  comparisons; module 102 files appear only as fake-board copies.
* **Generated files:** `../rehearsal/` (about 1.5 GB: copied raw traces and images, plus the processed `.npz` files). It is safe to
  delete; running `take_packet_walkthrough.ipynb` and then `process_packet_walkthrough.ipynb`
  recreates it.
* The scripts that generated these notebooks were kept outside the project. To change a notebook,
  edit it directly.

## Related documents

* `../docs/trace_fit_tools.tex`: the 39 comparison tools.
* `../docs/trace_features.tex`: every feature of `trace_features.py`.
* `../docs/ml_training_plan.tex`: the ML plan.
* `../docs/context/`: session records and the handoff for the data-taking and processing scripts.
