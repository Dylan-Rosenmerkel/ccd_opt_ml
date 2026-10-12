# Session context: 2026-10-09

Continuation of `2026-10-08_session_context.md`. Details of the scripts: `data_taking_scripts_handoff.md` (rewritten today).

## Requests

1. Read all context documents (done: the 10-08 context, the handoff, the ML training plan).
2. Revisit all DAQ commands for the packet script. **Images in master/slave mode** (one ACM leader, external clock); **traces in standalone mode**; use the group's take_trace/take_image approach.
3. **Take a reference trace for every data point** (training plan: changes relative to each amplifier's reference).
4. Start the **packet processing script**: trace format, FITS decompression, WADERs.
5. A step-by-step **notebook for each script**, very clear, before anything runs on the DAQ machine.

## What was learned from the DAQ code

* The group drives boards through the **cdaq queue** (`cdaq qadd "<python>" <client or pool>`); each client `exec`s the text with its `daq`. Synchronised images go to the pool after `daq.master(<leader>)` and `daq.use_external_clock()` (`cdaq/LBCReadme.md`); traces are standalone.
* acmpy `fpgacommand.py`: leader = EXS + ESS; follower = EXS + DSS; standalone = DXS + DSS; CEX/CIN select the clock. `get_system_status()` reports the master/slave/external-start/clock bits.
* With external start enabled, anything that starts the sequencer (trace, erase flush) waits for the leader's start: hence standalone for traces and the erase.
* `take_image`: the parameters are set, then a 0.5 s wait, then **EXPOSURE as a Python sleep**, then the sequencer starts. `take_trace` has no exposure and leaves the stored level at `[0, 3, 3, 3]`, so the image must always pass `acq_level`.
* The 10-06 image files of boards 101 and 102 have identical timestamps: they were read synchronously.
* **The DAQ machine runs acmpy `e750760`**, which is not in the local repository. The shell history uses an older API (`daq.ops.erase_CCD`, `daq.flush()`). All acmpy calls are therefore in `configs/daq_setup.toml` as editable text.
* The group's `scripts/take_image.py`, `take_trace.py` and `setup_run.py` are not available locally (only an old UChicago variant in another acmpy branch).
* FITS: HDU 1–4 = channels 0–3; WADERs `EXT<n>` = channel n−1; the `.meta` gives `ch<n>` → `CCD_A..D`.
* WADERs (pysimdamicm `ed622e7`, with local changes): the fit is a Gaussian-smeared Poisson; λ = MEDC/MEGAIN and σe = MESIG/MEGAIN confirmed from the source. Defaults: μ ±1, σ 0.001–100, λ 0.001–10, gain 1–100; `img_9.json` sets σ ≤ 0.5, gain 0.8–3, λ ≥ 1e-4. χ² is computed but not written.
* panaSKImg runs on this laptop with `conda run -n damic panaSKImg` (about 6 s per image). Importing ROOT without `conda run` crashes.

## What was built

* `src/daq_commands.py`, `src/daq_control.py` (new); `src/take_packet.py`, `src/scan_packets.py`, `src/fake_daq.py` (rewritten); `configs/daq_setup.toml`, `configs/packet_example.toml`, `configs/processing.toml` (new); `configs/scan_example.toml` (updated).
* `src/process_packet.py` (new): decode → `.npz` (lossless, verified) + features; FITS decompression (byte-identical, verified); WADERs with saved command/JSON/log/version; labels with fit-quality flags; `dataset.csv`/`.jsonl`.
* `notebooks/take_packet_walkthrough.ipynb` and `notebooks/process_packet_walkthrough.ipynb` (executed, outputs saved). They write into `rehearsal/` (about 160 MB; safe to delete; recreated by running the notebooks).
* The notebook generators are in the session scratch area only.

## Results of the rehearsal

* Fake packet: 10 steps in the expected order on both boards (standalone → erase → 4 reference traces → erase → 4 traces → leader/external clock → image on both), no e-purge, all receipts, manifest v2 complete. The fake catches: a trace in leader mode, an image without `acq_level` after a trace, a board that never answers (timeout), and a limits violation (0 texts sent).
* Scan dry run: 10 packets, about 1.2 h.
* Processing: 8/8 traces usable. Board 101: σe 0.106/0.105/0.109 e⁻ (EXT1/2/4), gain 1.27–1.34 ADU/e⁻, λ 1.03–1.25 e⁻/pixel/image. **EXT3 is flagged on both boards** (σ at its 0.5 ADU limit; also μ0 at 1 ADU on board 101), and this is already in the user's own 10-06 WADERs output. **Board 102**: almost no 1-e⁻ pixels, so λ (and on channel 1 the gain) are unconstrained; this illustrates the need for a charge image to get the gain.

## Open points (also in the handoff, section 7)

theta_boards (both or 101); trace clock; bias/clock relationships and the limits review; the WADERs JSON geometry for 100 × 130 images; how the gain will be obtained; the repeat flush in the queue; the acmpy `e750760` API check; first contact with `--status-only`.

---

## Second round (same day): the group's own scripts

The user shared `scripts/take_image.py`, `scripts/take_trace.py` and the command-line routine
(`setup_run.py -rd`, `daq.master(101)` and `daq.use_external_clock()` to acmpool, `set_vdd.py`,
`take_image.py`, then `daq.set_standalone()` and `take_trace.py acm101; acm102`), and asked for:

1. **separate settings for each board**;
2. **the external clock set once and never switched back**;
3. the packet sequence: **3 flushes, 3 reference traces with a flush between, then 3 new traces with
   a flush between, then an image of a specified size at the new parameters**;
4. good documentation and comments.

Changes (version 3; details in `data_taking_scripts_handoff.md`):

* The command texts now follow the scripts: `daq.standalone()` + 0.1 s, `daq.flush_ccd()`,
  `daq.take_trace(... take_trace.py values ...)`, `daq.master(101)`, `daq.take_image(fname, <image params>)`.
* θ and θ_ref are per board (`[theta]` + `[theta_board.<id>]`); each board gets its own apply
  step and ramp. Scan designs can target one board (`acm102.VR`).
* The image size and readout are a separate `[image]` table, like an image_params file; trace
  parameters are in `[trace]`, defaulting to take_trace.py. θ may not contain geometry (refused).
  Every acquisition sends its full geometry; the trace cuts are set to 0 (the 10-06 image had
  `NCOLPRE = 5300`, which a trace would otherwise inherit).
* The external clock is checked by each board in the `standalone` step and set only if it is not
  external. There is no clock change anywhere else.
* Sequence: 3 flushes → 3 reference sets (flush between) → θ → flush → 3 sets (flush between) →
  leader → image. There are 26 steps, about 12.7 min. The erase is optional (default `never`); the
  e-purge is never used.
* Both boards are traced (as in the routine).
* Checked in the acmpy source: `flush_ccd` blocks (`start_sequencer(wait=True)` sleeps for the
  FlushCCD duration, about 27 s at the reference timing).
* Fake boards: start on the internal clock (the rehearsal shows the clock set once over two
  packets); traces cycle through the real 10-08 repeats (the damaged one triggers the retake);
  `flush_ccd` is refused outside standalone mode.
* Processing: repeated sets combined (mean and run-to-run sd per tool), per-board θ/Δθ in the
  dataset (`trace_sd.*`, `ref_sd.*` columns).
* `trace_regions.py`: module 102 traces (`Vtransfer21`) have no detectable vertical-transfer
  period; the vertical region is now left out instead of crashing.
* Both notebooks were regenerated and executed (0 errors).

Open: the trace binning (20×20, from take_trace.py, differs from the image's 1×10); the `.npz`
volume (about 270 MB per packet); the WADERs JSON for 100×130 images; the gain; the limits review
and the bias/clock relationships; the acmpy attribute names listed in the handoff, section 6.

---

## Third round: trace features for ML

* Brainstorm of subtle trace features (no code), then `notebooks/advanced_trace_features.ipynb`: each
  feature computed on the 10-08 module 101 ch0 trace with plots, reframed (at the user's request) as
  **candidate inputs for a model**, not as ways to compute σe, λ or the gain from the trace. A test of
  single-electron quantization in the trace pixels found none at the image gain (units, offsets or too
  few pixels); it was removed from the notebook with the other physics derivations.
* Findings on that trace: the per-skip CDS has a common pattern rising ~30 ADU along the skips (baseline
  droop leaking in, coupling 2.1e-3, identical on ch0 and ch3); the level set by each reset scatters by
  ~146 ADU (cancelled by the CDS, 5.6 ADU left); the signal window is still settling (starts ~150 ADU low,
  ~2 µs); the output rings at ~3 MHz after the reset pulse; edge jitter 1.7 ns; the skip-to-skip variation
  is dominated (82 %) by the sub-sample phase walk at the edges; a 5 kHz peak in the CDS spectrum.
* **Merged** the 39 comparison tools and the advanced features into `src/trace_features.py`:
  `trace_features(trace)` returns 85 numbers, their pixel spread and 12 curves (0.4 s per trace); the
  `FEATURES` registry documents each. `process_packet.py` now uses it (dataset: 85 `trace.*`, `ref.*`,
  `diff.*`, `*_sd.*` columns). Earlier files are unchanged.
* `notebooks/trace_features_walkthrough.ipynb`: all features of one trace, the curves, missing values,
  redundancy (67 pairs with |r| > 0.98 across 19 traces), and separation for all 6 channel pairs
  (17 features separate all 6; 25 none).
* `docs/trace_features.tex` (generated from the registry; not compiled, no LaTeX here): every feature,
  definitions, the separation and redundancy tables.

---

## 2026-10-12: trace analysis for any geometry and timing

Request: make the trace analysis work for any clock timing, number of skips and binning.

* **Audit**: the code assumed the one real geometry (2x5 pixels, 500 skips, 20x20, reference timing):
  pixels grouped by the 4-bit pixel number (wraps after 16 pixels), fixed counts (skips 10-60 and
  450-500, skip 250, 150-sample windows, 75-sample half windows, blocks of 50, the reset pulse at
  ~sample 374), transfer periods assumed < 5000 / < 20000 samples, row starts from gap lengths,
  vertical transfers from flag bit 5 (which does not mark them on module 102).
* **Changes**: `decode_trace.py` groups pixels by the order of the window records; `trace_regions.py`
  rewritten (pixels from window order, exact period, horizontal transfers = flag bit 6, vertical =
  bit 7, flag glitches shorter than 10 samples ignored, row starts from NCOL or vertical edges,
  `.meta` fallback and cross-check via new `trace_geometry.py`); `compare_trace.py` and
  `trace_features.py`: every count scales with the trace (`skip_counts`, `trace_sizes`); features
  that cannot exist are NaN; a failing group no longer stops the trace.
* **Testing**: `simulate_trace.py` writes synthetic traces in the real format; `check_trace_analysis.py`
  runs 25 geometries/timings (1-2000 skips, 1-21 pixels, binning 1x1-20x20, windows 30-450 samples,
  fast/slow clocks, late reset, charged pixel): layout correct with and without the `.meta`, no group
  failures, only legitimate NaN. On the real traces all features are unchanged except the vertical
  tools (now aligned on bit 7; module 102 now has them) and "cycle variation after reset" (< 0.2 %).
