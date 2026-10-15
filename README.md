# ccd_opt_ml: ML-assisted optimization of skipper-CCD operating parameters

The goal is to find operating parameters for a skipper CCD (bias voltages VSUB, VDD, VDRAIN, VR; clock
rail levels; timing) that give calibrated images with **low read noise σe**, **low dark current λ** and
a **known gain**.

The approach is to take *packets*. Each packet holds short raw video **traces** and a calibrated
**image** taken at the same settings. A model learns how the settings and the trace features relate
to σe, λ and the gain (measured on the images with WADERs). An optimizer then uses the model to
suggest better settings.

**Status (October 2026):**
* the data-taking and processing scripts are written and rehearsed on fake boards, but not yet run
  on the DAQ;
* the trace decoder, the trace-comparison tools and 85 trace features are tested on real traces;
* no scan with varied parameters has been taken yet, and no model has been trained.

## Layout

| Folder | Contents |
|---|---|
| `lvl1/` | **Level 1**: can the image's noise parameters be predicted from the traces? Packets taken by hand, processed, every model trained and tested without amplifier labels (`train_lvl1.py`), synthetic test packets, walkthrough notebook; see `lvl1/README.md` |
| `src/` | All code (see below) |
| `daq_scripts/` | `manual_packet.sh`: one packet by hand with plain `cdaq qadd` commands (nominal → reference traces → change → traces → master → image → nominal), with the group's voltage rules and the ACM manual's hardware ranges checked before anything is queued, and a dry run; every trace is named `<PACKET_ID>_ref/new<set>` and a packet record is written so the packet can be processed; `manual_packet_commands.txt`: its command list |
| `configs/` | DAQ setup, packet and scan examples, the recommended packet and scan (`packet_recommended.toml`, `scan_recommended.toml`: 10 + 10 matched trace sets, 1×1 image), a mixed one-at-a-time + random design (`scan_mixed_example.toml`), parameter limits (`limits.toml`: relations and hardware ranges encoded, timing windows wide, voltage windows **not yet reviewed**), processing settings |
| `notebooks/` | Step-by-step walkthroughs with saved outputs; `notebooks/model_tutorials/`: one tutorial notebook per ML model, simplest first, from the basics; see `notebooks/README.md` |
| `docs/` | Notes: `overleaf_ml_guide/` (a learning guide to every model, from the basics to the advanced ones, with the equations explained and when to use each), `overleaf_ml_models/` (the models: what they do, inputs, data needs, results, next models and their prototypes, the real-data test plan, the data format), `overleaf_trace_features_technical/` (the math of every trace feature), `trace_fit_tools.tex` (comparison tools), `trace_features.tex` (all trace features), `ml_training_plan.tex` (ML plan), `packet_design.md` (image size, skips, binning, matched traces), `voltage_rules.txt` (the group's safe-operation rules), `Sequencer Notes.pdf` (ACM/LDAQ manual appendix: hardware ranges, sequencer; not in the repository), `context/` (session records and handoff notes) |
| `data/`, `rehearsal/`, `plots/` | Raw data, rehearsal output and figures: **not in the repository** (see `.gitignore`) |

## Code (`src/`)

| File | Role |
|---|---|
| `decode_trace.py` | Decode a raw trace (`.RJ45ADCDATA<ch>`): ADC samples, window flags, inline sums, quality check |
| `trace_regions.py` | Cut a trace into skip cycles, horizontal and vertical transfers; works for any geometry and timing |
| `trace_geometry.py` | Expected trace layout and durations from the `.meta` (cross-check and fallback) |
| `compare_trace.py` | 39 comparison tools and a permutation test: can two traces be told apart? |
| `trace_features.py` | `trace_features(trace)`: 85 features and 14 curves per trace, documented in a registry |
| `take_packet.py` | Take one packet through the cdaq queue: flushes, reference traces, traces, synchronized image, manifest |
| `daq_commands.py`, `daq_control.py` | The exact text sent to each board; the cdaq/fake controllers and step receipts |
| `scan_packets.py` | Run a series of packets (scan design, dry run, resume) |
| `ccd_limits.py` | Safety checks of every parameter set (windows, hardware ranges of the ACM manual, the group's voltage rules on the complete configuration); voltage ramps |
| `check_limits.py` | Regression check of the safety limits (every rule must refuse a breaking change; nominal must pass) |
| `fake_daq.py` | Fake boards for rehearsals (the fake cdaq queue is in `daq_control.py`) |
| `fake_cdaq_monitor.py` | Watch a fake rehearsal in the layout of the real tmux session (queues, logs, clients) |
| `simulate_trace.py` | Synthetic traces in the real file format, for any geometry and timing (for testing) |
| `check_trace_analysis.py` | Runs the trace analysis on 25 synthetic geometries/timings and on the real traces, checks the layout found |
| `process_packet.py` | Packet → decoded traces and features, decompressed FITS → WADERs labels and λ row by row (`image_rows.py`: dark current and CIC) → `dataset.csv` and one ready-to-train file per model family (`ml/real_inputs.py`) |
| `make_manual_manifest.py` | The manifest of a packet taken by hand (from its record and the `.meta` files: settings as reported, requested changes checked, image found by time); `--legacy-run` for older runs |
| `stage0_trace_vs_image.py`, `real_amplifier_check.py` | Real-data checks: trace-predicted vs image read noise; gains, noise and setting responses of the real amplifiers without labels |
| `plot_*.py`, `view_trace.py` | Plots and interactive trace viewers |
| `ml/` | The models: synthetic CCD world, data loading, ridge / boosted trees / Gaussian process / NN ensemble / 1D CNN on curves, grouped evaluation, Bayesian optimization, benchmark (`python -m ml.benchmark`); second round: physics-structured outputs + hierarchical GP (`structured.py`; the default model needs no amplifier labels), conformal error bars (`calibrate.py`), joint dark + CIC fit; remaining models: pre-training and deep kernels (`deep.py`), set models (`setmodels.py`), trust region / constraints / multi-fidelity / ParEGO (`bo_advanced.py`); benchmarks `ml.benchmark_structured`, `ml.benchmark_advanced`, `ml.benchmark_seeds` (robustness), `ml.benchmark_amp` (amplifier labels); see `docs/overleaf_ml_models/ml_models.tex` |
| `plot_feature_figures.py`, `make_trace_features_note.py` | Figures and generator of the trace-features note |

## Quick start

```bash
python src/decode_trace.py TRACE.RJ45ADCDATA0                          # decode and check a trace
python src/trace_features.py TRACE.RJ45ADCDATA0                        # all features of a trace (any geometry)
python src/check_trace_analysis.py                                     # test the analysis on many geometries
python src/check_limits.py                                             # test the safety limits and rules
python src/compare_trace.py TRACE_A TRACE_B --clean-pixels             # compare two traces
python src/take_packet.py --params configs/packet_example.toml --show-commands   # what a packet would send
python src/scan_packets.py --scan configs/scan_example.toml --dry-run  # plan a scan
python src/fake_cdaq_monitor.py --state /tmp/reh/fake_cdaq_state.json   # watch a rehearsal (2nd terminal)
python src/take_packet.py --params configs/packet_example.toml --backend fake --work-dir /tmp/reh --fake-time-scale 0.05
python src/process_packet.py MANIFEST.json                             # packet -> training data
```

The scripts need Python 3.11+ with numpy (and matplotlib for plots). The models need the packages of `requirements-ml.txt` (`python3 -m venv .venv && .venv/bin/pip install -r requirements-ml.txt`). Running on the DAQ needs the
group's `ccd_cdaq`/`acmpy`. Processing images needs WADERs (`pysimdamicm`, `panaSKImg`).

**Before using the real DAQ:** replace the placeholder voltage windows in `configs/limits.toml` by the group's
limits and set `reviewed = true` (the relations of `docs/voltage_rules.txt` and the manual's hardware ranges are
already enforced; `python src/check_limits.py` tests them). Then follow the checklist in `notebooks/take_packet_walkthrough.ipynb` and
`docs/context/data_taking_scripts_handoff.md`.
