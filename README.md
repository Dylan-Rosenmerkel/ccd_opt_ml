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
| `src/` | All code (see below) |
| `daq_scripts/` | `manual_packet.sh`: one packet by hand with plain `cdaq qadd` commands (nominal → reference traces → change → traces → master → image → nominal), with sanity checks and a dry run; `manual_packet_commands.txt`: its command list |
| `configs/` | DAQ setup, packet and scan examples, the recommended packet and scan (`packet_recommended.toml`, `scan_recommended.toml`: 10 + 10 matched trace sets, 1×1 image), parameter limits (`limits.toml`, **not yet reviewed**), processing settings |
| `notebooks/` | Step-by-step walkthroughs with saved outputs; see `notebooks/README.md` |
| `docs/` | Notes: `overleaf_ml_models/` (the models: what they do, inputs, data needs, results, next models and their prototypes, the real-data test plan, the data format), `overleaf_trace_features_technical/` (the math of every trace feature), `trace_fit_tools.tex` (comparison tools), `trace_features.tex` (all trace features), `ml_training_plan.tex` (ML plan), `packet_design.md` (image size, skips, binning, matched traces), `context/` (session records and handoff notes) |
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
| `ccd_limits.py` | Safety checks of every parameter set; voltage ramps |
| `fake_daq.py` | Fake boards for rehearsals (the fake cdaq queue is in `daq_control.py`) |
| `fake_cdaq_monitor.py` | Watch a fake rehearsal in the layout of the real tmux session (queues, logs, clients) |
| `simulate_trace.py` | Synthetic traces in the real file format, for any geometry and timing (for testing) |
| `check_trace_analysis.py` | Runs the trace analysis on 25 synthetic geometries/timings and on the real traces, checks the layout found |
| `process_packet.py` | Packet → decoded traces and features, decompressed FITS → WADERs labels → `dataset.csv` |
| `plot_*.py`, `view_trace.py` | Plots and interactive trace viewers |
| `ml/` | The models: synthetic CCD world, data loading, ridge / boosted trees / Gaussian process / NN ensemble / 1D CNN on curves, grouped evaluation, Bayesian optimization, benchmark (`python -m ml.benchmark`); second round: physics-structured outputs + hierarchical GP (`structured.py`), conformal error bars (`calibrate.py`), `python -m ml.benchmark_structured`; see `docs/overleaf_ml_models/ml_models.tex` |
| `plot_feature_figures.py`, `make_trace_features_note.py` | Figures and generator of the trace-features note |

## Quick start

```bash
python src/decode_trace.py TRACE.RJ45ADCDATA0                          # decode and check a trace
python src/trace_features.py TRACE.RJ45ADCDATA0                        # all features of a trace (any geometry)
python src/check_trace_analysis.py                                     # test the analysis on many geometries
python src/compare_trace.py TRACE_A TRACE_B --clean-pixels             # compare two traces
python src/take_packet.py --params configs/packet_example.toml --show-commands   # what a packet would send
python src/scan_packets.py --scan configs/scan_example.toml --dry-run  # plan a scan
python src/fake_cdaq_monitor.py --state /tmp/reh/fake_cdaq_state.json   # watch a rehearsal (2nd terminal)
python src/take_packet.py --params configs/packet_example.toml --backend fake --work-dir /tmp/reh --fake-time-scale 0.05
python src/process_packet.py MANIFEST.json                             # packet -> training data
```

The scripts need Python 3.11+ with numpy (and matplotlib for plots). The models need the packages of `requirements-ml.txt` (`python3 -m venv .venv && .venv/bin/pip install -r requirements-ml.txt`). Running on the DAQ needs the
group's `ccd_cdaq`/`acmpy`. Processing images needs WADERs (`pysimdamicm`, `panaSKImg`).

**Before using the real DAQ:** review `configs/limits.toml` and encode the bias/clock relationships
in `ccd_limits.py`. Then follow the checklist in `notebooks/take_packet_walkthrough.ipynb` and
`docs/context/data_taking_scripts_handoff.md`.
