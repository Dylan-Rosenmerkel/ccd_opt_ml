# Data-taking and packet-processing scripts: handoff notes

First written 2026-10-08. Rewritten 2026-10-09 (version 2: cdaq queue, master/slave), then
**revised again the same day (version 3)** after the user shared the group's `take_image.py`,
`take_trace.py` and command-line routine. **Updated 2026-10-13 (version 4):** the cdaq repeat
queue is handled, the image step is one queue entry from leader mode back to standalone, the fake
queue models the real runner, and `fake_cdaq_monitor.py` shows a rehearsal like the tmux session
(section 9). Companion documents: `2026-10-08_session_context.md`,
`2026-10-09_session_context.md`. Step-by-step walkthroughs:
`notebooks/take_packet_walkthrough.ipynb`, `notebooks/process_packet_walkthrough.ipynb`.

**Status: nothing has run on the real DAQ.** Everything was rehearsed on fake boards
(`src/fake_daq.py`) that run the exact command text and copy real example files. The processing
script, including WADERs, ran for real on those copies.

---

## 1. Files

| File | Role |
|---|---|
| `src/take_packet.py` | Plans one packet (all checks, no hardware), runs it step by step through a controller, writes the manifest. CLI: `--show-commands`, `--status-only`, `--backend fake/cdaq` |
| `src/daq_commands.py` | Builds the exact Python text each board runs (following the group's scripts); `wrap_step` adds the receipt frame; `DEFAULT_TRACE` = take_trace.py values |
| `src/daq_control.py` | `CdaqController` (`ccd_cdaq.add2queue`, repeat-queue save/clear/restore), `FakeController` (model of the cdaq queue), `repeat_queue_suspended`, `wait_for_receipts`, `StepError` |
| `src/fake_cdaq_monitor.py` | Terminal display of a fake rehearsal in the layout of the real tmux session (queues, logs, clients, published commands); `--tmux`, `--once` |
| `src/scan_packets.py` | Scan plan (per-board reference + design, reference repeated), `--dry-run` plans every packet, resumable, stops at the first failure |
| `src/fake_daq.py` | Strict fake boards (misuse that would hang raises); start on the internal clock; traces cycle through the real repeats |
| `src/ccd_limits.py` | Limits, hardware ranges, relationships, ramps (relationships from `docs/voltage_rules.txt` and the ACM manual's ranges encoded on 2026-10-14; `src/check_limits.py` tests them) |
| `src/process_packet.py` | Manifest → decoded traces (.npz) + features (repeats combined) → decompressed FITS → WADERs → labels → dataset.csv/jsonl |
| `src/trace_regions.py` | Changed 2026-10-09: no crash when the vertical transfers cannot be found (module 102, `Vtransfer21`); `v = None` |
| `configs/daq_setup.toml` | Queue, pool, boards, leader, θ boards, trace boards, work dir, timeouts, **the acmpy command texts** |
| `configs/packet_example.toml` | One packet: `[image]`, `[trace]`, `[theta]` + `[theta_board.102]`, `[theta_ref]` |
| `configs/scan_example.toml` | Example scan |
| `configs/limits.toml` | Windows (**provisional, reviewed = false**) |
| `configs/processing.toml` | Output dir, DAQ→local path map, WADERs command/JSON/version dir, image overrides |

## 2. How the DAQ is driven

Like the group's scripts, the packet script puts Python text in the **cdaq queue**
(`add2queue(text, target)`) for one client (`acm101`) or the pool (`acmpool`). The client `exec`s
it with its own `daq`. Every text ends by writing a **receipt** (`<work_dir>/receipts/<packet>/<step>.acm<board>.json`:
ok/error with traceback, files written, status bits). The script waits for one receipt per expected
board before sending the next step. If a receipt is missing after the timeout, it stops and says
the board state is unknown.

Correspondence with the group's routine:

| group | packet |
|---|---|
| `cdaq qadd "daq.use_external_clock()" acmpool` (once per session) | step `standalone` sends it only if a board does not report the external clock; never switched back |
| `take_trace.py`: `daq.standalone()`, `time.sleep(.1)`, 4 × `daq.take_trace(fname="trace", subroutine="AcquireImage", EXPOSURE=0, NROW=2, NCOL=5, NSBIN=20, NPBIN=20, NDCM=500, BOOLCLR=0, ch=i)` | step `standalone`; steps `ref_traces_n`/`traces_n` (same call, 4 channels in one queue entry, the 5 cuts also set to 0) |
| `take_image.py -f`: `daq.standalone()`, `daq.flush_ccd()` | steps `flush_xN` |
| `take_image.py -m 101 -f`: `daq.master(101)`, `daq.take_image(fname="img", <image_params>)` to `acmpool`, `daq.standalone()`, `daq.flush_ccd()` | step `image`: all of it as ONE queue entry (the `[image]` table) |
| `cdaq qadd "daq.flush_ccd()" acmpool -t repeat` (idle flushing) | saved and cleared during a packet/scan, restored afterwards (`repeat_queue = "suspend"`) |
| `set_vdd.py` / `daq["VDD"] = -19` | part of θ |

## 3. Steps of one packet (default counts)

| # | step | target | notes |
|---|---|---|---|
| 0 | checks | this computer | θ/θ_ref per board (no geometry allowed in θ), limits, image/trace windows, every ramp point; nothing is sent on failure |
| 1 | status | pool | run directory, loaded sequencer (must equal `reference_v5_pumping.seq`), sync/clock bits |
| 2 | standalone | pool | `daq.standalone()`, 0.1 s; external clock if not already set |
| 3 | apply_ref | each θ board | that board's θ_ref: timing constants, then voltages in steps ≤ max_step, 0.5 s apart |
| 4 | flush_x3 | pool | 3 × `daq.flush_ccd()` (blocks; about 27 s + margin each) |
| 5 | ref_traces_1..3 | each traced board | 4 channels per set; a flush between sets; decoded right away, a damaged set is retaken once |
| 6 | apply | each θ board | θ_ref → that board's θ |
| 7 | flush_x1 | pool | after the change |
| 8 | traces_1..3 | each traced board | a flush between sets |
| 9 | image | pool | ONE queue entry: `daq.master(leader)`; `take_image` with the complete `[image]` table, **`acq_level` always given**; `daq.standalone()` (in a `finally`); `n_final_flushes` × flush (default 1). Then waits for `avg_<image>.fits*` per board |

The erase (`daq.erase_ccd(highP=9, wait_for_flush=True)`) is still available
(`erase_policy`), default `never`. The e-purge is never used.

There are 25 steps, about 13.1 min per packet (8 flushes 3.9 min, image 6.3 min, 24 trace
acquisitions). Timeouts: traces 3×, flushes 2×, image (with its flush) 2× + 60 s, each plus a
300 s margin.

## 4. Manifest (version 3)

Per board: `theta`, `theta_ref`, `theta_id`, `reference_id`, `delta_theta`, `prev_theta`.
Also: `packet_id, scan_name, status, date_start/end, image, trace, prev_packet_id,
seconds_since_previous, daq{...}, sequence{counts, erase_policy, ...}, sequencer, erase_done,
estimated_seconds, limits_file, limits_reviewed, repeat_queue_suspended (the commands removed
from the repeat queue, or null if it was not touched), steps[{step, kind, target, boards, seconds,
receipts}], run_directories, external_clock_set_by_packet, reference_traces[{board, channel, repeat,
stem}], traces[...], images[{board, stem, fits[]}], trace_quality{stem: problems | "not checked"},
warnings, error`.

Not recorded yet: board readback values (they are in each `.meta`), and housekeeping
(temperature, pressure).

## 5. Processing (`process_packet.py`)

1. Files: trace `<stem>.RJ45ADCDATA<ch>`, image `.meta` and `avg_<stem>.fits*`; DAQ paths are mapped
   via `[paths]`.
2. Traces: decode → quality → regions → tools (usable pixels, median and MAD) → `.npz`
   (lossless: values and flags, checked against the raw decode). `combine_repeats` gives per
   (kind, board, channel) the mean over the 3 sets and their sd (run-to-run);
   `trace_differences` = mean(θ) − mean(ref).
3. Images: bz2 → `.fits` → `panaSKImg <fits> --acm -j <json> -o <dir>`; the command, JSON and its
   SHA-1, the log and the pysimdamicm commit are saved.
4. Labels: σe = MESIG/MEGAIN, λ = MEDC/MEGAIN, EXT n = channel n−1. Flags: a value at a fit limit,
   an error that is ~0, or an error > 50 % of the value.
5. Dataset: one row per (packet, board, channel) with `theta.*`, `delta.*` (that board),
   `label.*`, `trace.*`, `trace_sd.*`, `ref.*`, `ref_sd.*`, `diff.*`, `fake`.

Rehearsal: 48/48 traces usable. The run-to-run sd of board 101 (real 10-08 repeats) is e.g. pulse
peak ±67–73 ADU; θ − ref differences are of that size. Board 101 labels as before; EXT3 is flagged
on both boards; board 102's λ (and channel 1's gain) are unconstrained. Each packet's `.npz` files
take about 270 MB (`save_raw = false` keeps only the features and CDS matrices); processing takes
about 1 min per packet.

## 6. To check on the DAQ machine (acmpy `e750760`)

The group's scripts confirm `standalone()`, `set_standalone()`, `flush_ccd()`, `master()`,
`use_external_clock()`, `take_trace(...)`, `take_image(...)`. Still unconfirmed:
`daq.acquire._filepath`, `.directory`, `daq.sequencer.seqfile`, `daq.command.get_system_status()`
(the clock check falls back to sending the command), and the image builder's `avg_*.fits.bz2`. On
the cdaq side: the pool `acmpool` with both clients attached to the queue; `work_dir` writable
by the clients (receipts) and by the cdaq main process (`queue_backups/`, written by
`ccd_cdaq.queue2file`); after the first packet, `cdaq print queues` shows the repeat flush back.

## 7. Decisions still open

* Trace parameters: they follow take_trace.py (20×20 binning, NDCM 500), not the image's 1×10
  binning.
* ~~Bias/clock relationships in `ccd_limits.relational_constraints`~~ (done 2026-10-14, from
  `docs/voltage_rules.txt`); replace the placeholder voltage windows of `limits.toml` by the group's limits
  and set `reviewed = true`.
* The WADERs JSON for the 100×130 scan images (an overscan region must exist); `img_9.json` is for
  1000-column images.
* Gain: a charge image per voltage set, or a fixed gain.
* `.npz` storage (`save_raw`).

## 8. Safe first hardware sequence

1. `python src/take_packet.py --params <packet> --show-commands`.
2. `python src/take_packet.py --params <packet> --backend cdaq --status-only` (read-only).
3. Review limits (`reviewed = true`); one packet with θ = θ_ref, watching the client logs.
4. `python src/process_packet.py <manifest>`: its 6 trace sets are repeats at identical settings.
5. A short scan (`scan_packets.py --dry-run` first).

## 9. The cdaq repeat queue and the fake queue (added 2026-10-13)

**What the real queue does** (`Soft_JHU/cdaq/src/ccd_cdaq/_cdaqqueue.py`, `_main.py`):
* a queue has a simple part (each command runs once) and a repeat part (rotated, run again and again);
* the runner takes the next simple command if any, otherwise the next repeat command;
* it runs one command at a time and waits until the target client is back in standby.

The group keeps `daq.flush_ccd()` → `acmpool` in the repeat queue (seen in the tmux session of
2026-10-13).

**Why it matters:** the packet script queues one step, waits for its receipt, then queues the
next. So between two steps the simple queue is empty, and the runner starts a repeat flush:
* every gap gets an extra ~27 s flush that the packet did not ask for (a kept repeat queue gave
  36 flushes per board in the fake instead of 9);
* a gap while the boards are in leader/follower mode would start a flush that waits for an
  external start forever. This was the old `leader` → `image` boundary, and the end of the packet.

**What was changed:**
* **One image entry** (`daq_commands.synchronized_image_lines`): `daq.master(leader)`, the
  image, `daq.standalone()` in a `finally`, then `n_final_flushes` flushes (default 1, as
  `take_image.py -f`). The receipt confirms `standalone_after_image`.
* **Repeat queue suspended** (`[daq] repeat_queue = "suspend"`, the default;
  `daq_control.repeat_queue_suspended`):
  * how: `ccd_cdaq.queue2file` dumps the queue to `<work_dir>/queue_backups/<name>_<UTC>.json`;
    `clearqueue(queue, "repeat")` empties the repeat part; on normal exit each saved command is
    added back with `add2queue(..., queue_type="repeat")`;
  * when: `take_packet.py` (CLI) does this per packet, `scan_packets.py` once per scan;
  * after an error: the queue is not restored, and the warning and the backup file give the
    `cdaq qadd "<cmd>" <to> -q default -t repeat` lines;
  * `"keep"` leaves the queue alone.
* **The fake queue** (`FakeController`) models the runner:
  * it starts with the group's repeat flush;
  * before each submitted step it runs the next repeat command;
  * `idle()` runs one after the image;
  * pool commands run on all boards in parallel threads;
  * a repeat command that fails (flush outside standalone) leaves the board **hung**, and the next
    step stops with a `StepError` at once (the real DAQ would time out instead).

  The walkthrough's mistake (e) shows the old leader/image split hanging this way.
* **Display:**
  * the fake writes its state (queue, clients, pool, log, "Publishing" lines) to
    `<work_dir>/fake_cdaq_state.json`;
  * `src/fake_cdaq_monitor.py` draws it in the layout and formats of `cdaq print queues / logs /
    clients` (`ccd_cdaq/monitoring.py`): `--tmux` for a tmux session like the real one, `--once`
    for a snapshot;
  * `--fake-time-scale 0.05` slows the fake boards to 5 % of real time so it can be watched;
  * the fake boards' log lines imitate the real ones; only the FlushCCD line was seen in a real
    session.

**Still untested:** `CdaqController`'s repeat-queue calls (`queue2file`, `clearqueue`,
`add2queue(..., "repeat")`) have never run against the real cdaq; MQTT is not involved in any
rehearsal.

## 10. Recommended packet geometry and matched traces (added 2026-10-13)

See `docs/packet_design.md`.

* **Image:** 16×1000×500, 1×1, prescan 5300, clear on, no exposure: 4.8 min, σ ≈ 0.26 e⁻,
  ~1000 one-electron pixels, with λ fitted per row (DC = slope, CIC = intercept). 200 skips gives
  σ ≈ 0.41 e⁻, too noisy to count electrons.
* **Traces:** `[trace] match_image = true` copies `NDCM`, `NDCMPRE`, `NSBIN`, `NPBIN` and `subroutine`
  from `[image]` (`daq_commands.MATCHED_KEYS`). The size is 2×5 pixels after a 100-pixel prescan, no
  clear: 0.19 s, 11 MB per channel. The manifest records `trace_matches_image`, and `plan_packet`
  warns when the trace and image differ.
* **Config:** `configs/packet_recommended.toml`, 25 steps, ~11.5 min.
* `configs/limits.toml`: the image-size windows were widened to the CCD size, so that NCOL 1000 and
  NCOLPRE 5300 are allowed.

## 11. Packets taken by hand can be processed (added 2026-10-15)

* `daq_scripts/manual_packet.sh`: `PACKET_ID` (default `m<date>_<time>`); traces `<PACKET_ID>_ref<set>` /
  `<PACKET_ID>_new<set>`; a record `$RECORD_DIR/<PACKET_ID>.json` (default `~/packet_records`) written when queued.
* `src/make_manual_manifest.py <record> --run-dir <raw files>` → manifest (theta from the .meta files, requested changes
  checked, image = first image per board after the packet's last trace); `--legacy-run` for older runs.
* `process_packet.py` then also fits λ row by row (dark current / CIC, `image_rows.py`) and writes
  `model_inputs/*.npz` for every model family (`ml/real_inputs.py`). Level 1 training: `lvl1/train_lvl1.py`.


## 12. Planned packets and timing parameters (added 2026-10-16)

* `manual_packet.sh`: `CHANGES_FILE=<file>` reads the changes from a file (`lvl1/packet_plan/L1_NN.txt`); timing
  parameters (delay_V*, delay_H*, delayInt, delayRG*/SW*/OG*/DG*) are known names with their reference values, checked
  against the limits.toml timing windows, whole cycles only. Record field `changes_file`.
* `lvl1/packet_plan/PLAN.md`: 41 planned packets (session A timing only, can be taken now).
* Processing labels now come from comm_scripts' `run_WADERs` by default (`configs/processing.toml [waders] runner`),
  with `--autoconfig-args=--keep-dead`; see `lvl1/README.md` and `lvl1/rehearsal_report/REHEARSAL.md`.
