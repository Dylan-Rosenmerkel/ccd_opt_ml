# Level 1: can the image's noise parameters be predicted from the traces?

The project is staged in three levels:

| Level | Question | Who decides the next packet | Status |
|---|---|---|---|
| **1** | Can σe, λ and the gain of the image be predicted from an amplifier's traces? | you (packets taken by hand) | **this folder** |
| 2 | Predict, and *suggest* the next settings (Bayesian optimization inside the safety rules) | you, helped by the suggestion | after level 1 |
| 3 | Predict and run new scans | the optimizer, within limits | after levels 1 and 2 |

Level 1 is manual on purpose: packets are taken by hand, processed, and the models are trained and tested
offline. Nothing here sends anything to the hardware.

**No amplifier labels.** None of the level-1 models is told which amplifier it is looking at, except one reference
(`physics_hier`, model A+B) that is there only to measure what labels would add. A model sees an
amplifier's *own* traces (its trace and reference features or curves) and the settings. The board and channel are used
only to group rows for testing and scoring.

**One trace → one training example.** Each packet gives, per amplifier, its traces at the new settings, its reference
traces, and its own labels from the image (WADERs fits each amplifier's part of the image). So a packet read by 4
amplifiers gives 4 examples: 100 packets = 400 examples. The 4 examples of a packet are always kept together when the
data are split for testing: they share the CCD's state at that moment, and splitting them would make scores look better
than they are.

## The steps

```bash
# 1. take a packet by hand (on the DAQ machine): a planned packet (lvl1/packet_plan/PLAN.md) ...
DRY_RUN=1 PACKET_ID=L1_02 CHANGES_FILE=lvl1/packet_plan/L1_02.txt bash daq_scripts/manual_packet.sh   # check, send nothing
PACKET_ID=L1_02 CHANGES_FILE=lvl1/packet_plan/L1_02.txt bash daq_scripts/manual_packet.sh             # queue it
#    ... or your own changes in the SETTINGS block:  PACKET_ID=m0001 bash daq_scripts/manual_packet.sh
#    (each queued packet writes its record, ~/packet_records/<PACKET_ID>.json: copy it with the data)

# 2. (analysis computer) the packet's manifest, from its record and the raw files
python src/make_manual_manifest.py ~/packet_records/m0001.json --run-dir <raw-file directory> --out manifests

# 3. process: traces -> features and curves; image -> WADERs labels and λ row by row;
#    dataset.csv and one ready-to-train file per model family in <processed>/model_inputs/
cd src && python process_packet.py ../manifests/m0001.json [../manifests/m0002.json ...]

# 4. train and test every level-1 model (packet-grouped cross-validation), and write the report
cd .. && .venv/bin/python lvl1/train_lvl1.py --inputs processed/model_inputs
```

The walkthrough notebook [`lvl1_walkthrough.ipynb`](lvl1_walkthrough.ipynb) goes through every step, with the 10-12 packet
tests as a real-data example and synthetic packets for the training (real labelled packets do not exist yet).

## The packet plan (`lvl1/packet_plan/`)

`lvl1/make_packet_plan.py` writes [`packet_plan/PLAN.md`](packet_plan/PLAN.md): **41 packets in three sessions (~11 h)**,
each a ready changes file `L1_NN.txt`.

| Session | Packets | What changes | Status |
|---|---|---|---|
| A | 13 (~4 h) | timing only: delayInt 300–3000, the horizontal clock hold, the reset hold; nominal packets at the start, middle and end | **can be taken now** |
| B | ~17 | one voltage group at a time (VDD, VR, RG high/low, SW low, H lows, V lows, TG, VSUB), one `max_step` either way | provisional: the voltage windows are placeholders (`reviewed = false`) |
| C | 11 | joint changes (a Latin hypercube in the same box) and a final nominal | provisional, as B |

Rules:
* every configuration passes `ccd_limits` (windows, the ACM manual's hardware ranges, the group's voltage rules);
* every voltage stays within one `max_step` of nominal, so no ramp is needed going there and back;
* changes go to board 101 only;
* both readout stages' pins move together, so all four amplifiers see each change.

Every file also passes `manual_packet.sh`'s own dry-run checks. `manual_packet.sh` now reads `CHANGES_FILE` and knows the
timing parameters, checked against the `limits.toml` timing windows (whole clock cycles only).

## The WADERs route (`configs/processing.toml`, `[waders] runner`)

Labels come from **`run_WADERs`** (the `comm_scripts` repository) by default. It:
* builds each CCD's panaSKImg configuration from the image itself: geometry from the header, gain, noise, pedestal method;
* fits only the Poisson-like 0e⁻/1e⁻ peaks.

`process_packet.py` reads its `fit_quality.csv`.

The other route, one hand-written JSON for every image (`runner = "panaSKImg"`), did much worse in the rehearsal, even
with the planned geometry set (table below).

`--autoconfig-args=--keep-dead` is passed because `run_WADERs`' dead-channel test needs charged pixels that a 16-row image
with no exposure may not have. In the rehearsal it dropped a working channel.

## The along-row drift step (`configs/processing.toml`, `[drift]`; `src/image_drift.py`)

Board 101 / PA03 images have a baseline drift at the start of every row's readout: the zero-electron level starts ~1 ADU
low and recovers over a few hundred columns. The gain is unchanged, but the noise is raised in the first ~200 columns.
None of the tested settings removes it (exposure, prescan length, register clear: see
[`real_image_checks/2026-10-16_PA03_PQ04_16rows.md`](real_image_checks/2026-10-16_PA03_PQ04_16rows.md)).

`process_packet.py` therefore checks every image (Gaussian fits of the zero-electron peak per column block). If a drift
is found:
* the **first 200 columns are cut**, the same for every packet;
* the slow remainder is subtracted;
* WADERs and the per-row λ fit use that corrected copy.

Images without a drift are used unchanged. Every label records `drift_corrected`, `drift_cut` and `drift_warning` (the
drift differs between the first and second half of the rows, so the correction may be less reliable; recorded, not
flagged). On the real 16-row images this gives fits with χ²/ndf ≈ 1 and σe labels that repeat to 0.6–2 %.

The standalone tool is `python src/image_drift.py <images> --out <dir>` (or `lvl1/drift_correct.py`).

## Rehearsal: the whole chain on simulated packets of the planned geometry

`lvl1/rehearse_lvl1.py --out <dir>` simulates hand-taken packets and runs every step for both WADERs routes:
* the records, traces, and 16×1000×500 images with a known truth (`simulate_manual_run.py`);
* `make_manual_manifest.py`;
* `process_packet.py`;
* the labels against the truth (`check_rehearsal.py`);
* `train_lvl1.py`.

The result is in [`rehearsal_report/REHEARSAL.md`](rehearsal_report/REHEARSAL.md) (with the per-label tables `4_check_*.md` and the truth `truth.json`). Median label error against the truth (6 packets, 18 labels):

| | σe | λ | gain |
|---|---|---|---|
| `run_WADERs` (default) | +1.6 % | +4 % | 0 % |
| one JSON, geometry set | −8 % | −12 % | +11 % |

**What the rehearsal showed:**
* the noisy amplifier (ch2) is correctly left unfitted;
* the quoted σe errors are sometimes too small, so they should not be trusted below ~3 %;
* at λ ≈ 0.02–0.03 e⁻/pixel, the 16 rows give dark current and CIC only to ~20–40 % each (with honest error bars), while the
  total λ is fine.

## What each step writes

* **`manual_packet.sh`**:
  * every trace is named `<PACKET_ID>_ref<set>` / `<PACKET_ID>_new<set>`; cdaq appends the geometry, channel, board and time;
  * a **packet record** `<RECORD_DIR>/<PACKET_ID>.json` holds the packet's name, the requested changes, and the trace and image settings.

  The command sequence and all safety checks are unchanged.
* **`src/make_manual_manifest.py`** writes a manifest (the format `take_packet.py` writes), and its status is `ok`, `check` or `no image`:
  * **settings**: read from the `.meta` files, i.e. what the boards *report*;
  * **requested changes**: each one is compared with those reports; a change that did not arrive sets the status to `check` (fix it, or give the manifest explicitly to process it anyway);
  * **image**: per board, the first image after the packet's last trace.

  `--legacy-run DIR --board 101` does the same for older runs without records, grouping files by identical settings (used for the 2026-10-12 packet tests).
* **`src/process_packet.py`** writes, in the output directory:
  * `dataset.csv`: one row per packet × board × channel: settings, the 85 trace features (mean over the trace sets and their scatter), the reference features, the labels with WADERs' errors, the image geometry, and the per-row λ split into dark current and CIC (`src/image_rows.py`);
  * `model_inputs/*.npz`: one file per model family (table below). Load one with `ml.real_inputs.load(path)`.

| File | Models | Contains (besides ids, settings, labels and label errors) |
|---|---|---|
| `features.npz` | Bayesian ridge, boosted trees, GP, NN ensemble | own trace features, own reference features |
| `curves.npz` | CNN on curves; D (pre-trained); C (deep kernel) | 14 trace curves and 14 reference curves, 256 points each, log scales applied |
| `physics.npz` | A, physics-structured (+ joint dark current / CIC) | own features, the Allan curve (common pattern removed), skips, per-row λ fit |
| `sets.npz` | E, set model | every trace set's Allan curve, separately |
| `pretrain.npz` | D's pre-training | the curves of every usable trace, no labels |

Nothing in these files is fitted to the data (no scaling, no filling of gaps). Each model's preparation happens inside its
training folds, so test packets never influence training.

## The models (`lvl1/train_lvl1.py`)

| Name | What | Family file |
|---|---|---|
| `trace_physics` | no learning: the noise measured on the trace's Allan curve, divided by the training mean gain (the baseline: how far the trace alone gets) | physics |
| `physics` | **A, the default**: trace physics plus a learned gain and a learned correction (GPs on settings + packet state + own features) | physics |
| `physics_joint` | A with λ from the joint dark-current / CIC fit of the per-row λ | physics |
| `physics_hier` | **A+B, a reference WITH labels**: A on hierarchical GPs told the module and amplifier; marked "(labels)" in the report. Comparing it with `physics` measures what amplifier labels add on the real CCD | physics |
| `sets` | E: A with the 10 trace sets combined by a learned set network | sets |
| `ridge`, `trees`, `gp`, `nn` | the first-round models on settings + own trace + own reference features | features |
| `cnn` | 1D CNN on the trace and reference curves + settings | curves |
| `pretrained` | D: the CNN with its encoder pre-trained on every unlabeled trace, then fine-tuned | curves + pretrain |
| `deep_kernel` | C: a GP on the pre-trained encoder's embedding (frozen; no amplifier term) | curves + pretrain |

Options:

| Option | Default | Effect |
|---|---|---|
| `--models` | all | which models to run |
| `--board` | 101 | which board's amplifiers to use (module 102 is not used for trace analysis) |
| `--folds` | 5 | number of cross-validation folds |
| `--include-flagged` | off | also use labels that WADERs flagged |
| `--allow-fake` | off | also use rehearsal packets |
| `--quick` | off | short training, to test the pipeline |
| `--out` | `<processed>/lvl1_results/<date>/` | where to write the results |

## The report

`report.md` / `report.json` / `predictions.csv`, per model and per target (σe, λ, gain):

| Score | Meaning | Good |
|---|---|---|
| error | RMS of log(predicted/true): the typical relative error | close to the label error |
| bias | mean of log(predicted/true) | ≈ 0 |
| shape | error after removing each amplifier's own mean error: is the dependence on the traces right? | ≈ error, or below |
| rank | Spearman correlation of predicted and true within each amplifier: is the order right? | > 0.9 |
| cover 68 / 95 | fraction of truths within ±1σ / ±1.96σ of the prediction | 0.68 / 0.95 |
| calibrated | the same after conformal calibration (each fold's factor from the other folds) | 0.68 / 0.95 |
| factor | how much the error bars must be scaled for 68 % coverage | ≈ 1 |
| label error | median error of the labels (WADERs) | — (no model can be judged below it) |

**A suggested pass mark for level 1** (to agree on):
* the best label-free model predicts σe on held-out packets within about twice the label error;
* within each amplifier it ranks σe correctly (rank > 0.9);
* its calibrated error bars cover 68 % / 95 % of the truths.

## Testing without real data

`lvl1/make_test_data.py` writes synthetic packets (the toy CCD world: image-only noise, per-row λ, intermittent pickup,
damaged trace sets) in exactly the `model_inputs` format, with the label errors WADERs would give. They go to
`lvl1/test_data/`, which is not in the repository; regenerate them with the command below. On 100 such packets (400
examples), the full run gives these errors (labels carry 1.5 / 3 / 0.8 %):

| model | σe | λ | gain |
|---|---|---|---|
| trace_physics (no learning) | 35.5 % (in level: the gain is unknown without labels; the settings' order is right, rank 0.89) | 30.4 % | 31.3 % |
| **physics (A, default)** | **5.5 %** | **4.9 %** | **0.9 %** |
| physics_joint | 5.5 % | 4.6 % | 0.9 % |
| physics_hier (A+B, **with labels**) | 5.9 % | 4.9 % | 0.8 % |
| sets (E) | 5.1 % | 4.9 % | 0.9 % |
| ridge / trees / gp / nn | 10.5–11.5 % | 4.4–21.4 % | 0.8–3.6 % |
| cnn | 6.8 % | 9.3 % | 1.2 % |
| pretrained (D) | 5.8 % | 11.6 % | 1.2 % |
| deep_kernel (C) | 10.8 % | 5.0 % | 0.9 % |

These numbers compare methods; they say nothing about the real CCD.

```bash
.venv/bin/python lvl1/make_test_data.py --packets 100 --unlabeled 300
.venv/bin/python lvl1/train_lvl1.py --inputs lvl1/test_data/model_inputs --quick
```

## First real images (2026-10-16, `lvl1/check_real_images.py`)

The first 16 × 1000 × 500 images (3 repeats at nominal on board 101, EXPOSURE 10 s) are summarised in
[`real_image_checks/2026-10-16_PA03_PQ04_16rows.md`](real_image_checks/2026-10-16_PA03_PQ04_16rows.md).

* **Row order:** λ grows with the row, so FITS row 0 is the first row read (no `reverse` needed).
* **Along-row baseline:** the active region starts ~0.4 ADU below the overscan and recovers along each row. Removing this
  trend before `run_WADERs` brings χ²/ndf from 2.4–5.1 to 1.2–2.1, removes a ~12 % σe offset, and makes the σe labels
  repeat to 0.5–3 % (instead of 4–6 %).
* **Amplifiers:** ch2 is fitted in only 1 of 3 images, so expect about 3 usable amplifiers per packet.

## Before and at the first real packet

* **On the DAQ machine:** run a planned packet with `DRY_RUN=1` first. Check that cdaq names the files `<PACKET_ID>_ref1_…`
  and that `~/packet_records` (or `RECORD_DIR`) is written and copied with the data.
* **`run_WADERs`** must be reachable from the processing computer: the path is set in `configs/processing.toml`, and it
  finds the `damic` conda environment itself.
* **Manifest:** `make_manual_manifest.py` reports status `ok`, every requested change arrived, and `trace_matches_image` is true (same NDCM and binning).
* **Labels:** WADERs labels without flags for the amplifiers used.
* **Row order:** the per-row dark current (`label.lambda_dark`) is positive. If it is negative on every image, FITS row 0 is the last row read: set `[rows] reverse = true` in `configs/processing.toml`.
* **ch2:** its gain cannot be measured from a 1×1 image yet (its electron peaks are not resolved); its labels will usually be flagged and left out.
