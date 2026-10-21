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

`lvl1/make_packet_plan.py` writes [`packet_plan/PLAN.md`](packet_plan/PLAN.md): **43 packets in three sessions (~12 h)**,
each a ready changes file `L1_NN.txt`.

| Session | Packets | What changes | Status |
|---|---|---|---|
| A | 14 (~3.8 h) | timing only: delayInt 300–3000, the horizontal clock hold, the reset hold, in a shuffled order; nominal packets at the start, middle and end; delayInt 300 repeated | **can be taken now** |
| B | 18 (~4.9 h) | one voltage group at a time (VDD, VR, RG high/low, SW low, H lows, V lows, TG, VSUB), one `max_step` either way, the two signs of a group far apart in time; delayInt 3000 repeated | provisional: the voltage windows are placeholders (`reviewed = false`) |
| C | 11 | joint changes (a Latin hypercube in the same box) and a final nominal | provisional, as B |

**Run order (2026-10-21, `docs/literature/landscape_grounded.md`):** session A no longer scans delayInt in increasing order
(a slow drift during the session would be aliased with the strongest knob): its values alternate low/high
(1500, 300, 3000, 500, 2000, 700) with the delay_H and delayRGH packets between them. In session B the two signs of a group
are taken in two passes (the second in reverse order), never back to back. Two replicates away from nominal (delayInt
300 late in A, delayInt 3000 in the middle of B) measure the pure error and the drift where the noise is not nominal.
**Take the packets in the plan's order.**

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
| `mean` | **baseline**: always the training mean of each target (its error bar: the training spread) | features |
| `settings` | **baseline**: a GP on the settings alone, no trace and no amplifier label | features |
| `trace_physics` | no learning: the noise measured on the trace's Allan curve, divided by the training mean gain (the baseline: how far the trace alone gets) | physics |
| `physics` | **A, the default**: trace physics plus a learned gain and a learned correction (GPs on settings + packet state + own features); λ by a SAAS GP (since 2026-10-19) | physics |
| `physics_joint` | A with λ from the joint dark-current / CIC fit of the per-row λ, with the SAAS sparsity prior on its length scales (since 2026-10-20) | physics |
| `physics_hier` | **A+B, a reference WITH labels**: A on hierarchical GPs told the module and amplifier; marked "(labels)" in the report. Comparing it with `physics` measures what amplifier labels add on the real CCD | physics |
| `sets` | E: A with the 10 trace sets combined by a learned set network | sets |
| `ridge`, `trees`, `gp`, `nn` | the first-round models on settings + own trace + own reference features | features |
| `cnn` | 1D CNN on the trace and reference curves + settings | curves |
| `pretrained` | D: the CNN with its encoder pre-trained on every unlabeled trace, then fine-tuned | curves + pretrain |
| `deep_kernel` | C: a GP on the pre-trained encoder's embedding (frozen; no amplifier term) | curves + pretrain |

**GP priors (2026-10-19/20, `src/ml/bogp.py`, BoTorch).** With ~60-320 inputs and at most a few hundred examples, a GP
that fits one length scale per input freely has more knobs than data. The SAAS prior (Eriksson & Jankowiak 2021)
switches inputs off unless the data need them. On synthetic data it helps λ only, so it is used for λ only:

| λ error (synthetic) | 13 packets | 41 packets | 100 packets |
|---|---|---|---|
| `physics` before (`physics_gp_lambda`) | 19.1 % | 10.9 % | 4.9 % |
| `physics` now (λ by a SAAS GP, short NUTS chains; ~1 min per run at 100 packets) | 15.1 % | 7.4 % | 3.9 % |
| λ by a SAAS GP, full NUTS chains (`physics_saasfull_lambda`; 19 min at 100 packets) | 13.5 % | 7.3 % | 3.9 % |
| λ by our GP with the SAAS prior fitted by MAP (`physics_saasmap_lambda`; seconds) | 14.8 % | 9.7 % | 5.0 % |
| `physics_joint` before (`physics_joint_weak`) | 17.3 % | 5.2 % | 4.6 % |
| `physics_joint` now (SAAS prior, MAP) | 16.9 % | 4.6 % | 4.4 % |
| `physics_joint_select` (joint model on the inputs a SAAS GP picks) | 16.0 % | 5.5 % | 3.5 % |

(3 datasets each at 13 and 41 packets, 1 at 100.) σe and gain are unchanged. Neither SAAS nor the dimension-scaled prior
(`gp_dim`) improved the features GP (`gp`). Extra models for these comparisons, run only when named in `--models`:
`gp_dim`, `gp_saas`, `physics_dim`, `physics_saas`, `physics_gp_lambda`, `physics_joint_weak`, `physics_joint_select`,
`physics_saasmap_lambda`, `physics_saasfull_lambda`. Run times: `physics` 75 s and `physics_joint` 11 s for 5 folds
at 100 packets on this Mac.

**Baselines.** A model is only useful if it beats `mean`; the report's **vs mean** column is each model's error divided
by `mean`'s (below 1: it learned something). `settings` answers the level-1 question directly: a trace model that does
not beat it has not shown that the traces help.

**Safeguards for small data** (2026-10-19):

* `physics` (and every model built on it: `physics_joint`, `physics_hier`, `sets`): a training packet whose residual
  (image noise − trace physics) lies more than 4 robust standard deviations from the others is down-weighted in the
  learned correction (it counts as about one ordinary point). Without this, one packet with mostly damaged trace sets
  bent the correction for every packet with similar traces (41 synthetic packets: σe error 18.5 % → 15.0 %; the
  remaining error is that one packet itself, which no model reading the mean trace curve can predict; `sets` gets it
  to 13 %).
* `cnn`, `pretrained`, `nn`, and the set network: standardized inputs are clipped to ±10, and a prediction more than 3
  training standard deviations beyond the training labels is pulled back to that edge, with the distance added to
  its error bar. With 6 packets the CNN had predicted σe 1300 % off; now 12 %. On 100 packets nothing changes.

Options:

| Option | Default | Effect |
|---|---|---|
| `--models` | all | which models to run |
| `--board` | 101 | which board's amplifiers to use (module 102 is not used for trace analysis) |
| `--folds` | 5 | number of cross-validation folds |
| `--group-by` | `setting` | CV groups: packets at identical settings share a fold (`setting`), or each packet alone (`packet`, the behaviour before 2026-10-21); see [The report](#the-report) |
| `--setting-tol` | 0.01 | settings count as identical within this fraction of each setting's range over the packets |
| `--n-boot` | 1000 | packet-bootstrap resamples for the intervals and the paired comparison (0: none) |
| `--paired-ref` | `physics` | the reference model of the paired comparison |
| `--holdout-session` / `--holdout-packets` | none | the prospective test: hold these packets out of the CV (a session of `--plan`, e.g. `C` = L1_33…L1_43, and/or a list of packet ids) |
| `--plan` | `lvl1/packet_plan/plan.json` | the packet plan read by `--holdout-session` |
| `--loao` | off | also score leave-one-amplifier-out (a new-module indicator; amplifiers × folds extra fits) |
| `--include-flagged` | off | also use labels that WADERs flagged |
| `--allow-fake` | off | also use rehearsal packets |
| `--quick` | off | short training, to test the pipeline |
| `--feature-tiers` | all | which trace features the models see: `all`, `none` (settings only), `auto` (per-model defaults from the tolerance test: gp/nn/ridge `A`, trees `all`, the rest `A+B`), or tier labels joined by `+`: evidence `A1`, `A2`, `A` (= A1+A2), `B`, `C` (`lvl1/feature_study/evidence_tiers.json`) and implementation `impl0`, `impl1`, `impl2` |
| `--tier-ablation [SET ...]` | off | train every model with each feature set (default `none A1 A A+B all impl0`) on the same folds and compare each with its own `--ablation-ref` run (default `all`), with paired bootstrap intervals; adds a "Tier ablation" section to the report |
| `--tier-file` | `lvl1/feature_study/evidence_tiers.json` | the evidence tiers |
| `--out` | `<processed>/lvl1_results/<date>/` | where to write the results |

## The report

`report.md` / `report.json` / `predictions.csv`, per model and per target (σe, λ, gain). The run also prints a one-line
summary per model (error with its 68 % interval) and the paired comparison.

| Score | Meaning | Good |
|---|---|---|
| error | RMS of log(predicted/true): the typical relative error | close to the label error |
| vs mean | error ÷ the `mean` baseline's error | well below 1 |
| bias | mean of log(predicted/true) | ≈ 0 |
| shape | error after removing each amplifier's own mean error: is the dependence on the traces right? | ≈ error, or below |
| rank | Spearman correlation of predicted and true within each amplifier: is the order right? | > 0.9 |
| cover 68 / 95 | fraction of truths within ±1σ / ±1.96σ of the prediction | 0.68 / 0.95 |
| calibrated | the same after conformal calibration (each fold's factor from the other folds) | 0.68 / 0.95 |
| factor | how much the error bars must be scaled for 68 % coverage | ≈ 1 |
| 68 % CI | the 16–84 % range of the error over packet-bootstrap resamples (below) | narrow compared with the gaps between models |
| worst amp 68 | the calibrated 68 % coverage of the worst-covered amplifier (the average can hide one badly covered amplifier; per amplifier in `report.json`, `by_amp`) | ≈ 0.68 |
| CRPS | continuous ranked probability score of the Gaussian prediction in log space, mean over rows | lowest (≈ 0.56 × error for a calibrated Gaussian) |
| label error | median error of the labels (WADERs) | — (no model can be judged below it) |

**CRPS.** For a prediction N(μ, σ²) of log y it is σ·[z(2Φ(z)−1) + 2φ(z) − 1/√π] with z = (log y − μ)/σ (Gneiting &
Raftery 2007), in the units of the log target (0.05 ≈ 5 %). It is a *proper* score: it is lowest when σ is the true
spread, so it rewards accuracy and honest error bars in one number. A perfectly sharp prediction (σ → 0) scores its
absolute error; a calibrated Gaussian scores on average σ/√π ≈ 0.56 × its RMS error; too small or too large error bars
both score worse. It uses the raw error bars (before the conformal calibration).

**How the CV is grouped.** The rows of one packet are always in the same fold. In addition (default `--group-by
setting`), packets taken at *identical settings* share a fold: the nominal packets (L1_01, L1_07, L1_14, L1_32, L1_43)
and the two replicates with their originals (L1_13 with L1_04, L1_23 with L1_06). Otherwise a held-out nominal packet
has its twins in the training set, and the score of a "new" setting is flattered (Saeb et al. 2017; level 2 always
predicts at new settings). The groups come from the settings (`theta`, as the boards report them): two packets are
identical when every setting differs by at most `--setting-tol` (1 %) of that setting's range over all packets, and
chains of identical packets form one group. The run prints the number of groups and the merged ones (e.g.
`L1_01+L1_07+L1_14+L1_32`); check them. With `--group-by packet` every packet is its own group. `predictions.csv` has a
`cv group` column.

**The intervals (packet bootstrap).** The out-of-fold predictions are resampled by packet, with replacement, 1000 times
(no refitting: each packet keeps the prediction it got in the CV). Every score is recomputed on each resample; the report
gives the 16–84 % range ("68 %") and the 2.5–97.5 % range ("95 %"), in the main table for the error and in the section
*Bootstrap intervals* for error, vs mean, shape, rank, calibrated coverage, worst amplifier and CRPS. How to read them:
* two models whose 68 % intervals overlap widely are not distinguished by these packets; but the intervals of two models
  are correlated (same packets), so for "is A better than B" use the paired comparison, not the overlap;
* the width shrinks roughly as 1/√(packets) (synthetic, 13 → 41 packets: the 68 % range of `gp`'s σe error 6.3–8.5 → 4.2–5.2 %,
  of `physics_joint`'s λ error 12.7–18.6 → 4.0–5.0 %);
* with ~40 packets they are rough: a bootstrap over few clusters tends to be too narrow, and
  the conformal factor is not re-estimated per resample. Read them as "at least this uncertain".

**The paired comparison (vs `physics`, `--paired-ref`).** On the same resamples, the difference of each model's error and
the reference's (Δ error, in percentage points of log error; negative = the model is better), its 95 % interval, and
P(better), the fraction of resamples in which the model's error is the lower one (equivalently its mean squared log error
per row). A difference marked \* has a 95 % interval that excludes 0. With 14+ models compared on the same ~40 packets,
the best-looking one is partly lucky (Cawley & Talbot 2010): **change the default model only on a difference marked \*,
and confirm it on the prospective test.** JSON: `results[model]["paired_vs_physics"]`.
Example (41 synthetic packets, `--quick`): `mean` vs `physics` σe +42.5 pp [+32.0, +53.9] \*; `physics_joint` vs
`physics` λ −3.6 pp [−6.4, −1.1] \*; but `sets` vs `physics` σe −10.0 pp [−19.9, +0.9], P(better) 0.81: the 5.0 % vs
15.0 % gap is mostly one damaged packet, and `physics`' own σe interval is 4.4–20.7 %. A model with exactly the
reference's predictions for a target (e.g. `physics_joint`'s σe and gain are `physics`') is shown as "same".

**The prospective test (session C).** Session C (L1_33…L1_43, joint Latin-hypercube changes) is declared the
prospective test set. The workflow:
1. after sessions A and B: run `train_lvl1.py --inputs … --holdout-session C` (or before C exists, just `--inputs …`)
   and choose the model(s) on the cross-validation of A+B (main tables and paired comparison);
2. freeze that choice (write it down, with the date and the commit);
3. after session C: run once more with `--holdout-session C`. The section *Prospective test (held-out packets)* has every
   model trained once on all A+B packets (no CV) and scored on the C packets only, with the conformal factor taken from
   the A+B cross-validation, and its own paired comparison. Read it once, as the test of the frozen choice; do not go back
   and pick the model that happens to win on C.

`--holdout-session C` takes the session's packet ids from `lvl1/packet_plan/plan.json` (`--plan`); a processed packet id
matches when it equals a planned id or continues it with a non-digit (`L1_33`, `L1_33b`, `L1_33_retake`, not `L1_330`).
`--holdout-packets` takes ids directly. A held-out packet whose settings also occur in training (L1_43 is nominal) is
named in the report: it is in the scores, but it is not a new setting. JSON: `report.json["holdout"]`.

**Leave one amplifier out (`--loao`).** Each amplifier is predicted by models trained on the other amplifiers only,
crossed with the packet folds (so the test packets are not in training either). Level 1 always has the same amplifiers
in training, so this is only an indicator of how a model would do on a new module; it is expected to be poor, and a
model that fails it is not wrong for level 1. It costs amplifiers × folds fits per model.

**A suggested pass mark for level 1** (to agree on):
* the best label-free model predicts σe on held-out packets within about twice the label error, and clearly beats
  both baselines (`mean`, `settings`);
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
