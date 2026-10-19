# Session context, 2026-10-19

Continues `2026-10-16_session_context.md` (drift investigation, drift step in `process_packet.py`, commit 57ab15a).

## 1. Is the level-1 pipeline ready for real manual packets?

Checked end to end today:

* **Fresh rehearsal** (`lvl1/rehearse_lvl1.py`, 6 simulated packets, run_WADERs route, the drift step on): every step
  returns 0; 20/20 traces usable per packet, 3/4 labels unflagged (ch2 "not fitted", as on the real CCD); labels vs
  truth (median): σe +1.6 %, λ +4.1 %, gain 0.0 %, dark +0.5 %, CIC +5.4 %. Drift-free simulated images are left
  uncorrected.
* **Model inputs:** processing writes all five family files (`physics`, `sets`, `features`, `curves`, `pretrain`.npz)
  and **all 12 models (+ 2 new baselines) train** on processing output without reformatting. New packets: process them
  into the same output folder; `dataset.csv` and `model_inputs/` are rebuilt from every `*/result.json`.
* **Not yet tested:** a real manual packet (cdaq file naming with PACKET_ID, the copied record, manifest status). The
  image half was tested on real images 130/131, the trace half on the real 10-12 packets.
* **Open decision (user):** trace settings before the first packet: traces use `BOOLCLR=0, NDCMPRE=0`, images
  `BOOLCLR=1`; only the first pixel of each trace row is left out of the features (`trace_regions.py`, `p % NCOL == 0`),
  although real traces showed more disturbed row-start pixels. Changing this mid-dataset would split it.

## 2. Changes today (items 2-4 of the work list)

**Baselines in `train_lvl1.py`:** `mean` (training mean per target, training spread as error bar) and `settings`
(GP on the settings alone; no trace, no amplifier label). The report and the printed summary have a **vs mean**
column (error ÷ the mean baseline's). On the synthetic worlds most σe spread is between amplifiers (mean baseline
57 %), which settings alone cannot explain (settings ≈ mean for σe and gain).

**Why `physics` (A) had 18.5 % σe error on 41 synthetic packets** (4.2 % on 13, 5.5 % on 100):
* packet syn_00035 had 5 of its 10 trace sets damaged; the mean Allan curve's floor b ≈ 1 ADU made the trace physics
  ×2.5 too high (all 4 amplifiers);
* in the other folds that packet's huge residual was learned by the correction GP and applied (−0.45 in log) to
  packets 17 and 30 (1 damaged set each), whose trace physics was right within 5 %;
* median or trimmed set combination does not fix it (half the sets damaged; trimmed helps the 100-packet set
  5.3 → 4.0 % but not the 41: tuned to the synthetic glitch model, and on real data it could drop sets with genuine
  pickup; **not adopted**).

**Fix (`src/ml/structured.py`):** `robust_outliers()`; `PhysicsStructured(outlier_z=4.0)` down-weights training rows whose
residual is > 4 robust sd from the median (extra variance = deviation²) via the new `HierGP.fit(..., extra_var=)`
(added to the diagonal, not to the predictive label variance). Inherited by `physics_joint`, `physics_hier`, `sets`.
Result: 41 packets 18.5 → 15.0 % (only packet 35 left, which no mean-curve model can predict; `sets` 4.8 %);
100 packets 5.5 → 5.3 % (physics_hier 5.9 → 5.3 %); 13 packets unchanged. Packet 35's error bar stays too small
(4 %): an unusual trace (floor-dominated physics term) is not yet flagged.

**CNN divergence (`src/ml/models.py`, `src/ml/deep.py`):** with 6 packets the CNN predicted σe 1310 % / λ 2618 % off
(curve points with almost no spread among 12 training rows became huge standardized inputs). `_MeanVarHead`:
`INPUT_CLIP = 10` (standardized inputs clipped, at training and prediction, in CurveCNN, PretrainedCNN, MLPEnsemble),
`OUTPUT_MARGIN = 3` (a mean more than 3 training sd beyond the training labels is pulled to that edge, the distance
added to its sd). 6 packets: CNN σe 12 %, λ 84 %; 100 packets: unchanged within noise (guard off: 8.2 / 11.9 %,
on: 8.4 / 11.5 %).

**Synthetic results with everything (quick training; σe / λ / gain):**

| model | 13 packets | 41 packets | 100 packets |
|---|---|---|---|
| mean (baseline) | 57.8 / 21.8 / 31.3 | 57.5 / 28.6 / 31.3 | 57.1 / 30.4 / 31.3 |
| settings (baseline) | 57.8 / 17.0 / 31.3 | 57.5 / 26.8 / 31.3 | 56.9 / 23.1 / 31.3 |
| physics (A) | 4.2 / 14.0 / 1.3 | 15.0 / 10.5 / 0.9 | 5.3 / 4.9 / 0.9 |
| physics_joint | 4.2 / 14.2 / 1.3 | 15.0 / 5.0 / 0.9 | 5.3 / 4.6 / 0.9 |
| sets (E) | 6.0 / 14.0 / 1.3 | 4.8 / 10.5 / 0.9 | 5.0 / 4.9 / 0.9 |
| gp | 7.5 / 23.2 / 1.8 | 4.7 / 7.7 / 0.9 | 9.9 / 4.3 / 0.8 |
| cnn | 9.7 / 11.9 / 3.1 | 9.0 / 13.9 / 3.8 | 8.4 / 11.5 / 1.4 |
| pretrained | 16.4 / 15.1 / 8.9 | 7.0 / 22.1 / 3.2 | 8.0 / 21.8 / 2.9 |

Synthetic numbers compare methods only. The networks and deep_kernel are weak at the plan's size (13-41 packets).

## 3. Literature search

`docs/literature/ml_models_literature_review.md` (written by a search agent; citations spot-checked: arXiv:2509.21661
Cuevas-Zepeda et al., BO for sensor characterization on a low-noise CCD, and arXiv:2606.01427 Johnson et al., GPs often
beat TabPFN on scarce data, both confirmed; items the agent could not confirm are marked [unverified]). Shortlist:
1. TabPFN v2/2.5 (+TabICLv2) as a regressor and inside model A; compare at 13/40/100 packets (gated weights: pin version);
2. packet-aware conformal calibration (packet weights / one row per packet, bins of predicted sigma, worst-amplifier coverage);
3. SAAS / dimension-scaled length-scale priors for the GPs (also the level-2 surrogate);
4. trace-representation bake-off: MiniRocket vs frozen MOMENT/Mantis vs our encoder (needs raw skip sequences in model_inputs);
5. multi-output GP over settings with the trace noise as a cheap extra task (bridge to level 2).
Level 2: settings -> trace -> image as a "function network" BO with partial evaluations (Buathong et al. 2024),
step-limited safe BO (Kirschner et al. 2022), conformal is optimistic under BO feedback (Fannjiang et al. 2022).
No published work predicting image noise from readout traces was found.

## 4. Still open

* Trace-settings decision (above); then the row-start pixel exclusion as a setting to match it.
* Flag test packets whose traces are unusual (e.g. floor-dominated physics term) and widen their error bars.
* Add the row-start drift to the simulated images, so the rehearsal checks the drift correction against the truth.
* Re-execute the tutorial notebooks if their printed numbers should reflect today's model changes (not done).
