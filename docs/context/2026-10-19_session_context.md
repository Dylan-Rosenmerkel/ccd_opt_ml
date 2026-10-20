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

## 4. GP priors for many inputs / few points (literature item 3)

`src/ml/bogp.py` (`BoTorchGP`, BoTorch 0.18.1 + JAX/NumPyro, added to requirements-ml.txt): kind "dim" = SingleTaskGP
with the Hvarfner et al. 2024 prior (verified in the installed source: LogNormal(sqrt2 + log(d)/2, sqrt3), RBF);
kind "saas" = SaasFullyBayesianSingleTaskGP, NUTS 256/128/16 (quick 96/48/8). Inputs Normalize'd to the unit cube,
targets standardized, outlier rows (extra_var > 0) dropped. `make_gp` kinds "dim"/"saas"; `PhysicsStructured(lambda_kind=)`
swaps only the direct lambda GP. train_lvl1 extra models (not default): gp_dim, gp_saas, physics_dim, physics_saas,
physics_saas_lambda. Note: HierGP (physics) already had a weak dimension-scaled prior (log-normal at sqrt d, sd 3).

Full training, synthetic, 3 datasets each at 13 and 41 packets (seeds 0-2) + the 100-packet set; error % σe / λ / gain:

| model | 13 packets | 41 packets | 100 packets |
|---|---|---|---|
| gp (now) | 8.9 / 20.4 / 1.3 | 7.9 / 7.4 / 0.9 | 11.0 / 4.4 / 0.8 |
| gp_dim | 8.7 / 20.7 / 1.3 | 8.0 / 18.3 / 1.0 | 14.4 / 5.0 / 0.9 |
| gp_saas | 10.7 / 23.2 / 1.2 | 8.8 / 12.3 / 0.8 | 10.8 / 6.2 / 0.8 |
| physics (now) | 4.1 / 19.1 / 1.3 | 9.1 / 10.9 / 0.9 | 5.3 / 4.9 / 0.9 |
| physics_dim | 4.4 / 15.3 / 1.8 | 9.1 / 9.3 / 1.0 | 5.6 / 4.4 / 0.9 |
| physics_saas | 4.6 / 13.5 / 1.7 | 9.1 / 7.3 / 1.0 | 5.3 / 3.9 / 0.8 |
| physics_saas_lambda | 4.1 / 13.5 / 1.3 | 9.1 / 7.3 / 0.9 | – |
| physics_joint | 4.1 / 17.3 / 1.3 | 9.1 / 5.2 / 0.9 | (quick: 5.3 / 4.6 / 0.9) |

Conclusions: neither prior improves the features GP (gp_dim clearly worse on λ at 41 and σe at 100); their raw error
bars are better calibrated at 13 packets (factor 1-2 instead of 3), which conformal calibration fixes anyway. Inside the
physics model only λ gains (SAAS: 19→13.5, 11→7.3, 4.9→3.9 %); σe unchanged, gain slightly worse when all GPs are
SAAS → SAAS for λ only (physics_saas_lambda) keeps σe and gain identical. physics_joint is better at 41 packets
(5.2 %). Cost: SAAS λ ~1-2 min per fold alone (much longer when 7 runs share the CPU). Not tried: SAAS inside the joint
dark-current/CIC fit (JointLambdaGP), the natural combination.

### 4b. Decision (2026-10-20): SAAS for lambda, as default

User approved making SAAS-for-lambda the default and trying SAAS inside physics_joint.

* `ls_penalty()` in structured.py: "weak" (as before) or "saas" (SAAS prior for MAP fitting: tau ~ HalfCauchy(0.1),
  1/l^2 ~ HalfCauchy(tau), Jacobians included); used by `HierGP(ls_prior=)` and `JointLambdaGP(ls_prior=)`.
* `PhysicsStructured(joint_prior=, joint_select=)`: SAAS prior inside the joint model, or the joint model on the inputs a
  NUTS SAAS GP on log lambda finds relevant (relevance >= 0.1 x max; on a 41-packet set it kept 5 of 61 inputs, gap
  0.17 -> 0.04).

λ error, synthetic (3 datasets at 13 and 41 packets, 1 at 100); σe and gain identical in all; times alone on this Mac,
5 folds at 100 packets:

| λ fitted by | 13 | 41 | 100 | time |
|---|---|---|---|---|
| old GP (`physics_gp_lambda`) | 19.1 | 10.9 | 4.9 | 6 s |
| SAAS, full NUTS 256/128/16 (`physics_saasfull_lambda`) | 13.5 | 7.3 | 3.9 | 1143 s |
| **SAAS, short NUTS 96/48/8 (`physics`, default)** | 15.1 | 7.4 | 3.9 | 75 s |
| our GP + SAAS prior by MAP (`physics_saasmap_lambda`) | 14.8 | 9.7 | 5.0 | 9 s |
| joint, weak prior (`physics_joint_weak`) | 17.3 | 5.2 | 4.6 | 12 s |
| **joint, SAAS prior by MAP (`physics_joint`, default)** | 16.9 | 4.6 | 4.4 | 11 s |
| joint on SAAS-selected inputs (`physics_joint_select`) | 16.0 | 5.5 | 3.5 | very slow |

Run times measured with 7 jobs in parallel were up to 10x too long (JAX and torch each use all cores): time alone.

## 5. Still open

* Trace-settings decision (above); then the row-start pixel exclusion as a setting to match it.
* Flag test packets whose traces are unusual (e.g. floor-dominated physics term) and widen their error bars.
* Add the row-start drift to the simulated images, so the rehearsal checks the drift correction against the truth.
* Re-execute the tutorial notebooks if their printed numbers should reflect today's model changes (not done).
