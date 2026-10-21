# What in the ML landscape we can actually use: grounded triage

*Written 2026-10-21 by the grounded analyst of the three-agent team. Input: `docs/literature/ml_landscape_catalogue.md`
("the catalogue", entry IDs S01–U08, fields B1–B29, problem shapes C1–C22), the earlier review
`docs/literature/ml_models_literature_review.md` ("the prior review"), `lvl1/README.md`, `lvl1/train_lvl1.py`,
`docs/model_summary/make_model_summary.py`, `docs/context/2026-10-19_session_context.md`, the 2026-10-12 trace-feature
slides (`docs/slides/oct12_features.py`, `make_slide_figures.py`) and `lvl1/packet_plan/PLAN.md`. The question asked here
is narrow: what can we implement and test with our own data in the next weeks to months? Far-fetched cross-field ideas are
left to the exploratory analyst. No code was changed.*

---

## 0. Bottom line

1. **At ~120 examples (41 packets × ~3 usable amplifiers) the gains will come from evaluation, labels and physics, not from
   new model families.** The synthetic results already have the physics models near the label error (σe 5 %, gain 0.9 %).
   Most of the catalogue (deep architectures, self-supervision, RL, generative models, foundation models) needs 10³ or more
   examples, or a simulator we do not have. These are rejected below, with reasons.
2. **The real-data finding changes what has to be tested.** The hand-made trace numbers tell the settings apart perfectly
   (nearest centroid 35/35), but the noise per skip barely moved for the voltage changes tested on 2026-10-12. On real
   data most of the σe variation will therefore be (a) between amplifiers and (b) due to `delayInt`. A model can pass "beat
   `mean`" just by recognising the amplifier, and it can look good on rank just by reading `delayInt`. The baseline that
   matters is therefore **settings + the same amplifier's reference trace** (no trace at the new settings). That baseline
   does not exist yet. It is also exactly the Level-2 question, and it is the battery-literature question too: does the
   early signal add anything beyond what is already known? (§2 H4)
3. **The cheapest high-value change is to the data taking, before session A.** At present `delayInt` is scanned in
   increasing order (300 → 3000, packets L1_02–07), so any slow drift is aliased with the strongest knob. Randomise
   the order and add two non-nominal replicates (§4).
4. **The validation needs error bars of its own.** With 41 packets, 14+ models and 5 folds, differences of 1–2 points are
   noise, and choosing the best of 14 on the same folds is optimistic (Cawley & Talbot 2010). Add packet-bootstrap
   intervals and paired differences, and use CRPS. Group the nominal repeats as one setting. Declare session C a
   prospective test set now (§2 H1).

---

## 1. Triage table

"Have" means implemented in `lvl1/train_lvl1.py`, `src/ml/*` or processing. Value is the expected value *for us at
~120 examples*. Effort: **S** < 1 day, **M** 1–3 days, **L** > 1 week. Priorities marked "L2" are high for Level 2 but wait
until Level 1 is settled.

| Entry | Addresses | Have? | Value | Effort | Priority | Reason (one line) |
|---|---|---|---|---|---|---|
| **U08** use-case-matched (grouped) validation | validation | partly (packet-grouped 5-fold) | high | S | **HIGH (H1)** | Group the nominal repeats as one setting, add a leave-session-out split, and use packet-bootstrap CIs. This is what makes any comparison trustworthy. |
| **U03** proper scores (CRPS), recalibration | validation, uncertainty | partly (coverage, factor) | med | S | **HIGH (in H1)** | One number that rewards both accuracy and honest σ. It has a closed form for our Gaussian-in-log predictions. |
| **U04** conformal (+ hierarchical, prior review §5.2) | uncertainty | partly (cross-conformal, rows treated as exchangeable) | med | S | **HIGH (in H2)** | Packet weights and Mondrian bins by novelty. The guarantee only matters where the model is unsure. |
| **P15** anomaly / novelty detection | uncertainty, QC | new (open item in the 10-19 context) | high | S | **HIGH (H2)** | Synthetic packet 35 (damaged sets) got a 4 % error bar on a ×2.5 error. Unusual traces must widen σ. |
| **P16** shift detection | drift | new | med | S | MED-HIGH (in H2) | A two-sample check of each packet's *reference* traces against earlier nominal packets. These traces are free, taken at fixed settings. |
| **U05** simulation-based calibration (of the labels) | labels | partly (6-packet rehearsal) | high | S–M | **HIGH (H3)** | The rehearsal found σe label errors "sometimes too small". Every known-noise GP and the pass mark depend on them. |
| **S15** mixed effects / variance components | labels, validation | partly (`physics_hier`) | med | S | MED-HIGH (in H3) | Split the label variance of the nominal repeats into amplifier, packet-state and residual parts. That gives the floor for Level 1. |
| **B2** PCH-EM (Hendrickson & Haefner) | labels | equivalent exists (WADERs ML fit of Poisson–Gaussian peaks) | low–med | – | MED (reference for H3) | It is the same estimator family. Borrow its uncertainty analysis, not the code (§3). |
| **P18 / B11** early prediction → BO | Level 1→2 framing | partly (`trace_physics`, toy MF in `bo_advanced.py`) | high | S | **HIGH (H4)** | Frames the decisive ablation: what does the trace *at the new settings* add over settings + reference? |
| **S13** multi-output / multi-fidelity GP | Level 1→2 bridge | partly (`physics_joint` 2-output; toy MF screen) | high | M | **HIGH (H7)** | Settings → (trace σ_ADU, image labels). It uses every unlabelled trace and gives the Level-2 surrogate. Do it after H4. |
| **S03** PLS | small-n regression on curves | new | med | S | **HIGH (H5)** | The chemometrics/virtual-metrology standard for "collinear curve in, scalar out" at 10–10³ rows. A missing rung between ridge and CNN. |
| **S02** elastic net | small-n regression | new (BayesRidge only) | med | S | **HIGH (in H5)** | Severson's 124-cell model. A sparse, readable baseline with stable selection of correlated Allan points. |
| **S18** FDA / functional PCA | curves | new | low–med | S | MED (an arm of H5) | FPCA scores → GP is the GP-friendly way to use the 14×256 curves. |
| TabPFN (prior review §1; catalogue D01/B25 pointers) | small-n regression | new | med | S | MED-HIGH (arm of H5) | Still untested. About 30 lines. Johnson et al. 2026 warn that GPs often win when data are scarce. |
| **S25** spectral estimation → CDS filter physics | trace representation, Level 2 prior | partly ("noise vs window length" reintegration, within-cycle and CDS spectra are stored) | high | M | **HIGH (H6)** | Predicts σ at an *untried* `delayInt` from the nominal reference trace, and session A tests it directly. (Multitaper itself: LOW; see H6.) |
| **O04** safe / step-limited BO (prior review §7.3) | Level 2 | partly (trust region in `optimize.py`/`bo_advanced.py`) | high (L2) | M–L | MED now / HIGH-L2 | Kirschner et al. 2022 matches our one-`max_step` rule. Wait for Level-1 results. |
| **O09** multi-fidelity / early stopping | Level 2 | toy (`bo_advanced` trace-only screen) | depends on H4 | M | MED (conditional) | Only worth building if H4 shows that the trace adds information. |
| **O03** multi-objective BO (qNEHVI) | Level 2 | partly (ParEGO toy) | med | S | MED-L2 | A BoTorch swap once a real surrogate exists. It avoids fixing the σe/λ weight in advance. |
| **O02** TuRBO / trust regions | Level 2 | toy | med | S | MED-L2 | Already in `bo_advanced.py`. A trust region also implements the step limit. |
| **O05** prior-informed BO (πBO) | Level 2 | new | med | S–M | MED-L2 | Expert or physics beliefs about the optimum. The H6 physics prior *mean* is the stronger form of the same idea. |
| **O08** human-first BO (Kanarik 2023) | data strategy, Level 2 | implicit in the 41-packet plan | med | none | MED | Confirms the plan (expert packets first, BO for fine tuning). It informs when to hand over (§3). |
| **P10 / O20** active learning, Bayesian experimental design | data taking | new | med | S | MED | Hold back 3–4 session-C slots for model-chosen settings (§4). Keep the Latin hypercube for the rest. |
| **T06** stacking of predictive distributions | accuracy, uncertainty | new | low–med | S | LOW-MED | An equal-weight average of `sets` and `physics_joint` is worth one line. Learned weights are noise with ≤ 41 packets (the catalogue says so too). |
| **F02** MiniRocket (prior review §2.1) | trace representation | new; needs raw skip sequences in `model_inputs` | low–med | M | MED-LOW | The information is second-order, and the physics already sits near the label error on synthetic data. Keep it as a later check. |
| **F01** catch22 / tsfresh | trace representation | new | low | S | LOW | Useful only as a "did we miss a feature?" screen. With 120 rows and hundreds of features, false discoveries dominate. |
| **D15** wavelet scattering | trace representation | new | low | S | LOW | First-order scattering ≈ dyadic band power ≈ our Allan and PSD bands. |
| **S08/S09** celerite, spectral-mixture GP on raw skips | inverse problem | new | low–med | M | LOW | With 10 pixels × 500 skips × 10 sets, the PSD and Allan estimates are already precise. Revisit only if the uncertainty of the Allan fit (a, b) matters. |
| **G07** NPE for Allan parameters (prior review §4.2) | inverse problem | new | low | M | LOW | Same reason as S08/S09. The simulator gap (real pickup) adds risk. |
| **S14** history matching | Level 2 safety | new | low–med | M | LOW-MED (L2) | An implausibility filter on candidates (§3). Our hard limits plus the step limit already cover the hardware safety. |
| **O06 / O19 / O14 / U04-ACI** drift-aware BO, EWMA run-to-run, extremum seeking, adaptive conformal | Level 3 drift | new | med (L3) | S–M | LOW now | Real, but they need a stream of packets we will not have before Level 3. Note EWMA plus a deadband as the simplest Level-3 drift loop. |
| **S20 / S24** Kalman drift model, changepoints | drift | new | low | S–M | LOW now | 5 nominal packets per 41: a linear trend or a two-sample test is enough. |
| **U06/U07** SHAP, interpretable-by-design | insight | partly (SAAS relevances) | low–med | S | LOW-MED | SAAS relevances already say which inputs matter, at no extra cost. Use them in the report. |
| **S26** symbolic regression (PySR) | insight | new | low | S | LOW | σ vs `delayInt` already has a known law (Allan model; CDS filter, H6). |
| **B23** operating-condition normalisation | features | new | low–med | S | LOW-MED | Trace/reference *ratio* features (new − ref in log) help the linear arms of H5. GPs can learn them anyway. |
| **S11** heteroscedastic GP | uncertainty | have known label noise | low | – | LOW | Known per-row label variances are already in the likelihood. |
| **S16** GPBoost / MERF | grouping | new | low | S | LOW | ~4 rows per packet: the random-effect variance is poorly identified (catalogue note). |
| **P09** multi-task | targets | partly (`physics_joint`) | low | – | have | – |
| S07 GP, S12 DKL, U01 ensembles, D07/S19 sets | – | have | – | – | have | `gp`, `deep_kernel`, `nn`/`cnn` ensembles, `sets` (E). |

**Rejected for our use (with the reason):**
* **Too data-hungry at n ≈ 120, or no pretrained source that fits:** D02–D06, D08, D10, D13, D14, D16, D17 (CNN/RNN/
  transformer/Mamba/GNN/neural ODE/operators/MoE/FiLM/N-BEATS/NPs); P01–P05 (self-supervision, PEFT; our model D already
  tries this and trails the physics models); P17 science foundation models; G01–G05 (VAEs, flows, diffusion, GANs, EBMs);
  T01/T02/T04 (trees: `trees` already shows they are weak here).
* **Wrong problem shape:** S21 HMMs (damaged sets are handled by E); S22 GARCH/HAR; S27 SINDy; S28 DMD; S29 reservoirs;
  D11 INRs; D12 KANs; F03 path signatures (the amplifiers are traced one at a time, since `FILTER` selects one, so there are
  no simultaneous channels); B4 DeepClean witness channels (same reason); F04 augmentation (most augmentations change the
  noise label).
* **Not needed, or covered by the GP:** S04/T05 GAM/EBM; S05/T03 quantile models; S06 kernel ridge; S10 deep GPs; S17 PCE
  (needs input distributions; GPs handle known noise better); S23 DP mixtures.
* **We can intervene, so observational causal tools are unnecessary:** P13, P14, O07.
* **Too few tasks or environments:** P06–P08 (domain generalisation, test-time adaptation, meta-learning); P11 continual
  learning (refitting is cheaper); P12 federated; B21 matrix factorisation for a "new amplifier".
* **Wrong scale or immature:** O10 batch BO (one setting per packet); O11–O13 bandits/CMA-ES/NSGA-II (optimisers of the
  acquisition at most; BoTorch already does that); O15–O18, O24 RL/MPC/planning (need a simulator or 10⁴+ interactions;
  our system is static per setting); O21/O22 LLM optimisers and agents (reproducibility; Level-3 interface at most);
  P19; P20 AutoML (on 41 packets it optimises fold noise).
* **Conceptual analogies with nothing to implement:** B9, B13, B14 (DNSMOS), B22 histograms, B24, B27 TrueSkill, B28.

---

## 2. High-priority items

Every Level-1 experiment below uses the existing harness. A new method `def name(self, train, test) -> (mean, std)` goes
on `Runner` in `lvl1/train_lvl1.py`, on log targets, with the packet-grouped folds. The pass marks refer to the report's
columns: error, vs mean, shape, rank, calibrated 68/95, factor. Synthetic tests use `lvl1/make_test_data.py` at 13 / 41 /
100 packets, 3 seeds each. All conclusions are paired per packet against `physics` (H1).

### H1. Use-case-matched validation with error bars on the scores (U08, U03; Cawley & Talbot 2010)

**In our terms.** The packet-grouped CV is right (Saeb et al. 2017: record-wise CV "often massively overestimates" accuracy;
verified, GigaScience 6:gix019). It still has three gaps.

* **The nominal repeats.** L1_01/08/13/30/41 have identical settings. When one of them is held out, its twins sit in the
  training set, so 5 of 41 test packets are "seen settings". Level 2 always predicts at *new* settings.
* **The scores carry no uncertainty.** We pick among 14+ models on the same 41 packets. Cawley & Talbot (JMLR 11:2079,
  2010; verified) show that the variance of the CV estimate alone causes selection overfitting. The resulting loss can be
  as large as the gaps between algorithms.
* **The amplifier dimension.** The same 3 amplifiers sit in training and test. That matches Level 1 on this CCD, but it is
  "record-wise" with respect to amplifiers. The companion commentary to Saeb (Little et al. 2017, GigaScience gix020;
  abstract seen) notes that grouped CV still lets a model exploit identity–outcome confounding. Here that means the model
  can recognise the amplifier from its trace, which H4 measures.

**Plug-in.**
* `main()` in `train_lvl1.py`: add `--group-by {packet,setting}`. The setting key is the rounded `theta`, so all nominal
  packets fall in one group. `grouped_folds` takes any group array.
* Add `--split session`: train on A+B, test on C. Read the session from the packet id or `plan.json`.
* `scores()`:
  1. **CRPS** of the Gaussian in log space: `σ·[z(2Φ(z)−1) + 2φ(z) − 1/√π]`, with z = (y−μ)/σ (Gneiting & Raftery
     2007).
  2. **Packet bootstrap.** Resample packets with replacement (2000×) and report the 68 % interval of each error.
  3. **Paired difference** in RMS log error against `physics` on the same bootstrap draws.
  4. **Worst-amplifier coverage** (calibrated 68/95 per amplifier, minimum).
* Run 5 fold seeds (`src/ml/benchmark_seeds.py` exists) and average.

**Experiment.**
* On synthetic data at 41 packets: how wide are the bootstrap intervals? This tells us which current differences are real.
  For example, is `sets` 4.8 % vs `physics` 15.0 % at 41 packets driven by the single packet 35?
* On real data: report both groupings.
* Declare now that session C (11 packets, joint Latin-hypercube changes) is a **prospective test**. Freeze the model
  choice after A+B, then score C once.

**Adopt if:** always (this is measurement). Change the default model only when the paired interval excludes 0.

**Risks.** Bootstrap over 41 clusters is itself rough (it underestimates the variance for a few clusters). Setting-grouping
leaves fewer folds. **Package:** numpy/scipy (`properscoring`/`scoringrules` optional). **Effort:** S (half a day).

### H2. Error bars that know when a trace is unusual (P15, P16, U04; Dunn et al. 2023)

**In our terms.**
* On 41 synthetic packets, packet 35 (5 of 10 trace sets damaged) got a ×2.5 error with a 4 % error bar (context
  2026-10-19 §2). The cross-conformal factor in `scores()` is one global number, so it cannot help that row.
* Fit a novelty score on the training rows, inside the fold. The unlabelled reference traces from `pretrain.npz` may be
  added, since they carry no labels. Then calibrate *conditionally on that score*: **Mondrian conformal**, with the bins set
  by novelty.

**Plug-in.**
* A helper in `src/ml/calibrate.py`: `novelty(train_X, test_X)`.
  * Inputs: standardised `_features` of the own trace, plus the physics residual features (Allan floor b, the `a/b` ratio,
    set-to-set scatter).
  * Method: PCA to 5–10 components, then a robust Mahalanobis distance (`sklearn.covariance.MinCovDet`), with
    `IsolationForest` as an alternative.
* `scores()`:
  * replace `conformal_q(z[other], c)` by a quantile with weight 1/n_packet per row (Dunn, Wasserman & Ramdas, JASA 2023;
    prior review §5.2);
  * compute separate factors for the "usual" and "unusual" bins (e.g. novelty above the training 90th percentile).
* `process_packet.py` already writes quality flags. A per-trace novelty score there would serve as a data-quality monitor,
  in the spirit of B1's autoencoder DQM, but a linear method suffices at our size.
* **P16, the same machinery on the reference traces.** Compare each packet's reference-trace features with the previous
  nominal packets: a classifier two-sample test or MMD, as in Rabanser et al. 2018 (catalogue ref). The reference traces
  are taken at fixed settings, so a significant shift means the CCD changed, not the settings.

**Experiment.**
* Synthetic, 41 packets × 3 seeds, which contains damaged-set packets: calibrated coverage *on the flagged rows* and
  overall, worst-amplifier coverage, and mean interval width.
* Baseline: the current global cross-conformal factor.

**Adopt if:**
* flagged-row 68 % coverage rises from its current level to ≥ 0.6;
* overall coverage stays within 0.68 ± 0.05;
* mean width grows by < 10 %.

**Risks.**
* At 120 rows the bins hold ~12 unusual rows, so the factors are coarse. Merge the bins if a bin has < 8 packets.
* "Unusual" is not the same as "wrong". Widening on a genuine but rare setting is acceptable, because Level 2 *should* be
  cautious there.

**Packages:** scikit-learn (mature), `crepes` (Mondrian, mature) or our own 20 lines. **Effort:** S (1 day).

### H3. Audit the label errors before trusting the known-noise likelihood (U05, S15, B2)

**In our terms.**
* Every GP uses `y_var = Y_err²` from WADERs, and the pass mark is "within about twice the label error".
* The rehearsal found that the quoted σe errors "are sometimes too small" and "should not be trusted below ~3 %"
  (`lvl1/README.md`).
* On real nominal images the labels repeat to 0.6–2 %.

Simulation-based calibration (Talts 2018) turns this into a number: across simulated images with known truth,
z = (label − truth)/error should be N(0, 1).

**Plug-in, no model change.**
1. Simulate. `lvl1/rehearse_lvl1.py` / `simulate_manual_run.py` already makes 16×1000×500 images with a known truth.
   * Run ~60–100 images over the planned label range: σe 0.15–0.3 e⁻, λ 0.01–0.05, both gains.
   * Collect z for σe, λ, gain, dark and CIC.
   * Fit one scale factor per target. If needed, fit it as a function of λ: at λ ≈ 0.02 the 1 e⁻ peak holds few pixels, so
     the gain and λ errors are the fragile ones.
2. Real data, after session A. Take the 3–5 nominal packets per amplifier and fit a variance-components model
   (`statsmodels.MixedLM`): log y ~ amplifier + (1 | packet) + residual.
   * Compare the residual sd with the quoted errors.
   * Read the packet variance as the *state drift floor*. No trace model can beat it without seeing the state, and the
     reference traces are the only way to see it.
3. If the factor is > 1.3, apply it where `dataset.csv` writes the label errors (a `[labels] error_scale` entry in
   `configs/processing.toml`, read in `process_packet.py`). The "label error" column of the report changes with it.

**Adopt if:** the simulated z-sd differs from 1 by more than ~25 %, or the real nominal repeat scatter exceeds the quoted
error by more than ~1.5×.

**Risks.**
* The simulator has no row-start drift (an open item in the 10-19 context), so the drift-correction part of the label error
  is not covered. Add the drift to the simulated images first, or treat the real repeats as the arbiter.
* Five nominal repeats give a variance estimate good only to ~±35 %.

**B2 link.** PCH-EM (Hendrickson & Haefner, arXiv:2302.00090, JEDS 2023; abstract verified) is the same estimator family
as WADERs: a maximum-likelihood Poisson–Gaussian mixture for deep sub-electron read noise. It claims lower uncertainty
than photon transfer. Borrow the idea of checking our quoted errors against Monte-Carlo truth, which is what they did. No
reason to replace WADERs.

**Effort:** S–M (scripts plus WADERs run time).

### H4. The decisive ablation: settings → + reference trace → + trace at the new settings (P18/B11 framing, S13)

**In our terms.**
* Battery work asks whether a cheap early measurement predicts the expensive outcome beyond what is known in advance
  (Severson et al. 2019, 124 cells, first 100 cycles, 9.1 % test error; Attia et al. 2020 used it inside BO: 224
  protocols in 16 days instead of > 500; both verified at abstract/record level).
* Our version has three rungs:
  1. `settings` (have);
  2. **`settings_ref`, new**: settings + the *same amplifier's reference-trace features* (taken at nominal in the same
     packet) + packet state. This knows the amplifier and the CCD's state, but not the trace at the new settings.
  3. `physics` / `sets` (own trace at the new settings).
* The real-data finding (separation 35/35, but the noise per skip barely moved) makes rung 2 the honest bar.
  * If rung 3 ≈ rung 2, the trace at new settings adds little. Level 1 then "passes" only by recognising amplifiers, and
    Level 2 can be single-fidelity (no trace-only screening).
  * If rung 3 ≫ rung 2, multi-fidelity BO (O09, prior review §7.2) is justified.

**Plug-in.** `Runner.settings_ref`:

```python
d = self.f("features")
x = np.hstack([d["theta"], d["X_ref"], packet_state(d["packet_id"], d["X_ref"])])
xtr, xte, _ = prep_columns(x[train], x[test])
return models.GP(...).fit(xtr, self.Y[train]).predict(xte)
```

Also consider a SAAS variant (`BoTorchGP("saas")`). There are ~200 inputs, and the 10-19 tests showed SAAS helps mainly
λ, so run both.

A sharper variant, `ref_physics`, is model A with the Allan curve of the *reference* trace in place of the own trace.
It needs the reference Allan curve in `physics.npz` (`curves_ref` already has the Allan curves, so it is a small
`real_inputs.py` addition).

**Experiment.** Real 41 packets (and session A alone). Paired packet-bootstrap difference of error, shape and rank (H1),
rung 3 − rung 2, per target. On synthetic data the answer is known by construction (traces carry the signal), so this is a
real-data experiment.

**Adopt / decide:**
* Level 1 "passes" only if `physics` or `sets` beats `settings_ref` with a paired interval excluding 0 for σe.
* If the gap is < ~1 label error, record that the trace's value is amplifier and state recognition, and drop trace-only
  screening from the Level-2 design.

**Risks.** Session A is all timing. Within-amplifier variation comes almost entirely from `delayInt`, which `settings`
knows, so the comparison is decided by the voltage sessions B and C. **Effort:** S (half a day).

### H5. Cheap linear models on curves and features: PLS, elastic net, FPCA→GP (+ TabPFN) (S03, S02, S18)

**In our terms.** Between `ridge` (85+85 features + settings) and `cnn` (14×256 curves), the standard small-n tools for
"curve in, scalar out" are missing:
* **PLS**, the chemometrics and virtual-metrology workhorse (Wold 2001; catalogue S03, B7);
* the **elastic net**, Severson's model on 124 cells (S02);
* **functional PCA scores → GP** (S18).

None will beat `physics` for σe. Their job is to tell us whether `cnn`/`pretrained`/`deep_kernel` learn anything a
linear projection does not. If not, the network family can leave the default run, which saves time and false hope.

**Plug-in.** New `Runner` methods on the existing families:
* `pls`: `_curves(train, test)`. Flatten the own and reference log-curves (28×256), concatenate the standardised scalars,
  then `sklearn.cross_decomposition.PLSRegression`. Choose n_components ∈ {1..10} by an *inner* packet-grouped CV on the
  training rows. σ = the inner-CV residual sd per target (homoscedastic; conformal does the rest).
* `enet`: `_features` + `ElasticNetCV(l1_ratio=[.1,.5,.9], cv=GroupKFold)`, same σ rule. Optionally add the log
  trace/reference ratio features (B23).
* `fpca_gp`: PCA (fit on training rows plus the unlabelled `pretrain` curves, which is legitimate) to ~5 scores per curve
  type → `models.GP`.
* `tabpfn` (prior review §1.1): `TabPFNRegressor` on `_features`, using `predict(..., output_type="quantiles")`, with σ
  from the 16/84 % quantiles. Pin the open v2 weights.

**Experiment.** Synthetic 13/41/100 × 3 seeds, then real. Compare with `ridge`, `cnn`, `pretrained`, `deep_kernel` and
`physics` on error, vs mean and CRPS, with paired intervals.

**Adopt if:**
* `pls` or `fpca_gp` matches `cnn` within the paired interval at 41 packets → drop `cnn`/`pretrained` from the defaults.
* `tabpfn` beats `gp` on the features family → try it as the correction learner inside model A (prior review §1.1). It
  is unlikely to beat `physics` overall.

**Risks.**
* PLS and the elastic net give no native uncertainty, so rely on conformal.
* The selection must sit inside the folds (it does if done in `Runner`).
* TabPFN v2.5+ weights are non-commercial and gated.

**Packages:** scikit-learn (mature); `tabpfn` (mature, active). **Effort:** S (1 day for all four).

### H6. Physics of the CDS filter: predict σ at an untried `delayInt` from the nominal trace (S25, C20; prior review §4.1)

**In our terms.**
* The trace samples each pixel cycle at 15 MHz. `trace_features.py` already *reintegrates* the CDS at window lengths of
  10…150 samples (the "noise vs window length" curve, 0.67–10 µs). At the nominal `delayInt` = 1000 (10 µs) that covers
  windows up to the nominal one.
* It also stores the within-cycle spectrum (≥ 0.1 MHz) and the skip-to-skip CDS spectrum (< 15 kHz at 30.7 kHz skip
  rate).
* So the **reference trace taken at nominal** can predict the noise per skip at `delayInt` = 300, 500 and 700 directly. For
  1500–3000 it can do so through the textbook filter integral σ₁²(τ) ∝ ∫ S_v(f) |H_CDS(f;τ)|² df, using the measured PSD
  (Wüthrich et al. 2026, arXiv:2607.20126, and Chierchie et al., both in the prior review).
* Session A measures exactly these settings. It is a physics test that needs no ML and no labels beyond the image σe.
  It is also the **prior mean that Level 2 lacks**: a prediction at settings that have no trace yet.

**Plug-in.**
1. Analysis script, no pipeline change. For each session-A packet and amplifier, compare three numbers:
   * (a) the reintegrated σ at the packet's window length from the *reference* trace;
   * (b) the own trace's measured σ₁ at that `delayInt`;
   * (c) image σe × gain.
   Fit only an overall per-amplifier scale.
2. If (a) tracks (b) and (c), add "reference-predicted σ₁ at this `delayInt`" as a column in `physics.npz`
   (`src/ml/real_inputs.py`). Feed it to model A's correction GP, and use it as the prior mean of the Level-2 surrogate
   for the timing knobs.

**What to expect, honestly.** Reintegration inside a fixed cycle keeps the skip period of the nominal timing, while a real
change of `delayInt` also changes the skip rate. So the low-frequency (1/f) contribution will not transfer exactly. The
15–100 kHz gap between the two spectra has to be interpolated. Settling after the reset (the `delayRGH` packets) is not
in a stationary PSD. Mark the long-window extrapolation [to be checked].

**Multitaper itself.** It is not worth it here. The PSDs are averages over 500 skips × 10 pixels × several sets, so
variance is not the problem. The harmonic F-test for pickup lines would be the only gain, and `count_lines` exists.

**Adopt if:** the per-amplifier-scaled prediction (a) reproduces the session-A `delayInt` dependence of image σe within
~2× the label error (after H3). **Effort:** M (1–2 days after session A).

### H7. Multi-output GP: settings → (trace σ_ADU, σe, λ, gain), with the trace as a cheap task (S13; prior review #5)

**In our terms.**
* This is the prior review's experiment #5, kept HIGH, but it runs *after* H4 and H6 because they decide its value.
* Task 0 is the trace's own σ_ADU (Allan fit). It is observed on every trace, including the reference traces of every
  packet and any trace-only packet. Tasks 1–3 are the image labels.
* Inputs: settings + packet state, **no own-trace features**, plus the H6 physics prediction as an input or prior mean.
* Predicting tasks 1–3 at unseen settings is the Level-2 surrogate. Conditioning on an observed task 0 at the test
  settings (co-kriging) is the "trace screening" value. That repeats H4 in a model that can then drive BO.

**Plug-in.**
* A new class in `src/ml/bogp.py`: BoTorch `MultiTaskGP` / `KroneckerMultiTaskGP`, with known label variances per task
  (`train_Yvar`) and a SAAS or dimension-scaled prior on the settings length scales.
* `Runner.mtgp_settings` returns tasks 1–3 for the test rows, with and without the test row's task-0 observation.
* Later, `bo_advanced.py`'s decomposed surrogate is replaced by this model.

**Experiment.**
* Real 41 packets: error with vs without the task-0 observation.
* Compare with `settings` (single-task, settings only) and `settings_ref` (H4).

**Adopt if:** the multi-output model without task 0 beats `settings`, so the unlabelled traces help the surrogate. If
the task-0 conditioning adds ≥ 1 label error of improvement, the multi-fidelity design stays.

**Risks.**
* An ICM model with 4 tasks and 60 inputs on 120 labelled rows can mis-estimate the task covariance.
* The trace misses the gain and slow noise, so it is a *biased* cheap source (misoKG, prior review §7.2). Model a
  discrepancy, not just noise.

**Package:** BoTorch/GPyTorch (mature; BoTorch 0.18.1 is already installed). **Effort:** M (2–3 days).

---

## 3. The closest analogues the literature agent flagged

| Analogue | How closely it maps | What to borrow concretely |
|---|---|---|
| **Battery early prediction as a cheap fidelity in BO** (Severson 2019; Attia 2020; B11, P18) | **Structurally close, statistically different.** Both have ~120 units, hand-made features of a short early signal, an expensive end measurement, and BO on top. The difference is in range: battery cycle life spans 150–2300 cycles (an order of magnitude), while our σe moved little for the voltage changes tested. Their early signal had a large dynamic range to explain, and ours may not. | (1) The H4 ablation is their key question in our terms. (2) The elastic-net baseline on a few engineered features (H5). (3) A single dominant feature (their variance of ΔQ(V)) corresponds to our `trace_physics` term. Report its rank alone, as Severson reported single-feature models. (4) For Level 2, trust the early predictor only inside the range where it was validated, and confirm top candidates with full measurements. I recall Attia doing this with full cycling of the top protocols, but I did not see it in the abstract [unverified]. |
| **NN qubit noise spectroscopy** (Martina 2023, MLST 4:02LT01; Gupta 2025, Adv. Quantum Technol.; B6) | **Loose.** They infer a noise PSD from coherence decays, an indirect and ill-posed transform, so a network pretrained on synthetic data helps. Our traces give the PSD and Allan curve *directly* from ~5000 samples per set, so the inverse problem is easy. | Little to borrow. Their recipe (pretrain on synthetic data, then fine-tune) is already our model D, which trails the physics models. Revisit only if a hidden-parameter inverse problem appears, e.g. inferring settling constants from edges. |
| **PCH-EM for sub-electron read noise and gain** (Hendrickson & Haefner 2023–2025; B2) | **Close, on the label side.** It is a maximum-likelihood Poisson–Gaussian mixture fit, the same family as WADERs' fit of the 0e⁻/1e⁻ peaks. A 2025 annealed variant addresses bad initialisation, which is the failure we saw with the hand-written JSON (gain locked at ~2× in 4/18 fits) and which `run_WADERs` fixed. | (1) Validate quoted label errors against Monte-Carlo truth (H3). (2) Their papers analyse how gain uncertainty depends on exposure. At our λ ≈ 0.02–0.03 the gain rests on few 1 e⁻ pixels, so read the gain-error section of the 2024 TED paper before trusting gain labels below 1 % (I saw abstracts only). No need to replace WADERs. |
| **Semiconductor virtual metrology and "human first, computer last" BO** (B7; Kanarik et al., Nature 616:707, 2023) | **Close, conceptually.** Virtual metrology predicts expensive metrology from cheap in-situ sensors: that is Level 1. Kanarik (verified) found experts better early and algorithms more cost-efficient near tight tolerances, with human-first/computer-last halving cost-to-target versus experts alone. Caveat: it was a *virtual* process ("game") with a simulator, not wafers. | (1) PLS as the standard virtual-metrology baseline (H5). (2) The 41 hand packets are the "human first" phase. Hand over to BO only for *fine tuning* around a region the experts endorse, inside a trust region. Do not expect BO to beat expert coarse scans. (3) The virtual-metrology practice of moving-window or recency-weighted models under drift is a Level-3 idea; the details are in paywalled IEEE papers [unverified]. (4) Its sampling question, which wafer to measure fully, is our P10 slot idea (§4). |
| **Subject-wise vs record-wise CV** (Saeb 2017; U08) | **Direct.** A packet is a "subject" for the rows of one packet. The amplifier is a second, crossed grouping that we do *not* hold out. | H1: group nominal repeats by setting, use a leave-session-out prospective test, and report a leave-one-amplifier-out score only as a "new module" indicator (3 folds; it will be poor, and that is informative). Little et al.'s caution about identity confounding is exactly what `settings_ref` (H4) quantifies. |
| **History matching** (Williamson et al., Clim. Dyn. 41:1703, 2013; S14, B16) | **Moderate, for Level 2 only.** Emulators rule out about half of a climate model's parameter space with few observations (verified). Our hardware safety is already rule-based (`ccd_limits`, the step limit), so history matching would add a *data-driven* exclusion: settings whose predicted σe or λ is implausibly far from acceptable, given emulator, label and discrepancy variance. | A candidate filter before the acquisition: implausibility I(x) = \|target − μ(x)\| / √(σ²_em + σ²_label + σ²_disc), excluding I > 3 (the conventional cut, from my knowledge of the method, not re-checked). It is cheaper and more readable than SafeOpt for "don't go where noise is surely bad". It does not replace SafeOpt for unknown failure modes. LOW-MED until Level 2. |

---

## 4. Implications for taking the first 41 packets

1. **Randomise the run order within session A (high value, ~10 minutes in `make_packet_plan.py`).**
   * L1_02–07 scan `delayInt` 300 → 3000 in increasing order between nominal packets L1_01 and L1_08. Any monotone drift
     during those ~1.7 h (cool-down history, temperature) is aliased with the strongest knob. The bracketing nominal
     packets measure drift at only two points.
   * Standard design-of-experiments practice: shuffle, e.g. 1500, 300, 3000, 500, 2000, 700. Optionally interleave the
     `delay_H`/`delayRGH` packets with them.
   * In session B, alternate the ± pairs instead of taking them back to back.
   * The packets differ in duration (13–24 min). Shuffling does not change the total time.
2. **Add two non-nominal replicates (~35 min).** Repeat two session-A settings with the largest expected σe change (e.g.
   `delayInt` 300 and 3000) later in the session or in session B. That gives a pure-error estimate *away from nominal*
   (H3's variance components, a lack-of-fit check of the models), and a second test of drift.
3. **Keep the 5 nominal packets.** They feed the label audit (H3) and reference-trace drift detection (H2/P16). Make sure
   each packet record keeps the timestamp and any temperature readback the boards report.
4. **Trace-only extras, if cheap.** The prior review cites ~1 min for a trace-only packet versus ~16 min for a full one. I
   did not check whether `manual_packet.sh` supports this. If it does, a denser trace-only `delayInt` grid (≈ 10 values)
   in session A costs ~10 min. It feeds H6 (does the reintegrated reference prediction track the measured trace?) and
   H7's task 0. They are not training examples for the image models, and they must not change the trace settings
   (BOOLCLR/NDCMPRE) used for the labelled traces.
5. **Session C: keep the Latin hypercube, hold back a few slots.** A space-filling design is the right default before the
   length scales are known. If the plan can stay flexible, reserve 3–4 of the 10 joint packets to be chosen after A+B by
   the largest predictive variance of the settings → label model, or the largest disagreement between `physics` and
   `settings_ref` (P10 uncertainty sampling; BALD is the Bayesian version). The gain at this size is modest. The main
   value is to test the models where they are least sure. If C becomes the prospective test set (H1), keep at least 7
   LHS points untouched by model choice.
6. **Settle the open trace-settings decision before packet 1** (BOOLCLR/NDCMPRE and row-start exclusion; context 10-19
   §1). Every "grouped validation" argument assumes one consistent dataset. A mid-dataset change creates a second
   "environment", which at our size we cannot model (P06 rejected for that reason).
7. **No change to the image geometry.** Larger images would reduce label error, but they change the label definition
   between packets. Use H3 to quantify the label error instead.

---

## 5. Next 5 things to do (value per effort; Level 1 first)

| # | What | Effort | Why now |
|---|---|---|---|
| 1 | **Reorder session A** (shuffle `delayInt`, alternate ± pairs in B) and **add 2 non-nominal replicates** in `lvl1/make_packet_plan.py` (user decision; regenerate `PLAN.md`) | ~1 h | Must happen before the first packet. It costs nothing later and protects the main comparison from drift aliasing. |
| 2 | **H1 validation upgrade** in `train_lvl1.py`: packet-bootstrap CIs and paired differences vs `physics`, CRPS, worst-amplifier coverage, `--group-by setting`, 5 fold seeds; declare session C a prospective test | 0.5–1 day | Without it no later comparison (including 3–5 below) is interpretable at 41 packets. |
| 3 | **H4 + H5 baselines**: `settings_ref` (and `ref_physics`), `pls`, `enet`, `fpca_gp`, `tabpfn` as `Runner` methods | 1–1.5 days | `settings_ref` is the honest bar for "the trace helps" given the real-data finding. The linear and TabPFN arms show whether the network family earns its run time. |
| 4 | **H3 label audit**: SBC-lite on ~60–100 simulated images via the rehearsal tools now; variance components of the nominal repeats after session A; an error scale in processing if needed | 1 day + run time | Every known-noise GP and the pass mark ("within 2× the label error") rest on these errors. The rehearsal already showed they can be too small. |
| 5 | **H2 novelty-aware calibration**: in-fold robust-Mahalanobis novelty score, packet-weighted conformal, Mondrian factors by novelty bin; reference-trace shift test against nominal packets | 1 day | Closes the known failure (a ×2.5 error with a 4 % error bar) and the open "flag unusual traces" item. Level 2 needs it. |

Next, after session A data exist: **H6** (CDS-filter physics test on the `delayInt` scan, 1–2 days), then **H7** (multi-output
GP, 2–3 days), whose value H4 and H6 will have decided. Level-2 engine work (step-limited safe BO in BoTorch or Xopt,
qNEHVI, πBO/physics prior mean) waits until H4/H7 say whether it is single- or multi-fidelity.

**Things I would explicitly not spend time on now:**
* deep trace encoders beyond the existing three: on our data the information is second-order and the physics is near the
  label error;
* MiniRocket or foundation models: they need raw sequences written to `model_inputs`, for little expected gain;
* AutoML, RL, LLM optimisers;
* learned stacking weights;
* celerite/NPE inverse models: the direct PSD and Allan estimates are already precise.

---

## 6. Verification notes

Checked during this work (web search or arXiv page, abstract or record level):
* Saeb et al. 2017, GigaScience, DOI 10.1093/gigascience/gix019: record-wise CV "often massively overestimates" accuracy.
  Little et al. commentary gix020.
* Cawley & Talbot, JMLR 11:2079–2107 (2010): selection overfitting from the variance of CV estimates.
* Severson et al., Nat. Energy 4:383 (2019): 124 cells, first 100 cycles, 9.1 % test error. The elastic-net attribution
  comes from a secondary source and was not seen in the paper text.
* Attia et al., Nature 578:397 (2020): 224 protocols, 16 days vs > 500 days. The validation of the top protocols is from
  memory [unverified].
* Kanarik et al., Nature 616:707 (2023): human first–computer last halves cost-to-target vs experts alone, in a virtual
  process game.
* Hendrickson & Haefner, arXiv:2302.00090 (JEDS 2023): ML estimation of exposure, gain, bias and read noise for deep
  sub-electron read noise pixels, with less uncertainty than photon transfer. Later multisample (TED 2024) and annealed
  (arXiv:2509.13460) variants exist. Their exposure-dependence results were not read.
* Martina et al., MLST 4:02LT01 (2023); Gupta et al., Adv. Quantum Technol. (2025), DOI 10.1002/qute.202500109
  (first-author attribution from secondary pages).
* Williamson et al., Clim. Dyn. 41:1703 (2013): about half of HadCM3 parameter space ruled out.

From the prior review or the context (already checked there): Dunn–Wasserman–Ramdas JASA 2023; Johnson et al. 2026
(arXiv:2606.01427); Kirschner et al. PRAB 2022; Wüthrich et al. 2026; SAAS and dimension-scaled priors as installed.

Repository facts used (read in code):
* nominal `delayInt` = 1000;
* reintegration window lengths 10–150 samples (`trace_sizes`);
* within-cycle and CDS spectra stored as curves;
* one amplifier traced per acquisition (`oct12_features.py` docstring);
* session-A order in `PLAN.md`;
* `scores()` uses an unweighted cross-conformal quantile.

Not checked: whether `manual_packet.sh` can take trace-only packets; the long-window (`delayInt` > 1000) PSD extrapolation
in H6.
