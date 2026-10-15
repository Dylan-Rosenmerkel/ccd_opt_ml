# Session context: 2026-10-15

Continuation of `2026-10-14_session_context.md` (which ends with the DGL decision, the voltage limits still to
confirm, and the final state of the ML models).

## 1. Request

The user asked for a document that **only discusses the ML models**, written for a beginner: start at the basics
and work up to the advanced models with more equations; state the background clearly and explain every variable;
accurate; cover all models worked on; no need for the data and tests, but say when each model is favorable.

## 2. Done: `docs/overleaf_ml_guide/ml_guide.tex` (learning guide)

36 pages, compiles with tectonic, 0 overfull boxes, 0 undefined references. `docs/overleaf_ml_guide.zip` is the
Overleaf upload (tex + figures). The compiled PDF stays local (`docs/overleaf_*/*.pdf` is now in `.gitignore`).

Structure (each section tagged beginner / intermediate / advanced; each model has "the idea in words", the
equations with every symbol explained, "in our code", "when to use it"; "background you need" boxes):

* Notation table (λ always means dark charge; amplifier index j, module m).
* **Part I, foundations:** supervised learning, why log targets; loss, gradient descent, parameters vs
  hyperparameters; over/under-fitting, regularization, train/validation/test, grouped CV (leakage), standardization;
  probability (mean/variance/covariance, Gaussian and multivariate Gaussian, likelihood and NLL, Bayes' theorem and the
  evidence, aleatoric vs epistemic, calibration, error propagation / delta method).
* **Part II, classic models:** OLS → ridge → Bayesian ridge (posterior S, m, predictive variance); regression trees,
  gradient boosting, pinball loss and quantile error bars; MLPs (SiLU, backprop, mini-batches, Adam/AdamW, early
  stopping), mean–variance networks, deep ensembles; 1D CNN (convolution, pooling, our architecture, log-frequency
  spectra).
* **Part III:** Gaussian processes (prior over functions, Matérn-5/2 with ARD, conditioning formulas, log marginal
  likelihood, Cholesky cost).
* **Part IV:** Bayesian optimization (objective J, safe region, EI derivation); model F (decomposed surrogate, trust
  region, learned constraint, multi-fidelity / co-kriging, Pareto front, ParEGO, hypervolume).
* **Part V, project models:** A (skipper averaging, Allan deviation, a²/n + b² fit, common-pattern removal and the
  overlapping Allan curve, the residual and gain GPs, the packet state vector); B (multilevel models, kernel sums,
  new-amplifier shrinkage, G / meta-learning, amplifier labels with shape and rank scores); joint DC + CIC (per-row line,
  OLS covariance, log transform, delta method); H conformal (recipe, guarantee, K-fold caveat, not for new modules);
  D pre-training (masked reconstruction incl. whole curves, θ prediction, InfoNCE with temperature, frozen vs
  fine-tuned); C deep kernel (joint training, over-confidence, frozen encoder); E set models (DeepSets, attention
  pooling, mean of variances vs median, MAD rule).
* **Part VI:** a "which model when" table, the recommended pipeline today, questions to ask; glossary; further reading
  (standard textbooks and the original papers).

Teaching figures (toy examples, no CCD data): `src/ml/plot_guide_figures.py` →
`docs/overleaf_ml_guide/figures/guide_*.png` (fitting, trees, gp, ei, allan, hier, rows, pareto, conv).
`python -m ml.plot_guide_figures ../docs/overleaf_ml_guide/figures` from `src/`.

Accuracy checks made while writing: hyperparameters and settings quoted from the code (GBT 300 trees ν 0.05 ≤15
leaves ≥10/leaf; MLP 5×(64,64) AdamW 3e-3 / 1e-3; CNN 7/7/5 taps 16/32/32; GP subsets 400 / 1200; pre-training weights
1/1/0.3, τ 0.2, blocks of 32; DKL 250 steps, lr 0.03 / 3e-4; trust region 0.4, ×2 after 3, ÷2 after 5, restart <0.02;
ParEGO w ~ U(0.1, 2)). Noted where our versions differ from the textbook ones: ParEGO uses a weighted sum (original:
augmented Chebyshev); conformal uses K-fold (guarantee exact only for split conformal); the deep kernel has an
amplifier term (it is a labeled model).

## 3. State

* Nothing running. Open items unchanged from 2026-10-14 section 5.4 (voltage windows to confirm, `reviewed = false`;
  CCD-to-stage mapping; first real 1×1 500-skip packets).

## 4. Level 1 (user, later on 2026-10-15)

The user defined three levels: **lvl 1** = can noise parameters be predicted from the traces (packets taken by hand,
training by hand); **lvl 2** = predict and suggest new parameters; **lvl 3** = predict and run new scans. Lvl 1 and 2
must be properly prepared before lvl 3 is looked at. Everything **without amplifier labels**. The user's correction:
the models learn how σe, λ, gain depend on the **trace**, so every amplifier's trace is a training example
(100 packets = 400 examples); grouped CV by packet is kept for honest scores.

### Done (pushed as I went)
* **`daq_scripts/manual_packet.sh`** (906bb87): `PACKET_ID` (default `m<date>_<time>`, validated), traces named
  `<PACKET_ID>_ref<set>` / `<PACKET_ID>_new<set>`, a packet record `$RECORD_DIR/<PACKET_ID>.json` (default
  `~/packet_records`) written when queued (not in DRY_RUN); refuses an existing record. Tested with a stub `cdaq`.
  `manual_packet_commands.txt` regenerated.
* **`src/make_manual_manifest.py`**: record + run dir → manifest v3 subset; theta from the `.meta` files (readback);
  requested changes checked (mismatch → status `check`; tested: a wrong VDD1 request was caught); image = first image
  per board after the packet's last trace and before another packet's trace; `trace_matches_image`;
  `--legacy-run DIR --board 101` groups older runs by identical settings (10-12 run → 4 packets).
* **`src/image_rows.py`**: λ per row by ML with WADERs gain/σ fixed (active columns from the .meta geometry); weighted
  line λ_r = CIC + DC x_r with iterated weights from the fitted line (row-error weighting biased CIC low, fixed: pulls
  ~0 ± 1 on simulated images). Assumes FITS row 0 = first read; `[rows] reverse` in processing.toml otherwise.
* **`src/process_packet.py`** (e93b21b): per-row λ after WADERs (failure → warning only); dataset rows gain
  `label.sigma_err`, `dc_err`, `lambda_dark/cic` (+ errors, covariance, χ²), `image.*` geometry; `--packets-dir` skips
  non-ok manifests with a message; then `ml/real_inputs.build_all`.
* **`src/ml/real_inputs.py`**: `model_inputs/{features,curves,physics,sets,pretrain}.npz` (common part: ids, Y, Y_err,
  label_ok, fake, theta; nothing fitted). Tested on the 10-12 legacy packets (16 examples, 85 features, 14 curves, 3 sets
  per row, 96 unlabeled traces). `data.curve_tensor` no longer re-log-frequencies the already log-frequency spectrum.
* **Models**: `HierGP.fit(..., y_var)` known label variances; `PhysicsStructured` takes `X["n_skips"]` per row and
  `X["y_var"]` (residual var = var σe − var g); `JointLambdaGP` takes `X["lambda_cov"]` (real per-row fit covariance);
  `DeepKernelGP(amp_term=False)`; `SetStructured` per-row skips.
* **`lvl1/`** (8c54cfb): `train_lvl1.py` (11 models, all label-free; packet-grouped CV; error, bias, shape, rank,
  coverage 68/95 raw and cross-conformal, factor, label error, for σe, λ AND gain; report.md/json, predictions.csv),
  `make_test_data.py` (synthetic world v4 in the model_inputs format; allan = median over sets like real processing),
  `README.md`, `make_walkthrough.py` → `lvl1_walkthrough.ipynb`. `lvl1/test_data/` and `lvl1/demo/` are gitignored.
* Quick run on 100 synthetic packets (400 examples): trace_physics σe 35.5 % (gain unknown; rank 0.89); physics 5.5 %,
  λ 4.9 %, gain 0.9 %; physics_joint λ 4.6 %; sets σe 5.1 %; ridge 11.5 %; gp 9.9 %; cnn 8.4 %; pretrained 8.0 %;
  deep_kernel 8.9 % (σe; labels carry 1.5 / 3 / 0.8 % error).
* Full (non-quick) run on the same 100 synthetic packets, σe / λ / gain error (labels 1.5 / 3 / 0.8 %):
  trace_physics 35.5 / 30.4 / 31.3 % (σe rank 0.89); **physics 5.5 / 4.9 / 0.9**; physics_joint 5.5 / 4.6 / 0.9;
  sets 5.1 / 4.9 / 0.9; ridge 11.5 / 11.8 / 0.8; trees 10.6 / 21.4 / 3.6; gp 11.0 / 4.4 / 0.8; nn 10.5 / 10.6 / 1.1;
  cnn 6.8 / 9.3 / 1.2; pretrained 5.8 / 11.6 / 1.2; deep_kernel 10.8 / 5.0 / 0.9.
* Fix after the notebook showed it: with known label variances the GP's predictive variance lacked the label noise of
  a NEW image (gain coverage 0.43); `HierGP` now adds the mean known variance → coverage 0.71 / 0.75 / 0.71.
* Notebook `lvl1/lvl1_walkthrough.ipynb` executed (10-12 legacy manifests and processing of 2 packets with
  --skip-waders; row-λ demo; quick training; report tables; predicted-vs-true plots).

### Still open for level 1
* First real packets (1×1, 500 skips) taken with the new manual_packet.sh; check row order (DC > 0), label flags, ch2.
* `models.GP` (sklearn, features family) does not use the label errors (only the physics models do).
* Pass mark for level 1 to agree with the user (suggested in lvl1/README.md).

## 5. Model tutorials (user: a folder of notebooks, one per model, simplest first, from the basics)

`notebooks/model_tutorials/` (bb895e2, 82b0a02): 15 executed notebooks, same order and notation as the learning guide:
00 foundations, 01 ridge, 02 trees, 03 networks, 04 CNN, 05 GP, 06 BO, 07 better BO (F), 08 A, 09 B (+ labels), 10 joint
DC/CIC, 11 conformal (H), 12 pre-training (D), 13 deep kernel (C), 14 set models (E). Each: idea → equations with symbols →
from-scratch version checked against our code → our code on the toy CCD world (no amplifier labels except the comparison
in 09) → when to use it. Generator `make_tutorials.py` (+ `tutorials_part1.py`, `tutorials_part2.py`; `make_tutorials.py
<names>` rebuilds only those, dropping their outputs), helpers `tutorial_utils.py` (`toy_scan`, `grouped_cv`, `report`),
`README.md` index. Single-run numbers are stated honestly next to the multi-scan results (e.g. 09: labelled 1.8 vs
label-free 3.3 % after 2 packets in this run, 2.0 vs 2.2 % over 3 scans; 10: direct λ ahead of joint in this run; 12:
pre-training did not help in this run). Notable demos: 00 leakage (row split λ 10.4 % vs packet split 27.7 %); 04
log-frequency (σe 10.1 → 4.8 % at 120 packets); 08 model A σe 3.0 % from 15 packets vs 20.6 % for a GP; 14 learned set
combination 4.4 % vs best fixed rule 11.3 %.

## 6. Model A+B (user: "what about A+B?" → "go ahead with both")

* **Level 1** (cb1e2dc): `physics_hier` = `PhysicsStructured(kind="hier")` with module/amplifier ids built from board and channel,
  a REFERENCE marked "(labels)" in every report; runs by default next to `physics`. Synthetic 100 packets: σe 5.9 % (labels)
  vs 5.5 % (no labels), λ 4.9 / 4.9 %, gain 0.8 / 0.9 %.
* **Tutorial** `notebooks/model_tutorials/09b_physics_plus_hierarchical.ipynb` (between 09 and 10; 09's "Next" points to it):
  inside A+B, the gain GP's variance is 99.5 % amplifier offset, the correction and λ are shared; known amplifiers (1 module):
  A+B ahead with few packets (30 packets: σe 2.2 vs 3.7 %, gain 1.0 vs 2.6 %), equal by 60 (gain still better 0.3 vs 0.9 %);
  new module: 0 packets error dominated by ONE amplifier with an unusual gain (A+B 67 %, A 53 %; the other three within
  3–7 % for both); A+B gives every new amplifier a wide ±31 % bar, A narrow ±9 % bars; after 2–5 packets A+B slightly
  ahead in this run (1.8 vs 3.3 %, 1.9 vs 2.4 %). Conclusion written in the notebook: the price of no labels is mainly the gain
  with few packets; whether it exists on the real CCD (gains differ ~2 % on module 101) is for level 1 to show.
