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
