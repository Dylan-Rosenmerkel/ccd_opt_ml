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
