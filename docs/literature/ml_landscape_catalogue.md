# A wide map of machine-learning model families and application fields

*Written 2026-10-21 for `ccd_opt_ml` by the literature agent of a three-agent team. Its purpose is **breadth**: to catalogue
model families and techniques, and the application fields that use them (related and seemingly unrelated to ours), so that two
analysts can later (i) sort out what is directly applicable and (ii) work out how unrelated ideas could be adapted. It
deliberately makes no deep applicability judgements. Where an entry carries a short remark about our problem, read it as a
hint for the analysts, not a recommendation.*

*It extends the earlier, narrower review `docs/literature/ml_models_literature_review.md` ("prior review"). Items covered
there (TabPFN and in-context tabular models, ROCKET/MiniRocket, time-series foundation models, TS2Vec/TF-C/PatchTST, neural
processes and PACOH, SBI basics, VBLL/Laplace/SNGP/evidential regression, hierarchical conformal, Set Transformer, multi-fidelity
and function-network BO, SafeOpt/LineBO/SCBO, SAASBO, CCD/accelerator prior work) appear here only as short stubs that point
back.*

**How references were checked.** Every arXiv reference was verified during this search against the arXiv API: the ID resolves
to the stated title, first author and year. Non-arXiv references were verified by web search of publisher, repository or index
pages, and are marked "(abstract only)" or "(record only)" when I saw no more than that. Anything I could not confirm is
marked **[unverified]**. Package maturity ratings are my own judgement. See Part D for caveats.

**Our application in one paragraph (for context).** We predict a skipper-CCD amplifier's image-noise parameters (read noise
σe, dark charge λ, gain) from a short raw readout trace (~0.2 s; 10 pixels × 500 skips at 15 MHz) plus ~61 operating settings
(Level 1). Labels come from image fits (WADERs) with known errors. There are ~40-400 labelled examples in packets of 4
correlated amplifiers, and a few times more unlabelled traces. Inputs per example: 85 hand-made trace features, 14 curves ×
256 points (Allan deviation, spectra, waveform shapes), several repeated trace sets, and the settings. Uncertainties must be
calibrated. Level 2 suggests the next settings (safe BO within hardware limits). Level 3 runs autonomous scans.

---

## Table of contents

* [Summary table](#summary-table)
* [Part A: model and technique families](#part-a-model-and-technique-families-detailed-entries)
  * [A1. Classical statistics, kernels and GPs (S01–S29)](#a1-classical-statistics-kernels-and-gaussian-processes)
  * [A2. Trees and ensembles (T01–T06)](#a2-trees-and-ensembles)
  * [A3. Deep-learning architectures (D01–D17)](#a3-deep-learning-architectures)
  * [A4. Feature engineering for series and curves (F01–F04)](#a4-feature-engineering-for-series-and-curves)
  * [A5. Generative models and simulation-based inference (G01–G08)](#a5-generative-models-and-simulation-based-inference)
  * [A6. Learning paradigms (P01–P20)](#a6-learning-paradigms)
  * [A7. Sequential decisions, optimization and control (O01–O24)](#a7-sequential-decisions-optimization-and-control)
  * [A8. Uncertainty, evaluation and interpretability (U01–U08)](#a8-uncertainty-quantification-evaluation-and-interpretability)
* [Part B: application fields (B1–B29)](#part-b-application-fields)
  * [B-I. Fields close to ours (B1–B10)](#b-i-fields-close-to-ours)
  * [B-II. Fields that seem unrelated (B11–B29)](#b-ii-fields-that-seem-unrelated)
* [Part C: problem shapes (C1–C22)](#part-c-problem-shapes)
* [Part D: gaps and caveats](#part-d-notable-gaps-and-caveats-of-this-catalogue)
* [References](#references)

**Field codes used throughout:**
B1 particle/nuclear detectors · B2 CCD/sensor characterization · B3 astronomy instrumentation · B4 gravitational-wave
detectors · B5 accelerators · B6 quantum devices · B7 semiconductor manufacturing · B8 analog circuit design · B9 electronics/RF
· B10 metrology/timekeeping · B11 batteries · B12 medicine/biosignals · B13 seismology · B14 speech/audio · B15 finance · B16
climate/weather/hydrology · B17 materials/self-driving labs · B18 drug discovery/chemistry · B19 proteins · B20 robotics/control
· B21 recommender systems · B22 agriculture/remote sensing · B23 predictive maintenance · B24 ecology · B25 NLP/LLMs · B26
computer vision · B27 sports · B28 gaming · B29 hyperparameter optimization/AutoML.

---

## Summary table

116 technique entries (Part A) + 29 field entries (Part B). "Data regime" is the typical *labelled* data size, plus the input
type. "n" means rows.

| ID | Entry | Family | Main fields | Data regime | Python package (maturity) |
|---|---|---|---|---|---|
| S01 | GLMs, ridge, Bayesian ridge | linear | all | any n; tabular | scikit-learn, statsmodels (mature) |
| S02 | LASSO / elastic net / compressed sensing | sparse linear | B11, B17, B9 | p ≫ n | scikit-learn, cvxpy (mature) |
| S03 | PLS / PCR | latent linear | chemometrics, B22, B7 | 10–10³ n; collinear curves | scikit-learn (mature) |
| S04 | Splines / GAMs | additive nonparametric | B24, B16, B12 | 10²+ n; < 50 inputs | pygam (moderate) |
| S05 | Quantile regression | statistical | B15, B16 | any | statsmodels, lightgbm (mature) |
| S06 | Kernel ridge / SVR / random features | kernel | B18, B7 | ≤ 10⁴ n | scikit-learn (mature) |
| S07 | GP regression (prior review) | Bayesian nonparametric | B1–B11 | 10–10⁴ n | gpytorch/botorch, GPflow (mature) |
| S08 | Sparse variational GPs, celerite | scalable GP | B3, B16 | long 1-D series / big n | gpytorch, celerite2 (mature) |
| S09 | Spectral-mixture kernels, BNSE | spectral GP | B9, B3, B12 | short noisy series | gpytorch (mature) |
| S10 | Deep GPs | hierarchical GP | B22, B16 | 10²–10⁶ n | gpytorch, GPflux (moderate) |
| S11 | Warped / heteroscedastic GPs | GP likelihoods | B16, B15 | like GP | GPy, gpytorch |
| S12 | Deep kernel learning (prior review) | NN + GP | B17, B18 | 10²–10⁵ n | gpytorch |
| S13 | Multi-output / multi-fidelity GPs (prior review) | GP | B5, B16, B17, B11 | cheap many + expensive few | botorch, emukit |
| S14 | Bayesian calibration (KOH), history matching | inverse problems | B16 | simulator + few obs | (custom) |
| S15 | Mixed-effects / hierarchical Bayes, PPLs | multilevel | B12, B27, B24 | many small groups | statsmodels, PyMC, NumPyro (mature) |
| S16 | MERF, GPBoost | trees + random effects | B12, B22 | clustered rows | gpboost (mature), merf |
| S17 | Polynomial chaos expansion | UQ surrogate | engineering, B8 | 10–10³ runs; < 20 inputs | chaospy, OpenTURNS (mature) |
| S18 | Functional data analysis | curve statistics | B12, B16, B22 | 10–10³ curves | scikit-fda (moderate) |
| S19 | Distribution regression / mean embeddings | kernel on sets | B24, B1 | bags of samples | (custom), alibi-detect |
| S20 | State-space models, Kalman (+ neural) | latent dynamics | B16, B20, B15, B10 | streams | statsmodels, filterpy, dynamax |
| S21 | Hidden Markov models | discrete latent | B14, B15, B12 | sequences | hmmlearn, pomegranate |
| S22 | ARIMA / GARCH / HAR | econometric TS | B15, B10 | single long series | statsmodels, arch (mature) |
| S23 | Bayesian nonparametrics (DP mixtures) | Bayesian | B24, B1 | small–medium | scikit-learn, PyMC |
| S24 | Bayesian online changepoint detection | sequential Bayes | B15, B12, B23 | streams | ruptures, river |
| S25 | Multitaper / classical spectral estimation | signal processing | B13, B3, B4, B10 | short records | scipy, mne, allantools |
| S26 | Symbolic regression (PySR, AI Feynman, SISSO) | equation discovery | physics, B17 | 10–10³ n; < 10 relevant inputs | pysr (mature) |
| S27 | SINDy | sparse system ID | fluids, B20 | dynamic time series | pysindy (mature) |
| S28 | Koopman / DMD | operator models | fluids, B16 | snapshot sequences | pydmd (mature) |
| S29 | Reservoir computing | random RNN + ridge | chaos, B9 | time series | reservoirpy (moderate) |
| T01 | Random forests | bagged trees | all | 10²–10⁶ n | scikit-learn (mature) |
| T02 | Gradient boosting (XGBoost, LightGBM, CatBoost) | boosted trees | B15, B21, B1, B23 | 10³+ n | xgboost, lightgbm, catboost (mature) |
| T03 | Quantile forests, NGBoost (prior review) | probabilistic trees | — | — | quantile-forest, ngboost |
| T04 | BART | Bayesian trees | causal inference, B12 | 10²–10⁵ n | pymc-bart (moderate) |
| T05 | EBM / NAM | glass-box additive | B12, B15 | 10³+ n | interpret (mature) |
| T06 | Stacking of predictive distributions | ensembling | B16, B12 | out-of-fold predictions | arviz (mature) |
| D01 | Tabular MLPs (RealMLP, TabM, FT-Transformer) | NN tabular | B15, B21 | 10³+ n | pytabkit, tabm (moderate) |
| D02 | 1-D CNN / ResNet / InceptionTime / TCN / WaveNet | CNN | B12, B13, B4, B14 | 10³+ labelled series | torch, tsai, aeon |
| D03 | LSTM / DeepAR | RNN | B16, B15, B23 | many related series | gluonts, neuralhydrology |
| D04 | Transformers | attention | B25, B26, B13 | large / pretrained | transformers (mature) |
| D05 | S4 / Mamba | deep state space | B14, B12 | long sequences | mamba-ssm (active) |
| D06 | Graph neural networks | message passing | B1, B17, B8, B16 | graphs, 10³+ | torch_geometric (mature) |
| D07 | DeepSets (Set Transformer in prior review) | set models | B1, B3 | sets | torch |
| D08 | Neural ODE / SDE / CDE | continuous-time NN | B12, B15, B20 | (irregular) series | torchdiffeq, diffrax (mature) |
| D09 | PINNs, PGNN, UDEs | hybrid physics-ML | B16, B17, B5 | few labels + equations | deepxde (mature) |
| D10 | FNO / DeepONet | neural operators | PDE surrogates, B16 | 10³+ function pairs | neuraloperator (moderate) |
| D11 | Implicit neural representations (SIREN) | coordinate nets | B26 | per-signal | (research) |
| D12 | KANs | spline networks | science | small, low-dim | pykan (research) |
| D13 | Mixture of experts | conditional | B25, B15, B20 | any | torch |
| D14 | Hypernetworks, FiLM | conditional | B26, B14 | medium | torch |
| D15 | Wavelet scattering, SincNet | signal front-ends | B14, B12, B13, B23 | small–medium; no training (scattering) | kymatio (mature) |
| D16 | N-BEATS | NN forecasting | B15 | many series | darts, neuralforecast (mature) |
| D17 | Neural processes (prior review) | meta-learned processes | — | many tasks | (research) |
| F01 | tsfresh / catch22 / hctsa | feature extraction | B12, B23, B24 | any | tsfresh, pycatch22 (mature) |
| F02 | ROCKET / MiniRocket (prior review) | random features | TSER | any | aeon, sktime (mature) |
| F03 | Path signatures | rough-path features | B12, B15 | multichannel series | iisignature, signatory |
| F04 | Time-series augmentation, mixup | regularization | B12, B14 | small labelled sets | tsaug, torchaudio |
| G01 | VAEs | latent generative | B1, B18, B12 | 10³+ unlabelled | torch, pythae |
| G02 | Normalizing flows | exact-density generative | B1, B3, B4 | 10³–10⁶ samples | zuko, nflows |
| G03 | Diffusion / score models (incl. CSDI) | generative | B26, B14, B16 | large | diffusers (mature) |
| G04 | GANs | adversarial generative | B1, B26, B14 | large | torch |
| G05 | Energy-based models | unnormalized density | B26, B19 | medium–large | torch |
| G06 | ABC | likelihood-free (classical) | genetics, B24 | simulator | pyabc (mature) |
| G07 | Neural SBI (NPE/NLE/NRE) (prior review) | likelihood-free (neural) | B4, B3, B1 | simulator, 10⁴+ sims | sbi (mature) |
| G08 | Physics-based synthetic data, domain randomization, Noise2Noise | data generation | B20, B26, B3, B2 | simulator + few real | torch |
| P01 | Contrastive SSL (SimCLR, CPC) | self-supervised | B26, B14, B12 | many unlabelled | lightly |
| P02 | Masked modelling (BERT, MAE, wav2vec 2.0, HuBERT) | self-supervised | B25, B14, B12 | many unlabelled | transformers |
| P03 | BYOL / VICReg | self-supervised | B26, B12 | unlabelled, small batch | lightly, solo-learn |
| P04 | Semi-supervised (mean teacher, FixMatch, self-training) | semi-supervised | B26, B12, B14 | few labels + unlabelled | torch, USB |
| P05 | Transfer / PEFT (LoRA, domain-adaptive pretraining) | transfer | B25, B19, B12 | few labels | peft (mature) |
| P06 | Domain adaptation / generalization (DANN, IRM, Group DRO) | shift-robust learning | B12, B13 | several environments | adapt, WILDS |
| P07 | Test-time adaptation (Tent) | adaptation | B26, B12 | unlabelled test | (research) |
| P08 | Meta-learning / few-shot (MAML, ProtoNets) | meta-learning | B18, B20 | many tasks | learn2learn |
| P09 | Multi-task learning | shared representation | B25, B18 | any | torch |
| P10 | Active learning (BALD, uncertainty sampling) | data acquisition | B18, B17, B7 | pool + costly labels | modAL, botorch, baal |
| P11 | Continual learning (EWC) | streaming learning | B20, B21 | sequential tasks | avalanche |
| P12 | Federated learning | distributed | B12 | multi-site | flower (mature) |
| P13 | Causal inference (double ML, DoWhy) | causal | B12, econometrics | observational | dowhy, econml, doubleml (mature) |
| P14 | Causal discovery (PC, NOTEARS) | causal | B23, B16 | 10²+ n; < 50 vars | causal-learn (mature) |
| P15 | Anomaly detection (iForest, Deep SVDD, AEs) | unsupervised | B1, B23, B15, B4 | mostly normal | scikit-learn, pyod (mature) |
| P16 | Drift / shift detection | monitoring | B15, B21, B23, B7 | streams | alibi-detect, river (mature) |
| P17 | Physics / science foundation models (MPP, The Well, AstroCLIP, ClimaX, biosignal FMs) | pretrained | B3, B16, B12 | pretrain huge | HF hub |
| P18 | Early prediction / learning-curve extrapolation | predictive | **B11**, B29 | partial + few complete runs | (problem-specific), botorch |
| P19 | Neuro-symbolic / program synthesis | hybrid | science | structured tasks | (research) |
| P20 | AutoML / HPO (auto-sklearn, AutoGluon, Optuna, Hyperband) | automation | all | any | autogluon, optuna (mature) |
| O01 | Bayesian optimization (core) | BO | B5–B8, B17, B20, B28, B29 | 10–10³ evals | botorch/ax (mature), xopt |
| O02 | TuRBO, BAxUS (SAASBO in prior review) | high-dim BO | B5, B8 | 10²+ evals; 20–1000 dims | botorch |
| O03 | Multi-objective BO (qNEHVI) | MOBO | B17, B8, B11 | like BO | botorch, pymoo |
| O04 | Safe / constrained BO (prior review) | safe BO | B5, B20 | like BO | botorch, SafeOpt |
| O05 | Transfer / prior-informed BO (RGPE, πBO, HEBO) | BO with priors | B29, B5 | past tasks / expert beliefs | botorch, HEBO, SMAC3 |
| O06 | Time-varying BO | drift-aware BO | B5, B10 | streams | botorch (custom) |
| O07 | Causal BO | BO + causal graph | B12, B7 | graph + few interventions | (research) |
| O08 | Preferential / human-in-the-loop BO | BO + human | **B7**, B26 | few evals + expert | botorch PairwiseGP |
| O09 | Multi-fidelity / early stopping (Hyperband, freeze-thaw) | resource allocation | B29, B11 | cheap partial evals | optuna, ray tune |
| O10 | Batch BO | parallel BO | B8, B17 | parallel evals | botorch |
| O11 | Bandits, Thompson sampling, LinUCB | sequential decisions | B21, B12 | streams | vowpalwabbit, mabwiser |
| O12 | CMA-ES | evolution strategy | B20, B5, B6 | 10²–10⁴ evals | cma (mature) |
| O13 | Genetic algorithms, NSGA-II | evolutionary MOO | B8, B7 | 10³+ evals | pymoo, DEAP (mature) |
| O14 | Extremum seeking | model-free feedback | **B5** | continuous stream | (custom) |
| O15 | Model-free RL (DQN, PPO, SAC) | RL | B28, B20, B5, B8 | 10⁴–10⁸ interactions | stable-baselines3 (mature) |
| O16 | Model-based / safe RL (PILCO, PETS) | RL | B20, B3 | 10–10³ real interactions | mbrl-lib |
| O17 | Offline RL, Decision Transformer | RL from logs | B12, B20 | large logs | d3rlpy (mature) |
| O18 | Learning-based MPC | control | B20, B7 | system streams | do-mpc, acados, casadi |
| O19 | EWMA run-to-run control | process control | **B7** | run stream | trivial |
| O20 | Bayesian experimental design | info-optimal design | B12, B17, B10 | model + few runs | pyro (OED) |
| O21 | LLM optimizers (OPRO, LLAMBO, FunSearch) | LLM-driven search | B29, maths, B18 | few evals | (research) |
| O22 | LLM lab agents (Coscientist, ChemCrow, k-agents) | agents | B18, B6, B5 | tools + docs | (fast-moving) |
| O23 | Self-driving labs | closed loop | B17, B18, B11 | 10–10³ autonomous runs | botorch + lab code |
| O24 | AlphaZero / MuZero planning | search + learning | B28 | cheap simulator | mctx, open_spiel |
| U01 | Deep ensembles (prior review) | UQ | all | — | torch |
| U02 | Bayesian NNs (BBB, MC dropout) | UQ | B12, B20 | medium | bayesian-torch, laplace-torch |
| U03 | Temperature scaling, proper scores (CRPS) | calibration / evaluation | B16, B12 | validation set | properscoring, scoringrules |
| U04 | Conformal incl. adaptive (ACI) | distribution-free UQ | B15, B12, B16 | calibration set/stream | mapie, crepes (mature) |
| U05 | Simulation-based calibration | inference checking | Bayesian stats, B4 | simulator | sbi, arviz |
| U06 | SHAP | attribution | B12, B15, B7 | any model | shap (mature) |
| U07 | Interpretable-by-design models | interpretability | B12, B15 | any | interpret, imodels |
| U08 | Use-case-matched (grouped) validation | evaluation | **B12**, B13, B16 | grouped data | scikit-learn GroupKFold |

Field entries (Part B) with their closest-shape notes are not repeated in the table. See the Part C table for the mapping of
fields to problem shapes.
---

# Part A: model and technique families (detailed entries)

Each entry follows the same layout. **What** is a plain-language description. **Data** is the typical data regime.
**Fields** lists where it is used (codes refer to Part B). **+/−** gives strengths and weaknesses. **Py** lists Python
packages with my maturity rating. **Refs** are short keys into the reference list at the end.

Entries marked **(prior review §x)** are already covered in `ml_models_literature_review.md`. They are listed here only
briefly, for completeness of the map.

---

## A1. Classical statistics, kernels and Gaussian processes

### S01. Generalized linear models (GLMs), ridge and Bayesian ridge
* **Family:** linear / statistical.
* **What:** The prediction is a linear combination of the inputs passed through a link function: $g(\mathbb E[y]) = w^\top x$.
  For example, a log link gives a multiplicative model for a positive target such as σe or λ. Ridge adds an L2 penalty. Bayesian
  ridge puts a Gaussian prior on $w$ and so gives a predictive variance. The model is fast, convex and easy to interpret, and it
  is the baseline every other method must beat.
* **Data:** any size, including tens of rows. Tabular inputs.
* **Fields:** everywhere: B12 (clinical risk scores), B15, B27 (expected-goals models), B7.
* **+/−:** + it is hard to overfit once regularized, and its uncertainty is cheap. − it cannot capture interactions unless you
  hand-craft them, and Gaussian errors may be wrong for counts (λ).
* **Py:** `scikit-learn` (mature), `statsmodels` (mature; full GLM families, inference).
* **Refs:** [Pedregosa 2011]; [Seabold 2010]. (Our Bayesian ridge model already belongs here.)

### S02. Sparse linear models: LASSO, elastic net, compressed sensing
* **Family:** linear / sparse estimation.
* **What:** An L1 penalty $\lambda\|w\|_1$ drives most coefficients to exactly zero. The method therefore selects a small set
  of inputs. Compressed sensing is the signal-processing side of the same idea: a sparse signal can be recovered from far fewer
  measurements than its length if the measurements are "incoherent". The elastic net mixes L1 and L2 so that correlated inputs
  (for example, neighbouring Allan-curve points) are kept together.
* **Data:** p ≫ n is the classical use case: hundreds of inputs and tens of rows.
* **Fields:** B11 (Severson et al. used an elastic net on early-cycle battery features), genomics, B9 (sparse spectral
  recovery), B17 (SISSO builds on it).
* **+/−:** + automatic feature selection and an interpretable model. − selection is unstable with correlated inputs (stability
  selection or the elastic net helps), and the uncertainty after selection is not straightforward.
* **Py:** `scikit-learn` (`Lasso`, `ElasticNetCV`; mature), `cvxpy` (mature).
* **Refs:** [Tibshirani 1996]; [Donoho 2006]; [Severson 2019].

### S03. Partial least squares (PLS) and principal-component regression
* **Family:** linear latent-variable / chemometrics.
* **What:** PLS finds a few latent directions in X that are maximally *covariant with y*, then regresses on them. It is the
  workhorse of spectroscopy, where hundreds of collinear wavelength channels predict one concentration. Principal-component
  regression (PCR) uses PCA directions instead, which ignores y when the directions are chosen.
* **Data:** tens to hundreds of samples, hundreds to thousands of highly collinear inputs (spectra, curves).
* **Fields:** chemometrics, B22 (NIR spectroscopy of crops and soils), B7 (virtual metrology from optical-emission spectra),
  food science.
* **+/−:** + ideal for "a curve in, a scalar out" with few rows, robust and fast. − linear unless kernelized, and the number of
  components must be chosen by CV.
* **Py:** `scikit-learn` (`PLSRegression`; mature).
* **Refs:** [Wold 2001]. Relevant to our 14 × 256 curves.

### S04. Splines and generalized additive models (GAMs)
* **Family:** nonparametric additive regression.
* **What:** The model is $g(\mathbb E[y]) = \sum_j f_j(x_j)$ (+ selected pairwise terms), where each $f_j$ is a smooth
  penalized spline. Each input's effect can be plotted as a curve, so the model is a "glass box" that still captures
  nonlinearity.
* **Data:** hundreds to millions of rows. Works best with fewer than ~50 inputs.
* **Fields:** epidemiology, ecology (B24), B16 (air quality), B15 (credit scoring), B27.
* **+/−:** + interpretable, smooth, and well calibrated with REML smoothing selection. − interactions must be specified, and
  hundreds of inputs are impractical without selection.
* **Py:** `pygam` (moderately mature), `statsmodels.gam` (basic), `interpret` (EBM, see T05). R `mgcv` is the reference.
* **Refs:** [Hastie & Tibshirani 1986]; [Servén & Brummitt 2018].

### S05. Quantile regression
* **Family:** statistical regression of conditional quantiles.
* **What:** It estimates the τ-quantile of y given x by minimizing the asymmetric "pinball" loss
  $\rho_\tau(u) = u(\tau - \mathbb 1[u<0])$, so it models the spread directly and needs no Gaussian assumption. Any learner can
  use the pinball loss (linear, GBM, NN).
* **Data:** any size.
* **Fields:** B15 (value-at-risk), econometrics, B16 (probabilistic forecasting), B12 (growth charts).
* **+/−:** + heteroscedastic intervals without distributional assumptions. − separately fitted quantiles can cross, and the
  coverage is only asymptotic (fix this with CQR, prior review §5.2).
* **Py:** `statsmodels.QuantReg` (mature); `HistGradientBoostingRegressor(loss="quantile")`, `lightgbm`; `quantile-forest`.
* **Refs:** [Koenker & Hallock 2001].

### S06. Kernel ridge regression, support-vector regression, random features
* **Family:** kernel methods.
* **What:** These methods replace inner products by a kernel $k(x,x')$, which gives nonlinear regression with a convex fit.
  Kernel ridge gives the same mean as a GP but no variance. SVR uses an ε-insensitive loss. Random Fourier features
  approximate a kernel with an explicit finite feature map, which makes the method scale.
* **Data:** up to ~10⁴ rows exactly. Any vector input, or custom kernels on curves and sets.
* **Fields:** B18 (QSAR), B7 (virtual metrology), B13 (older pickers), bioinformatics.
* **+/−:** + strong at small n with a good kernel. − kernel choice is everything, and there is no built-in uncertainty (use
  a GP for that).
* **Py:** `scikit-learn` (`KernelRidge`, `SVR`, `RBFSampler`; mature).
* **Refs:** [Pedregosa 2011]; [Rasmussen & Williams 2006].

### S07. Gaussian-process regression (baseline) (prior review §7)
* **Family:** Bayesian nonparametric / kernel.
* **What:** A prior over functions defined by a mean and a covariance kernel. Conditioning on data gives a closed-form Gaussian
  predictive distribution. ARD kernels learn one length scale per input.
* **Data:** 10–10⁴ rows exactly. ≲ 20–50 effective inputs before ARD needs strong priors (SAAS, prior review §7.4).
* **Fields:** everywhere in B1–B11. The default surrogate in BO.
* **+/−:** + calibrated uncertainty, priors are explicit, works with known noise. − O(n³) cost, and stationarity is assumed.
* **Py:** `gpytorch`/`botorch` (mature), `GPflow` (mature), `scikit-learn` (basic).
* **Refs:** [Rasmussen & Williams 2006]; [Gardner 2018].

### S08. Scalable and structured GPs: sparse variational GPs, celerite
* **Family:** GP approximations.
* **What:** Sparse variational GPs summarize the data with m ≪ n "inducing points" and optimize a variational bound. This
  allows minibatching and non-Gaussian likelihoods (e.g. Poisson for counts). *celerite* exploits the fact that a 1-D
  stationary GP whose kernel is a sum of complex exponentials (damped oscillators, i.e. a Lorentzian-shaped PSD) can be solved
  in O(n). It was built for stellar light curves.
* **Data:** sparse GPs: 10⁴–10⁷ rows. celerite: very long 1-D time series.
* **Fields:** B3 (light curves, exoplanet transits, asteroseismology), B16 (spatial statistics), B24.
* **+/−:** + celerite fits a *physically meaningful PSD* directly to a raw 1-D series with an exact likelihood. That is close
  to "fit a noise model to a trace", and a 500-skip trace fits trivially. − celerite is limited to 1-D and to a specific family of
  PSD shapes. 1/f noise needs a sum of terms.
* **Py:** `gpytorch` (SVGP; mature), `GPflow` (mature), `celerite2` (mature, astronomy community).
* **Refs:** [Hensman 2013]; [Foreman-Mackey 2017].

### S09. Spectral-mixture kernels and Bayesian nonparametric spectral estimation
* **Family:** GP kernels designed in the frequency domain.
* **What:** By Bochner's theorem, a stationary kernel is the Fourier transform of a spectral density. The spectral-mixture
  kernel models that density as a mixture of Gaussians, so the GP *learns the PSD*. Tobar's BNSE puts a GP on the signal and
  derives a posterior over its spectrum, giving error bars on a PSD estimate from short, noisy, possibly unevenly sampled data.
* **Data:** single or few series of 10²–10⁴ points.
* **Fields:** B9, B3, B12 (heart-rate variability), B13.
* **+/−:** + error bars on spectra from short records, which suits 0.2 s traces. − multimodal hyperparameter landscape, and
  1/f (scale-free) behaviour is not natural for Gaussian mixtures.
* **Py:** `gpytorch.kernels.SpectralMixtureKernel` (mature); BNSE research code (`mogptk` [unverified]).
* **Refs:** [Wilson & Adams 2013]; [Tobar 2018].

### S10. Deep Gaussian processes
* **Family:** hierarchical GP.
* **What:** Several GPs are composed: the output of one is the input of the next. This yields non-stationary and non-Gaussian
  functions while keeping a probabilistic treatment. It is trained with doubly stochastic variational inference.
* **Data:** hundreds to 10⁶ rows. Works best when a single GP is clearly too stationary.
* **Fields:** B22 (You et al. use a GP layer on top of a CNN/LSTM for crop yield), B16, surrogate modelling.
* **+/−:** + flexible and still Bayesian. − hard to train and tune, and it can underfit at very small n.
* **Py:** `gpytorch` (DeepGP module), `GPflux` (moderate).
* **Refs:** [Damianou & Lawrence 2012]; [Salimbeni & Deisenroth 2017]; [You 2017].

### S11. Warped and heteroscedastic GPs
* **Family:** GP likelihood extensions.
* **What:** A warped GP learns a monotone transformation of y so that the transformed target is Gaussian. This is a learned
  version of our "log target". A heteroscedastic GP lets the noise variance itself vary with x (a second GP on log-noise), which
  fits when label noise depends on the settings (e.g. λ is much noisier at low dark current).
* **Data:** like a GP.
* **Fields:** B16 (rainfall), B15, BO with input-dependent noise.
* **+/−:** + captures skew and input-dependent noise. − more hyperparameters, and when label errors are *known* a fixed noise
  model is simpler.
* **Py:** `GPy` (warped GP; maintenance mode), `gpytorch` (heteroskedastic via custom likelihood), `botorch` (`HeteroskedasticSingleTaskGP` [unverified name in current version]).
* **Refs:** [Snelson 2003].

### S12. Deep kernel learning (prior review §3; model in repo)
* **Family:** neural feature extractor + GP.
* **What:** The GP kernel acts on the output of a neural network, $k(g_\theta(x), g_\theta(x'))$, and θ and the kernel are
  trained jointly by marginal likelihood.
* **Data:** hundreds to 10⁵ rows. It overfits the marginal likelihood at very small n.
* **Fields:** B17, B18, few-shot BO.
* **+/−:** + learned features with GP uncertainty. − known to be overconfident when n is small.
* **Py:** `gpytorch` (mature).
* **Refs:** [Wilson 2015].

### S13. Multi-output, multi-fidelity GPs and co-kriging (prior review §7)
* **Family:** GP for several correlated outputs or fidelity levels.
* **What:** In co-kriging, $f_\text{hi}(x) = \rho f_\text{lo}(x) + \delta(x)$. Multi-output GPs share information between
  outputs through a coregionalization matrix. Neural variants learn the low-to-high mapping with networks (Meng & Karniadakis'
  composite NN). Surveys organize multi-fidelity methods into adaptation, fusion and filtering.
* **Data:** many cheap evaluations plus few expensive ones.
* **Fields:** aerospace design, B16, B17, B5, B11.
* **+/−:** + uses cheap data such as trace-only packets. − needs the fidelity levels to be correlated, and a biased cheap source
  needs a discrepancy term.
* **Py:** `botorch` (mature), `emukit` (moderate).
* **Refs:** [Peherstorfer 2018]; [Fernández-Godino 2016]; [Meng & Karniadakis 2019].

### S14. Bayesian calibration of computer models and history matching
* **Family:** statistical inverse problems with emulators.
* **What:** In Kennedy–O'Hagan calibration, a slow simulator $\eta(x,\theta)$ with unknown parameters θ is emulated by a GP.
  The method infers θ from observations while modelling a *discrepancy* term between simulator and reality. History matching
  (climate science) uses emulators and an "implausibility" measure to *rule out* regions of parameter space instead of finding
  a best point. That suits safe exploration.
* **Data:** 10²–10⁴ simulator runs plus few real observations.
* **Fields:** B16 (climate-model tuning), nuclear engineering, oil reservoirs, epidemiology.
* **+/−:** + explicit model discrepancy, and ruling out implausible space is a natural safety idea. − the setup is
  statistically demanding, and identifiability between θ and the discrepancy is weak.
* **Py:** no dominant package. Build from `gpytorch`/`emukit`. R and MATLAB tools exist.
* **Refs:** [Kennedy & O'Hagan 2001]; [Williamson 2013].

### S15. Mixed-effects and hierarchical Bayesian models (incl. probabilistic programming)
* **Family:** multilevel statistics.
* **What:** Parameters vary by group (amplifier, packet, board) around a population mean, with partial pooling:
  $y_{ij} = x_{ij}^\top\beta + u_j + \epsilon_{ij}$, $u_j \sim \mathcal N(0,\tau^2)$. Groups with little data are "shrunk" toward
  the population. Probabilistic programming languages (Stan, PyMC, NumPyro) let you write any such model and get posteriors by
  MCMC or variational inference.
* **Data:** small to large. Its strength is *many small groups*.
* **Fields:** B12 (patients), B27 (teams and players, e.g. hierarchical football models), B24, education, B22 (field trials).
* **+/−:** + the textbook answer to "rows within a packet are correlated", with honest uncertainty. − linear unless combined
  with GPs or trees (S16), and MCMC is slow for big models.
* **Py:** `statsmodels.MixedLM` (mature), `PyMC` (mature), `NumPyro` (mature), `bambi` (mature formula interface). R `lme4` is the
  reference.
* **Refs:** [Bates 2014]; [Salvatier 2015]; [Phan 2019]; [Blei 2016]; [Baio & Blangiardo 2010].

### S16. Mixed-effects machine learning: MERF and GPBoost
* **Family:** trees + random effects.
* **What:** MERF alternates (EM) between fitting a random forest on the fixed effects and estimating per-cluster random effects.
  GPBoost combines gradient-boosted trees for the mean with grouped random effects and/or a GP for the residual correlation.
  Both keep tree flexibility while respecting clustered data.
* **Data:** hundreds to millions of rows in clusters.
* **Fields:** B12 (longitudinal patients), B22, real estate, small-area estimation.
* **+/−:** + addresses grouping explicitly, and fixed-effect predictions are valid for *new* groups. − with ~4 rows per group the
  random-effect variance is poorly identified.
* **Py:** `merf` (small, maintained), `gpboost` (mature, active).
* **Refs:** [Hajjem 2014]; [Sigrist 2020].

### S17. Polynomial chaos expansion (PCE)
* **Family:** spectral surrogate for uncertainty quantification.
* **What:** It expands the output in orthogonal polynomials of the (random) inputs. The coefficients give Sobol sensitivity
  indices for free. It is the engineering-UQ workhorse for smooth simulators.
* **Data:** tens to thousands of runs, ≲ 20 inputs (sparse PCE for more).
* **Fields:** structural, aerospace and nuclear engineering, circuit tolerance analysis (B8).
* **+/−:** + very sample-efficient for smooth responses, and sensitivities come free. − it needs input distributions, and
  non-smooth responses are a problem.
* **Py:** `chaospy` (mature), `OpenTURNS` (mature).
* **Refs:** [Feinberg & Langtangen 2015].

### S18. Functional data analysis (FDA)
* **Family:** statistics on curves.
* **What:** It treats each curve (e.g. an Allan deviation vs τ, or a PSD) as one observation of a smooth function. Tools include
  functional PCA, which gives a few scores per curve, and functional linear regression $y = \int \beta(t)x(t)\,dt$, where the
  coefficient is itself a smooth curve that you can plot and interpret.
* **Data:** tens to thousands of curves.
* **Fields:** B12 (growth curves), B16, biomechanics, B22 (spectra), B7.
* **+/−:** + made for "curve in, scalar out" at small n, and the coefficient curves are interpretable. − mostly linear.
* **Py:** `scikit-fda` (moderate, active).
* **Refs:** [Wang 2015].

### S19. Distribution regression and kernel mean embeddings
* **Family:** kernel methods on sets/distributions.
* **What:** Each input is a *bag of samples*, e.g. all skip values of a trace, or the 10 repeated sets. A bag is mapped to
  the mean of its feature embeddings, $\mu_P = \mathbb E_{x\sim P}[\phi(x)]$, and regression is done on that. The same embedding
  gives the MMD two-sample test used for drift detection (P16).
* **Data:** tens to thousands of bags, each with many samples.
* **Fields:** aerosol optical depth from satellite pixels, ecology (B24), B1 (event ensembles), B12.
* **+/−:** + principled theory, and it directly handles "a set of samples → scalar". It is a kernel cousin of DeepSets. − kernel
  and bandwidth choice matters, and quadratic cost.
* **Py:** no dominant package. Easy with `scikit-learn` kernels. MMD in `alibi-detect` (mature).
* **Refs:** [Szabó 2014]; [Gretton 2008].

### S20. State-space models and the Kalman filter (incl. neural variants)
* **Family:** latent dynamical models.
* **What:** A hidden state evolves linearly with Gaussian noise and is observed with noise. The Kalman filter gives the exact
  posterior over the state in real time. Extensions are the EKF/UKF (nonlinear), the ensemble Kalman filter (weather), and
  learned variants (Deep Kalman Filters, KalmanNet).
* **Data:** streaming time series of any length.
* **Fields:** B16 (data assimilation), B20 (navigation), B15, B10 (clock ensembles), B5 (orbit feedback).
* **+/−:** + explicit noise models (process vs measurement noise), optimal for linear-Gaussian systems. For us it is a natural
  model of *slow drift* of noise parameters between packets. − nonlinear and non-Gaussian systems need approximations.
* **Py:** `filterpy` (stable), `statsmodels.tsa.statespace` (mature), `pykalman` (old), `dynamax` (JAX, active).
* **Refs:** [Kalman 1960]; [Evensen 1994]; [Krishnan 2015]; [Revach 2021].

### S21. Hidden Markov models (HMMs)
* **Family:** discrete-state latent models.
* **What:** A hidden discrete state (e.g. "normal / glitching / saturated") switches with Markov probabilities and emits
  observations. The forward-backward and Viterbi algorithms give state posteriors and the most likely state path.
* **Data:** sequences, any length.
* **Fields:** B14 (classical speech recognition), genomics, B15 (regime switching), B12 (sleep staging), ion-channel recordings.
* **+/−:** + interpretable regimes, and it can segment a trace into good and bad stretches (pickup bursts, telegraph noise).
  − Markov and emission assumptions.
* **Py:** `hmmlearn` (maintenance), `pomegranate` (active), `dynamax`.
* **Refs:** [Rabiner 1989].

### S22. ARIMA, GARCH and HAR: classical time-series and volatility models
* **Family:** econometric time series.
* **What:** ARIMA models the mean of a series as autoregressive plus moving-average terms. GARCH models the *variance* as
  depending on past squared shocks and past variance. HAR models the *realized volatility* as a sum of daily, weekly and
  monthly averages, which mimics long memory. Volatility estimation from high-frequency data is the finance twin of
  "estimate the noise level of a waveform at several time scales", much like an Allan deviation.
* **Data:** single long series, or panels.
* **Fields:** B15, B16, energy, B10.
* **+/−:** + fast, interpretable, and noise-level-centric (GARCH/HAR). − univariate, linear, assumes stationarity.
* **Py:** `statsmodels` (ARIMA, SARIMAX; mature), `arch` (GARCH, HAR; mature).
* **Refs:** [Bollerslev 1986]; [Corsi 2009]; [Seabold 2010].

### S23. Bayesian nonparametrics (Dirichlet-process mixtures, etc.)
* **Family:** infinite-capacity Bayesian models.
* **What:** Priors whose complexity grows with the data. For example, a Dirichlet-process mixture lets the data decide the
  number of clusters. The family also includes the Indian buffet process (latent features) and GPs.
* **Data:** small to medium.
* **Fields:** B24, genomics, B1 (clustering of event types), B15.
* **+/−:** + automatic model size, and it could discover "regimes" of CCD behaviour without fixing their number. − inference
  is slow and fiddly.
* **Py:** `scikit-learn` (`BayesianGaussianMixture`; mature), `PyMC`/`NumPyro` (custom).
* **Refs:** [Gershman & Blei 2011].

### S24. Bayesian online changepoint detection (BOCPD)
* **Family:** sequential Bayesian inference.
* **What:** It maintains a posterior over the "run length" since the last change in the data-generating process. It flags,
  online, when a stream (e.g. per-packet noise or pedestal level) has shifted.
* **Data:** streams, univariate or multivariate.
* **Fields:** B15, B12 (EEG and ECG events), B23 (fault onset), B10.
* **+/−:** + principled, online, gives a probability of a change. − needs a model for each segment, and hazard-rate tuning.
* **Py:** `ruptures` (offline changepoints; mature), `bayesian_changepoint_detection` (small), `river` (drift detectors).
* **Refs:** [Adams & MacKay 2007].

### S25. Multitaper and classical spectral estimation
* **Family:** signal processing (not ML, but a feature source).
* **What:** Thomson's multitaper method averages periodograms computed with several orthogonal tapers (Slepian sequences). The
  result is a low-variance, low-leakage PSD estimate from a *short* record, plus an F-test for line components (pickup tones).
  Welch averaging and Allan variance are the usual alternatives.
* **Data:** single series of any length. It is especially valuable for short ones.
* **Fields:** B13, B3, B12 (EEG), B10, B4.
* **+/−:** + better PSD features from 500-skip traces. The harmonic F-test detects pickup lines. − it is a feature
  extractor, not a predictor.
* **Py:** `scipy.signal.windows.dpss` (mature), `spectrum` (moderate), `mne.time_frequency.psd_array_multitaper` (mature),
  `allantools` (Allan variance; stable).
* **Refs:** [Thomson 1982].

### S26. Symbolic regression (PySR, AI Feynman, SISSO)
* **Family:** equation discovery / genetic programming.
* **What:** It searches over mathematical expressions (built from +, ×, exp, log, √ and so on) for a compact formula that fits
  the data, trading accuracy against complexity along a Pareto front. AI Feynman adds physics-inspired tricks (dimensional
  analysis, symmetry and separability tests). SISSO builds a huge space of candidate descriptors and selects a few by
  compressed sensing. A neural network can also be "distilled" into symbols (Cranmer et al. 2020).
* **Data:** tens to thousands of rows, ≲ 10 relevant inputs.
* **Fields:** physics (B1, B3), B17 (SISSO descriptors), B16 (parameterizations), B13.
* **+/−:** + it can return a *readable law*, e.g. how σe depends on CDS window and clock voltage. − combinatorial search,
  sensitive to noise and to the inputs offered, and it struggles with more than a handful of variables.
* **Py:** `pysr` (mature, active), `gplearn` (stable), `aifeynman` (research), SISSO (Fortran; research).
* **Refs:** [Cranmer 2023]; [Udrescu & Tegmark 2019]; [Udrescu 2020]; [Ouyang 2017]; [Cranmer 2020].

### S27. Sparse identification of nonlinear dynamics (SINDy)
* **Family:** sparse regression for dynamical systems.
* **What:** It builds a library of candidate terms (polynomials, sines and so on) of the state, and uses sparse regression to
  find the few terms in $\dot x = \Theta(x)\xi$. The result is an interpretable differential equation learned from time series.
* **Data:** time series of a few state variables, often hundreds to thousands of samples.
* **Fields:** fluid dynamics, B20, biology, plasma physics, B5.
* **+/−:** + interpretable dynamics from little data. − needs derivatives (noise-sensitive) and a well-chosen library. Our
  traces are noise-dominated, so the direct use is limited, but the same idea applies to slow drift of operating points.
* **Py:** `pysindy` (mature, active).
* **Refs:** [Brunton 2016]; [Kaptanoglu 2021].

### S28. Koopman operator and dynamic mode decomposition (DMD)
* **Family:** linear-operator models of nonlinear dynamics.
* **What:** DMD finds the best linear operator that advances snapshots in time. Its eigenvalues give oscillation frequencies
  and growth or decay rates. Koopman methods learn a feature space (often with a neural network) in which nonlinear dynamics
  become linear.
* **Data:** snapshot sequences, often high-dimensional fields.
* **Fields:** fluid dynamics, power grids, B12 (neural recordings), B16.
* **+/−:** + extracts coherent modes and frequencies, which could separate pickup modes across the 4 amplifiers' traces. − assumes
  approximately linear evolution in the chosen space.
* **Py:** `pydmd` (mature), `pykoopman` (moderate).
* **Refs:** [Tu 2013]; [Lusch 2017].

### S29. Reservoir computing (echo-state networks)
* **Family:** fixed random recurrent network + linear readout.
* **What:** A large random recurrent network (the "reservoir") is driven by the input series. Only a linear readout is
  trained. It is the recurrent analogue of ROCKET's random convolutions.
* **Data:** time series. The training cost is tiny because only ridge regression is fitted.
* **Fields:** chaotic-system forecasting, B9 (channel equalization), neuromorphic and photonic hardware.
* **+/−:** + very cheap and few trained parameters. − sensitive to reservoir hyperparameters, and it shines at forecasting
  rather than at extracting noise statistics.
* **Py:** `reservoirpy` (moderate, active).
* **Refs:** [Pathak 2017].

---

## A2. Trees and ensembles

### T01. Random forests
* **Family:** bagged decision trees.
* **What:** Many deep trees, each grown on a bootstrap sample and considering a random subset of inputs at each split, are
  averaged. The out-of-bag error and permutation importance come free.
* **Data:** 10²–10⁶ rows. Tabular inputs with mixed types.
* **Fields:** everywhere: B1, B12, B22, B24 (species-distribution models), B7.
* **+/−:** + robust defaults, little tuning, handles irrelevant inputs. − poor extrapolation (piecewise constant), and its
  variance estimate is not a calibrated uncertainty (use quantile forests or conformal).
* **Py:** `scikit-learn` (mature).
* **Refs:** [Breiman 2001].

### T02. Gradient-boosted trees (GBM, XGBoost, LightGBM, CatBoost)
* **Family:** boosted decision trees.
* **What:** Trees are added sequentially, each fitted to the gradient of the loss of the current ensemble (Friedman's gradient
  boosting). XGBoost added regularization and engineering, LightGBM added histogram binning and gradient-based sampling, and
  CatBoost added ordered boosting against target leakage. On medium-size tabular data, tuned GBMs remain the benchmark to beat
  (Grinsztajn et al. 2022; Shwartz-Ziv & Armon 2021).
* **Data:** 10³–10⁸ rows. At ≲ 100 rows they tend to lose to GPs and TabPFN.
* **Fields:** B15, B21, B7, B1 (BDTs for pulse-shape discrimination in Majorana), B23.
* **+/−:** + state of the art on tabular data, monotonic constraints possible (useful for physics: noise rises with a given
  setting). − weak at tiny n, no native uncertainty, no extrapolation.
* **Py:** `xgboost`, `lightgbm`, `catboost` (all mature), `scikit-learn` `HistGradientBoosting`.
* **Refs:** [Friedman 2001]; [Chen & Guestrin 2016]; [Ke 2017]; [Prokhorenkova 2017]; [Grinsztajn 2022]; [Shwartz-Ziv 2021]; [Arnquist 2022].

### T03. Quantile regression forests and NGBoost (prior review §5.1)
* **Family:** probabilistic tree ensembles.
* **What:** Quantile forests keep all training targets in the leaves and read off conditional quantiles. NGBoost boosts the
  parameters of a distribution using natural gradients.
* **Py:** `quantile-forest`, `ngboost` (both mature). **Refs:** see prior review.

### T04. Bayesian additive regression trees (BART)
* **Family:** Bayesian sum-of-trees.
* **What:** It models $f(x) = \sum_m g(x; T_m, M_m)$ with a prior that keeps each tree small (a "weak learner" prior), and
  samples the posterior over trees by MCMC. It is a tree-based model with a full posterior, and it is robust at small n.
* **Data:** 10²–10⁵ rows.
* **Fields:** causal inference (B12 treatment effects, econometrics), B24, policy evaluation.
* **+/−:** + good default uncertainty for a tree model, and automatic relevance via split counts. − MCMC cost, and it
  extrapolates poorly like all trees.
* **Py:** `pymc-bart` (moderate, active); R `BART`/`dbarts` are the references.
* **Refs:** [Chipman 2008].

### T05. Glass-box additive models: Explainable Boosting Machines (EBM) and Neural Additive Models (NAM)
* **Family:** interpretable additive ensembles.
* **What:** An EBM learns a GAM ($\sum_j f_j(x_j) + \sum f_{jk}(x_j,x_k)$) by cyclic boosting of shallow trees, one feature at a
  time, and selects pairwise interactions automatically. A NAM learns each $f_j$ with a small neural network. Both give an exact
  per-feature contribution plot.
* **Data:** 10³–10⁶ rows typical. They work at a few hundred rows with strong regularization.
* **Fields:** B12 (healthcare risk), B15 (credit), high-stakes decisions.
* **+/−:** + often matches GBM accuracy while being fully interpretable. − interactions are limited to pairs, and there is no
  native predictive uncertainty.
* **Py:** `interpret` (EBM; mature), NAM (research code).
* **Refs:** [Nori 2019]; [Agarwal 2020]; [Rudin 2018].

### T06. Stacking and Bayesian model averaging of predictive distributions
* **Family:** ensemble combination.
* **What:** It combines the predictive distributions of several models with weights chosen to maximize leave-one-out (or
  grouped-CV) log score ("stacking of predictive distributions"). Unlike Bayesian model averaging it does not assume that one
  model is true.
* **Data:** any. It needs out-of-fold predictions.
* **Fields:** B16 (multi-model ensembles), B12 (epidemic forecasting hubs), Kaggle-style practice.
* **+/−:** + can combine physics (model A), GP and TabPFN predictions into one calibrated distribution. − with ≤ 100 packets the
  weights themselves are noisy.
* **Py:** `arviz` (stacking weights; mature), `scikit-learn` `StackingRegressor` (point predictions).
* **Refs:** [Yao 2018].

---

## A3. Deep-learning architectures

### D01. MLPs and deep models for tabular data (RealMLP, TabM, FT-Transformer)
* **Family:** neural networks for tables.
* **What:** Well-tuned MLPs with careful preprocessing ("RealMLP"), parameter-efficient ensembles of MLPs that share most
  weights (TabM), and transformers that embed each feature as a token (FT-Transformer). Benchmarks (Grinsztajn 2022;
  Shwartz-Ziv & Armon 2021) show GBMs still win on many medium-size tables. The newer MLP variants have narrowed the gap.
* **Data:** 10³–10⁶ rows. Tiny n favours in-context models (TabPFN, prior review §1) instead.
* **Fields:** B15, B21, B12, B7.
* **+/−:** + differentiable (usable inside BO or end-to-end with trace encoders), and ensembles give uncertainty. − worse than
  trees or GPs at our n without strong priors.
* **Py:** `pytabkit` (RealMLP; moderate), `tabm` (moderate), `rtdl` (moderate).
* **Refs:** [Holzmüller 2024]; [Gorishniy 2024]; [Gorishniy 2021]; [Grinsztajn 2022].

### D02. Convolutional networks for 1-D signals (ResNet, InceptionTime, TCN, WaveNet)
* **Family:** CNN.
* **What:** Stacks of 1-D convolutions learn local waveform patterns. Residual connections, multi-scale Inception blocks and
  dilated causal convolutions (TCN, WaveNet) grow the receptive field exponentially. These are standard baselines for time-series
  classification and regression.
* **Data:** 10³–10⁶ labelled series, or pretrained (P01–P02) when labels are few.
* **Fields:** B12 (ECG arrhythmia detection at cardiologist level), B13 (PhaseNet), B4 (DeepClean), B14 (WaveNet), B9 (radio
  modulation), B23.
* **+/−:** + end-to-end from raw samples, and the inductive bias suits waveforms. − data-hungry, and noise statistics
  (second-order) are learned only indirectly. Global pooling of squared activations helps.
* **Py:** `torch` (mature), `tsai` (moderate), `aeon` deep-learning module (moderate).
* **Refs:** [Wang 2016]; [Ismail Fawaz 2019 InceptionTime]; [Ismail Fawaz 2018 review]; [Bai 2018]; [van den Oord 2016]; [Rajpurkar 2017].

### D03. Recurrent networks (LSTM, GRU) and probabilistic RNN forecasters (DeepAR)
* **Family:** RNN.
* **What:** Gated recurrent cells carry a memory through a sequence. DeepAR trains one RNN across *many related series* and
  outputs a predictive distribution at each step. In hydrology, LSTMs trained across hundreds of catchments, with static
  catchment attributes as extra inputs, beat physics models. They also beat per-catchment models: "never train on a single
  basin".
* **Data:** many related series, 10²–10⁵ series.
* **Fields:** B16 (rainfall-runoff), B15, retail demand, B12, B23 (remaining-useful-life).
* **+/−:** + learns across many units with static attributes, a pattern analogous to amplifiers or boards with settings.
  − slower than CNNs and transformers, and data-hungry.
* **Py:** `torch`, `gluonts` (DeepAR; mature), `neuralhydrology` (moderate).
* **Refs:** [Kratzert 2018]; [Kratzert 2019]; [Salinas 2017].

### D04. Transformers (incl. time-series transformers)
* **Family:** attention models.
* **What:** Self-attention lets every position (token, patch, feature, set element) weigh every other position. In time
  series, inputs are cut into patches (PatchTST, prior review §2.3). Transformers are the backbone of all modern foundation
  models (P19).
* **Data:** large, unless pretrained.
* **Fields:** B25, B26, B14, B13 (EQTransformer), B16, B19.
* **+/−:** + flexible, scales, permutation-aware variants work for sets. − needs pretraining or lots of data, and with 100
  labels it will overfit.
* **Py:** `torch`, `transformers` (mature).
* **Refs:** [Vaswani 2017]; [Mousavi 2020].

### D05. Structured state-space sequence models (S4, Mamba)
* **Family:** deep state-space models.
* **What:** Each layer is a learned *linear* state-space system $\dot h = Ah + Bx$, $y = Ch$, discretized and computed as a
  long convolution (S4) or with an input-dependent "selective" scan (Mamba). They handle sequences of 10⁴–10⁶ samples in linear
  time, and they relate directly to classical filters.
* **Data:** long sequences, medium to large datasets.
* **Fields:** B14 (raw audio), B12 (long ECG/EEG), genomics, B25.
* **+/−:** + could read a *whole raw trace* (all pixels × skips, ~10⁵ samples), and the linear-filter structure resembles CDS
  filtering. − needs data or pretraining, and the tooling is younger.
* **Py:** `mamba-ssm` (active, CUDA), `s4` research code.
* **Refs:** [Gu 2021]; [Gu & Dao 2023].

### D06. Graph neural networks (GNNs)
* **Family:** message passing on graphs.
* **What:** Nodes exchange messages along edges and update their states, so the model respects the topology and is invariant
  to node ordering. Examples are detector hits connected into tracks, atoms in crystals, mesh cells in simulations, and circuit
  netlists.
* **Data:** graphs with 10–10⁶ nodes. Datasets of 10³+ graphs.
* **Fields:** B1 (Exa.TrkX tracking, ParticleNet jet tagging), B17 (CGCNN), B18, B8 (GCN-RL transistor sizing), B16 (GraphCast),
  B21.
* **+/−:** + natural for structured systems. A CCD module (amplifiers sharing clocks and bias lines) or the clock and bias
  network could be a graph. − needs structure and data, and is overkill for 4 nodes.
* **Py:** `torch_geometric` (mature), `dgl` (mature).
* **Refs:** [Kipf & Welling 2016]; [Battaglia 2018]; [Ju 2021]; [Qu & Gouskos 2019]; [Pfaff 2020].

### D07. Set models: DeepSets (Set Transformer in prior review §6)
* **Family:** permutation-invariant networks.
* **What:** $f(X) = \rho(\sum_{x\in X}\phi(x))$ is a universal form for functions of sets. Attention-based pooling generalizes it.
* **Fields:** B1 (particle clouds), B3 (point sources), B19, our model E.
* **Py:** `torch` (no package needed). **Refs:** [Zaheer 2017].

### D08. Neural ODEs, SDEs and CDEs
* **Family:** continuous-time deep models.
* **What:** A neural network defines the vector field of an ODE, $\dot h = f_\theta(h,t)$, which is solved with an ODE
  solver. Neural SDEs add a learned diffusion term and so model *noise processes* themselves. They can be trained as
  generative models of time series. Neural CDEs handle irregularly sampled data.
* **Data:** time series (often irregular). Small to medium datasets with physics-like structure.
* **Fields:** B12 (irregular clinical series), B15 (SDE market models), B20, B16, B17 (kinetics).
* **+/−:** + a neural SDE is a learnable *generative model of the video noise* (white + 1/f + drift) that could serve as a
  simulator for SBI (G07). − slow training, stiffness problems.
* **Py:** `torchdiffeq` (mature), `torchsde` (moderate), `diffrax` (JAX; mature), `torchcde` (moderate).
* **Refs:** [Chen 2018]; [Kidger 2021 SDE]; [Kidger 2020 CDE].

### D09. Physics-informed and physics-guided neural networks (PINNs, PGNN, UDEs, theory-guided data science)
* **Family:** hybrid physics–ML.
* **What:** PINNs add the residual of a known differential equation to the loss, so the network satisfies physics at
  "collocation points" without labels. Physics-guided NNs feed physics-model outputs in as inputs and penalize physically
  inconsistent predictions. Universal differential equations embed neural networks *inside* known equations so that they learn
  only the unknown term. This is the general version of model A ("Allan physics + learned correction").
* **Data:** few labels plus known equations.
* **Fields:** B16 (lake temperature PGNN, NeuralGCM), fluid mechanics, B17, B12 (pharmacokinetics), B5.
* **+/−:** + data efficiency and physically plausible extrapolation. − PINN training is notoriously finicky, and the gain
  depends on how good the physics is.
* **Py:** `deepxde` (mature), `neuromancer` (moderate), Julia `DiffEqFlux`/`SciML` (mature).
* **Refs:** [Raissi 2017]; [Daw 2017]; [Karpatne 2016]; [Rackauckas 2020]; [Kochkov 2023].

### D10. Neural operators (FNO, DeepONet)
* **Family:** learning maps between function spaces.
* **What:** They learn an operator $G: u(\cdot) \mapsto v(\cdot)$, e.g. PDE coefficient field → solution field. The Fourier
  Neural Operator applies learned filters in Fourier space. DeepONet uses a "branch" network for the input function and a
  "trunk" network for the query location.
* **Data:** 10³–10⁵ simulated function pairs.
* **Fields:** PDE surrogates, B16 (weather), materials, B20.
* **+/−:** + resolution-independent and fast surrogates. A possible map is "settings → full Allan curve or PSD" (a function
  output). − needs many training pairs, usually from simulators.
* **Py:** `neuraloperator` (moderate, active), `deepxde` (DeepONet).
* **Refs:** [Li 2020]; [Lu 2019].

### D11. Implicit neural representations (SIREN, NeRF-style)
* **Family:** coordinate networks.
* **What:** A small MLP maps coordinates (t, or x,y) to signal values. Sine activations (SIREN) represent fine detail and
  derivatives. The network weights *are* the compressed signal, and conditioning on a code gives a family of signals.
* **Data:** one or a few signals (per-signal fitting), or many signals with a shared hypernetwork.
* **Fields:** B26 (3-D scenes), medical imaging, B14 (audio compression).
* **+/−:** + continuous representation of curves, e.g. Allan deviation as a function of log τ. − rarely better than
  simple smoothing for 1-D curves, and the high-frequency noise content is the hard part.
* **Py:** research code. Easy in `torch`.
* **Refs:** [Sitzmann 2020].

### D12. Kolmogorov–Arnold networks (KANs)
* **Family:** spline-based networks.
* **What:** They put learnable 1-D functions (splines) on edges instead of fixed activations on nodes. After training,
  individual edge functions can be inspected or replaced by symbolic formulas, which bridges toward symbolic regression.
* **Data:** small to medium, low-dimensional scientific data.
* **Fields:** physics and maths discovery (proposed), early applications in many fields.
* **+/−:** + interpretable 1-D pieces, "symbolification". − the claims of superior accuracy are contested in follow-up work
  [unverified], training is slower, and it is young.
* **Py:** `pykan` (research, active).
* **Refs:** [Liu 2024].

### D13. Mixture of experts (MoE)
* **Family:** conditional computation.
* **What:** A gating network routes each input to one or a few "expert" sub-models, and the output is a gate-weighted sum. The
  classical (small) version fits different regimes with different models. In modern LLMs, sparse MoE layers scale capacity.
* **Data:** any. The classical small-n version works with GPs or linear experts ("mixture of GP experts").
* **Fields:** B25 (sparse MoE LLMs), B15 (regime models), B20 (multiple dynamics regimes).
* **+/−:** + handles regime changes (e.g. clock-settings regions where a different noise source dominates). − gating needs
  enough data per regime.
* **Py:** `torch`; small-scale: write by hand, or `sklearn` mixture + per-cluster regressors.
* **Refs:** [Shazeer 2017].

### D14. Hypernetworks and FiLM conditioning
* **Family:** conditional architectures.
* **What:** A hypernetwork generates the weights of another network from a conditioning input. FiLM, a cheaper version, has the
  conditioning input produce a per-channel scale and shift, $\gamma(c)\odot h + \beta(c)$, applied to feature maps.
* **Data:** medium.
* **Fields:** B26 (visual question answering), B14 (speaker-conditioned TTS), B20, B16.
* **+/−:** + a clean way to make a trace encoder *settings-aware* (settings → FiLM parameters of the trace CNN), so that the same
  waveform pattern can mean different things at different CDS windows. − more parameters to learn.
* **Py:** `torch` (a few lines), `hypnettorch` (research).
* **Refs:** [Ha 2016]; [Perez 2017].

### D15. Learnable and fixed signal front-ends: wavelet scattering, SincNet
* **Family:** signal-processing-inspired layers.
* **What:** The wavelet scattering transform cascades wavelet filtering, modulus and averaging *with fixed filters*. It gives
  stable, translation-invariant features that capture second- and higher-order statistics. For audio it is close to mel
  spectra plus their modulations, and it works well with little data. SincNet makes the first CNN layer a bank of learnable
  band-pass filters (only cut-off frequencies are learned).
* **Data:** small to medium. Scattering needs no training.
* **Fields:** B14, B12 (ECG/EEG), B13, B23 (vibration), B3.
* **+/−:** + scattering coefficients are essentially *multi-scale noise-power features*. They are a principled, untrained
  generalization of Allan deviation and PSD bands, and are therefore well matched to noise-level regression from 500-skip traces.
  − fixed filters may miss very specific pickup lines.
* **Py:** `kymatio` (mature), SincNet (research code; also in `speechbrain`).
* **Refs:** [Mallat 2011]; [Andén & Mallat 2013]; [Andreux 2018]; [Ravanelli 2018].

### D16. Neural basis-expansion forecasters (N-BEATS)
* **Family:** deep MLP for time series.
* **What:** Stacks of fully connected blocks that output coefficients of basis functions (trend polynomials, Fourier terms).
  The forecast decomposes into interpretable components.
* **Data:** many series.
* **Fields:** B15, retail, energy.
* **+/−:** + interpretable decomposition. − aimed at forecasting, not at extracting noise statistics.
* **Py:** `darts`, `neuralforecast` (mature).
* **Refs:** [Oreshkin 2019].

### D17. Neural processes (prior review §3)
* **Family:** meta-learned stochastic processes.
* **What:** A network maps a context set of (x, y) pairs plus a query to a predictive distribution. It is amortized GP-like
  inference.
* **Refs:** [Garnelo 2018]; see prior review.

---

## A4. Feature engineering for series and curves

### F01. Massive feature extraction (tsfresh, catch22, hctsa)
* **Family:** feature-based time-series analysis.
* **What:** It computes hundreds to thousands of interpretable summary statistics per series (autocorrelations, entropy,
  spectral moments, stationarity tests, extreme values, …) and then selects the relevant ones by hypothesis tests or a
  classifier. hctsa has ~7 700 features. catch22 is a 22-feature distilled subset.
* **Data:** any number of series. Works at small n when combined with selection and regularization.
* **Fields:** B12, B23, B24, neuroscience, B7.
* **+/−:** + interpretable, cheap, a systematic alternative to our 85 hand-made features, and a good check on what we missed.
  − many correlated features invite false discoveries at n ≈ 100. Selection must be inside CV.
* **Py:** `tsfresh` (mature), `pycatch22` (mature), hctsa (MATLAB), `aeon`/`sktime` wrappers.
* **Refs:** [Christ 2016]; [Lubba 2019]; [Fulcher & Jones 2016].

### F02. Random-convolution features (ROCKET / MiniRocket) (prior review §2.1)
* Untrained random kernels + linear head. A strong cheap baseline for time-series extrinsic regression. See prior review.

### F03. Path signatures
* **Family:** rough-path theory features.
* **What:** The signature of a path is the sequence of its iterated integrals, $S(X)^{i_1\dots i_k} = \int dX^{i_1}\cdots dX^{i_k}$.
  It is a universal, reparametrization-invariant description of a multichannel series, and linear models on signatures can
  approximate any continuous function of the path.
* **Data:** small to medium. Multichannel series (e.g. the 4 amplifiers' traces together, or trace + clock phase).
* **Fields:** B12 (psychiatric mood series, sepsis prediction), B15, handwriting recognition.
* **+/−:** + captures *cross-channel ordering effects* (cross-talk, correlated pickup between amplifiers) with few
  parameters. − the feature count grows fast with depth and channel number, and it needs scaling and lead-lag tricks.
* **Py:** `iisignature` (stable), `signatory` (PyTorch; moderate), `esig`.
* **Refs:** [Chevyrev & Kormilitzin 2016]; [Kidger & Lyons 2020].

### F04. Data augmentation for time series (and mixup)
* **Family:** regularization.
* **What:** It creates label-preserving variants of training signals (jitter, scaling, time warping, window slicing,
  permutation). mixup trains on convex combinations of pairs of examples and their labels.
* **Data:** small labelled sets.
* **Fields:** B12, B14, B23, B26.
* **+/−:** + cheap regularization. − many standard augmentations *change noise statistics* (scaling changes σ, smoothing changes the
  PSD), so for our task only noise-preserving ones are valid (see prior review §2.3).
* **Py:** `tsaug` (small), `torchaudio` transforms (mature).
* **Refs:** [Iwana & Uchida 2020]; [Zhang 2017 mixup].

---

## A5. Generative models and simulation-based inference

### G01. Variational autoencoders (VAEs)
* **Family:** latent-variable generative model.
* **What:** An encoder maps x to a distribution over a low-dimensional latent z, and a decoder reconstructs x. It is trained
  by maximizing the evidence lower bound. The latent space is a compressed, often disentangled summary, usable for anomaly
  scores (reconstruction error), for semi-supervised learning, and for clustering.
* **Data:** 10³+ unlabelled examples.
* **Fields:** B1 (anomaly detection, fast simulation), B18 (molecule generation), B12, B23.
* **+/−:** + uses unlabelled traces, and gives a probabilistic latent. − blurry reconstructions, and the latent may ignore
  low-amplitude noise structure.
* **Py:** `torch`, `pythae` (moderate).
* **Refs:** [Kingma & Welling 2013].

### G02. Normalizing flows
* **Family:** exact-likelihood generative models.
* **What:** A chain of invertible, differentiable transformations of a simple base distribution, with exact density via the
  change-of-variables formula. Flows are the density estimators inside neural SBI (G07).
* **Data:** 10³–10⁶ samples, moderate dimension.
* **Fields:** B1 (density estimation, unfolding), B3/B4 (posterior estimation), lattice field theory.
* **+/−:** + exact likelihoods and conditional densities (a full predictive distribution for σe, λ, gain). − hard to train in very
  high dimensions.
* **Py:** `zuko` (active), `nflows` (stable), `normflows` (moderate).
* **Refs:** [Papamakarios 2019 review]; [Dinh 2016].

### G03. Diffusion and score-based models
* **Family:** iterative denoising generative models.
* **What:** Data are gradually noised. A network learns to reverse the process by estimating the score $\nabla_x\log p_t(x)$.
  Sampling runs the learned reverse SDE. Conditional versions do imputation (CSDI) and inverse problems.
* **Data:** large.
* **Fields:** B26 (images), B14, B16 (ensemble weather), B19 (protein design), B12.
* **+/−:** + state-of-the-art sample quality, and flexible conditioning. − data-hungry, slow sampling, heavy for our scale.
* **Py:** `diffusers` (mature), CSDI research code.
* **Refs:** [Ho 2020]; [Song 2020]; [Tashiro 2021].

### G04. Generative adversarial networks (GANs)
* **Family:** adversarial generative models.
* **What:** A generator and a discriminator are trained against each other. In physics they were used as fast simulators
  (CaloGAN for calorimeter showers). For audio, WaveGAN generates raw waveforms.
* **Data:** large.
* **Fields:** B1, B26, B14.
* **+/−:** + fast sampling. − unstable training, mode collapse, no likelihood.
* **Py:** `torch`.
* **Refs:** [Goodfellow 2014]; [Paganini 2017]; [Donahue 2018].

### G05. Energy-based models (EBMs)
* **Family:** unnormalized density models.
* **What:** They learn a scalar energy $E_\theta(x)$ with $p(x)\propto e^{-E_\theta(x)}$. Low energy means "plausible". Training
  and sampling use MCMC (Langevin dynamics). They are useful for out-of-distribution detection and compositional constraints.
* **Data:** medium to large.
* **Fields:** B26, B19, anomaly detection.
* **+/−:** + flexible and composable. − expensive and unstable training.
* **Py:** `torch` (research code).
* **Refs:** [Du & Mordatch 2019].

### G06. Approximate Bayesian computation (ABC)
* **Family:** likelihood-free inference (classical).
* **What:** It simulates data from parameters drawn from the prior and keeps the parameters whose simulated summary statistics
  are close to the observed ones, optionally correcting by regression. It is the ancestor of neural SBI.
* **Data:** a simulator, plus 10⁴–10⁶ simulations.
* **Fields:** population genetics, B24, epidemiology, B16.
* **+/−:** + simple, any simulator. − inefficient, and the results depend on the choice of summaries and tolerance.
* **Py:** `pyabc` (mature), `ELFI` (moderate).
* **Refs:** [Beaumont 2002].

### G07. Neural simulation-based inference (NPE, NLE, NRE) (prior review §4.2)
* **Family:** amortized likelihood-free inference.
* **What:** Networks trained on simulated (θ, x) pairs learn the posterior (NPE), the likelihood (NLE) or a likelihood ratio
  (NRE). Inference on new data is then a single forward pass. DINGO does real-time gravitational-wave parameter estimation on
  detector noise. Simulation-based calibration checks that the posteriors are calibrated.
* **Data:** a simulator, plus 10⁴–10⁶ simulations.
* **Fields:** B4, B3, B1, B12 (neuroscience models), B24.
* **+/−:** + calibrated posteriors over *noise-model parameters* from a trace, if a trace simulator exists. − simulator
  mismatch.
* **Py:** `sbi` (mature), `lampe` (moderate), `swyft` (moderate).
* **Refs:** [Papamakarios & Murray 2016]; [Papamakarios 2018]; [Hermans 2019]; [Dax 2021]; [Talts 2018].

### G08. Physics-based synthetic data, domain randomization and self-supervised denoising
* **Family:** data generation for learning.
* **What:** It trains on synthetic data from a physical noise model whose parameters are randomized widely, so that the real
  world looks like "one more variation" (domain randomization, from robotics). In imaging, physics-based CCD noise synthesis
  (shot, dark, read, PRNU, cosmic rays) creates training pairs for denoisers. Noise2Noise shows that a denoiser can be trained
  from pairs of *independent noisy* observations alone. That idea transfers to repeated skips and sets of the same pixel.
* **Data:** an unlimited synthetic supply plus few real examples.
* **Fields:** B20 (sim-to-real), B26 (camera denoising), B3 (astronomical CCD denoising), B2.
* **+/−:** + addresses label scarcity. Our `src/ml/synthetic.py` toy world is exactly such a generator. − the reality gap:
  pickup and drift that the simulator lacks.
* **Py:** `torch`. No specific package.
* **Refs:** [Tobin 2017]; [Lehtinen 2018]; [Liu 2026 CCD noise].

---

## A6. Learning paradigms

### P01. Contrastive self-supervised learning (SimCLR, CPC; TS2Vec and TF-C in prior review §2.3)
* **Family:** self-supervised representation learning.
* **What:** It learns an encoder by pulling together embeddings of two "views" of the same example (augmentations, or
  past/future segments in CPC) and pushing apart those of different examples (the InfoNCE loss). Labels are not needed. A small
  head is then trained on the few labelled examples.
* **Data:** 10³–10⁷ unlabelled examples, then few labels.
* **Fields:** B26, B14 (CPC on raw audio), B12 (biosignal foundation models), B13.
* **+/−:** + uses the unlabelled traces. − everything hinges on augmentations that preserve the label. For noise regression,
  most standard views do *not* (see F04).
* **Py:** `lightly` (mature), `torch`.
* **Refs:** [Chen 2020 SimCLR]; [van den Oord 2018 CPC].

### P02. Masked modelling (BERT, MAE, wav2vec 2.0, HuBERT)
* **Family:** self-supervised by reconstruction or prediction of hidden parts.
* **What:** It hides parts of the input (words, image patches, audio frames) and trains the network to predict them, either the
  raw values (MAE) or discrete targets (wav2vec 2.0, HuBERT). In speech, a model pretrained this way on unlabelled audio
  reaches good recognition with very few labelled minutes.
* **Data:** large unlabelled corpora, then few labels.
* **Fields:** B25, B26, B14, B12, B13.
* **+/−:** + the strongest recipe for "few labels, many raw signals" in speech and biosignals. − predicting masked *noise* is
  impossible by definition, so the model learns the predictable part. Masked modelling of the *noise statistics* (e.g. a
  masked PSD band) is the relevant adaptation.
* **Py:** `transformers` (mature), `fairseq` (stable), `torch`.
* **Refs:** [Devlin 2018]; [He 2021 MAE]; [Baevski 2020]; [Hsu 2021].

### P03. Non-contrastive self-supervision (BYOL, VICReg)
* **Family:** self-supervised without negative pairs.
* **What:** BYOL predicts one view's embedding from the other's through a slowly updated "target" network. VICReg uses explicit
  variance, invariance and covariance regularizers to prevent collapse. Neither needs large batches of negatives.
* **Data:** unlabelled sets of moderate size.
* **Fields:** B26, B12, remote sensing.
* **+/−:** + works with smaller batches, which suits a few thousand traces. − same augmentation caveat as P01.
* **Py:** `lightly`, `solo-learn` (moderate).
* **Refs:** [Grill 2020]; [Bardes 2021].

### P04. Semi-supervised learning (mean teacher, FixMatch, self-training)
* **Family:** learning from labelled + unlabelled data jointly.
* **What:** Consistency regularization makes predictions stable under perturbations of unlabelled inputs. A "mean teacher" is
  an exponential moving average of the student. Pseudo-labelling or self-training labels confident unlabelled examples and
  retrains (FixMatch, Noisy Student). For regression, pseudo-labels need uncertainty filtering.
* **Data:** few labels plus many unlabelled examples from the *same* distribution.
* **Fields:** B26, B12, B14, B1.
* **+/−:** + directly targets our labelled/unlabelled ratio. − it can confirm its own errors, and the gains are smaller in
  regression than in classification.
* **Py:** `torch`; `USB` (unified semi-supervised benchmark; moderate).
* **Refs:** [Tarvainen & Valpola 2017]; [Sohn 2020]; [Xie 2019].

### P05. Transfer learning and parameter-efficient fine-tuning (LoRA, domain-adaptive pretraining)
* **Family:** reuse of pretrained models.
* **What:** It starts from a model pretrained on a big source domain and adapts it with few target labels. LoRA adds trainable
  low-rank matrices to frozen weights, so very few parameters are trained. *Continued* pretraining on unlabelled in-domain data
  before fine-tuning ("don't stop pretraining") gives consistent gains in NLP.
* **Data:** few target labels.
* **Fields:** B25, B26, B12, B19 (protein-LM embeddings + small heads).
* **+/−:** + low data needs. − needs a relevant source model. For skipper-CCD noise, "relevant" pretraining would have to come
  from our own (or other experiments') traces.
* **Py:** `peft` (mature), `transformers`.
* **Refs:** [Hu 2021 LoRA]; [Gururangan 2020].

### P06. Domain adaptation and domain generalization (DANN, IRM, Group DRO)
* **Family:** learning under distribution shift between "environments".
* **What:** DANN trains features from which a discriminator cannot tell source from target domain. Invariant risk minimization
  looks for features whose optimal predictor is the same in every training environment. Group DRO minimizes the *worst-group*
  loss. Environments for us could be boards, cool-downs, or amplifiers.
* **Data:** several labelled environments, and possibly an unlabelled target.
* **Fields:** B12 (multi-hospital models), B26, B13 (transfer between regions), B23.
* **+/−:** + targets the "new board / new cool-down" problem and the worst-amplifier metric. − IRM-type methods are fragile in
  practice, and they need several environments.
* **Py:** `adapt` (domain adaptation; moderate), `torch`; `WILDS` benchmark.
* **Refs:** [Ganin 2015]; [Arjovsky 2019]; [Sagawa 2019].

### P07. Test-time adaptation
* **Family:** adapting at deployment without labels.
* **What:** It updates a few parameters (e.g. normalization statistics and affine terms) on the unlabelled test batch, for example by
  minimizing prediction entropy (Tent). It handles gradual drift without new labels.
* **Data:** a trained model plus unlabelled test batches.
* **Fields:** B26, B12, B20.
* **+/−:** + cheap drift handling. − entropy minimization is for classification. For regression, re-estimating normalization
  statistics from reference traces is the transferable idea.
* **Py:** research code.
* **Refs:** [Wang 2020 Tent].

### P08. Meta-learning and few-shot learning (MAML, prototypical networks; NPs and PACOH in prior review §3)
* **Family:** learning to learn across tasks.
* **What:** MAML learns an initialization from which a few gradient steps fit a new task. Prototypical networks classify by
  distance to class-mean embeddings. Both need many training *tasks*.
* **Data:** many related small tasks.
* **Fields:** B26, B18 (few-shot molecular property prediction), B20, B14 (speaker adaptation).
* **+/−:** + designed for "a new unit with few points". − we have few tasks (amplifiers or boards).
* **Py:** `learn2learn` (moderate), `higher` (stable).
* **Refs:** [Finn 2017]; [Snell 2017]; [Parnami & Lee 2022].

### P09. Multi-task learning
* **Family:** shared representations across related targets.
* **What:** It trains one model to predict several targets (σe, λ, gain, trace noise, settings reconstruction) with a shared
  trunk. Auxiliary tasks act as regularizers.
* **Data:** small to large.
* **Fields:** B25, B26, B18, B12.
* **+/−:** + a natural use of our three targets plus cheap auxiliary targets. − negative transfer when tasks conflict, and the
  loss weighting matters.
* **Py:** `torch`; multi-output GPs (S13) are the Bayesian version.
* **Refs:** [Ruder 2017].

### P10. Active learning
* **Family:** choosing which examples to label.
* **What:** The model picks the next example to label (here: which settings to run as a full image packet) to maximize
  information. Examples are uncertainty sampling, expected model change, and BALD (maximize mutual information between the label
  and the model parameters). It is closely related to BO, but the goal is model accuracy, not an optimum.
* **Data:** a pool or stream of unlabelled candidates, and expensive labels.
* **Fields:** B18, B17, B12 (annotation), B7 (selecting wafers to measure in virtual metrology), B13.
* **+/−:** + the right framing for "which packets to take to improve Level 1 most". − it needs calibrated uncertainty, and
  batch selection needs diversity.
* **Py:** `modAL` (stable), `botorch` (`qNegIntegratedPosteriorVariance`; mature), `baal` (moderate).
* **Refs:** [Settles 2009]; [Houlsby 2011]; [Gal 2017].

### P11. Continual learning
* **Family:** learning from a stream without forgetting.
* **What:** Models updated on new data (new board, new cool-down) tend to forget old regimes. Elastic weight consolidation (EWC)
  penalizes changes to parameters important for old tasks. Replay buffers keep old examples.
* **Data:** sequential datasets.
* **Fields:** B20, B12, B21.
* **+/−:** + relevant for Level 3 over months. − for GPs and small models, simply refitting on all data is easier.
* **Py:** `avalanche` (moderate, active).
* **Refs:** [Kirkpatrick 2016].

### P12. Federated learning
* **Family:** distributed training without sharing raw data.
* **What:** Each site trains locally, and a server averages model updates (FedAvg).
* **Data:** data split across sites.
* **Fields:** B12 (hospitals), mobile keyboards, B23.
* **+/−:** + would let several CCD labs (e.g. DAMIC-M, SENSEI, Oscura test stands) pool models without moving data. − at our
  scale it is easier to share data.
* **Py:** `flower` (mature), `fedml` (moderate).
* **Refs:** [McMahan 2016].

### P13. Causal inference (double ML, DoWhy)
* **Family:** estimating effects of interventions.
* **What:** It separates the *causal effect* of a setting on noise from confounded correlations, e.g. when settings were changed
  together in historical scans. Double/debiased ML uses ML for nuisance functions and an orthogonalized estimator for the
  effect. DoWhy structures the analysis as model, identify, estimate, refute.
* **Data:** observational data, hundreds+ rows.
* **Fields:** B12, econometrics, B21 (A/B tests), B16.
* **+/−:** + relevant when learning from *historical, non-randomized* scans. − needs assumptions about confounding. When we can
  intervene directly (we can), randomized designs are better.
* **Py:** `dowhy` (mature), `econml` (mature), `doubleml` (mature).
* **Refs:** [Chernozhukov 2016]; [Sharma & Kiciman 2020].

### P14. Causal discovery (PC, NOTEARS)
* **Family:** learning causal graphs from data.
* **What:** It infers a directed acyclic graph among variables from conditional-independence tests (PC) or by continuous
  optimization with an acyclicity constraint (NOTEARS). It could suggest which settings drive which noise components.
* **Data:** hundreds to thousands of rows, ≲ 50 variables.
* **Fields:** biology, B16, B23 (root-cause analysis), B7.
* **+/−:** + a hypothesis generator. − unreliable at small n, and sensitive to scaling (NOTEARS) and to assumptions.
* **Py:** `causal-learn` (mature), `gcastle` (moderate), `cdt`.
* **Refs:** [Zheng 2018].

### P15. Anomaly and novelty detection (isolation forest, Deep SVDD, autoencoders)
* **Family:** unsupervised outlier detection.
* **What:** It scores how unusual an example is. Isolation forests count how few random splits isolate a point. One-class
  methods (OC-SVM, Deep SVDD) enclose normal data. Autoencoders use reconstruction error. In HEP, autoencoders monitor detector
  data quality online and catch problems that rule-based monitoring missed.
* **Data:** mostly normal examples, few or no labelled anomalies.
* **Fields:** B1 (CMS ECAL and muon data-quality monitoring, LHC Olympics anomaly searches), B23, B15 (fraud), B4 (glitch
  classification).
* **+/−:** + flags damaged trace sets, broken amplifiers, or out-of-distribution settings, which Level 3 safety needs.
  − threshold setting, and "anomalous" is not the same as "bad".
* **Py:** `scikit-learn` (`IsolationForest`; mature), `pyod` (mature), `alibi-detect` (mature).
* **Refs:** [Liu 2008]; [Ruff 2020]; [Harilal 2023]; [Pol 2018]; [Kasieczka 2021].

### P16. Distribution-shift and concept-drift detection
* **Family:** monitoring.
* **What:** Two-sample tests (MMD, KS on embeddings, classifier-based tests) check whether new data come from the training
  distribution. Concept-drift methods detect changes in $p(y\mid x)$ and adapt (windowing, reweighting). An empirical study
  ("Failing loudly") found that dimensionality reduction + two-sample testing works well in practice.
* **Data:** streams or batches.
* **Fields:** B15, B21, B23, B12, B7.
* **+/−:** + tells Level 3 when the CCD (or a board) has changed enough to recalibrate. − power is low with small batches.
* **Py:** `alibi-detect` (mature), `river` (mature), `frouros` (moderate).
* **Refs:** [Rabanser 2018]; [Gretton 2008]; [Lu 2020].

### P17. Foundation models for physics and science
* **Family:** large pretrained models (tabular and time-series FMs are in prior review §1–2).
* **What:** Models pretrained on many simulated physical systems (Multiple Physics Pretraining; "The Well" dataset), on
  multimodal astronomy data (AstroCLIP: images + spectra in one embedding space), on weather (ClimaX), or on wearable
  biosignals (PPG/ECG). Time-series FMs also include Chronos, Lag-Llama, and LLM reprogramming (Time-LLM).
* **Data:** pretrain huge, adapt with few labels.
* **Fields:** B3, B16, B12, fluid dynamics.
* **+/−:** + shows what a "detector-trace foundation model" could look like if several CCD experiments pooled traces. − none is
  pretrained on detector readout noise, and domain mismatch is large.
* **Py:** model-specific repositories, Hugging Face hub.
* **Refs:** [McCabe 2023]; [Ohana 2024]; [Parker 2023]; [Nguyen 2023 ClimaX]; [Abbaspourazad 2023]; [Ansari 2024]; [Rasul 2023]; [Jin 2023].

### P18. Early prediction and learning-curve extrapolation
* **Family:** predicting a long or expensive outcome from a short early measurement.
* **What:** In battery research, cycle life (~hundreds to thousands of cycles) is predicted from features of the *first 100
  cycles* (Severson 2019). The early predictor is then used inside Bayesian optimization of charging protocols, which cut the
  search from >500 days to 16 days (Attia 2020). In AutoML, the final accuracy of a training run is extrapolated from its first
  epochs (Domhan 2015; LC-PFN with prior-fitted networks), and freeze-thaw BO decides which runs to continue.
* **Data:** tens to hundreds of complete examples, many partial ones.
* **Fields:** B11, B29, B12 (early-warning scores).
* **+/−:** + **the closest structural analogue to Level 1 → Level 2**: short trace ≈ early cycles, image noise ≈ cycle life,
  and early-prediction-inside-BO is exactly the proposed trace-screening loop. − early signals may miss slow effects (in our
  case, slow noise and CIC that the 0.2 s trace does not see).
* **Py:** problem-specific; `botorch` for the BO part; LC-PFN code (research).
* **Refs:** [Severson 2019]; [Attia 2020]; [Domhan 2015]; [Adriaensen 2023]; [Swersky 2014]; [Wistuba & Pedapati 2020].

### P19. Neuro-symbolic learning and program synthesis
* **Family:** combining neural learning with symbolic structure.
* **What:** It mixes neural perception with symbolic reasoning or program search. DreamCoder learns a library of reusable program
  components while solving tasks. The neurosymbolic "third wave" view argues for knowledge representation inside learning.
  LLM-driven program search (FunSearch, O21) is a recent variant.
* **Data:** small to medium, structured tasks.
* **Fields:** program synthesis, physics law discovery, B17.
* **+/−:** + interpretable, compositional models (e.g. a readable rule for choosing CDS windows). − immature tooling, narrow
  successes.
* **Py:** research code.
* **Refs:** [Garcez & Lamb 2020]; [Ellis 2020].

### P20. AutoML and hyperparameter optimization (auto-sklearn, AutoGluon, Optuna, Hyperband)
* **Family:** automated model selection and tuning.
* **What:** It searches over pipelines and hyperparameters with BO, bandit-based early stopping (Hyperband), meta-learning of
  good starting configurations, and stacking (AutoGluon).
* **Data:** any. At small n the danger is overfitting the validation folds.
* **Fields:** all of applied ML.
* **+/−:** + a fair, automated baseline sweep. − at 13–100 packets, nested grouped CV is mandatory, and the AutoML result may
  just be noise.
* **Py:** `autogluon` (mature), `auto-sklearn` (maintenance), `optuna` (mature), `ray[tune]` (mature).
* **Refs:** [Feurer 2020]; [Erickson 2020]; [Akiba 2019]; [Li 2016 Hyperband].

---

## A7. Sequential decisions, optimization and control

### O01. Bayesian optimization (core: GP + acquisition function)
* **Family:** sample-efficient black-box optimization.
* **What:** A probabilistic surrogate (usually a GP) models the objective. An acquisition function (expected improvement, UCB,
  knowledge gradient, entropy search) picks the next point by trading off exploration and exploitation. GP-UCB has regret
  guarantees.
* **Data:** 10–10³ expensive evaluations, ≲ 20 dimensions classically.
* **Fields:** B29 (hyperparameters), B5, B6, B7, B8, B17, B20, B11, B28 (AlphaGo tuning).
* **+/−:** + the standard for expensive experiments. − struggles with high dimension, non-stationarity and safety unless extended
  (O02–O09).
* **Py:** `botorch`/`ax` (mature), `scikit-optimize` (maintenance), `xopt` (accelerators; active).
* **Refs:** [Snoek 2012]; [Frazier 2018]; [Srinivas 2009]; [Chen 2018 AlphaGo].

### O02. Trust-region and high-dimensional BO (TuRBO, BAxUS; SAASBO in prior review §7.4)
* **Family:** BO for many parameters.
* **What:** TuRBO runs local GP models inside trust regions that grow or shrink with success, which is robust in 10–100+
  dimensions. BAxUS optimizes in nested random subspaces that expand as evaluations accumulate.
* **Data:** hundreds to thousands of evaluations, 20–1000 dimensions.
* **Fields:** B5, B8, B17, robotics, aerospace.
* **+/−:** + fits 61 settings, and local steps match hardware step-size limits. − TuRBO needs more evaluations than we can
  afford in images, and BAxUS assumes a low effective dimension.
* **Py:** `botorch` tutorials (mature), BAxUS research code.
* **Refs:** [Eriksson 2019]; [Papenmeier 2023].

### O03. Multi-objective BO (qNEHVI, ParEGO)
* **Family:** Pareto-front BO.
* **What:** It optimizes several objectives at once (e.g. σe, λ, readout time) and returns the Pareto front instead of one
  weighted optimum. qNEHVI maximizes expected hypervolume improvement with noisy observations and batch proposals.
* **Data:** like BO.
* **Fields:** B17, B8, B18, B7, B11.
* **+/−:** + avoids choosing the noise/dark-current weight in advance. − the cost grows with the number of objectives.
* **Py:** `botorch` (mature), `pymoo` (evolutionary; mature).
* **Refs:** [Daulton 2021].

### O04. Safe and constrained BO (prior review §7.3)
* SafeOpt, LineBO, step-limited safe BO, SCBO. Added here: safe controller tuning on quadrotors, the origin of
  multi-constraint SafeOpt in robotics.
* **Refs:** [Berkenkamp 2015 quadrotor]; see prior review.

### O05. Transfer, meta and prior-informed BO (RGPE, πBO, HEBO)
* **Family:** BO that uses previous studies or expert beliefs.
* **What:** RGPE ranks and weights GPs fitted to past tasks to warm-start a new task. πBO multiplies the acquisition by a
  user-supplied belief about where the optimum is, with a weight that decays over time. HEBO adds input and output warping
  (heteroscedastic, non-stationary objectives) and multi-acquisition ensembles. It is reported to have won a NeurIPS 2020 BO challenge
  [unverified].
* **Data:** prior tasks or expert knowledge, few new evaluations.
* **Fields:** B29, B5 (archived-scan priors), B7, B17.
* **+/−:** + expert CCD knowledge (e.g. "the optimal delayInt is near X") enters formally, and old boards warm-start new ones.
  − misleading priors slow convergence (πBO limits the damage).
* **Py:** `botorch` (custom), `HEBO` (moderate), `SMAC3` (mature, includes πBO-style priors [unverified]).
* **Refs:** [Feurer 2018]; [Hvarfner 2022]; [Cowen-Rivers 2020].

### O06. Time-varying and drift-aware BO
* **Family:** BO for non-stationary objectives.
* **What:** It models the objective as varying in time, with a kernel over (x, t) and a forgetting factor. Old observations
  count less, and the optimizer keeps tracking a moving optimum.
* **Data:** a stream of evaluations.
* **Fields:** B5 (drifting machines), B20, B21 (changing user preferences), B10.
* **+/−:** + CCD state drifts (temperature, cool-down history). Level 3 needs this. − the drift rate must be learned or
  assumed.
* **Py:** `botorch` (custom kernel), research code.
* **Refs:** [Bogunovic 2016].

### O07. Causal Bayesian optimization
* **Family:** BO with a causal graph.
* **What:** It uses a known causal graph among variables to decide *which subset of variables to intervene on*, and pools
  observational and interventional data.
* **Data:** a causal graph plus few interventions.
* **Fields:** B12 (proposed), B16, B7.
* **+/−:** + if we know which settings act through which intermediate quantities (e.g. clock voltage → transfer efficiency →
  noise), this cuts the search space. − needs the graph.
* **Py:** research code.
* **Refs:** [Aglietti 2020].

### O08. Human-in-the-loop and preferential BO
* **Family:** BO with human feedback or collaboration.
* **What:** Preferential BO learns from pairwise comparisons ("image A looks better than B") instead of numeric scores. In
  semiconductor plasma-etch development, a controlled study found "human first, computer last" halved the cost-to-target
  compared with human experts alone. Humans were better early, the algorithm better near tight tolerances.
* **Data:** few evaluations, expert in the loop.
* **Fields:** B7, B26 (image tuning), B29, B5 (operator-assisted tuning).
* **+/−:** + matches how CCD experts work. Expert-chosen early scans followed by BO is the published best practice in a closely
  analogous industry. − human time is expensive, and preferences are noisy.
* **Py:** `botorch` (`PairwiseGP`; mature).
* **Refs:** [González 2017]; [Kanarik 2023].

### O09. Multi-fidelity and early-stopping optimization (freeze-thaw BO, Hyperband)
* **Family:** optimization with cheap partial evaluations.
* **What:** It allocates budget across many configurations at low fidelity and promotes only the promising ones (successive
  halving / Hyperband). Freeze-thaw BO models partial learning curves to decide which runs to pause or resume. The BO versions
  for physical experiments are in the prior review §7.2 (multi-fidelity KG, function networks).
* **Data:** cheap partial evaluations plus a few full ones.
* **Fields:** B29, B11, B5.
* **+/−:** + "trace-only packets as a cheap fidelity" in algorithmic form. − needs the low-fidelity ranking to correlate with
  the high-fidelity one.
* **Py:** `optuna` (Hyperband pruner), `ray[tune]`, `botorch` (mature).
* **Refs:** [Li 2016 Hyperband]; [Swersky 2014].

### O10. Batch BO
* **Family:** parallel proposals.
* **What:** It proposes several points per round, e.g. one per amplifier or per packet slot, with diversity. In analog circuit
  sizing, an ensemble of acquisition functions treated as a multi-objective problem picks the batch from their Pareto front.
* **Data:** like BO, with parallel evaluation capacity.
* **Fields:** B8, B17, B29.
* **+/−:** + fits a packet structure if one packet can test several settings. − less sample-efficient per evaluation.
* **Py:** `botorch` (q-acquisitions; mature).
* **Refs:** [Lyu 2018].

### O11. Bandits and Thompson sampling (incl. contextual bandits)
* **Family:** sequential decision with exploration.
* **What:** Choose among arms (configurations) to maximize cumulative reward. Thompson sampling draws a plausible model from the
  posterior and acts greedily on it. Contextual bandits (LinUCB) condition on side information (here, for example, today's CCD
  temperature).
* **Data:** streams of decisions.
* **Fields:** B21 (news recommendation), B12 (adaptive trials), B15, B29.
* **+/−:** + simple and robust, good for choosing among a *discrete* set of candidate settings during operation. − less
  sample-efficient than BO for continuous spaces.
* **Py:** `mabwiser` (moderate), `vowpalwabbit` (mature), `botorch` (Thompson sampling).
* **Refs:** [Russo 2017]; [Li 2010].

### O12. Evolution strategies (CMA-ES)
* **Family:** derivative-free stochastic search.
* **What:** It samples candidates from a multivariate Gaussian and adapts its mean and covariance toward successful samples. It
  is robust, invariant to monotone transformations of the objective, and scales to ~100 dimensions.
* **Data:** hundreds to thousands of evaluations.
* **Fields:** B20 (gait learning), B5, B6, B16 (model tuning), B28.
* **+/−:** + robust to noise and to bad conditioning, simple. − needs many more evaluations than BO, so it is better suited to
  optimizing a cheap *surrogate* or the acquisition function.
* **Py:** `cma` (mature), `evosax` (JAX; moderate), `nevergrad` (mature).
* **Refs:** [Hansen 2016].

### O13. Genetic and multi-objective evolutionary algorithms (NSGA-II)
* **Family:** population-based search.
* **What:** It evolves a population through selection, crossover and mutation. NSGA-II sorts by Pareto dominance and crowding
  distance to approximate the whole Pareto front.
* **Data:** thousands of evaluations.
* **Fields:** B8, B7, aerospace, B17.
* **+/−:** + handles discrete and mixed variables and many objectives. − evaluation-hungry, so use it on surrogates.
* **Py:** `pymoo` (mature), `DEAP` (mature).
* **Refs:** [Deb 2002].

### O14. Extremum seeking and model-free feedback tuning
* **Family:** control-theoretic online optimization.
* **What:** It dithers parameters with small sinusoids, demodulates the measured objective to estimate the gradient, and moves
  uphill continuously. It is robust to noise and *tracks time-varying optima*. It is used at LANL, CERN (AWAKE), LCLS and
  European XFEL for continuous tuning. "Safe" variants keep losses below a threshold.
* **Data:** a continuous stream of cheap measurements.
* **Fields:** B5, B20, power electronics, B6.
* **+/−:** + continuous drift compensation with minimal modelling, a candidate for Level 3 *maintenance* around an optimum found
  by BO. − slow for many parameters, and each step needs a measurement (an image packet is too slow, a trace might not be).
* **Py:** no standard package (a few dozen lines).
* **Refs:** [Scheinker 2020 ACM4PA].

### O15. Model-free reinforcement learning (DQN, PPO, SAC)
* **Family:** learning policies from reward.
* **What:** An agent learns a policy (state → action) by trial and error to maximize cumulative reward. It suits sequential
  decisions where actions change the state. It is sample-hungry, so in accelerators the policy is usually trained on a
  simulator or with careful sample-efficient methods, then deployed.
* **Data:** 10⁴–10⁸ interactions (often simulated).
* **Fields:** B28 (Atari), B20, B5 (CERN AWAKE/LINAC4 steering), B8 (AutoCkt), B6 (quantum control), B3 (adaptive optics).
* **+/−:** + learned policies act instantly at run time. − far too sample-hungry for real CCD packets. Only plausible with a
  good simulator of the CCD response, which we lack.
* **Py:** `stable-baselines3` (mature), `gymnasium` (mature), `cleanrl` (moderate).
* **Refs:** [Mnih 2013]; [Schulman 2017]; [Haarnoja 2018]; [Kain 2020]; [Settaluri 2020]; [Niu 2018].

### O16. Model-based and safe reinforcement learning (PILCO, PETS, safe MBRL)
* **Family:** RL with a learned dynamics model.
* **What:** It learns a probabilistic model of the system (a GP in PILCO, an ensemble of NNs in PETS) and plans or optimizes the
  policy using the model, propagating its uncertainty. Safe MBRL adds stability guarantees via Lyapunov functions. It is much
  more data-efficient than model-free RL.
* **Data:** tens to thousands of real interactions.
* **Fields:** B20, B3 (model-based RL for adaptive optics), B5.
* **+/−:** + feasible with tens of real episodes, and the uncertainty is explicit. − our "state" dynamics (CCD condition) are
  slow and poorly observed.
* **Py:** `mbrl-lib` (moderate), research code; `gpytorch` for GP models.
* **Refs:** [Deisenroth & Rasmussen 2011]; [Chua 2018]; [Berkenkamp 2017]; [Nousiainen 2021].

### O17. Offline RL and sequence-model RL (CQL, Decision Transformer)
* **Family:** learning policies from logged data only.
* **What:** It learns a policy from a fixed dataset of past decisions without new interaction. Conservative Q-learning penalizes
  values of actions not seen in the data. Decision Transformer treats RL as sequence prediction conditioned on the desired
  return.
* **Data:** large logs of past operations.
* **Fields:** B12 (treatment policies), B20, B21.
* **+/−:** + could learn from years of logged expert tuning sessions, if they existed in structured form. − needs much logged data
  with good coverage.
* **Py:** `d3rlpy` (mature).
* **Refs:** [Levine 2020]; [Kumar 2020]; [Chen 2021 DT].

### O18. Learning-based model predictive control (MPC)
* **Family:** control with learned models.
* **What:** At each step it optimizes a sequence of future actions using a model (here learned, e.g. a GP), subject to
  constraints, applies the first action, and repeats. Learning-based MPC focuses on improving the model from data while keeping
  safety constraints.
* **Data:** streaming system data.
* **Fields:** B20, process control (B7), B16 (building climate), automotive.
* **+/−:** + explicit hard constraints and multi-step planning (e.g. ramp voltages safely over several steps). − needs a dynamic
  model of the system, while our system is mostly static per setting.
* **Py:** `do-mpc` (mature), `acados` (mature), `casadi` (mature).
* **Refs:** [Hewing 2020].

### O19. Run-to-run (R2R) control with EWMA
* **Family:** statistical process control + feedback.
* **What:** In semiconductor fabs, after each run (wafer or lot) the process recipe is nudged using an exponentially weighted
  moving average of the observed offset from target. It compensates for slow drift. Deadbands and shift detection avoid
  chasing noise.
* **Data:** a stream of runs with metrology.
* **Fields:** B7.
* **+/−:** + a very simple, well-understood drift compensator for Level 3, e.g. re-centre the output-gate voltage after each
  packet. − single-loop and model-light, so it needs an approximately linear local response.
* **Py:** trivial to implement.
* **Refs:** [Sachs 1995] (via secondary citations; full text not read).

### O20. Bayesian experimental design (BED)
* **Family:** choosing experiments to maximize information.
* **What:** It chooses the experiment design ξ that maximizes the expected information gain about parameters θ,
  $\mathrm{EIG}(\xi) = \mathbb E[\log p(\theta\mid y,\xi) - \log p(\theta)]$. Modern versions are amortized with neural networks
  and run sequentially.
* **Data:** a model of the experiment plus few expensive runs.
* **Fields:** B12 (adaptive trials), B17, psychology, B10.
* **+/−:** + the formal answer to "which settings should the next packet test to learn the noise model best", as opposed to "to
  find the optimum". − EIG estimation is expensive, and it needs a likelihood or a simulator.
* **Py:** `pyro` (OED module; moderate), `botorch` (information-based acquisitions).
* **Refs:** [Rainforth 2023].

### O21. LLMs as optimizers and LLM-enhanced BO (OPRO, LLAMBO, FunSearch)
* **Family:** language-model-driven optimization.
* **What:** An LLM is prompted with past (configuration, score) pairs and asked to propose better configurations (OPRO). LLAMBO
  uses LLMs to warm-start, propose candidates and act as a surrogate inside BO. FunSearch evolves *programs*, with an LLM
  proposing code edits and an automatic evaluator scoring them. It found new constructions in combinatorics.
* **Data:** few evaluations. The prior knowledge sits in the LLM.
* **Fields:** B29, maths, B18, B17.
* **+/−:** + can inject textual domain knowledge (manuals, logbooks) and propose interpretable heuristics. − poor numerical
  calibration, unclear uncertainty, and results are hard to reproduce. A verifier or evaluator must be in the loop.
* **Py:** research code; `funsearch` (reference implementation).
* **Refs:** [Yang 2023 OPRO]; [Liu 2024 LLAMBO]; [Romera-Paredes 2024].

### O22. LLM agents for laboratories and facilities
* **Family:** tool-using language-model agents.
* **What:** LLMs that call tools (search, code execution, instrument APIs) to plan and run experiments. Examples: Coscientist
  (GPT-4-driven chemistry experiments on liquid-handling robots), ChemCrow (chemistry tools), agents that calibrate
  superconducting qubits on real hardware ("k-agents"), natural-language accelerator tuning, and "AI Scientist" pipelines that
  write papers end to end.
* **Data:** documentation, tools, few experiments.
* **Fields:** B18, B6, B5, B17.
* **+/−:** + a possible Level-3 operator interface ("why did noise rise? check the logbook, re-run the reference trace").
  − reliability, safety and reproducibility. They should act through a restricted, validated tool set (our `ccd_limits.py`
  pattern).
* **Py:** agent frameworks (fast-moving).
* **Refs:** [Boiko 2023]; [Bran 2023]; [Cao 2024]; [Kaiser 2024 LLM]; [Lu 2024].

### O23. Self-driving laboratories (closed-loop autonomous experimentation)
* **Family:** integrated ML + robotics + experiment.
* **What:** A loop of propose (BO or active learning), execute (robot or instrument), measure (automatic analysis), and update.
  Examples are the "Ada" thin-film lab, PHOENICS-driven chemistry optimization, and closed-loop battery charging optimization.
* **Data:** tens to hundreds of autonomous experiments.
* **Fields:** B17, B18, B11, B6.
* **+/−:** + **the template for Level 3**: lessons on automation of analysis, failure handling and human oversight transfer
  directly. − hardware integration dominates the effort.
* **Py:** `botorch`, `atlas`/`olympus` (Aspuru-Guzik group; moderate) [unverified names], lab-specific code.
* **Refs:** [MacLeod 2019]; [Häse 2018]; [Attia 2020].

### O24. Planning with learned models (AlphaZero, MuZero)
* **Family:** search + learned value and policy.
* **What:** Monte-Carlo tree search guided by a learned value and policy network, with self-play training. MuZero also learns the
  dynamics model, so it plans without knowing the rules.
* **Data:** millions of simulated games.
* **Fields:** B28, chip floor-planning, B17.
* **+/−:** + planning over *sequences* of setting changes under step limits is formally a planning problem. − needs a cheap
  simulator, which we do not have.
* **Py:** `mctx` (JAX; moderate), `open_spiel` (mature).
* **Refs:** [Silver 2017]; [Schrittwieser 2019].

---

## A8. Uncertainty quantification, evaluation and interpretability

### U01. Deep ensembles (prior review §5.1; used in repo)
* Train M networks from different initializations and use the spread as epistemic uncertainty. A strong practical baseline.
  **Refs:** [Lakshminarayanan 2016].

### U02. Bayesian neural networks (Bayes by Backprop, MC dropout; Laplace and VBLL in prior review)
* **Family:** approximate Bayesian deep learning.
* **What:** It places distributions on network weights. Bayes by Backprop learns a Gaussian per weight variationally. MC
  dropout keeps dropout on at test time and averages, which is interpretable as approximate variational inference.
* **Data:** medium.
* **Fields:** B12, B20, B3.
* **+/−:** + cheap uncertainty (MC dropout). − often miscalibrated, so post-hoc or conformal calibration is still needed.
* **Py:** `torch`, `bayesian-torch` (moderate), `laplace-torch` (moderate).
* **Refs:** [Blundell 2015]; [Gal & Ghahramani 2015].

### U03. Calibration after training, and proper scoring rules
* **Family:** evaluation and recalibration.
* **What:** Modern networks are often over-confident. Temperature scaling (a single scalar fitted on validation data) fixes much
  of this for classifiers, and the regression analogue rescales the predicted σ. Proper scoring rules (log score, CRPS, interval
  score) reward honest predictive distributions. CRPS generalizes absolute error to distributions and is the natural single
  number for comparing our probabilistic models.
* **Data:** a validation set (grouped by packet).
* **Fields:** B16 (forecast verification), B12, B26.
* **+/−:** + cheap, and CRPS would complement our coverage + width metrics. − global rescaling cannot fix shape errors.
* **Py:** `properscoring` (stable), `scoringrules` (active), `scikit-learn` (calibration).
* **Refs:** [Guo 2017]; [Gneiting & Raftery 2007].

### U04. Conformal prediction, incl. adaptive conformal under drift (hierarchical variants in prior review §5.2)
* **Family:** distribution-free predictive intervals.
* **What:** Any model's errors on calibration data give intervals with guaranteed marginal coverage under exchangeability.
  *Adaptive conformal inference* updates the miscoverage level online, $\alpha_{t+1} = \alpha_t + \gamma(\alpha - \mathrm{err}_t)$,
  so coverage holds in the long run even under arbitrary distribution shift. That is relevant for Level 3 operation over drifting
  conditions.
* **Data:** a calibration set or stream.
* **Fields:** B15, B12, B16, B18.
* **+/−:** + coverage without model assumptions. − marginal, not conditional, coverage.
* **Py:** `mapie` (mature), `crepes` (mature), `puncc` (moderate).
* **Refs:** [Angelopoulos & Bates 2021]; [Gibbs & Candès 2021].

### U05. Simulation-based calibration (SBC) of Bayesian inference
* **Family:** checking Bayesian inference algorithms.
* **What:** It draws θ from the prior, simulates data, and runs the inference. The rank of the true θ among the posterior draws
  must be uniform. Deviations show biased or over/under-dispersed posteriors. Applies to MCMC, VI, NPE and our WADERs-style
  fits.
* **Data:** a simulator plus repeated inference runs.
* **Fields:** Bayesian statistics, B4, B3.
* **+/−:** + checks whether *label* uncertainties (WADERs fit errors) are themselves calibrated, using synthetic images.
  − needs a simulator and many fits.
* **Py:** `sbi` (diagnostics), `arviz` (rank plots), `simuk` [unverified].
* **Refs:** [Talts 2018].

### U06. SHAP and feature attribution
* **Family:** post-hoc explanation.
* **What:** It attributes a prediction to its inputs with Shapley values from cooperative game theory. TreeSHAP is exact and fast
  for trees. KernelSHAP is model-agnostic.
* **Data:** any model.
* **Fields:** B12, B15, B7, B1.
* **+/−:** + shows which settings and trace features drive σe predictions. − attributions of correlated inputs are ambiguous, and
  they explain the model, not the physics.
* **Py:** `shap` (mature).
* **Refs:** [Lundberg & Lee 2017].

### U07. Interpretable-by-design models
* **Family:** sparse, additive and rule-based models.
* **What:** The argument (Rudin) is that for high-stakes decisions one should use models that are interpretable by construction
  (sparse linear, GAM/EBM, scoring systems, short rule lists) instead of explaining black boxes, because there is often little
  accuracy cost.
* **Fields:** B12, B15, criminal justice, B7.
* **+/−:** + Level-2/3 decisions about hardware settings benefit from readable models. − possible accuracy cost on raw
  signals.
* **Py:** `interpret`, `imodels` (moderate), `pysr`.
* **Refs:** [Rudin 2018]; [Nori 2019].

### U08. Use-case-matched (grouped) validation
* **Family:** evaluation methodology.
* **What:** In clinical ML, *record-wise* cross-validation (records of the same subject in train and test) massively
  overestimates accuracy compared with *subject-wise* CV, which matches the real use case of a new patient. This is the same issue
  as our packet-grouped folds, and the medical literature is the best-documented source of evidence and arguments.
* **Fields:** B12, B23, B13 (station-wise splits), B16 (spatial blocking).
* **+/−:** + justifies packet- (and board-) grouped evaluation. − grouped CV has high variance with few groups.
* **Py:** `scikit-learn` (`GroupKFold`, `LeaveOneGroupOut`; mature).
* **Refs:** [Saeb 2017].

---

# Part B: application fields

Each field lists its **problem shape** (stated abstractly, so that it can be matched to ours), its **signature methods** (entry
codes from Part A), **notable examples** (with references), and **a transferable idea** (one line, as a prompt for the analysts,
not a recommendation).

## B-I. Fields close to ours

### B1. Particle and nuclear physics detectors
* **Problem shape:** huge simulated + real datasets. Event-level classification and regression from hits, pulses or point clouds.
  Online data-quality monitoring of thousands of channels. Searches for anomalies without labels.
* **Signature methods:** T02 (BDTs, historically dominant), D06 (GNN tracking, ParticleNet), D07 (sets / point clouds), G02–G04
  (fast simulation, unfolding), P15 (autoencoder data-quality monitoring, anomaly searches), G07 (SBI), S07 (GP background fits).
* **Examples:** GNN tracking for HL-LHC [Ju 2021]. ParticleNet [Qu & Gouskos 2019]. CaloGAN [Paganini 2017]. Interpretable BDTs
  for pulse-shape discrimination in the Majorana Demonstrator (germanium detectors) [Arnquist 2022]. Autoencoder online
  data-quality monitoring of the CMS ECAL [Harilal 2023] and muon chambers [Pol 2018]. LHC Olympics anomaly challenge
  [Kasieczka 2021]. GPs for smooth background modelling [Frate 2017]. A living review indexes the field [Feickert & Nachman 2021].
  Skipper-CCD experiments use classical likelihood clustering and cuts, and only early ML event classification (CONNIE,
  conference work) was found (context: [Cababie 2023]).
* **Transferable idea:** channel-level unsupervised data-quality monitoring (P15) maps directly onto "is this amplifier's trace
  set healthy?".

### B2. CCD / image-sensor characterization (incl. camera noise estimation)
* **Problem shape:** estimate read noise, gain, dark current and other sensor parameters from images or raw samples, often with
  sub-electron noise. Small number of devices. Strong physics.
* **Signature methods:** classical maximum likelihood / EM (photon-counting histograms), S07/S14 (physics fits), G08 (physics-based
  noise synthesis), O01 (BO of operating points).
* **Examples:** PCH-EM, a maximum-likelihood (EM) replacement for the photon-transfer method for deep sub-electron read-noise
  pixels, gives lower-variance read-noise and gain estimates from the same data [Hendrickson 2024; Hendrickson 2025].
  It is the CMOS/QIS analogue of our WADERs labels. Single-image Poisson–Gaussian noise parameter estimation in raw camera data
  [Foi 2008]. Physics-based CCD noise synthesis for astronomical denoising [Liu 2026 CCD noise]. BO for SiSeRO/skipper
  characterization and skipper-CCD noise work: see prior review §8.
* **Transferable idea:** label-side improvements (EM/likelihood fits of photon-counting histograms with calibrated errors) are
  as valuable as model-side ones. The computer-vision noise-estimation literature already solves "noise parameters from
  data".

### B3. Astronomy instrumentation and data
* **Problem shape:** few instruments, much data. Correlated noise in time series. Real-time control (adaptive optics).
  Artefact removal.
* **Signature methods:** S08 (celerite GPs for light curves), S09, D02 (deepCR cosmic-ray masks), O16 (model-based RL for adaptive
  optics), P17 (AstroCLIP), G07 (SBI).
* **Examples:** celerite fast GPs for stellar variability [Foreman-Mackey 2017]. deepCR cosmic-ray rejection on CCD images
  [Zhang & Bloom 2019]. Model-based RL for adaptive-optics control [Nousiainen 2021]. AstroCLIP cross-modal embeddings
  [Parker 2023].
* **Transferable idea:** in astronomy, GPs with physically shaped PSD kernels (S08) are the standard noise model for 1-D
  series. Our traces are the same kind of object.

### B4. Gravitational-wave detectors (noise characterization)
* **Problem shape:** instruments dominated by non-stationary noise. Many auxiliary sensor channels. Glitches. Real-time
  parameter estimation.
* **Signature methods:** D02 (CNN noise regression from auxiliary channels), P15 + citizen-science labels (glitch
  classification), G07 (NPE: DINGO).
* **Examples:** DeepClean subtracts noise linearly and non-linearly coupled from auxiliary witness channels [Ormiston 2020].
  Gravity Spy combines ML and citizen science to classify glitches [Zevin 2016]. DINGO gives real-time posteriors with NPE
  [Dax 2021].
* **Transferable idea:** "witness channels" (environment, clock-line monitors, other amplifiers) used to *regress out* pickup
  noise, i.e. DeepClean for CCD video.

### B5. Accelerator tuning
* **Problem shape:** many knobs (10–100), expensive and noisy evaluations, safety constraints, drift, human operators, archives of
  past tuning.
* **Signature methods:** O01–O04 (BO incl. safe and trust-region), O14 (extremum seeking), O15/O16 (RL, often sim-trained), S13
  (multi-fidelity), O22 (LLM operators), D09 (physics-informed priors).
* **Examples:** sample-efficient RL at CERN AWAKE and LINAC4 [Kain 2020]. RL-trained optimizers vs BO in a comparative study
  [Kaiser 2024 Sci. Rep.]. Extremum seeking and adaptive ML [Scheinker 2020 ACM4PA]. LLM-driven tuning through natural language
  [Kaiser 2024 LLM]. Overview [Edelen 2018]. More in prior review §7–8.
* **Transferable idea:** "RL-trained optimizer" (an optimizer policy trained on a simulator, then deployed) versus BO. Accelerator
  physicists have published the trade-off study we would otherwise have to run.

### B6. Quantum device tuning (quantum dots, superconducting qubits)
* **Problem shape:** a device with many gate voltages, a narrow "good" region, expensive measurements, device-to-device
  variation, and noise spectra that must be characterized.
* **Signature methods:** D02 (CNN state recognition on charge-stability diagrams), O01 (BO), O15 (RL for gates), P01/G01
  (unsupervised embeddings for fine tuning), dependency graphs for calibration, neural noise spectroscopy, O22.
* **Examples:** CNN state recognition and auto-tuning of quantum dots [Kalantre 2017]. Fully automatic tuning faster than experts
  [Moon 2020]. Unsupervised embedding learning for fine-tuning [van Esbroeck 2020]. Review [Zwolak & Taylor 2021]. Qubit
  calibration organized as a directed acyclic graph of calibration steps ("Optimus") [Kelly 2018]. Deep-RL quantum control
  [Niu 2018]. **Neural networks that infer noise PSDs from qubit coherence measurements**, pretrained on synthetic data and
  fine-tuned [Martina 2023; Gupta 2025]. LLM agents for qubit calibration [Cao 2024].
* **Transferable ideas:** (i) neural noise spectroscopy is the closest published analogue of "infer noise parameters from a
  measured waveform with a network pretrained on synthetic noise". (ii) DAG-structured calibration suits our ordered clock,
  voltage and timing tuning.

### B7. Semiconductor manufacturing and process control
* **Problem shape:** expensive metrology, cheap in-situ sensor data (spectra, traces), drift between runs, tight safety and
  yield constraints, experts in the loop.
* **Signature methods:** S03/S07 (virtual metrology: PLS, GP regression), O19 (EWMA run-to-run), O08 (human–machine BO), D02
  (wafer-map CNN), P15/P16.
* **Examples:** "human first, computer last" BO for plasma-etch recipe development [Kanarik 2023]. CNN wafer-map defect
  classification [Nakazawa & Kulkarni 2018] (bibliographic details from citing papers). EWMA run-to-run control [Sachs 1995].
  GP-regression virtual metrology of plasma-etch rate (Lynn, Ringwood, MacGearailt; Maynooth theses and papers, abstract only
  [unverified details]).
* **Transferable idea:** **virtual metrology** is our Level 1 under another name: predict an expensive end measurement (image
  noise) from cheap in-process signals (trace). The fab literature on drift-windowed GPs and on choosing *which* wafer to measure
  (active learning) carries over.

### B8. Analog circuit design
* **Problem shape:** size transistors and bias currents (10–100 parameters) to meet specifications. Each evaluation is a SPICE
  simulation (cheap) or silicon (expensive). Multiple objectives and constraints.
* **Signature methods:** O10 (batch BO), O03, O15 (RL, AutoCkt), D06 (GCN-RL transferable across topologies), S17 (PCE for
  tolerance).
* **Examples:** batch BO with an acquisition ensemble [Lyu 2018]. Deep-RL sizing (AutoCkt) [Settaluri 2020]. GCN-RL
  transferable sizing [Wang 2020 GCN-RL].
* **Transferable idea:** our clock and bias voltages *are* analog design parameters of the CCD output stage. Simulation-trained
  policies that transfer across topologies suggest training on a SPICE model of the output amplifier, then fine-tuning on
  hardware.

### B9. Electronics and RF signal processing
* **Problem shape:** classify or characterize signals from raw I/Q or voltage samples. Strong noise. Hardware impairments.
* **Signature methods:** D02 (CNN/ResNet on raw I/Q), S25 (spectral estimation), S29 (reservoir equalizers), S09.
* **Examples:** over-the-air radio modulation classification with deep nets [O'Shea 2017].
* **Transferable idea:** RF ML learns *hardware fingerprints* (impairments) from raw samples. The same idea applies to
  identifying the amplifier, and its health, from the trace alone.

### B10. Metrology and timekeeping
* **Problem shape:** characterize and predict oscillator noise with Allan statistics. Clock ensembles. Drift.
* **Signature methods:** S20 (Kalman filters for clock ensembles), S22, neural networks on Allan/Hadamard deviation features, O20.
* **Examples:** neural networks for clock-ensemble weighting from Allan and Hadamard deviation inputs, and for clock-error
  prediction, were found only through abstracts in journals (MAPAN, IEEE TIM) [unverified details, abstract only].
* **Transferable idea:** our Allan-deviation features come from this field. Its Kalman-filter clock models (separate white-FM,
  flicker-FM and random-walk states) are a principled way to model noise components over time.

## B-II. Fields that seem unrelated

### B11. Batteries and electrochemistry
* **Problem shape:** the outcome of interest (cycle life) is extremely slow to measure. Early cheap measurements (first cycles) are
  predictive. Experiments are parallel but expensive. The design space is protocol parameters.
* **Signature methods:** S02 (elastic net on hand-crafted early features), P18 (early prediction), O01/O09 (BO with early
  prediction as cheap fidelity), O23.
* **Examples:** cycle-life prediction from the first 100 cycles, with 124 cells and an elastic net on discharge-curve features
  [Severson 2019] (abstract only). Closed-loop BO of fast-charging protocols with an early-prediction model: 224 protocols in 16
  days instead of >500 [Attia 2020] (abstract only).
* **Transferable idea:** **the strongest structural analogue found**. Few devices, hand-made features of a short early
  waveform, a simple regularized linear model, then used as a cheap fidelity inside BO.

### B12. Medicine and biosignals (ECG, EEG, PPG)
* **Problem shape:** many unlabelled waveforms, few labels, strong between-subject variation (data grouped by subject), drift
  across devices and hospitals, high stakes.
* **Signature methods:** D02 (CNNs on raw ECG), D15 (filterbank front-ends), P01–P02 (self-supervised biosignal foundation
  models), U08 (subject-wise CV), P06 (multi-site shift), P12, T05/U07.
* **Examples:** cardiologist-level arrhythmia detection with a 34-layer 1-D CNN [Rajpurkar 2017]. EEGNet, a compact CNN for small EEG
  datasets [Lawhern 2016]. Foundation models trained on wearable PPG/ECG from a very large cohort with self-supervision
  [Abbaspourazad 2023]. Record-wise vs subject-wise CV [Saeb 2017].
* **Transferable idea:** EEGNet-style *compact, depthwise-separable* CNNs were designed for exactly "few hundred labelled
  multichannel waveforms". Subject-wise CV maps directly to packet-wise CV.

### B13. Seismology
* **Problem shape:** continuous noisy waveforms from many stations, detection and phase picking, denoising, and a
  labelled-catalogue bias.
* **Signature methods:** D02/D04 (PhaseNet U-Net, EQTransformer attention), denoising networks, S25 (multitaper), P05 (transfer
  between regions), shared toolboxes.
* **Examples:** PhaseNet [Zhu & Beroza 2018]. EQTransformer [Mousavi 2020]. DeepDenoiser (signal/noise decomposition in the
  time-frequency domain) [Zhu 2018 denoise]. SeisBench, a unified toolbox and benchmark collection [Woollam 2021].
* **Transferable idea:** DeepDenoiser learns a time-frequency *mask separating signal from noise*. Applied to video traces, it
  could separate pickup and 1/f components from white noise as a learned decomposition.

### B14. Speech and audio
* **Problem shape:** raw waveforms at high rate, huge unlabelled corpora, few labelled hours for many languages, robust
  front-ends, and quality estimation without a reference.
* **Signature methods:** D15 (SincNet, scattering), P02 (wav2vec 2.0, HuBERT), D02/D05 (WaveNet, state-space models), D04 (AST),
  S21 (classical HMMs), non-intrusive quality metrics.
* **Examples:** wav2vec 2.0 [Baevski 2020]. HuBERT [Hsu 2021]. AST [Gong 2021]. SincNet [Ravanelli 2018]. CPC [van den Oord
  2018]. **DNSMOS**, a network that predicts human quality ratings of noisy or denoised speech *without a clean reference*
  [Reddy 2020].
* **Transferable idea:** non-intrusive quality estimation (predict a costly human/lab quality score from the raw signal alone) is
  the audio twin of predicting σe from a trace.

### B15. Finance and econometrics
* **Problem shape:** non-stationary noisy series, estimating *volatility (noise level)* at several time scales, regime changes,
  and calibrating a model's parameters to market observables.
* **Signature methods:** S22 (GARCH, HAR realized volatility), S21 (regime switching), S24, S05 (quantiles, value-at-risk), D03 (DeepAR),
  NN surrogates of pricing maps for fast calibration.
* **Examples:** GARCH [Bollerslev 1986]. HAR realized-volatility cascade over daily, weekly and monthly horizons [Corsi 2009]. "Deep
  learning volatility": train a network on the map (model parameters → implied-volatility surface), then calibrate parameters to
  data by inverting the fast surrogate [Horvath 2019].
* **Transferable ideas:** (i) HAR is an "Allan-deviation regression", forecasting a noise level from noise levels at several
  scales. (ii) The Horvath recipe (learn a forward surrogate, then invert it) is a template for Level 2: learn settings → noise
  curve, then invert for settings.

### B16. Climate, weather and hydrology
* **Problem shape:** expensive simulators with uncertain parameters, sparse observations, data assimilation, emulation, and
  learning across many sites with static attributes.
* **Signature methods:** S14 (history matching, calibration), S20 (EnKF), D06 (GraphCast), D09 (NeuralGCM hybrid), P17 (ClimaX),
  D03 (LSTMs across catchments).
* **Examples:** history matching of HadCM3 with emulators rules out about half of parameter space [Williamson 2013]. EnKF
  [Evensen 1994]. GraphCast [Lam 2022]. NeuralGCM [Kochkov 2023]. ClimaX [Nguyen 2023 ClimaX]. LSTM rainfall–runoff across 241
  catchments [Kratzert 2018], with catchment attributes as static inputs [Kratzert 2019].
* **Transferable idea:** **history matching**, which rules out implausible settings instead of searching for the best, is a natural
  formalism for the *safe region* of CCD settings.

### B17. Materials discovery and self-driving labs
* **Problem shape:** small expensive datasets, large design spaces, multi-objective, physics descriptors, autonomous robots.
* **Signature methods:** O01/O03/O23 (BO, self-driving labs), D06 (CGCNN), S26 (SISSO), P10.
* **Examples:** Ada self-driving thin-film lab [MacLeod 2019]. PHOENICS optimizer [Häse 2018]. CGCNN [Xie & Grossman 2017]. SISSO
  descriptors [Ouyang 2017].
* **Transferable idea:** SISSO-style *descriptor search* (build millions of candidate formulas from primary features and select
  a few by sparse regression) for a compact law of σe(settings).

### B18. Drug discovery and chemistry
* **Problem shape:** few assay labels, many unlabelled molecules, multi-fidelity assays (cheap computational screens vs
  expensive wet lab), active learning loops, agentic automation.
* **Signature methods:** D06 (message-passing networks such as Chemprop), P10, S13, P08 (few-shot), O22 (Coscientist, ChemCrow).
* **Examples:** Chemprop, directed message-passing networks for property prediction [Yang 2019]. Coscientist [Boiko 2023].
  ChemCrow [Bran 2023].
* **Transferable idea:** the "computational screen then wet lab" funnel is structurally our "trace-only packet then image
  packet" funnel.

### B19. Protein structure and engineering
* **Problem shape:** huge unlabelled sequence databases, tiny labelled fitness sets (tens of variants), expensive assays.
* **Signature methods:** P05 (pretrained protein LMs + small heads), D04/D06 (AlphaFold), P10/O01 (ML-guided directed evolution).
* **Examples:** AlphaFold2 [Jumper 2021]. ESM-2/ESMFold [Lin 2023]. "Low-N" engineering: a pretrained sequence model with as few as 24 measured
  variants guided successful design [Biswas 2021] (abstract only).
* **Transferable idea:** low-N success came from **unsupervised pretraining on the unlabelled domain + evolutionary fine-tuning
  on unlabelled *related* sequences + a tiny linear/ridge head** (details beyond the abstract from memory [unverified]). That is a recipe for our unlabelled traces.

### B20. Robotics and control
* **Problem shape:** expensive and possibly damaging real trials, simulators with a reality gap, safety constraints, continuous
  control.
* **Signature methods:** O01 (BO for gaits and controllers), O04 (safe BO on quadrotors), O16 (PILCO, PETS, safe MBRL), G08 (domain
  randomization), O18 (learning-based MPC).
* **Examples:** BO for bipedal gaits [Calandra 2016]. Safe controller tuning on quadrotors [Berkenkamp 2015 quadrotor]. PILCO
  [Deisenroth & Rasmussen 2011]. Domain randomization [Tobin 2017]. Learning-based MPC review [Hewing 2020].
* **Transferable idea:** "a few tens of real trials with a GP model" is routine in robot learning, and robotics has the most mature
  safety tooling.

### B21. Recommender systems
* **Problem shape:** a sparse user × item matrix, cold start for new users or items, explore/exploit trade-off, drift in
  preferences.
* **Signature methods:** matrix factorization, O11 (contextual bandits), neural collaborative filtering, P16.
* **Examples:** matrix factorization [Koren 2009]. LinUCB contextual bandits for news [Li 2010]. Neural collaborative filtering
  [He 2017].
* **Transferable idea:** treat *amplifiers × settings* (or boards × settings) as a sparse matrix and factorize it, sharing latent
  "device factors" across settings. This is a low-rank cousin of multi-task GPs, and it answers "cold start for a new amplifier".

### B22. Agriculture and remote sensing
* **Problem shape:** few ground-truth labels (yield, soil properties), rich spectral or remote-sensing inputs, spatial
  correlation.
* **Signature methods:** S03 (PLS on NIR spectra), D02/D03 + GP layer, S07, S15.
* **Examples:** a deep GP for crop yield from remote-sensing histograms. Images were converted to pixel-value *histograms* to cope
  with few labels, and a GP layer captured spatial correlation [You 2017].
* **Transferable idea:** summarizing a high-dimensional input as a *histogram of values* (permutation-invariant, low-dimensional)
  before learning, e.g. histograms of skip values or CDS differences per trace.

### B23. Manufacturing predictive maintenance
* **Problem shape:** vibration or sensor waveforms, rare failures, remaining-useful-life regression, operating-condition changes.
* **Signature methods:** D02/D03 (CNN, LSTM on run-to-failure data), F01, P15, S24, D15.
* **Examples:** C-MAPSS turbofan run-to-failure simulation benchmark [Saxena 2008]. Survey of deep learning for machine health
  monitoring [Zhao 2016].
* **Transferable idea:** operating-condition normalization (clustering data by operating regime and normalizing features within
  each regime) before learning resembles normalizing trace features within settings groups.

### B24. Ecology and bioacoustics
* **Problem shape:** presence-only or biased labels, spatial models, massive unlabelled audio recordings, hierarchical sampling.
* **Signature methods:** MaxEnt species distribution models, T01, S15, D02 (BirdNET spectrogram CNN), S04.
* **Examples:** MaxEnt [Phillips 2006]. BirdNET [Kahl 2021] (bibliographic details only).
* **Transferable idea:** presence-only learning, i.e. learning from "good examples only", resembles learning a model of
  *acceptable* operating regions from the history of settings that experts kept.

### B25. NLP and large language models
* **Problem shape:** pretraining on huge unlabelled corpora, adaptation with few labels, in-context learning.
* **Signature methods:** D04, P02 (BERT), in-context learning (GPT-3), P05 (LoRA), D13 (MoE).
* **Examples:** BERT [Devlin 2018]. GPT-3 few-shot in-context learning [Brown 2020]. LoRA [Hu 2021 LoRA].
* **Transferable idea:** in-context learning is the mechanism behind TabPFN (prior review §1). Prediction is "conditioning on a
  context set", which also suits "this packet's other amplifiers as context".

### B26. Computer vision (incl. camera noise)
* **Problem shape:** large labelled and unlabelled image sets, pretrained backbones, self-supervision, and sensor noise modelling
  for denoising.
* **Signature methods:** D02 (ResNet), D04 (ViT), P01–P03 (SimCLR, MAE, BYOL), multimodal CLIP, G08 (Noise2Noise, physics noise
  synthesis), U03 (calibration).
* **Examples:** ResNet [He 2015]. ViT [Dosovitskiy 2020]. CLIP [Radford 2021]. MAE [He 2021 MAE]. Noise2Noise [Lehtinen 2018].
  Raw-image Poisson–Gaussian noise fitting [Foi 2008]. Temperature scaling [Guo 2017].
* **Transferable idea:** Noise2Noise-style training on *repeated independent readouts of the same pixel* (our skips) can learn
  without clean targets, and it gives a self-supervised task tied to noise.

### B27. Sports analytics
* **Problem shape:** few observations per player or team, strong hierarchy (player within team within league), rating under
  uncertainty, changing form.
* **Signature methods:** S15 (hierarchical Bayesian models), Bayesian skill ratings, S01 (expected-goals logistic models), S20
  (dynamic ratings).
* **Examples:** hierarchical Bayesian football model [Baio & Blangiardo 2010]. TrueSkill, a Bayesian skill rating with explicit
  uncertainty and message passing [Herbrich 2006].
* **Transferable idea:** TrueSkill-style *rating with uncertainty that drifts over time* gives a per-amplifier "health rating" for
  Level 3.

### B28. Gaming and game AI
* **Problem shape:** perfect simulators, sequential decisions, planning, and tuning of very expensive training runs.
* **Signature methods:** O15 (DQN), O24 (AlphaZero, MuZero), O01 (BO tuned AlphaGo's hyperparameters).
* **Examples:** DQN on Atari [Mnih 2013]. AlphaZero [Silver 2017]. MuZero [Schrittwieser 2019]. **BO in AlphaGo**: GP-BO tuned
  the hyperparameters of the MCTS and value networks during development [Chen 2018 AlphaGo].
* **Transferable idea:** even the most compute-rich projects used plain GP-BO for *a few expensive knobs*. This supports BO over
  RL when evaluations are scarce.

### B29. ML engineering: hyperparameter optimization and AutoML (as an application field)
* **Problem shape:** expensive training runs, partial learning curves as cheap signals, many past tuning tasks to transfer from.
* **Signature methods:** O09 (Hyperband, freeze-thaw), P18 (learning-curve extrapolation), O05 (transfer BO), P20.
* **Examples:** Hyperband [Li 2016 Hyperband]. Freeze-thaw BO [Swersky 2014]. Learning-curve extrapolation [Domhan 2015;
  Adriaensen 2023]. RGPE transfer BO [Feurer 2018]. Learning to rank learning curves [Wistuba & Pedapati 2020].
* **Transferable idea:** the HPO community's tooling for "cheap partial signal → decide what to fully evaluate" is mature and
  open source, and could be borrowed almost unchanged.

---

# Part C: problem shapes

Abstract problem types, each with the entries (Part A) and fields (Part B) that address it. Our application exhibits every
shape marked ★.

| # | Problem shape | Entries that address it | Fields where it is the core problem |
|---|---|---|---|
| C1 ★ | **Few-shot regression with grouped / clustered data** (rows correlated within packets; few groups) | S15, S16, S07+S13, T04, P08, D17, U08, prior-review conformal (§5.2) | B12, B27, B22, B16 (catchments), B6 |
| C2 ★ | **p ≫ n: hundreds of inputs, tens–hundreds of rows** | S02, S03, S18, S26 (SISSO), S07 with sparse priors (SAAS), T04, T05, prior-review TabPFN | B11, B17, B22 (spectra), B7 (virtual metrology), genomics |
| C3 ★ | **Few labels + many unlabelled waveforms** | P01–P04, P05, G01, G08, F02, D15 (untrained scattering), S19 | B12, B14, B13, B19, B1 |
| C4 ★ | **Learning from waveforms where the information is in second-order statistics (noise level, PSD shape)** rather than in shape | S08 (celerite), S09 (spectral GPs), S22 (GARCH/HAR), S25 (multitaper), D15 (scattering), F01, G07, D08 (neural SDEs) | B15 (volatility), B10, B6 (noise spectroscopy), B4, B3 |
| C5 ★ | **Several repeated measurements (a set) per example** | D07, S19 (distribution regression), F03, P15 (flag bad set members), B22 histogram trick | B1, B3, B22 |
| C6 ★ | **Curves / functions as inputs (Allan curves, PSDs)** | S03 (PLS), S18 (FDA), D10 (neural operators), D11 (INR), F01 | chemometrics, B22, B12 (growth curves), B7 |
| C7 ★ | **Predict an expensive or slow outcome from a short, cheap early measurement; use that inside optimization** | P18, S13, O09, prior-review multi-fidelity BO (§7.2) | B11, B29, B7 (virtual metrology), B18 (screen → assay) |
| C8 ★ | **Expensive black-box optimization with safety constraints** | O01, O02, O04, S14 (history matching = ruling out unsafe space), O18, O14 (safe ES) | B5, B20, B7, B6 |
| C9 ★ | **High-dimensional settings (~60 knobs) with few evaluations** | O02, prior-review SAAS, O05 (priors), O07 (causal structure), S26/S02 (find the few relevant knobs) | B5, B8, B29 |
| C10 ★ | **Multi-objective trade-offs** (σe vs λ vs readout time vs CTE) | O03, O13, O08 (preference) | B8, B17, B11 |
| C11 ★ | **Drift / non-stationarity** (cool-downs, temperature, ageing) | S20, S24, P16, P07, O06, O14, O19, U04 (adaptive conformal), P11 | B7, B5, B15, B10, B23 |
| C12 ★ | **Transfer to a new unit (new board or amplifier): cold start** | O05 (RGPE), P08, P06, S15, B21 matrix factorization, P05 | B21, B12 (new hospital), B6 (device variability), B16 (ungauged basins) |
| C13 ★ | **Calibrated uncertainty and honest evaluation** | U01–U05, U08, S07, T04, G02 | B16 (forecast verification), B12, B4 |
| C14 ★ | **Physics partially known (grey box)** | D09, S14, S13 (discrepancy), S26, S27, O05 (physics priors) | B16, B5, B3, B17 |
| C15 | **A simulator exists but is imperfect (sim-to-real)** | G06–G08, S14, O15 (sim-trained policies), D10 | B20, B4, B1, B8 |
| C16 ★ | **Labels are themselves noisy fits with known errors** | S11 (heteroscedastic / known noise), S15, U05 (are the label errors calibrated?), B2 (EM fits of photon-counting histograms) | B2, B3, B1 |
| C17 ★ | **Wanting scientific insight, not just predictions** | S04, S26, S27, T05, U06, U07, P14 | B17, B16, B12 |
| C18 ★ | **Instrument health / anomaly / fault detection** | P15, S21 (HMM regimes), S24, P16 | B1, B4, B23, B7 |
| C19 ★ | **Autonomous closed loop with human oversight** | O23, O22, O08, O14 | B17, B6, B5, B7 |
| C20 ★ | **Inverse problem: infer noise-model parameters from a measured signal** | G07 (NPE), S08/S09 (likelihood fits of PSD-shaped GPs), B6 neural noise spectroscopy, B15 inverse surrogate (Horvath) | B6, B4, B15, B2 |
| C21 | **Witness/auxiliary channels to remove coupled noise** | D02 (DeepClean-style regression), S20 | B4, B9 |
| C22 ★ | **Choosing which experiment to run to learn the most (not to optimize)** | P10, O20 | B17, B18, B12, B7 |

**Reading tip for the exploratory analyst.** The unrelated fields with the closest overall match (most ★ shapes at once) are
**batteries (B11)**: C2, C6, C7, C8 and C10. Next come **biosignals (B12)**: C1, C3, C4 and C13, and **quantum-device tuning
(B6)**: C1, C4, C8, C12 and C20. **Finance volatility (B15)** is a close match for C4 and C20 only.

---

# Part D: notable gaps and caveats of this catalogue

* **No ML work on skipper-CCD *readout traces* was found** (consistent with the prior review). The nearest neighbours are qubit
  noise spectroscopy with neural networks (B6), PCH-EM sensor characterization (B2), and DeepClean (B4).
* **Metrology (B10)** gave only journal abstracts (clock prediction with NNs). No arXiv-quality ML-for-Allan-variance work was
  found.
* **Semiconductor virtual metrology (B7)** is mostly in IEEE journals (paywalled). Only abstracts and theses were seen.
* **Sports, gaming, ecology** entries are thin by design: one or two signature works each.
* **Not covered in depth:** graph-structured BO, quantum ML, spiking/neuromorphic networks, optimal transport, kernel two-sample
  methods beyond MMD, Bayesian quadrature, probabilistic numerics, recommender cold-start specifics, and time-series
  classification ensembles (HIVE-COTE). These are listed here so the analysts know they were not searched.
* **Package maturity ratings are my assessment** as of writing, not checked per package. Names marked [unverified] may have
  changed.
* **Venues for arXiv papers.** For every arXiv entry I checked the ID, title, first author and year against the arXiv API during
  this search. The journal or conference venues in the reference list come from my knowledge and the arXiv metadata. Those not
  re-checked against the publisher are given as I know them. Treat a venue as [unverified] if it matters.

---

# References

All references were found during this search, through arXiv API metadata (ID, title, first author, year checked) or web
search of publisher, repository or index pages. "(abstract only)" means I saw only the abstract or a bibliographic record.
"[unverified]" marks details I could not confirm. Venues of arXiv papers come from my knowledge where the arXiv record did
not show them (see Part D). Items already in the prior review are not repeated here unless cited above.

**A–C**
* **[Abbaspourazad 2023]** Abbaspourazad et al. Large-scale training of foundation models for wearable biosignals. ICLR 2024. arXiv:2312.05409
* **[Adams & MacKay 2007]** Adams, MacKay. Bayesian online changepoint detection. arXiv:0710.3742
* **[Adriaensen 2023]** Adriaensen et al. Efficient Bayesian learning curve extrapolation using prior-data fitted networks. NeurIPS 2023. arXiv:2310.20447
* **[Agarwal 2020]** Agarwal et al. Neural additive models: interpretable machine learning with neural nets. NeurIPS 2021. arXiv:2004.13912
* **[Aglietti 2020]** Aglietti, Lu, Paleyes, González. Causal Bayesian optimization. AISTATS 2020. arXiv:2005.11741
* **[Akiba 2019]** Akiba et al. Optuna: a next-generation hyperparameter optimization framework. KDD 2019. arXiv:1907.10902
* **[Andén & Mallat 2013]** Andén, Mallat. Deep scattering spectrum. IEEE TSP 2014. arXiv:1304.6763
* **[Andreux 2018]** Andreux et al. Kymatio: scattering transforms in Python. JMLR 2020. arXiv:1812.11214
* **[Angelopoulos & Bates 2021]** Angelopoulos, Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511
* **[Ansari 2024]** Ansari et al. Chronos: learning the language of time series. TMLR 2024. arXiv:2403.07815
* **[Arjovsky 2019]** Arjovsky, Bottou, Gulrajani, Lopez-Paz. Invariant risk minimization. arXiv:1907.02893
* **[Arnquist 2022]** Arnquist et al. (Majorana Collaboration). Interpretable boosted decision tree analysis for the Majorana Demonstrator. arXiv:2207.10710
* **[Attia 2020]** Attia et al. Closed-loop optimization of fast-charging protocols for batteries with machine learning. *Nature* 578:397–402 (2020). doi:10.1038/s41586-020-1994-5 (abstract only)
* **[Baevski 2020]** Baevski, Zhou, Mohamed, Auli. wav2vec 2.0. NeurIPS 2020. arXiv:2006.11477
* **[Bai 2018]** Bai, Kolter, Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling (TCN). arXiv:1803.01271
* **[Baio & Blangiardo 2010]** Baio, Blangiardo. Bayesian hierarchical model for the prediction of football results. *J. Appl. Stat.* 37(2):253–264 (2010). doi:10.1080/02664760802684177
* **[Bardes 2021]** Bardes, Ponce, LeCun. VICReg. ICLR 2022. arXiv:2105.04906
* **[Bates 2014]** Bates, Mächler, Bolker, Walker. Fitting linear mixed-effects models using lme4. *J. Stat. Softw.* 2015. arXiv:1406.5823
* **[Battaglia 2018]** Battaglia et al. Relational inductive biases, deep learning, and graph networks. arXiv:1806.01261
* **[Beaumont 2002]** Beaumont, Zhang, Balding. Approximate Bayesian computation in population genetics. *Genetics* 162(4):2025–2035 (2002). doi:10.1093/genetics/162.4.2025 (abstract only)
* **[Berkenkamp 2015 quadrotor]** Berkenkamp, Schoellig, Krause. Safe controller optimization for quadrotors with Gaussian processes. ICRA 2016. arXiv:1509.01066
* **[Berkenkamp 2017]** Berkenkamp, Turchetta, Schoellig, Krause. Safe model-based reinforcement learning with stability guarantees. NeurIPS 2017. arXiv:1705.08551
* **[Biswas 2021]** Biswas, Khimulya, Alley, Esvelt, Church. Low-N protein engineering with data-efficient deep learning. *Nat. Methods* 18:389–396 (2021). doi:10.1038/s41592-021-01100-y; bioRxiv 10.1101/2020.01.23.917682 (abstract only)
* **[Blei 2016]** Blei, Kucukelbir, McAuliffe. Variational inference: a review for statisticians. *JASA* 2017. arXiv:1601.00670
* **[Blundell 2015]** Blundell et al. Weight uncertainty in neural networks. ICML 2015. arXiv:1505.05424
* **[Bogunovic 2016]** Bogunovic, Scarlett, Cevher. Time-varying Gaussian process bandit optimization. AISTATS 2016. arXiv:1601.06650
* **[Boiko 2023]** Boiko, MacKnight, Kline, Gomes. Autonomous chemical research with large language models. *Nature* 624 (2023) (news and abstract only)
* **[Bollerslev 1986]** Bollerslev. Generalized autoregressive conditional heteroskedasticity. *J. Econometrics* 31(3):307–327 (1986). doi:10.1016/0304-4076(86)90063-1
* **[Bran 2023]** Bran et al. ChemCrow: augmenting large-language models with chemistry tools. *Nat. Mach. Intell.* 2024. arXiv:2304.05376
* **[Breiman 2001]** Breiman. Random forests. *Mach. Learn.* 45:5–32 (2001). doi:10.1023/A:1010933404324
* **[Brown 2020]** Brown et al. Language models are few-shot learners. NeurIPS 2020. arXiv:2005.14165
* **[Brunton 2016]** Brunton, Proctor, Kutz. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *PNAS* 2016. arXiv:1509.03580
* **[Cababie 2023]** Cababie. Origin and characterization of single-electron events in Skipper-CCDs for light dark matter searches (thesis). arXiv:2309.10458
* **[Calandra 2016]** Calandra, Seyfarth, Peters, Deisenroth. Bayesian optimization for learning gaits under uncertainty. *Ann. Math. Artif. Intell.* 76:5–23 (2016) (abstract only)
* **[Cao 2024]** Cao et al. Agents for self-driving laboratories applied to quantum computing. arXiv:2412.07978
* **[Chen & Guestrin 2016]** Chen, Guestrin. XGBoost: a scalable tree boosting system. KDD 2016. arXiv:1603.02754
* **[Chen 2018]** Chen, Rubanova, Bettencourt, Duvenaud. Neural ordinary differential equations. NeurIPS 2018. arXiv:1806.07366
* **[Chen 2018 AlphaGo]** Chen, Huang, Wang et al. Bayesian optimization in AlphaGo. arXiv:1812.06855
* **[Chen 2020 SimCLR]** Chen, Kornblith, Norouzi, Hinton. A simple framework for contrastive learning of visual representations. ICML 2020. arXiv:2002.05709
* **[Chen 2021 DT]** Chen et al. Decision Transformer: reinforcement learning via sequence modeling. NeurIPS 2021. arXiv:2106.01345
* **[Chernozhukov 2016]** Chernozhukov et al. Double/debiased machine learning for treatment and causal parameters. *Econometrics J.* 2018. arXiv:1608.00060
* **[Chevyrev & Kormilitzin 2016]** Chevyrev, Kormilitzin. A primer on the signature method in machine learning. arXiv:1603.03788
* **[Chipman 2008]** Chipman, George, McCulloch. BART: Bayesian additive regression trees. *Ann. Appl. Stat.* 2010. arXiv:0806.3286
* **[Christ 2016]** Christ, Kempa-Liehr, Feindt. Distributed and parallel time series feature extraction for industrial big data applications (tsfresh). arXiv:1610.07717
* **[Chua 2018]** Chua, Calandra, McAllister, Levine. Deep reinforcement learning in a handful of trials using probabilistic dynamics models (PETS). NeurIPS 2018. arXiv:1805.12114
* **[Corsi 2009]** Corsi. A simple approximate long-memory model of realized volatility. *J. Financ. Econometrics* 7(2):174–196 (2009). doi:10.1093/jfec/nbp001
* **[Cowen-Rivers 2020]** Cowen-Rivers et al. HEBO: pushing the limits of sample-efficient hyperparameter optimisation. *JAIR* 2022. arXiv:2012.03826
* **[Cranmer 2020]** Cranmer et al. Discovering symbolic models from deep learning with inductive biases. NeurIPS 2020. arXiv:2006.11287
* **[Cranmer 2023]** Cranmer. Interpretable machine learning for science with PySR and SymbolicRegression.jl. arXiv:2305.01582

**D–G**
* **[Damianou & Lawrence 2012]** Damianou, Lawrence. Deep Gaussian processes. AISTATS 2013. arXiv:1211.0358
* **[Daulton 2021]** Daulton, Balandat, Bakshy. Parallel Bayesian optimization of multiple noisy objectives with expected hypervolume improvement (qNEHVI). NeurIPS 2021. arXiv:2105.08195
* **[Daw 2017]** Daw, Karpatne, Watkins, Read, Kumar. Physics-guided neural networks (PGNN): an application in lake temperature modeling. arXiv:1710.11431
* **[Dax 2021]** Dax et al. Real-time gravitational-wave science with neural posterior estimation (DINGO). *PRL* 2021. arXiv:2106.12594
* **[Deb 2002]** Deb, Pratap, Agarwal, Meyarivan. A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE TEC* 6(2):182–197 (2002). doi:10.1109/4235.996017
* **[Deisenroth & Rasmussen 2011]** Deisenroth, Rasmussen. PILCO: a model-based and data-efficient approach to policy search. ICML 2011, pp. 465–472
* **[Devlin 2018]** Devlin, Chang, Lee, Toutanova. BERT. NAACL 2019. arXiv:1810.04805
* **[Dinh 2016]** Dinh, Sohl-Dickstein, Bengio. Density estimation using Real NVP. ICLR 2017. arXiv:1605.08803
* **[Domhan 2015]** Domhan, Springenberg, Hutter. Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves. IJCAI 2015, pp. 3460–3468 (bibliographic record only)
* **[Donahue 2018]** Donahue, McAuley, Puckette. Adversarial audio synthesis (WaveGAN). ICLR 2019. arXiv:1802.04208
* **[Donoho 2006]** Donoho. Compressed sensing. *IEEE Trans. Inf. Theory* 52(4):1289–1306 (2006). doi:10.1109/TIT.2006.871582
* **[Dosovitskiy 2020]** Dosovitskiy et al. An image is worth 16×16 words (ViT). ICLR 2021. arXiv:2010.11929
* **[Du & Mordatch 2019]** Du, Mordatch. Implicit generation and generalization in energy-based models. NeurIPS 2019. arXiv:1903.08689
* **[Edelen 2018]** Edelen et al. Opportunities in machine learning for particle accelerators. arXiv:1811.03172
* **[Ellis 2020]** Ellis et al. DreamCoder: growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning. arXiv:2006.08381
* **[Erickson 2020]** Erickson et al. AutoGluon-Tabular. arXiv:2003.06505
* **[Eriksson 2019]** Eriksson, Pearce, Gardner, Turner, Poloczek. Scalable global optimization via local Bayesian optimization (TuRBO). NeurIPS 2019. arXiv:1910.01739
* **[Evensen 1994]** Evensen. Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics. *JGR* 99(C5):10143–10162 (1994). doi:10.1029/94JC00572 (abstract only)
* **[Feickert & Nachman 2021]** Feickert, Nachman. A living review of machine learning for particle physics. arXiv:2102.02770
* **[Feinberg & Langtangen 2015]** Feinberg, Langtangen. Chaospy: an open source tool for designing methods of uncertainty quantification. *J. Comput. Sci.* 11:46–57 (2015). doi:10.1016/j.jocs.2015.08.008
* **[Fernández-Godino 2016]** Fernández-Godino et al. Review of multi-fidelity models. arXiv:1609.07196
* **[Feurer 2018]** Feurer, Letham, Hutter, Bakshy. Practical transfer learning for Bayesian optimization (RGPE). arXiv:1802.02219
* **[Feurer 2020]** Feurer et al. Auto-sklearn 2.0: hands-free AutoML via meta-learning. *JMLR* 2022. arXiv:2007.04074
* **[Finn 2017]** Finn, Abbeel, Levine. Model-agnostic meta-learning (MAML). ICML 2017. arXiv:1703.03400
* **[Foi 2008]** Foi, Trimeche, Katkovnik, Egiazarian. Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data. *IEEE TIP* 17(10):1737–1754 (2008) (citation record and author page only)
* **[Foreman-Mackey 2017]** Foreman-Mackey, Agol, Ambikasaran, Angus. Fast and scalable Gaussian process modeling with applications to astronomical time series (celerite). *AJ* 2017. arXiv:1703.09710
* **[Frate 2017]** Frate, Cranmer, Kalia, Vandenberg-Rodes, Whiteson. Modeling smooth backgrounds and generic localized signals with Gaussian processes. arXiv:1709.05681
* **[Frazier 2018]** Frazier. A tutorial on Bayesian optimization. arXiv:1807.02811
* **[Friedman 2001]** Friedman. Greedy function approximation: a gradient boosting machine. *Ann. Stat.* 29(5):1189–1232 (2001). doi:10.1214/aos/1013203451
* **[Fulcher & Jones 2016]** Fulcher, Jones. hctsa: a computational framework for automated time-series phenotyping using massive feature extraction. *Cell Systems* 5:527–531 (2017). arXiv:1612.05296
* **[Gal & Ghahramani 2015]** Gal, Ghahramani. Dropout as a Bayesian approximation. ICML 2016. arXiv:1506.02142
* **[Gal 2017]** Gal, Islam, Ghahramani. Deep Bayesian active learning with image data. ICML 2017. arXiv:1703.02910
* **[Ganin 2015]** Ganin et al. Domain-adversarial training of neural networks (DANN). *JMLR* 2016. arXiv:1505.07818
* **[Garcez & Lamb 2020]** d'Avila Garcez, Lamb. Neurosymbolic AI: the 3rd wave. arXiv:2012.05876
* **[Gardner 2018]** Gardner, Pleiss, Bindel, Weinberger, Wilson. GPyTorch. NeurIPS 2018. arXiv:1809.11165
* **[Garnelo 2018]** Garnelo et al. Conditional neural processes. ICML 2018. arXiv:1807.01613
* **[Gershman & Blei 2011]** Gershman, Blei. A tutorial on Bayesian nonparametric models. arXiv:1106.2697
* **[Gibbs & Candès 2021]** Gibbs, Candès. Adaptive conformal inference under distribution shift. NeurIPS 2021. arXiv:2106.00170
* **[Gneiting & Raftery 2007]** Gneiting, Raftery. Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477):359–378 (2007). doi:10.1198/016214506000001437
* **[Gong 2021]** Gong, Chung, Glass. AST: audio spectrogram transformer. Interspeech 2021. arXiv:2104.01778
* **[González 2017]** González, Dai, Damianou, Lawrence. Preferential Bayesian optimization. ICML 2017. arXiv:1704.03651
* **[Goodfellow 2014]** Goodfellow et al. Generative adversarial networks. arXiv:1406.2661
* **[Gorishniy 2021]** Gorishniy, Rubachev, Khrulkov, Babenko. Revisiting deep learning models for tabular data (FT-Transformer). NeurIPS 2021. arXiv:2106.11959
* **[Gorishniy 2024]** Gorishniy, Kotelnikov, Babenko. TabM: advancing tabular deep learning with parameter-efficient ensembling. arXiv:2410.24210
* **[Gretton 2008]** Gretton et al. A kernel method for the two-sample problem (MMD). arXiv:0805.2368
* **[Grill 2020]** Grill et al. Bootstrap your own latent (BYOL). NeurIPS 2020. arXiv:2006.07733
* **[Grinsztajn 2022]** Grinsztajn, Oyallon, Varoquaux. Why do tree-based models still outperform deep learning on tabular data? NeurIPS 2022 D&B. arXiv:2207.08815
* **[Gu 2021]** Gu, Goel, Ré. Efficiently modeling long sequences with structured state spaces (S4). ICLR 2022. arXiv:2111.00396
* **[Gu & Dao 2023]** Gu, Dao. Mamba: linear-time sequence modeling with selective state spaces. arXiv:2312.00752
* **[Guo 2017]** Guo, Pleiss, Sun, Weinberger. On calibration of modern neural networks. ICML 2017. arXiv:1706.04599
* **[Gupta 2025]** Gupta et al. Expedited noise spectroscopy of transmon qubits. arXiv:2502.00679
* **[Gururangan 2020]** Gururangan et al. Don't stop pretraining: adapt language models to domains and tasks. ACL 2020. arXiv:2004.10964

**H–K**
* **[Ha 2016]** Ha, Dai, Le. HyperNetworks. ICLR 2017. arXiv:1609.09106
* **[Haarnoja 2018]** Haarnoja, Zhou, Abbeel, Levine. Soft actor-critic. ICML 2018. arXiv:1801.01290
* **[Hajjem 2014]** Hajjem, Bellavance, Larocque. Mixed-effects random forest for clustered data. *J. Stat. Comput. Simul.* 84(6):1313–1328 (2014). doi:10.1080/00949655.2012.741599 (abstract only)
* **[Hansen 2016]** Hansen. The CMA evolution strategy: a tutorial. arXiv:1604.00772
* **[Harilal 2023]** Harilal, Park, Andrews, Paulini. Autoencoder-based online data quality monitoring for the CMS electromagnetic calorimeter. arXiv:2308.16659
* **[Hastie & Tibshirani 1986]** Hastie, Tibshirani. Generalized additive models. *Statist. Sci.* 1(3):297–318 (1986)
* **[He 2015]** He, Zhang, Ren, Sun. Deep residual learning for image recognition. CVPR 2016. arXiv:1512.03385
* **[He 2017]** He et al. Neural collaborative filtering. WWW 2017. arXiv:1708.05031
* **[He 2021 MAE]** He et al. Masked autoencoders are scalable vision learners. CVPR 2022. arXiv:2111.06377
* **[Hendrickson 2024]** Hendrickson et al. PCH-EM: a solution to information loss in the photon transfer method. arXiv:2403.04498
* **[Hendrickson 2025]** Hendrickson et al. Initialization-robust characterization of deep sub-electron read noise pixels via annealed PCH-EM. arXiv:2509.13460
* **[Hensman 2013]** Hensman, Fusi, Lawrence. Gaussian processes for big data. UAI 2013. arXiv:1309.6835
* **[Herbrich 2006]** Herbrich, Minka, Graepel. TrueSkill: a Bayesian skill rating system. NIPS 2006 (NeurIPS proceedings page)
* **[Hermans 2019]** Hermans, Begy, Louppe. Likelihood-free MCMC with amortized approximate ratio estimators. ICML 2020. arXiv:1903.04057
* **[Hewing 2020]** Hewing, Wabersich, Menner, Zeilinger. Learning-based model predictive control: toward safe learning in control. *Annu. Rev. Control Robot. Auton. Syst.* 3:269–296 (2020) (record only)
* **[Ho 2020]** Ho, Jain, Abbeel. Denoising diffusion probabilistic models. NeurIPS 2020. arXiv:2006.11239
* **[Holzmüller 2024]** Holzmüller, Grinsztajn, Steinwart. Better by default: strong pre-tuned MLPs and boosted trees on tabular data (RealMLP). NeurIPS 2024. arXiv:2407.04491
* **[Horvath 2019]** Horvath, Muguruza, Tomas. Deep learning volatility. *Quant. Finance* 2021. arXiv:1901.09647
* **[Houlsby 2011]** Houlsby, Huszár, Ghahramani, Lengyel. Bayesian active learning for classification and preference learning (BALD). arXiv:1112.5745
* **[Hsu 2021]** Hsu et al. HuBERT. IEEE/ACM TASLP 2021. arXiv:2106.07447
* **[Hu 2021 LoRA]** Hu et al. LoRA: low-rank adaptation of large language models. ICLR 2022. arXiv:2106.09685
* **[Hvarfner 2022]** Hvarfner et al. πBO: augmenting acquisition functions with user beliefs for Bayesian optimization. ICLR 2022. arXiv:2204.11051
* **[Häse 2018]** Häse, Roch, Kreisbeck, Aspuru-Guzik. PHOENICS: a universal deep Bayesian optimizer. *ACS Cent. Sci.* 2018. arXiv:1801.01469
* **[Ismail Fawaz 2018 review]** Ismail Fawaz et al. Deep learning for time series classification: a review. *DMKD* 2019. arXiv:1809.04356
* **[Ismail Fawaz 2019 InceptionTime]** Ismail Fawaz et al. InceptionTime: finding AlexNet for time series classification. *DMKD* 2020. arXiv:1909.04939
* **[Iwana & Uchida 2020]** Iwana, Uchida. An empirical survey of data augmentation for time series classification with neural networks. *PLoS ONE* 2021. arXiv:2007.15951
* **[Jin 2023]** Jin et al. Time-LLM: time series forecasting by reprogramming large language models. ICLR 2024. arXiv:2310.01728
* **[Ju 2021]** Ju et al. (Exa.TrkX). Performance of a geometric deep learning pipeline for HL-LHC particle tracking. *EPJC* 2021. arXiv:2103.06995
* **[Jumper 2021]** Jumper et al. Highly accurate protein structure prediction with AlphaFold. *Nature* 596:583–589 (2021). doi:10.1038/s41586-021-03819-2 (abstract only)
* **[Kahl 2021]** Kahl, Wood, Eibl, Klinck. BirdNET: a deep learning solution for avian diversity monitoring. *Ecol. Inform.* 61:101236 (2021) (citation record only)
* **[Kain 2020]** Kain, Hirlander, Goddard, Velotti, Zevi Della Porta, Bruchon, Valentino. Sample-efficient reinforcement learning for CERN accelerator control. *PRAB* 23:124801 (2020). doi:10.1103/PhysRevAccelBeams.23.124801
* **[Kaiser 2024 LLM]** Kaiser et al. Large language models for human-machine collaborative particle accelerator tuning through natural language. arXiv:2405.08888
* **[Kaiser 2024 Sci. Rep.]** Kaiser, Xu, Eichler, Santamaria Garcia et al. Reinforcement learning-trained optimisers and Bayesian optimisation for online particle accelerator tuning. *Sci. Rep.* 14:15733 (2024). doi:10.1038/s41598-024-66263-y (abstract only)
* **[Kalantre 2017]** Kalantre et al. Machine learning techniques for state recognition and auto-tuning in quantum dots. *npj Quantum Inf.* 2019. arXiv:1712.04914
* **[Kalman 1960]** Kalman. A new approach to linear filtering and prediction problems. *J. Basic Eng.* 82(1):35–45 (1960)
* **[Kanarik 2023]** Kanarik et al. Human–machine collaboration for improving semiconductor process development. *Nature* 616:707–711 (2023). doi:10.1038/s41586-023-05773-7; open access PMC10132970 (abstract only)
* **[Kaptanoglu 2021]** Kaptanoglu et al. PySINDy: a comprehensive Python package for robust sparse system identification. *JOSS* 2022. arXiv:2111.08481
* **[Karpatne 2016]** Karpatne et al. Theory-guided data science: a new paradigm for scientific discovery from data. *IEEE TKDE* 2017. arXiv:1612.08544
* **[Kasieczka 2021]** Kasieczka et al. The LHC Olympics 2020: a community challenge for anomaly detection in high energy physics. *Rep. Prog. Phys.* 2021. arXiv:2101.08320
* **[Ke 2017]** Ke et al. LightGBM: a highly efficient gradient boosting decision tree. NeurIPS 2017
* **[Kelly 2018]** Kelly, O'Malley, Neeley, Neven, Martinis. Physical qubit calibration on a directed acyclic graph. arXiv:1803.03226
* **[Kennedy & O'Hagan 2001]** Kennedy, O'Hagan. Bayesian calibration of computer models. *JRSS B* 63(3):425–464 (2001). doi:10.1111/1467-9868.00294
* **[Kidger & Lyons 2020]** Kidger, Lyons. Signatory: differentiable computations of the signature and logsignature transforms. ICLR 2021. arXiv:2001.00706
* **[Kidger 2020 CDE]** Kidger, Morrill, Foster, Lyons. Neural controlled differential equations for irregular time series. NeurIPS 2020. arXiv:2005.08926
* **[Kidger 2021 SDE]** Kidger, Foster, Li, Oberhauser, Lyons. Neural SDEs as infinite-dimensional GANs. ICML 2021. arXiv:2102.03657
* **[Kingma & Welling 2013]** Kingma, Welling. Auto-encoding variational Bayes. ICLR 2014. arXiv:1312.6114
* **[Kipf & Welling 2016]** Kipf, Welling. Semi-supervised classification with graph convolutional networks. ICLR 2017. arXiv:1609.02907
* **[Kirkpatrick 2016]** Kirkpatrick et al. Overcoming catastrophic forgetting in neural networks (EWC). *PNAS* 2017. arXiv:1612.00796
* **[Kochkov 2023]** Kochkov et al. Neural general circulation models for weather and climate (NeuralGCM). *Nature* 2024. arXiv:2311.07222
* **[Koenker & Hallock 2001]** Koenker, Hallock. Quantile regression. *J. Econ. Perspect.* 15(4):143–156 (2001). doi:10.1257/jep.15.4.143
* **[Koren 2009]** Koren, Bell, Volinsky. Matrix factorization techniques for recommender systems. *IEEE Computer* 42(8):30–37 (2009). doi:10.1109/MC.2009.263
* **[Kratzert 2018]** Kratzert, Klotz, Brenner, Schulz, Herrnegger. Rainfall–runoff modelling using long short-term memory (LSTM) networks. *HESS* 22:6005–6022 (2018). doi:10.5194/hess-22-6005-2018
* **[Kratzert 2019]** Kratzert et al. Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets (EA-LSTM). *HESS* 2019. arXiv:1907.08456
* **[Krishnan 2015]** Krishnan, Shalit, Sontag. Deep Kalman filters. arXiv:1511.05121
* **[Kumar 2020]** Kumar, Zhou, Tucker, Levine. Conservative Q-learning for offline reinforcement learning. NeurIPS 2020. arXiv:2006.04779

**L–P**
* **[Lakshminarayanan 2016]** Lakshminarayanan, Pritzel, Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS 2017. arXiv:1612.01474
* **[Lam 2022]** Lam et al. GraphCast: learning skillful medium-range global weather forecasting. *Science* 2023. arXiv:2212.12794
* **[Lawhern 2016]** Lawhern et al. EEGNet: a compact convolutional network for EEG-based brain-computer interfaces. *J. Neural Eng.* 2018. arXiv:1611.08024
* **[Lehtinen 2018]** Lehtinen et al. Noise2Noise: learning image restoration without clean data. ICML 2018. arXiv:1803.04189
* **[Levine 2020]** Levine, Kumar, Tucker, Fu. Offline reinforcement learning: tutorial, review, and perspectives on open problems. arXiv:2005.01643
* **[Li 2010]** Li, Chu, Langford, Schapire. A contextual-bandit approach to personalized news article recommendation (LinUCB). WWW 2010. arXiv:1003.0146
* **[Li 2016 Hyperband]** Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar. Hyperband. *JMLR* 2018. arXiv:1603.06560
* **[Li 2020]** Li et al. Fourier neural operator for parametric partial differential equations. ICLR 2021. arXiv:2010.08895
* **[Lin 2023]** Lin et al. Evolutionary-scale prediction of atomic-level protein structure with a language model (ESM-2/ESMFold). *Science* 379:1123–1130 (2023). doi:10.1126/science.ade2574 (abstract only)
* **[Liu 2008]** Liu, Ting, Zhou. Isolation forest. ICDM 2008, pp. 413–422
* **[Liu 2024]** Liu et al. KAN: Kolmogorov–Arnold networks. ICLR 2025. arXiv:2404.19756
* **[Liu 2024 LLAMBO]** Liu, Astorga, Seedat, van der Schaar. Large language models to enhance Bayesian optimization (LLAMBO). ICLR 2024. arXiv:2402.03921
* **[Liu 2026 CCD noise]** Liu et al. Denoising the deep sky: physics-based CCD noise formation for astronomical imaging. arXiv:2601.23276 (abstract only)
* **[Lu 2019]** Lu, Jin, Karniadakis. DeepONet. *Nat. Mach. Intell.* 2021. arXiv:1910.03193
* **[Lu 2020]** Lu et al. Learning under concept drift: a review. *IEEE TKDE* 2019. arXiv:2004.05785
* **[Lu 2024]** Lu et al. The AI Scientist: towards fully automated open-ended scientific discovery. arXiv:2408.06292
* **[Lubba 2019]** Lubba et al. catch22: CAnonical Time-series CHaracteristics. *DMKD* 2019. arXiv:1901.10200
* **[Lundberg & Lee 2017]** Lundberg, Lee. A unified approach to interpreting model predictions (SHAP). NeurIPS 2017. arXiv:1705.07874
* **[Lusch 2017]** Lusch, Kutz, Brunton. Deep learning for universal linear embeddings of nonlinear dynamics. *Nat. Commun.* 2018. arXiv:1712.09707
* **[Lyu 2018]** Lyu, Yang, Yan, Zhou, Zeng. Batch Bayesian optimization via multi-objective acquisition ensemble for automated analog circuit design. ICML 2018, PMLR 80:3306–3314
* **[MacLeod 2019]** MacLeod et al. Self-driving laboratory for accelerated discovery of thin-film materials (Ada). *Sci. Adv.* 2020. arXiv:1906.05398
* **[Mallat 2011]** Mallat. Group invariant scattering. *CPAM* 2012. arXiv:1101.2286
* **[Martina 2023]** Martina et al. Deep learning enhanced noise spectroscopy of a spin qubit environment. *Mach. Learn.: Sci. Technol.* 2023. arXiv:2301.05079
* **[McCabe 2023]** McCabe et al. Multiple physics pretraining for physical surrogate models. arXiv:2310.02994
* **[McMahan 2016]** McMahan et al. Communication-efficient learning of deep networks from decentralized data (FedAvg). AISTATS 2017. arXiv:1602.05629
* **[Meng & Karniadakis 2019]** Meng, Karniadakis. A composite neural network that learns from multi-fidelity data. *J. Comput. Phys.* 2020. arXiv:1903.00104
* **[Mnih 2013]** Mnih et al. Playing Atari with deep reinforcement learning. arXiv:1312.5602
* **[Moon 2020]** Moon et al. Machine learning enables completely automatic tuning of a quantum device faster than human experts. *Nat. Commun.* 11:4161 (2020). arXiv:2001.02589
* **[Mousavi 2020]** Mousavi, Ellsworth, Zhu, Chuang, Beroza. Earthquake transformer. *Nat. Commun.* 11:3952 (2020). doi:10.1038/s41467-020-17591-w
* **[Nakazawa & Kulkarni 2018]** Nakazawa, Kulkarni. Wafer map defect pattern classification and image retrieval using convolutional neural network. *IEEE TSM* 31(2):309–314 (2018). doi:10.1109/TSM.2018.2795466 (via citing papers only)
* **[Nguyen 2023 ClimaX]** Nguyen et al. ClimaX: a foundation model for weather and climate. ICML 2023. arXiv:2301.10343
* **[Niu 2018]** Niu, Boixo, Smelyanskiy, Neven. Universal quantum control through deep reinforcement learning. *npj Quantum Inf.* 2019. arXiv:1803.01857
* **[Nori 2019]** Nori, Jenkins, Koch, Caruana. InterpretML: a unified framework for machine learning interpretability. arXiv:1909.09223
* **[Nousiainen 2021]** Nousiainen, Rajani, Kasper, Helin. Adaptive optics control using model-based reinforcement learning. arXiv:2104.13685 (venue unverified; later lab work in JATIS 2024, arXiv:2401.00242)
* **[O'Shea 2017]** O'Shea, Roy, Clancy. Over-the-air deep learning based radio signal classification. *IEEE JSTSP* 2018. arXiv:1712.04578
* **[Ohana 2024]** Ohana et al. The Well: a large-scale collection of diverse physics simulations for machine learning. NeurIPS 2024 D&B. arXiv:2412.00568
* **[Oreshkin 2019]** Oreshkin, Carpov, Chapados, Bengio. N-BEATS. ICLR 2020. arXiv:1905.10437
* **[Ormiston 2020]** Ormiston, Nguyen, Coughlin, Adhikari, Katsavounidis. Noise reduction in gravitational-wave data via deep learning (DeepClean). *Phys. Rev. Research* 2:033066 (2020). arXiv:2005.06534
* **[Ouyang 2017]** Ouyang et al. SISSO: a compressed-sensing method for identifying the best low-dimensional descriptor. *Phys. Rev. Materials* 2018. arXiv:1710.03319
* **[Paganini 2017]** Paganini, de Oliveira, Nachman. Accelerating science with generative adversarial networks: an application to 3D particle showers in multi-layer calorimeters (CaloGAN). *PRL* 2018. arXiv:1705.02355
* **[Papamakarios & Murray 2016]** Papamakarios, Murray. Fast ε-free inference of simulation models with Bayesian conditional density estimation. NeurIPS 2016. arXiv:1605.06376
* **[Papamakarios 2018]** Papamakarios, Sterratt, Murray. Sequential neural likelihood. AISTATS 2019. arXiv:1805.07226
* **[Papamakarios 2019 review]** Papamakarios et al. Normalizing flows for probabilistic modeling and inference. *JMLR* 2021. arXiv:1912.02762
* **[Papenmeier 2023]** Papenmeier, Nardi, Poloczek. Increasing the scope as you learn: adaptive Bayesian optimization in nested subspaces (BAxUS). NeurIPS 2022. arXiv:2304.11468
* **[Parker 2023]** Parker et al. AstroCLIP: a cross-modal foundation model for galaxies. *MNRAS* 2024. arXiv:2310.03024
* **[Parnami & Lee 2022]** Parnami, Lee. Learning from few examples: a summary of approaches to few-shot learning. arXiv:2203.04291
* **[Pathak 2017]** Pathak, Lu, Hunt, Girvan, Ott. Using machine learning to replicate chaotic attractors and calculate Lyapunov exponents from data (reservoir computing). *Chaos* 2017. arXiv:1710.07313
* **[Pedregosa 2011]** Pedregosa et al. Scikit-learn: machine learning in Python. *JMLR* 12 (2011). arXiv:1201.0490
* **[Peherstorfer 2018]** Peherstorfer, Willcox, Gunzburger. Survey of multifidelity methods in uncertainty propagation, inference, and optimization. *SIAM Rev.* 60(3):550–591 (2018). arXiv:1806.10761
* **[Perez 2017]** Perez, Strub, de Vries, Dumoulin, Courville. FiLM: visual reasoning with a general conditioning layer. AAAI 2018. arXiv:1709.07871
* **[Pfaff 2020]** Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia. Learning mesh-based simulation with graph networks. ICLR 2021. arXiv:2010.03409
* **[Phan 2019]** Phan, Pradhan, Jankowiak. Composable effects for flexible and accelerated probabilistic programming in NumPyro. arXiv:1912.11554
* **[Phillips 2006]** Phillips, Anderson, Schapire. Maximum entropy modeling of species geographic distributions. *Ecol. Model.* 190:231–259 (2006)
* **[Pol 2018]** Pol et al. Detector monitoring with artificial neural networks at the CMS experiment at the CERN LHC. arXiv:1808.00911
* **[Prokhorenkova 2017]** Prokhorenkova et al. CatBoost: unbiased boosting with categorical features. NeurIPS 2018. arXiv:1706.09516

**Q–Z**
* **[Qu & Gouskos 2019]** Qu, Gouskos. ParticleNet: jet tagging via particle clouds. *PRD* 2020. arXiv:1902.08570
* **[Rabanser 2018]** Rabanser, Günnemann, Lipton. Failing loudly: an empirical study of methods for detecting dataset shift. NeurIPS 2019. arXiv:1810.11953
* **[Rabiner 1989]** Rabiner. A tutorial on hidden Markov models and selected applications in speech recognition. *Proc. IEEE* 77(2):257–286 (1989)
* **[Rackauckas 2020]** Rackauckas et al. Universal differential equations for scientific machine learning. arXiv:2001.04385
* **[Radford 2021]** Radford et al. Learning transferable visual models from natural language supervision (CLIP). ICML 2021. arXiv:2103.00020
* **[Rainforth 2023]** Rainforth, Foster, Ivanova, Bickford Smith. Modern Bayesian experimental design. *Statist. Sci.* 2024. arXiv:2302.14545
* **[Raissi 2017]** Raissi, Perdikaris, Karniadakis. Physics informed deep learning (Part I). arXiv:1711.10561 (journal version *J. Comput. Phys.* 2019)
* **[Rajpurkar 2017]** Rajpurkar, Hannun et al. Cardiologist-level arrhythmia detection with convolutional neural networks. arXiv:1707.01836 (later *Nat. Med.* 2019, Hannun et al.)
* **[Rasmussen & Williams 2006]** Rasmussen, Williams. *Gaussian Processes for Machine Learning*. MIT Press 2006. http://gaussianprocess.org/gpml/
* **[Rasul 2023]** Rasul et al. Lag-Llama: towards foundation models for probabilistic time series forecasting. arXiv:2310.08278
* **[Ravanelli 2018]** Ravanelli, Bengio. Speaker recognition from raw waveform with SincNet. SLT 2018. arXiv:1808.00158
* **[Reddy 2020]** Reddy, Gopal, Cutler. DNSMOS: a non-intrusive perceptual objective speech quality metric to evaluate noise suppressors. ICASSP 2021. arXiv:2010.15258
* **[Revach 2021]** Revach et al. KalmanNet: neural network aided Kalman filtering for partially known dynamics. *IEEE TSP* 2022. arXiv:2107.10043
* **[Romera-Paredes 2024]** Romera-Paredes et al. Mathematical discoveries from program search with large language models (FunSearch). *Nature* 625:468–475 (2024). doi:10.1038/s41586-023-06924-6
* **[Ruder 2017]** Ruder. An overview of multi-task learning in deep neural networks. arXiv:1706.05098
* **[Rudin 2018]** Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nat. Mach. Intell.* 2019. arXiv:1811.10154
* **[Ruff 2020]** Ruff et al. A unifying review of deep and shallow anomaly detection. *Proc. IEEE* 2021. arXiv:2009.11732
* **[Russo 2017]** Russo, Van Roy, Kazerouni, Osband, Wen. A tutorial on Thompson sampling. arXiv:1707.02038
* **[Sachs 1995]** Sachs, Hu, Ingolfsson. Run by run process control: combining SPC and feedback control. *IEEE Trans. Semicond. Manuf.* 8:26–43 (1995) (via secondary citations only)
* **[Saeb 2017]** Saeb, Lonini, Jayaraman, Mohr, Kording. The need to approximate the use-case in clinical machine learning. *GigaScience* 6(5) (2017). doi:10.1093/gigascience/gix019
* **[Sagawa 2019]** Sagawa, Koh, Hashimoto, Liang. Distributionally robust neural networks for group shifts (Group DRO). ICLR 2020. arXiv:1911.08731
* **[Salimbeni & Deisenroth 2017]** Salimbeni, Deisenroth. Doubly stochastic variational inference for deep Gaussian processes. NeurIPS 2017. arXiv:1705.08933
* **[Salinas 2017]** Salinas, Flunkert, Gasthaus. DeepAR: probabilistic forecasting with autoregressive recurrent networks. *IJF* 2020. arXiv:1704.04110
* **[Salvatier 2015]** Salvatier, Wiecki, Fonnesbeck. Probabilistic programming in Python using PyMC. *PeerJ CS* 2016. arXiv:1507.08050
* **[Saxena 2008]** Saxena, Goebel, Simon, Eklund. Damage propagation modeling for aircraft engine run-to-failure simulation (C-MAPSS). PHM 2008 (abstract only)
* **[Scheinker 2020 ACM4PA]** Scheinker et al. Advanced Control Methods for Particle Accelerators (ACM4PA) 2019 workshop report. arXiv:2001.05461
* **[Schrittwieser 2019]** Schrittwieser et al. Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero). *Nature* 2020. arXiv:1911.08265
* **[Schulman 2017]** Schulman et al. Proximal policy optimization algorithms. arXiv:1707.06347
* **[Seabold 2010]** Seabold, Perktold. statsmodels: econometric and statistical modeling with Python. Proc. SciPy 2010
* **[Servén & Brummitt 2018]** Servén, Brummitt. pyGAM: generalized additive models in Python. Zenodo 2018. doi:10.5281/zenodo.1208723
* **[Settaluri 2020]** Settaluri et al. AutoCkt: deep reinforcement learning of analog circuit designs. DATE 2020. arXiv:2001.01808
* **[Settles 2009]** Settles. Active learning literature survey. Univ. Wisconsin–Madison CS Tech. Rep. 1648 (2009)
* **[Severson 2019]** Severson et al. Data-driven prediction of battery cycle life before capacity degradation. *Nat. Energy* 4:383–391 (2019). doi:10.1038/s41560-019-0356-8 (abstract only)
* **[Sharma & Kiciman 2020]** Sharma, Kiciman. DoWhy: an end-to-end library for causal inference. arXiv:2011.04216
* **[Shazeer 2017]** Shazeer et al. Outrageously large neural networks: the sparsely-gated mixture-of-experts layer. ICLR 2017. arXiv:1701.06538
* **[Shwartz-Ziv 2021]** Shwartz-Ziv, Armon. Tabular data: deep learning is not all you need. *Inf. Fusion* 2022. arXiv:2106.03253
* **[Sigrist 2020]** Sigrist. Gaussian process boosting. *JMLR* 2022. arXiv:2004.02653
* **[Silver 2017]** Silver et al. Mastering chess and shogi by self-play with a general reinforcement learning algorithm (AlphaZero). arXiv:1712.01815
* **[Sitzmann 2020]** Sitzmann et al. Implicit neural representations with periodic activation functions (SIREN). NeurIPS 2020. arXiv:2006.09661
* **[Snell 2017]** Snell, Swersky, Zemel. Prototypical networks for few-shot learning. NeurIPS 2017. arXiv:1703.05175
* **[Snelson 2003]** Snelson, Rasmussen, Ghahramani. Warped Gaussian processes. NIPS 2003 (NeurIPS 16), pp. 337–344
* **[Snoek 2012]** Snoek, Larochelle, Adams. Practical Bayesian optimization of machine learning algorithms. NeurIPS 2012. arXiv:1206.2944
* **[Sohn 2020]** Sohn et al. FixMatch. NeurIPS 2020. arXiv:2001.07685
* **[Song 2020]** Song et al. Score-based generative modeling through stochastic differential equations. ICLR 2021. arXiv:2011.13456
* **[Srinivas 2009]** Srinivas, Krause, Kakade, Seeger. Gaussian process optimization in the bandit setting (GP-UCB). ICML 2010. arXiv:0912.3995
* **[Swersky 2014]** Swersky, Snoek, Adams. Freeze-thaw Bayesian optimization. arXiv:1406.3896
* **[Szabó 2014]** Szabó, Sriperumbudur, Póczos, Gretton. Learning theory for distribution regression. *JMLR* 2016. arXiv:1411.2066
* **[Talts 2018]** Talts, Betancourt, Simpson, Vehtari, Gelman. Validating Bayesian inference algorithms with simulation-based calibration. arXiv:1804.06788
* **[Tarvainen & Valpola 2017]** Tarvainen, Valpola. Mean teachers are better role models. NeurIPS 2017. arXiv:1703.01780
* **[Tashiro 2021]** Tashiro, Song, Song, Ermon. CSDI: conditional score-based diffusion models for probabilistic time series imputation. NeurIPS 2021. arXiv:2107.03502
* **[Thomson 1982]** Thomson. Spectrum estimation and harmonic analysis. *Proc. IEEE* 70(9):1055–1096 (1982). doi:10.1109/PROC.1982.12433 (citation via software references)
* **[Tibshirani 1996]** Tibshirani. Regression shrinkage and selection via the lasso. *JRSS B* 58(1):267–288 (1996)
* **[Tobar 2018]** Tobar. Bayesian nonparametric spectral estimation. NeurIPS 2018. arXiv:1809.02196
* **[Tobin 2017]** Tobin et al. Domain randomization for transferring deep neural networks from simulation to the real world. IROS 2017. arXiv:1703.06907
* **[Tu 2013]** Tu, Rowley, Luchtenburg, Brunton, Kutz. On dynamic mode decomposition: theory and applications. arXiv:1312.0041
* **[Udrescu & Tegmark 2019]** Udrescu, Tegmark. AI Feynman: a physics-inspired method for symbolic regression. *Sci. Adv.* 2020. arXiv:1905.11481
* **[Udrescu 2020]** Udrescu et al. AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity. NeurIPS 2020. arXiv:2006.10782
* **[Vaswani 2017]** Vaswani et al. Attention is all you need. NeurIPS 2017. arXiv:1706.03762
* **[van den Oord 2016]** van den Oord et al. WaveNet: a generative model for raw audio. arXiv:1609.03499
* **[van den Oord 2018 CPC]** van den Oord, Li, Vinyals. Representation learning with contrastive predictive coding. arXiv:1807.03748
* **[van Esbroeck 2020]** van Esbroeck et al. Quantum device fine-tuning using unsupervised embedding learning. *New J. Phys.* 2020. arXiv:2001.04409
* **[Wang 2015]** Wang, Chiou, Müller. Review of functional data analysis. *Annu. Rev. Stat. Appl.* 2016. arXiv:1507.05135
* **[Wang 2016]** Wang, Yan, Oates. Time series classification from scratch with deep neural networks: a strong baseline. IJCNN 2017. arXiv:1611.06455
* **[Wang 2020 GCN-RL]** Wang et al. GCN-RL circuit designer: transferable transistor sizing with graph neural networks and reinforcement learning. DAC 2020. arXiv:2005.00406
* **[Wang 2020 Tent]** Wang, Shelhamer, Liu, Olshausen, Darrell. Tent: fully test-time adaptation by entropy minimization. ICLR 2021. arXiv:2006.10726
* **[Williamson 2013]** Williamson, Goldstein et al. History matching for exploring and reducing climate model parameter space using observations and a large perturbed physics ensemble. *Clim. Dyn.* 41:1703–1729 (2013). doi:10.1007/s00382-013-1896-4 (abstract only)
* **[Wilson & Adams 2013]** Wilson, Adams. Gaussian process kernels for pattern discovery and extrapolation (spectral mixture kernel). ICML 2013. arXiv:1302.4245
* **[Wilson 2015]** Wilson, Hu, Salakhutdinov, Xing. Deep kernel learning. AISTATS 2016. arXiv:1511.02222
* **[Wistuba & Pedapati 2020]** Wistuba, Pedapati. Learning to rank learning curves. ICML 2020. arXiv:2006.03361
* **[Wold 2001]** Wold, Sjöström, Eriksson. PLS-regression: a basic tool of chemometrics. *Chemom. Intell. Lab. Syst.* 58:109–130 (2001). doi:10.1016/S0169-7439(01)00155-1
* **[Woollam 2021]** Woollam et al. SeisBench: a toolbox for machine learning in seismology. *Seismol. Res. Lett.* 2022. arXiv:2111.00786
* **[Xie & Grossman 2017]** Xie, Grossman. Crystal graph convolutional neural networks (CGCNN). *PRL* 2018. arXiv:1710.10324
* **[Xie 2019]** Xie, Luong, Hovy, Le. Self-training with Noisy Student improves ImageNet classification. CVPR 2020. arXiv:1911.04252
* **[Yang 2019]** Yang et al. Analyzing learned molecular representations for property prediction (Chemprop). *J. Chem. Inf. Model.* 2019. arXiv:1904.01561
* **[Yang 2023 OPRO]** Yang et al. Large language models as optimizers (OPRO). ICLR 2024. arXiv:2309.03409
* **[Yao 2018]** Yao, Vehtari, Simpson, Gelman. Using stacking to average Bayesian predictive distributions. *Bayesian Anal.* 2018. arXiv:1704.02030
* **[You 2017]** You, Li, Low, Lobell, Ermon. Deep Gaussian process for crop yield prediction based on remote sensing data. AAAI 2017. doi:10.1609/aaai.v31i1.11172
* **[Zaheer 2017]** Zaheer et al. Deep sets. NeurIPS 2017. arXiv:1703.06114
* **[Zevin 2016]** Zevin et al. Gravity Spy: integrating Advanced LIGO detector characterization, machine learning, and citizen science. *CQG* 2017. arXiv:1611.04596
* **[Zhang & Bloom 2019]** Zhang, Bloom. deepCR: cosmic ray rejection with deep learning. *ApJ* 2020. arXiv:1907.09500
* **[Zhang 2017 mixup]** Zhang, Cisse, Dauphin, Lopez-Paz. mixup: beyond empirical risk minimization. ICLR 2018. arXiv:1710.09412
* **[Zhao 2016]** Zhao et al. Deep learning and its applications to machine health monitoring: a survey. arXiv:1612.07640
* **[Zheng 2018]** Zheng, Aragam, Ravikumar, Xing. DAGs with NO TEARS: continuous optimization for structure learning. NeurIPS 2018. arXiv:1803.01422
* **[Zhu & Beroza 2018]** Zhu, Beroza. PhaseNet: a deep-neural-network-based seismic arrival time picking method. *GJI* 2019. arXiv:1803.03211
* **[Zhu 2018 denoise]** Zhu, Mousavi, Beroza. Seismic signal denoising and decomposition using deep neural networks (DeepDenoiser). *IEEE TGRS* 2019. arXiv:1811.02695
* **[Zwolak & Taylor 2021]** Zwolak, Taylor. Colloquium: advances in automation of quantum dot devices control. *Rev. Mod. Phys.* 2023. arXiv:2112.09362
