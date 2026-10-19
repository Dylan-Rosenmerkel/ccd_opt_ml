# Newer models for the skipper-CCD noise problem: a literature review

*Written 2026-10-19 for `ccd_opt_ml`. It covers Level 1 (predict σe, λ and gain from traces and settings) and Level 2 (suggest
the next settings). Every reference below was found and checked during this search. Anything I could not fully verify is
marked **[unverified]**. The package notes reflect the state of things at the time of writing.*

---

## 0. The problem, and what any new method has to cope with

| Property of our data | What it means for a method |
|---|---|
| 13 to ~100 packets × 3-4 amplifiers = **40-400 labelled rows** | Deep models trained from scratch are at a disadvantage. Strong priors (physics, GP kernels, pretrained priors) are what wins. |
| Rows of one packet are **correlated** (same CCD state) | Scores and conformal calibration need packet-grouped data. Plain conformal assumes exchangeable *rows*, but only the *packets* are exchangeable here. |
| **Unlabelled** traces exist (reference traces, a few × the labelled ones) | Self-supervised pretraining, pretrained time-series encoders, or a multi-fidelity "cheap task" can use them. |
| Inputs: ~61 settings, 85 + 85 trace and reference features, 14 curves × 256, 10 trace sets | That is ~300 scalar inputs, more than there are rows. A per-input ARD GP fitted by maximum likelihood is at risk of overfitting. |
| Labels have **known errors** (σe 1-2 %, gain 0.5-1 %, λ larger) | Use fixed, known heteroscedastic noise in the likelihood. No model can be judged below the label error. |
| Need **calibrated** error bars for Level 2 / 3 | This favours models with a coherent predictive distribution (GPs, TabPFN, ensembles), followed by conformal calibration. |
| No amplifier identity as input | Methods that need a task ID (multi-task GPs, neural processes keyed by amplifier) can only be references, or must infer the "task" from the trace. |

**A point that matters for Level 2.** At Level 1 the trace *at the new settings* is an input. When Level 2 proposes
settings that have not been run yet, that trace does not exist. A Level-2 surrogate therefore has to be either a function of
the settings (plus packet state) alone, or a two-stage model: settings → trace quantities → image quantities. Several
methods below (multi-output and multi-fidelity GPs, function-network BO) target exactly this two-stage structure.

---

## 1. Tabular foundation models and in-context learners

### 1.1 TabPFN v2 / TabPFN-2.5 (Prior Labs)

**What it is (plain language).** TabPFN is a transformer pretrained *only on millions of synthetic tables* that were drawn
from a broad prior over data-generating processes (random causal graphs, noise, and so on). At use time you do not train it
in the usual sense. You give it the training table as "context" and the test rows as "queries", and in one forward pass it
outputs a predictive distribution for each query. The network has learned to approximate Bayesian inference under that
prior: it outputs an approximation of the posterior predictive distribution

$$q_\phi(y \mid x_\text{test}, D_\text{train}) \approx p(y \mid x_\text{test}, D_\text{train}) = \int p(y \mid x_\text{test}, f)\,p(f \mid D_\text{train})\,df .$$

For regression the output is a histogram over many bins (a "bar distribution"). From it you get the mean, median,
quantiles, or the full density.

**Evidence.**
* Hollmann et al. (2025, *Nature* 637:319) report that TabPFN v2 dominates tuned baselines on datasets of up to 10 000 rows,
  using far less compute.
* The TabPFN-2.5 report (Prior Labs, arXiv:2511.08667) extends this to 50 000 rows and 2 000 features. It reports an 85 %
  win rate against default XGBoost on regression for the larger datasets.
* The package's `TabPFNRegressor.predict(X, output_type="quantiles" | "full" | "main")` returns quantiles and the full bar
  distribution (Prior Labs docs).

**Counter-evidence that matters for us.** Johnson et al. (2026, arXiv:2606.01427) compared TabPFN-2.5 with default GPs on
engineering regression problems with 4-40 inputs. They found that **GPs often give better accuracy and uncertainty when data
are scarce**, especially when the kernel is a good prior. TabPFN's *intervals* stayed well behaved even when its means were
less accurate. Small-scale coverage checks with split conformal around TabPFN reach nominal marginal coverage, with
undercoverage for some subgroups (e.g. arXiv:2606.30702). These are application papers, not a systematic benchmark.

**Why it may suit us.**
* We have ~300 inputs and 40-400 rows, which is well inside its range.
* It needs no hyperparameter tuning, and nothing it learned came from our test packets.
* It gives a full predictive distribution, which is what conformal calibration and Level 2 need.
* It is a natural replacement for the *learned* parts of model A: the gain GP $\hat g(x)$ and the residual GP $\hat r(x)$.
  The Allan physics stays in place.

**Risks.**
* Its prior is generic, not physics.
* With <100 rows a well-specified GP may beat it (see Johnson et al. above).
* Context rows are treated as i.i.d. That is harmless for prediction, but it does not exploit the packet structure.
* Licences: the TabPFN v2 weights are under the "Prior Labs License" (open, commercial use with attribution). The 2.5 / 2.6 /
  3 weights are **gated and non-commercial**, which is fine for research. The newest package defaults to the newest model,
  so pin the version.
* It runs on CPU for our sizes. Prior Labs says CPU is only for small data, which ours is.

**Effort:** very low, about 30 lines (`pip install tabpfn`, scikit-learn API). **Mature:** yes, actively maintained.
**Priority: HIGH.**

### 1.2 Other in-context tabular models (TabICL / TabICLv2, TabDPT, LimiX)

* **TabICL** (Qu, Holzmüller, Varoquaux, Le Morvan; ICML 2025, arXiv:2502.05564) was classification-only.
  **TabICLv2** (arXiv:2602.11139, 2026) adds regression and ties RealTabPFN-2.5 on the TabArena regression subset, by the
  authors' own report. It is open (`soda-inria/tabicl`).
* **TabDPT** (Ma et al., NeurIPS 2025, arXiv:2410.18164) is pretrained on real tables. TabICLv2's authors raise a possible
  benchmark-leakage concern about its regression results.
* **LimiX** (arXiv:2509.03505) and LimiX-2 (arXiv:2609.17488) report top TabArena regression scores (self-reported).
* All of these papers are written by the methods' own authors, and the rankings change with the benchmark.

**For us:** use them as a second opinion to TabPFN in the same harness. Do not expect a big difference between them at
n ≈ 100. **Effort:** low. **Priority: MEDIUM** (run TabICLv2 alongside TabPFN; skip the rest unless TabPFN looks very promising).

### 1.3 PFNs as Bayesian-optimization surrogates

**PFNs4BO** (Müller, Feurer, Hollmann, Hutter; ICML 2023, arXiv:2305.17535) trains PFNs to imitate a GP, an "advanced" GP,
or a Bayesian neural network, and uses them as BO surrogates. It can include user priors about where the optimum is.
**FIRE** (Yu, Sung, Ahmed; arXiv:2601.22371, 2026) does *training-free multi-fidelity regression* with tabular foundation
models: a high-fidelity model is conditioned on the low-fidelity model's predictive distribution. Our "trace = cheap,
image = expensive" pair is exactly this setting (see §7).
**Priority: LOW for Level 1, MEDIUM to watch for Level 2.**

---

## 2. Time-series foundation models and pretrained encoders for raw traces

Our traces are short 1D sensor records: per pixel, a sequence of 500 skips. They come with 14 derived curves (Allan
deviation, PSDs, ...) of 256 points each. The task, "one scalar per series", is called **time-series extrinsic regression
(TSER)** in the literature.

### 2.1 Random-convolution features (ROCKET / MiniRocket): the strong cheap baseline

**What it is.** Convolve each series with thousands of *random* 1D kernels (random lengths, dilations and biases). Keep two
summary numbers per kernel: the maximum, and the "proportion of positive values". Then fit a linear model on these features.
Nothing is learned in the feature step.
* ROCKET: Dempster, Petitjean, Webb, *DMKD* 34 (2020), arXiv:1910.13051.
* MiniRocket: Dempster, Schmidt, Webb, KDD 2021, arXiv:2012.08791. It is almost deterministic and up to 75× faster.

**Evidence for regression.**
* Tan, Bergmeir, Petitjean, Webb, "Time series extrinsic regression" (*DMKD* 35, 2021, arXiv:2006.12672): ROCKET adapted
  to regression was the most accurate of the methods tested on 19 TSER datasets, ahead of XGBoost, random forests and SVR.
* Guijo-Rubio et al. (*DMKD* 38:2141, 2024, arXiv:2305.01429) extended the archive to 63 problems. Feature-based pipelines
  did well there (FreshPRINCE: many summary statistics + rotation forest; DrCIF: random-interval features + trees), and a
  rotation forest was hard to beat.

**Why it may suit us.**
* Nothing needs training except a ridge or GP head, so it cannot overfit 100 labels the way a CNN can.
* It works on the raw 500-skip sequences, so it could find effects our 85 hand-made features miss.

**Risks.**
* ~10 000 features with 100 rows makes the head (ridge, PLS, or GP after PCA) carry all the regularisation.
* Random features do not know the noise physics: noise is a *second-order* property of the trace, and ROCKET's "max" and
  "proportion of positive values" capture variance only indirectly. Hand-made Allan and PSD features may already be close to
  optimal for σe.

**Packages:** `aeon` and `sktime` (MiniRocket transformers, ROCKET regressors). **Mature:** yes.
**Effort:** low, *but* the raw skip sequences would have to be written to `model_inputs` (today only curves and features
are). **Priority: MEDIUM-HIGH**, as the baseline any deep trace encoder must beat.

### 2.2 Pretrained time-series foundation models (frozen encoders)

* **MOMENT** (Goswami et al., ICML 2024, arXiv:2402.03885): a family of open T5-style encoders pretrained by masked
  reconstruction on the "Time-series Pile". It is meant for forecasting, classification, anomaly detection and imputation
  with little task-specific data (`momentfm`, Hugging Face). I believe it uses fixed 512-sample input windows, which would
  fit our 500 skips with little padding. **[unverified detail]**
* **Chronos** (Ansari et al., *TMLR* 2024, arXiv:2403.07815) tokenizes scaled values and trains T5 language models.
  **Chronos-2** (arXiv:2510.15821, 2025) is a 120M-parameter encoder with group attention over related series and covariates.
* **TimesFM** (Das, Kong, Sen, Zhou; ICML 2024, arXiv:2310.10688) is a decoder-only forecaster.
* **Mantis** (Feofanov et al., arXiv:2502.15637, 2025) is a ViT-style model pretrained *contrastively for classification*.
  It reports the best accuracy and lowest calibration error among the foundation models it compared, frozen or fine-tuned.
* Auer, Klotz, Böck, Hochreiter (NeurIPS 2025 workshop, arXiv:2510.26777) found that *frozen forecasting* models are strong
  zero-shot feature extractors for classification. A 2026 study using frozen Chronos-2 features for remaining-useful-life
  regression (arXiv:2606.11990) warns that naive use often underperforms specialised models.

**Fit to our problem.** These models were pretrained on forecasting-type data (demand, weather, physiology), where the
*shape* of the signal is the information. In our traces the information is the **noise statistics**: white versus 1/f
level, pickup lines, pedestal steps, and how they change with the CDS window. Forecasting pretraining tends to smooth noise
away, and per-series scaling (Chronos' mean scaling, MOMENT's RevIN-style normalisation) can **remove the absolute
noise level**, which is exactly what we need. It would have to be fed back as an extra scalar.

**Priority: MEDIUM-LOW.** Try one frozen model (Mantis or MOMENT) in the same "embedding → ridge or GP" head as MiniRocket.
Do not fine-tune with 100 labels. **Effort:** medium (GPU helps; the packages exist; data plumbing for the raw sequences).

### 2.3 Self-supervised pretraining on our own unlabelled traces (what model D already does)

The relevant literature for model D:
* **TS2Vec** (Yue et al., AAAI 2022, arXiv:2106.10466): hierarchical contrastive learning per timestamp.
* **TF-C** (Zhang, Zhao, Tsiligkaridis, Zitnik; NeurIPS 2022, arXiv:2206.08496): time-domain and frequency-domain
  embeddings of the same series are made to agree. This is attractive here because our physics lives in the PSD.
* **PatchTST** (Nie et al., ICLR 2023, arXiv:2211.14730): masked-patch pretraining with a transformer.

Our D model already combines masked reconstruction, settings prediction and InfoNCE. The literature suggests two concrete
upgrades:
1. A **time-frequency consistency** loss (TF-C) between the Allan/PSD curves and the raw sequence.
2. **Augmentations that do not change the label**: a pixel shuffle within a trace, time reversal of a skip sequence, and
   adding a constant pedestal are all noise-preserving. Scaling the amplitude is *not* noise-preserving (it changes σ in ADU).

Our "±5 % scale per curve" augmentation in InfoNCE teaches the encoder to *ignore* a 5 % level change, which is about the
accuracy we want. That is worth a check.
**Priority: MEDIUM** (a modest change to existing code).

---

## 3. Neural processes, meta-learning and multi-task GPs (few-shot across amplifiers and boards)

**Plain language.** Treat each amplifier (or board) as a separate "task": a related but different function from settings
to noise. A meta-learner trains on many tasks so that, for a new amplifier, a handful of points are enough.
* **Multi-task GPs** (Bonilla, Chai, Williams, NeurIPS 2007) do this with a kernel
  $k((x,i),(x',j)) = B_{ij}\,k_x(x,x')$, where $B$ is a learned task covariance (the "intrinsic coregionalisation model";
  model B is a hierarchical variant).
* **Conditional / neural processes** (Garnelo et al., ICML 2018) learn a network that maps a context set
  $\{(x_c, y_c)\}$ plus a query $x$ to a predictive distribution. Transformer NPs (Nguyen & Grover, ICML 2022) are the
  stronger modern variant. Survey: Jha et al., arXiv:2209.00517.
* **Deep kernel transfer** (Patacchiola et al., NeurIPS 2020, arXiv:1910.05199) meta-learns a deep kernel across tasks.
* **PACOH / F-PACOH** (Rothfuss et al., ICML 2021, arXiv:2002.05551; NeurIPS 2021) meta-learn **GP priors** with
  PAC-Bayes regularisation. F-PACOH keeps epistemic uncertainty high where the meta-training data were thin, and was shown
  to work well for BO.
* **Few-shot BO with deep kernels** (Wistuba & Grabocka, ICLR 2021, arXiv:2101.07667).

**Fit to our problem.** These methods need **many tasks** for meta-training: tens to thousands of related functions. We
have ~4 amplifiers on one board, 2-3 boards eventually, and no amplifier ID as input. A neural process with "context = this
packet's other amplifiers" is conceivable, but with ≤100 packets it would overfit.

**When they become useful:** at Level 2/3, when *a new module or board* is installed. A GP prior meta-learned on the old
boards (F-PACOH-style), or simply a multi-task GP with a board index, would let the optimiser start from an informed prior
instead of from scratch.

**Priority:** LOW for Level 1, MEDIUM for later transfer across boards. **Packages:** GPyTorch/BoTorch (`MultiTaskGP`,
`KroneckerMultiTaskGP`) are mature. NP and PACOH code are research repos (not mature).

---

## 4. Physics-informed / grey-box models and simulation-based inference

### 4.1 Grey-box GPs (what model A is) and physics prior means

Model A is already a grey-box model: the Allan physics gives the noise in ADU, and GPs learn the gain and a residual. The
literature supports this direction:
* Hanuka et al. (arXiv:1911.01538; arXiv:2009.03566) built "physics-informed GPs" for accelerator tuning. The kernel or
  covariance comes from a physics model, and it beat data-trained GPs on SPEAR3.
* Boltz et al. (arXiv:2403.03225, 2024) use neural networks trained on simulations as **GP prior means** for accelerator BO.
* Reviews: Karniadakis et al., "Physics-informed machine learning", *Nat. Rev. Phys.* 3:422 (2021); von Rüden et al.,
  "Informed machine learning", *IEEE TKDE* 35:614 (2023).

**A specific physics ingredient worth adding.** Wüthrich, Fazan, MacBride, Soares-Santos ("Skipper CCD readout time
optimization for astronomical applications", Proc. SPIE 14157, 141572K, 2026; arXiv:2607.20126) model the trade-off between
CDS integration time and number of skips *from the sensor's noise PSD*. They verify it on an Oscura-type skipper CCD and
find that the optimum is governed by the per-sample charge-transfer time.

The standard CDS filter view (textbook, not taken from that paper) has the form

$$\sigma_1^2(\tau) \;\propto\; \int_0^\infty S_v(f)\,|H_\text{CDS}(f;\tau)|^2\,df,\qquad |H_\text{CDS}(f;\tau)|^2 = \frac{4\sin^4(\pi f\tau)}{(\pi f\tau)^2}$$

(two adjacent boxcar windows of length τ, subtracted). Here $S_v(f)$ is the video noise PSD and τ the integration window
(our `delayInt`).

**Why it helps:**
* A feature $\hat\sigma_1(\tau)$ computed from the *reference* trace PSD at the *candidate* delayInt is a physics prediction
  that does not need a trace at the new setting. That is exactly what Level 2 lacks (see §0).
* At Level 1 it gives model A's residual GP a better baseline.

Chierchie et al. ("Detailed modeling of the video signal and optimal readout of CCDs", *Int. J. Circuit Theory Appl.*,
DOI 10.1002/cta.2784; Fermilab-PUB-20-319) derive an optimal filter that includes charge-transfer and feedthrough effects.
That is a more complete version of the same idea.

**Priority: HIGH** for adding the PSD-filtered noise prediction as a model-A feature or prior mean. The effort is modest if
the PSD curves are already stored, which they are (14 curves).

### 4.2 Simulation-based inference (neural posterior estimation)

**What it is.** If you can *simulate* data $x$ from parameters $\theta$ but cannot write the likelihood, train a
conditional density network $q_\phi(\theta \mid x)$ on simulated pairs, $\min_\phi \mathbb E_{\theta,x}[-\log q_\phi(\theta\mid x)]$.
For real data it then gives an approximate posterior in one forward pass. Review: Cranmer, Brehmer, Louppe, *PNAS* 117:30055
(2020). Package: `sbi` (Boelts et al., *JOSS* 10:7754, 2025; arXiv:2411.17337), PyTorch, mature.

**Fit to our problem.**
* SBI is *not* a direct replacement for the Level-1 regressors. The mapping settings → image noise depends on a CCD whose
  physics we cannot simulate faithfully, and that mapping is what we are learning.
* It **could replace the weighted-least-squares Allan fit** inside model A. Simulate skip sequences with white + 1/f +
  pickup + pedestal steps (our toy world `src/ml/synthetic.py` already does this), and train NPE to infer
  $(a, b, f_\text{corner}, \dots)$ with posteriors from a trace's curves. That turns "a, b measured on the trace" into
  calibrated distributions, which propagate into σe's error bar.
* I found no SBI work on CCD readout noise. The nearest is gravitational-wave work that handles real detector noise and PSD
  uncertainty (e.g. arXiv:2406.03935; review arXiv:2507.11192).

**Risk:** a simulator mismatch on real traces (pickup and drift that the toy world does not reproduce).
**Effort:** medium. **Priority: MEDIUM-LOW** (worth it only if the Allan-fit uncertainty turns out to dominate the σe error).

---

## 5. Better uncertainty: probabilistic models and calibration

### 5.1 Model-side uncertainty

| Method | One-line idea | Fit to our data | Package / maturity | Priority |
|---|---|---|---|---|
| **Deep ensembles** (Lakshminarayanan et al., NeurIPS 2017; already in the guide) | train M nets; mean ± spread | already used (NN, CNN) | — | (have) |
| **Bayesian last layer, VBLL** (Harrison, Willes, Snoek; ICLR 2024, arXiv:2404.11599) | the network's last layer is Bayesian linear regression, $y = w^\top\phi(x)+\epsilon$, $w\sim\mathcal N(\mu,\Sigma)$; closed-form predictive variance $\phi^\top\Sigma\phi + \sigma^2$ | cheap epistemic uncertainty for the CNN or the pretrained encoder; close to what model C does with a GP head | `vbll` (PyPI, young) | MEDIUM-LOW |
| **Laplace approximation** (Daxberger et al., NeurIPS 2021, arXiv:2106.14806) | Gaussian posterior around the trained weights, using curvature | post-hoc on the CNNs, last-layer variant | `laplace-torch` (moderately mature) | LOW-MEDIUM |
| **SNGP** (Liu et al., NeurIPS 2020, arXiv:2006.10108) | spectral normalisation + GP output layer, so uncertainty grows with distance from the training data | good for flagging novel settings (Level 3 safety); single model | official code in TensorFlow; PyTorch ports are unofficial | LOW-MEDIUM |
| **Deep evidential regression** (Amini et al., NeurIPS 2020, arXiv:1910.02600) | the network outputs Normal-Inverse-Gamma parameters | Meinert, Gawlikowski, Lavin (AAAI 2023, arXiv:2205.10060) show the uncertainty is a heuristic, not exact | `evidential-deep-learning` | **LOW (avoid)** |
| **NGBoost** (Duan et al., ICML 2020, arXiv:1910.03225) | boosting of the parameters of a distribution (μ, log σ) via natural gradients | gives the trees family a real σ; trees are weak here anyway | `ngboost` (mature) | LOW |
| **Quantile regression forests** (Meinshausen, JMLR 7:983, 2006) | keep all leaf values; conditional quantiles from them | same remark as NGBoost | `quantile-forest` (sklearn-compatible) | LOW |

### 5.2 Calibration-side: conformal methods that respect packets

Our current method computes $z_i = |y_i-\mu_i|/s_i$ on out-of-fold rows and scales the error bars by the empirical quantile.
This is "normalised" conformal in cross-validation form, close to **CV+** (Barber, Candès, Ramdas, Tibshirani, *Ann. Stat.*
49:486, 2021, arXiv:1905.02928).

**The gap.** The guarantee assumes the calibration *rows* are exchangeable with the new row. In our data only the
**packets** are. Two papers treat exactly this two-level structure:
* **Dunn, Wasserman, Ramdas**, "Distribution-free prediction sets for two-layer hierarchical models", *JASA* 118:2491
  (2023), arXiv:1809.07441.
  * *CDF pooling*: weight each group equally, i.e. each row by $1/n_\text{packet}$. It is recommended when asymptotic
    validity is enough.
  * *Repeated subsampling*: take one row per group, compute the quantile, and repeat. It is recommended when a finite-sample
    guarantee is needed.
* **Lee, Barber, Willett**, "Distribution-free inference with hierarchical data", arXiv:2306.06342. It defines a
  hierarchical exchangeability and extends conformal and jackknife+ to it, with a stronger "second-moment" coverage.
  (I could not confirm a peer-reviewed venue.)

**Useful refinements:**
* **Conformalized quantile regression (CQR)** (Romano, Patterson, Candès; NeurIPS 2019, arXiv:1905.03222). Calibrate
  quantile outputs, e.g. TabPFN's 16/84 % quantiles or the trees' quantiles, instead of μ ± s. It adapts to heteroscedasticity.
* **Mondrian conformal**: compute a separate factor within each category. Categories can be bins of the model's own
  predicted σ ("difficulty"), which needs no amplifier ID. See Boström, Johansson, Löfström, COPA 2021 (PMLR 152) and the
  `crepes` package (Boström, COPA 2022, PMLR 179; mature, model-agnostic). Mondrian *by amplifier* would give per-amplifier
  coverage, but it uses the amplifier identity at calibration time. That is allowed only if we decide calibration may know the
  channel (the model itself still would not).

**Priority: HIGH**, because it is cheap and directly addresses the "calibrated uncertainty" requirement. Use packet-weighted
quantiles, and report worst-amplifier coverage, not only the average. **Effort:** tens of lines in `scores()` /
`calibrate.py`. `MAPIE` and `crepes` are mature alternatives, but our own grouped implementation is simpler.

---

## 6. Set and attention models for the repeated trace sets

**Set Transformer** (Lee, Lee, Kim, Kosiorek, Choi, Teh; ICML 2019, arXiv:1810.00825) replaces the DeepSets sum or mean
with self-attention between set elements and *pooling by multi-head attention* (PMA). PMA lets the model learn to
down-weight a damaged set (pickup, glitch) by comparing it with the others, which is what model E is for. DeepSets can only
weight each set independently.

**Fit:** we have 10 sets per example and 40-400 examples. A one-layer attention pool with a handful of heads has few
parameters, but the gain over model E (5.1 % vs 5.5 % σe in the toy world) leaves little room. A **robust, non-learned
combiner** (median or trimmed mean of per-set Allan values, or a weight from each set's agreement with the others) is a
cheaper baseline to compare first.
**Priority: LOW-MEDIUM. Effort:** low (≈50 lines of PyTorch). No package needed.

---

## 7. Multi-output and multi-fidelity GPs, and their link to Level 2

### 7.1 Multi-output GPs over the targets (and over trace quantities)

σe, λ and the gain are correlated through shared settings (e.g. RG/SW voltages affect both CIC and noise). An ICM/LMC
multi-output GP (Bonilla et al. 2007 kernel above) shares statistical strength between the targets.

The bigger win: **add the trace-measured noise in ADU as an extra output**. It is observed on *every* trace, including the
unlabelled reference traces and future trace-only packets, so the "cheap" output has many more points than the image
labels.

This is classical co-kriging: $f_\text{img}(\theta) = \rho\,f_\text{trace}(\theta) + \delta(\theta)$ (Kennedy & O'Hagan
2000, in the guide). A nonlinear version is NARGP (Perdikaris et al., *Proc. R. Soc. A* 473:20160751, 2017).
**Packages:** BoTorch/GPyTorch (`MultiTaskGP`, `SingleTaskMultiFidelityGP`, mature); Emukit (linear and nonlinear
multi-fidelity models; works, less actively developed).

**Priority: MEDIUM-HIGH.** It is the most natural way to (a) use the unlabelled traces without a neural encoder, and (b) get
a settings-only surrogate for Level 2.

### 7.2 Multi-fidelity, multi-information-source and function-network BO

* **Multi-information-source optimisation (misoKG)**: Poloczek, Wang, Frazier, NeurIPS 2017, arXiv:1603.00389. A cheap
  source may be *biased*, not just noisier. That is our case: the trace misses slow noise and the gain.
* **Multi-fidelity knowledge gradient**: Wu, Toscano-Palmerin, Frazier, Wilson, UAI 2019, arXiv:1903.04703 (in BoTorch).
  **MF-MES**: Takeno et al., ICML 2020, arXiv:1901.08275.
* **BO of composite functions** (Astudillo & Frazier, ICML 2019, arXiv:1906.01537). Model the vector
  $h(\theta) = (\log\sigma_e, \log\lambda)$ with a multi-output GP and optimise a *known* cheap function
  $J = g(h(\theta))$. This is our decomposed surrogate (model F), with a principled acquisition (EI-CF).
* **BO of function networks** (Astudillo & Frazier, NeurIPS 2021) and **with partial evaluations** (Buathong, Wan,
  Astudillo, Daulton, Balandat, Frazier; ICML 2024, arXiv:2311.02146). The objective is a network of nodes
  (settings → trace quantities → image quantities). The cost-aware acquisition decides *which node to evaluate*: a ~1-minute
  trace-only packet or a ~16-minute full packet. **This is the closest match in the literature to our Level-2 design.**
* Tabular-foundation-model multi-fidelity: FIRE (arXiv:2601.22371), §1.3.

### 7.3 Safe and constrained BO

* **SafeOpt** (Sui, Gotovos, Burdick, Krause; ICML 2015). Only evaluate where a GP lower confidence bound on a safety
  function is above a threshold, and expand the safe set gradually. Berkenkamp, Krause, Schoellig (*Mach. Learn.* 2023,
  arXiv:1602.04450) extend it to multiple safety constraints that are separate from the objective.
* **LineBO** (Kirschner, Mutný, Hiller, Ischebeck, Krause; ICML 2019, arXiv:1902.03229) does safe BO in high dimension by
  optimising along 1-D lines. It was demonstrated on SwissFEL with up to 40 parameters.
* **Step-size-limited safe BO** (Kirschner et al., *PRAB* 25:062802, 2022, arXiv:2203.13968) tuned SwissFEL and HIPA with
  up to 16 parameters and >200 constraints. It matches our "one `max_step` per change" rule closely.
* Constrained BO with *unknown* constraints: Gardner et al., ICML 2014 (EI × P(feasible)). The high-dimensional version is
  **SCBO** (Eriksson & Poloczek, AISTATS 2021, arXiv:2002.08526; in BoTorch).

### 7.4 GP priors when there are many inputs and few points

* **SAASBO** (Eriksson & Jankowiak, UAI 2021, arXiv:2103.00349) puts a sparsity-inducing half-Cauchy prior on the inverse
  length scales and uses full Bayesian inference (NUTS). Most inputs are switched off unless the data demand them. It is in
  BoTorch as `SaasFullyBayesianSingleTaskGP`.
* **Dimension-scaled length-scale priors** (Hvarfner, Hellsten, Nardi; ICML 2024, arXiv:2402.02229) scale the GP length
  scale prior with $\sqrt d$. They show that vanilla BO then works well in high dimension. I believe recent BoTorch defaults
  adopt this prior **[unverified: check the installed version]**.

**Relevance:** our `models.GP` fits one ARD length scale per input by maximum likelihood, with ~300 inputs and ≤400 rows.
Both priors address exactly this over-parameterisation. They also matter for Level 2, with 61 settings.
**Priority: HIGH** (the same change improves Level 1 GPs and Level 2 surrogates).

### 7.5 Conformal + BO

* **Conformal BO** (Stanton, Maddox, Wilson; AISTATS 2023, arXiv:2210.12496) steers queries towards regions where the
  model's predictions keep guaranteed coverage.
* **Feedback covariate shift** (Fannjiang, Bates, Angelopoulos, Listgarten, Jordan; *PNAS* 119:e2204569119, 2022,
  arXiv:2202.03613). When the model chooses the next settings, the test point depends on the training data, and ordinary
  conformal coverage breaks. This paper gives a valid correction.
* Weighted conformal under covariate shift: Tibshirani, Barber, Candès, Ramdas, NeurIPS 2019, arXiv:1904.06019.

These matter for Level 2/3. Our conformal factor is measured on *hand-chosen* packets, and it will be optimistic for
optimiser-chosen packets at the edge of the explored region.

---

## 8. Prior work on ML and automation for CCD / detector / accelerator tuning

| Work | What | Relevance |
|---|---|---|
| **Cuevas-Zepeda, Chavez, Estrada, Noonan, Nord, Saffold, Sofo-Haro, Spinola e Castro, Trivedi**, "Automating Sensor Characterization with Bayesian Optimization", arXiv:2509.21661 (Fermilab, 2025/26). The OSTI record lists it as **JINST 21 P10012 (2026)** [title match on the journal page not fully confirmed]. | Closed-loop BO on a SiSeRO CCD over 5 parameters (V_DS, V_GS, offset, isolation guard voltage, CDS samples). The objective is 1/SNR plus empirical penalties (negative gain, saturation, non-uniformity, CIC). 150 random points, then 350 BO points. The text says "<10 h" in one place and "a couple of days" in another. It also says it was validated on a skipper CCD (no details). | The **closest prior work**. It is image-based single-fidelity BO with soft penalties: no trace surrogate, no safety model, no multi-fidelity. Our Level 1 / Level 2 design goes beyond it. |
| **Hope, Fernandez-Moroni, Gamero, Drlica-Wagner**, "Machine Learning for Skipper CCD Parameter Selection", Fermilab poster, July 2026 (indico.fnal.gov event 73552) **[seen only through a search snippet; page returned 403]** | GP + EI BO of the horizontal-clock rails (HH, HL) on a skipper CCD. The objective trades overscan noise against transfer efficiency, and the minima are reported to be repeatable. | The same group of parameters as our H-clock voltages. Worth contacting the authors. |
| **Wüthrich et al.**, SPIE 14157 (2026), arXiv:2607.20126 | An analytic optimum of (CDS integration time, number of skips) from the noise PSD | A physics prior for delayInt (§4.1) |
| **Marrufo Villalpando et al.**, "Characterization and Optimization of Skipper CCDs for the SOAR Integral Field Spectrograph", *PASP* 136:045001 (2024), arXiv:2311.00813 | Manual optimisation of readout time, voltages for full well, and CIC | Shows the expert-scan baseline and the CIC versus full-well trade-off |
| **Chierchie et al.**, smart readout, *PRL* 127:241101 (2021), arXiv:2111.09905; optimal filter, *Int. J. Circuit Theory Appl.* (DOI 10.1002/cta.2784) | Per-pixel adaptive number of skips; an optimal digital filter including transfer and feedthrough | Physics of the video chain; a source of grey-box features |
| DAMIC-M production modules, arXiv:2509.06943; DAMIC-M LBC, arXiv:2407.17872 | Manual optimisation of clock and bias voltages and timings (CIC, glow, CTI) | Context; no ML |
| **Duris et al.**, "Bayesian optimization of a free-electron laser", *PRL* 124:124801 (2020) | GP BO for LCLS. Kernel hyperparameters come from archived scans, and device correlations from a beam model. | A template for "fit the GP prior on historical packets" |
| **Roussel et al.**, "Bayesian optimization algorithms for accelerator physics", *PRAB* 27:084801 (2024), arXiv:2312.05667 | A review of BO practice: constraints, multi-objective, trust regions, priors | **The best single practical reference for Level 2** |
| **Xopt / Badger** (SLAC; JACoW / ICALEPCS 2025 proceedings) | Open Python BO stack for live control rooms, including trust-region BO and constraints | A candidate engine for Levels 2/3 instead of in-house BO code |
| **Moon et al.**, "Machine learning enables completely automatic tuning of a quantum device faster than human experts", *Nat. Commun.* 11:4161 (2020) | Autonomous gate-voltage tuning in up to 8 dimensions | Analogous to Level 3: autonomous scans of a sensitive device |
| Bernardes et al., BO of EMCCD operating modes for SPARC4, *JATIS* 7:038001 (2021) (cited by Cuevas-Zepeda et al.) | BO among EMCCD readout modes | Older precedent in astronomy detectors |

**Gap:** I found **no published work** that predicts skipper-CCD *image* noise, dark current or CIC from short readout
traces, nor any that uses traces as a cheap fidelity. LSST/Rubin sensor characterisation (crosstalk, residual charge,
read noise) uses conventional pipelines, not learned models, in what I found. Level 1 appears to be new.

---

## 9. Ranked shortlist: what to try first

Every experiment below plugs into `lvl1/train_lvl1.py` as a new method `def <name>(self, train, test) -> (mean, std)` on
the log targets (σe, λ, gain), using the existing packet-grouped folds and `scores()`.

**Caution about the comparisons.** With 13-100 packets, differences of 1-2 percentage points are noise. Run each
comparison over several fold seeds (`src/ml/benchmark_seeds.py` exists) and look at the **paired per-packet difference**
in log error against `physics`, with a bootstrap over packets.

### 1. TabPFN (and TabICLv2) as a regressor, and inside model A (**HIGH**)

* **`tabpfn`**: `xtr, xte = self._features(train, test)` (optionally add `packet_state`). For each target, fit
  `TabPFNRegressor` on the rows with finite labels, then call `predict(xte, output_type="quantiles", quantiles=[0.16, 0.5, 0.84])`.
  Set mean = median and std = (q84 − q16)/2. Pin `ModelVersion.V2` (open weights) and also try 2.5.
* **`physics_tabpfn`**: model A with the gain GP $\hat g(x)$ and the residual GP $\hat r(x)$ replaced by TabPFN fitted on
  the same targets. Then $s^2_{\log\sigma_e} = s^2_g + s^2_r$ as before.
* **Read off:** error / shape / rank / factor against `gp`, `physics` and `physics_hier`. Also time per fold on CPU.
* **Expectation:** it beats `ridge`, `trees`, `gp` and `nn` on the features family. Whether it beats the GPs inside model A
  is open. Johnson et al. (2026) found GPs ahead when data are scarce, so test both at 13, 40 and 100 packets (subsample the
  synthetic set).

### 2. Packet-aware conformal calibration + Mondrian-by-difficulty + CQR (**HIGH**, smallest effort)

* In `scores()` (and `calibrate.Conformal`), replace `conformal_q(z[other], c)` with a **weighted quantile, weight
  $1/n_\text{packet}$ per row** (CDF pooling, Dunn et al. 2023). Add a **repeated one-row-per-packet subsampling** variant
  (finite-sample valid).
* Add **Mondrian by predicted-σ tercile** (no amplifier ID needed), e.g. with `crepes`. For quantile models (TabPFN,
  trees), add **CQR**.
* **Report** the worst-amplifier and worst-board coverage and the mean interval width, not just the average coverage. The
  pass mark should be "every amplifier within ±0.1 of 0.68 / 0.95".

### 3. Sparse / well-specified GP priors: SAAS and dimension-scaled length scales (**HIGH**)

* **`gp_saas`**: BoTorch `SaasFullyBayesianSingleTaskGP` on the same standardized features, with known label noise
  `train_Yvar = Y_err**2` (check that your BoTorch version accepts it). Use NUTS with modest settings (e.g. 256 warm-up,
  128 samples, thinning 16).
* **`gp_dimprior`**: `SingleTaskGP` with a $\sqrt d$-scaled log-normal length-scale prior (Hvarfner et al. 2024).
* Swap these in for the GPs inside `physics` too (`physics_saas`). Check which inputs SAAS keeps: that is a free
  feature-relevance analysis, and Level 2 uses the same surrogate.

### 4. Trace-representation bake-off: MiniRocket vs a frozen foundation model vs our pretrained encoder (**MEDIUM-HIGH**)

* **Prerequisite:** write the raw per-pixel skip sequences (500 skips) to a new `model_inputs/raw.npz`. That is a later
  code change, not made here.
* **`minirocket`**:
  * fit `aeon`/`sktime` MiniRocket on the *unlabelled* traces (allowed, since it uses no labels);
  * transform each example's sequences and average over pixels and sets;
  * add log-variance of the sequences explicitly (MiniRocket is scale-sensitive only indirectly);
  * head: `RidgeCV` (std from the leave-one-out residuals of the training folds), or PCA(20-50) → GP.
* **`mantis_frozen`** / **`moment_frozen`**: the same head on frozen embeddings, with the per-series scale fed back as a
  scalar.
* **Compare** with `cnn`, `pretrained` and `deep_kernel`. If none beats the hand-made features plus model A, stop investing
  in deep trace encoders.

### 5. Multi-output GP with the trace noise as a cheap extra task (**MEDIUM-HIGH**; the bridge to Level 2)

* **`mtgp_settings`**: a GPyTorch/BoTorch ICM multi-task GP on inputs = settings + packet state (**no own-trace features**).
  * Tasks: (0) log σ_ADU from the trace's Allan fit, observed on *all* traces including reference and unlabelled ones;
    (1) log σe; (2) log λ; (3) log gain, the last three observed on labelled rows only.
  * Predict tasks 1-3 for test packets from settings alone. This is **the Level-2 question**: how well can we predict
    *before* running the settings?
  * Then condition on the test packet's own task-0 value (co-kriging). This shows how much the trace adds on top.
* **Read off:** error with versus without the trace observation. If the gap is small, cheap trace-only packets are worth
  little and Level 2 can be single-fidelity. If it is large, the function-network / multi-fidelity BO of §7.2 is justified.

**Runner-up ideas** (after the top 5):
* the PSD-filtered CDS noise feature $\hat\sigma_1(\texttt{delayInt})$ from the reference PSD (§4.1), cheap if done
  together with #1 or #3;
* TF-C-style frequency-consistency pretraining (§2.3);
* a robust set combiner or Set Transformer (§6);
* NPE for the Allan parameters (§4.2).

Things I would **not** spend time on now:
* evidential regression (unsound uncertainty);
* neural processes and meta-learning (too few tasks);
* fine-tuning large time-series foundation models on 100 labels.

---

## 10. What is most useful for Level 2 (suggesting settings)

1. **The surrogate must work without the trace at the candidate settings.** Model it as **settings → (trace quantities,
   image quantities)** with a multi-output GP (experiment #5). If trace-only packets are cheap, choose between a trace-only
   and a full packet with a cost-aware **function-network / partial-evaluation BO** acquisition (Buathong et al., ICML 2024)
   or multi-fidelity KG (Wu et al. 2019; BoTorch). This is the literature's formal version of the guide's
   "screen 3 candidates with trace-only packets" plan.
2. **Optimise a known function of modelled outputs.** $J = \overline{\log\sigma_e} + w\,\overline{\log\lambda}$ over the
   good amplifiers is a **composite objective** (Astudillo & Frazier 2019). Use a multi-output model and EI-CF / qEI on
   posterior samples of $J$, rather than a GP on $J$ itself. This matches model F's decomposed surrogate. If $w$ is
   contentious, use qNEHVI (multi-objective) in BoTorch.
3. **Use well-specified priors in 61 dimensions.**
   * SAAS or $\sqrt d$-scaled length scales (Eriksson & Jankowiak 2021; Hvarfner et al. 2024);
   * trust regions (TuRBO; already in the guide; SCBO for unknown constraints);
   * physics-informed prior means (Hanuka et al.; Boltz et al.), e.g. the PSD-based CDS noise model for delayInt;
   * kernel hyperparameters fitted on historical packets (Duris et al. 2020).
4. **Safety.**
   * Known rules (`ccd_limits.py`) → hard bounds and a candidate filter.
   * The "≤ one `max_step` per change" rule → a step-size-limited trust region (Kirschner et al., *PRAB* 2022).
   * Unknown failure modes (e.g. CIC blow-up, saturation, a lost amplifier) → a SafeOpt-style GP on a safety metric, where
     only candidates with a lower confidence bound inside the limit are allowed (Sui et al. 2015; Berkenkamp et al. 2023).
     Xopt/Badger already implement constrained and trust-region BO for control rooms.
5. **Uncertainty under the optimiser's own choices.** Conformal factors measured on hand-chosen packets are optimistic for
   optimiser-chosen ones (feedback covariate shift; Fannjiang et al., *PNAS* 2022). Either inflate the intervals at the edge
   of the explored region, or restrict proposals to regions of valid coverage (conformal BO, Stanton et al. 2023).
6. **Expect a GP-based surrogate.** TabPFN may win Level-1 accuracy, but GPs (BoTorch) remain the practical choice for
   Level 2: they give exact posterior samples, gradients for acquisition optimisation, multi-fidelity and safety tooling,
   and known-noise likelihoods. PFNs4BO-style surrogates are an option to watch, not a default.
7. **Compare with the Fermilab BO work** (Cuevas-Zepeda et al.; Hope et al.). Their objectives (1/SNR + penalties; overscan
   noise versus CTE) and convergence budgets (hundreds of image evaluations) are the natural baseline: Level 2 should need
   far fewer *image* packets.

---

## References (all found during this search)

**Tabular foundation models**
* Hollmann, Müller, Purucker, Krishnakumar, Körfer, Hoo, Schirrmeister, Hutter. Accurate predictions on small data with a tabular foundation model. *Nature* 637:319-326 (2025). doi:10.1038/s41586-024-08328-6
* Prior Labs. TabPFN-2.5: Advancing the state of the art in tabular foundation models. arXiv:2511.08667 (2025)
* Prior Labs docs: predictive distribution / regression (`output_type`), https://docs.priorlabs.ai/capabilities/predictive-distribution
* Qu, Holzmüller, Varoquaux, Le Morvan. TabICL. ICML 2025, arXiv:2502.05564; TabICLv2, arXiv:2602.11139 (2026)
* Ma et al. TabDPT. NeurIPS 2025, arXiv:2410.18164
* LimiX, arXiv:2509.03505 (2025); LimiX-2, arXiv:2609.17488 (2026)
* Johnson, Ben-Jacob, Negarandeh, Vendrell-Gallart, Bostanabad. On the uncertainty quantification ability of tabular foundation models. arXiv:2606.01427 (2026)
* Müller, Feurer, Hollmann, Hutter. PFNs4BO. ICML 2023, arXiv:2305.17535
* Yu, Sung, Ahmed. FIRE: multi-fidelity regression with tabular foundation models. arXiv:2601.22371 (2026)

**Time series**
* Dempster, Petitjean, Webb. ROCKET. *DMKD* 34 (2020), arXiv:1910.13051
* Dempster, Schmidt, Webb. MiniRocket. KDD 2021, arXiv:2012.08791
* Tan, Bergmeir, Petitjean, Webb. Time series extrinsic regression. *DMKD* 35 (2021), arXiv:2006.12672
* Guijo-Rubio, Middlehurst, Arcencio, Silva, Bagnall. Unsupervised feature based algorithms for TSER. *DMKD* 38:2141 (2024), arXiv:2305.01429
* Goswami et al. MOMENT. ICML 2024, arXiv:2402.03885
* Ansari et al. Chronos. *TMLR* 2024, arXiv:2403.07815; Chronos-2, arXiv:2510.15821
* Das, Kong, Sen, Zhou. TimesFM. ICML 2024, arXiv:2310.10688
* Feofanov et al. Mantis. arXiv:2502.15637 (2025)
* Auer, Klotz, Böck, Hochreiter. Pre-trained forecasting models: strong zero-shot feature extractors for TSC. arXiv:2510.26777 (2025)
* Time-series foundation model embeddings for remaining useful life estimation, arXiv:2606.11990 (2026)
* Yue et al. TS2Vec. AAAI 2022, arXiv:2106.10466
* Zhang, Zhao, Tsiligkaridis, Zitnik. TF-C. NeurIPS 2022, arXiv:2206.08496
* Nie, Nguyen, Sinthong, Kalagnanam. PatchTST. ICLR 2023, arXiv:2211.14730

**Meta-learning / multi-task**
* Bonilla, Chai, Williams. Multi-task Gaussian process prediction. NeurIPS 2007
* Garnelo et al. Conditional neural processes. ICML 2018, arXiv:1807.01613
* Nguyen, Grover. Transformer neural processes. ICML 2022, arXiv:2207.04179
* Jha et al. The neural process family: survey. arXiv:2209.00517
* Patacchiola, Turner, Crowley, O'Boyle, Storkey. Deep kernel transfer. NeurIPS 2020, arXiv:1910.05199
* Rothfuss et al. PACOH. ICML 2021, arXiv:2002.05551; F-PACOH, NeurIPS 2021
* Wistuba, Grabocka. Few-shot BO with deep kernel surrogates. ICLR 2021, arXiv:2101.07667
* Lee et al. Set Transformer. ICML 2019, arXiv:1810.00825

**Physics-informed / SBI**
* Karniadakis et al. Physics-informed machine learning. *Nat. Rev. Phys.* 3:422 (2021)
* von Rüden et al. Informed machine learning. *IEEE TKDE* 35:614 (2023)
* Hanuka et al. arXiv:1911.01538; arXiv:2009.03566 (physics-informed GP for accelerators)
* Boltz et al. Prior mean models for accelerator BO. arXiv:2403.03225 (2024)
* Cranmer, Brehmer, Louppe. The frontier of simulation-based inference. *PNAS* 117:30055 (2020), arXiv:1911.01429
* Boelts, Deistler et al. sbi reloaded. *JOSS* 10:7754 (2025), arXiv:2411.17337
* SBI for GWs in real noise, arXiv:2406.03935; review arXiv:2507.11192

**Uncertainty and conformal**
* Liu et al. SNGP. NeurIPS 2020, arXiv:2006.10108
* Amini, Schwarting, Soleimany, Rus. Deep evidential regression. NeurIPS 2020, arXiv:1910.02600
* Meinert, Gawlikowski, Lavin. The unreasonable effectiveness of deep evidential regression. AAAI 2023, arXiv:2205.10060
* Duan et al. NGBoost. ICML 2020, arXiv:1910.03225
* Meinshausen. Quantile regression forests. JMLR 7:983 (2006)
* Harrison, Willes, Snoek. Variational Bayesian last layers. ICLR 2024, arXiv:2404.11599
* Daxberger et al. Laplace Redux. NeurIPS 2021, arXiv:2106.14806
* Romano, Patterson, Candès. Conformalized quantile regression. NeurIPS 2019, arXiv:1905.03222
* Barber, Candès, Ramdas, Tibshirani. Predictive inference with the jackknife+. *Ann. Stat.* 49:486 (2021), arXiv:1905.02928
* Dunn, Wasserman, Ramdas. Distribution-free prediction sets for two-layer hierarchical models. *JASA* 118:2491 (2023), arXiv:1809.07441
* Lee, Barber, Willett. Distribution-free inference with hierarchical data. arXiv:2306.06342
* Boström, Johansson, Löfström. Mondrian conformal predictive distributions. COPA 2021, PMLR 152; Boström, crepes, COPA 2022, PMLR 179
* Tibshirani, Barber, Candès, Ramdas. Conformal prediction under covariate shift. NeurIPS 2019, arXiv:1904.06019
* Fannjiang, Bates, Angelopoulos, Listgarten, Jordan. Conformal prediction under feedback covariate shift. *PNAS* 119:e2204569119 (2022), arXiv:2202.03613
* Stanton, Maddox, Wilson. Bayesian optimization with conformal prediction sets. AISTATS 2023, arXiv:2210.12496

**Bayesian optimization**
* Balandat et al. BoTorch. NeurIPS 2020, arXiv:1910.06403
* Sui, Gotovos, Burdick, Krause. SafeOpt. ICML 2015, PMLR 37:997
* Berkenkamp, Krause, Schoellig. BO with safety constraints. *Mach. Learn.* (2023), arXiv:1602.04450
* Kirschner, Mutný, Hiller, Ischebeck, Krause. LineBO. ICML 2019, arXiv:1902.03229
* Kirschner et al. Tuning particle accelerators with safety constraints using BO. *PRAB* 25:062802 (2022), arXiv:2203.13968
* Gardner et al. BO with inequality constraints. ICML 2014
* Eriksson, Poloczek. SCBO. AISTATS 2021, arXiv:2002.08526
* Eriksson, Jankowiak. SAASBO. UAI 2021, arXiv:2103.00349
* Hvarfner, Hellsten, Nardi. Vanilla BO performs great in high dimensions. ICML 2024, arXiv:2402.02229
* Astudillo, Frazier. BO of composite functions. ICML 2019, arXiv:1906.01537; BO of function networks, NeurIPS 2021
* Buathong, Wan, Astudillo, Daulton, Balandat, Frazier. BO of function networks with partial evaluations. ICML 2024, arXiv:2311.02146
* Poloczek, Wang, Frazier. Multi-information source optimization. NeurIPS 2017, arXiv:1603.00389
* Wu, Toscano-Palmerin, Frazier, Wilson. Practical multi-fidelity BO. UAI 2019, arXiv:1903.04703
* Takeno et al. MF-MES. ICML 2020, arXiv:1901.08275
* Perdikaris et al. Nonlinear information fusion (NARGP). *Proc. R. Soc. A* 473:20160751 (2017); Emukit, https://github.com/EmuKit/emukit
* Roussel et al. BO algorithms for accelerator physics. *PRAB* 27:084801 (2024), arXiv:2312.05667
* Duris et al. BO of a free-electron laser. *PRL* 124:124801 (2020), arXiv:1909.05963
* Xopt / Badger (SLAC), e.g. ICALEPCS 2025 WESV001

**CCD / detectors**
* Cuevas-Zepeda et al. Automating sensor characterization with Bayesian optimization. arXiv:2509.21661 (FERMILAB-PUB-25-0671); JINST 21 P10012 (2026) per OSTI
* Hope, Fernandez-Moroni, Gamero, Drlica-Wagner. Machine learning for skipper CCD parameter selection. Fermilab poster (2026) [snippet only]
* Wüthrich, Fazan, MacBride, Soares-Santos. Skipper CCD readout time optimization for astronomical applications. Proc. SPIE 14157 (2026), arXiv:2607.20126
* Marrufo Villalpando et al. Characterization and optimization of skipper CCDs for the SOAR IFS. *PASP* 136:045001 (2024), arXiv:2311.00813
* Chierchie et al. Smart readout of nondestructive image sensors with single-photon sensitivity. *PRL* 127:241101 (2021), arXiv:2111.09905
* Chierchie et al. Detailed modeling of the video signal and optimal readout of charge-coupled devices. *Int. J. Circuit Theory Appl.*, doi:10.1002/cta.2784
* DAMIC-M: First production of skipper-CCD modules, arXiv:2509.06943; Low Background Chamber, arXiv:2407.17872
* Moon et al. Machine learning enables completely automatic tuning of a quantum device. *Nat. Commun.* 11:4161 (2020)
