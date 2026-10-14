# Session context: 2026-10-14

Continuation of `2026-10-13_session_context.md` (which ends with the 5 + 10 trace-set packets, the scan
time estimates, the image-time check and the noise question).

## 1. Inputs from the user

* **Nominal configuration** (the .bcf content):
  * `[bias]`: VDD1/2 −19, VR1/2 −3, VDRAIN1/2 −23, VSUB 45;
  * `[clock_offsets]`: OFS_RS1/2, OFS_CLK −10;
  * `[clock_rails]`: V1/V2/V3 A/B/C high 9.9, low 7.9/7.7; TG A/B 7.4/7.4; H1/H2/H3 high 9.5 (H2_C 9.0),
    low 6.7/6.5; SW 1.0/−6.0, OG 0.0/−6.0, RG 9.0/3.0, DG 0.0/−5.0, for amplifier sets 1 and 2;
  * `[sequencer]`: vertical `Vtransfer22`, horizontal `HtransferUU`, ch0..3 = CCD_D, C, A, B.
* **Configuration files:** acm101 reads `module_PA03.bcf`, acm102 reads `ccd_PQ04.bcf`
  (`cdaq qadd "daq.read_config('<file>')" <board>`).
* **The group's command style:**
  * `cdaq qadd "daq['H1_A_L'] = 6.2" acmpool`;
  * `daq.take_trace(...)`, `daq.master(101)`, `daq.use_external_clock()`, `daq.set_standalone()`;
  * `python scripts/take_image.py image_params/tc1/img.toml`.
* `img.toml` is now **16×1000×500**, 1×1, prescan 5300 (updated by the user).
* **The user is compiling the voltage and timing limits** of these CCDs and the relations that must hold
  between them. These go into `configs/limits.toml` / `ccd_limits.py` (still `reviewed = false`) and
  the checks of the manual script.

## 2. `daq_scripts/manual_packet.sh`: one packet by hand

Plain bash (works with bash 3.2+). It only **queues** plain `cdaq qadd` commands; the cdaq queue runs
them in order. Current default sequence:

1. `daq.read_config('module_PA03.bcf')` → acm101, `daq.read_config('ccd_PQ04.bcf')` → acm102 (start
   from nominal); `daq.set_standalone()`; `daq.use_external_clock()`;
2. 3 flushes;
3. **5 reference trace sets**: one trace per channel 0–3 to `acmpool`, no flush between the channels of
   a set, **a flush between sets**;
4. the parameter changes, `daq['NAME'] = value` (`NAME=VALUE` for both boards via acmpool,
   `acm101:NAME=VALUE` for one board); a flush;
5. **10 trace sets** at the new parameters, a flush between sets;
6. `daq.master(101)`; `take_image.py image_params/tc1/img.toml`;
7. `daq.set_standalone()` **first** (a flush, e.g. the repeat queue's, would hang in master mode), then
   both nominal .bcf files again and a flush.

**Trace, matched to the image** (`docs/packet_design.md` section 6):
`daq.take_trace(fname=..., ch=c, subroutine='AcquireImage', EXPOSURE=0, NROW=2, NCOL=5, NPBIN=1, NSBIN=1,
NDCM=500, BOOLCLR=0, NDCMPRE=0, NROWPRE=0, NROWPOST=0, NCOLPRE=100, NCOLPOST=0)`.
* It has the same skips and binning as img.toml, with its own small size; every cut is given, so
  nothing is inherited from the last image; 0.19 s, ~11 MB per channel.
* The group's earlier command (5×5 binning, 100 skips) is kept as a comment.
* If the image's NDCM/NSBIN/NPBIN change, `TRACE_ARGS` must be changed with them.

**Checks before anything is sent:**
* unknown parameter names (typos);
* values that are not numbers;
* any rail with HIGH < LOW after the change (the TG_B case of 2026-10-12).

It then shows a plan table and asks for confirmation. `DRY_RUN=1 bash manual_packet.sh` prints every
command and sends nothing. The checks were tested with deliberate mistakes.

**Default run:** 18 flushes, 60 traces, the 16-row image: **about 16 min**.

**Settings** (top of the file):
* `N_REF_SETS` (5), `N_SETS` (10), `N_INITIAL_FLUSHES` (3);
* `FLUSH_BETWEEN`: `"set"` (default), `"trace"` (a flush before every trace, ~44 min with the old
  35-row image) or `"none"` (~9 min);
* `FLUSH_AFTER_CHANGE`, `TRACE_CHANNELS`, `TRACE_ARGS`, `CHANGES`, `IMAGE_TOML`, `BCF`, `CDAQ_DIR`,
  `PYTHON`.

`daq_scripts/manual_packet_commands.txt` holds the 92 commands of the default run, for reference.

**History of the defaults today** (on request):
1. first the group's 5×5/100-skip trace with a flush between sets only;
2. then the matched trace with a flush before every trace;
3. then the matched trace with no flush between the channels of a set (current).

Commits: `035d4bd`, `85f6852`, `d4b9d6a`, `3edf635`.

## 2b. ML models (in progress, started 2026-10-14)

**Request:** think about the model types, write code, install software, and draft a document (basic and
advanced) on what the models do, their inputs, and the training data needed. Answer two questions:
full trace or features, and can a model find its own features.

**Software:**
* project venv `.venv/` (gitignored) from `requirements-ml.txt`: numpy, scipy, pandas, matplotlib,
  astropy, scikit-learn 1.9, torch 2.14 (MPS), nbformat/nbclient/ipykernel; plus pymupdf;
* run with `cd src && ../.venv/bin/python -m ml.benchmark`.

**Code `src/ml/`:**
* `synthetic.py`: toy CCD world in the `dataset.csv` format. It has:
  * 10 parameters with real names, 4 amplifiers per board (ch2 4× noisier);
  * toy physics for σe, λ and gain, with a forbidden region VDRAIN − VDD > −2.5 that the design
    respects (`allowed`);
  * a drifting hidden state (λ ±35 %), intermittent pickup (40 % of packets, a noise floor visible
    only in the curves), 17 hand features and 4 curves.
* `data.py`: `load_dataset`, `make_inputs` (groups theta/delta/amp/trace/ref/diff),
  `make_targets` (logs), `curve_tensor`, `curves_from_processed` (the real `.npz` curves).
* `models.py`: BayesRidge, GBT (with quantile std), GP (ARD Matern; capped at 400 points and 150
  iterations), MLPEnsemble, and CurveCNN (on the MPS GPU), all with `fit` / `predict(mean, std)`.
* `evaluate.py` (grouped CV, scores incl. coverage) and `optimize.py` (EI Bayesian optimization,
  trust region, `allowed` hook).
* `benchmark.py`, with five experiments: inputs, relevance, holdout (new module, 0/5/20 of its
  packets), learning, bo.
* plots: `plot_representations.py` (real-trace ladder), `plot_overview.py`.

**Performance fixes:**
* sklearn HistGradientBoosting must run with 1 thread (threadpoolctl): 100× faster;
* this torch build has no fast CPU convolutions, so the CNN runs on MPS with batch 256.

**First results (synthetic, quick run):**
* θ alone cannot know the amplifier (an `amp` one-hot was added);
* the reference traces reveal the hidden state: λ error 3.6 % (GP) vs 16–24 % from θ alone;
* GP ARD on 44 inputs with little data is worse for σe than GP on θ;
* BO beats random search;
* no model generalizes to a never-seen module from one training module (hence the k-shot test).

The full benchmark is finished (results in `docs/ml_results/*.json`, figures in
`docs/overleaf_ml_models/figures/`). See 2d.

**Document:** `docs/overleaf_ml_models/ml_models.tex`, drafted. It has basic/advanced boxes,
summary, two roles (predictor P, surrogate S + BO), inputs table, five model families with
equations, full trace vs features (representation ladder), own-feature discovery, data needs,
validation and BO. The results section is filled (see 2d), and a new section covers the next
models (2e). It compiles to 19 pages with no overflow; `docs/overleaf_ml_models.zip` was rebuilt.

## 2c. Trace-features technote (in progress)

**Request:** fix the formatting of `docs/trace_features.tex` (figures off the page, unreadable
text) and write a separate, more mathematical technical note on how every feature is computed.

* Tectonic 0.17 was installed in the scratchpad (`texenv`, conda-forge) to compile and check layouts.
* `fit_damped_ringing`, `skip_tools`, `transfer_tools`, `trace_features` and the helpers were
  re-read for the equations.

**Update (later on 2026-10-14):**
* **First full benchmark (v1 world):**
  * the CNN on curves gave σe 2.7 % against 11–13 % for every feature model (the intermittent
    pickup is visible only in the curves);
  * λ was best with the reference features (GP 3.8 %, NN 4.5 %);
  * a new module never seen in training failed for all models;
  * BO beat random.
* **The user asked: features + trace, or trace alone?**
  * A new benchmark experiment `representation` compares features only, curves only (θ and
    reference curves) and both, with and without θ, at 60–960 packets.
  * The toy curves were made consistent with the real curve definitions: they carry the same
    information as the features except the absolute levels, which real curves remove.
  * There are now 7 curves (plus `ref:` versions).
  * The benchmark was restarted with this world (the old log is `mlfull_v1.log` in the scratchpad).
* **Trace-features note fixed:**
  * new `src/plot_feature_figures.py` redraws all 25 figures from the real 2026-10-08 trace at
    page width (6.9 in / 3.4 in, 8–9 pt);
  * the generator moved into the repo as `src/make_trace_features_note.py`: natural widths, a
    height cap, the curve grid split into two, full pages for separation/correlation, narrower
    table columns;
  * the correlation matrix now has group bands instead of 85 unreadable names;
  * the separation numbers were recomputed with the corrected ringing fit (73 redundant pairs);
  * compiled with tectonic: 31 overfull warnings went down to 3 tiny ones.
* **Technical note (math of every feature):** being written in `docs/overleaf_trace_features_technical/`.

**Technical note on the trace features (done):** `docs/overleaf_trace_features_technical/trace_features_technical.tex`
(+ zip), 16 pages, 0 overflow warnings. It covers:
* notation: samples, windows, exact cycle, cycle matrix, baseline, CDS, transfers, usable / row-start
  pixels, the counts table;
* common operations: rsd (1.4826 MAD), detrending (degree min(4, n/50); why no running mean), residual
  cycles, block noise, autocorrelation, Allan deviation, spectral density, largest-step t, line count;
* fitting: variable projection with a log grid, fine grid and parabola; two exponentials; the corrected
  damped ringing; the noise model;
* the exact formula of every one of the 85 features and 12 curves, with the code's index ranges and
  constants;
* the packet-level combination with outlier rejection.

It reuses the readable figures. `docs/overleaf_trace_features.zip` was rebuilt as well.

## 2d. ML benchmark results (synthetic world v2, full run)

**Inputs** (240 training packets, 200 test packets), relative error on new packets:
* θ + amp: σe 12 %, λ 26 % (θ cannot know the drifting state), gain 0.9 %;
* θ + amp + features: GP λ 3.6 %, NN 4.2 %, σe ~12 % for every feature model (the intermittent pickup
  is invisible to the features);
* curves + θ + amp (CNN): σe 8.5 %, λ 4.7 %.

**Coverage:** GP and ensembles 0.67–0.80 (target 0.68); boosted-tree quantiles are too narrow (0.4–0.6).

**Learning curves:** the feature models level off at σe ~11–12 % from ~120 packets. The CNN on curves
matches them up to ~240 packets, then reaches σe 1.9 % at 480 and 1.7 % at 960 (it found the pickup
floor). For λ: GP + features 3.5 % from ~120 packets; CNN 3.3 % at 960.

**GP relevance:** σe ← CDS rms; λ ← reference baseline (the state) + V_low; gain ← amplifier.

**New module:** with 0 of its packets, σe/gain are off by 35–70 % for every model; with 5 packets, gain is
2.5 % (GP); with 20 packets, σe is 5–15 %. Start each new module with ~20 packets.

**Representation** (features vs curves vs both, each with θ and reference traces; n = 60…960):
* without θ, λ stays ~16 % for every representation: λ depends on rails the traces do not show, so
  θ must be an input;
* σe: at ≤120 packets all ~11–15 %; at 240, curves only 5.6 % and both 7.3 %, against features only
  12 %; at 480, curves only 2.2 %, both 2.1 %, features only 11.5 %;
* σe at 960: curves only 1.7 %, both 1.8 %, features only 10.8 %;
* with θ + amp: λ 3.1–3.6 % for all representations at 480–960;
* **answer to "features + trace, or trace alone?":**
  * with enough packets (≥ ~480), the curves alone are as good as curves + features;
  * below ~240 packets the features help a little, because they are pre-digested;
  * adding them never hurts, so the recommended input is θ + amp + curves (+ features).

**Bayesian optimization** (GP surrogate + EI, 5 seeds, 10 random start packets, 60 packets each):
* objective J = log σe + ½ log λ; nominal −2.49, reference optimum −2.81;
* BO comes within 0.04 of the optimum by 30 packets in every seed and ends at −2.77 to −2.82;
* random search ends at −2.61 to −2.72;
* σe√λ ends about 27 % below nominal.

**Process note:** watcher loops of the form `until ! pgrep -f "<pattern>"` never end, because they match
their own command line. Four were killed. Use `pgrep -f "[m]l.benchmark"` or wait on the PID instead.

**Notebook:** `notebooks/ml_models_walkthrough.ipynb`, run with the `.venv` kernel (~70 s), saved with its
outputs. It covers:
* the synthetic world, the inputs and targets, and the five model types;
* the coverage of the error bars and the GP relevance;
* the learning and representation curves (read from `docs/ml_results/`);
* the new-module test, a short BO run, and how to plug in real processed packets.
`README.md` and `notebooks/README.md` point to it.

## 2e. User question: are there more complex models that would do better?

**What the first round taught us:**
* the σe error left over comes from information the hand features lack (pickup); the curves have it;
* the drifting state is read from the reference traces;
* a new module is the hardest case;
* σe, λ and gain are physically linked;
* the GBT error bars are too narrow;
* labeled packets are few, while unlabeled traces are many.

**Next models** (section 12 of `ml_models.tex`):
* **A. Physics-structured outputs:** predict the noise terms (white, 1/f, pickup), gain and dark terms,
  and combine them with the known formulas.
* **B. Hierarchical / multi-task GP:** shared kernel plus per-module/amplifier offsets; a new module
  needs only ~5 packets.
* **C. Deep kernel learning:** CNN encoder on the curves feeding a GP (CNN accuracy plus GP error bars,
  for BO).
* **D. Self-supervised pre-training** on unlabeled traces, then fine-tuning with few labels.
* **E. Fusion / set models** over the 10 repeats and 5 references (DeepSets/attention).
* **F. Better optimization:** multi-fidelity (traces vs images), trust-region (TuRBO), constrained and
  multi-objective BO.
* **G. Few-shot / meta-learning** for new modules.
* **H. Conformal calibration** of the error bars.

**Priorities:**
1. A + B;
2. H;
3. D + C, once real unlabeled traces exist;
4. F, when the first real scan runs.

Nothing has been prototyped yet.

## 2f. What is running

Nothing. The full benchmark (PID 61289) finished; no background jobs or watchers are left.

## 3. Not yet known or checked

* What `read_config` resets: the mode is set explicitly afterwards, but whether it touches the
  sequencer constants or the clock is unknown.
* Whether `take_image.py` without `-m` keeps the master mode set just before (it should: the commands
  run in queue order).
* The script has only been dry-run on the Mac; nothing has been sent to the DAQ.

## 4. Open

* Run `DRY_RUN=1 bash daq_scripts/manual_packet.sh` on the DAQ machine, then a first real packet; then
  process it (σ at 500 skips ≈ 0.26 e⁻? λ per row; matched traces decode without layout warnings).
* Put the user's limits and relations into `configs/limits.toml` / `ccd_limits.py` and the manual
  script's checks.
* Decide whether to keep the flush between trace sets (test sets with and without it, compare the
  trace-to-trace scatter).
* Everything open from 2026-10-13 (WADERs JSON with per-row dark current; the first scan design).

## 2g. Second-round models A + B (prototyping, started later on 2026-10-14)

**User request:**
* prototype A (physics-structured outputs) + B (hierarchical GP) on the synthetic data, then the other
  candidates if time allows;
* update the context and the tex as we go, and start a new notebook;
* plan how to test all models on real data, and say whether the packets need different data;
* think about a better data format for new models;
* **back up the context often** (the session is near its limit).

**Code, done:**
* `src/ml/synthetic.py`: two optional flags. The defaults reproduce world v2 bit for bit (checked against
  HEAD).
  * `image_only_noise`: a noise term only the image sees: lf = 0.10 (1 + 0.5 (VR+3)^2)(1 + 0.3 dDH) e-,
    on average 29 % of σe;
  * `per_row_lambda`: λ is measured on each of the 16 rows, λ_r = cic + dark (r+½)/8 with 12 % scatter per
    row; a line fit gives the labels `label.lambda_dark` (12 % error) and `label.lambda_cic` (26 %);
  * world v3 = both flags; there are also the new truth columns `truth.dark`, `truth.cic` and
    `truth.image_only`.
* `src/ml/structured.py`:
  * `allan_fit`: a²/n + b² on the Allan curve gives a = σ1·g and b = floor·g in ADU;
  * `StructuredPrep`: x = θ + a packet state vector. The state vector is the reference features minus each
    amplifier's training mean (only amplifiers with ≥ 20 rows), averaged over the packet's amplifiers;
  * `HierGP`: torch float64 exact GP with kernel shared + [same module](c + s·M) + [same amp](c + s·M),
    fitted with L-BFGS;
  * `DirectGP` (B);
  * `PhysicsStructured` (A or A+B): log σe = ½ log(a²/N + b²) − log ĝ + r(x), with ĝ and the residual r
    as GPs; λ is a GP on log λ, or exp(dark GP) + exp(CIC GP).
* `src/ml/benchmark_structured.py`:
  * `learning` (worlds v2, v3; n = 15…480; scored against the labels and against the truth);
  * `newmodule` (k = 0, 1, 2, 5, 10, 20).

**First numbers** (60 training packets, 100 test packets):
* σe:
  * A / A+B: 2.4 % (1.9 % against the truth) in v2 and 3.4 % in v3;
  * first round at 60 packets: 13–15 %;
  * B alone: 14 %.
* λ: 5.4 % (hierarchical GP); the dark + CIC split is not better at 60 packets (the CIC labels are noisy).
* Gain: hierarchical 0.8 % (0.2 % against the truth); ordinary GP 0.9 % (0.4 %).
* New module with k = 0 (quick run): the σe error equals the gain error (~33 %). The trace cannot show the
  gain, so ≥ 1 imaged packet of a new module is needed. The hierarchical GP's coverage stays 0.75; the
  others drop to 0.
* A "no curves" variant was dropped: the toy hand features carry large per-amplifier offsets, so the toy
  CDS-rms feature is not a physical σ1. Real data has "skip: CDS rms" and "fit: correlated noise floor".

**Next:**
* full benchmark run, figures, ml_models.tex section "Prototypes of the next models";
* notebook `notebooks/ml_structured_models.ipynb`;
* H (conformal), then others;
* test plan + data needs + data format section.

**Update (A+B work, continued):**
* **Bug fixed:** in the new-module test the state vector must be built on whole packets. The test rows of
  module 101 alone gave state = 0, which caused the λ error of 26 %. `run_structured(full=…)` now builds
  the inputs on the full df and then splits. With the fix, a never-seen module (k = 0) has λ error
  3.1–3.6 % (first round: 50–118 %); σe and gain are still ~33 % because the gain is unknown without an
  image.
  * Caveat: with one training module the module-level variance cannot be learned. It stays at its
    prior, so B's λ error bars are too wide there (coverage 1.0); conservative, not wrong.
* **New `src/ml/calibrate.py` (H):**
  * `Conformal(make_model).fit(X, Y, groups, extra)`: 5-fold grouped out-of-fold normalized errors give
    the factors for 68 % and 95 %; then a refit on everything;
  * `coverage_report`.
* **New experiment `design`:** the planned one-at-a-time scan (`synthetic.oat_thetas(20)`, 197 allowed
  points) vs random points vs half and half, tested on random points.
  * Why: `configs/scan_recommended.toml` / `scan_packets.py` support `oat` and `random`, and the "10 × 20"
    plan is one-at-a-time, but every synthetic test so far trained on random joint points.
* **Full run** (learning, newmodule, conformal, design) running in the background. Log in the scratchpad
  `structured_full.log` (PID in `structured_full.pid`).
* **Drafted** `docs/overleaf_ml_models/draft_realdata_sections.tex`, to be merged into ml_models.tex.
  It has three sections:
  1. testing every model on real data, in stages:
     * stage 0: label noise + "does the trace physics hold" + per-row λ;
     * stage 1: scan splits (grouped, blocked in time, leave one parameter out), scores, learning curves;
     * stage 2: new module;
     * stage 3: offline replay BO, then prospective;
     * stage 4: C/D/E/F;
  2. do the packets need to change:
     * no longer packets;
     * add per-row λ (+σ) from WADERs;
     * keep the un-averaged image skips if cdaq can;
     * use a design with joint changes;
     * take trace-only packets;
     * record temperatures / conditions / all pins;
     * nominal image packets (already reference_every = 5);
     * 2–5 imaged packets per new module;
  3. data format: keep the raw files; add Parquet/HDF5 tables packets, trace_sets (per set, not
     combined), curves, labels with errors, image_rows, schema; `dataset.csv` becomes a view.

**Results so far (full run):**
* **New module** (240 packets of module 100 + k packets of module 101; error against the labels):

  | | k = 0 | k = 1 | k = 2 |
  |---|---|---|---|
  | first-round GP | σe 72 %, λ 118 %, gain 35 % | σe 24 %, λ 37 %, gain 3 % | σe 18 %, λ 39 %, gain 3 % |
  | B | σe 58 %, λ 3.6 %, gain 33 % | | σe 7.4 %, λ 3.1 %, gain 0.9 % |
  | A+B | σe 33 %, λ 3.6 %, gain 33 % | σe 3.3 %, λ 3.1 %, gain 3.4 % | σe 2.4 %, λ 3.1 %, gain 0.9 % |

  * With an ordinary GP, A breaks at k = 1 (λ 37 %): the one-hot of a nearly unseen amplifier. The
    hierarchy is robust.
  * **One or two imaged packets of a new module are enough with A+B** (the first round needed ~20).
* **F, decomposed BO** (new experiment `bo` in benchmark_structured; `src/ml/benchmark_structured.py`
  `decomposed_propose`): hierarchical GPs on log σe and log λ per good amplifier, combined into J, against
  the GP on J. 5 seeds, 10 random start packets + 50.
  * Best true J after 30 packets: decomposed −2.797 to −2.820; GP on J −2.747 to −2.791.
  * After 60 both reach −2.80 to −2.82 (reference −2.81).
  * So decomposing about halves the packets needed; the final value is the same.
  * Outputs: `docs/ml_results/structured_bo.json`, figure `ml_structured_bo.png`.
* **C, D, E are not testable meaningfully on this synthetic world:** A+B already uses everything the
  toy curves contain (the Allan floor); the image-only term is invisible to traces by construction. They
  belong to stage 4 on real data.
* **ml_models.tex:**
  * the draft sections are merged before "Plan": "Prototypes of the next models (A, B, F, H)" (sec:proto),
    "Testing every model on real data" (sec:realtest), "Do the packets need to change?" (sec:dataneeds),
    "A better data format" (sec:format);
  * new summary bullets and updated Plan/Software;
  * 5 placeholders (SUMMARY-PROTO, PROTO-SUMMARY, PROTO-LEARNING, PROTO-DESIGN, PROTO-CONFORMAL) wait for the
    full run;
  * the draft file was removed.
* **Notebook generator:** written in the scratchpad (`make_structured_nb.py`, writes
  `notebooks/ml_structured_models.ipynb`), not run yet. It reads `structured_learning/newmodule/design/
  conformal/bo.json`.

**Learning results (error against the truth, 200 test packets; `docs/ml_results/structured_learning.json`):**
* **σe, v2:** A+B 2.4 % at 15 packets and 2.0 % from 30 on (flat: limited by the precision of the trace
  measurement, 10 sets); A with an ordinary GP is about the same from 120 on. First-round GP: 18 → 12–14 %;
  CNN: 12.7 % (60), 8.5 % (240), **1.2 % (480)**, so the CNN overtakes A+B only at ~480 packets.
  Without the floor measurement: 12–17 %.
* **σe, v3** (image-only noise): A+B 3.3 % (15), 4.4 % (30), 3.1 % (60), 2.2 % (240–480); first-round GP
  10–17 %; CNN 7.7 % (240), 1.5 % (480).
* **λ:** A+B is the same as the first-round GP (λ is not what A changes): v2 21 % (15), 10 % (30), 7 % (60),
  3 % (120), 1.3–1.6 % (240–480); the first-round GP is better at 30–120 packets (8.5 %, 4.4 %, 2.1 %).
  * **The dark + CIC split (v3) is worse at every size from 30 packets on** (34 %, 32 %, 9.6 %, 5.7 %, 2.3 %, 2.3 %, against
    41 %, 18 %, 6.2 %, 3.9 %, 1.6 %, 1.5 % direct) and its coverage is too wide (0.9–1.0). The 16-row
    components are too noisy (CIC 26 %). The per-row λ is still worth having as a stage-0 physics check
    and for longer images; a joint likelihood would be the right way to use it.
* **Gain:** hierarchical 0.1–0.6 %; ordinary 0.2–0.9 %.
* **Coverage:** A+B 0.56–0.59 at 15 packets, 0.66–0.74 from 30 on.
* Figures redrawn with plain tick labels: `ml_structured_learning.png`, `ml_structured_newmodule.png`.
* `models.n_rows` now handles dict inputs without curves; conformal + design were rerun (log
  `structured_cd.log`).

**Second round finished (A, B, H, F prototyped):**
* **Conformal** (`structured_conformal.json`; 68 % coverage of σe/λ/gain, raw → conformal):
  * boosted trees at 60 packets: 0.36/0.22/0.60 → 0.72/0.77/0.67;
  * boosted trees at 240: 0.44/0.59/0.60 → 0.63/0.72/0.72;
  * NN, CNN and A+B at 240 were already 0.61–0.79 and end at 0.65–0.72;
  * at 60 packets the factors are noisy (A+B λ 0.75 → 0.57), so use it from ~100 packets and also report
    the 95 % interval.
* **Design** (`structured_design.json`; 197 packets each, error against the truth on random test points):
  * λ after a one-at-a-time scan: 8.9 % (GP), 11.4 % (B, A+B);
  * after a random scan: 1.9 % (GP), 1.2 % (B, A+B); half and half: 2.3 % / 1.9 %;
  * trees: 12–15 % in every design;
  * σe from A+B: 1.9–2.1 % in every design;
  * **recommendation: half one-at-a-time, half random** (`scan_packets.py` already has `type = "random"`).
* **Corrections** (verified in the notebook):
  * the trace alone with the true gain gives σe to ~3 %, not 1 %; A+B's learned correction brings it to 2 %;
  * the per-row components are measured to 13–15 % each, not 26 % (the 26 % came from clipping at 1e-4).
    With the 5 % clipping used by the model they are anti-correlated, and their sum is known to 3 %.
    That is why two separate fits lose; a joint likelihood is the fix.
* **ml_models.tex:**
  * all placeholders filled (main results, learning, design table, conformal table);
  * 28 pages, 0 overfull, 0 undefined; zip rebuilt;
  * figures redrawn with plain tick labels.
* **Notebook** `notebooks/ml_structured_models.ipynb`: executed, 0 errors, 27 s. The generator is in the
  scratchpad (`make_structured_nb.py`).
* READMEs updated.

**Answers to the user's questions:**
* **Different data?** The packet as designed is fine. Add (no extra packet time):
  * the scan design: half random (the most important);
  * per-row λ / σ from WADERs;
  * the un-averaged image skips if cdaq can save them;
  * trace-only packets;
  * recorded temperatures / conditions / all pins;
  * 2–5 imaged packets per new module.
* **Data format:** keep the raw files; add the tables packets, trace_sets (per set), curves, labels (with
  errors), image_rows, schema; `dataset.csv` becomes a view.

**Possible next steps:**
* joint likelihood for per-row λ;
* use WADERs' label errors as GP noise;
* stage-0 check on the first real packets;
* C/D/E on real traces;
* write the trace_sets / labels tables in process_packet.py.

## 2h. Remaining models C, D, E, F (rest), G (started later on 2026-10-14)

**User request:**
* write code for the remaining models and sections on their implementation;
* give an opinion on today's work and what to improve now.

**Plan:**
* `synthetic.py` world v4 (`within_packet_pickup`, `damaged_fraction`, per-set Allan curves) for E;
* `src/ml/deep.py`:
  * D: curve encoder, pre-training by masked reconstruction + θ prediction + contrastive, fine-tuning;
  * C: deep kernel GP;
* `src/ml/setmodels.py`: E, a DeepSets/attention pooling of the trace sets inside A (learned combination);
* `src/ml/bo_advanced.py`: F, trust region (TuRBO-like), learned feasibility (constrained EI), multi-fidelity
  trace-only screening, ParEGO multi-objective;
* G: HierGP across several training modules (empirical-Bayes meta-learning) in the benchmark;
* `src/ml/benchmark_advanced.py` with the experiments; tex sections; maybe a third notebook.
* **Progress (C/D/E):**
  * `synthetic.py`: world v4 flags `within_packet_pickup`, `damaged_fraction`; v2 still identical. In v4
    with the true gain, averaging the Allan curves of the sets gives σe with a 35 % bias (damaged sets),
    the median 10 % error (it destroys the duty cycle);
  * `src/ml/deep.py`: `Pretrainer` (D: masked reconstruction + θ prediction + contrastive),
    `PretrainedHead` (frozen), `PretrainedCNN` (fine-tune), `DeepKernelGP` (C).
  * First tests at 60 labeled packets: no σe gain over the CNN from scratch (~14 %). The DKL
    fine-tuned on 60 packets over-fits (coverage 0.2–0.3).
  * Probe: the pre-trained embedding knows the state (R² 0.96) but not the floor (R² 0.14), because masking
    16-point blocks is too easy. Added whole-curve masking (p = 0.5) and 32-point blocks; testing.
* **User rule (2026-10-14):** if the session limit is reached, stop any commands and update (and push) the
  context documents. Push as we go.
* **Implemented:**
  * `src/ml/setmodels.py` (E: `combine_sets` with mean / median / robust mean / mean of variances /
    robust mean of variances; `SetPooling` DeepSets + attention; `SetStructured` = A+B with any combiner);
  * `src/ml/bo_advanced.py` (F: trust region, learned / known / no constraint, multi-fidelity trace-only
    screening, ParEGO + 2-D hypervolume);
  * `src/ml/benchmark_advanced.py` (experiments sets, modules (G), bo, pretrain (C/D));
  * `data.curve_tensor(log_frequency_spectra=True)` and `benchmark.inputs_for(..., log_frequency_spectra)`.
* **Finding (D):**
  * the self-supervised embedding learns the state (probe R² 0.95–0.98) but NOT the noise floor
    (R² 0.08–0.15) with linear-frequency spectra; with LOG-FREQUENCY spectra the floor is learned (R² 0.70);
  * on a linear axis the floor occupies ~2 of 256 points of the CDS spectrum;
  * **real-pipeline implication:** store and feed spectra on a log-frequency axis.
* **Hypervolume:** the reference is nominal + log 1.2 in both coordinates. The nominal σe is already near
  its best, so no point beats nominal in both; the gains in J come from λ.
* **Correction (E):** with two training scans the learned set combination does NOT consistently beat the
  fixed rules:
  * n = 60: 6.1 % vs best fixed 9.4 % (seed 1) but 3.4 % vs 3.4 % (seed 2);
  * n = 30, seed 2: 7.2 % vs 5.7 %;
  * n = 120: 3.3–3.4 % vs 3.7–4.0 %.
  The plain mean and the mean of variances fail (32–48 %, damaged sets). Learned ≈ robust rules, with
  wider (conservative) error bars, coverage ~0.9.
* **Full run:** `benchmark_advanced` is running (PID in the scratchpad `advanced_full.pid`, log
  `advanced_full.log`).
* **Started, interrupted:** a stage-0 check on the REAL 2026-10-12 data (module 101). Script in the
  scratchpad `stage0.py`. It compares the image overscan noise (NDCM 500) with √(a²/500 + b²) from each
  trace set's Allan fit. The notes of 10-12 already suggest overscan 0.42–0.66 ADU vs a white-noise
  extrapolation of the ch0 CDS noise 5.7/√500 = 0.25 ADU, i.e. the image noise is ~2× the white-noise
  prediction (a floor or image-only noise; needs checking: binned 20×20 overscan, NCOLPRE 185 traces).
