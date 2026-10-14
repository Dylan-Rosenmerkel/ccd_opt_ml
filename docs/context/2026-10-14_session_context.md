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
