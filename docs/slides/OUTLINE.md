# Project overview deck: outline (6 slides)

**Built deck (2026-10-20):** https://claude.ai/artifact/6TYq1G1Lz1dzK172GLKDmB, 12 slides in black and white (plots in
colour): cover · goal · what a trace is (`t_trace_anatomy.png`) · from a trace to numbers (`t_trace_noise.png`) · drift ·
run_WADERs labels · gain-correction side track · from a packet to training data (`p_processing.png`) · the models (table) ·
how we test them (`m_cv.png`) · results · next steps. The outline below is the first, 6-slide version.

Figures are in `docs/slides/figures/`; remake them with `.venv/bin/python docs/slides/make_slide_figures.py`.
Style: at most 3 short callouts per slide; key numbers large; one colour per idea throughout
(trace = blue, image = orange, model = green, problem = red). Exact numbers go in the speaker notes.

---

## Slide 1 — The goal: predict CCD noise from a short trace

**Message:** a 2-second trace should tell us what a full image would show.

| Figure | Where |
|---|---|
| Pipeline: settings → trace → image → labels → models (main graphic) | `figures/s1_pipeline.png` |
| Level staircase: lvl 1 (we are here) → lvl 2 → lvl 3 | `figures/s1_levels.png` |
| Optional inset: a trace and its image | `docs/context/figures/oct12_noise_trace_and_image.png` |

**Callouts:** 1 packet = 4 amplifiers = 4 training examples · the models are never told which amplifier they see.

---

## Slide 2 — From raw data to clean labels: the row-start drift

**Message:** the labels are only as good as the image. We found a real hardware effect and corrected for it.

| Figure | Where |
|---|---|
| Zero-electron level along the row, as read out vs after the drift step (image 130, ch0 and ch1) | `figures/s2_drift_profile.png` |
| WADERs fit χ²/ndf per channel, as is vs corrected (images 127–131) | `figures/s2_chi2.png` |

**Callouts:**
* the level sags by **−1.0 ADU** at the row start and recovers over ~300–600 columns; on PA03/board 101 since
  2026-09-22, absent on module 106; not caused by exposure, prescan length or the H clock;
* fit χ²/ndf **2.7–5.1 → 0.65–1.15** (ch0/1/3); labels repeat within 0.6–2 %;
* σe after correction: 0.155 / 0.161 / 0.200 e⁻ (ch0 / ch1 / ch3).

**Speaker notes:** detection by Gaussian fits of the zero peak per column block; the first 200 columns are cut and the
slow remainder subtracted; images without drift are left untouched; cause still open (question for a front-end expert).

---

## Slide 3 — comm_scripts: automatic WADERs fits

**Message:** your scripts replaced hand-written JSON configurations and are now the default labelling route.

| Figure | Where |
|---|---|
| run_WADERs flow: image → autoconfig → one JSON per CCD → panaSKImg in parallel → fit_quality.csv → labels | `figures/s3_runwaders_flow.png` |
| Label error on simulated packets (known truth): hand-written JSON vs run_WADERs | `figures/s3_label_error.png` |
| What comes out: fit_quality.csv for image 130 | `figures/s3_fit_quality_table.png` |

**Callouts:**
* geometry from the header, only the Poisson-like 0e⁻/1e⁻ peaks fitted → unbiased dark current;
* label error: hand JSON **8–12 %** off → run_WADERs **0–4 %** (median);
* on real images the hand JSON silently locked the gain at ~2× in 4 of 18 fits.

**Speaker notes:** how we call it: `run_WADERs images.txt --no-clusters --autoconfig-args=--keep-dead`
(16-row images have too few charged pixels for the dead-channel test); `process_packet.py` reads `fit_quality.csv` and
flags a label if the CCD was not fitted, an error is > 50 %, or χ²/ndf > 3.

---

## Slide 4 — Side track: run_WADERs `--gain-correction`

**Message (a divergence from the noise project):** the gain falls slightly as a pixel fills with charge, so X-ray
energies come out low and differ between CCDs; `--gain-correction` measures that fall from the image itself and
corrects the cluster energies.

| Figure | Where |
|---|---|
| The effect: measured gain vs electrons per pixel, 12 CCDs (falls ~3 % by 1000 e⁻) | `figures/gc_gain_curves.png` (report p. 2) |
| One module's 12 CCDs, Fe-55 K-alpha without / with: peaks line up | `figures/gc_module_alignment.png` (report p. 7) |
| Am-241 59.5 keV on EXT4: −6.0 % → −0.7 % | `figures/gc_am241.png` (Am-241 report p. 2) |
| Every CCD of every dataset, peak / true energy, without / with | `figures/gc_all_datasets.png` (report p. 10; backup or second panel) |

Suggested layout: gain curve (left, small) → module alignment (centre, large) → Am-241 (right).

**Callouts:**
* Fe-55 K-alpha over 82 corrected CCDs: **0.9835 → 0.9977** of the true energy; spread between CCDs **2.1 % → 0.32 %**;
* Am-241 59.5 keV: EXT3 0.976 → 0.998 (peak width 0.41 → 0.26 keV), EXT4 0.940 → 0.993;
* no source peak used anywhere: the correction comes from the electron comb of the image itself.

**Speaker notes:** usage `run_WADERs images.txt --gain-correction` writes corrected cluster files to `root_gaincorr/`
(the originals are kept); the gain curve g(n) = a + b·n is fitted per CCD from the spacing of the electron peaks at
50, 300, 800 e⁻ …; ~20 % of CCDs get no reliable curve and are left unchanged; 1-skip images are skipped; a residual
−0.2 to −0.7 % comes from charge lost at the cluster edge (clustering threshold), not from the gain. Not used in the
noise project (we run with `--no-clusters`). Source: `comm_scripts/docs/gain_correction_report.pdf`,
`gain_correction_Am241.pdf`.

---

## Slide 5 — The models, and how we test them honestly

**Message:** 12 models from simple to complex, compared against baselines; the 4 examples of a packet always stay together
when the data are split for testing.

| Figure | Where |
|---|---|
| Model ladder, simple → complex | `figures/s4_model_ladder.png` |
| Error per model at the planned dataset size (41 synthetic packets), with the "predict the average" line | `figures/s4_model_errors.png` |
| SAAS GP for λ: before / after at 13 / 41 / 100 packets | `figures/s4_saas.png` |
| Optional: what a GP does | `docs/overleaf_ml_guide/figures/guide_gp.png` |

**Callouts:**
* the physics-based models land far below the "predict the average" line;
* SAAS GP: λ error **19 → 15 %, 11 → 7.4 %, 4.9 → 3.9 %** (13 / 41 / 100 packets);
* robustness: one bad packet can no longer bend the learned correction, and the CNN no longer blows up on tiny datasets.

**Speaker notes:** at 41 packets the best σe model is `sets` (5.7 %), the best λ model `physics_joint` (4.6 %), gain
~0.9 % for most; the wide σe range of the physics models comes from one synthetic dataset with a packet whose trace sets
are mostly damaged (only `sets`, which looks at each set, handles it). Synthetic numbers compare methods only, not the real CCD; the "labels" reference model (A+B), which is
told the amplifier, measures what amplifier labels would add.

---

## Slide 6 — Where we are and what's next

| Figure | Where |
|---|---|
| Packet plan: session A (13) → B (17) → C (11), ~11 h | `figures/s5_timeline.png` |

Status board (build as a 3-column slide table with coloured ticks):

| ✅ Ready | 🟡 To decide | ⏭ Next |
|---|---|---|
| Packet script → manifest → processing → training: one command each | Trace settings (BOOLCLR, NDCMPRE, row-start pixels) | Take session A: 14 timing packets, ~3.8 h (43-packet plan) |
| Drift detection and correction in processing | Confirm the voltage windows | First real lvl-1 report: do traces beat the settings-only baseline? |
| Rehearsal passes end to end; all models train | Ask a front-end expert about the drift | Packet-aware error-bar calibration; flag unusual traces |
| Literature review; SAAS GPs adopted for λ | | Lvl 2: suggest settings (step-limited, safe Bayesian optimization) |

**Closing line:** if the traces predict σe within ~2× the label error and beat both baselines, level 1 passes.
