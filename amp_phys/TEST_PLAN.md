# amp_phys: characterising amplifiers we cannot probe

**Goal.** From data we already take (traces, skipper images, the voltage scans), estimate physical properties of each
CCD output amplifier and the silicon it reads: capacitances and couplings, transistor operating point, noise model,
traps and defects, geometry and doping. Give each one an uncertainty, and use them to diagnose amplifiers that cannot
be probed directly. The result is one **amplifier datasheet** per amplifier, plus a check of whether the datasheet
explains the amplifiers we already know are odd.

Status: plan written 2026-11-16. Phases 1 and 2 use processed data only; phases 3 and 4 read raw images.

## 1. Data we have

| Data | What it gives | Amplifiers |
|---|---|---|
| **v3n voltage scan** (`processed_v3n`, 127 packets, 2026-11-10) | Board 101: 80 Sobol voltage settings (every voltage moved, timing fixed) → every trace feature and label as a function of every voltage. Board 102: 127 packets **all at nominal** → stability and repeatability. | 101 ch0–3, 102 ch3 (+ dead 102 ch0–2 as controls) |
| **v2 timing + voltage scan** (`processed_v2`, sessions D/E) | Integration time, `delay_H`, `delayRGH` varied → time constants (traps, settling) | 101 ch0–3, 102 |
| **v1** (`processed_v1`) | Earlier settings, repeat stability over weeks | 101, 102 |
| **Pocket-pump runs** (2026-10-12 `PocketPump1`, 2026-10-22 `PP_Testing`) | Trap maps in the pixel array (dipoles), trap density per quadrant | 101, 102 |
| **Skip-noise run** (2026-10-14) | Noise against number of skips (100–10000) → 1/f floor, read-noise model | 101, 102 |
| **Larger images** (400×1500, 250×1000, 1300×6300) | Cosmic tracks (thickness, diffusion), hot columns, CTI per column, dark-current maps | 101, 102 |

**Board 102 has only one working amplifier** (user, 2026-11-16). The data point to **ch3**: it is the only board-102
channel with charge in its pixels (charged-pixel skewness defined only there; cosmic tracks only in its image extension).
Channels 0–2 of board 102 read the electronics without a working amplifier. They are useful as **controls**: whatever
the "datasheet" finds on them comes from the readout chain, not from an amplifier. Caution: the label fit still marks
92–117 of their rows "ok" with plausible gains (1.24–1.29 ADU/e⁻); every earlier analysis that used all of board 102
(the v1/v2 new-module test) mixed dead channels in, and the label quality flag must be fixed to catch this.

Known odd cases (the test set for the diagnosis):

| Case | What is known | What the datasheet should explain |
|---|---|---|
| **101 ch2** | Left out of the models as anomalous since v1 | Which physical quantity is off |
| **101 ch0** | Raises λ with VSUB about 3× less than ch1 and ch3 (+13 vs +43/+47 %/SD) | Amplifier or silicon? Depletion, doping or a dark/CIC difference? |
| **102 ch0, ch1, ch2** | Not working (no amplifier signal) | **Negative controls**: the datasheet must flag all three as non-functional, and show which quantities are pure electronics (present on dead channels too) |
| **102 ch3** | The only working amplifier on the second chip | Chip-to-chip comparison with board 101 |

## 2. What we will estimate

Every estimate is a fit coefficient with a bootstrap uncertainty (resampling packets). Most are **ratios**
(C_x / C_sn, noise in e⁻), because the preamp gain is not calibrated absolutely; ratios are what matter for
performance and for comparing amplifiers.

### Phase 1: electrical datasheet (processed data only)

| # | Quantity | How | Physics |
|---|---|---|---|
| 1.1 | Conversion gain, and **C_sn relative to the other amplifiers** | Image gain (ADU/e⁻) at nominal, mean over the repeated nominal packets | C_sn ∝ 1/gain (same chain) |
| 1.2 | **Clock coupling matrix**: C_k / C_sn for SW, RG, OG/DG, H1–3, V1–3 | Slope of the feedthrough features (`dip per volt SW`, `pulse per volt RG`, H and V peak-to-peak, pre-pulse dip) against that clock's swing in v3, divided by the gain → e⁻ per volt | Overlap capacitance of each gate to the sense node |
| 1.3 | **Output transistor operating point** | d log(gain)/dVDD, d log(gain)/dVR; settling and rise time against VDD; the VR knee | Source-follower transconductance and headroom |
| 1.4 | **Noise model** in e⁻: white density, 1/f coefficient, corner, Allan floor | Trace noise fit / gain; skip-noise run for the floor; dependence on VDD | e_n² ∝ kT/g_m; S_1/f ∝ K_f/(C_ox² W L) |
| 1.5 | **Bandwidth / output RC** | Rise time, recovery τ1/τ2, ringing frequency and amplitude | Output-node R·C, loading |
| 1.6 | **RTS / telegraph noise** (traps in the output FET) | CDS excess kurtosis, baseline steps, discrete-level features; amplitude and switching rate | Single oxide traps |
| 1.7 | **Serial-register charge transfer** | Charge-loss slope, charged-pixel skewness; against `delay_H` in v2 → emission time constants | Register traps |
| 1.8 | **Dark rate vs clock-induced charge** | λ split into dark and CIC (already in the label fit), overscan vs active; CIC per volt of H and V swing | Spurious charge per clock |
| 1.9 | **Stability** | Drift and scatter of gain, white noise, baseline and λ over the repeated nominal packets (board 102: 127 repeats; board 101: 14 + 5); mains-frequency (line) coherence | Aging, temperature, grounding |
| 1.10 | **Thresholds (knees)** | Piecewise-linear fits to each output against each voltage: VSUB knee in λ, VR knee in σe, low-H-high limit, RG-low reset limit | One voltage per amplifier with an error bar |

### Phase 2: physical makeup (inversion with priors)

| # | Quantity | Inputs | Comment |
|---|---|---|---|
| 2.1 | Output FET **gate area W·L**, twice | (a) C_sn from 1.1, (b) the 1/f coefficient from 1.4 | Both need C_ox (oxide thickness, design prior). Agreement checks the model; disagreement points to parasitic capacitance or the oxide |
| 2.2 | Output FET **aspect ratio W/L** | White noise (1.4) and its VDD dependence (1.3) | Needs the mobility (prior) and the drain current (from VDD/VR, known bias network) |
| 2.3 | **Oxide trap density** | K_f from 2.1, and the number and size of RTS traps (1.6) | RTS step ∝ 1/(W·L): a third area estimate |
| 2.4 | **Overlap geometry** around the sense node | Ratios in the coupling matrix (1.2) | Identical designs → identical ratios; differences = mask alignment or oxide thickness |
| 2.5 | **Full-depletion voltage → doping (resistivity)** per quadrant | VSUB knee (1.10) | V_dep ≈ q N_d d² / 2ε, with the known thickness d |
| 2.6 | **Channel potentials** | Knees of the clock rails (1.10) | Implant uniformity between amplifiers |

The inversion is a small Bayesian fit per amplifier: design values as priors, the phase-1 numbers with their errors as
data. It reports which combinations the data determine and which stay at the prior.

### Phase 3: images (raw files, read only)

| # | Quantity | How |
|---|---|---|
| 3.1 | **Thickness and diffusion** | Cosmic-ray (muon) tracks: length against angle → thickness; track width → lateral diffusion → field-free depth. Compare quadrants, and VSUB where images exist at several VSUB |
| 3.2 | **Defect maps** | Hot columns and pixels, dark-current map, per-column CTI from the larger images |
| 3.3 | **Trap density in the array** | Pocket-pump dipoles per quadrant; emission time constants where several pump phase times exist |

### Phase 4: diagnosis

| # | Test | Pass criterion |
|---|---|---|
| 4.1 | Datasheet of the 5 working amplifiers (+ 3 dead controls), side by side; an outlier score per quantity (distance from the others in units of its error) | Every quantity has an error bar; repeated estimates (v1, v2, v3, board-102 nominal repeats) agree within errors |
| 4.2 | Known odd cases | 101 ch2 and 101 ch0 each come out as outliers in at least one quantity, with a physical reason; 102 ch0–2 are flagged as non-functional by a simple, automatic rule (a new label/trace quality check) |
| 4.2b | Electronics vs amplifier | Every quantity is also computed on the dead 102 channels: a quantity that looks the same there is set by the readout chain, not the amplifier |
| 4.3 | Independent routes agree | W·L from C_sn ≈ from 1/f ≈ from RTS; noise in e⁻ from the trace ≈ from the image; coupling from falling ≈ rising edges |
| 4.4 | Chip vs amplifier | Board 102 ch3 against the spread of the four board-101 amplifiers: is the second chip's amplifier inside the within-chip spread? (one amplifier: a first look only) |
| 4.5 | Usefulness for the models | Do the datasheet numbers explain the model errors on a new amplifier (e.g., add them as inputs: does the new-amplifier λ error drop?) |

## 3. Limits

- **Ratios, not absolute values.** No calibrated preamp gain or reference capacitance, so C_sn and the couplings are
  relative unless a design value anchors one of them.
- **Priors matter for phase 2.** The oxide thickness, mobility, K_f and silicon thickness come from design or
  literature values; the fit shows how much the data move them.
- **Few amplifiers.** 5 working amplifiers on 2 chips (4 + 1): enough to estimate each one and flag outliers, too few to learn a latent
  "health space" from the data. That needs more CCDs, ideally including one that has been probed in the lab as ground
  truth.
- **Board 102 did not move its voltages in v3**, so the couplings and knees of 102 ch3 come from v1/v2 (if they varied
  there) or stay unmeasured; its 127 nominal repeats give the best stability numbers we have.
- Some features (RTS, kurtosis) mix amplifier effects with readout electronics; repeated measurements and the
  comparison with board 102 separate them where possible.

## 4. Outputs

- `amp_phys/datasheet.py` → `amp_phys/results/datasheet.csv/.md`: phase 1, every quantity per amplifier with errors.
- `amp_phys/geometry.py` → phase 2 inversion.
- `amp_phys/images.py` → phase 3 (tracks, defect maps, pocket pumping).
- `amp_phys/diagnose.py` → phase 4: outlier scores, the known-case check, figures.
- `amp_phys/REPORT.md` (and a LaTeX version): the results.

## 5. Order of work

1. Phase 1 on v3n board 101 (voltage responses) + 102 ch3 nominal repeats (stability) + the dead 102 channels as
   controls, then v2 for the time constants.
2. Phase 4.1–4.2 right away on the phase-1 numbers: does it already flag the four known cases?
3. Phase 3 images (cosmic tracks first: cheapest, most direct geometry).
4. Phase 2 inversion once priors are collected (design values for oxide thickness, W, L, silicon thickness and
   resistivity: please send any you have).
