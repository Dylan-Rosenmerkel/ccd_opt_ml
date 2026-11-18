# What we can characterise about a CCD amplifier from its traces (state 2026-11-16)

This summary covers: what each quantity is, how it is measured, the value on PA03 / board 101, whether the method is
standard or new, and what it is used for. Details and caveats are in [REPORT.md](REPORT.md) (§ numbers below); the
next measurements are in [NEXT_RUN_PLAN.md](NEXT_RUN_PLAN.md).

**Legend.** "Novel" = a measurement or method not part of standard CCD characterisation, as far as we know.
"Standard" = known quantities, now automated per trace with error bars.

> **Correction (REPORT §23):** the per-channel traces are separate acquisitions (~1 s apart), so the cross-channel
> trace quantities (#12 common share, the trace-based setup index) describe a sequence-locked pattern, not simultaneous
> setup noise. Setup noise is measured from the simultaneous images: October ρ_sim ≈ 0.25 → November ≈ 0.05, σ_0e −30 %
> on ch0/ch1 after the cabling change. Use ρ_sim / the image CMS gain until traces are recorded simultaneously.

## 1. From traces alone (no image, no documents)

| # | quantity | value (board 101, v3) | method | novel? | used for |
|---|---|---|---|---|---|
| 1 | Amplifier alive / dead | dead channels: signal-window settling 82–84 ADU vs 150–202 live | automatic rule (§1) | novel rule (data-derived) | flags dead or disconnected outputs; catches false "label ok" |
| 2 | White noise per skip | 5.6–6.2 ADU (≈ 4 e⁻) | trace noise fit | standard | noise budget |
| 3 | 1/f corner, exponent, Allan floor | per amplifier (§3) | spectra / Allan deviation | standard | slow noise; where skipping stops helping |
| 4 | Telegraph (RTS) noise | 101 ch2 only (CDS kurtosis z = +4.5, jump share, steeper 1/f) | discreteness features (§3) | standard features, new use as a flag | single oxide traps in the output FET |
| 5 | Stability at nominal | gain ±0.8 %, λ drift +0.08…+0.21 %/h | repeats over 49 h (§1.9) | standard | aging / health baseline |
| 6 | Output-FET operating point | gain vs VDD −1.55…−1.62 %/V (same on all four); vs VR −0.7…−3.0 %/V (differs) | regression on the voltage scan (§3) | novel as a per-amplifier datasheet | follower transconductance; threshold differences |
| 7 | Reset headroom | σe/gain knee in VR: −2.5 to −3.8 V | hinge fits (§12) | novel | reset-transistor channel potential |
| 8 | Channel-potential knees | H3-low knee 0.85 V lower on CCD_B (ch3) | hinge fits with the other 17 rails regressed out (§12) | **novel** | implant-dose / oxide-charge variation between CCDs (≈ 0.6 % of the dose) |
| 9 | Clock → sense-node couplings | RG 0.15–0.37 fF, SW 0.08–0.13 fF, H1–3 0.011–0.016 fF, V1–3 < 0.003 fF | edges in the raw traces, regressed across settings, minus dead-channel pickup (§15, §19, §20) | **novel** (per-line capacitances from operating traces) | which lines can inject noise; overlap geometry |
| 10 | Pickup vs real coupling | RG ≈ 70 %, SW ≈ 50 % of the "feedthrough" is board pickup; H transfers ≈ 90 % real | dead channels as electronics-only controls (§2, §20) | **novel** | separates the board from the CCD |
| 11 | Reset charge injection | 2.1–2.4 fF per volt of (RG_high − VR) | reset edges vs rails (§20) | **novel use** | → buried-channel depth (§4 below) |
| 12 | Common slow wander (setup) | 61 % → 52 % of the per-skip variance after the cabling change; white part uncorrelated | simultaneous channels, with/without per-pixel detrend (§11, §13, §18) | **novel as a grounding metric** | grounding/cabling tests (§3 below) |
| 13 | Crosstalk matrix | ±0.6…1.6 × 10⁻⁴ between channels | tracks in images (§14) | standard idea, new decomposition | ghost subtraction; sub-electron searches |
| 14 | Video-line mutual capacitance | 0.1–0.85 pF per pair (largest for neighbouring lines) | crosstalk vs CDS window: a/τ term (§17) | **novel** | cable / flex layout |
| 15 | Shared return impedance | ≈ 2–4 Ω between ch0/ch2 and ch1/ch3 | crosstalk vs CDS window: τ-independent negative term (§17) | **novel** | a specific grounding defect to fix |
| 16 | Sense-node mutual capacitance in the package | ≲ 0.01 fF (not seen) | same fit (§17) | novel bound | the CCDs do not couple inside the package |

## 2. Traces + documents (schematics, recipe, layout)

| # | quantity | value | method | novel? | used for |
|---|---|---|---|---|---|
| 17 | Readout-chain gain | 0.853 ± 0.027 ADU/µV (design 0.875) | **dead-channel Johnson noise of the 20 kΩ load as a built-in reference** (§6) | **novel** | absolute calibration with no injection hardware |
| 18 | Conversion gain at the video input | 1.51–1.62 µV/e⁻ | gain / chain gain | new (absolute without a source) | absolute sensitivity |
| 19 | Input-referred white noise | 19–22 nV/√Hz; electronics alone (open) 17.1 | chain gain | new (absolute) | noise budget in physical units |
| 20 | Sense-node capacitance, follower gain, g_m | C_sn ≈ 80–90 fF, A ≈ 0.8–0.9, g_m ≈ 200–400 µS (§8) | layout W/L + the bootstrapping relation C_sn = C_fixed + C_gs(1 − A) + device physics | **novel route** | amplifier design check. Still open: one DC reading (§5) decides between this and a low-g_m branch |
| 21 | Gate vs wiring capacitance | C_gs ≈ 100 fF vs C_fixed ≈ 60–80 fF | same | **novel** | where the conversion gain is lost (the ~100 µm sense line) |
| 22 | Overlap length under RG | 0.08–0.14 µm (ch2 ≈ 1.7× the others) | RG coupling / (C_ox × gate width) (§15) | **novel** | lateral diffusion / mask alignment without cross-sections |
| 23 | Buried-channel depth | ≈ 0.6 µm (range 0.3–0.9) | reset charge injection (§4 below) | **novel** | channel depth without SIMS |
| 24 | Thermal-noise bound of the output FET | ≤ 4.9 nV/√Hz at the input (130 K, 20 kΩ load) | physics | new | proves the readout, not the FET's thermal noise, dominates |
| 25 | Sensitivity of each line as a noise injector | VDD 0.6, OG 5, RG 7, SW 19, VSUB ~60 µV/√Hz needed | C_sn / C_line (§19) | **novel** | which supplies must be quiet |

## 3. Noise: what is intrinsic to the CCD and what is the setup

| component | size | intrinsic or setup? | evidence |
|---|---|---|---|
| Frontend load (20 kΩ Johnson) + preamp | 17.1 nV/√Hz with the input open; on a live channel reduced by the follower's shunting (to ~5–7 nV/√Hz if g_m ≈ 300 µS, ~13–15 if g_m ≈ 20 µS) | **setup** (frontend board) | dead channels; schematic |
| Output-FET channel thermal noise | ≤ 4.9 nV/√Hz at the input | **intrinsic** | physics bound |
| Unexplained white term | ≈ 11–14 nV/√Hz above the open floor; uncorrelated between channels | **per channel**: either the load resistor (if g_m is low) or excess FET noise; not a shared supply/ground | channels uncorrelated after detrend (§18) |
| 1/f, Allan floor | per amplifier; ch3 has an excess | **intrinsic** (oxide traps) | trace spectra |
| RTS / telegraph noise | 101 ch2 only | **intrinsic** (a trap in ch2's output FET) | kurtosis, jumps, instability |
| Slow common wander / droop (< ~250 Hz) | 52 % of the per-skip variance (was 61 % before the cabling change) | **setup** (grounding/cabling, AC coupling) | correlated on all channels; anti-correlated between boards; fell with the cabling change |
| Clock pickup on the video lines | RG ≈ 70 %, SW ≈ 50 % of the edges | **setup** (board/cable) | dead channels |
| Crosstalk via video lines + shared return | 10⁻⁴; 0.1–0.85 pF; 2–4 Ω | **setup** (cable/flex/grounding) | window dependence |
| Charge noise in the image (dark + clock-induced charge) | the part of σe that follows λ (VSUB, serial register) | **intrinsic to each CCD** (silicon / serial register) | σe ∝ λ; serial-register VSUB response per CCD |
| Reset charge injection | 2.1–2.4 fF/V, deterministic | intrinsic, but **cancelled by CDS** | equal in both windows |

**The decisive open measurement:** the DC level of each video line (or the VDD current). It gives the follower current,
hence g_m. That splits the unexplained white term into "load resistor (setup, fixable on the frontend)" or "excess
FET noise (intrinsic)".

## 4. The grounding / noise-reduction toolkit

Per configuration, one simultaneous trace set on all channels of both boards (~1 min), then:

| metric | script | what it tests | good value |
|---|---|---|---|
| slow common share per board | `common_mode.py` (mean only) | grounding/cabling loops, AC-coupling droop | → 0 (now 52 %) |
| cross-board correlation and sign | `common_mode.py` | a loop between the two boards / modules | → 0 (now −0.3) |
| white common share | `common_mode.py --detrend 3` | supply/substrate/clock injection (shared) | stays ≈ 0 |
| coherence spectrum | `common_mode.py` | which frequencies are shared (mains, regulators) | flat ≈ 0 |
| dead-channel excess over 17.1 nV/√Hz | `chain_calibration.py` | injection into the frontend itself | → 0 |
| per-skip white noise in nV/√Hz | `chain_calibration.py` | the total, in physical units | ↓ toward the amplifier floor |
| crosstalk + shared-return Z | `crosstalk.py` + `mutual_capacitance.py` | return paths (ch0/ch2, ch1/ch3 now 2–4 Ω) | Z → 0 |
| high-skip floor with/without common-mode subtraction | `cms_images.py` | whether the slow wander still limits | no gain from subtraction |

## 5. The buried-channel depth (§20)

**What is measured.** At every reset, the output jumps by an amount that depends only on RG_high − VR: equal and
opposite coefficients, 2.1–2.4 fF-equivalent per volt, the same on all four CCDs within ±7 %.

**Why that is charge injection.**
- The reset transistor is a buried p-channel MOSFET (6 × 6 µm, layout).
- While it is on, its channel holds mobile charge Q = C_eff · W · L · (gate overdrive). The overdrive is set by the
  gate rail (RG_high) relative to the channel potential, which is tied to VR.
- When it switches off, that charge leaves through the two ends, roughly half into the sense node and half into VR.
- So the sense node receives ½ C_eff W L per volt of (RG_high − VR). That is exactly the measured dependence, and it
  scales with the gate area, not with an edge overlap.

**Why C_eff measures the depth.**
- In a buried channel the charge does not sit at the oxide surface (as in an ordinary MOSFET). It sits at a potential
  maximum a depth x_ch inside the silicon (holes, in this p-channel device).
- The gate therefore couples to it through two capacitors in series: the gate dielectric (50 nm oxide + 50 nm nitride
  = 76 nm oxide-equivalent, 0.454 fF/µm²) and the silicon between the surface and the channel (ε_Si / x_ch):

      1 / C_eff = t_eq / ε_ox + x_ch / ε_Si

- Measured: ½ C_eff × 36 µm² = 2.25 fF → C_eff ≈ 0.125 fF/µm².
- So x_ch / ε_Si = 1/0.125 − 1/0.454 µm²/fF → x_ch ≈ 1.036 pF/cm / (0.172 fF/µm²) ≈ **0.60 µm**.

**Check against the recipe.** The buried channel is a boron implant at 150 keV, 1.3 × 10¹³ cm⁻², with a projected
range ≈ 0.45 µm and straggle ≈ 0.1 µm. The later oxidations and RTAs diffuse it deeper. A channel maximum at 0.5–0.7 µm
is what the process should give.

**Uncertainty.** The split of the injected charge between the sense node and VR depends on the edge speed and on the
impedances at the two ends.
- A 30–70 % split gives x_ch ≈ 0.3–0.9 µm.
- The gate area is read from the layout (±10 %).
- Pinning the split: vary the RG edge speed (an RC on RG), or compare with the channel-potential knees, which depend
  on the dose × depth (§12).

**Why it matters.** The channel depth sets the full well, the channel potential vs gate voltage (the knees), the
transconductance (deeper means less gate coupling) and the 1/f noise (deeper means farther from the oxide traps).
Measured per CCD, it tracks implant-energy and anneal variation between wafers, from data taken during normal
operation and without sectioning a die.

## 6. Retracted or corrected along the way
- "Low g_m from the board-102 rise times" (§6) → superseded by the layout inversion (§8). The DC-level measurement
  still decides it.
- "~50 % of the white noise is shared setup noise" (§11) → it is slow wander. The white noise is uncorrelated (§18).
- "102 ch3 is the working amplifier" (§1) → it settles like a live amplifier, but no charge reaches it (images, H
  transfers). In the October images only 102 ch2 collects charge. The mapping is to be checked.
- Single-electron counting inside the current traces does not work (the trace pixels carry varying charge or level).
