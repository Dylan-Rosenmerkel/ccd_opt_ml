# amp_phys: first results (phase 1, v3n data), 2026-11-16

The plan is in [TEST_PLAN.md](TEST_PLAN.md). This is phase 1, run with
`amp_phys/datasheet.py` on `processed_v3n` (127 packets, 2026-11-10/12):
- **Board 101:** the voltage scan (80 settings) and its nominal repeats.
- **Board 102:** 127 packets, all at nominal.

Every number is in `results/v3n/datasheet.md` (value ± bootstrap error over packets). The outlier scores are in
`results/v3n/outlier_z.csv` and `outliers.png`; the knee fits are in `knees.png`.

## 1. Which channels have a working amplifier

Board 102 has one working amplifier, and the data identify it: **ch3**. The rule below was found from the data and is
now automatic. On a channel without an amplifier, the **signal-window settling** (the output's response to the
summing-well / output-gate edge, which needs a live sense node) is about half its normal value.

| channel | signal-window settling (ADU) | ratio to board 101 | working |
|---|---|---|---|
| 101 ch0 / ch1 / ch2 / ch3 | 160 / 168 / 151 / 180 | 0.92–1.10 | yes |
| 102 ch0 / ch1 / ch2 | 84 / 81 / 84 | 0.50–0.51 | **no** |
| 102 ch3 | 202 | 1.24 | yes |

**Warning for earlier work.** The label fit marks 92–98 rows "ok" on the dead channels 102 ch1 and ch2 (only 9 on ch0)
and returns plausible gains (1.24–1.29 ADU/e⁻). 102 ch2 even gets a λ of 1.18. Any earlier result that used all of
board 102 (the v1/v2 new-module test) included dead channels. The label quality flag should use this rule.

## 2. Electronics vs amplifier: what the dead channels teach

A dead channel still goes through the full readout chain. Whatever appears there does not come from the amplifier.

| quantity | dead channels (102 ch0–2) | 102 ch3 (same board, working) | board 101 (working) | reading |
|---|---|---|---|---|
| white noise per skip (ADU) | 4.50–4.78 | 5.06 | 5.59–6.15 | The electronics floor is ~75–85 % of the white-noise **variance**. The amplifier adds ≈2 ADU (102 ch3) to ≈3–4 ADU (board 101) in quadrature |
| null-CDS noise (ADU) | 4.77–4.88 | 5.19 | 6.92–7.97 | Same reading |
| RG pulse per volt (÷ gain, e⁻/V) | 4580–4830 | 4780 | 5980–7010 | **Most of the RG "feedthrough" is pickup in the electronics**, not coupling to the sense node |
| SW dip per volt (÷ gain, e⁻/V) | 655–717 | 926 | 1190–1510 | About half is pickup |
| in-window steps (rate) | 2–3 % | 6.5 % | 0–0.03 % | Board 102's electronics make steps (telegraph-like) on their own; 102 ch3's steps are half electronics |
| rise time (ns) | 505–579 | 594 | 353–436 | Set mostly by board 102's chain, not its amplifier |

(On the dead channels the "gain" is the label fit's meaningless value, 1.24–1.29, close to the real ones; in ADU/V the
SW dip is 850–890 on dead channels against 1640–1970 on board 101, the same ratio.)

Consequences:
- The "clock coupling" features must be corrected for the pickup measured on dead channels before they mean
  C_clock / C_sn.
- On board 102, the RTS-like steps and the slow edges are electronics.
- A spare dead channel per board is a valuable reference; it is worth recording on purpose.

## 3. The board-101 amplifiers

| | ch0 | ch1 | ch2 | ch3 |
|---|---|---|---|---|
| gain (ADU/e⁻) | 1.41 | 1.39 | 1.33 | 1.34 |
| C_sn relative to mean | 0.99 | 1.00 | 1.05 | 1.04 |
| σe (e⁻) at nominal | 0.144 | 0.147 | 0.164 | 0.173 |
| λ (e⁻/pix) at nominal | 0.20 | 0.28 | 0.49 | 0.43 |
| of which CIC | 0.08 | 0.13 | 0.32 | 0.23 |
| λ vs VSUB (%/V) | **1.3** | 4.4 | **0.8** | 4.8 |
| CIC vs VSUB (%/V) | 1.9 | 6.4 | 1.4 | 4.6 |
| overscan λ vs VSUB (%/V) | 2.2 | 3.7 | 0.5 | 2.3 |
| λ knee in VSUB (V) | 54 ± 1 | 56.5 ± 3 | 57.5 ± 2 | 57 ± 5 |
| gain vs VR (%/V) | −2.6 | −0.7 | −1.3 | −3.0 |
| σe knee in VR (V) | **−2.7 ± 0.3** | −3.2 ± 0.3 | −3.35 ± 0.3 | −3.6 ± 0.4 |
| CDS excess kurtosis | −0.05 | −0.03 | **+0.04** | −0.02 |
| σe scatter at nominal (%) | 1.2 | 1.0 | **3.2** | 1.6 |
| white-noise scatter at nominal (%) | 0.8 | 0.9 | **2.3** | **2.2** |

### What it says about the known odd cases

- **101 ch2, excluded since v1, has an RTS (telegraph-noise) signature.** It is the only amplifier with positive excess
  kurtosis of the CDS values (z = +4.5 against the others), the largest jump share of the CDS variance (z = +3.2) and a
  steeper 1/f spectrum (z = +3.2). Its σe and white noise are also the least stable at nominal (3.2 % and 2.3 %
  scatter, against ~1 %). This is the textbook sign of one or a few **charge traps in the output transistor's gate
  oxide**: random switching between levels gives a non-Gaussian, bursty noise that changes from hour to hour. Next
  step: look for two-level switching directly in its raw traces, and measure the switching rate.
- **101 ch0's weak VSUB response is not unique: ch2 has it too.** The amplifiers pair up: ch0 and ch2 respond weakly
  (λ +0.8 to +1.3 %/V), ch1 and ch3 strongly (+4.4 to +4.8 %/V).
  - VSUB acts on λ almost entirely through **clock-induced charge** (CIC +1.4 to +6.4 %/V), not through the dark rate,
    whose VSUB slopes are small and inconsistent.
  - The weak pair is weak in the **overscan** too, i.e. in the serial register, not only in the pixel array.
  - Working hypothesis: the pairing follows the geometry of the chip, i.e. which side of the serial register (or which
    half of the substrate) each amplifier reads. To test: the amplifier-to-quadrant map, and the images (phase 3).
  - ch0 also has the lowest λ knee (54 V) and the highest σe knee in VR (−2.7 V against −3.2 to −3.6). Its reset-drain
    headroom is smaller: its noise starts rising at a less negative VR.
- **101 ch3:** excess slow noise (Allan deviation at 50 skips, z = +4.3) and a less stable white noise (2.2 % scatter).

### Operating point
- Gain falls with VDD by the same amount on all four (−1.5 to −1.7 %/V): the source followers are alike.
- Gain falls with VR by very different amounts (−0.7 to −3.0 %/V). The reset-drain dependence of the conversion gain
  differs between amplifiers; that is a sense-node property (capacitance or reset-transistor operation), and matches the
  spread of the σe knee in VR.

### Common to all four (not amplifier-specific)
- The white noise rises with VSUB by the same 0.11–0.13 %/V on every amplifier: a common effect (substrate bias on the
  shared electronics or the output stage), not a property of one amplifier.
- The σe rise with VSUB follows each amplifier's λ response, not its white noise. Most of what VSUB does to the noise is
  extra charge (CIC), not amplifier noise. This matches the level-1 report: VSUB raises σe 4× more than the trace's
  white noise.

## 4. Board 102 ch3 (the second chip's only amplifier)
Compared with board 101, 102 ch3 has much higher λ (1.35, half dark, half CIC), a slower edge and different ringing,
and more steps. Most of the edge and step differences are also on the dead board-102 channels, so they come from
board 102's electronics. **Its high λ and dark rate are not**: the dead channels show no charge. With one amplifier, this
is a first look only; it does not yet measure chip-to-chip uniformity.

## 5. Next steps
1. **Correct the coupling features for pickup** (subtract the dead-channel level per board) → C_clock / C_sn per clock.
2. **RTS test on 101 ch2:** two-level switching in the raw traces, switching rate and amplitude → trap time constants.
3. **v1 and v2 data:** repeat the datasheet (stability over weeks; v2's timing scan → time constants, charge transfer
   against `delay_H`). Also check how many board-102 rows in the earlier new-module test were dead channels.
4. **Images (phase 3):** cosmic tracks in the 400×1500 and 250×1000 images (thickness, diffusion per quadrant), the
   pocket-pump runs (trap density per quadrant), the skip-noise run (noise floor per amplifier).
5. **Geometry inversion (phase 2):** needs design values: oxide thickness, output FET W and L, silicon thickness and
   resistivity, the preamp gain. Please send any you have.
6. Map amplifiers to quadrants and register sides, to test the ch0/ch2 vs ch1/ch3 hypothesis.

## 6. Absolute calibration from the readout documents (2026-11-16)

Priors from the process recipe, the ACM guide and the frontend schematic are in [priors.toml](priors.toml);
the script is `amp_phys/chain_calibration.py`, with results in `results/v3n/chain_calibration.md` and `sf_gm_scan.csv`.

**Chain gain, measured two independent ways.**
- The frontend loads each video line with 20 kΩ to ground (∥100 kΩ after the coupling capacitor).
- A dead channel therefore reads pure Johnson noise of 16.7 kΩ at 300 K plus the ADA4817 preamp: 17.1 nV/√Hz, or
  5.40 µV per skip for the 10 µs CDS windows.
- The dead channels measure 4.50–4.78 ADU per skip, i.e. **0.853 ± 0.027 ADU/µV**.
- The schematic gives 13.64 × 2.00 (frontend) / 31.25 µV per LSB (18-bit ADC, 8.192 V) = **0.875 ADU/µV**.
- They agree within 2.5 %. That validates the noise model and the trace units, and implies the ACM input buffer has unity
  gain.

**Per amplifier (needs only the chain gain).**

| | 101 ch0 | 101 ch1 | 101 ch2 | 101 ch3 | 102 ch3 |
|---|---|---|---|---|---|
| conversion gain at the video input (µV/e⁻) | 1.62 | 1.59 | 1.52 | 1.53 | 1.51 |
| white noise at the video input (nV/√Hz) | 20.2 | 20.6 | 21.4 | 22.2 | 18.8 |
| above the dead-channel (electronics) floor (nV/√Hz) | 10.8 | 11.5 | 12.8 | 14.2 | 7.7 |

**What the noise says about the output transistor.**
- With a 20 kΩ load at ~140 K, the follower's own thermal channel noise can reach at most ~5 nV/√Hz at the video input,
  for any transconductance. It is therefore a small part of the 19–22 nV/√Hz measured.
- The white noise per skip is set mostly by:
  - the frontend's load network and preamp: 60–83 % of the variance if the follower does not shunt the load;
  - an extra **12–19 nV/√Hz that neither term explains**.
- The extra term is the same order on every amplifier and rises with VSUB by the same 0.11–0.13 %/V on all four. That
  points to noise injected from a bias or substrate supply or the clock lines, not to the transistors.
- **Actionable:** the read noise per skip is limited by the readout (load-resistor noise and supply injection), not by
  the CCD amplifiers.

**What stays open: the follower gain A, and with it C_sn.**
- The noise alone cannot fix g_m. The board-102 rise-time ratio (live 594 ns vs dead 549 ns: no shortening) suggests a
  low g_m (≲ 20–30 µS). That would mean A ≈ 0.2–0.4, a sense-node conversion gain of 4–8 µV/e⁻ and C_sn ≈ 20–40 fF
  (area-equivalent 45–90 µm² of the oxide-nitride gate stack at 0.454 fF/µm²).
- A high g_m (≥ 100 µS) would instead give A ≥ 0.67 and C_sn ≈ 66–90 fF.
- Either of these would decide it:
  - (a) the output-FET W and L from the layout or a higher-magnification image: C_gate ≈ ⅔ C_ox W L + overlaps fixes
    C_sn directly, then A and g_m;
  - (b) a direct follower-gain measurement: step VR by a known amount with the frontend DC-coupled, or inject a known
    step on the video input.

## 7. Depletion and doping (CCD at 130 K, n-type substrate ~10 kΩ·cm; user, 2026-11-16)

- **Doping.** N_d = 1/(q μ_n ρ) = **4.6 × 10¹¹ cm⁻³** at 10 kΩ·cm (μ_n = 1350 cm²/V s, the room-temperature resistivity).
  The range is 2.3–9.3 × 10¹¹ for 20–5 kΩ·cm.
- **Full depletion of 670 µm** needs V = q N_d d² / 2ε = **160 V** at 10 kΩ·cm (80 V at 20 kΩ·cm, 320 V at
  5 kΩ·cm). The ACM's VSUB supply only reaches 90 V.
- **The λ knee at 54–57.5 V** (§3) is where the dark/clock-induced charge starts climbing. If that knee is full
  depletion, then **thickness and resistivity are tied**:
  - 670 µm requires ≈ 29 kΩ·cm;
  - 10 kΩ·cm requires ≈ 390–400 µm (275–285 µm at 5 kΩ·cm, 550–570 µm at 20 kΩ·cm).
  - The cosmic-track thickness (phase 3) closes this: thickness + knee → resistivity (doping) per quadrant.
- **Uniformity.** The knee voltages of the four board-101 amplifiers agree within their errors (54 ± 1, 56.5 ± 3,
  57.5 ± 2, 57 ± 5 V). Any doping difference between the quadrants is below ~10 %. So the ch0/ch2 vs ch1/ch3 difference
  in the VSUB slope **above** the knee is not a doping difference. It sits in how fast clock-induced charge grows once
  depleted, which again points to the serial-register / clock side.
- **Thermal-noise bound at 130 K.** The follower's channel noise is at most 4.9 nV/√Hz at the video input (§6), so the
  conclusion that the readout chain limits the noise is unchanged.

## 8. Output transistor from the layout (user's layout screenshot, 2026-11-16)

**Scale.** The scale bar is 40 µm in two 20 µm halves (~95 px each). The serial-register period, 15 µm per pixel, is
~75 px. Both give ~5 px/µm (±5 %), and they confirm the 15 µm pixel.

**Geometry** (layer identification inferred, to be confirmed):
- Output FET gate: the poly strip between the source and drain contact columns, **L ≈ 6 µm, W ≈ 55–60 µm**
  (W/L ≈ 10, W·L ≈ 330–360 µm²). C_ox·W·L = 150–163 fF; in saturation C_gs ≈ ⅔ of that = 100–109 fF.
- Reset FET ≈ 6 × 6 µm.
- The gate poly continues ~100 µm to the end of the serial register. Its wiring capacitance adds to the sense node.

**Closing the follower inversion.** In a source follower the sense node sees C_sn = C_fixed + C_gs (1 − A).
C_fixed = gate–drain overlap + floating diffusion + wiring. The measured conversion gain at the video input,
1.52–1.62 µV/e⁻ = A q / C_sn, then gives A as a function of C_fixed:

| C_fixed (fF) | follower gain A | C_sn (fF) | sense-node conversion gain (µV/e⁻) | g_m (µS, 20 kΩ load) |
|---|---|---|---|---|
| 20 | 0.60 | 62 | 2.60 | 75 |
| 40 | 0.70 | 72 | 2.24 | 115 |
| 60 | 0.79 | 82 | 1.97 | 193 |
| 80 | 0.89 | 91 | 1.75 | 406 |

Independently, device physics for this geometry (buried p-channel at 130 K, hole mobility 1000–2500 cm²/V s, effective
gate coupling 0.14–0.45 fF/µm² including the channel depth, 100–250 µA through the 20 kΩ load) gives
**g_m ≈ 170–750 µS → A ≈ 0.77–0.94**.

**Consistent picture:** A ≈ 0.8–0.9, g_m ≈ 200–400 µS, **C_sn ≈ 80–90 fF**, a sense-node conversion gain of
**≈ 1.8–2.0 µV/e⁻**, and C_fixed ≈ 60–80 fF. That much fixed capacitance is plausible for ~100 µm of sense line plus the
diffusion and overlaps. Retracted: the low-g_m reading from the board-102 rise times (§6). The dead channels' edges
are pickup, and their shape does not measure the follower's output resistance.

**What follows.**
- **Noise.** With g_m ≈ 200–400 µS the follower's output resistance (2.5–5 kΩ) does shunt the load resistor. The live
  electronics floor is then only ~5–7 nV/√Hz, and the follower's channel noise ~3–4 nV/√Hz. Most of the measured
  19–22 nV/√Hz at the video input is the **unexplained term (~18–20 nV/√Hz, ≈ 10 nV/√Hz·(1/A) at the sense node)**. That
  term is common to all amplifiers and rises with VSUB, so supply, substrate or clock injection is the leading candidate.
  This is now the main noise lead.
- **Amplifier-to-amplifier differences.** The relative C_sn differs by up to 5 % (ch2/ch3 vs ch0/ch1). That corresponds to
  ~5 % in gate area, oxide thickness or follower gain: process variation of the size the recipe allows.
- **Next.** The 1/f coefficient K_f from the 1/f corner with W·L known gives an oxide trap density per amplifier. This
  should come out high for 101 ch2 (its RTS signature).

## 9. Physics equations, tested leave one amplifier out (`physics_model.py`)

Fit on three board-101 amplifiers, predict the fourth (v3 G+H). Errors are RMS of log(pred/true) in %.
"One image" = one nominal image of the new amplifier.

| target | equation | no image | one image | best ML, new amplifier |
|---|---|---|---|---|
| gain | ln g = ln g_nom + b_VDD ΔVDD + b_VR ΔVR + c ΔVR² (3 shared constants) | **3.5** | 3.8 | 3.7 |
| λ | ln λ = ln λ_nom + β ΔVSUB + γ max(0, VSUB − 56) + Σ b_k ΔV_k | 57 | **32** | 40 |
| σe | σe = √(s² σ_read,trace² + κ λ g²) / g (2 global constants) | 20 (9–14 on ch0/ch1, 21–35 on ch2/ch3) | 19 | 11.9 |

- **Gain:** three physical constants match the ML. The source-follower slopes are universal: VDD −1.55 to −1.62 %/V
  whichever amplifier is left out. The VR slope varies (−1.4 to −2.3 %/V).
- **λ:** without an image, physics is no better than ML, because each channel reads its own CCD (image agent). With a
  single nominal image of the new CCD, the shared voltage slopes predict all its other settings to 32 %, better than any
  ML model without that image.
- **σe:** the equation works where the noise is well behaved (ch0, ch1) and fails exactly on the two amplifiers with
  anomalous slow noise (ch2: RTS; ch3: Allan excess). The failure is itself a health flag: extrapolating the trace's
  Allan deviation to 500 skips is valid only for white + stationary 1/f noise.
- **Negative results.**
  - Counting single electrons inside the traces (`trace_electrons.py`) does not work with the current traces.
  - The trace pixels' means vary by ~4 ADU between repeat traces on live amplifiers, against 0.15 ADU on dead channels.
    So they carry varying charge or level, not clean empty pixels.
  - A direct "image noise from traces" (`trace_read_noise.py`) works on dead channels (0.19 ADU = image) but not yet on
    live ones.

## 10. Resistivity from thickness + depletion (image agent + v3)

- The muons give 615–655 µm thickness, and the diffusion profile shows full depletion (field-free layer ≲ 50 µm).
- If full depletion holds at VSUB ≤ 45–55 V (v3 nominal 45 V), then N_d ≤ 1.3–1.9 × 10¹¹ cm⁻³, i.e. **ρ ≳ 25–35 kΩ·cm**.
  That is above the "order 10 kΩ·cm" spec.
- The λ knee at 54–57 V is then **not** full depletion. It is a serial-register effect (the image agent: the VSUB
  response sits in the serial-register dark and the overscan, not the array).
- To check: the VSUB of the October track images.

## 11. The hidden noise is common to the setup (`common_mode.py`, `cms_images.py`)

**The test.** The ACM digitises its four channels simultaneously, so shared noise shows up as a correlation between
channels, skip by skip. 475 simultaneous trace sets, pixel means removed:
- **Board 101:** the four channels correlate at 0.52–0.56, so **~50 % of each channel's per-skip noise variance is
  common**.
- **Board 102:** 12–15 % common.
- **Between the boards:** **−0.3** (anti-correlated) across two ACMs. That is a ground-loop-like coupling between the
  modules.
- **Frequency:** the shared part sits below ~0.05 cycles/skip ≈ 1.5 kHz (coherence 0.3–0.5; ~0.02 above).

**Source.** With all 18 voltages regressed out, the board-101 common share rises with the **RG high rail
(+17 percentage points/V)**, falls with **VR (−16 pp/V)**, and rises with the H1 high rail. It barely follows VSUB.
Hypothesis: noise on the DC-held clock rails (RG, H high during the skips) couples through the gate overlaps into every
sense node. The rails are shared and only filtered at 33 ns.

**What subtracting it does.** Subtracting the common mode across the simultaneous amplifiers:
- **v3 images (500 skips): no gain.** The 500-skip average and the row baseline already suppress the shared band.
- **October skip-noise images: 7–14 % lower at ≤ 500 skips and 8–30 % at 1000–10000 skips**, where the slow floor
  dominates. One image per skip count, so these numbers are rough.
- The October/November difference at 500 skips suggests that the noise environment changed between the runs.

**Noise thermometry** on the dead channels is swamped by the same injection (±1.6 % variance ↔ ±5 K apparent). It
becomes usable only once the common mode is removed.

**Plan.** The grounding / filtering study is written up in [NEXT_RUN_PLAN.md](NEXT_RUN_PLAN.md) §A, with
`common_mode.py` as the per-configuration metric.

## 12. Channel-potential knees per CCD (`channel_knees.py`, v3 scan)

Hinge fits of each output against each rail, with all other rails regressed out. Kept: p < 0.01, found on ≥ 2 CCDs
(`results/v3n/channel_knees.md`).

| knee | CCD_D (ch0) | CCD_C (ch1) | CCD_A (ch2) | CCD_B (ch3) | reading |
|---|---|---|---|---|---|
| H3 low rail (white noise; H3_A and H3_B agree) | 6.2–6.4 V | 6.25–6.45 V | – | **5.35–5.55 V** | where the serial-register barrier under H3 starts to matter |
| VR (σe, gain) | −2.45 to −2.66 V | – | −2.95 V | −2.94 to −3.62 V | reset-drain headroom (reset-FET channel potential) |
| VSUB (λ) | 54.6 ± 0.8 V | 56.9 ± 2.5 V | – | 56.9 ± 7 V | serial-register onset (§10: not depletion) |
| V2 low (CIC) | 8.18 ± 0.06 V | – | – | 8.06 ± 0.07 V | vertical-clock CIC onset |

**New quantity: a per-CCD channel-potential offset.**
- The H3-low knee of CCD_B sits **0.85 ± 0.3 V** below CCD_D and CCD_C.
- For a buried channel, the potential shifts by ΔV ≈ q ΔN_s (t_eq/ε_ox + x_j/ε_Si). Here t_eq = 50 nm oxide + 50 nm
  nitride ≈ 76 nm oxide-equivalent, and x_j ≈ 0.45 µm (B, 150 keV), giving ≈ 1.05 × 10⁻¹¹ V cm².
- So 0.85 V ≈ **8 × 10¹⁰ cm⁻² ≈ 0.6 % of the 1.3 × 10¹³ cm⁻² buried-channel dose**, or the same amount of fixed oxide
  charge.
- This is a process-variation measurement that would otherwise need SIMS or C–V on a broken-out die.
- Precision is limited by the Sobol design (every rail moves at once). Single-rail stepping (NEXT_RUN_PLAN B4) gives
  ±0.05 V.

## 13. The cabling change, seen in the data (`common_mode.py` on v1, v2, v3)

The user confirms the grounding/cabling changed. Nominal reference traces only, board 101, per-skip variance split into
the part shared with the other channels and the amplifier's own:

| run | dates | shared share | shared variance (ADU²) | own variance (ADU²) | per-skip RMS | dead-channel RMS (board 102) |
|---|---|---|---|---|---|---|
| v1 | Nov 2–3 | 61.2 % | 68.8 | 43.6 | 10.60 | 5.42 |
| v2 | Nov 5–7 | 60.7 % | 68.0 | 43.8 | 10.54 | 5.44 |
| v3 | Nov 10–12 | 52.5 % | **47.0** | 42.5 | **9.46** | 5.41 |

- **What changed.** Between Nov 7 and Nov 10 the shared variance fell by a third. Each amplifier's own noise is
  unchanged (43.6 → 42.5), and the board-102 electronics are unchanged (5.42 → 5.41). This is the signature of a
  setup (grounding) improvement, not of the amplifiers, and it shows `common_mode.py` works as a grounding metric.
- **What is left.** Removing the remaining shared part would bring the per-skip RMS from 9.46 to ~6.5 ADU (−31 %).

## 14. Crosstalk between the channels (`crosstalk.py`, October board-101 images)

Charge read on one amplifier appears simultaneously on the others at the ~10⁻⁴ level. Target e⁻ per 10⁴ source e⁻:

| source \ target | ch0 | ch1 | ch2 | ch3 |
|---|---|---|---|---|
| ch0 | – | +1.2 | **−1.6** | +1.2 |
| ch1 | +1.2 | – | +0.6 | **−0.7** |
| ch2 | **−0.6** | +1.4 | – | +1.4 |
| ch3 | +0.6 | **−0.9** | +1.4 | – |

- **The pattern.** The pairs ch0↔ch2 and ch1↔ch3 couple with negative sign; every other pair couples positively.
  That points to the routing (the cable or connector pairing of the video lines, or a shared return), not to the CCDs.
- **Why it matters.** A 10⁴ e⁻ track makes ±1 e⁻ ghosts on the other CCDs. That can fake single-electron events
  unless it is subtracted, and the matrix makes subtraction possible.

## 15. Clock → sense-node coupling capacitances (absolute, with C_sn known)

Feedthrough per volt (in equivalent electrons) minus the pickup seen on the dead channels, times q, gives the
capacitance of each clock to the sense node. The method assumes board 102's pickup equals board 101's.

| | ch0 | ch1 | ch2 | ch3 |
|---|---|---|---|---|
| RG → sense node (fF) | 0.26 | 0.21 | **0.37** | 0.21 |
| SW → sense node (fF) | 0.12 | 0.08 | 0.13 | 0.10 |
| H → first skip (fF) | 0.009 | 0.010 | 0.010 | 0.010 |

- **A new process quantity.** 0.21–0.37 fF over the 6 µm-wide reset gate at 0.454 fF/µm² is an overlap of
  ≈ 0.08–0.14 µm. That is the lateral diffusion / alignment of the sense-node implant under RG, which is otherwise
  only measurable in cross-section.
- **101 ch2 has ~1.7× the RG overlap of the others.** That is consistent with its other anomalies (RTS, noise, glow
  at the amplifier end).

## 16. Doing this on a new amplifier in another setup

What transfers, and what each step needs:

| step | needs | transfers to a new setup if |
|---|---|---|
| chain calibration (ADU/µV) | one channel with a known passive input: an unconnected channel, a dead amplifier, or a switchable known resistor | the frontend's load and preamp are known (schematic) |
| conversion gain, injected noise, follower inversion | the chain calibration, the layout W/L, C_ox (recipe), the CCD temperature | similar geometry: same equations, new priors |
| common-mode / grounding metric | ≥ 2 channels digitised simultaneously | always |
| crosstalk matrix | one image with tracks | always |
| clock couplings (fF) | feedthrough features + a pickup reference (dead channel) | always |
| knees / channel potentials | single-rail voltage steps | always; the dose interpretation needs the recipe |
| gain / dark-charge equations | the 3 shared slopes (VDD −1.6 %/V is universal here) + one image of the new CCD | same transistor design; refit the slopes for a new design |
| transconductance per amplifier | the DC output level (a voltmeter on the video line, or the VDD supply current) | always; one bench reading closes it: g_m = 2 I_D / V_ov with I_D = V_out / R_L |

The ML models do not transfer this way: they learn amplifier identities. The physics quantities are defined
independently of the setup, so a new setup only needs new priors (schematic, layout, recipe), not new training data.

## 17. Mutual capacitances between the channels (`crosstalk.py` at three CDS windows, `mutual_capacitance.py`)

**Method.** v2 varied the CDS window (delayInt). The board-101 crosstalk was measured on images grouped by window:
short (mean 441 cycles, 21 images), nominal (1000, 16), long (2046, 22). The nominal matrix reproduces the October one
(e.g. ch0→ch2 −1.54 vs −1.6 × 10⁻⁴), so it is stable across runs.

**Model per pair:** k(τ) = a/τ + b.
- **a/τ: capacitance between the video lines after the followers.** A source step injects a pulse of area
  R_out·C_m·ΔV into the target line, and the CDS window averages it, so C_m = a / R_out. R_out = 2.5–5 kΩ (follower
  ‖ 20 kΩ load, §8).
- **b: τ-independent.**
  - Charge sharing at the sense nodes would be positive, with C_m,sn = b·C_sn.
  - A shared return impedance gives a negative b, with Z = −b·R_L.

| pair (both directions) | C_m between video lines (pF) | τ-independent b (10⁻⁴) | reading of b |
|---|---|---|---|
| ch0–ch1 | 0.2–0.6 | ≈ 0 | – |
| ch2–ch3 | 0.24–0.85 | ≈ 0 (−0.6 ± 1.4, +0.4 ± 0.7) | – |
| ch0–ch3 | 0.08–0.35 | ≈ 0 | – |
| ch1–ch2 | 0.11–0.26 | ≈ 0 | – |
| ch0–ch2 | 0.11–0.41 | **−2.0 ± 0.9, −1.2 ± 0.4** | shared return **Z ≈ 2–4 Ω** |
| ch1–ch3 | 0.06–0.16 | **−1.3 ± 0.7, −1.3 ± 0.5** | shared return **Z ≈ 2.6 Ω** |

**Readings.**
- The neighbouring lines (ch0–ch1, ch2–ch3) have the largest capacitance, ~0.3–0.6 pF. That is consistent with
  adjacent conductors over a cable / flex length.
- ch0/ch2 and ch1/ch3 share a few-ohm return path. That is a grounding defect with a number on it, and it is the same
  pairing as the negative entries of the crosstalk matrix.
- Sense-node (in-cryostat) mutual capacitance: no positive τ-independent term is seen. So C_m,sn ≲ 0.01 fF
  (|b| ≲ 1 × 10⁻⁴ × 85 fF): the CCDs do not couple inside the package at a measurable level.

**Caveats.**
- The a terms carry ~30–100 % errors: 3 window groups, and the other settings also vary between the groups.
- The model assumes the coupled pulse decays well within the window (τ_line ≈ R_out × C_line ≲ 1 µs ≪ 3–20 µs).

**To pin it down.** One image per window length at fixed nominal voltages (5 windows, 300–3000 cycles), plus a
short-circuit / cable-swap test of the ch0/ch2 and ch1/ch3 returns.
