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

## 18. CORRECTION to §11 and §13: the "common-mode noise" is slow baseline wander, not white noise

`common_mode.py --detrend 3` removes a cubic per pixel (the AC-coupling droop and wander below ~250 Hz; the frontend's
100 nF / 100 kΩ coupling has τ ≈ 10 ms ≈ one 500-skip pixel). After that, the channels are **uncorrelated**:

| run | per-pixel mean removed: common share / per-skip RMS | cubic removed: common share / per-skip RMS |
|---|---|---|
| v1 (Nov 2–3) | 61 % / 10.60 ADU | 0.1 % / 5.79 ADU |
| v3 (Nov 10–12) | 52 % / 9.46 ADU | 0.1 % / 5.75 ADU |

What this means:
- **The shared part is slow** baseline droop/wander, common to all channels.
  - Its size tracks the in-trace drift (ρ = +0.80) and RG_high − VR (the video DC level through the reset), with equal
    and opposite RG_H (+17) and VR (−15 pp/V) coefficients.
  - It is mostly removed by the per-pixel/row baseline. That is why common-mode subtraction gives nothing at 500 skips
    and 8–30 % at 1000–10000 skips, where a pixel lasts long enough for the slow wander to become pixel-to-pixel noise.
- **The cabling change (between Nov 7 and Nov 10) reduced the slow wander by about a third.** The white noise is
  unchanged.
- **The white per-skip noise (5.8 ADU, 19–22 nV/√Hz at the input) has no channel-to-channel correlation.** The
  unexplained white term of §6/§8 is therefore per channel, not a shared supply/substrate/ground injection.
- **Leading candidate now:** the frontend's 20 kΩ load Johnson noise, insufficiently shunted by the follower if its
  drain current is low. That reopens the low-g_m branch of §6.
- **Deciding measurement:** the DC level of each video line (or the VDD current) → I_D → g_m.
- **Grounding metric:** report both bands. Slow-wander share (mean only) for grounding/cabling, white share (detrended)
  for supply injection. Both are in `common_mode.py`.

## 19. Mutual capacitances between the sense node and the voltage lines (`sense_node_coupling.py`)

| line | C to sense node | how | status |
|---|---|---|---|
| RG (reset gate) | **0.21–0.37 fF** (ch2 0.37) | edge feedthrough minus dead-channel pickup (§15) | measured |
| SW (summing well) | **0.08–0.13 fF** | same | measured |
| H clocks (first skip) | **0.009–0.010 fF** | same | measured |
| all lines, settled level | ≤ 0.01–0.03 fF effective | regression of the empty-skip CDS on all 18 rails | CDS cancels every line's step (same level in both windows): by design |
| OG (output gate) | ~0.2–0.6 fF (estimate) | layout: adjacent gate, the measured RG overlap length 0.08–0.14 µm | not varied in v3 |
| VDD / drain (output FET C_gd) | **~2–4 fF** (estimate) | W = 55–60 µm × the same overlap × 0.454 fF/µm² | not varied with a switching edge; the largest coupling |
| VR (reset drain) | ≪ 0.01 fF with RG off | only through the reset channel | – |
| VSUB (substrate) | ~0.01–0.03 fF (estimate) | sense-node area / 635 µm of depleted Si | – |
| DG, TG, V clocks | – | not resolvable: not switching in the skip cycle, or not varied | – |

Edge amplitudes per volt of each rail (pulse and dip heights, regressed on all rails) are in
`results/v3n/sense_node_coupling.md`. They include cable pickup, and the lockstep A/B rails limit the separation.

**Noise sensitivity per line** = injected noise × C_sn / C_line. The source noise each line would need to make
22 nV/√Hz at the sense node:

| line | needed noise |
|---|---|
| VDD/drain | **0.6 µV/√Hz** |
| OG | 5 µV/√Hz |
| RG | 7 µV/√Hz |
| SW | 19 µV/√Hz |
| VSUB | ~60 µV/√Hz |
| H | ~190 µV/√Hz |

VDD/drain is the most sensitive line. Any per-channel noise on VDD1/VDD2 (each regulator feeds two channels; their
noise would correlate within a pair, which §18 does not see) would have to stay below ~0.6 µV/√Hz.

**How to measure every line directly ("capacitance tomography").** During one trace set, dither each line with a small
square wave at its own frequency (e.g. 20 mV, frequencies 1–20 kHz chosen not to alias with the skip rate), then
lock-in demodulate the CDS at each frequency. The amplitude at f_k / the dither amplitude = A·C_k/C_sn. That gives the
full sense-node capacitance vector in one acquisition, including DC-held lines (VDD, VR, VSUB, OG, DG, TG). It needs a
firmware option to add a small modulation to the DAC / pot settings.

## 20. H and V clock edges read directly from the raw traces (`clock_edges.py`, `v_edges.py`)

**Method.** The raw trace samples carry the sequencer flags: bit 6 = horizontal transfer, bit 7 = vertical transfer.
(The pipeline's H/V features are empty for v3: its region finder did not handle this run's trace format.)
- Per trace: the median waveform around each transfer, and the fast jump at each edge (samples +3..+12 minus
  −12..−3).
- That jump is regressed, across the 80 v3 settings, on all varied rails.
- C (fF-equivalent at the sense node) = |ADU/V| / gain × q.

**The horizontal-transfer waveform** (`clock_edges.png`):
- On board 101 it swings ±40 000 ADU (≈ ±47 mV at the input), relaxing with τ ≈ 5–9 µs after each edge.
- The board-102 dead channels see ~6–10 % of that (pickup), so most of the response comes through the CCD amplifier.
- 102 ch3 looks like the dead channels, consistent with the image agent (no charge reaches it).

| line → sense node | result | edge(s) |
|---|---|---|
| **H1, H2, H3 (high and low rails)** | **0.011–0.016 fF** (z > 4 on all four amplifiers) | H-transfer edges 1, 3, 4 |
| **V1, V2, V3** | **< 0.002–0.003 fF** (3σ, not detected) | V-transfer edges. The 2.35 µs RC slows the V edges; the gates are far from the sense node |
| RG low rail (edge) | 0.15–0.22 fF | reset edges (agrees with §15/§19: 0.21–0.37 fF from the feedthrough feature) |
| **Reset charge injection, per volt of (RG_high − VR)** | **2.1–2.4 fF-equivalent**: equal and opposite coefficients for RG_H (+17 500 to +20 200 ADU/V) and VR (−16 000 to −18 700 ADU/V) | reset edges 0/6, plus their relaxation at 2/5 |
| SW low | < 0.008 fF at the H edges (SW does not switch there; its 0.08–0.13 fF is from the skip cycle) | – |

**New quantity: the buried-channel depth from the reset charge injection.**
- When the reset FET turns off, its channel charge Q = C_eff·W·L·(V_G − V_ch − V_T) is shared, roughly half onto the
  sense node. So the measured 2.1–2.4 fF per volt ≈ ½ C_eff · W·L.
- With the layout's 6 × 6 µm reset gate: **C_eff ≈ 0.12–0.13 fF/µm²**.
- In a buried channel, 1/C_eff = t_eq/ε_ox + x_ch/ε_Si. With t_eq = 76 nm (50 nm oxide + 50 nm nitride), that gives
  **x_ch ≈ 0.6 µm**: the depth of the buried-channel potential maximum.
- The recipe (B, 150 keV) gives a ~0.45 µm projected range, plus diffusion.
- Caveats: the ½ split and the 36 µm² gate are assumptions (±30 %). The depth scales accordingly.
- It is consistent across the four CCDs (2.10–2.43 fF). ch2 is the highest, as for every RG-related quantity.

**Updated sense-node mutual-capacitance table** (supersedes §19 where they overlap):

| line | C to sense node | source |
|---|---|---|
| RG | 0.15–0.37 fF | trace edges + feedthrough feature |
| SW | 0.08–0.13 fF | skip-cycle feedthrough |
| H1/H2/H3 | 0.011–0.016 fF | H-transfer edges (measured) |
| V1/V2/V3 | < 0.003 fF | V-transfer edges (upper limit) |
| OG | ~0.2–0.6 fF | layout estimate (not varied) |
| VDD/drain | ~2–4 fF | layout estimate (C_gd of the output FET) |
| VSUB | ~0.01–0.03 fF | depletion estimate |
| TG, DG | – | not varied in v3; tomography (NEXT_RUN_PLAN B8) |

## 21. Oxide (dielectric) thickness: what the data can and cannot separate

**The degeneracy.** Every electrical handle on the gates measures the gate-to-channel coupling of the buried channel:

    1 / C_eff = t_eq / ε_ox + x_ch / ε_Si

with t_eq the oxide-equivalent dielectric thickness and x_ch the depth of the channel maximum. This one combination
appears in the reset charge injection, in the channel potential vs dose (the knees: V_pinch ≈ q N_s / C_eff when the
potential maximum sits mid-layer), and in the output-FET gate capacitance.

- **Measured:** 1/C_eff ≈ **8.0 µm²/fF** (C_eff ≈ 0.125 fF/µm²; ±30 % from the charge-partition assumption). It is
  the same on the four CCDs within ±7 % (injection 2.10–2.43 fF/V).
- **Split with one prior:**

| assumed | gives |
|---|---|
| t_eq = 76 nm (recipe: 50 nm SiO₂ + 50 nm Si₃N₄) | x_ch ≈ 0.60 µm |
| x_ch = 0.45 µm (boron 150 keV projected range, no diffusion) | t_eq ≈ 126 nm |
| x_ch = 0.6 µm | t_eq ≈ 76 nm (the recipe value) |

- The recipe values (76 nm, ~0.5–0.6 µm after diffusion) are mutually consistent with the measured combination.
- **What does not work yet:** the vertical-clock steps in the traces. Their relaxation should contain the V line's
  RC (R_series × C_line, with C_line = ε_ox × gate area / t_eq). But the steps overlap and mix several couplings, and
  exponential fits are unstable (τ from 4 to 60 µs, many failures).

**How to separate them (cheap, non-destructive).** Each of these weighs the oxide alone, because the gate couples to a
surface layer (the channel stops / accumulated surface) over a large area, not to the buried channel:
1. **Clock-line capacitance at the connector** (LCR meter, warm and at 130 K, CCD powered down): C_line = ε_ox · A / t_eq
   for each V and H phase, with A from the layout (array area per phase). A V phase is ~1/3 of the array, giving
   ~0.1–0.3 µF at 76 nm, easily measured. The ratio between phases checks the area bookkeeping.
2. **Scope the clock edge at the cryostat feedthrough:** τ = R_series × C_line with R_series known from the schematics
   (V: 49.9 + 5.1 Ω; H/RG/SW: 100 + 15 Ω).
3. **The charge-partition check** (RG edge speed varied with an RC) pins the ½ split, which tightens C_eff to ~10 %.

With t_eq from 1 or 2 and C_eff from the reset injection, the buried-channel depth follows per CCD to ~10–15 %.

**Other layer quantities and where they stand:**

| quantity | status | how |
|---|---|---|
| silicon thickness | **615–655 µm** (measured) | muon tracks (§10) |
| full depletion, field-free layer | depleted at nominal; field-free ≲ 50 µm | diffusion profile of tracks |
| substrate doping / resistivity | **ρ ≳ 25–35 kΩ·cm** | thickness + depletion (§10) |
| buried-channel dose variation between CCDs | ~0.6 % (CCD_B vs D/C) | knees (§12) |
| lateral overlap under RG | 0.08–0.14 µm | coupling / (C_ox × width) (§15). Assumes t_eq = 76 nm: scales ∝ t_eq |
| field (thick) oxide under the sense line | only a bound | C_fixed (60–80 fF) over ~100 µm × 6 µm of line → the field dielectric must be ≳ 0.2–0.3 µm-equivalent if the line dominates C_fixed |
| poly, metal, passivation thickness | not accessible | they do not enter any electrical quantity we measure |

## 22. A single setup-noise number (`setup_noise_index.py`)

**Inputs.** One simultaneous trace set (~1 min, all channels of a board). Per channel, the per-skip CDS values are
averaged over blocks of n skips. The deterministic droop curve (the same in every pixel) is removed with
`--remove-pattern`. Each channel's block noise is split into a shared part (explained by the board's other channels)
and its own part.

**Recommended figures of merit, to compute and minimise per configuration:**
1. **σ_shared (e⁻)**: the shared (setup) noise per skip-block. On board 101 it is **constant with n**: it does not
   average down.
2. **SNI_n = σ_shared / σ_own** at the operating averaging n: the factor of setup noise over the amplifier's own noise.
3. **e_w (nV/√Hz)**: the white input noise density, against the 17.1 nV/√Hz open-frontend floor.

| run | board | e_w (nV/√Hz) | σ_shared (e⁻), n = 1 / 10 / 120 | σ_own (e⁻), n = 1 / 10 / 120 | SNI_1 / SNI_10 / SNI_120 |
|---|---|---|---|---|---|
| v1 (Nov 2–3) | 101 | 20.7 | 5.25 / 5.67 / 5.45 | 4.86 / 1.37 / 0.33 | 1.09 / 4.19 / 16.5 |
| v2 (Nov 5–7) | 101 | 21.0 | 5.18 / 5.62 / 5.39 | 4.91 / 1.40 / 0.34 | 1.08 / 4.12 / 15.9 |
| v3 (Nov 10–12, after the cabling change) | 101 | 20.5 | **4.31 / 4.80 / 4.62** | 4.74 / 1.33 / 0.33 | **0.92 / 3.62 / 14.1** |
| v1 / v2 / v3 | 102 | 16.8–17.0 (at the open floor) | – | – | 0.33–0.35 / 1.5–1.6 / 5.7–5.9 |

**Reading.**
- The shared component is a **slow wander of the CDS offset** (≈ 5 e⁻ equivalent). The CDS cancels baseline drift, so
  it must be the clock-feedthrough offset wandering, identically on every channel.
- That points to a slow wander of the shared clock/bias levels (it scales with RG_high − VR, §18).
- The cabling change reduced it by ~15 % (5.4 → 4.6 e⁻). Each amplifier's own noise is unchanged (≈ 4.8 e⁻ per skip,
  falling as 1/√n).
- Most of it averages out within a 500-skip pixel and the row baseline removes the rest. That is why common-mode
  subtraction does not help the 500-skip images, but does help at 1000–10000 skips.
- Board 102's white noise sits at the open-frontend Johnson floor (16.8–17.0 vs 17.1 nV/√Hz). Its setup index is ~3×
  lower than board 101's.

**For a new setup:**
- Take one simultaneous trace set per configuration.
- Minimise σ_shared (or SNI at the operating skip count), and check that e_w approaches the frontend floor.
- One number per board, ~1 min of data, no image needed.

## 23. CORRECTION: the traces of the four channels are NOT simultaneous; the valid October vs November comparison

**What was wrong.** Each channel's trace is its own acquisition, about 1 s apart (e.g. ch0 18:56:04, ch1 :05, ch2 :06,
ch3 :07; the other three streams of each file are empty frames, `process_packet.py`). Consequences:
- The cross-channel trace analyses of **§11, §13, §18 and the setup noise index of §22 do not measure simultaneous
  (setup) noise.**
- Their "shared" component (~50 % of the per-skip variance, ~5 e⁻) is a **pattern locked to the readout sequence**: the
  baseline/offset evolve identically in every acquisition.
- "White noise uncorrelated between channels" is trivially true for separate acquisitions. It says nothing about shared
  supplies.
- The v1/v2 → v3 change of that pattern is not the cabling change.

**Still valid** (the images are simultaneous: the four amplifiers are read into one file): crosstalk (§14), mutual
capacitances (§17), and the image common-mode tests (§11, `cms_images.py`).

**October vs November, the cabling change** (`oct_nov.py`). Board 101, 500-skip unbinned images at October's nominal
voltages (VSUB 45.1, VR −3.0, VDD −19.0, H1 high 9.5, V1 high 9.9, delay_H 1800): 12 October images vs 14–25 per
November period. Pixels where all four amplifiers are in the 0-e⁻ peak; row baseline removed. Medians:

| period | σ_0e ch0 / ch1 / ch2 / ch3 (e⁻) | simultaneous correlation between amplifiers | common-mode subtraction gain |
|---|---|---|---|
| **October** | **0.207 / 0.210 / 0.532 / 0.270** | **0.19–0.28** | **5–8 %** |
| Nov 2–3 (v1) | 0.154 / 0.179 / 0.517 / 0.371 | 0.03–0.06 | 0.1–0.7 % |
| Nov 5–7 (v2) | 0.144 / 0.147 / 0.349 / 0.247 | 0.02–0.08 | 0.1–0.7 % |
| Nov 10–12 (v3) | **0.145 / 0.148 / 0.332 / 0.291** | **0.02–0.07** | **≈ 0 %** |

Without the row baseline the October correlation is 0.20–0.30 (Nov v3: 0.07–0.12).

**What this shows.**
- **In October ~20–28 % of the pixel noise was common to the four amplifiers.** After the cabling change it is 2–7 %,
  and subtracting it gains nothing.
- **σ_0e fell ≈ 30 % on ch0 and ch1** (0.21 → 0.145–0.15 e⁻) at the same nominal settings.
- ch2 and ch3 also improved but scatter more: ch2 has its own RTS / high-floor problem, and ch3 varies.
- The October images had longer exposures (10–100 s vs 0–1 s). At ~40–75 e⁻/pix/day that is ≤ 0.1 e⁻/pix, and the
  0-e⁻ selection excludes it.
- The ADC/gain scale also moved by 6 % between October and November, on all amplifiers alike (image agent). That is
  consistent with a change in the readout chain at the same time.

**Trace white noise per channel** (valid: one acquisition per channel), at the video input:

| period | ch0 | ch1 | ch2 | ch3 (nV/√Hz) |
|---|---|---|---|---|
| Oct 8 | 20.2 | 20.8 | (broken traces) | 23.8 |
| Oct 12 | 20.7 | 21.2 | (broken traces) | 23.7 |
| Nov 2–3 / 5–7 / 10–12 | 19.9 / 19.9 / 19.8 | 20.4 / 20.4 / 20.2 | 21.2–21.7 | 22.1 / 22.3 / 21.9 |

- The white part improved by only 2–8 %. The cabling change mainly removed **simultaneous, low-frequency common-mode
  noise**, which is what the 500-skip pixels are sensitive to.

**A valid single setup-noise number.** From one image, or from traces once they are recorded simultaneously:
- **ρ_sim**: the mean simultaneous correlation of 0-e⁻ pixels between the amplifiers of a board, row baseline removed.
- or, equivalently, **the common-mode subtraction gain in %**.
- October: ρ_sim ≈ 0.25 / 5–8 %. November: ρ_sim ≈ 0.05 / ≈ 0 %.
- Target: ρ_sim → 0 with the white trace density at its floor.

**To make it a ~1 min trace measurement:** record the traces of all four channels in the same acquisition (a firmware /
script option: the streams of the other three channels are currently empty frames). Then `setup_noise_index.py`
measures simultaneous noise as intended.

## 24. A setup-noise metric from one few-second acquisition (L1 strips; `strip_noise.py`)

**Why strips, not traces.** Traces stream raw samples one channel at a time, and their slow noise is ~10× the images'.
That slow noise did not change with the cabling (`trace_spectrum.py`: σ at 500 skips 2.2–2.3 → 2.2 → 1.9 e⁻,
October → v1 → v3). L1 strips (2 × 1000 pixels × 500 skips, ~30 s) use the image's clocking, record **all four
channels simultaneously** and keep every skip's value. They are already taken in every packet.

**What the strips show** (board 101; per-skip residuals with each pixel's mean removed; the shared part = the mean of
the channels):

| | October 6 (L1, 3000 skips) | November 10–12 (v3 strips) |
|---|---|---|
| shared / own noise power, 0–200 Hz | **23** | **3.9** |
| shared / own, 200–1000 Hz | 1.3 | 0.7 |
| shared / own, 1–5 kHz | 1.6 | 1.5 |
| correlation between channels, per skip | 0.29–0.30 | 0.36–0.38 |
| at 250-skip averages: correlation; slow noise above white | 0.66; 2.9–3.4× | (pixels too short to measure) |
| shared narrow lines | 2532 Hz, 5064 Hz | 2240–2490 Hz, 5050–5240 Hz |

**Readings.**
- **The cabling change cut the shared slow (< 200 Hz) noise ~6×.** The images showed the same change: correlation
  0.25 → 0.05, σ_0e −30 %.
- **A ~2.5 kHz interference line and its harmonic reach every channel**, in both months. Its source is to be found
  (a switching converter or a periodic digital process). The 500-skip pixel (16.3 ms) averages it out; other skip counts
  may not.
- 60 Hz hum is not the issue (no 120/180 Hz harmonics).

**Recommended setup metric (no image):** one L1 strip per configuration (~30 s):
1. the **shared/own power ratio below 200 Hz** (target → 0);
2. the list of **shared lines**;
3. the per-channel white noise per skip.

**Caveats.**
- The October point is a single acquisition (100 × 1000 × 3000, ch2 broken).
- The November strips' 480-skip pixels give only ~64 Hz resolution, so the 0–200 Hz band is coarse.

## 25. Reset (kTC) noise in the per-skip pedestal level: a direct route to C_sn (preliminary)

**The idea.** The sense node is reset every skip, and each reset freezes kTC charge, √(kT·C_sn)/q electrons. That
charge cancels in the CDS but appears in the per-skip level, identically in the pedestal and signal windows.

**What is measured.** The per-skip level common to both windows is 161–168 ADU on the board-101 amplifiers. But it is
145–148 ADU even on the dead board-102 channels, so most of it is an electronics level jump per skip (reset-edge pickup
charging the frontend's coupling capacitor).

**Live minus dead.** √(166² − 147²) ≈ 77 ADU ≈ 56 e⁻ → **C_sn = (q σ_e)²/(kT) ≈ 45 fF** at 130 K, and then
A = CG·C_sn/q ≈ 0.44.

**What it implies.** This disagrees with the layout inversion (80–90 fF, A ≈ 0.85). It agrees with the low-g_m branch
of §6, in which the load resistor explains most of the white noise.

**Status.**
- The subtraction assumes board 101's electronics jitter equals board 102's.
- 102 ch3 adds almost nothing above its dead neighbours (149 vs 145–148), consistent with its reset not producing charge.
- **Decisive:** the DC level of the video line.

## 26. Corrections (2026-11-16, after review)

- **Single-trace cabling comparison: compare equal lengths.** 9 of the 12 October traces in `trace_anatomy.csv` are
  1.1 s long (Nov: 0.19 s) and sit ~40 % higher below 1 kHz than the three 0.19 s October traces. Like-for-like
  (0.2 s October, n = 3 → Nov 10–12): spectrum 200 Hz–1 kHz 48.8 → 34.8 µV/√Hz (−29 %), 1–10 kHz 5.2 → 3.7 (−28 %),
  10–100 kHz −15 %; skip-to-skip correlation 0.71 → 0.68 (not a useful metric at this level). The earlier 37–49 % and
  0.87 → 0.68 are withdrawn. The images (ρ_sim 0.25 → 0.05, σ_0e −30 %) remain the solid result.
- **Own white (9–13 nV/√Hz, window fit) vs dead-channel frontend (17.0): inconsistent, not an agreement.** The frontend
  noise is per-channel, so it belongs to "own"; own white below 17 requires the live follower to shunt it (high g_m),
  which §25 argues against, or the fit to count band-limited white noise (≈0.9 µs settling) as 1/f. Until the window
  fit is run on the dead board-102 channels of the v2 strips, the 1/f share and N_t are upper limits. The window
  optimum uses only the fitted σ₁(τ) and is unaffected.

## 27. The 1/f level cross-checked without the CDS; an amplifier-health number in electrons (2026-11-16)

**Single-trace window fit.** `trap_from_trace.py` forms 2–9.3 µs windows in software inside each 10 µs window of one
raw trace and fits σ₁²(τ) = S_w I_w + K I_f with the exact filter. On board 101 (8 v3 traces per channel) it reproduces the
strip result (§24) within 15 % on ch1/ch3 and ×1.6 on ch0/ch2. Same model, so a second implementation, not a test.

**Independent test: LSM static records** (DM07/09/10, ACM 106/109/110, 180 K; all clocks frozen, 0.1 s).
`static_fit.py` fits S(f) = (S_w + K/f)·L(f) + S_0 directly (2 kHz–4 MHz, L = 1/(1+(f/f_c)^2n)); f_c = 470–690 kHz,
n ≈ 1.4–2.2, S_0 ≈ 4.5–5 nV/√Hz.

| quantity | window method / static spectrum (9 clean channels) |
|---|---|
| 1/f level K | **1.01 ± 0.13** |
| white level | 1.28 (1.39 with the roll-off added to the window fit) |
| 1/f corner | not consistent (static 84–181 kHz, windows 48–99 kHz) |

- K is robust; N_t ∝ K, so the trap density (still an upper limit) is now backed by two methods.
- The white/corner split is not robust; adding the measured roll-off makes it worse, so band-limited white counted as
  1/f (§26) is not the main effect. The corner is dropped as a health number.
- Static records miss episodes: ACM109 ch1 (telegraph) and ACM110 ch3 (30.6 kHz comb) look normal when static.
- 1/f at 10 kHz is 21–28 nV/√Hz on all 12 LSM channels: transistor design, or 1/f on a shared rail (VDD/VR).
  Test: simultaneous static records on several channels.

**Health number.** σ_1/f(500) = √(K·I_f(500))/gain, the transistor's 1/f noise left in a 500-skip pixel (through the
CDS, 1/f still averages ×0.040 vs ×0.045 for white from 1 to 500 skips):

| | 101 ch0 | 101 ch1 | 101 ch2 | 101 ch3 | LSM (12) |
|---|---|---|---|---|---|
| σ_1/f(500) (e) | 0.084 | 0.111 | 0.114 | 0.128 | 0.11–0.16 |
| image σ, 500 skips (e) | 0.144 | 0.147 | 0.164 | 0.173 | 0.19–0.33 |
| share of image variance | 34 % | 57 % | 49 % | 55 % | 14–64 % |

About half of board 101's image variance is the transistors' own 1/f noise (upper limit). Used in
`physics_sheet.py` and the scorecard (good ≤ 0.10 e, fair ≤ 0.15 e). The board-102 channels meant as an open-input
check have transistors (1/f corner 100–130 kHz, N_t ≈ 2×10¹⁶), so a true open-input test is still missing.
Figure: `results/lsm_hightemp/static_fit/static_fit.png`.

**Validity filter (added).** Operating points where the readout breaks (merged peaks, most columns hot;
`label.ok = False`, 27 of 127 board-101 packets) are left out of the ML scorecards, the correlation matrices,
`physics_model.py` and `datasheet.py`. Today's trace analyses (per-channel anatomy, `trap_from_trace.py`,
`skip_averaging.py`) had not filtered them; rerun on the 93 packets valid on all four channels: per-skip noise and
slow-noise medians move < 3 %, ch2 N_t 1.78 → 1.87, ch2 transistor 1/f at 500 skips 0.114 → 0.117 e; ch2's
highest-kurtosis traces (N3_094, N3_116, N3_026) are all valid packets, so its telegraph episodes stand.

## 28. Buried-channel depth from one trace (`depth_from_trace.py`, 2026-11-16)

The depth (§ depth) uses the slope of the reset charge against RG_high − VR, s = ½ C_eff W L. One trace gives
Q = s (RG_high − VR − V*), so it needs V*. The scan's intercepts (valid packets) are V* = 9.38 / 9.57 / 9.44 / 9.50 V on
ch0–3: a design constant (spread 0.2 V). Nominal sits only 2.4–2.6 V above it, so V* must be known to ~0.1 V.

| | scan slope → depth | 1 nominal trace, V* from the other 3 amps | 1 nominal trace, V* = 9.47 ± 0.08 | single traces (93), median slope |
|---|---|---|---|---|
| ch0 | 2.23 fF/V → 0.61 µm | 0.63 µm | 0.60 (0.57–0.63) µm | 2.38 |
| ch1 | 2.32 → 0.58 | 0.69 | 0.63 (0.60–0.66) | 2.22 |
| ch2 | 2.53 → 0.51 | 0.56 | 0.54 (0.52–0.57) | 2.61 |
| ch3 | 2.20 → 0.62 | 0.70 | 0.65 (0.62–0.68) | 2.19 |

One trace reproduces the scan depth to ≤ 0.1 µm (≤ 0.05 µm with the shared V*), ch2 thinnest either way. The
injection-split assumption (30–70 %) still dominates the absolute value (≈ 0.25–1.0 µm), as for the scan. The overlap
(feedthrough ≈ 1 fC on top of 5–6 fC injection at the same edge) cannot be separated from one trace: it needs one
extra trace with RG_low changed. Not yet applicable to the LSM traces: their reset event has a different shape and
position (sequencer `refrence_v4_DM10.seq`, RG high 8 µs); the edge must be mapped from that sequence first.

**Valid-operating-point rerun of the scan regressions** (`clock_edges.py`, `sense_node_coupling.py`, 2026-11-16).
Reset injection slope (edge 0, RG_1_H) 2.22/2.13/2.43/2.10 → **2.27/2.39/2.60/2.27 fF/V** (depth 0.59/0.55/0.49/0.59 µm;
mean 0.55 µm); RG_1_L feedthrough at that edge 0.15–0.22 → 0.12–0.18 fF; H couplings 0.012–0.023 fF (unchanged range).
New with the broken points removed: the reset edge also depends on H2 high/low (0.20–0.23 / 0.11–0.12 fF): it sits on
the H2 waveform's tail. The settled mutual-capacitance matrix is unchanged except ch1 RG_1_H (3.2σ → 1.1σ).

**Mapping the reset edge on another sequencer (LSM).** `reset_pulse.py` finds the reset pulses from the waveform alone
(no flags): PA03 −280/+1626, LSM −268/+1626 samples from the H flag. Their heights are not comparable: PA03 pulses are
11–14 samples (the sense node still charging toward VR, τ ≈ 0.4–0.7 µs, when they end; the two pulses agree, 47k/50k
ADU), LSM pulses 2.7–5.5 samples (a feedthrough + charging fit is degenerate; the two pulses of one ACM106 trace differ,
32k/42k). The V* calibration of §28 therefore holds per sequencer. Sequencer-independent route: two traces on the same
setup differing only in RG_high (the dynamics cancel in the difference) give that setup's slope and V*; a third with
RG_low changed gives the overlap. Requested for the LSM boards; no LSM depths are reported until then.

## 29. The setup in electrons at N skips (`setup_noise_N.py`, 2026-11-16)

Per-skip σ_CDS is blind to slow pickup (CDS ∝ f² below ~1 kHz; Oct→Nov σ_CDS −3 % while images −30 %). A noise spectrum
integrated through the exact CDS + N-skip filter gives the setup's share of an N-skip pixel, in electrons.

- **LSM static records** (validation: predicted per-skip σ / trace σ = 0.97 median): setup band 100 Hz–1 kHz at 500
  skips = ACM110 0.012–0.024 e (0.7–1.9 % of the variance); ACM106 ch1/ch3, ACM109 ch2/ch3 0.04–0.07 e (11–22 %);
  ACM109 ch0/ch1, ACM106 ch2 0.09–0.12 e (36–41 %); ACM106 ch0 0.14 e (67 %). Images are 1.4× the predicted total
  (180 K dark charge).
- **Clocked traces cannot give it.** The waveform-removed residual is dominated by reset-pulse jitter and cycle joins
  (σ_1 ≈ 1400 e on PA03: unusable). The trace CDS series itself (time domain, `skip_averaging.py`) gives σ_128 ≈ 3.5 e on
  board 101 (99 % excess) and 1.6 e on board 102, ~12× what board 101's images show (0.144 e at 500 skips). On the LSM
  the clocked-trace 0.2–1 kHz level is 2–50× the static one. Clocked-trace setup numbers (§ cabling, §26) hold for
  like-for-like comparisons only; PA03 needs static records for the electron number.
