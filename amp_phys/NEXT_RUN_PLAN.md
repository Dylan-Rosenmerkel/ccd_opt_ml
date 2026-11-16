# amp_phys: the next run (and the grounding study)

Written 2026-11-16. Each item names what it measures, why it was not measurable before, what to take, and how it
will be analysed (scripts in `amp_phys/`).

## A. The setup's common-mode noise: find it and reduce it by changing the grounding

**What we know (v3n, `common_mode.py`):**
- On board 101, ~50 % of each channel's per-skip noise variance is common to its four channels (correlation 0.52–0.56).
- Board 101 and board 102 are **anti-correlated (−0.3)** although two ACMs digitise them. A shared source with
  opposite sign on two boards is the signature of a **ground-loop current** between the two boards' references.
- The shared part is at low frequency (coherence 0.3–0.5 below ~0.05 cycles/skip ≈ 1.5 kHz, ~0.02 above).
- Subtracting it across the simultaneous amplifiers (`cms_images.py`) does not help at 500 skips in the v3 images. It
  lowers the noise by **8–30 % at 1000–10000 skips** in the October skip-noise images, where the slow floor dominates.
- The October 500-skip images also gain 6–14 % while the November ones gain nothing, so the environment changed
  between the runs.

**Where it comes from (v3n, with all 18 voltages varied):** the common share on board 101 tracks the **reset-gate high
rail** (Spearman +0.54; +17 percentage points per volt with the other rails held fixed), **VR** (−16 pp/V) and the
**H1 high rail** (+0.42). It tracks VSUB only weakly (+0.22). The cross-board coupling tracks the same rails.
Leading hypothesis: noise on the DC-held clock rails couples through the gate overlaps onto every sense node.
- During the 500 skips, RG and the H clocks sit at their high levels.
- The amplifiers share these DAC rails.
- The frontend filters RG/H/SW only with 100 Ω × 330 pF (33 ns), which passes kHz noise unfiltered.
- That makes the noise common to the channels, and stronger with a larger RG high level.

**First hardware tests:**
- A heavier RC on the RG and H rails, compatible with their timing: the held level only needs to be quiet below
  ~1.5 kHz.
- Measure the AD5371 DAC + THS6072 buffer output noise on RG_H directly.
- Drive RG_H from a quiet external reference for one trace set.

**The metric for a grounding study.** For each configuration, take one simultaneous trace set on all 8 channels
(both boards, ~1 min, nominal settings, the dead board-102 channels included). Then run `common_mode.py`, which reports:
1. the common share per board (target: → 0);
2. the cross-board correlation and its sign (ground loop: → 0);
3. the coherence spectrum, i.e. which frequencies are shared (mains and harmonics, switching regulators: the VSUB boost
   converter LT3580 switches at a fixed rate, and its harmonics would appear as lines);
4. the dead-channel noise. That is pure Johnson noise + preamp if no injection reaches the frontend, so its excess over
   17.1 nV/√Hz is the injection at the frontend itself.

**Configurations to step through, one change at a time, two trace sets each:**

| # | Change | Tests |
|---|---|---|
| 0 | as is (repeat ×3 spread over the session) | baseline + its stability |
| 1 | the two ACMs on one common ground point (star) / separate | the cross-board ground loop |
| 2 | cryostat body to ACM ground strap: on / off / moved | the cryostat as the loop |
| 3 | frontend-board shield (JP200 / R200: shield to GND) open / closed | cable-shield return path |
| 4 | VSUB at 40, 50, 60, 70 V with everything else fixed | the injection grows with VSUB (it does: 0.12 %/V on all amps). An LDO's output noise scales with its output voltage, so a VSUB-supply origin predicts injected variance ∝ VSUB² |
| 5 | VSUB supply from an external battery / linear lab supply | removes the on-board boost converter: if the common mode drops, it is VSUB |
| 6 | VDD and VR from an external supply (one at a time) | same test for the bias generators (TPS7A3301) |
| 7 | the other board's ACM off (board 102 unpowered) | whether board 102 drives the loop |
| 8 | Ethernet / VME cables: disconnect what is not needed during the trace | digital return currents |

**Success criterion:** the common share on board 101 drops from ~50 % to ≲ 10 % and the cross-board correlation to ~0.
Then the per-skip white noise should fall from 19–22 toward the ~7–8 nV/√Hz that the amplifier + load network alone
give (`chain_calibration.py`), and the high-skip floor (0.1 e⁻) should drop.

## B. Ideas that need new data

| # | Measurement | What it gives (not measurable before) | What to take | Analysis |
|---|---|---|---|---|
| B1 | **Noise thermometry.** Dead-channel noise logged continuously | The frontend board temperature from the Johnson noise of its 16.7 kΩ load, and electronics drift vs CCD drift. Only after the common mode is removed (A): today the injection's ±1.6 % variance swamps it | Nothing extra: the board-102 dead channels in every trace. A deliberately unconnected channel on board 101 would add the same for board 101 | `common_mode.py` (thermometry panel) after common-mode removal |
| B2 | **VR fine scan** (0.1 V steps over −4.0 to −2.2 V, 20 points, nominal else) | Separates the gate capacitance from the fixed (wiring + diffusion) capacitance per amplifier: the sense node sees C_fixed + C_gs(1 − A), and A moves with VR. Also the reset-headroom knee (now −2.5 to −3.8 V) to ±0.05 V | 20 packets | extend `physics_model.py` gain equation with the bootstrapped C_sn |
| B3 | **VSUB dither.** A small square wave (±1 V, ~100 Hz) on VSUB during a trace | The substrate → sense-node coupling C_sub/C_sn per amplifier, by lock-in detection. It tests directly whether the injected noise enters through the substrate | 1 trace set per amplifier with the dither (firmware: modulate the VSUB pot) | lock-in on the trace CDS at the dither frequency |
| B4 | **Rail stepping** (one clock rail at a time, 10 points across its window, others nominal: H highs, H lows, V highs, V lows, SW, OG, RG) | Channel potentials per CCD, the non-destructive stand-in for SIMS/C–V. The knees found in v3 (`channel_knees.py`) are blurred by the Sobol design (every rail moves at once); single-rail steps give the knee to ±0.05 V. A 0.9 V knee shift ≈ 0.7 % implant dose or equivalent oxide charge | ~70 packets (7 rails × 10) | `channel_knees.py` |
| B5 | **Traces with known charge.** Flush the register, then trace 10 pixels after a short exposure (1–5 s) | Single electrons resolved inside a 0.2 s trace → absolute gain and charge occupancy without an image (it failed on the current traces: their pixels carry varying charge or level) | 1 trace set per packet | `trace_electrons.py` |
| B6 | **Pocket pumping at 6 phase times** (10 µs – 10 ms), board 102 configuration on board 101 | Trap emission time constants → trap energies per CCD (non-destructive DLTS) | 6 PP images per CCD | `amp_phys/images/pp.py` |
| B7 | **⁵⁵Fe or a known-energy source** (if available) | Absolute energy scale → pixel pitch and thickness to ±1 % (now ±7–10 % from muons) | 1 h exposure | `amp_phys/images/tracks.py` + peak fit |

## C. Open questions for the user
- Board 102: in the October images only ch2 (EXT3) collects charge, in v3 the trace of ch3 is the live one. Is the
  channel mapping different between traces and FITS, or is ch3's amplifier alive with no charge reaching it?
- Which traces are theta1–3 / ref / post (settings, flush before)?
- Was anything about grounding or cabling changed between 2026-10-14 and 2026-11-10?
