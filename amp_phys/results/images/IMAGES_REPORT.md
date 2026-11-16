# amp_phys phase 3: what the images say about each amplifier (2026-11-16)

This covers the skipper images in `/pool/storage/DAQ1_Data/run/PA_PQ_testing`, which were only read, never changed.
The code is in `amp_phys/images/`: `inventory.py`, `units.py`, `tracks.py`, `dark.py`, `defects.py` and `pp.py`, plus
the shared helpers in `common.py`. Each script has a docstring and a CLI. Every number below comes from the CSV files
in this folder.

## 0. Results per amplifier

| | 101 ch0 (CCD_D) | 101 ch1 (CCD_C) | 101 ch2 (CCD_A) | 101 ch3 (CCD_B) | 102 ch2 (EXT3) |
|---|---|---|---|---|---|
| gain (ADU/e⁻), single-e⁻ peaks, Oct 14–15 | 1.324 ± 0.003 | 1.304 ± 0.003 | 1.248 ± 0.012 | 1.256 ± 0.002 | 1.238 ± 0.003 |
| noise per single sample σ₁ (e⁻) | 4.2 ± 0.5 | 4.5 ± 0.5 | **8.1 ± 0.5** | 5.4 ± 0.4 | 4.0 ± 0.4 |
| noise floor at high N (e⁻) | 0.095 ± 0.016 | 0.097 ± 0.016 | **0.20 ± 0.02** | 0.103 ± 0.012 | 0.19 ± 0.02 |
| σ at N = 5000 / 10000 skips (e⁻) | 0.131 / 0.082 | 0.126 / 0.084 | 0.210 / 0.213 | 0.114 / 0.107 | 0.181 / 0.245 |
| active thickness from muons (µm, 15 µm pitch) | 620 ± 50 | 655 ± 60 | 615 ± 65 | 625 ± 50 | 635 ± 60 |
| pixel pitch from MIP dE/dx (µm) | 15.2 ± 1.5 | 15.1 ± 1.5 | 15.3 ± 1.5 | 15.1 ± 1.5 | 15.3 ± 1.5 |
| diffusion σ, front / back (px) | 0.26 / 1.40 | 0.25 / 1.39 | 0.25 / 1.38 | 0.26 / 1.39 | 0.31 / 1.37 |
| array dark D_arr (e⁻/pix/day), v3 nominal | 36 ± 2 | 41 ± 2 | 75 ± 2 | 67 ± 3 | n/a |
| serial-register dark D_ser (e⁻/pix/s), v3 nominal | **0.0098 ± 0.0006** | 0.0187 ± 0.0011 | 0.0256 ± 0.0005 | 0.0225 ± 0.0010 | n/a |
| **D_ser vs VSUB (%/V)** | **+2.0 ± 0.6** | +7.5 ± 0.5 | **+0.8 ± 0.2** | +5.8 ± 0.6 | n/a |
| D_arr vs VSUB (%/V) | +1.5 ± 0.5 | +1.4 ± 0.3 | +0.6 ± 0.2 | +2.7 ± 0.6 | n/a |
| overscan λ vs VSUB (%/V) | **+3.9 ± 0.6** | +8.9 ± 0.4 | **+1.0 ± 0.2** | +6.6 ± 0.6 | n/a |
| vertical deferred charge, (trail − lead)/edge (10⁻³) | 0.24 ± 0.09 | 0.50 ± 0.09 | **1.11 ± 0.17** | **1.06 ± 0.13** | 0.35 ± 0.15 |
| serial deferred charge (10⁻³) | −0.3 ± 1.3 | 0.8 ± 1.5 | 1.1 ± 0.6 | −0.1 ± 1.5 | 4.5 ± 3.2 |
| glow (peak column, e⁻/pix in the 1-h dark image) | col 3691, 19 | col 1913, 84 | **col 136 (amplifier end), 22** | col 1910, 37 + hot spot | noisy channel |
| persistent hot pixels (>25 e⁻ in ≥8/10) | 1 | 0 | 0 | 0 outside the hot-spot dome | 220 |
| pocket-pump trap dipoles | hidden by an artefact (§5) | hidden | hidden | hidden | no excess over the unpumped image |

Errors are statistical (fit or bootstrap) unless marked otherwise. For the thickness, the error includes a systematic
from the choice of fit window and from how the MIP deposit depends on path length; it is mostly common to all CCDs.

## 1. What the data are (inventory: `inventory.csv`, 697 avg_img files)

- **Each board-101 channel reads its own CCD, not a quadrant.** The header maps ch0 = CCD_D, ch1 = CCD_C,
  ch2 = CCD_A and ch3 = CCD_B. Each channel has 8 prescan columns and then **6144 active columns**: the step from active
  area to overscan sits at serial index 6152 in every channel (index 852 in the images with NCOLPRE = 5300). Tracks and
  defects differ from channel to channel. So module PA03 has four 6144 × 1536 CCDs, each read through one amplifier.
  The "quadrant" comparison in the test plan is really a **CCD-to-CCD** comparison: every image quantity mixes the
  amplifier with its own CCD. Board 102 reads CCD PQ04 with `Vtransfer21`; board 101 uses `Vtransfer22`.
- **Board 102: the images contradict the phase-1 label.** In the October images, only **102 ch2 (EXT3)** sees charge.
  It has tracks, single-electron peaks (gain 1.238 ADU/e⁻), an active-to-overscan step and glow. **102 ch3 sees none**:
  in a 400 × 6144 image with a 1-hour exposure, 2 pixels pass 6σ, against about 10⁵ track pixels on the other CCDs. All
  it shows is crosstalk from ch2 of about 2–5 × 10⁻⁴. The session notes of 2026-10-12 and 2026-10-16 say the same: the
  CCD channel of board 102 is EXT3. On 2026-10-01 no board-102 channel saw any charge. In this report, 102 ch2 is the
  board-102 CCD channel, and ch0, ch1 and ch3 are the no-charge controls. The phase-1 conclusion that "102 ch3 is the
  working amplifier", which came from settling, needs checking against these images: either the EXT-to-channel mapping
  differs between traces and FITS files, or the amplifier settles normally but no charge reaches it.
- **What each run varied.**
  - Skip-noise run: N = 1, 10, 100, 500, 1000, 2000, 3000, 5000, 10000, with all other settings fixed.
  - PP_Testing: PP_400×6300×5 with N_pump = 15, 50, 150, 500, 5000, 50000, plus an unpumped reference with the same
    geometry and a 3708 s exposure.
  - PocketPump1: ten PP_1300×6300×1 images (N = 5000, 100 s exposure) and four PP_1300×6300×5 images (N = 50000, 3600 s).
  - Every pocket-pump image has the same phase time, delay_Va/b = 6000 clocks = 60 µs, and PP_SHIFT = 3.
  - The 32×1000×500 packet runs vary all voltages together (VSUB 35.7–70 V).
- **Timing.** From the time stamps, a skip takes 33.1 µs and a pixel has about 0.3 ms of overhead. The `.meta`/FITS
  headers record no temperature (the run names suggest about 130 K).
- **ADC scale anomaly.** In the 10000-skip images, the peak spacing is 3.33 times the gain of the other images on every
  channel (4.41 against 1.324 ADU/e⁻ on ch0). This is consistent with a sum over 10000 samples divided by 3000. The
  noise in e⁻ is unaffected, but any ADU-level use of NDCM > ~8000 images must rescale.

## 2. Units and noise (`units.py` → `units_summary.csv`, `units_fits.csv`, `noise_vs_skips.png`, `peaks_*.png`)

- **Gain.** It comes from a comb-of-Gaussians fit to the single-electron peaks of the 500–5000-skip images. The
  per-image values agree to 0.2–0.5 %, so the gain does not depend on N (except at N = 10000, see above). The October
  gains are **6.2 % below the phase-1 (v3, November) gains on all four board-101 amplifiers alike** (ratios
  0.939/0.938/0.938/0.937). That is a common scale change between epochs or methods, not an amplifier effect. The
  ordering ch0 > ch1 > ch3 ≈ ch2 is the same in both.
- **Noise against number of skips.** For N ≤ 100 the noise comes from the overscan; for N ≥ 1000 it is the width of the
  0-e⁻ peak. The model is σ(N) = √(σ₁²/N + floor²).
  - The floor equals the white term at N ≈ 2000 (1600–2700) on board 101.
  - **101 ch2 has twice the single-sample noise (8.1 e⁻) and twice the floor (0.20 e⁻) of the others.** In October it
    did not reach 0.1 e⁻ at any N. This fits the RTS/1/f excess found in phase 1.
  - 102 ch2 has a white noise as low as the best board-101 amplifiers (4.0 e⁻) but a 0.19 e⁻ floor, and its image has
    strong column-correlated baseline structure.
- **The readout chain alone averages down cleanly.** In ADU, the overscan noise of the no-charge channels (102 ch0,
  ch1, ch3) falls as 1/√N up to 1.1–1.7 times white at N = 5000. On board 101 the overscan noise grows again above
  N ≈ 1000 (4–18 times white at 5000), because the baseline drifts within a row. The excess floor therefore comes from
  the CCD and amplifier (or board 101's low-frequency drift), not from the ADC chain.

## 3. Cosmic tracks (`tracks.py` → `tracks_summary.csv`, `tracks_Q_vs_L.png`, `tracks_diffusion_thickness.png`, `tracks_angles_cti.png`)

- **Clusters.** 4σ threshold, 8-connected, charge integrated over the cluster grown by 2 px. For each cluster: a
  charge-weighted PCA, the length L = √(12·(var∥ − σ⊥² − 1/12)), and the transverse width in 4-px segments.
- **Image sample.** Only images where tracks are not pumped: the unpumped 400×6300×5 image, the 400×1500×500 image, and
  the PP_1300×6300×1 images, whose exposure (100 s) is much shorter than their readout (45 min). In the 3600 s pumped
  images, tracks that arrived before pumping come out broken (see §5), so those images are excluded.
- **Thickness.** A track that is contained in the image crosses the whole depth T, so Q = q₀·√(L² + T²).
  - A median fit of the muon band, about 1000–1200 straight tracks per CCD, gives **T = 41–43.5 px**: 615–655 µm at a
    15 µm pitch.
  - Changing the fit window (L ≥ 12/20/30 px) and the path-length dependence of the median MIP deposit (0 or +4.5 % per
    e-fold) moves T by ±3–4 px. That systematic is common to all CCDs.
  - **All five CCDs agree within errors and with the DAMIC-M nominal 670 µm, to within 1σ (~−5 %).** CCD_C (101 ch1)
    is 2 ± 1.5 px thicker than the others, which is not significant.
  - The fit is a ratio, so T in pixels does not depend on the gain.
- **Pixel pitch.** The slope at large L is q₀ = 1230–1243 e⁻ per pixel of path, the same on all CCDs within 1 %. That
  also confirms the relative gains. Divided by the expected median MIP deposit for 0.6–2 mm silicon paths
  (≈0.305 keV/µm at 3.75 eV per pair, ±10 %), it gives **15.1–15.3 ± 1.5 µm**, consistent with 15 µm. A better check
  than ±10 % is not possible without an absolute energy reference.
- **Diffusion.** The stacked width profile of the muon tracks (orientation fixed with even segments, width measured on
  odd ones) rises smoothly from **σ ≈ 0.25 px (4 µm) at the narrow end to 1.39 px (21 µm) at the wide end**. It is
  identical on all four board-101 CCDs to ±0.02 px; 102 ch2 is 0.31 → 1.37 px.
  - There is no sudden widening or charge loss at the back, and T is close to nominal. So the CCDs are **fully depleted
    within the precision**: any field-free layer is ≲ 50 µm thick, limited by the thickness error.
  - For full depletion with a uniform field, σ_back ≈ √(2kT/q·T²/V). That gives 14.5 µm at 130 K and 45 V.
    The measured 21 µm points to a weaker field near the back (normal close to the full-depletion voltage) or a higher
    temperature. The temperature is not recorded, so this cannot be pinned down further.
- **Mounting.** The track azimuths are not isotropic: 74 % lie closer to the column direction than to the row
  direction. The module is probably mounted with its columns vertical or tilted. The azimuth therefore cannot be used to
  check the pixel aspect ratio.
- **Charge-transfer proxies.** The proxy is the charge just after the trailing edge minus the charge just before the
  leading edge of each cluster row or column, divided by the edge-pixel charge.
  - **Serial: no significant deferred charge** on any CCD (|x| < 1.5 × 10⁻³, with no slope over 6150 transfers).
    Serial CTI is below about 10⁻⁶ per transfer.
  - **Vertical: a small, row-independent deferred fraction**, 0.24/0.50/1.11/1.06 × 10⁻³ on ch0/ch1/ch2/ch3. It does not
    grow with the number of transfers (CTI_v < 10⁻⁶ per transfer). It is therefore a one-off loss, most likely at the
    last vertical-to-serial transfer (TG or the summing well), and it is twice as large on CCD_A and CCD_B as on CCD_D
    and CCD_C.

## 4. Dark charge and defects (`dark.py` → `dark_summary.csv`, `dark_vsub.csv`, `dark_vs_vsub.png`; `defects.py` → `defects_summary.csv`, `defects_hot_columns.csv`, `defects_profiles.png`, `defects_hot_pixels.png`)

**Splitting the dark charge into array and serial register.** In electron-counting images, the mean charge of an
unmasked pixel at read row r and read column j is fitted as λ = a + D_arr·r·t_row + D_ser·j·t_pix. Rows read later wait
longer in the array. Columns read later wait longer in the serial register.

- At nominal settings (v3, 33 images), the serial register generates charge **20–50 times faster per pixel-second than
  the array**:
  - D_ser = 0.010/0.019/0.026/0.022 e⁻/pix/s;
  - D_arr = 4.2/4.8/8.7/7.8 × 10⁻⁴ e⁻/pix/s, i.e. 36/41/75/67 e⁻/pix/day.
- The long unpumped image gives the same order for D_arr: 25 ± 4, 28 ± 4, 25 ± 3 and 57 ± 3 e⁻/pix/day from its row
  slope.
- The skip-noise images give lower values. They are a different epoch with a much longer readout.

**VSUB response, with the other voltages regressed out.** v3 has 127 images, and all voltages vary together, so this is
an OLS fit on 18 non-collinear voltages.
- **The split between the weak ch0/ch2 pair and the strong ch1/ch3 pair is in the serial register, not the array:**
  - D_ser rises by +2.0 ± 0.6 and +0.8 ± 0.2 %/V on ch0 and ch2, against +7.5 ± 0.5 and +5.8 ± 0.6 %/V on ch1 and ch3.
  - In absolute terms: 2.0 and 2.1 × 10⁻⁴ against 14.0 and 13.0 × 10⁻⁴ e⁻/pix/s/V.
  - The overscan λ follows the same pattern: +3.9 and +1.0 against +8.9 and +6.6 %/V.
- The array dark D_arr responds about equally on all four (+0.6 to +2.7 %/V, no pairing).
- The New_Packet_Tests run (fewer images, 9 regressors) repeats the pattern with large errors: D_ser +6.7 ± 3.1,
  +8.9 ± 2.7, +0.9 ± 2.7 and +9.2 ± 3.2 %/V. Its ch0 is ambiguous; its ch2 is clearly weak.

**Glow and hot spots.** Column profiles of the 1-hour unpumped image (`defects_profiles.png`):
- **CCD_B (101 ch3) has a strong localized emitter**. It is a saturating dome starting at read row ≈ 800 and widening
  toward the far rows, centred at column ≈ 1950, with a bright column streak. This makes 26 % of its columns unusable in
  the 1300-row images.
- **CCD_C (101 ch1) has a broad glow centred on the same column (1913)**: 84 e⁻/pix peak, FWHM 680 columns, growing with
  readout time. It is most likely light from the CCD_B hot spot reaching the neighbouring CCD.
- **CCD_A (101 ch2) has a glow at the amplifier end of the register** (peak at column 136, about 22 e⁻/pix, decaying
  over about 1300 columns). It is constant in rows, so it builds up during the exposure, not during the readout. It is
  absent in a 1 s exposure. This fits light from the CCD_A output stage while it is idle-biased; it is the only glow
  tied to an amplifier.
- CCD_D (101 ch0) has a narrow glow at column 3691 (19 e⁻/pix, FWHM 270), away from the amplifier.

**Hot columns and hot pixels.**
- Hot columns (more than 5σ and more than 1 e⁻ above a 151-column running median, in 1 h): 94/157/57/152 on
  ch0/ch1/ch2/ch3. Most come in small adjacent groups; the lists are in `defects_hot_columns.csv`.
- Persistent hot pixels (> 25 e⁻ in at least 8 of 10 images) are almost absent: 1/0/0 on ch0/ch1/ch2, and none on ch3
  outside the hot-spot dome.
- Board 102 ch2 has 195 "hot" columns and 220 hot pixels, mostly at its noisy low-column end; that channel's column
  structure is strong.

## 5. Pocket pumping (`pp.py` → `pp_dipoles.csv`, `pp_summary.csv`, `pp_density_vs_charge.csv`, `pp_dipoles.png`)

**Method.** A dipole candidate is a pair of vertical neighbours, one more than 3σ above and one more than 3σ below a
local running median. Clusters and noisy columns are masked. The count is compared with the unpumped image of the same
geometry, and horizontal pairs serve as the control.

**Board 101: an artefact hides any trap dipoles.**
- All four CCDs show **3–5 % of pixel pairs as "dipoles" at every pump count, from N = 15 to N = 50000, with no trend in
  N**. The scatter of active pixels is 1.4–1.8 times that of the unpumped image.
- 2.8 times more pairs have the upper (later-read) pixel bright than the reverse.
- Horizontal pairs show the excess too.
- The rate grows with the local charge: 0.3–1.5 % below 5 e⁻/pix, 6–8 % at 5–10 e⁻ and 12–15 % above 10 e⁻. The dark
  member of each pair sits at about 0 e⁻ in absolute charge.
- So charge is moved, until the pixel is empty, into a neighbour in a large fraction of pixels already after 15 pumps.
  That is far too dense and too efficient for bulk traps (expected 10⁻⁴–10⁻³ per pixel, amplitude growing with N).
- The rate is the same on all four CCDs and absent on board 102. It is therefore an artefact of the pocket-pump clocking
  with board 101's configuration (`Vtransfer22`, PP_SHIFT = 3), for example charge spilling over a barrier phase during
  the pump or the return.
- It is absent in the 1-skip, 100 s, N = 5000 images: density 10⁻⁴, because those pixels held almost no charge when
  pumped.

**Board 102 ch2: no trap signal.**
- Pumped and unpumped images agree at every N. The excess is −0.7 ± 0.06 per kpix at N = 50000, i.e. no excess.
- In the 5–10 e⁻ charge bin the density is 0.0104 / 0.0098 at N = 15 / 50000 against 0.0097 unpumped. Pumpable traps
  with an amplitude above about 5 e⁻ at 60 µs are therefore fewer than about 10⁻³ per pixel.

**Emission time constants cannot be determined.** Every pocket-pump image uses the same phase time (60 µs), and trap
dipoles are hidden on board 101.

## 6. What separates 101 ch0 and ch2 from ch1 and ch3

1. **Serial-register dark response to VSUB.** This is the main physical result. The ch0/ch2 vs ch1/ch3 pairing from
   phase 1 is entirely in D_ser (×7 in absolute slope) and in the overscan charge. The array dark has no such pairing.
   Because each channel is a different CCD on the module, this is a property of each CCD's **serial register** (for
   example how VSUB changes the depletion or surface state under the horizontal gates or the channel stop): CCD_D and
   CCD_A against CCD_C and CCD_B. It is not an amplifier property in the transistor sense.
   - ch0 also has the lowest nominal D_ser (0.010 against 0.019–0.026 e⁻/pix/s), but ch2 does not, so the level and
     the VSUB slope are separate effects.
2. **101 ch2 alone:**
   - twice the single-sample noise and noise floor;
   - the only glow at the amplifier end of its register;
   - single electrons not resolved at 500–2000 skips in October.
   These point to the output stage (amplifier noise and luminescence) and fit the phase-1 RTS signature.
3. **Not distinguishing:**
   - thickness, depletion and diffusion are identical on all four CCDs;
   - serial CTI is below the sensitivity on all four;
   - the vertical deferred charge pairs ch2/ch3 against ch0/ch1, not ch0/ch2;
   - the pocket-pump artefact is the same on all four.

## 7. Not determinable from these data

- **Trap emission times.** There is a single phase time, and on board 101 the pocket-pump artefact hides any trap.
  This needs a scan of delay_V, and on board 101 a fixed or checked pocket-pump sequence.
- **Absolute pixel pitch** better than ±10 %, and **thickness** better than ±7 %. These need an energy calibration
  (for example ⁵⁵Fe, which also gives the gain independently).
- **Temperature.** It is not recorded, so neither the dark-current activation nor the absolute diffusion field can be
  interpreted.
- **102 ch3.** No charge reaches it in any image, so neither its gain in ADU/e⁻ nor any CCD quantity is known for it.
- **Which side of the module the CCDs sit on.** The geometric relations between the CCDs, and between amplifier and
  register end, are inferred from glow positions only. A module drawing is needed to test the light-from-CCD_B
  explanation for CCD_C's glow.
- **Vertical CTE per transfer** below 10⁻⁶, and serial CTE below 10⁻⁶. Measuring these needs EPER or charge-injection
  images.

## 8. Reproduce

```
cd amp_phys/images; PY=../../.venv/bin/python
$PY inventory.py; $PY units.py; $PY tracks.py --jobs 48; $PY dark.py --jobs 60; $PY defects.py; $PY pp.py
```

Decompressed images are cached as `.npz` (the `AMP_IMG_CACHE` environment variable, by default the session scratch
directory). `tracks_clusters.csv.gz` (98 MB) holds every cluster, so `tracks.py --analyse-only` can redo the fits without
re-reading the images.
