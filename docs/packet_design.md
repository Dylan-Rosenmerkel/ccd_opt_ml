# Packet design: image size, binning, skips, and a trace that matches the image

Written 2026-10-13. Implemented as `configs/packet_recommended.toml` and the `[trace] match_image = true`
option (`src/daq_commands.py`). All times come from `take_packet.formula_acquisition_seconds` (the
sequencer formulas). That formula reproduces the 2026-10-12 trace files exactly: 0.203 s → 12.2 MB and
1.137 s → 68.2 MB.

## 1. What a packet must deliver

| part | purpose | requirement |
|---|---|---|
| image | the **labels**: σe, gain, λ from WADERs | photon counting: most pixels hold 0 or 1 e⁻, and the 0 and 1 e⁻ peaks are separated |
| traces | the **inputs**: 85 features per channel | describe the readout that produced the image; repeatable; small files |
| both | ~6 packets per hour | image ≈ 5 min, packet ≈ 10–12 min |

The 2026-10-12 run showed why the image part matters. Its 20×20 binned images held 20–35 e⁻ per bin, so
no WADERs label could be made.

## 2. Number of skips: the noise needed to count electrons

With $N$ skips per pixel and white noise, the noise of the averaged pixel is

$$\sigma_N = \frac{\sigma_1}{\sqrt N},$$

where $\sigma_1$ is the single-skip noise. From the WADERs fit of the 2026-10-06 1×1 image (3000 skips):

$$\sigma_{3000} = 0.106\ e^- \;(\text{ch0}) \;\Rightarrow\; \sigma_1 = 0.106\sqrt{3000} \approx 5.8\ e^- .$$

Channels 1 and 3 are the same within 5 %. Channel 2 is about 4× noisier in every trace and cannot count
electrons at any practical $N$: it would need $(23/0.25)^2 \approx 8500$ skips.

A pixel with 0 e⁻ is misread as ≥ 1 e⁻ when its value exceeds 0.5 e⁻:

$$P_\text{mis}(N) = 1 - \Phi\!\left(\frac{0.5\,e^-}{\sigma_N}\right).$$

| $N$ | $\sigma_N$ (e⁻) | $P_\text{mis}$ |
|---|---|---|
| 200 | 0.41 | 0.111 |
| 300 | 0.34 | 0.068 |
| 400 | 0.29 | 0.042 |
| **500** | **0.26** | **0.027** |
| 800 | 0.21 | 0.007 |
| 1000 | 0.18 | 0.003 |

The dark charge per pixel is only $\lambda \approx 0.08$ e⁻ (section 4). For λ to be measurable, $P_\text{mis}$ must
be well below λ. WADERs fits the Gaussian tails rather than cutting at 0.5 e⁻, but λ and σ become strongly
correlated when the peaks overlap.

* **200 skips is too few.** Misread zeros (11 %) outnumber the real electrons. This corrects the 200-skip
  image recommended on 2026-10-12; that estimate of σ was too optimistic.
* **500 skips** gives $P_\text{mis} = 2.7\,\%$, a third of λ. This is the minimum that works.
* More skips cost pixels: the time per pixel grows like $N$ (section 3).

**Choice: $N = 500$.** Caveat: this assumes white noise from 500 to 3000 skips. The first packet must check
$\sigma_{500}$ directly. If it is above ~0.3 e⁻ (low-frequency noise), go to 800 skips.

## 3. Readout time

Times in sequencer ticks of 10 ns, with the reference delays:

$$t_\text{CDS} = \text{OGH+OGL+RGL+RGH} + 2\,\text{Int} + \text{SWH+SWL} = 3258 \text{ ticks} = 32.58\ \mu s$$
$$t_H = \text{RGL} + 3H_a + 3H_b + H_c = 126.3\ \mu s, \qquad t_V = \text{RGL} + 3V_a + 3V_b = 360\ \mu s$$

One read pixel, one skipped (prescan) pixel, and one row:

$$t_\text{pix} = N\,t_\text{CDS} + N_\text{SBIN}\,t_H + t_\text{ped} + t_\text{DG} \approx 16.5\ \text{ms} \quad (N = 500,\ 1\times1)$$
$$t_\text{skip} = N_\text{SBIN}\,t_H + t_\text{DG}$$
$$t_\text{row} = N_\text{COL}\,t_\text{pix} + N_\text{COLPRE}\,t_\text{skip} + N_\text{PBIN}\,t_V + B_\text{CLR}\,t_\text{clear}, \qquad t_\text{clear} = 616\,(10\,t_H + t_\text{DG}) = 0.78\ \text{s}$$

The image takes $t_\text{img} = N_\text{ROW}\,t_\text{row} + \text{EXPOSURE}$.

For 1000 columns after a 5300-column prescan with the clear on:

$$t_\text{row} = 16.5 + 0.67 + 0.78 \approx 17.9\ \text{s}.$$

So **16 rows take 4.8 min**.

## 4. Dark charge: how many electrons the image counts

Pixel $(r, c)$ collects dark charge from the end of the last flush until it is read:

$$\lambda_{rc} \approx \text{DC}\,\bigl(\text{EXPOSURE} + r\,t_\text{row} + c\,t_\text{pix}\bigr) + \lambda_\text{CIC}$$

* DC ≈ 5.5×10⁻⁴ e⁻/pix/s (2026-10-12: 4.4–6.7×10⁻⁴).
* $\lambda_\text{CIC}$ is the clock-induced charge from the transfers.

Averaged over the image, $\bar\lambda \approx \text{DC}\,(\bar r\,t_\text{row} + t_\text{row}/2) \approx 0.08$ e⁻. The
number of one-electron pixels and the statistical precision of λ are

$$N_1 = \sum_{rc} \lambda_{rc} e^{-\lambda_{rc}} \approx 1000, \qquad \frac{\delta\lambda}{\lambda} \approx \frac{1}{\sqrt{N_1}} \approx 3\,\% .$$

The gain comes from the position of the 1 e⁻ peak:

$$\frac{\delta G}{G} \approx \frac{\sigma_N}{\sqrt{N_1}} \approx 0.8\,\% .$$

The read noise comes from the width of the 0 e⁻ peak (about 13,000 pixels):

$$\frac{\delta\sigma}{\sigma} \approx \frac{1}{\sqrt{2 N_0}} < 1\,\% .$$

**The rows separate dark current from clock-induced charge.** Row $r$ waits $r\,t_\text{row}$ longer than row 0.
Fitting λ per row,

$$\lambda(r) = \lambda_0 + \text{DC}\cdot t_\text{row}\cdot r ,$$

gives DC as the slope and $\lambda_0$ (CIC plus the charge collected before the first row) as the intercept.
With 16 rows spread over 4.8 min the slope is well constrained. For the ML scan this matters: changing the
clock rails changes the **CIC**, and changing the substrate or the temperature changes the **DC**, so both
should be labels. (WADERs has a per-row dark-current option; `configs/processing.toml` should use it.)

## 5. Options compared (all ~5 min)

| option | layout | time | active pixels | $\bar\lambda$ | $N_1$ | $\delta\lambda/\lambda$ | $\sigma_N$ |
|---|---|---|---|---|---|---|---|
| earlier (10-12) | 35×1000×200, 1×1 | 4.9 min | 29,800 | 0.084 | 2,300 | 2.1 % | **0.41** ✗ |
| **A (recommended)** | **16×1000×500, 1×1, prescan 5300** | **4.8 min** | **13,600** | **0.079** | **990** | **3.2 %** | **0.26** |
| B | 3×6300×500, 1×1, full width | 5.2 min | 18,400 | 0.086 | 1,460 | 2.6 % | 0.26 |
| C | 30×500×500, 2×2 binned, prescan 2650 | 4.9 min | 12,800 bins | 0.32/bin | ~3,000 | 1.8 % | 0.26 |
| D | 12×1000×800, 1×1 | 5.5 min | 10,200 | 0.091 | 850 | 3.4 % | 0.21 |

* **A** reads the last 852 active columns and 148 overscan columns of each row. That is the column layout
  of the existing WADERs JSON (`img_9.json`), with 16 rows for the per-row fit.
* **B** reads every column (no prescan skipping, so 35 % more pixels per minute) but only 3 rows, so the
  per-row DC fit has 3 points. It is a good alternative if the whole register width should be seen
  (column defects).
* **C** collects 4× the charge per bin and reads half as many bins per row, so it fits 30 rows. λ is
  more precise, but each bin mixes 4 pixels, CIC changes with
  binning, and the matched trace would also be 2×2.
* **D** buys margin in noise (if $\sigma_1$ turns out higher) at the cost of pixels.

**Binning: 1×1.** It keeps the labels per pixel and in the photon-counting regime, and it is what the trace
matches (section 6).

## 6. The trace should read its pixels like the image

The traces are the model's input, and the image gives the labels. Features measured on a **different**
readout describe a different operating state. In 2026-10-12 the traces used 20×20 binning: each pixel
saw 20 horizontal transfers and its row 20 vertical transfers, and each bin collected 20–35 e⁻ of
dark charge. The images used other settings. So the trace and the label did not describe the same
pixels.

**Rule: the trace copies every key that decides how one pixel is read**
(`daq_commands.MATCHED_KEYS`):

| key | why it must match |
|---|---|
| `NDCM` (and `NDCMPRE`) | same number of skips: baseline droop, ringing and low-frequency noise per pixel |
| `NSBIN` | same number of horizontal transfers before every pixel (H-transfer features; serial CIC) |
| `NPBIN` | same vertical transfers per row (V-transfer features; parallel CIC) |
| `subroutine` | same sequencer routine |
| timing and voltages | already the same: they are θ and apply to both |

**What cannot match, and why.** A trace stores every ADC sample:

$$\text{size} = 4\ \text{bytes} \times 15\ \text{MHz} \times t_\text{trace} = 60\ \text{MB per second of readout}.$$

| key | image | matched trace | reason |
|---|---|---|---|
| `NROW` × `NCOL` | 16 × 1000 | 2 × 5 | a full image row of traces would be 16.5 s ≈ 1 GB |
| `NCOLPRE` | 5300 | **100** | the whole prescan is 0.67 s ≈ 40 MB per row; 100 skipped pixels give the read pixels a stretch of horizontal clocking before them, as in the image, for 0.8 MB per row |
| `BOOLCLR` | 1 | 0 | the register clear is 0.78 s ≈ 47 MB per row; it is 6160 ordinary horizontal transfers, already described by the H features |
| `EXPOSURE` | 0 | 0 | — |

Matched trace: 2 rows × 5 pixels × 500 skips, 1×1, prescan 100. That is **0.19 s and 11.4 MB per
channel**, the same size as today's 12 MB traces. With 6 trace sets × 4 channels × 2 boards that is
0.55 GB per packet (55 GB per 100 packets). Use `[trace] match_image = true`. Setting a matched key by hand
is refused, so the match cannot be broken by mistake. The manifest records `trace_matches_image`. When a
packet's trace and image differ, `plan_packet` logs a warning listing the keys.

**Consequences for the features:**
* **Horizontal-transfer features:** each pixel now has 1 transfer instead of 20. Each row adds a burst of
  100 transfers, so there are still ~210 transfers per trace.
* **Vertical-transfer features:** 1 transfer per row, so 2 per trace instead of ~40. The V waveform is
  thousands of ADU against ~10 ADU of noise, so it is still measured. If more is wanted, use 4 × 3 pixels
  (4 rows, 0.25 s, 15 MB).
* **Pixels are empty** (λ ≈ 0 in 0.2 s): the CDS features are not disturbed by dark charge, unlike the 20×20
  traces.
* **The analysis handles this layout:** `trace_regions` now accepts the row-start pixels that carry the
  prescan transfers (`NSBIN·(NCOLPRE+1)`), and `check_trace_analysis.py` has a "matched to a 1x1 image
  (prescan 100)" case. All 27 configurations pass.

## 7. The recommended packet

`configs/packet_recommended.toml`:
3 flushes → 3 reference trace sets (flush between) → θ → flush → 3 trace sets → image
16×1000×500, 1×1 (leader mode, image, back to standalone, flush).

**25 steps, about 11.5 min** (8 flushes 3.9 min, image 4.8 min, 24 trace acquisitions), i.e. ~5 packets
per hour.

## 8. To check on the first real packet

1. $\sigma_{500}$ from WADERs on channels 0, 1 and 3 should be ≈ 0.26 e⁻. If it is above 0.3 e⁻, use
   800 skips (option D).
2. λ per row should rise linearly. The slope gives DC; the intercept gives CIC plus earlier charge.
3. The matched traces should decode without loss and with no layout warnings.
4. The WADERs JSON: `img_9.json` with `ccd_rows = 16`, `ccd_cols = 852`, and per-row dark current.
5. Also note: the 2026-10-12 binned overscan noise (0.42–0.66 ADU at 500 skips) was above the white-noise
   prediction (0.35 ADU). The cause may be charge in the 20-pixel serial bins or extra low-frequency
   noise; item 1 will tell.

## 9. One big trace or several small ones?

**Answer: several small traces, separate acquisitions.** If more trace data is wanted per packet, add more
small traces rather than making each trace bigger.

**The measurement** (2026-10-12, board 101, settings 2–5, 3 trace sets each, 47 traces). Most features are
the median over the ~8 usable pixels of a trace. That gives two ways to estimate a feature's noise:

* **within one trace**, from the pixel-to-pixel spread $s_\text{pix}$ (robust, `TraceFeatures.spread`):
  $\text{SE}_\text{pix} = 1.2533\, s_\text{pix}/\sqrt{n_p}$, with $n_p = 8$ (1.2533 converts to the standard
  error of a median);
* **between traces**, from the scatter of the trace-level value over the repeated trace sets of each
  setting, pooled over the four settings: $s_\text{trace}$.

If each acquisition added nothing of its own, $s_\text{trace} \approx \text{SE}_\text{pix}$. Write the
between-trace variance as an acquisition part plus a pixel part:

$$s_\text{trace}^2 = \sigma_\text{acq}^2 + \frac{\sigma_\text{pix}^2}{n_p}, \qquad
R = \frac{s_\text{trace}}{\text{SE}_\text{pix}} .$$

For the 52 per-pixel features (median over the 4 channels):

| ratio $R$ | features | meaning |
|---|---|---|
| ≈ 1 (median 1.1, quartiles 0.9–1.4) | about two thirds: CDS noise, spectra, levels | no acquisition term: **only the total number of pixels counts**, in one trace or many |
| 4–170 | the amplifier-response features: dip recovery amplitude (167) and time (31), settling amplitude (8.7) and time (7.4), SW and RG feedthrough per volt (7.6, 4.1), dip height (6.8), pulse height (4.1) | **each acquisition has its own value**: $\sigma_\text{acq}$ dominates |

The second group holds exactly the features that respond to VDD (section 4 of the packet-test analysis).
For them the uncertainty of a packet's mean over $N_t$ traces is

$$\operatorname{Var}(\bar f) = \frac{1}{N_t}\left(\sigma_\text{acq}^2 + \frac{\sigma_\text{pix}^2}{n_p}\right).$$

Example: the settling time has $R \approx 7$, so $\sigma_\text{acq}^2 = 48\,\text{SE}_\text{pix}^2$.
* **10× more pixels per trace** ($n_p = 80$): the variance goes from 49 to 48.1, so the noise drops by **1 %**.
* **4× more traces**: the noise halves.

**Other reasons for several small traces:**
1. **Honest noise estimates.** The model needs the trace-to-trace scatter as each feature's noise
   (`trace_sd.*` in `dataset.csv`). This scatter can only be measured with separate acquisitions. Pixels
   inside one trace give the pixel part only, and underestimate the noise of the features that matter most
   (the earlier pixel-level comparison test made the same mistake: it flagged repeats as "different").
2. **Lost data.** The chance that a trace loses UDP packets grows with its length (2 of ~80 traces so far,
   at 12–68 MB). A damaged trace is refused and retaken, which costs 0.2 s for a small trace and the whole
   acquisition for a big one.
3. **Same cost per pixel.** Disk use is 60 MB per second of readout either way. The overhead is ~1.5 s per
   `take_trace` call, small next to the 27 s flushes.

**What a big trace would add:**
* lower frequencies in the noise features: a trace of length $T$ reaches down to $\sim 1/T$. The 0.19 s
  matched trace covers ≥ 5 Hz. One image pixel (500 skips) takes 16.5 ms, i.e. 60 Hz, so the noise that
  matters for $\sigma_e$ is covered;
* more row-level features (vertical transfers, row starts) per trace. More small traces give these too, or
  use 4 × 3 pixels instead of 2 × 5.

**Suggestion:**
* keep 2 × 5-pixel matched traces;
* if the time budget allows, use **5 reference + 5 θ sets** instead of 3 + 3. That is about +2.7 min per
  packet (each set is ~8 trace calls plus a flush) and gives ~30 % smaller errors on the amplifier
  features;
* worth testing once on the DAQ: whether the flush **between** trace sets is needed. Without it, a set
  costs ~14 s instead of ~41 s, so many more repeats fit. Compare $s_\text{trace}$ of sets taken with and
  without flushes.

These ratios were measured on the 20×20 traces of 2026-10-12. Repeat the check on the first matched 1×1
traces.
