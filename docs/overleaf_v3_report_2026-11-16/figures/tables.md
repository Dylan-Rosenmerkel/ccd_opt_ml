# v3 report tables

## What each voltage does (top 5 per output, % per SD [68 %]; per V)

| output | R² | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|---|
| σe (image noise) | 0.78 | VSUB +4.9* (+0.5/V) | SW low +1.5* (+2.5/V) | VDD +1.4* (+1.4/V) | VR +1.2* (+2.4/V) | RG high -1.2* (-1.8/V) |
| λ (dark charge) | 0.95 | VSUB +34.1* (+3.5/V) | SW low +4.7* (+7.8/V) | VDD +2.1* (+2.1/V) | RG low +1.9* (+1.8/V) | RG high -1.7* (-2.4/V) |
| gain | 0.87 | VDD -1.6* (-1.6/V) | VR -1.1* (-2.1/V) | SW low -0.4* (-0.7/V) | H3 low +0.3* (+0.6/V) | RG high +0.3* (+0.4/V) |
| trace white noise | 0.87 | VSUB +1.2* (+0.1/V) | VDD -1.1* (-1.1/V) | V3 low -0.3* (-2.1/V) | V1 low -0.3* (-1.9/V) | SW low -0.2* (-0.4/V) |
| trace 1/f corner | -0.05 | H3 low -4.9* (-11.4/V) | H1 high +4.7* (+10.0/V) | H2 low -4.7* (-10.6/V) | H2 high +4.2* (+10.7/V) | H1 low -3.6* (-8.5/V) |
| trace Allan dev (n=50) | 0.41 | SW low -1.4* (-2.4/V) | V3 low -1.0* (-6.7/V) | VDD -0.9* (-1.0/V) | V2 low -0.9* (-5.7/V) | V1 low -0.9* (-5.8/V) |
| trace power 0.1–1 MHz | 0.95 | VDD -0.0* (-0.0/V) | VR -0.0* (-0.1/V) | RG high -0.0* (-0.0/V) | RG low -0.0* (-0.0/V) | H3 high -0.0* (-0.0/V) |
| trace power 1–3 MHz | 0.96 | RG low -0.1* (-0.1/V) | H3 high -0.1* (-0.1/V) | H1 low +0.1* (+0.2/V) | H2 low +0.0* (+0.1/V) | RG high -0.0* (-0.1/V) |
| trace power 3–7.5 MHz | 0.61 | RG low -0.4* (-0.4/V) | H3 high -0.4* (-0.7/V) | H1 low +0.3* (+0.7/V) | V1 high -0.3* (-3.2/V) | H2 low +0.2* (+0.5/V) |
| trace slow/fast CDS power | 0.28 | VSUB -6.2* (-0.6/V) | SW low -2.2* (-3.6/V) | VR +1.3 (+2.5/V) | VDD -0.9 (-0.9/V) | H2 high +0.9 (+2.2/V) |

## Frozen top 3, absolute (typical error %; session G → sealed session H)

| cell | target | 1st | 2nd | 3rd |
|---|---|---|---|---|
| trained | noise σe | trace: physics: 6.3 → 7.6 | settings + amplifier: trees: 6.4 → 7.7 | trace: trees: 6.5 → 8.0 |
| trained | dark charge λ | settings + amplifier: gp: 11.1 → 14.6 | settings + amplifier: trees: 11.9 → 10.5 | trace: trees: 20.6 → 26.1 |
| trained | gain | settings + amplifier: gp: 0.8 → 0.6 | settings + amplifier: trees: 1.0 → 0.9 | trace: physics: 1.1 → 1.1 |
| new | noise σe | trace: trees: 11.9 → 12.7 | trace: physics: 14.0 → 13.5 | trace: nn: 15.3 → 14.2 |
| new | dark charge λ | trace: trees: 39.9 → 49.5 | trace: physics: 51.5 → 50.9 | trace: nn: 57.2 → 51.1 |
| new | gain | settings: trees: 3.7 → 3.6 | settings: gp: 3.7 → 3.9 | trace: physics: 4.0 → 4.3 |

## Frozen top 3, relative (size error % (random pairs); session G → sealed session H)

| cell | target | 1st | 2nd | 3rd |
|---|---|---|---|---|
| trained | noise σe | ensemble: settings + trace: 8.0 → 10.0 | settings + trace: combined trees: 8.1 → 11.4 | ensemble: trace models: 8.5 → 10.3 |
| trained | dark charge λ | settings + trace: combined trees: 14.0 → 15.3 | absolute settings + amplifier: gp, differenced: 14.4 → 19.3 | absolute settings + amplifier: trees, differenced: 16.4 → 14.6 |
| trained | gain | absolute settings + amplifier: gp, differenced: 1.0 → 0.8 | absolute settings + amplifier: trees, differenced: 1.4 → 1.3 | settings + trace: combined trees: 1.4 → 0.9 |
| new | noise σe | ensemble: settings + trace: 9.3 → 11.2 | ensemble: trace models: 9.4 → 11.5 | trace change: difference curves PCA + trees: 9.7 → 14.2 |
| new | dark charge λ | ensemble: settings + trace: 25.3 → 34.3 | settings + trace: combined trees: 25.3 → 29.9 | settings + trace: residual trees: 32.6 → 33.5 |
| new | gain | settings change: gp: 2.3 → 1.9 | settings change: trees: 2.3 → 1.8 | ensemble: settings + trace: 2.3 → 2.1 |

## Partial correlation, top 5 per output (plain correlation in brackets)

| output | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| σe (noise) | VSUB +0.55* (+0.63) | VDD +0.20 (-0.04) | RG high -0.17 (-0.06) | VR +0.16 (+0.17) | SW low +0.15 (-0.21) |
| λ (dark charge) | VSUB +0.84* (+0.88) | SW low +0.16 (-0.18) | VDD +0.10 (-0.20) | RG low +0.09 (+0.13) | V3 high +0.09 (+0.02) |
| gain | VDD -0.75* (-0.63) | VR -0.58* (-0.40) | SW low -0.20 (+0.10) | RG high +0.18 (-0.20) | H2 low +0.15 (+0.07) |
| trace: white noise | VDD -0.51* (-0.54) | VSUB +0.50* (+0.55) | H3 low +0.09 (-0.10) | RG high +0.09 (-0.01) | SW low -0.09 (-0.03) |
| trace: 1/f corner | RG low +0.09 (+0.03) | VSUB -0.08 (-0.02) | VDD -0.08 (-0.01) | RG high +0.07 (+0.01) | V1 high -0.05 (-0.03) |
| trace: Allan dev n=50 | SW low -0.12 (-0.01) | VDD -0.11 (-0.12) | RG high -0.08 (-0.08) | RG low +0.08 (+0.06) | V3 low +0.07 (-0.06) |
| trace: power 0.1-1 MHz | VDD -0.29* (-0.29) | VR -0.24* (-0.28) | RG high -0.18 (-0.28) | RG low -0.13 (-0.06) | V1 low +0.10 (+0.00) |
| trace: power 1-3 MHz | RG low -0.25* (-0.23) | RG high -0.13 (-0.09) | VR -0.11 (-0.11) | V1 low +0.11 (-0.04) | H2 low +0.09 (+0.06) |
| trace: power 3-7.5 MHz | RG low -0.21 (-0.18) | RG high -0.12 (-0.09) | VSUB +0.12 (+0.14) | V1 low +0.10 (-0.01) | VR -0.10 (-0.06) |
| trace: slow/fast CDS power | VSUB -0.42* (-0.43) | SW low -0.12 (+0.05) | V1 low -0.10 (-0.09) | VR +0.10 (+0.03) | H1 low -0.09 (+0.06) |

