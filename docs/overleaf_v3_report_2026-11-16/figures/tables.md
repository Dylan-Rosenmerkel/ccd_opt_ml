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
| σe (noise) | VSUB +0.55* (+0.63) | VDD +0.20 (-0.04) | RG swing -0.17 (-0.10) | H3 low -0.16 (-0.35) | VR +0.16 (+0.17) |
| λ (dark charge) | VSUB +0.84* (+0.88) | VDD +0.10 (-0.20) | V1 low +0.09 (+0.29) | H swing -0.08 (-0.02) | RG swing -0.08 (-0.12) |
| gain | VDD -0.75* (-0.63) | VR -0.58* (-0.40) | H3 low +0.27* (+0.20) | RG swing +0.18 (-0.20) | V3 low -0.18 (-0.17) |
| trace: white noise | VDD -0.51* (-0.54) | VSUB +0.50* (+0.55) | V3 low -0.16 (+0.03) | H1 low -0.15 (-0.11) | V1 low -0.14 (+0.07) |
| trace: 1/f corner | H swing +0.38* (+0.44) | SW swing +0.14 (+0.01) | RG low +0.11 (+0.03) | V1 low -0.10 (-0.01) | V swing -0.10 (-0.01) |
| trace: Allan dev n=50 | VDD -0.11 (-0.12) | H1 low -0.11 (-0.07) | V2 low -0.10 (-0.08) | V3 low -0.09 (-0.06) | SW swing +0.09 (+0.01) |
| trace: power 0.1-1 MHz | VDD -0.29* (-0.29) | VR -0.24* (-0.28) | RG low -0.22 (-0.06) | RG swing -0.18 (-0.10) | H swing -0.14 (-0.17) |
| trace: power 1-3 MHz | RG low -0.24* (-0.23) | H swing -0.19 (-0.23) | H1 low +0.18 (+0.14) | H3 low -0.17 (-0.09) | RG swing -0.13 (+0.14) |
| trace: power 3-7.5 MHz | RG low -0.22 (-0.18) | H3 low -0.19 (-0.15) | V1 low -0.18 (-0.01) | H1 low +0.17 (+0.11) | H swing -0.14 (-0.18) |
| trace: slow/fast CDS power | VSUB -0.42* (-0.43) | V2 low -0.10 (-0.19) | VR +0.10 (+0.03) | H swing +0.09 (+0.05) | V3 low -0.09 (-0.16) |

