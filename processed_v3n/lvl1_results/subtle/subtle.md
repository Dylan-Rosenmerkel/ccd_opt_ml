# Subtle changes: pairs where the dominant knob barely moved

'held' = the dominant knob changed by less than 0.25 SD between the two packets. Pairs: random pairs, within packet (image), n -> n+1, n -> n+2. rho = Spearman of predicted vs true change; size = RMS error of the change (%).

## trained amplifiers

| target | subset | n | true change RMS % (label noise) | settings only | traces only | traces only (ensemble) | settings + traces | settings + traces (ensemble) |
|---|---|---|---|---|---|---|---|---|
| noise σe | all pairs | 2291 | 12.0 (1.9) | ρ 0.69, 9.6 | ρ 0.69, 9.2 | ρ 0.73, 8.5 | ρ 0.79, 8.0 | ρ 0.79, 7.9 |
| noise σe | VSUB held | 532 | 8.4 (1.9) | ρ 0.50, 6.7 | ρ 0.51, 6.6 | ρ 0.53, 6.3 | ρ 0.57, 6.1 | ρ 0.59, 6.0 |
| dark charge λ | all pairs | 2291 | 53.5 (12.0) | ρ 0.91, 27.5 | ρ 0.77, 32.2 | ρ 0.83, 27.3 | ρ 0.97, 13.8 | ρ 0.95, 17.7 |
| dark charge λ | VSUB held | 532 | 30.6 (12.0) | ρ 0.73, 15.7 | ρ 0.30, 19.7 | ρ 0.37, 17.5 | ρ 0.76, 10.1 | ρ 0.72, 11.0 |
| gain | all pairs | 2291 | 3.1 (0.8) | ρ 0.81, 1.9 | ρ 0.77, 2.2 | ρ 0.83, 1.9 | ρ 0.91, 1.4 | ρ 0.90, 1.4 |
| gain | VDD and VR held | 242 | 2.3 (0.8) | ρ 0.84, 1.3 | ρ 0.79, 1.6 | ρ 0.80, 1.6 | ρ 0.89, 1.0 | ρ 0.86, 1.1 |

## new amplifier

| target | subset | n | true change RMS % (label noise) | settings only | traces only | traces only (ensemble) | settings + traces | settings + traces (ensemble) |
|---|---|---|---|---|---|---|---|---|
| noise σe | all pairs | 2291 | 12.0 (1.9) | ρ 0.62, 10.7 | ρ 0.65, 10.2 | ρ 0.66, 9.4 | ρ 0.65, 10.5 | ρ 0.68, 9.3 |
| noise σe | VSUB held | 532 | 8.4 (1.9) | ρ 0.43, 7.3 | ρ 0.49, 7.1 | ρ 0.50, 6.7 | ρ 0.54, 7.5 | ρ 0.54, 6.5 |
| dark charge λ | all pairs | 2291 | 53.5 (12.0) | ρ 0.88, 35.8 | ρ 0.73, 36.1 | ρ 0.73, 33.8 | ρ 0.93, 24.9 | ρ 0.91, 25.2 |
| dark charge λ | VSUB held | 532 | 30.6 (12.0) | ρ 0.71, 20.3 | ρ 0.32, 22.4 | ρ 0.29, 20.3 | ρ 0.74, 15.8 | ρ 0.64, 16.8 |
| gain | all pairs | 2291 | 3.1 (0.8) | ρ 0.77, 2.2 | ρ 0.71, 2.4 | ρ 0.69, 2.5 | ρ 0.78, 2.3 | ρ 0.78, 2.3 |
| gain | VDD and VR held | 242 | 2.3 (0.8) | ρ 0.79, 1.6 | ρ 0.73, 1.8 | ρ 0.71, 1.7 | ρ 0.80, 1.6 | ρ 0.79, 1.6 |

