# T4: the obvious part (timing) and what is left for ML

Amplifier offsets removed first (the amplifier's mean; one calibration image), GP models, cross-validated by packet; 'new amp' = the amplifier left out AND the fleet trained on other time blocks. R2 = share of the remaining variance explained.

| target | spread % | R2 timing only | R2 all settings | voltages on the timing residual: R2 trained / new amp | left after all % | label noise % |
|---|---|---|---|---|---|---|
| sigma_e | 14.1 | 0.50 | 0.62 | 0.13 / 0.08 | 8.8 | 1.8 |
| lambda | 38.6 | 0.19 | 0.80 | 0.68 / 0.43 | 17.3 | 4.1 |
| gain | 2.1 | 0.00 | 0.74 | 0.73 / 0.61 | 1.1 | 1.0 |

## sigma_e: voltage effects on the timing residual (% per SD of the knob)

| knob | shared | amplifier-to-amplifier spread | slope error | amplifier-specific? |
|---|---|---|---|---|
| VSUB | +3.8 | 1.9 | 0.9 | yes |
| SW_1_L (+1) | +2.3 | 1.4 | 1.2 |  |
| V1_B_L (+1) | +1.3 | 0.5 | 0.5 |  |
| VDD1 (+1) | +1.1 | 1.1 | 0.8 |  |
| VR1 (+1) | +1.1 | 0.6 | 0.8 |  |
| V2_C_L | +1.0 | 0.6 | 0.5 |  |
| V3_B_L (+1) | +1.0 | 0.2 | 0.5 |  |
| H1_B_L (+1) | -0.7 | 0.5 | 0.6 |  |
| RG_1_H (+1) | -0.6 | 1.0 | 0.8 |  |
| H1_B_H (+1) | -0.6 | 0.3 | 0.6 |  |
| RG_1_L (+1) | +0.5 | 0.8 | 0.8 |  |
| V1_B_H (+1) | +0.5 | 0.6 | 0.7 |  |
| H2_C_H | +0.5 | 0.4 | 0.7 |  |
| H3_B_L (+1) | -0.4 | 0.4 | 0.7 |  |
| H3_B_H (+1) | -0.4 | 0.6 | 0.6 |  |
| H2_C_L | +0.3 | 0.1 | 0.6 |  |
| TG_B_H (+3) | +0.3 | 0.5 | 0.7 |  |
| V3_B_H (+1) | -0.1 | 0.3 | 0.7 |  |
| V2_C_H | +0.0 | 0.5 | 0.6 |  |

## lambda: voltage effects on the timing residual (% per SD of the knob)

| knob | shared | amplifier-to-amplifier spread | slope error | amplifier-specific? |
|---|---|---|---|---|
| VSUB | +31.7 | 16.7 | 1.3 | yes |
| SW_1_L (+1) | +5.3 | 3.5 | 1.8 |  |
| VDD1 (+1) | +2.4 | 1.5 | 1.1 |  |
| TG_B_H (+3) | -2.2 | 0.7 | 1.0 |  |
| H1_B_L (+1) | +2.2 | 0.5 | 0.9 |  |
| H2_C_L | +1.9 | 0.9 | 0.8 |  |
| H3_B_H (+1) | -1.5 | 1.3 | 0.9 |  |
| V3_B_L (+1) | +1.3 | 0.6 | 0.7 |  |
| V3_B_H (+1) | +1.2 | 0.7 | 1.1 |  |
| V1_B_L (+1) | +1.2 | 1.1 | 0.7 |  |
| V1_B_H (+1) | +0.9 | 0.4 | 1.1 |  |
| H3_B_L (+1) | +0.9 | 0.4 | 1.0 |  |
| H2_C_H | -0.7 | 0.2 | 1.0 |  |
| V2_C_L | +0.7 | 0.6 | 0.7 |  |
| RG_1_H (+1) | -0.6 | 0.7 | 1.2 |  |
| RG_1_L (+1) | +0.3 | 0.6 | 1.1 |  |
| H1_B_H (+1) | -0.2 | 0.5 | 0.9 |  |
| VR1 (+1) | -0.1 | 0.5 | 1.2 |  |
| V2_C_H | -0.0 | 0.8 | 0.9 |  |

## gain: voltage effects on the timing residual (% per SD of the knob)

| knob | shared | amplifier-to-amplifier spread | slope error | amplifier-specific? |
|---|---|---|---|---|
| VDD1 (+1) | -1.3 | 0.1 | 0.1 |  |
| VR1 (+1) | -1.1 | 0.5 | 0.1 | yes |
| RG_1_H (+1) | +0.3 | 0.1 | 0.1 |  |
| H3_B_L (+1) | +0.2 | 0.1 | 0.1 |  |
| VSUB | +0.2 | 0.2 | 0.1 |  |
| V2_C_H | +0.1 | 0.1 | 0.1 |  |
| SW_1_L (+1) | -0.1 | 0.1 | 0.2 |  |
| H3_B_H (+1) | +0.1 | 0.1 | 0.1 |  |
| H1_B_L (+1) | +0.1 | 0.2 | 0.1 |  |
| H2_C_H | -0.1 | 0.1 | 0.1 |  |
| V3_B_L (+1) | -0.1 | 0.0 | 0.1 |  |
| V1_B_L (+1) | -0.1 | 0.0 | 0.1 |  |
| V1_B_H (+1) | +0.1 | 0.0 | 0.1 |  |
| H1_B_H (+1) | +0.1 | 0.1 | 0.1 |  |
| H2_C_L | -0.0 | 0.0 | 0.1 |  |
| RG_1_L (+1) | -0.0 | 0.0 | 0.1 |  |
| TG_B_H (+3) | +0.0 | 0.1 | 0.1 |  |
| V2_C_L | +0.0 | 0.0 | 0.1 |  |
| V3_B_H (+1) | -0.0 | 0.0 | 0.1 |  |
