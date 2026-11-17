# T4: the obvious part (timing) and what is left for ML

Amplifier offsets removed first (the amplifier's mean; one calibration image), GP models, cross-validated by packet; 'new amp' = the amplifier left out AND the fleet trained on other time blocks. R2 = share of the remaining variance explained.

| target | spread % | R2 timing only | R2 all settings | voltages on the timing residual: R2 trained / new amp | left after all % | label noise % |
|---|---|---|---|---|---|---|
| sigma_e | 8.9 | 0.00 | 0.31 | 0.31 / 0.12 | 7.4 | 1.4 |
| lambda | 39.1 | 0.00 | 0.79 | 0.79 / 0.60 | 18.0 | 3.8 |
| gain | 2.3 | 0.00 | 0.77 | 0.77 / 0.62 | 1.1 | 1.1 |

## sigma_e: voltage effects on the timing residual (% per SD of the knob)

| knob | shared | amplifier-to-amplifier spread | slope error | amplifier-specific? |
|---|---|---|---|---|
| VSUB | +4.7 | 2.4 | 0.8 | yes |
| SW_1_L (+1) | +1.9 | 1.8 | 1.0 |  |
| VDD1 (+1) | +1.5 | 1.5 | 0.7 | yes |
| VR1 (+1) | +1.5 | 0.9 | 0.7 |  |
| RG_1_H (+1) | -1.2 | 1.1 | 0.7 |  |
| V1_B_L (+1) | +1.2 | 0.6 | 0.5 |  |
| H1_B_H (+1) | -0.9 | 0.4 | 0.5 |  |
| RG_1_L (+1) | +0.8 | 0.8 | 0.7 |  |
| V2_C_L | +0.8 | 1.0 | 0.5 |  |
| H3_B_H (+1) | -0.7 | 0.7 | 0.5 |  |
| V3_B_L (+1) | +0.7 | 0.5 | 0.5 |  |
| H2_C_L | +0.6 | 0.4 | 0.5 |  |
| V1_B_H (+1) | +0.5 | 0.7 | 0.6 |  |
| H1_B_L (+1) | -0.4 | 0.4 | 0.6 |  |
| V3_B_H (+1) | -0.3 | 0.2 | 0.6 |  |
| H3_B_L (+1) | -0.2 | 0.2 | 0.6 |  |
| V2_C_H | -0.0 | 0.5 | 0.5 |  |
| H2_C_H | +0.0 | 0.5 | 0.6 |  |

## lambda: voltage effects on the timing residual (% per SD of the knob)

| knob | shared | amplifier-to-amplifier spread | slope error | amplifier-specific? |
|---|---|---|---|---|
| VSUB | +35.3 | 18.9 | 1.3 | yes |
| SW_1_L (+1) | +3.0 | 3.0 | 1.8 |  |
| VDD1 (+1) | +2.1 | 1.9 | 1.2 |  |
| H3_B_H (+1) | -2.1 | 1.4 | 0.8 |  |
| H1_B_L (+1) | +1.2 | 0.3 | 1.0 |  |
| H2_C_H | -1.2 | 0.6 | 1.0 |  |
| RG_1_L (+1) | +1.2 | 0.8 | 1.2 |  |
| RG_1_H (+1) | -1.1 | 1.1 | 1.3 |  |
| VR1 (+1) | -0.7 | 0.4 | 1.3 |  |
| V1_B_H (+1) | +0.7 | 0.3 | 1.0 |  |
| V2_C_L | -0.7 | 0.8 | 0.9 |  |
| H3_B_L (+1) | -0.7 | 0.6 | 1.0 |  |
| H2_C_L | +0.6 | 0.7 | 0.9 |  |
| H1_B_H (+1) | -0.5 | 0.7 | 0.8 |  |
| V3_B_H (+1) | +0.4 | 0.7 | 1.0 |  |
| V2_C_H | -0.3 | 0.9 | 0.8 |  |
| V3_B_L (+1) | -0.2 | 0.6 | 0.9 |  |
| V1_B_L (+1) | -0.0 | 1.2 | 0.9 |  |

## gain: voltage effects on the timing residual (% per SD of the knob)

| knob | shared | amplifier-to-amplifier spread | slope error | amplifier-specific? |
|---|---|---|---|---|
| VDD1 (+1) | -1.5 | 0.1 | 0.1 |  |
| VR1 (+1) | -1.3 | 0.7 | 0.1 | yes |
| RG_1_H (+1) | +0.4 | 0.1 | 0.1 |  |
| SW_1_L (+1) | -0.3 | 0.3 | 0.2 |  |
| VSUB | +0.3 | 0.2 | 0.1 |  |
| H3_B_L (+1) | +0.2 | 0.1 | 0.1 |  |
| V3_B_L (+1) | -0.2 | 0.1 | 0.1 |  |
| H1_B_L (+1) | +0.2 | 0.1 | 0.1 |  |
| V2_C_H | +0.2 | 0.1 | 0.1 |  |
| H2_C_H | -0.2 | 0.1 | 0.1 |  |
| V1_B_L (+1) | -0.1 | 0.2 | 0.1 |  |
| H3_B_H (+1) | +0.1 | 0.1 | 0.1 |  |
| V1_B_H (+1) | +0.1 | 0.0 | 0.1 |  |
| RG_1_L (+1) | -0.1 | 0.0 | 0.1 |  |
| H1_B_H (+1) | +0.0 | 0.1 | 0.1 |  |
| H2_C_L | -0.0 | 0.0 | 0.1 |  |
| V2_C_L | -0.0 | 0.1 | 0.1 |  |
| V3_B_H (+1) | -0.0 | 0.0 | 0.1 |  |
