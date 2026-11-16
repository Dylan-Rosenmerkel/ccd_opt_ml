# Correlations, v3 voltage-only data (session G)

232 rows, 80 packets, amplifiers [np.int64(0), np.int64(1), np.int64(3)]; within amplifier; outputs in log units. Significant (95 %, n_eff = packets) when |r| >= 0.22. Partial = with the other voltages and the amplifier held fixed.

Knob groups that moved together: SW low = SW_1_L+SW_2_L; RG high = RG_1_H+RG_2_H; RG low = RG_1_L+RG_2_L; V1 high = V1_B_H+V1_A_H; V1 low = V1_B_L+V1_A_L; V3 high = V3_B_H+V3_A_H; V3 low = V3_B_L+V3_A_L; H1 high = H1_B_H+H1_A_H; H1 low = H1_B_L+H1_A_L; H3 high = H3_B_H+H3_A_H; H3 low = H3_B_L+H3_A_L; VDD = VDD1+VDD2; VR = VR1+VR2

| output | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| σe (noise) | VSUB +0.54 | VDD +0.20 (n.s.) | RG swing -0.17 (n.s.) | VR +0.16 (n.s.) | H3 low -0.16 (n.s.) |
| λ (dark charge) | VSUB +0.84 | VDD +0.10 (n.s.) | V1 low +0.09 (n.s.) | H swing -0.08 (n.s.) | RG swing -0.08 (n.s.) |
| gain | VDD -0.75 | VR -0.58 | H3 low +0.27 | RG swing +0.18 (n.s.) | V3 low -0.18 (n.s.) |
| trace: white noise | VDD -0.51 | VSUB +0.50 | V3 low -0.16 (n.s.) | H1 low -0.15 (n.s.) | V1 low -0.14 (n.s.) |
| trace: 1/f corner | H swing +0.38 | SW swing +0.14 (n.s.) | RG low +0.11 (n.s.) | V1 low -0.10 (n.s.) | V swing -0.10 (n.s.) |
| trace: Allan dev n=50 | VDD -0.11 (n.s.) | H1 low -0.11 (n.s.) | V2 low -0.10 (n.s.) | V3 low -0.09 (n.s.) | SW swing +0.09 (n.s.) |
| trace: power 0.1-1 MHz | VDD -0.29 | VR -0.24 | RG low -0.22 (n.s.) | RG swing -0.18 (n.s.) | H swing -0.14 (n.s.) |
| trace: power 1-3 MHz | RG low -0.24 | H swing -0.19 (n.s.) | H1 low +0.18 (n.s.) | H3 low -0.17 (n.s.) | RG swing -0.13 (n.s.) |
| trace: power 3-7.5 MHz | RG low -0.22 (n.s.) | H3 low -0.19 (n.s.) | V1 low -0.18 (n.s.) | H1 low +0.17 (n.s.) | H swing -0.14 (n.s.) |
| trace: slow/fast CDS power | VSUB -0.42 | V2 low -0.10 (n.s.) | VR +0.09 (n.s.) | H swing +0.09 (n.s.) | V3 low -0.09 (n.s.) |

Largest correlations between outputs (within amplifier):

- trace: power 1-3 MHz ~ trace: power 3-7.5 MHz: +0.94
- σe (noise) ~ λ (dark charge): +0.74
- trace: power 0.1-1 MHz ~ trace: power 3-7.5 MHz: +0.56
- trace: power 0.1-1 MHz ~ trace: power 1-3 MHz: +0.56
- λ (dark charge) ~ trace: slow/fast CDS power: -0.49
- λ (dark charge) ~ trace: white noise: +0.46
- trace: white noise ~ trace: power 0.1-1 MHz: +0.44
- σe (noise) ~ trace: slow/fast CDS power: -0.43
- gain ~ trace: white noise: +0.38
- gain ~ trace: power 0.1-1 MHz: +0.34

Design: largest |correlation| between two voltages over the packets 1.00, median 0.15.
