# Correlations, v3 voltage-only data (session A+B+C+D+E+F+G+H)

578 rows, 197 packets, amplifiers [np.int64(0), np.int64(1), np.int64(3)]; within amplifier; outputs in log units. Significant (95 %, n_eff = packets) when |r| >= 0.14. Partial = with the other voltages and the amplifier held fixed.

Knob groups that moved together: delay_Ha (+2) = delay_Ha+delay_Hb+delay_Hc; SW low = SW_1_L+SW_2_L; RG high = RG_1_H+RG_2_H; RG low = RG_1_L+RG_2_L; V1 high = V1_B_H+V1_A_H; V1 low = V1_B_L+V1_A_L; V3 high = V3_B_H+V3_A_H; V3 low = V3_B_L+V3_A_L; TG_B_H (+3) = TG_B_H+TG_B_L+TG_A_H+TG_A_L; H1 high = H1_B_H+H1_A_H; H1 low = H1_B_L+H1_A_L; H3 high = H3_B_H+H3_A_H; H3 low = H3_B_L+H3_A_L; VDD = VDD1+VDD2; VR = VR1+VR2

| output | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| σe (noise) | delayInt -0.58 | VSUB +0.24 | H2 low +0.14 (n.s.) | V2 low +0.13 (n.s.) | VDD +0.13 (n.s.) |
| λ (dark charge) | VSUB +0.68 | delayInt +0.54 | delayRGH +0.19 | H swing -0.09 (n.s.) | VDD +0.08 (n.s.) |
| gain | VDD -0.55 | VR -0.44 | H2 low -0.11 (n.s.) | V swing -0.11 (n.s.) | V3 low -0.10 (n.s.) |
| trace: white noise | delayInt -0.60 | VSUB +0.16 | delay_Ha (+2) +0.12 (n.s.) | H2 low +0.08 (n.s.) | V2 low +0.05 (n.s.) |
| trace: 1/f corner | delayInt -0.45 | delayRGH -0.34 | H swing +0.18 | V1 low -0.06 (n.s.) | RG low +0.06 (n.s.) |
| trace: Allan dev n=50 | H2 low +0.09 (n.s.) | V1 low -0.08 (n.s.) | H swing +0.07 (n.s.) | V3 low +0.05 (n.s.) | VSUB +0.05 (n.s.) |
| trace: power 0.1-1 MHz | delayInt +0.98 | V2 low +0.12 (n.s.) | V3 low +0.11 (n.s.) | H swing +0.10 (n.s.) | delayRGH +0.10 (n.s.) |
| trace: power 1-3 MHz | delayInt +0.99 | delay_Ha (+2) -0.21 | TG_B_H (+3) +0.13 (n.s.) | RG low -0.10 (n.s.) | H1 low +0.08 (n.s.) |
| trace: power 3-7.5 MHz | delayInt +0.89 | H1 low +0.17 | H swing +0.11 (n.s.) | RG low -0.10 (n.s.) | V3 low +0.09 (n.s.) |
| trace: slow/fast CDS power | delayInt +0.15 | H3 low -0.08 (n.s.) | V1 low -0.08 (n.s.) | VSUB -0.07 (n.s.) | H swing +0.06 (n.s.) |

Largest correlations between outputs (within amplifier):

- trace: power 0.1-1 MHz ~ trace: power 1-3 MHz: +0.97
- trace: power 1-3 MHz ~ trace: power 3-7.5 MHz: +0.88
- trace: power 0.1-1 MHz ~ trace: power 3-7.5 MHz: +0.88
- σe (noise) ~ trace: white noise: +0.65
- trace: white noise ~ trace: power 1-3 MHz: -0.63
- trace: Allan dev n=50 ~ trace: slow/fast CDS power: +0.61
- σe (noise) ~ trace: power 1-3 MHz: -0.58
- trace: white noise ~ trace: power 0.1-1 MHz: -0.58
- σe (noise) ~ trace: power 0.1-1 MHz: -0.48
- trace: white noise ~ trace: Allan dev n=50: +0.47

Design: largest |correlation| between two voltages over the packets 1.00, median 0.09.
