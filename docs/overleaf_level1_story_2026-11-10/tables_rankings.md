# What matters most: top five per view

## noise σe

| view | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| Measured effect on a change (%/SD, all settings together) | delayInt −13.1 | V lows +5.2 | VSUB +2.4 | RG_H +2.2 | SW_L −1.9 |
| Correlation: setting change vs label change | delayInt −0.68 | V lows +0.42 | SW_L −0.29 | VSUB +0.20 | TG −0.14 |
| Correlation: setting vs label (levels) | delayInt −0.59 | V lows +0.38 | SW_L −0.36 | H lows −0.16 | RG_L +0.10 |
| Settings GP (absolute): relevance (largest = 1) | delayInt 1.00 | V lows 0.19 | TG 0.14 | H lows 0.13 | delayRGH 0.08 |
| Settings change GP: relevance (largest = 1) | delayInt 1.00 | VDD 0.52 | TG 0.25 | RG_L 0.23 | RG_H 0.22 |
| Settings change GP: predicted effect (%/SD) | delayInt −12.4 | delay_H −2.6 | RG_L +1.6 | H lows −1.6 | VDD +1.3 |
| Trace combined trees: predicted effect (%/SD) | delayInt −9.4 | SW_L −4.0 | V lows +3.1 | VR −2.7 | RG_L −1.5 |

## dark charge λ

| view | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| Measured effect on a change (%/SD, all settings together) | delayInt +26.7 | VSUB +7.4 | delayRGH +5.9 | delay_H +3.6 | VDD +2.1 |
| Correlation: setting change vs label change | delayInt +0.89 | RG_L −0.27 | delay_H +0.19 | TG +0.18 | H lows +0.13 |
| Correlation: setting vs label (levels) | delayInt +0.63 | RG_L −0.20 | H lows +0.17 | VSUB +0.16 | delay_H +0.07 |
| Settings GP (absolute): relevance (largest = 1) | delayInt 1.00 | VSUB 0.21 | delayRGH 0.20 | V lows 0.01 | H lows 0.01 |
| Settings change GP: relevance (largest = 1) | delayInt 1.00 | delayRGH 0.31 | VSUB 0.29 | delay_H 0.22 | VR 0.14 |
| Settings change GP: predicted effect (%/SD) | delayInt +26.9 | VSUB +6.8 | delay_H +3.7 | delayRGH +3.7 | VR −2.6 |
| Trace combined trees: predicted effect (%/SD) | delayInt +25.5 | delayRGH +4.3 | RG_L −2.4 | TG −2.1 | delay_H +2.1 |

## gain

| view | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| Measured effect on a change (%/SD, all settings together) | VR −0.7 | VDD −0.6 | delay_H +0.2 | delayInt −0.2 | delayRGH −0.2 |
| Correlation: setting change vs label change | VDD −0.61 | VR −0.55 | delayRGH −0.21 | delay_H +0.12 | delayInt −0.09 |
| Correlation: setting vs label (levels) | VDD −0.30 | VR −0.28 | delayRGH −0.08 | delay_H +0.07 | RG_H +0.04 |
| Settings GP (absolute): relevance (largest = 1) | not learned: the GP fitted this label as noise (all length scales at their start value) |  |  |  |  |
| Settings change GP: relevance (largest = 1) | VR 1.00 | VDD 0.68 | delay_H 0.37 | delayRGH 0.28 | delayInt 0.28 |
| Settings change GP: predicted effect (%/SD) | VR −0.6 | VDD −0.6 | delay_H +0.2 | delayInt −0.2 | delayRGH −0.1 |
| Trace combined trees: predicted effect (%/SD) | VR −0.5 | VDD −0.4 | RG_H +0.2 | delayRGH −0.1 | V lows +0.1 |

## The trace features the trace models use most

| model, label | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| trace trees (absolute), noise σe | signal-window settling (ADU) | line frequency, whole trace (k… | noise of 10-skip average | noise power 0.1-1 MHz | row-start recovery time (skips) |
| trace trees (absolute), dark charge λ | PC1 variance fraction | ref.: cycle variation in signal wind… | drift (ADU/ms) | in-window step rate (%) | waveform modes for 90 % of the… |
| trace trees (absolute), gain | ref.: cycle variation in signal wind… | dip per volt SW (ADU/V) | cycle variation in signal wind… | ref.: ringing rms (ADU) | dip recovery time (us) |
| trace combined trees (change), noise σe | row-start recovery time (skips) | signal-window settling (ADU) | 1/f corner frequency (kHz) | transient time constant (us) | CDS-slope lever arm (samples) |
| trace combined trees (change), dark charge λ | row-start recovery time (skips) | baseline droop time constant (… | noise power 3-7.5 MHz | ringing frequency (MHz) | recovery amplitude ratio |
| trace combined trees (change), gain | dip recovery time (us) | dip per volt SW (ADU/V) | Hadamard deviation n=50 (ADU) | baseline coupling (1e-3) | signal-window settling (ADU) |
