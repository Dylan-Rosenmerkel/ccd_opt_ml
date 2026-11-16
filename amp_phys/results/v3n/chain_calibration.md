# Readout-chain calibration and source-follower inversion

CDS window tau = 10.0 us; load 20 k || 100 k = 16.67 k at 300 K; ADA4817 4 nV/rtHz.

## 1. Chain gain (ADU per uV at the video input)

Dead channel (open input): e_n = sqrt(4kTR + e_op^2) = 17.1 nV/rtHz -> sigma per skip 5.40 uV.

| channel | white noise per skip (ADU) | chain gain (ADU/uV) |
|---|---|---|
| 102 ch0 (dead) | 4.553 ± 0.005 | 0.842 ± 0.001 |
| 102 ch1 (dead) | 4.778 ± 0.006 | 0.884 ± 0.001 |
| 102 ch2 (dead) | 4.500 ± 0.005 | 0.833 ± 0.001 |

**Board 102 from the dead channels: 0.853 ± 0.027 ADU/uV** (spread between the three). **Design value: 0.875 ADU/uV** (27.34 V/V / 31.25 uV per LSB). Ratio measured/design = 0.975.

Board 101 has no dead channel; it uses the design value (same hardware), with the board-102 measured/design ratio as its uncertainty.

## 2-3. Per amplifier (needs only the chain gain)

Chain-gain uncertainty: 6 % (propagates 1:1 to the conversion gains).

|                                                    |   101 ch0 |   101 ch1 |   101 ch2 |   101 ch3 |   102 ch3 |
|:---------------------------------------------------|----------:|----------:|----------:|----------:|----------:|
| chain gain (ADU/uV)                                |     0.875 |     0.875 |     0.875 |     0.875 |     0.853 |
| conversion gain at AIN (uV/e)                      |     1.62  |     1.59  |     1.52  |     1.53  |     1.51  |
| white noise at AIN (nV/rtHz)                       |    20.2   |    20.6   |    21.4   |    22.2   |    18.8   |
| above the dead-channel floor (nV/rtHz)             |    10.8   |    11.5   |    12.8   |    14.2   |     7.74  |
| electronics share of the variance if unshunted (%) |    71.5   |    68.8   |    64.1   |    59.2   |    83     |
| rise time (ns)                                     |   436     |   367     |   353     |   363     |   594     |

## 4. Source follower: what the noise can and cannot say

With a 20 k load at 130 K, the follower's channel thermal noise seen at AIN is at most 4.9 nV/rtHz (at gm R = 1), whatever gm is. The measured 19-22 nV/rtHz is therefore mostly the frontend's load network and preamp (17.1 nV/rtHz open), unless the follower's output resistance shunts it, which needs a large gm. The noise alone cannot fix gm; the scan below shows each gm's consequences, and a second observable is needed (rise-time ratio, or a direct follower-gain measurement).

Rise-time check (board 102): the edge rise time scales with the output resistance (R_L || R_b || 1/gm) times the input capacitance. Dead channels (open): 549 ns; 102 ch3 (live, same board): 594 ns, ratio 1.08. No shortening means 1/gm is not much below 17 k, i.e. gm of order 10-30 uS or less, a follower gain A of 0.2-0.4, and then a sense-node conversion gain of 4-8 uV/e (C_sn 20-40 fF). Caveat: the dead channels' edges are pickup, whose own shape may differ.

| gm (uS) | A | output R (k) | live floor | channel noise | unexplained | CG node (uV/e) | C_sn (fF) | rise/dead |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.09 | 15.4 | 15.9 | 2.8 | 12.2 | 17.78 | 9 | 0.92 |
| 10 | 0.17 | 14.3 | 14.8 | 3.6 | 13.3 | 9.70 | 17 | 0.86 |
| 20 | 0.29 | 12.5 | 13.1 | 4.4 | 14.8 | 5.66 | 28 | 0.75 |
| 50 | 0.50 | 9.1 | 9.9 | 4.9 | 16.9 | 3.23 | 50 | 0.55 |
| 100 | 0.67 | 6.3 | 7.4 | 4.6 | 18.2 | 2.42 | 66 | 0.38 |
| 200 | 0.80 | 3.8 | 5.5 | 3.9 | 19.0 | 2.02 | 79 | 0.23 |
| 500 | 0.91 | 1.8 | 4.4 | 2.8 | 19.5 | 1.78 | 90 | 0.11 |

(scan shown for 101 ch0; all amplifiers in sf_gm_scan.csv)
