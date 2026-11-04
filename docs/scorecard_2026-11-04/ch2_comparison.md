# One extra amplifier: board 101 ch2

Runs: without `2026-11-04_scorecard`, with `2026-11-04_scorecard_with_ch2`. Direction = right on clear changes (|change| > sigma_e 5.0 %, lambda 15.0 %, gain 1.2 %); size = RMS error of the change (%).

## 1. The amplifier offset on ch2 as a new amplifier

Mean error (prediction - truth, % in log) on its rows (absolute) or pairs (change models, n -> n+1). An absolute model carries the amplifier's offset; a change model should not.

| model | sigma_e | lambda | gain |
|---|---|---|---|
| absolute settings: gp | -7.7 | -54.3 | +3.8 |
| absolute mean | -6.3 | -53.5 | +3.8 |
| absolute trace: physics | -5.0 | -99.7 | +3.2 |
| absolute trace: trees | -13.6 | -75.5 | +4.7 |
| absolute trace: pretrained cnn | -14.2 | -83.6 | +6.2 |
| change: absolute settings: gp, differenced | +0.2 | -0.0 | -0.0 |
| change: absolute trace: physics, differenced | +0.1 | -0.6 | -0.0 |
| change: absolute trace: pretrained cnn, differenced | +0.1 | -0.7 | -0.0 |
| change: absolute trace: trees, differenced | +0.2 | -0.6 | -0.0 |
| change: no change | +0.3 | +0.3 | +0.0 |
| change: settings change: gp | -0.6 | -2.3 | -0.1 |
| change: settings change: trees | -3.2 | -6.1 | -0.2 |
| change: trace change: hand-picked bands + gp | +1.8 | +0.3 | -0.1 |
| change: trace change: shared-encoder net | -0.7 | -1.2 | +0.1 |
| change: trace change: trees | +1.4 | -7.1 | -0.0 |

## 2. ch2 as a new amplifier (trained on the other three) against the other amplifiers

### within packet

clear changes going up: sigma_e ch2 100 % / others 96 %, lambda ch2 83 % / others 76 %, gain ch2 50 % / others 45 %

| model | sigma_e direction ch2 / others | sigma_e size ch2 / others | lambda direction ch2 / others | lambda size ch2 / others | gain direction ch2 / others | gain size ch2 / others |
|---|---|---|---|---|---|---|
| absolute settings: gp, differenced | nan / nan % | 0.0 / 0.0 | nan / nan % | 0.0 / 0.0 | nan / nan % | 0.0 / 0.0 |
| absolute trace: physics, differenced | nan / nan % | 0.0 / 0.0 | nan / nan % | 0.0 / 0.0 | nan / nan % | 0.0 / 0.0 |
| absolute trace: pretrained cnn, differenced | nan / nan % | 0.0 / 0.0 | nan / nan % | 0.0 / 0.0 | nan / nan % | 0.0 / 0.0 |
| absolute trace: trees, differenced | nan / nan % | 0.0 / 0.0 | nan / nan % | 0.0 / 0.0 | nan / nan % | 0.0 / 0.0 |
| no change | 50 / 50 % | 16.9 / 17.4 | 50 / 50 % | 32.1 / 34.2 | 50 / 50 % | 0.8 / 1.0 |
| settings change: gp | 42 / 27 % | 18.9 / 17.0 | 92 / 83 % | 36.2 / 38.9 | 62 / 77 % | 0.8 / 0.9 |
| settings change: trees | 26 / 35 % | 23.8 / 21.1 | 92 / 79 % | 38.2 / 41.8 | 100 / 95 % | 0.7 / 0.8 |
| trace change: hand-picked bands + gp | 63 / 78 % | 13.2 / 15.4 | 67 / 76 % | 32.1 / 38.3 | 88 / 91 % | 0.9 / 1.4 |
| trace change: shared-encoder net | 68 / 65 % | 20.7 / 22.9 | 75 / 55 % | 36.6 / 72.9 | 62 / 64 % | 2.3 / 2.3 |
| trace change: trees | 84 / 80 % | 12.6 / 13.8 | 92 / 93 % | 36.3 / 35.2 | 100 / 91 % | 0.7 / 0.9 |

### n -> n+1

clear changes going up: sigma_e ch2 47 % / others 51 %, lambda ch2 52 % / others 53 %, gain ch2 50 % / others 54 %

| model | sigma_e direction ch2 / others | sigma_e size ch2 / others | lambda direction ch2 / others | lambda size ch2 / others | gain direction ch2 / others | gain size ch2 / others |
|---|---|---|---|---|---|---|
| absolute settings: gp, differenced | 79 / 67 % | 10.1 / 10.9 | 83 / 80 % | 35.7 / 40.2 | 50 / 41 % | 1.3 / 1.5 |
| absolute trace: physics, differenced | 76 / 71 % | 12.5 / 12.6 | 71 / 83 % | 42.1 / 35.3 | 67 / 77 % | 1.3 / 1.4 |
| absolute trace: pretrained cnn, differenced | 82 / 87 % | 10.9 / 11.2 | 90 / 81 % | 36.5 / 41.1 | 67 / 64 % | 1.2 / 1.7 |
| absolute trace: trees, differenced | 71 / 85 % | 13.7 / 11.4 | 86 / 84 % | 39.4 / 37.7 | 83 / 74 % | 1.2 / 1.7 |
| no change | 50 / 50 % | 13.4 / 15.1 | 50 / 50 % | 44.7 / 46.7 | 50 / 50 % | 1.4 / 1.5 |
| settings change: gp | 41 / 60 % | 13.7 / 13.6 | 80 / 80 % | 51.2 / 52.1 | 73 / 75 % | 1.3 / 1.3 |
| settings change: trees | 35 / 70 % | 20.8 / 17.5 | 90 / 79 % | 52.2 / 54.1 | 91 / 92 % | 0.8 / 1.0 |
| trace change: hand-picked bands + gp | 65 / 68 % | 13.4 / 13.2 | 60 / 79 % | 44.7 / 48.7 | 91 / 83 % | 1.3 / 1.9 |
| trace change: shared-encoder net | 76 / 70 % | 23.6 / 28.6 | 80 / 67 % | 48.2 / 85.7 | 91 / 58 % | 2.9 / 3.0 |
| trace change: trees | 76 / 68 % | 13.0 / 13.0 | 90 / 89 % | 45.3 / 43.5 | 82 / 92 % | 0.9 / 1.2 |

### n -> n+2

clear changes going up: sigma_e ch2 42 % / others 52 %, lambda ch2 47 % / others 44 %, gain ch2 43 % / others 44 %

| model | sigma_e direction ch2 / others | sigma_e size ch2 / others | lambda direction ch2 / others | lambda size ch2 / others | gain direction ch2 / others | gain size ch2 / others |
|---|---|---|---|---|---|---|
| absolute settings: gp, differenced | 82 / 85 % | 9.8 / 10.9 | 82 / 74 % | 35.7 / 39.9 | 71 / 61 % | 1.1 / 1.3 |
| absolute trace: physics, differenced | 74 / 66 % | 11.5 / 13.8 | 76 / 92 % | 43.3 / 37.3 | 57 / 72 % | 1.1 / 1.2 |
| absolute trace: pretrained cnn, differenced | 89 / 83 % | 10.5 / 14.1 | 82 / 79 % | 39.2 / 41.1 | 71 / 69 % | 1.1 / 1.5 |
| absolute trace: trees, differenced | 68 / 78 % | 13.2 / 13.6 | 71 / 87 % | 40.9 / 38.9 | 86 / 75 % | 1.0 / 1.5 |
| no change | 50 / 50 % | 16.0 / 19.1 | 50 / 50 % | 38.2 / 42.2 | 50 / 50 % | 1.0 / 1.3 |
| settings change: gp | 44 / 57 % | 15.4 / 18.5 | 79 / 72 % | 48.1 / 52.6 | 80 / 75 % | 1.0 / 1.2 |
| settings change: trees | 44 / 65 % | 20.4 / 21.5 | 86 / 79 % | 49.5 / 53.6 | 80 / 79 % | 0.9 / 1.1 |
| trace change: hand-picked bands + gp | 62 / 61 % | 14.6 / 18.8 | 57 / 77 % | 38.2 / 43.0 | 100 / 89 % | 1.0 / 1.6 |
| trace change: shared-encoder net | 81 / 61 % | 20.1 / 31.3 | 71 / 62 % | 42.8 / 88.7 | 80 / 79 % | 2.8 / 2.7 |
| trace change: trees | 88 / 73 % | 10.3 / 16.3 | 79 / 89 % | 44.1 / 38.2 | 100 / 86 % | 0.8 / 1.1 |

### absolute, new amplifier: typical error (%)

| model | sigma_e ch2 / others | lambda ch2 / others | gain ch2 / others |
|---|---|---|---|
| settings: gp | 11.8 / 15.1 | 60.5 / 47.0 | 3.9 / 3.4 |
| mean | 15.5 / 19.2 | 61.7 / 50.3 | 3.9 / 3.4 |
| trace: physics | 13.0 / 12.6 | 104.9 / 79.6 | 3.4 / 4.3 |
| trace: trees | 17.9 / 17.3 | 81.3 / 59.8 | 4.8 / 4.1 |
| trace: pretrained cnn | 17.3 / 17.0 | 88.1 / 55.2 | 6.2 / 4.4 |

## 3. ch2 in training: the other amplifiers' scores without -> with it (same test pairs and rows)

Change (with - without) and its packet-bootstrap 95 % interval; direction in percentage points (+ = better), size and absolute error in % (- = better).

### trained amplifiers, within packet (105 pairs)

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| absolute settings + amplifier: gp, differenced | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] |
| absolute settings: gp, differenced | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] |
| absolute trace: physics, differenced | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] |
| absolute trace: pretrained cnn, differenced | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] |
| absolute trace: trees, differenced | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] |
| no change | +0 [+0, +0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | +0.0 [+0.0, +0.0] |
| settings + amplifier: gp | +0 [-21, +21] | -0.3 [-4.4, +4.0] | +17 [+0, +38] | -0.8 [-2.4, +0.0] | +0 [+0, +0] | +0.0 [-0.0, +0.0] |
| settings change: gp | -4 [-23, +15] | -0.0 [-2.6, +2.8] | +12 [+0, +31] | +0.2 [-0.1, +0.3] | +0 [+0, +0] | +0.0 [-0.0, +0.0] |
| settings change: trees | +0 [+0, +0] | -0.6 [-2.2, +1.1] | +0 [+0, +0] | -0.3 [-1.3, +0.3] | +0 [+0, +0] | -0.0 [-0.1, -0.0] |
| trace change: hand-picked bands + gp | +10 [-5, +30] | +0.5 [-0.9, +2.6] | +0 [-16, +14] | -0.7 [-8.0, +6.6] | +0 [-27, +27] | +0.0 [-0.1, +0.2] |
| trace change: shared-encoder net | +10 [-2, +22] | -2.8 [-6.9, -0.4] | -21 [-40, -6] | +5.5 [-2.3, +11.7] | -14 [-31, +0] | -0.2 [-0.7, +0.2] |
| trace change: trees | +2 [-15, +23] | +1.2 [-1.6, +4.5] | -7 [-15, +0] | +3.9 [-2.5, +13.2] | +5 [+0, +14] | +0.0 [-0.0, +0.1] |

### trained amplifiers, n -> n+1 (117 pairs)

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| absolute settings + amplifier: gp, differenced | +9 [-15, +38] | -0.6 [-2.1, +0.5] | +6 [+0, +19] | -1.3 [-5.9, +4.3] | +3 [+0, +17] | +0.1 [-0.1, +0.3] |
| absolute settings: gp, differenced | +5 [-29, +41] | +0.4 [-0.0, +1.2] | +5 [-22, +30] | +0.7 [-2.2, +2.5] | +0 [+0, +0] | -0.0 [-0.0, +0.0] |
| absolute trace: physics, differenced | +2 [-21, +29] | -0.1 [-1.9, +1.3] | -11 [-46, +10] | -2.1 [-9.4, +14.2] | +3 [-17, +26] | +0.0 [-0.3, +0.3] |
| absolute trace: pretrained cnn, differenced | -5 [-19, +0] | +1.7 [-0.0, +5.1] | -6 [-24, +6] | +1.6 [-0.4, +3.7] | +5 [-13, +29] | -0.0 [-0.1, +0.1] |
| absolute trace: trees, differenced | +2 [-7, +17] | +0.5 [-1.9, +3.0] | +2 [-9, +13] | -2.4 [-9.1, +3.8] | +0 [-19, +16] | -0.1 [-0.3, +0.1] |
| no change | +0 [+0, +0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | +0.0 [+0.0, +0.0] |
| settings + amplifier: gp | -10 [-38, +23] | +1.9 [-1.4, +6.7] | +2 [-21, +31] | -0.5 [-3.3, +0.3] | +11 [+0, +43] | -0.0 [-0.0, +0.0] |
| settings change: gp | -8 [-46, +25] | +0.4 [-1.5, +3.5] | -2 [-23, +24] | +0.1 [-0.3, +0.5] | +8 [+0, +40] | -0.0 [-0.0, +0.0] |
| settings change: trees | +0 [+0, +0] | +0.5 [-1.3, +2.8] | +0 [+0, +0] | -0.3 [-2.5, +1.2] | +0 [+0, +0] | -0.0 [-0.1, +0.0] |
| trace change: hand-picked bands + gp | -18 [-45, -2] | +1.8 [-0.7, +5.1] | +5 [-21, +31] | -4.9 [-21.2, +5.5] | -6 [-42, +33] | +0.1 [-0.2, +0.5] |
| trace change: shared-encoder net | +10 [-10, +34] | -1.9 [-12.6, +4.3] | -15 [-41, +0] | +6.7 [-6.9, +21.5] | -6 [-23, +0] | -0.3 [-1.2, +0.6] |
| trace change: trees | -4 [-32, +26] | +1.9 [-1.1, +5.4] | -3 [-25, +26] | +1.5 [-5.7, +9.9] | +0 [+0, +0] | +0.0 [-0.1, +0.2] |

### trained amplifiers, absolute (120 rows): typical error change (%)

| model | sigma_e | lambda | gain |
|---|---|---|---|
| settings: gp | +0.0 [-0.3, +0.3] | +2.0 [-1.2, +6.0] | +0.2 [+0.1, +0.3] |
| settings + amplifier: gp | +0.2 [-0.3, +0.8] | +0.4 [-2.0, +3.1] | +0.1 [-0.0, +0.2] |
| mean | +0.0 [-0.4, +0.5] | +1.8 [-1.5, +5.8] | +0.2 [+0.1, +0.3] |
| amplifier mean | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| trace: physics | -0.3 [-1.1, +0.3] | -1.4 [-4.9, +6.8] | +0.0 [-0.1, +0.1] |
| trace: trees | +0.9 [-0.2, +1.9] | -1.2 [-3.4, +1.8] | -0.1 [-0.2, -0.0] |
| trace: pretrained cnn | +1.2 [-0.0, +2.6] | +0.7 [-0.4, +1.7] | +0.1 [-0.0, +0.1] |

### new amplifier, within packet (105 pairs)

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| absolute settings: gp, differenced | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] |
| absolute trace: physics, differenced | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] |
| absolute trace: pretrained cnn, differenced | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] |
| absolute trace: trees, differenced | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] | +nan [+nan, +nan] | +0.0 [+0.0, +0.0] |
| no change | +0 [+0, +0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | +0.0 [+0.0, +0.0] |
| settings change: gp | -18 [-36, -4] | -1.3 [-3.2, +0.6] | +5 [+0, +11] | -0.2 [-1.7, +0.6] | +0 [-17, +20] | -0.0 [-0.1, -0.0] |
| settings change: trees | -2 [-8, +0] | -1.2 [-2.5, +0.1] | -7 [-20, +0] | -0.7 [-2.8, +0.8] | +0 [+0, +0] | -0.1 [-0.1, -0.0] |
| trace change: hand-picked bands + gp | +2 [-5, +12] | -0.0 [-2.4, +2.8] | -5 [-19, +8] | +4.9 [-1.8, +10.5] | +5 [-11, +19] | -0.0 [-0.1, +0.0] |
| trace change: shared-encoder net | +18 [+6, +31] | -1.9 [-6.4, +1.6] | -5 [-17, +7] | +21.8 [+13.0, +31.1] | +0 [-15, +14] | -0.2 [-0.5, +0.2] |
| trace change: trees | +14 [-2, +31] | +1.2 [-1.9, +4.3] | +0 [-7, +7] | +3.6 [-0.7, +10.4] | +0 [+0, +0] | -0.1 [-0.2, +0.0] |

### new amplifier, n -> n+1 (117 pairs)

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| absolute settings: gp, differenced | +3 [-4, +15] | +0.0 [-0.7, +0.9] | -5 [-23, +11] | +0.3 [-4.4, +4.3] | -13 [-38, +0] | +0.1 [-0.0, +0.2] |
| absolute trace: physics, differenced | +5 [-9, +23] | -0.9 [-2.8, +0.9] | -2 [-20, +13] | -6.0 [-12.9, -0.0] | +3 [-30, +29] | -0.0 [-0.2, +0.2] |
| absolute trace: pretrained cnn, differenced | +4 [-10, +18] | +0.8 [-0.8, +2.8] | +0 [-7, +7] | -0.7 [-3.8, +1.1] | -3 [-32, +22] | +0.0 [-0.1, +0.2] |
| absolute trace: trees, differenced | +7 [-16, +35] | +0.3 [-3.1, +3.3] | -6 [-18, +6] | -0.6 [-5.3, +4.6] | -5 [-30, +25] | +0.4 [-0.0, +0.7] |
| no change | +0 [+0, +0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | +0.0 [+0.0, +0.0] | +0 [+0, +0] | +0.0 [+0.0, +0.0] |
| settings change: gp | -12 [-38, +7] | -0.4 [-3.8, +3.4] | -2 [-14, +8] | +0.8 [-0.8, +3.6] | +6 [+0, +23] | -0.0 [-0.1, +0.0] |
| settings change: trees | -2 [-9, +0] | -1.2 [-3.3, +1.2] | -7 [-33, +0] | -1.8 [-8.2, +0.6] | -3 [-19, +0] | -0.1 [-0.1, +0.0] |
| trace change: hand-picked bands + gp | -6 [-40, +22] | +0.5 [-4.5, +5.4] | +0 [-12, +11] | +5.8 [-7.2, +16.6] | +14 [-7, +49] | -0.0 [-0.2, +0.2] |
| trace change: shared-encoder net | +8 [-14, +31] | +3.9 [-5.1, +11.2] | -7 [-21, +9] | +29.4 [+6.9, +54.8] | -3 [-27, +17] | -0.3 [-1.0, +0.4] |
| trace change: trees | -4 [-32, +26] | +0.1 [-3.4, +2.8] | +7 [-4, +22] | +0.7 [-7.6, +5.7] | +3 [+0, +13] | -0.0 [-0.1, +0.1] |

### new amplifier, absolute (120 rows): typical error change (%)

| model | sigma_e | lambda | gain |
|---|---|---|---|
| settings: gp | -0.6 [-1.1, -0.1] | -2.0 [-5.2, +1.7] | -0.1 [-0.2, +0.0] |
| mean | -0.8 [-1.2, -0.4] | +0.2 [-3.4, +4.2] | -0.1 [-0.2, +0.0] |
| trace: physics | -2.1 [-2.9, -1.4] | +24.1 [+16.5, +32.0] | +0.9 [+0.7, +1.0] |
| trace: trees | +1.0 [-0.2, +1.9] | +10.6 [+6.4, +14.3] | -0.4 [-0.5, -0.3] |
| trace: pretrained cnn | -1.5 [-2.8, -0.1] | +4.9 [+2.7, +7.4] | +1.0 [+0.9, +1.1] |
