# Level-1 scorecard, v2

Board 101 ch0/1/3; 232 rows from 80 packets ({'G': 232}); 5 contiguous time blocks of session D; fractional-window rows left out. Trace models never see the settings. Pairs: {'within packet (image)': 190, 'within packet (strip)': 158, 'n -> n+1': 229, 'n -> n+2': 226, 'random pairs': 8857}.

Image-label change models train on the within-packet (image), n -> n+1, n -> n+2 and random pairs (random capped per split); strip-label models on the strip pairs; both orders. A test pair never shares a time block with a training pair.

Label noise of a change: sigma_e 1.9 %, lambda 12.0 %, gain 0.8 %; clear = sigma_e > 5.0 %, lambda > 15.0 %, gain > 1.2 %.

## Absolute, trained amplifiers: typical error % (vs settings-only `settings + amplifier: gp`)

| model | sigma_e | lambda | gain |
|---|---|---|---|
| settings: gp | 13.0 (worse) | 42.9 (worse) | 2.5 (worse) |
| settings: trees | 12.7 (worse) | 44.0 (worse) | 2.5 (worse) |
| settings + amplifier: gp | 7.1 (reference) | 11.1 (reference) | 0.8 (reference) |
| settings + amplifier: trees | 6.4 (same) | 11.9 (same) | 1.0 (worse) |
| mean | 13.8 (worse) | 54.9 (worse) | 3.1 (worse) |
| amplifier mean | 8.7 (worse) | 38.4 (worse) | 2.2 (worse) |
| trace: physics | 6.3 (better) | 24.0 (worse) | 1.1 (worse) |
| trace: gp | 10.2 (worse) | 38.7 (worse) | 2.0 (worse) |
| trace: trees | 6.5 (same) | 20.6 (worse) | 1.6 (worse) |
| trace: nn | 7.1 (same) | 23.7 (worse) | 2.2 (worse) |
| trace: cnn | 7.4 (same) | 26.6 (worse) | 2.3 (worse) |
| trace: pretrained cnn | 7.6 (same) | 31.6 (worse) | 2.3 (worse) |
| trace: deep kernel | 10.1 (worse) | 38.3 (worse) | 2.8 (worse) |

## Absolute, new amplifier: typical error % (vs settings-only `settings: gp`)

| model | sigma_e | lambda | gain |
|---|---|---|---|
| settings: gp | 17.7 (reference) | 67.3 (reference) | 3.7 (reference) |
| settings: trees | 17.6 (same) | 63.7 (better) | 3.7 (same) |
| mean | 18.2 (same) | 70.4 (worse) | 4.0 (worse) |
| trace: physics | 14.0 (better) | 51.5 (better) | 3.9 (worse) |
| trace: gp | 17.9 (same) | 66.3 (same) | 4.9 (worse) |
| trace: trees | 11.9 (better) | 39.9 (better) | 4.8 (worse) |
| trace: nn | 15.3 (better) | 57.2 (better) | 5.9 (worse) |
| trace: cnn | 20.4 (worse) | 59.9 (better) | 5.9 (worse) |
| trace: pretrained cnn | 18.7 (worse) | 63.2 (better) | 4.8 (worse) |
| trace: deep kernel | 23.1 (worse) | 84.0 (worse) | 5.0 (worse) |

## Relative, trained amplifiers, within packet (image)

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 73 clear of 190, true change RMS 11.9 %; lambda: 117 clear of 190, true change RMS 49.4 %; gain: 112 clear of 190, true change RMS 2.6 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 100 % | 11.9 | 3.3 | - | 50 % (worse) | 86 % | 49.4 | 20.8 | - | 50 % (worse) | 35 % | 2.6 | 1.4 | - |
| settings change: gp | 100 % (same) | 100 % | 8.5 | 3.5 | 0.50 | 99 % (same) | 86 % | 23.5 | 13.6 | 0.89 | 97 % (same) | 35 % | 1.5 | 0.7 | 0.85 |
| settings change: trees | 100 % (same) | 100 % | 8.1 | 3.3 | 0.58 | 99 % (same) | 86 % | 22.3 | 13.6 | 0.88 | 96 % (same) | 35 % | 1.4 | 0.7 | 0.88 |
| settings + amplifier: gp | 99 % (reference) | 100 % | 7.8 | 3.3 | 0.65 | 98 % (reference) | 86 % | 13.0 | 7.2 | 0.93 | 99 % (reference) | 35 % | 1.3 | 0.7 | 0.87 |
| settings + amplifier: trees | 100 % (same) | 100 % | 7.0 | 2.8 | 0.69 | 99 % (same) | 86 % | 12.9 | 6.4 | 0.94 | 98 % (same) | 35 % | 1.1 | 0.6 | 0.91 |
| trace change: trees | 93 % (same) | 100 % | 8.0 | 3.7 | 0.49 | 84 % (worse) | 86 % | 25.5 | 11.7 | 0.69 | 96 % (same) | 35 % | 1.6 | 0.8 | 0.84 |
| trace: combined trees [A, B, B-A] | 95 % (same) | 100 % | 7.9 | 2.8 | 0.50 | 91 % (worse) | 86 % | 22.0 | 9.3 | 0.75 | 93 % (worse) | 35 % | 1.6 | 0.8 | 0.83 |
| trace: direction classifier [A, B, B-A] | 99 % (same) | 100 % | - | - | 0.42 | 89 % (worse) | 86 % | - | - | 0.69 | 96 % (same) | 35 % | - | - | 0.84 |
| trace change: shared-encoder net | 88 % (worse) | 100 % | 9.8 | 4.5 | 0.40 | 79 % (worse) | 86 % | 35.5 | 19.9 | 0.57 | 83 % (worse) | 35 % | 2.0 | 1.1 | 0.69 |
| trace change: difference curves PCA + trees | 95 % (same) | 100 % | 9.0 | 4.6 | 0.46 | 88 % (worse) | 86 % | 32.1 | 15.5 | 0.60 | 77 % (worse) | 35 % | 2.3 | 1.3 | 0.50 |
| trace change: hand-picked bands + gp | 93 % (same) | 100 % | 8.2 | 3.3 | 0.49 | 80 % (worse) | 86 % | 27.5 | 12.5 | 0.64 | 83 % (worse) | 35 % | 2.3 | 1.2 | 0.57 |
| trace: physics difference (sigma_e) | 85 % (worse) | 100 % | 10.6 | 4.1 | 0.25 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 97 % (same) | 100 % | 7.8 | 3.7 | 0.50 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 96 % (same) | 100 % | 7.0 | 2.3 | 0.66 | 99 % (same) | 86 % | 11.2 | 4.8 | 0.96 | 99 % (same) | 35 % | 1.0 | 0.6 | 0.93 |
| settings + trace: residual trees | 97 % (same) | 100 % | 7.7 | 3.0 | 0.63 | 99 % (same) | 86 % | 13.0 | 7.2 | 0.93 | 96 % (same) | 35 % | 1.4 | 0.7 | 0.86 |
| settings + trace: direction classifier | 100 % (same) | 100 % | - | - | 0.58 | 99 % (same) | 86 % | - | - | 0.89 | 98 % (same) | 35 % | - | - | 0.90 |
| ensemble: trace models | 97 % (same) | 100 % | 7.5 | 3.2 | 0.52 | 86 % (worse) | 86 % | 22.0 | 9.8 | 0.75 | 96 % (same) | 35 % | 1.5 | 0.8 | 0.85 |
| ensemble: settings + trace | 100 % (same) | 100 % | 7.1 | 2.8 | 0.63 | 99 % (same) | 86 % | 11.1 | 5.4 | 0.95 | 96 % (same) | 35 % | 1.1 | 0.6 | 0.91 |

## Relative, trained amplifiers, within packet (strip)

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 94 clear of 158, true change RMS 12.2 %; lambda: 123 clear of 158, true change RMS 69.1 %; gain: 111 clear of 158, true change RMS 7.0 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 96 % | 12.2 | 6.0 | - | 50 % (worse) | 80 % | 69.1 | 34.1 | - | 50 % (worse) | 50 % | 7.0 | 2.1 | - |
| settings change: gp | 96 % (better) | 96 % | 7.9 | 3.9 | 0.60 | 96 % (same) | 80 % | 30.3 | 18.9 | 0.88 | 71 % (same) | 50 % | 7.1 | 1.9 | 0.30 |
| settings change: trees | 96 % (better) | 96 % | 8.2 | 3.8 | 0.56 | 95 % (same) | 80 % | 33.0 | 18.2 | 0.87 | 59 % (worse) | 50 % | 8.1 | 2.8 | 0.16 |
| settings + amplifier: gp | 83 % (reference) | 96 % | 10.0 | 5.1 | 0.30 | 97 % (reference) | 80 % | 20.8 | 9.5 | 0.93 | 72 % (reference) | 50 % | 7.0 | 2.1 | 0.37 |
| settings + amplifier: trees | 95 % (better) | 96 % | 8.4 | 3.8 | 0.55 | 95 % (same) | 80 % | 23.7 | 11.2 | 0.92 | 65 % (same) | 50 % | 8.3 | 2.8 | 0.23 |
| trace change: trees | 94 % (same) | 96 % | 7.3 | 3.8 | 0.60 | 76 % (worse) | 80 % | 47.1 | 25.5 | 0.63 | 61 % (same) | 50 % | 7.7 | 3.0 | 0.25 |
| trace: combined trees [A, B, B-A] | 94 % (same) | 96 % | 6.9 | 3.4 | 0.66 | 81 % (worse) | 80 % | 44.9 | 18.9 | 0.64 | 64 % (same) | 50 % | 7.5 | 2.6 | 0.24 |
| trace: direction classifier [A, B, B-A] | 96 % (better) | 96 % | - | - | 0.53 | 85 % (worse) | 80 % | - | - | 0.67 | 76 % (same) | 50 % | - | - | 0.49 |
| trace change: shared-encoder net | 88 % (same) | 96 % | 10.1 | 6.5 | 0.36 | 72 % (worse) | 80 % | 56.7 | 32.8 | 0.47 | 53 % (worse) | 50 % | 10.6 | 6.0 | 0.07 |
| trace change: difference curves PCA + trees | 71 % (same) | 96 % | 11.3 | 7.2 | 0.47 | 71 % (worse) | 80 % | 61.5 | 41.2 | 0.59 | 55 % (worse) | 50 % | 7.7 | 3.1 | 0.13 |
| trace change: hand-picked bands + gp | 78 % (same) | 96 % | 10.4 | 6.7 | 0.51 | 74 % (worse) | 80 % | 52.9 | 34.6 | 0.62 | 68 % (same) | 50 % | 7.7 | 2.4 | 0.27 |
| settings + trace: combined trees | 95 % (better) | 96 % | 6.9 | 3.1 | 0.64 | 96 % (same) | 80 % | 23.2 | 12.6 | 0.92 | 60 % (same) | 50 % | 7.5 | 2.7 | 0.23 |
| settings + trace: residual trees | 93 % (same) | 96 % | 8.0 | 4.3 | 0.53 | 91 % (worse) | 80 % | 31.3 | 17.4 | 0.86 | 59 % (same) | 50 % | 8.5 | 3.3 | 0.16 |
| settings + trace: direction classifier | 96 % (better) | 96 % | - | - | 0.53 | 95 % (same) | 80 % | - | - | 0.79 | 76 % (same) | 50 % | - | - | 0.49 |
| ensemble: trace models | 95 % (same) | 96 % | 6.9 | 3.5 | 0.66 | 78 % (worse) | 80 % | 44.7 | 21.3 | 0.64 | 62 % (same) | 50 % | 7.4 | 2.5 | 0.27 |
| ensemble: settings + trace | 96 % (better) | 96 % | 7.2 | 3.5 | 0.62 | 95 % (same) | 80 % | 23.9 | 11.6 | 0.92 | 59 % (worse) | 50 % | 7.7 | 2.7 | 0.19 |

## Relative, trained amplifiers, n -> n+1

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 91 clear of 217, true change RMS 12.1 %; lambda: 130 clear of 217, true change RMS 54.6 %; gain: 151 clear of 217, true change RMS 3.1 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 49 % | 12.1 | 3.3 | - | 50 % (worse) | 55 % | 54.6 | 22.5 | - | 50 % (worse) | 50 % | 3.1 | 1.9 | - |
| settings change: gp | 85 % (same) | 49 % | 10.1 | 4.6 | 0.57 | 100 % (same) | 55 % | 26.2 | 14.3 | 0.91 | 93 % (same) | 50 % | 2.0 | 0.9 | 0.81 |
| settings change: trees | 88 % (same) | 49 % | 9.4 | 3.8 | 0.69 | 100 % (same) | 55 % | 27.2 | 15.2 | 0.90 | 91 % (same) | 50 % | 2.0 | 0.9 | 0.79 |
| settings + amplifier: gp | 89 % (reference) | 49 % | 9.3 | 4.2 | 0.67 | 99 % (reference) | 55 % | 22.7 | 9.8 | 0.91 | 93 % (reference) | 50 % | 1.8 | 0.9 | 0.85 |
| settings + amplifier: trees | 89 % (same) | 49 % | 9.1 | 3.9 | 0.68 | 100 % (same) | 55 % | 17.7 | 8.0 | 0.93 | 92 % (same) | 50 % | 1.7 | 0.8 | 0.85 |
| trace change: trees | 86 % (same) | 49 % | 9.8 | 3.5 | 0.65 | 86 % (worse) | 55 % | 34.8 | 14.8 | 0.75 | 94 % (same) | 50 % | 2.0 | 0.9 | 0.84 |
| trace: combined trees [A, B, B-A] | 82 % (same) | 49 % | 9.4 | 3.4 | 0.64 | 90 % (same) | 55 % | 27.7 | 11.8 | 0.82 | 93 % (same) | 50 % | 1.9 | 0.8 | 0.86 |
| trace: direction classifier [A, B, B-A] | 85 % (same) | 49 % | - | - | 0.65 | 94 % (same) | 55 % | - | - | 0.80 | 94 % (same) | 50 % | - | - | 0.84 |
| trace change: shared-encoder net | 76 % (same) | 49 % | 12.5 | 6.1 | 0.48 | 85 % (worse) | 55 % | 46.0 | 24.3 | 0.63 | 84 % (same) | 50 % | 2.5 | 1.3 | 0.70 |
| trace change: difference curves PCA + trees | 80 % (same) | 49 % | 9.7 | 4.7 | 0.60 | 82 % (worse) | 55 % | 36.1 | 18.6 | 0.69 | 79 % (worse) | 50 % | 2.7 | 1.5 | 0.54 |
| trace change: hand-picked bands + gp | 84 % (same) | 49 % | 9.6 | 4.0 | 0.58 | 78 % (worse) | 55 % | 36.2 | 14.6 | 0.67 | 75 % (worse) | 50 % | 2.8 | 1.5 | 0.49 |
| trace: physics difference (sigma_e) | 77 % (same) | 49 % | 11.2 | 4.1 | 0.40 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 89 % (same) | 49 % | 9.9 | 3.5 | 0.66 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 92 % (same) | 49 % | 8.5 | 2.6 | 0.76 | 100 % (same) | 55 % | 14.4 | 6.2 | 0.96 | 97 % (same) | 50 % | 1.3 | 0.7 | 0.93 |
| settings + trace: residual trees | 90 % (same) | 49 % | 8.9 | 3.4 | 0.70 | 100 % (same) | 55 % | 17.5 | 8.9 | 0.94 | 89 % (same) | 50 % | 1.9 | 0.9 | 0.80 |
| settings + trace: direction classifier | 92 % (same) | 49 % | - | - | 0.77 | 100 % (same) | 55 % | - | - | 0.92 | 97 % (same) | 50 % | - | - | 0.89 |
| absolute settings: gp, differenced | 87 % (same) | 49 % | 9.9 | 3.0 | 0.65 | 100 % (same) | 55 % | 25.3 | 12.5 | 0.93 | 93 % (same) | 50 % | 1.7 | 0.8 | 0.88 |
| absolute settings: trees, differenced | 88 % (same) | 49 % | 9.5 | 3.4 | 0.69 | 95 % (same) | 55 % | 28.2 | 15.8 | 0.88 | 94 % (same) | 50 % | 1.8 | 1.1 | 0.82 |
| absolute settings + amplifier: gp, differenced | 90 % (same) | 49 % | 9.3 | 3.0 | 0.74 | 100 % (same) | 55 % | 14.5 | 5.4 | 0.96 | 96 % (same) | 50 % | 1.0 | 0.5 | 0.95 |
| absolute settings + amplifier: trees, differenced | 90 % (same) | 49 % | 8.7 | 3.8 | 0.75 | 98 % (same) | 55 % | 16.4 | 7.3 | 0.93 | 93 % (same) | 50 % | 1.4 | 0.7 | 0.89 |
| absolute trace: physics, differenced | 93 % (same) | 49 % | 8.5 | 3.5 | 0.69 | 92 % (same) | 55 % | 35.3 | 13.9 | 0.77 | 96 % (same) | 50 % | 1.5 | 0.7 | 0.90 |
| absolute trace: gp, differenced | 60 % (worse) | 49 % | 13.9 | 5.6 | 0.20 | 65 % (worse) | 55 % | 53.2 | 25.3 | 0.37 | 81 % (same) | 50 % | 2.8 | 1.3 | 0.61 |
| absolute trace: trees, differenced | 87 % (same) | 49 % | 8.7 | 3.3 | 0.65 | 86 % (worse) | 55 % | 27.1 | 13.3 | 0.75 | 85 % (same) | 50 % | 2.3 | 1.3 | 0.69 |
| absolute trace: nn, differenced | 76 % (same) | 49 % | 10.1 | 3.8 | 0.51 | 82 % (worse) | 55 % | 36.2 | 15.1 | 0.67 | 70 % (worse) | 50 % | 2.9 | 1.7 | 0.40 |
| absolute trace: cnn, differenced | 79 % (same) | 49 % | 10.0 | 4.4 | 0.50 | 86 % (worse) | 55 % | 38.2 | 18.5 | 0.67 | 56 % (worse) | 50 % | 3.2 | 1.7 | 0.12 |
| absolute trace: pretrained cnn, differenced | 75 % (same) | 49 % | 10.7 | 4.2 | 0.42 | 70 % (worse) | 55 % | 45.4 | 22.4 | 0.50 | 52 % (worse) | 50 % | 3.2 | 1.9 | 0.09 |
| absolute trace: deep kernel, differenced | 71 % (same) | 49 % | 13.3 | 7.0 | 0.31 | 68 % (worse) | 55 % | 55.4 | 28.0 | 0.34 | 48 % (worse) | 50 % | 3.7 | 2.4 | -0.03 |
| ensemble: trace models | 88 % (same) | 49 % | 8.9 | 3.1 | 0.70 | 93 % (same) | 55 % | 29.6 | 11.9 | 0.84 | 96 % (same) | 50 % | 1.7 | 0.7 | 0.90 |
| ensemble: settings + trace | 92 % (same) | 49 % | 8.3 | 2.8 | 0.77 | 100 % (same) | 55 % | 18.9 | 7.0 | 0.95 | 95 % (same) | 50 % | 1.4 | 0.6 | 0.92 |

## Relative, trained amplifiers, n -> n+2

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 97 clear of 202, true change RMS 12.6 %; lambda: 138 clear of 202, true change RMS 54.3 %; gain: 130 clear of 202, true change RMS 3.1 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 51 % | 12.6 | 4.6 | - | 50 % (worse) | 54 % | 54.3 | 30.0 | - | 50 % (worse) | 55 % | 3.1 | 1.9 | - |
| settings change: gp | 95 % (same) | 51 % | 10.2 | 4.7 | 0.70 | 100 % (same) | 54 % | 28.1 | 15.9 | 0.92 | 92 % (same) | 55 % | 1.9 | 1.0 | 0.80 |
| settings change: trees | 94 % (same) | 51 % | 9.6 | 5.3 | 0.74 | 100 % (same) | 54 % | 27.2 | 18.6 | 0.91 | 94 % (same) | 55 % | 1.8 | 0.9 | 0.80 |
| settings + amplifier: gp | 92 % (reference) | 51 % | 9.8 | 4.8 | 0.72 | 98 % (reference) | 54 % | 19.4 | 9.0 | 0.94 | 95 % (reference) | 55 % | 1.6 | 0.9 | 0.85 |
| settings + amplifier: trees | 96 % (same) | 51 % | 9.0 | 3.9 | 0.77 | 100 % (same) | 54 % | 17.2 | 8.4 | 0.96 | 94 % (same) | 55 % | 1.6 | 0.8 | 0.84 |
| trace change: trees | 95 % (same) | 51 % | 9.2 | 3.7 | 0.76 | 88 % (same) | 54 % | 30.0 | 13.2 | 0.79 | 91 % (same) | 55 % | 1.9 | 1.1 | 0.81 |
| trace: combined trees [A, B, B-A] | 96 % (same) | 51 % | 9.2 | 3.7 | 0.76 | 88 % (same) | 54 % | 25.8 | 11.9 | 0.83 | 90 % (same) | 55 % | 1.9 | 0.9 | 0.83 |
| trace: direction classifier [A, B, B-A] | 96 % (same) | 51 % | - | - | 0.76 | 90 % (same) | 54 % | - | - | 0.83 | 92 % (same) | 55 % | - | - | 0.83 |
| trace change: shared-encoder net | 85 % (same) | 51 % | 12.2 | 5.8 | 0.59 | 80 % (worse) | 54 % | 43.3 | 20.1 | 0.62 | 81 % (worse) | 55 % | 2.4 | 1.4 | 0.69 |
| trace change: difference curves PCA + trees | 91 % (same) | 51 % | 9.5 | 5.0 | 0.64 | 80 % (worse) | 54 % | 36.6 | 19.4 | 0.65 | 71 % (worse) | 55 % | 2.9 | 1.8 | 0.45 |
| trace change: hand-picked bands + gp | 89 % (same) | 51 % | 9.1 | 4.9 | 0.67 | 77 % (worse) | 54 % | 31.5 | 17.9 | 0.69 | 81 % (same) | 55 % | 2.8 | 1.5 | 0.56 |
| trace: physics difference (sigma_e) | 81 % (same) | 51 % | 11.7 | 4.8 | 0.47 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 93 % (same) | 51 % | 9.1 | 3.9 | 0.75 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 99 % (same) | 51 % | 7.9 | 3.0 | 0.84 | 100 % (same) | 54 % | 13.7 | 5.8 | 0.97 | 96 % (same) | 55 % | 1.4 | 0.7 | 0.91 |
| settings + trace: residual trees | 92 % (same) | 51 % | 9.0 | 4.1 | 0.78 | 97 % (same) | 54 % | 17.0 | 10.3 | 0.95 | 92 % (same) | 55 % | 1.7 | 0.9 | 0.82 |
| settings + trace: direction classifier | 98 % (same) | 51 % | - | - | 0.83 | 100 % (same) | 54 % | - | - | 0.93 | 95 % (same) | 55 % | - | - | 0.88 |
| absolute settings: gp, differenced | 89 % (same) | 51 % | 10.9 | 4.0 | 0.61 | 100 % (same) | 54 % | 25.1 | 14.7 | 0.93 | 95 % (same) | 55 % | 1.6 | 0.9 | 0.87 |
| absolute settings: trees, differenced | 95 % (same) | 51 % | 9.7 | 3.8 | 0.75 | 96 % (same) | 54 % | 28.9 | 16.1 | 0.88 | 89 % (same) | 55 % | 1.6 | 0.8 | 0.86 |
| absolute settings + amplifier: gp, differenced | 93 % (same) | 51 % | 9.6 | 3.2 | 0.75 | 100 % (same) | 54 % | 14.4 | 5.6 | 0.98 | 98 % (same) | 55 % | 1.0 | 0.5 | 0.94 |
| absolute settings + amplifier: trees, differenced | 97 % (same) | 51 % | 8.6 | 3.2 | 0.80 | 98 % (same) | 54 % | 16.6 | 7.5 | 0.95 | 97 % (same) | 55 % | 1.3 | 0.8 | 0.89 |
| absolute trace: physics, differenced | 94 % (same) | 51 % | 8.5 | 4.0 | 0.78 | 91 % (same) | 54 % | 34.6 | 16.1 | 0.76 | 95 % (same) | 55 % | 1.5 | 0.8 | 0.88 |
| absolute trace: gp, differenced | 66 % (worse) | 51 % | 14.5 | 6.8 | 0.27 | 71 % (worse) | 54 % | 52.6 | 24.7 | 0.45 | 78 % (worse) | 55 % | 2.8 | 1.3 | 0.57 |
| absolute trace: trees, differenced | 87 % (same) | 51 % | 8.7 | 4.0 | 0.70 | 82 % (worse) | 54 % | 27.9 | 13.9 | 0.76 | 86 % (same) | 55 % | 2.3 | 1.3 | 0.71 |
| absolute trace: nn, differenced | 82 % (same) | 51 % | 9.7 | 4.0 | 0.64 | 83 % (worse) | 54 % | 32.7 | 17.7 | 0.73 | 67 % (worse) | 55 % | 3.1 | 1.7 | 0.30 |
| absolute trace: cnn, differenced | 85 % (same) | 51 % | 9.8 | 5.2 | 0.57 | 80 % (worse) | 54 % | 35.9 | 19.1 | 0.70 | 55 % (worse) | 55 % | 3.4 | 1.9 | 0.08 |
| absolute trace: pretrained cnn, differenced | 76 % (same) | 51 % | 10.6 | 4.9 | 0.53 | 74 % (worse) | 54 % | 43.1 | 26.2 | 0.54 | 47 % (worse) | 55 % | 3.3 | 2.0 | 0.01 |
| absolute trace: deep kernel, differenced | 69 % (worse) | 51 % | 14.0 | 7.5 | 0.28 | 69 % (worse) | 54 % | 51.2 | 30.9 | 0.42 | 43 % (worse) | 55 % | 4.0 | 2.5 | -0.12 |
| ensemble: trace models | 96 % (same) | 51 % | 8.5 | 3.3 | 0.79 | 91 % (same) | 54 % | 27.0 | 11.7 | 0.84 | 92 % (same) | 55 % | 1.7 | 0.8 | 0.87 |
| ensemble: settings + trace | 98 % (same) | 51 % | 8.0 | 3.2 | 0.84 | 99 % (same) | 54 % | 18.2 | 8.6 | 0.95 | 95 % (same) | 55 % | 1.3 | 0.7 | 0.91 |

## Relative, trained amplifiers, random pairs

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 749 clear of 1682, true change RMS 11.9 %; lambda: 1075 clear of 1682, true change RMS 53.8 %; gain: 1084 clear of 1682, true change RMS 3.2 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 50 % | 11.9 | 4.0 | - | 50 % (worse) | 49 % | 53.8 | 26.0 | - | 50 % (worse) | 47 % | 3.2 | 1.9 | - |
| settings change: gp | 90 % (same) | 50 % | 10.3 | 5.1 | 0.62 | 99 % (same) | 49 % | 28.3 | 15.9 | 0.91 | 93 % (same) | 47 % | 1.9 | 1.0 | 0.83 |
| settings change: trees | 89 % (same) | 50 % | 9.7 | 4.9 | 0.67 | 99 % (same) | 49 % | 28.1 | 15.9 | 0.91 | 93 % (same) | 47 % | 2.0 | 0.9 | 0.81 |
| settings + amplifier: gp | 89 % (reference) | 50 % | 9.5 | 4.3 | 0.67 | 98 % (reference) | 49 % | 20.6 | 9.4 | 0.93 | 95 % (reference) | 47 % | 1.8 | 0.9 | 0.85 |
| settings + amplifier: trees | 91 % (same) | 50 % | 9.1 | 3.9 | 0.70 | 99 % (better) | 49 % | 18.5 | 8.2 | 0.94 | 94 % (same) | 47 % | 1.7 | 0.8 | 0.85 |
| trace change: trees | 88 % (same) | 50 % | 9.2 | 3.8 | 0.68 | 87 % (worse) | 49 % | 32.8 | 14.3 | 0.77 | 90 % (worse) | 47 % | 2.3 | 1.0 | 0.75 |
| trace: combined trees [A, B, B-A] | 87 % (same) | 50 % | 9.1 | 3.4 | 0.69 | 91 % (worse) | 49 % | 26.4 | 11.7 | 0.83 | 89 % (worse) | 47 % | 2.2 | 1.0 | 0.76 |
| trace: direction classifier [A, B, B-A] | 88 % (same) | 50 % | - | - | 0.68 | 93 % (worse) | 49 % | - | - | 0.82 | 90 % (worse) | 47 % | - | - | 0.76 |
| trace change: shared-encoder net | 79 % (worse) | 50 % | 11.9 | 5.8 | 0.53 | 83 % (worse) | 49 % | 43.1 | 21.5 | 0.66 | 83 % (worse) | 47 % | 2.6 | 1.4 | 0.67 |
| trace change: difference curves PCA + trees | 86 % (same) | 50 % | 9.4 | 4.6 | 0.62 | 84 % (worse) | 49 % | 35.7 | 17.8 | 0.71 | 75 % (worse) | 47 % | 2.8 | 1.6 | 0.48 |
| trace change: hand-picked bands + gp | 82 % (same) | 50 % | 9.4 | 4.4 | 0.60 | 83 % (worse) | 49 % | 33.4 | 15.4 | 0.73 | 76 % (worse) | 47 % | 2.8 | 1.5 | 0.50 |
| trace: physics difference (sigma_e) | 74 % (worse) | 50 % | 11.0 | 4.8 | 0.38 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 87 % (same) | 50 % | 9.3 | 3.9 | 0.66 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 93 % (same) | 50 % | 8.0 | 2.8 | 0.78 | 99 % (better) | 49 % | 14.0 | 5.8 | 0.96 | 97 % (same) | 47 % | 1.4 | 0.7 | 0.91 |
| settings + trace: residual trees | 90 % (same) | 50 % | 9.1 | 3.9 | 0.71 | 98 % (same) | 49 % | 19.0 | 9.5 | 0.94 | 92 % (same) | 47 % | 1.9 | 0.9 | 0.82 |
| settings + trace: direction classifier | 93 % (same) | 50 % | - | - | 0.77 | 99 % (same) | 49 % | - | - | 0.93 | 97 % (same) | 47 % | - | - | 0.88 |
| absolute settings: gp, differenced | 85 % (same) | 50 % | 10.2 | 3.9 | 0.59 | 99 % (better) | 49 % | 25.2 | 13.3 | 0.93 | 96 % (same) | 47 % | 1.7 | 0.8 | 0.87 |
| absolute settings: trees, differenced | 87 % (same) | 50 % | 9.7 | 4.4 | 0.65 | 96 % (same) | 49 % | 28.7 | 15.7 | 0.89 | 94 % (same) | 47 % | 1.8 | 0.9 | 0.84 |
| absolute settings + amplifier: gp, differenced | 88 % (same) | 50 % | 9.3 | 3.0 | 0.72 | 99 % (same) | 49 % | 14.4 | 5.7 | 0.96 | 98 % (better) | 47 % | 1.0 | 0.5 | 0.95 |
| absolute settings + amplifier: trees, differenced | 88 % (same) | 50 % | 8.7 | 3.4 | 0.71 | 98 % (same) | 49 % | 16.4 | 7.0 | 0.94 | 96 % (same) | 47 % | 1.4 | 0.8 | 0.90 |
| absolute trace: physics, differenced | 88 % (same) | 50 % | 8.6 | 3.7 | 0.68 | 89 % (worse) | 49 % | 32.9 | 14.1 | 0.76 | 93 % (same) | 47 % | 1.7 | 0.8 | 0.85 |
| absolute trace: gp, differenced | 63 % (worse) | 50 % | 13.9 | 6.2 | 0.20 | 71 % (worse) | 49 % | 54.0 | 24.4 | 0.42 | 80 % (worse) | 47 % | 2.8 | 1.4 | 0.58 |
| absolute trace: trees, differenced | 85 % (same) | 50 % | 9.0 | 3.7 | 0.63 | 85 % (worse) | 49 % | 28.0 | 13.8 | 0.76 | 85 % (worse) | 47 % | 2.4 | 1.4 | 0.68 |
| absolute trace: nn, differenced | 84 % (same) | 50 % | 9.5 | 3.9 | 0.61 | 85 % (worse) | 49 % | 33.6 | 16.1 | 0.74 | 72 % (worse) | 47 % | 2.9 | 1.7 | 0.43 |
| absolute trace: cnn, differenced | 81 % (same) | 50 % | 9.8 | 4.2 | 0.54 | 84 % (worse) | 49 % | 36.4 | 18.1 | 0.70 | 57 % (worse) | 47 % | 3.2 | 2.0 | 0.14 |
| absolute trace: pretrained cnn, differenced | 74 % (worse) | 50 % | 10.4 | 4.3 | 0.44 | 74 % (worse) | 49 % | 43.6 | 23.5 | 0.53 | 50 % (worse) | 47 % | 3.3 | 2.0 | 0.07 |
| absolute trace: deep kernel, differenced | 65 % (worse) | 50 % | 13.6 | 6.7 | 0.25 | 65 % (worse) | 49 % | 54.6 | 28.8 | 0.33 | 49 % (worse) | 47 % | 3.8 | 2.5 | -0.01 |
| ensemble: trace models | 89 % (same) | 50 % | 8.5 | 3.4 | 0.72 | 91 % (worse) | 49 % | 27.5 | 11.5 | 0.84 | 92 % (same) | 47 % | 1.9 | 0.8 | 0.82 |
| ensemble: settings + trace | 92 % (same) | 50 % | 8.0 | 2.9 | 0.78 | 99 % (same) | 49 % | 18.1 | 7.3 | 0.95 | 96 % (same) | 47 % | 1.5 | 0.7 | 0.90 |

## Relative, new amplifier, within packet (image)

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 73 clear of 190, true change RMS 11.9 %; lambda: 117 clear of 190, true change RMS 49.4 %; gain: 112 clear of 190, true change RMS 2.6 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 100 % | 11.9 | 3.3 | - | 50 % (worse) | 86 % | 49.4 | 20.8 | - | 50 % (worse) | 35 % | 2.6 | 1.4 | - |
| settings change: gp | 95 % (reference) | 100 % | 10.1 | 4.6 | 0.32 | 97 % (reference) | 86 % | 31.0 | 18.0 | 0.82 | 97 % (reference) | 35 % | 1.8 | 0.8 | 0.81 |
| settings change: trees | 99 % (same) | 100 % | 9.2 | 3.9 | 0.41 | 99 % (same) | 86 % | 30.7 | 16.5 | 0.82 | 95 % (same) | 35 % | 1.8 | 0.7 | 0.82 |
| trace change: trees | 96 % (same) | 100 % | 8.8 | 4.5 | 0.38 | 81 % (worse) | 86 % | 30.4 | 17.4 | 0.64 | 93 % (same) | 35 % | 1.8 | 1.0 | 0.77 |
| trace: combined trees [A, B, B-A] | 97 % (same) | 100 % | 10.0 | 4.3 | 0.46 | 78 % (worse) | 86 % | 31.6 | 14.0 | 0.56 | 88 % (worse) | 35 % | 2.0 | 1.2 | 0.68 |
| trace: direction classifier [A, B, B-A] | 86 % (same) | 100 % | - | - | 0.33 | 77 % (worse) | 86 % | - | - | 0.55 | 84 % (worse) | 35 % | - | - | 0.70 |
| trace change: shared-encoder net | 74 % (worse) | 100 % | 12.9 | 7.8 | 0.28 | 69 % (worse) | 86 % | 52.3 | 35.3 | 0.37 | 78 % (worse) | 35 % | 2.6 | 1.3 | 0.59 |
| trace change: difference curves PCA + trees | 97 % (same) | 100 % | 8.4 | 4.3 | 0.43 | 85 % (worse) | 86 % | 32.0 | 17.8 | 0.60 | 75 % (worse) | 35 % | 2.4 | 1.2 | 0.45 |
| trace change: hand-picked bands + gp | 82 % (worse) | 100 % | 8.5 | 3.8 | 0.42 | 77 % (worse) | 86 % | 31.0 | 16.1 | 0.54 | 71 % (worse) | 35 % | 2.6 | 1.6 | 0.44 |
| trace: physics difference (sigma_e) | 89 % (same) | 100 % | 10.5 | 3.8 | 0.22 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 95 % (same) | 100 % | 8.8 | 4.8 | 0.36 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 100 % (better) | 100 % | 8.8 | 3.8 | 0.55 | 99 % (same) | 86 % | 22.5 | 7.8 | 0.90 | 97 % (same) | 35 % | 1.8 | 0.7 | 0.81 |
| settings + trace: residual trees | 96 % (same) | 100 % | 9.4 | 4.6 | 0.42 | 97 % (same) | 86 % | 27.3 | 13.0 | 0.85 | 89 % (worse) | 35 % | 1.9 | 0.8 | 0.76 |
| settings + trace: direction classifier | 100 % (better) | 100 % | - | - | 0.48 | 99 % (same) | 86 % | - | - | 0.84 | 96 % (same) | 35 % | - | - | 0.79 |
| ensemble: trace models | 96 % (same) | 100 % | 8.4 | 4.2 | 0.45 | 84 % (worse) | 86 % | 28.4 | 11.7 | 0.67 | 91 % (worse) | 35 % | 1.8 | 1.0 | 0.76 |
| ensemble: settings + trace | 99 % (same) | 100 % | 8.1 | 4.0 | 0.48 | 98 % (same) | 86 % | 24.2 | 9.8 | 0.88 | 96 % (same) | 35 % | 1.8 | 0.7 | 0.80 |

## Relative, new amplifier, within packet (strip)

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 94 clear of 158, true change RMS 12.2 %; lambda: 123 clear of 158, true change RMS 69.1 %; gain: 111 clear of 158, true change RMS 7.0 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 96 % | 12.2 | 6.0 | - | 50 % (worse) | 80 % | 69.1 | 34.1 | - | 50 % (worse) | 50 % | 7.0 | 2.1 | - |
| settings change: gp | 95 % (reference) | 96 % | 8.3 | 4.7 | 0.52 | 93 % (reference) | 80 % | 41.8 | 25.6 | 0.84 | 63 % (reference) | 50 % | 7.0 | 2.0 | 0.22 |
| settings change: trees | 95 % (same) | 96 % | 8.0 | 3.9 | 0.50 | 95 % (same) | 80 % | 40.5 | 25.7 | 0.82 | 62 % (same) | 50 % | 8.7 | 2.7 | 0.11 |
| trace change: trees | 96 % (same) | 96 % | 7.5 | 3.9 | 0.59 | 76 % (worse) | 80 % | 49.8 | 29.6 | 0.61 | 55 % (same) | 50 % | 8.1 | 3.4 | 0.09 |
| trace: combined trees [A, B, B-A] | 93 % (same) | 96 % | 7.3 | 3.6 | 0.61 | 80 % (worse) | 80 % | 45.0 | 22.4 | 0.65 | 51 % (same) | 50 % | 8.6 | 3.6 | 0.00 |
| trace: direction classifier [A, B, B-A] | 96 % (same) | 96 % | - | - | 0.48 | 81 % (worse) | 80 % | - | - | 0.61 | 67 % (same) | 50 % | - | - | 0.26 |
| trace change: shared-encoder net | 85 % (worse) | 96 % | 11.3 | 7.3 | 0.35 | 65 % (worse) | 80 % | 65.1 | 44.7 | 0.37 | 59 % (same) | 50 % | 10.3 | 5.3 | 0.25 |
| trace change: difference curves PCA + trees | 57 % (worse) | 96 % | 13.3 | 8.6 | 0.38 | 62 % (worse) | 80 % | 68.3 | 53.6 | 0.55 | 58 % (same) | 50 % | 8.0 | 3.2 | 0.14 |
| trace change: hand-picked bands + gp | 48 % (worse) | 96 % | 13.8 | 9.5 | 0.27 | 65 % (worse) | 80 % | 61.7 | 40.3 | 0.55 | 59 % (same) | 50 % | 8.3 | 3.0 | 0.07 |
| settings + trace: combined trees | 95 % (same) | 96 % | 7.2 | 3.9 | 0.59 | 96 % (same) | 80 % | 31.3 | 20.3 | 0.87 | 54 % (same) | 50 % | 8.5 | 3.5 | 0.04 |
| settings + trace: residual trees | 93 % (same) | 96 % | 8.5 | 4.5 | 0.50 | 89 % (same) | 80 % | 41.2 | 25.0 | 0.79 | 57 % (same) | 50 % | 10.0 | 4.7 | 0.12 |
| settings + trace: direction classifier | 97 % (same) | 96 % | - | - | 0.44 | 96 % (same) | 80 % | - | - | 0.72 | 72 % (same) | 50 % | - | - | 0.32 |
| ensemble: trace models | 95 % (same) | 96 % | 7.0 | 3.8 | 0.64 | 80 % (worse) | 80 % | 45.6 | 22.2 | 0.65 | 56 % (same) | 50 % | 8.1 | 3.3 | 0.06 |
| ensemble: settings + trace | 91 % (same) | 96 % | 7.3 | 3.4 | 0.57 | 94 % (same) | 80 % | 34.0 | 20.5 | 0.85 | 58 % (same) | 50 % | 8.8 | 3.6 | 0.10 |

## Relative, new amplifier, n -> n+1

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 91 clear of 217, true change RMS 12.1 %; lambda: 130 clear of 217, true change RMS 54.6 %; gain: 151 clear of 217, true change RMS 3.1 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 49 % | 12.1 | 3.3 | - | 50 % (worse) | 55 % | 54.6 | 22.5 | - | 50 % (worse) | 50 % | 3.1 | 1.9 | - |
| settings change: gp | 88 % (reference) | 49 % | 11.0 | 4.9 | 0.59 | 98 % (reference) | 55 % | 37.5 | 19.3 | 0.86 | 89 % (reference) | 50 % | 2.4 | 1.0 | 0.77 |
| settings change: trees | 86 % (same) | 49 % | 10.7 | 4.7 | 0.59 | 98 % (same) | 55 % | 35.2 | 17.4 | 0.87 | 90 % (same) | 50 % | 2.3 | 0.9 | 0.79 |
| trace change: trees | 87 % (same) | 49 % | 10.4 | 4.2 | 0.66 | 85 % (worse) | 55 % | 37.4 | 18.1 | 0.73 | 89 % (same) | 50 % | 2.3 | 1.0 | 0.78 |
| trace: combined trees [A, B, B-A] | 82 % (same) | 49 % | 11.1 | 5.5 | 0.59 | 78 % (worse) | 55 % | 38.2 | 16.4 | 0.63 | 87 % (same) | 50 % | 2.5 | 1.1 | 0.70 |
| trace: direction classifier [A, B, B-A] | 76 % (same) | 49 % | - | - | 0.55 | 83 % (worse) | 55 % | - | - | 0.70 | 89 % (same) | 50 % | - | - | 0.74 |
| trace change: shared-encoder net | 65 % (worse) | 49 % | 13.5 | 7.8 | 0.31 | 71 % (worse) | 55 % | 52.2 | 36.2 | 0.45 | 77 % (same) | 50 % | 2.8 | 1.5 | 0.59 |
| trace change: difference curves PCA + trees | 82 % (same) | 49 % | 9.9 | 4.7 | 0.56 | 82 % (worse) | 55 % | 38.3 | 19.1 | 0.66 | 70 % (worse) | 50 % | 2.9 | 1.6 | 0.44 |
| trace change: hand-picked bands + gp | 82 % (same) | 49 % | 10.8 | 4.1 | 0.54 | 74 % (worse) | 55 % | 37.1 | 18.4 | 0.57 | 70 % (same) | 50 % | 2.9 | 1.7 | 0.42 |
| trace: physics difference (sigma_e) | 78 % (same) | 49 % | 11.2 | 4.0 | 0.39 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 84 % (same) | 49 % | 10.3 | 4.3 | 0.63 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 90 % (same) | 49 % | 10.9 | 4.1 | 0.68 | 100 % (same) | 55 % | 24.9 | 10.0 | 0.93 | 93 % (same) | 50 % | 2.3 | 0.9 | 0.80 |
| settings + trace: residual trees | 80 % (same) | 49 % | 11.1 | 5.1 | 0.60 | 98 % (same) | 55 % | 30.9 | 14.9 | 0.89 | 89 % (same) | 50 % | 2.4 | 1.0 | 0.76 |
| settings + trace: direction classifier | 91 % (same) | 49 % | - | - | 0.73 | 100 % (same) | 55 % | - | - | 0.89 | 90 % (same) | 50 % | - | - | 0.79 |
| absolute settings: gp, differenced | 80 % (same) | 49 % | 12.3 | 4.1 | 0.53 | 89 % (same) | 55 % | 48.2 | 22.0 | 0.70 | 83 % (same) | 50 % | 2.6 | 1.3 | 0.63 |
| absolute settings: trees, differenced | 87 % (same) | 49 % | 10.5 | 3.9 | 0.69 | 95 % (same) | 55 % | 36.2 | 20.1 | 0.84 | 88 % (same) | 50 % | 2.4 | 1.1 | 0.75 |
| absolute trace: physics, differenced | 70 % (same) | 49 % | 10.9 | 4.8 | 0.34 | 70 % (worse) | 55 % | 42.4 | 19.7 | 0.50 | 83 % (same) | 50 % | 3.1 | 1.8 | 0.60 |
| absolute trace: gp, differenced | 53 % (worse) | 49 % | 12.2 | 4.2 | 0.01 | 45 % (worse) | 55 % | 58.1 | 24.4 | -0.16 | 49 % (worse) | 50 % | 3.2 | 1.8 | -0.02 |
| absolute trace: trees, differenced | 78 % (same) | 49 % | 10.1 | 4.1 | 0.52 | 69 % (worse) | 55 % | 39.9 | 21.5 | 0.53 | 60 % (worse) | 50 % | 3.3 | 2.1 | 0.14 |
| absolute trace: nn, differenced | 76 % (same) | 49 % | 10.2 | 4.1 | 0.48 | 73 % (worse) | 55 % | 41.9 | 23.0 | 0.57 | 74 % (same) | 50 % | 2.7 | 1.5 | 0.51 |
| absolute trace: cnn, differenced | 76 % (same) | 49 % | 10.6 | 5.0 | 0.47 | 83 % (worse) | 55 % | 41.7 | 23.2 | 0.67 | 60 % (worse) | 50 % | 3.2 | 1.8 | 0.14 |
| absolute trace: pretrained cnn, differenced | 64 % (worse) | 49 % | 12.2 | 5.3 | 0.20 | 63 % (worse) | 55 % | 54.2 | 25.4 | 0.28 | 51 % (worse) | 50 % | 3.2 | 2.0 | 0.04 |
| absolute trace: deep kernel, differenced | 68 % (same) | 49 % | 12.3 | 4.4 | 0.20 | 63 % (worse) | 55 % | 53.4 | 31.6 | 0.24 | 53 % (worse) | 50 % | 3.2 | 1.9 | 0.03 |
| ensemble: trace models | 87 % (same) | 49 % | 9.8 | 3.8 | 0.66 | 83 % (worse) | 55 % | 35.5 | 13.5 | 0.71 | 89 % (same) | 50 % | 2.4 | 1.1 | 0.77 |
| ensemble: settings + trace | 89 % (same) | 49 % | 9.6 | 3.6 | 0.69 | 94 % (same) | 55 % | 25.4 | 13.5 | 0.89 | 91 % (same) | 50 % | 2.3 | 1.0 | 0.80 |

## Relative, new amplifier, n -> n+2

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 97 clear of 202, true change RMS 12.6 %; lambda: 138 clear of 202, true change RMS 54.3 %; gain: 130 clear of 202, true change RMS 3.1 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 51 % | 12.6 | 4.6 | - | 50 % (worse) | 54 % | 54.3 | 30.0 | - | 50 % (worse) | 55 % | 3.1 | 1.9 | - |
| settings change: gp | 91 % (reference) | 51 % | 11.2 | 5.6 | 0.60 | 99 % (reference) | 54 % | 36.7 | 22.1 | 0.86 | 91 % (reference) | 55 % | 2.3 | 1.1 | 0.74 |
| settings change: trees | 95 % (same) | 51 % | 10.8 | 5.5 | 0.68 | 100 % (same) | 54 % | 35.9 | 21.4 | 0.87 | 89 % (same) | 55 % | 2.3 | 1.1 | 0.74 |
| trace change: trees | 92 % (same) | 51 % | 10.7 | 5.0 | 0.69 | 85 % (worse) | 54 % | 33.8 | 17.3 | 0.77 | 88 % (same) | 55 % | 2.2 | 1.1 | 0.75 |
| trace: combined trees [A, B, B-A] | 90 % (same) | 51 % | 10.7 | 5.5 | 0.65 | 83 % (worse) | 54 % | 33.8 | 17.1 | 0.72 | 85 % (same) | 55 % | 2.4 | 1.2 | 0.68 |
| trace: direction classifier [A, B, B-A] | 82 % (same) | 51 % | - | - | 0.63 | 85 % (worse) | 54 % | - | - | 0.77 | 86 % (same) | 55 % | - | - | 0.69 |
| trace change: shared-encoder net | 71 % (worse) | 51 % | 14.5 | 8.1 | 0.39 | 68 % (worse) | 54 % | 52.2 | 33.2 | 0.48 | 77 % (same) | 55 % | 2.9 | 1.7 | 0.53 |
| trace change: difference curves PCA + trees | 87 % (same) | 51 % | 9.6 | 5.2 | 0.62 | 78 % (worse) | 54 % | 37.9 | 22.6 | 0.66 | 75 % (same) | 55 % | 2.9 | 1.7 | 0.48 |
| trace change: hand-picked bands + gp | 85 % (same) | 51 % | 10.0 | 5.0 | 0.61 | 75 % (worse) | 54 % | 35.1 | 21.5 | 0.59 | 78 % (same) | 55 % | 2.8 | 1.5 | 0.49 |
| trace: physics difference (sigma_e) | 81 % (same) | 51 % | 11.7 | 4.8 | 0.47 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 93 % (same) | 51 % | 10.7 | 4.5 | 0.70 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 91 % (same) | 51 % | 11.0 | 4.2 | 0.69 | 100 % (same) | 54 % | 24.0 | 9.9 | 0.94 | 91 % (same) | 55 % | 2.3 | 0.9 | 0.77 |
| settings + trace: residual trees | 94 % (same) | 51 % | 10.8 | 5.8 | 0.68 | 99 % (same) | 54 % | 31.0 | 15.1 | 0.90 | 86 % (same) | 55 % | 2.4 | 1.1 | 0.72 |
| settings + trace: direction classifier | 96 % (same) | 51 % | - | - | 0.77 | 100 % (same) | 54 % | - | - | 0.90 | 88 % (same) | 55 % | - | - | 0.79 |
| absolute settings: gp, differenced | 85 % (same) | 51 % | 12.6 | 4.9 | 0.59 | 88 % (worse) | 54 % | 48.0 | 29.4 | 0.71 | 88 % (same) | 55 % | 2.6 | 1.1 | 0.65 |
| absolute settings: trees, differenced | 89 % (same) | 51 % | 10.9 | 4.2 | 0.69 | 96 % (same) | 54 % | 37.3 | 23.0 | 0.85 | 88 % (same) | 55 % | 2.3 | 1.0 | 0.75 |
| absolute trace: physics, differenced | 74 % (same) | 51 % | 11.2 | 6.3 | 0.40 | 72 % (worse) | 54 % | 43.4 | 26.1 | 0.52 | 82 % (same) | 55 % | 3.1 | 1.8 | 0.56 |
| absolute trace: gp, differenced | 49 % (worse) | 51 % | 13.3 | 5.2 | -0.13 | 51 % (worse) | 54 % | 57.8 | 29.7 | -0.09 | 42 % (worse) | 55 % | 3.2 | 1.9 | -0.07 |
| absolute trace: trees, differenced | 86 % (same) | 51 % | 9.7 | 4.8 | 0.60 | 72 % (worse) | 54 % | 36.7 | 20.4 | 0.62 | 50 % (worse) | 55 % | 3.4 | 2.1 | 0.01 |
| absolute trace: nn, differenced | 84 % (same) | 51 % | 9.7 | 5.5 | 0.60 | 80 % (worse) | 54 % | 38.1 | 21.6 | 0.64 | 64 % (worse) | 55 % | 3.0 | 2.0 | 0.31 |
| absolute trace: cnn, differenced | 79 % (same) | 51 % | 10.5 | 4.9 | 0.57 | 79 % (worse) | 54 % | 41.5 | 26.5 | 0.65 | 54 % (worse) | 55 % | 3.4 | 2.0 | 0.08 |
| absolute trace: pretrained cnn, differenced | 70 % (worse) | 51 % | 12.3 | 6.4 | 0.34 | 70 % (worse) | 54 % | 51.5 | 31.0 | 0.39 | 47 % (worse) | 55 % | 3.3 | 2.0 | -0.00 |
| absolute trace: deep kernel, differenced | 60 % (worse) | 51 % | 12.8 | 5.2 | 0.10 | 64 % (worse) | 54 % | 54.8 | 33.0 | 0.23 | 57 % (worse) | 55 % | 3.3 | 2.1 | 0.06 |
| ensemble: trace models | 93 % (same) | 51 % | 9.8 | 4.7 | 0.69 | 84 % (worse) | 54 % | 32.3 | 16.4 | 0.75 | 88 % (same) | 55 % | 2.4 | 1.2 | 0.74 |
| ensemble: settings + trace | 94 % (same) | 51 % | 9.6 | 4.3 | 0.72 | 96 % (same) | 54 % | 24.6 | 15.1 | 0.91 | 91 % (same) | 55 % | 2.3 | 1.1 | 0.77 |

## Relative, new amplifier, random pairs

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 749 clear of 1682, true change RMS 11.9 %; lambda: 1075 clear of 1682, true change RMS 53.8 %; gain: 1084 clear of 1682, true change RMS 3.2 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 50 % | 11.9 | 4.0 | - | 50 % (worse) | 49 % | 53.8 | 26.0 | - | 50 % (worse) | 47 % | 3.2 | 1.9 | - |
| settings change: gp | 84 % (reference) | 50 % | 11.4 | 6.0 | 0.54 | 98 % (reference) | 49 % | 38.0 | 20.8 | 0.86 | 91 % (reference) | 47 % | 2.3 | 1.1 | 0.78 |
| settings change: trees | 88 % (better) | 50 % | 10.9 | 5.5 | 0.60 | 99 % (same) | 49 % | 36.4 | 20.1 | 0.87 | 90 % (same) | 47 % | 2.3 | 1.1 | 0.77 |
| trace change: trees | 85 % (same) | 50 % | 10.3 | 4.7 | 0.63 | 85 % (worse) | 49 % | 36.8 | 17.7 | 0.73 | 86 % (worse) | 47 % | 2.4 | 1.2 | 0.69 |
| trace: combined trees [A, B, B-A] | 83 % (same) | 50 % | 10.6 | 5.1 | 0.59 | 80 % (worse) | 49 % | 36.3 | 17.4 | 0.66 | 83 % (worse) | 47 % | 2.6 | 1.4 | 0.61 |
| trace: direction classifier [A, B, B-A] | 79 % (same) | 50 % | - | - | 0.56 | 85 % (worse) | 49 % | - | - | 0.73 | 84 % (worse) | 47 % | - | - | 0.64 |
| trace change: shared-encoder net | 70 % (worse) | 50 % | 14.2 | 8.0 | 0.36 | 72 % (worse) | 49 % | 52.6 | 32.1 | 0.49 | 76 % (worse) | 47 % | 3.0 | 1.7 | 0.53 |
| trace change: difference curves PCA + trees | 85 % (same) | 50 % | 9.7 | 4.8 | 0.59 | 83 % (worse) | 49 % | 38.1 | 19.3 | 0.70 | 73 % (worse) | 47 % | 2.9 | 1.6 | 0.45 |
| trace change: hand-picked bands + gp | 80 % (same) | 50 % | 10.0 | 4.7 | 0.55 | 77 % (worse) | 49 % | 37.3 | 18.8 | 0.63 | 71 % (worse) | 47 % | 3.0 | 1.7 | 0.41 |
| trace: physics difference (sigma_e) | 75 % (worse) | 50 % | 11.0 | 4.7 | 0.39 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 84 % (same) | 50 % | 10.3 | 4.9 | 0.63 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 86 % (same) | 50 % | 10.5 | 4.5 | 0.62 | 99 % (better) | 49 % | 25.3 | 9.8 | 0.93 | 92 % (same) | 47 % | 2.3 | 1.0 | 0.77 |
| settings + trace: residual trees | 86 % (same) | 50 % | 11.0 | 5.5 | 0.62 | 98 % (same) | 49 % | 32.6 | 16.2 | 0.89 | 88 % (worse) | 47 % | 2.4 | 1.2 | 0.74 |
| settings + trace: direction classifier | 88 % (same) | 50 % | - | - | 0.70 | 99 % (same) | 49 % | - | - | 0.89 | 89 % (same) | 47 % | - | - | 0.77 |
| absolute settings: gp, differenced | 80 % (same) | 50 % | 11.8 | 4.6 | 0.53 | 90 % (worse) | 49 % | 47.5 | 26.9 | 0.72 | 86 % (worse) | 47 % | 2.6 | 1.2 | 0.64 |
| absolute settings: trees, differenced | 84 % (same) | 50 % | 10.6 | 4.7 | 0.62 | 96 % (worse) | 49 % | 36.5 | 21.5 | 0.85 | 89 % (same) | 47 % | 2.3 | 1.0 | 0.75 |
| absolute trace: physics, differenced | 75 % (worse) | 50 % | 10.5 | 5.5 | 0.39 | 71 % (worse) | 49 % | 42.6 | 22.3 | 0.53 | 81 % (worse) | 47 % | 3.1 | 1.8 | 0.58 |
| absolute trace: gp, differenced | 47 % (worse) | 50 % | 12.4 | 4.6 | -0.09 | 53 % (worse) | 49 % | 56.5 | 26.5 | -0.02 | 47 % (worse) | 47 % | 3.2 | 1.9 | -0.05 |
| absolute trace: trees, differenced | 81 % (same) | 50 % | 9.8 | 4.3 | 0.55 | 73 % (worse) | 49 % | 38.5 | 20.3 | 0.60 | 54 % (worse) | 47 % | 3.3 | 2.1 | 0.07 |
| absolute trace: nn, differenced | 83 % (same) | 50 % | 9.9 | 4.9 | 0.56 | 79 % (worse) | 49 % | 40.5 | 22.5 | 0.63 | 70 % (worse) | 47 % | 2.8 | 1.8 | 0.44 |
| absolute trace: cnn, differenced | 80 % (same) | 50 % | 10.6 | 4.9 | 0.49 | 82 % (worse) | 49 % | 41.8 | 24.8 | 0.65 | 60 % (worse) | 47 % | 3.2 | 1.9 | 0.19 |
| absolute trace: pretrained cnn, differenced | 64 % (worse) | 50 % | 11.9 | 6.0 | 0.26 | 69 % (worse) | 49 % | 52.1 | 27.8 | 0.37 | 52 % (worse) | 47 % | 3.2 | 2.0 | 0.05 |
| absolute trace: deep kernel, differenced | 62 % (worse) | 50 % | 12.1 | 5.2 | 0.16 | 59 % (worse) | 49 % | 54.5 | 31.7 | 0.20 | 53 % (worse) | 47 % | 3.2 | 2.0 | 0.03 |
| ensemble: trace models | 86 % (same) | 50 % | 9.4 | 4.4 | 0.65 | 83 % (worse) | 49 % | 34.3 | 15.0 | 0.74 | 86 % (worse) | 47 % | 2.5 | 1.3 | 0.68 |
| ensemble: settings + trace | 89 % (same) | 50 % | 9.3 | 4.5 | 0.67 | 96 % (same) | 49 % | 25.3 | 14.6 | 0.91 | 91 % (same) | 47 % | 2.3 | 1.1 | 0.78 |

## (5) Ranking: shortlist the settings from the traces, joint objective J(w) = w dlog sigma_e + (1-w) dlog lambda

Each setting (non-nominal packet) applies to the whole board, so its objective is the mean over the 3 amplifiers (predicted and true; truth = the setting's effect relative to nominal, drift removed). Weight w:(1-w) of sigma_e:lambda. Shortlist rules (lvl1/shortlist.py): weighted = by J(w); pareto = the predicted Pareto front first. Cells: top-5 regret (%) = how much worse the best of the 5 shortlisted settings truly is than the truly best setting / whether the truly best is on the 10-setting shortlist (1 = yes). The 3 amplifiers share their settings, so each column is ONE trial: read the pattern, not single numbers.

### trained amplifiers (60 settings)

| model | 1:0 weighted | 0.75:0.25 weighted | 0.5:0.5 weighted | 0.25:0.75 weighted | 0:1 weighted | 1:0 pareto | 0.75:0.25 pareto | 0.5:0.5 pareto | 0.25:0.75 pareto | 0:1 pareto |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 2.1 / 0.18 | 8.3 / 0.17 | 15.8 / 0.15 | 23.4 / 0.16 | 31.1 / 0.17 | 2.1 / 0.16 | 8.5 / 0.17 | 15.6 / 0.17 | 23.8 / 0.17 | 30.7 / 0.16 |
| no change | 1.8 / 0 | 8.5 / 0 | 16.2 / 0 | 22.4 / 0 | 32.5 / 0 | 2.1 / 0 | 9.0 / 0 | 17.5 / 0 | 20.6 / 0 | 32.8 / 0 |
| settings change: gp | 1.1 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.1 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings change: trees | 1.1 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.1 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings + amplifier: gp | 1.7 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.7 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings + amplifier: trees | 1.0 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.0 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| trace change: trees | 3.0 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 2.2 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| trace: combined trees [A, B, B-A] | 0.0 / 1 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 2.2 / 1 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| trace: direction classifier [A, B, B-A] | 1.3 / 0 | 3.1 / 1 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 1.3 / 0 | 3.1 / 1 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| trace change: shared-encoder net | 3.0 / 0 | 6.8 / 0 | 11.1 / 0 | 15.4 / 0 | 19.7 / 0 | 3.0 / 0 | 6.8 / 0 | 11.1 / 0 | 15.4 / 0 | 19.7 / 0 |
| trace change: difference curves PCA + trees | 0.0 / 1 | 6.6 / 1 | 11.1 / 1 | 15.4 / 1 | 19.7 / 1 | 0.0 / 1 | 6.6 / 1 | 11.1 / 1 | 15.4 / 1 | 19.7 / 1 |
| trace change: hand-picked bands + gp | 1.6 / 0 | 11.0 / 0 | 22.2 / 0 | 33.5 / 0 | 44.7 / 0 | 3.0 / 0 | 11.0 / 0 | 22.2 / 0 | 33.5 / 0 | 47.2 / 0 |
| trace: physics difference (sigma_e) | 1.6 / 0 | 12.5 / 0 | 24.1 / 0 | 35.6 / 0 | 31.6 / 0 | 1.6 / 0 | 12.5 / 0 | 24.1 / 0 | 35.6 / 0 | 34.7 / 0 |
| trace: physics difference + correction (sigma_e) | 3.0 / 0 | 12.5 / 1 | 24.1 / 1 | 35.6 / 1 | 30.8 / 0 | 3.0 / 0 | 12.5 / 1 | 24.1 / 1 | 35.6 / 1 | 28.6 / 0 |
| settings + trace: combined trees | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings + trace: residual trees | 1.0 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.1 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings + trace: direction classifier | 1.7 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 2.0 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| ensemble: trace models | 3.0 / 1 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 2.2 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| ensemble: settings + trace | 1.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.1 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute settings: gp | 1.1 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.1 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute settings: trees | 1.3 / 0 | 5.0 / 1 | 8.7 / 1 | 12.4 / 1 | 16.1 / 1 | 1.7 / 0 | 5.0 / 1 | 8.7 / 1 | 15.4 / 1 | 19.7 / 1 |
| absolute settings + amplifier: gp | 1.1 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 2.0 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute settings + amplifier: trees | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute trace: physics | 0.0 / 1 | 6.6 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 0.0 / 1 | 6.6 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| absolute trace: gp | 1.7 / 0 | 6.8 / 0 | 11.1 / 0 | 15.4 / 0 | 19.7 / 0 | 1.7 / 0 | 6.8 / 0 | 11.1 / 0 | 15.4 / 0 | 19.7 / 0 |
| absolute trace: trees | 0.0 / 1 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 2.5 / 1 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| absolute trace: nn | 1.6 / 0 | 10.6 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 1.7 / 0 | 6.8 / 0 | 11.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| absolute trace: cnn | 1.7 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 1.7 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| absolute trace: pretrained cnn | 1.7 / 0 | 6.6 / 0 | 11.1 / 0 | 15.4 / 0 | 23.2 / 0 | 1.7 / 0 | 6.6 / 0 | 11.1 / 0 | 15.4 / 0 | 23.2 / 0 |
| absolute trace: deep kernel | 1.0 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 1.0 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| hybrid: sigma_e absolute trees, lambda change trees | 0.0 / 1 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 2.2 / 1 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |

### new amplifier (60 settings)

| model | 1:0 weighted | 0.75:0.25 weighted | 0.5:0.5 weighted | 0.25:0.75 weighted | 0:1 weighted | 1:0 pareto | 0.75:0.25 pareto | 0.5:0.5 pareto | 0.25:0.75 pareto | 0:1 pareto |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 2.1 / 0.18 | 8.3 / 0.17 | 15.8 / 0.15 | 23.4 / 0.16 | 31.1 / 0.17 | 2.1 / 0.16 | 8.5 / 0.17 | 15.6 / 0.17 | 23.8 / 0.17 | 30.7 / 0.16 |
| no change | 1.8 / 0 | 8.5 / 0 | 16.2 / 0 | 22.4 / 0 | 32.5 / 0 | 2.1 / 0 | 9.0 / 0 | 17.5 / 0 | 20.6 / 0 | 32.8 / 0 |
| settings change: gp | 1.0 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.0 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings change: trees | 1.1 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.1 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| trace change: trees | 1.6 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 1.6 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| trace: combined trees [A, B, B-A] | 1.6 / 0 | 12.5 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 1.6 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| trace: direction classifier [A, B, B-A] | 1.7 / 1 | 11.0 / 0 | 24.1 / 0 | 9.1 / 0 | 12.0 / 0 | 1.6 / 1 | 12.5 / 0 | 24.1 / 0 | 35.6 / 0 | 47.2 / 0 |
| trace change: shared-encoder net | 1.6 / 0 | 12.6 / 0 | 17.3 / 0 | 24.6 / 0 | 31.9 / 0 | 1.6 / 0 | 12.6 / 0 | 25.6 / 0 | 24.6 / 0 | 31.9 / 0 |
| trace change: difference curves PCA + trees | 1.7 / 1 | 6.6 / 1 | 11.1 / 1 | 15.4 / 1 | 19.7 / 1 | 1.7 / 1 | 6.6 / 1 | 11.1 / 1 | 15.4 / 1 | 19.7 / 1 |
| trace change: hand-picked bands + gp | 1.0 / 0 | 10.6 / 0 | 24.0 / 0 | 34.8 / 0 | 42.4 / 0 | 1.6 / 0 | 10.6 / 0 | 24.0 / 0 | 34.8 / 0 | 45.7 / 0 |
| trace: physics difference (sigma_e) | 1.6 / 0 | 12.5 / 0 | 24.1 / 0 | 35.6 / 0 | 26.4 / 0 | 1.6 / 0 | 12.5 / 0 | 24.1 / 0 | 35.6 / 0 | 30.5 / 0 |
| trace: physics difference + correction (sigma_e) | 1.6 / 0 | 6.8 / 1 | 11.1 / 1 | 15.4 / 1 | 29.5 / 0 | 1.6 / 0 | 6.8 / 1 | 11.1 / 1 | 15.4 / 1 | 17.9 / 0 |
| settings + trace: combined trees | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 3.1 / 1 | 6.1 / 1 | 9.1 / 1 | 12.0 / 1 |
| settings + trace: residual trees | 1.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.0 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings + trace: direction classifier | 1.7 / 0 | 6.6 / 1 | 11.1 / 1 | 9.1 / 1 | 0.0 / 1 | 1.7 / 0 | 6.6 / 1 | 11.1 / 1 | 15.4 / 1 | 0.0 / 1 |
| ensemble: trace models | 1.6 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 1.6 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| ensemble: settings + trace | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 2.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute settings: gp | 1.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.1 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute settings: trees | 1.0 / 0 | 3.1 / 1 | 6.1 / 1 | 12.4 / 1 | 16.1 / 1 | 1.0 / 0 | 5.4 / 0 | 10.4 / 0 | 15.4 / 0 | 20.3 / 0 |
| absolute trace: physics | 2.9 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 1.7 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| absolute trace: gp | 2.0 / 0 | 0.0 / 1 | 24.0 / 1 | 15.4 / 1 | 20.3 / 1 | 3.0 / 0 | 12.5 / 1 | 24.0 / 1 | 34.8 / 1 | 45.7 / 0 |
| absolute trace: trees | 1.7 / 0 | 10.6 / 0 | 21.6 / 0 | 32.6 / 0 | 43.6 / 0 | 1.7 / 0 | 10.6 / 0 | 21.6 / 0 | 32.6 / 0 | 43.6 / 0 |
| absolute trace: nn | 1.6 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 1.6 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
| absolute trace: cnn | 1.7 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 1.7 / 0 | 3.1 / 0 | 6.1 / 0 | 12.4 / 0 | 16.1 / 0 |
| absolute trace: pretrained cnn | 3.4 / 0 | 5.0 / 0 | 8.7 / 0 | 12.4 / 0 | 16.1 / 0 | 2.9 / 0 | 5.0 / 0 | 8.7 / 0 | 12.4 / 0 | 16.1 / 0 |
| absolute trace: deep kernel | 1.6 / 0 | 12.5 / 0 | 20.2 / 0 | 28.9 / 0 | 38.9 / 0 | 1.7 / 0 | 12.5 / 0 | 20.2 / 0 | 29.9 / 0 | 39.6 / 0 |
| hybrid: sigma_e absolute trees, lambda change trees | 1.7 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 | 1.7 / 0 | 3.1 / 0 | 6.1 / 0 | 9.1 / 0 | 12.0 / 0 |
