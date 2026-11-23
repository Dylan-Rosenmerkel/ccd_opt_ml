# Level-1 scorecard, v2

Board 101 ch0/1/3; 299 rows from 103 packets ({'G': 232, 'H': 67}); 5 contiguous time blocks of session D; fractional-window rows left out. Trace models never see the settings. Pairs: {'within packet (image)': 242, 'within packet (strip)': 193, 'n -> n+1': 293, 'n -> n+2': 287, 'random pairs': 9572}.

Image-label change models train on the within-packet (image), n -> n+1, n -> n+2 and random pairs (random capped per split); strip-label models on the strip pairs; both orders. A test pair never shares a time block with a training pair.

Label noise of a change: sigma_e 1.9 %, lambda 12.0 %, gain 0.8 %; clear = sigma_e > 5.0 %, lambda > 15.0 %, gain > 1.2 %.

## Absolute, trained amplifiers, session E: typical error % (vs settings-only `settings + amplifier: gp`)

| model | sigma_e | lambda | gain |
|---|---|---|---|
| settings: gp | 13.8 (worse) | 46.3 (worse) | 2.5 (worse) |
| settings: trees | 13.7 (worse) | 45.8 (worse) | 2.5 (worse) |
| settings + amplifier: gp | 9.5 (reference) | 14.6 (reference) | 0.6 (reference) |
| settings + amplifier: trees | 7.7 (better) | 10.5 (better) | 0.9 (worse) |
| mean | 14.6 (worse) | 61.1 (worse) | 3.3 (worse) |
| amplifier mean | 9.8 (same) | 44.1 (worse) | 2.4 (worse) |
| trace: physics | 7.6 (same) | 24.2 (worse) | 1.1 (worse) |
| trace: trees | 8.0 (same) | 26.1 (worse) | 1.4 (worse) |
| trace: nn | 8.4 (same) | 25.6 (worse) | 1.9 (worse) |

## Absolute, new amplifier, session E: typical error % (vs settings-only `settings: gp`)

| model | sigma_e | lambda | gain |
|---|---|---|---|
| settings: gp | 18.8 (reference) | 71.7 (reference) | 3.9 (reference) |
| settings: trees | 18.1 (same) | 66.4 (better) | 3.6 (same) |
| mean | 18.9 (same) | 76.1 (worse) | 4.1 (worse) |
| trace: physics | 13.5 (better) | 50.9 (better) | 4.3 (worse) |
| trace: trees | 12.7 (better) | 49.5 (better) | 4.8 (worse) |
| trace: nn | 14.2 (better) | 51.1 (better) | 6.0 (worse) |

## Relative, trained amplifiers, session E, within packet (image)

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 28 clear of 52, true change RMS 14.2 %; lambda: 38 clear of 52, true change RMS 58.2 %; gain: 40 clear of 52, true change RMS 2.8 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 100 % | 14.2 | 5.6 | - | 50 % (worse) | 84 % | 58.2 | 31.2 | - | 50 % (worse) | 45 % | 2.8 | 2.2 | - |
| settings change: gp | 100 % (same) | 100 % | 11.3 | 4.6 | 0.42 | 100 % (same) | 84 % | 24.2 | 18.0 | 0.91 | 100 % (same) | 45 % | 1.4 | 0.8 | 0.87 |
| settings change: trees | 100 % (same) | 100 % | 9.9 | 3.0 | 0.62 | 100 % (same) | 84 % | 25.4 | 18.5 | 0.90 | 100 % (same) | 45 % | 1.2 | 0.8 | 0.91 |
| settings + amplifier: gp | 93 % (reference) | 100 % | 10.1 | 3.8 | 0.63 | 100 % (reference) | 84 % | 11.2 | 5.2 | 0.97 | 98 % (reference) | 45 % | 1.1 | 0.8 | 0.92 |
| settings + amplifier: trees | 100 % (same) | 100 % | 9.1 | 2.3 | 0.78 | 100 % (same) | 84 % | 11.5 | 5.3 | 0.97 | 100 % (same) | 45 % | 1.0 | 0.6 | 0.95 |
| trace change: trees | 93 % (same) | 100 % | 11.3 | 3.1 | 0.49 | 79 % (worse) | 84 % | 38.0 | 18.2 | 0.69 | 98 % (same) | 45 % | 1.4 | 0.6 | 0.86 |
| trace: combined trees [A, B, B-A] | 93 % (same) | 100 % | 10.0 | 4.3 | 0.43 | 89 % (same) | 84 % | 23.8 | 10.1 | 0.89 | 92 % (same) | 45 % | 1.4 | 0.7 | 0.88 |
| trace change: difference curves PCA + trees | 93 % (same) | 100 % | 13.2 | 6.1 | 0.37 | 82 % (worse) | 84 % | 37.8 | 17.8 | 0.66 | 78 % (worse) | 45 % | 2.5 | 1.5 | 0.59 |
| trace: physics difference (sigma_e) | 79 % (same) | 100 % | 12.2 | 6.1 | 0.50 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 89 % (same) | 100 % | 9.3 | 3.5 | 0.56 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 100 % (same) | 100 % | 9.0 | 3.2 | 0.80 | 100 % (same) | 84 % | 11.9 | 6.1 | 0.98 | 100 % (same) | 45 % | 0.7 | 0.5 | 0.97 |
| settings + trace: residual trees | 100 % (same) | 100 % | 8.6 | 3.4 | 0.72 | 95 % (same) | 84 % | 15.4 | 7.2 | 0.94 | 98 % (same) | 45 % | 1.3 | 0.6 | 0.92 |
| settings + trace: direction classifier | 96 % (same) | 100 % | - | - | 0.54 | 100 % (same) | 84 % | - | - | 0.92 | 100 % (same) | 45 % | - | - | 0.91 |
| ensemble: trace models | 93 % (same) | 100 % | 9.8 | 3.0 | 0.52 | 79 % (worse) | 84 % | 29.8 | 13.0 | 0.77 | 95 % (same) | 45 % | 1.4 | 0.6 | 0.87 |
| ensemble: settings + trace | 100 % (same) | 100 % | 8.4 | 3.0 | 0.79 | 100 % (same) | 84 % | 12.7 | 4.9 | 0.97 | 98 % (same) | 45 % | 0.9 | 0.6 | 0.96 |

## Relative, trained amplifiers, session E, within packet (strip)

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 23 clear of 35, true change RMS 20.9 %; lambda: 31 clear of 35, true change RMS 62.6 %; gain: 29 clear of 35, true change RMS 16.1 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 78 % | 20.9 | 8.0 | - | 50 % (worse) | 84 % | 62.6 | 41.4 | - | 50 % (worse) | 62 % | 16.1 | 2.6 | - |
| settings change: gp | 78 % (same) | 78 % | 18.5 | 6.6 | 0.40 | 94 % (same) | 84 % | 42.6 | 28.4 | 0.76 | 72 % (same) | 62 % | 16.1 | 2.6 | 0.39 |
| settings change: trees | 78 % (same) | 78 % | 16.1 | 5.4 | 0.69 | 94 % (same) | 84 % | 40.6 | 32.2 | 0.73 | 72 % (same) | 62 % | 16.0 | 2.0 | 0.38 |
| settings + amplifier: gp | 91 % (reference) | 78 % | 15.0 | 4.0 | 0.71 | 94 % (reference) | 84 % | 29.8 | 13.9 | 0.87 | 79 % (reference) | 62 % | 16.1 | 2.6 | 0.48 |
| settings + amplifier: trees | 78 % (same) | 78 % | 16.2 | 4.9 | 0.72 | 94 % (same) | 84 % | 30.3 | 7.5 | 0.89 | 62 % (same) | 62 % | 15.6 | 2.9 | 0.34 |
| trace change: trees | 87 % (same) | 78 % | 16.4 | 5.8 | 0.67 | 94 % (same) | 84 % | 41.2 | 19.4 | 0.79 | 69 % (same) | 62 % | 15.9 | 2.5 | 0.37 |
| trace: combined trees [A, B, B-A] | 87 % (same) | 78 % | 15.8 | 4.9 | 0.73 | 90 % (same) | 84 % | 35.2 | 19.4 | 0.84 | 45 % (worse) | 62 % | 16.0 | 2.3 | 0.17 |
| trace change: difference curves PCA + trees | 70 % (same) | 78 % | 18.9 | 11.0 | 0.64 | 65 % (worse) | 84 % | 58.0 | 43.7 | 0.67 | 52 % (worse) | 62 % | 15.2 | 3.2 | 0.31 |
| settings + trace: combined trees | 91 % (same) | 78 % | 15.8 | 4.9 | 0.75 | 94 % (same) | 84 % | 29.8 | 9.2 | 0.88 | 62 % (same) | 62 % | 15.6 | 2.7 | 0.38 |
| settings + trace: residual trees | 78 % (same) | 78 % | 16.1 | 5.2 | 0.66 | 90 % (same) | 84 % | 34.2 | 15.0 | 0.85 | 66 % (same) | 62 % | 15.2 | 4.1 | 0.39 |
| settings + trace: direction classifier | 91 % (same) | 78 % | - | - | 0.67 | 94 % (same) | 84 % | - | - | 0.76 | 79 % (same) | 62 % | - | - | 0.49 |
| ensemble: trace models | 91 % (same) | 78 % | 16.0 | 5.8 | 0.73 | 94 % (same) | 84 % | 35.7 | 15.8 | 0.85 | 69 % (same) | 62 % | 15.9 | 2.8 | 0.39 |
| ensemble: settings + trace | 83 % (same) | 78 % | 15.9 | 4.3 | 0.72 | 94 % (same) | 84 % | 30.5 | 10.8 | 0.88 | 69 % (same) | 62 % | 15.3 | 3.2 | 0.43 |

## Relative, trained amplifiers, session E, n -> n+1

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 42 clear of 64, true change RMS 14.0 %; lambda: 50 clear of 64, true change RMS 72.9 %; gain: 54 clear of 64, true change RMS 3.8 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (same) | 52 % | 14.0 | 7.6 | - | 50 % (worse) | 48 % | 72.9 | 42.5 | - | 50 % (worse) | 52 % | 3.8 | 3.4 | - |
| settings change: gp | 74 % (same) | 52 % | 12.2 | 6.4 | 0.64 | 100 % (same) | 48 % | 32.9 | 21.8 | 0.95 | 98 % (same) | 52 % | 1.7 | 0.9 | 0.88 |
| settings change: trees | 79 % (same) | 52 % | 12.1 | 4.3 | 0.68 | 98 % (same) | 48 % | 34.6 | 28.6 | 0.92 | 98 % (same) | 52 % | 1.8 | 1.1 | 0.86 |
| settings + amplifier: gp | 76 % (reference) | 52 % | 12.2 | 4.3 | 0.63 | 100 % (reference) | 48 % | 14.4 | 9.0 | 0.98 | 96 % (reference) | 52 % | 1.9 | 1.2 | 0.88 |
| settings + amplifier: trees | 79 % (same) | 52 % | 11.8 | 3.6 | 0.72 | 100 % (same) | 48 % | 16.9 | 7.1 | 0.97 | 98 % (same) | 52 % | 1.6 | 0.7 | 0.90 |
| trace change: trees | 81 % (same) | 52 % | 13.1 | 3.3 | 0.63 | 90 % (same) | 48 % | 43.3 | 14.7 | 0.84 | 91 % (same) | 52 % | 1.9 | 1.2 | 0.89 |
| trace: combined trees [A, B, B-A] | 76 % (same) | 52 % | 14.6 | 6.1 | 0.53 | 92 % (same) | 48 % | 26.6 | 15.3 | 0.93 | 94 % (same) | 52 % | 1.9 | 1.0 | 0.90 |
| trace change: difference curves PCA + trees | 76 % (same) | 52 % | 12.5 | 4.0 | 0.62 | 94 % (same) | 48 % | 42.7 | 22.2 | 0.86 | 83 % (same) | 52 % | 3.0 | 2.1 | 0.67 |
| trace: physics difference (sigma_e) | 83 % (same) | 52 % | 11.6 | 5.1 | 0.66 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 90 % (same) | 52 % | 10.6 | 4.1 | 0.76 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 83 % (same) | 52 % | 12.5 | 3.7 | 0.71 | 100 % (same) | 48 % | 13.0 | 7.8 | 0.99 | 100 % (same) | 52 % | 0.9 | 0.5 | 0.97 |
| settings + trace: residual trees | 79 % (same) | 52 % | 11.0 | 4.0 | 0.71 | 98 % (same) | 48 % | 20.7 | 13.2 | 0.97 | 98 % (same) | 52 % | 1.8 | 0.9 | 0.88 |
| settings + trace: direction classifier | 83 % (same) | 52 % | - | - | 0.68 | 100 % (same) | 48 % | - | - | 0.95 | 100 % (same) | 52 % | - | - | 0.91 |
| absolute settings: gp, differenced | 86 % (same) | 52 % | 13.2 | 3.9 | 0.63 | 100 % (same) | 48 % | 31.5 | 19.3 | 0.94 | 98 % (same) | 52 % | 1.5 | 0.9 | 0.91 |
| absolute settings: trees, differenced | 86 % (same) | 52 % | 13.0 | 5.0 | 0.63 | 100 % (same) | 48 % | 32.3 | 21.6 | 0.94 | 94 % (same) | 52 % | 1.6 | 1.0 | 0.92 |
| absolute settings + amplifier: gp, differenced | 79 % (same) | 52 % | 14.6 | 3.9 | 0.65 | 100 % (same) | 48 % | 18.8 | 10.1 | 0.97 | 100 % (same) | 52 % | 0.8 | 0.5 | 0.97 |
| absolute settings + amplifier: trees, differenced | 90 % (same) | 52 % | 11.6 | 3.0 | 0.75 | 98 % (same) | 48 % | 13.7 | 7.6 | 0.98 | 96 % (same) | 52 % | 1.3 | 0.6 | 0.94 |
| absolute trace: physics, differenced | 88 % (same) | 52 % | 10.8 | 2.9 | 0.76 | 96 % (same) | 48 % | 33.7 | 13.6 | 0.88 | 100 % (same) | 52 % | 1.4 | 1.1 | 0.94 |
| absolute trace: trees, differenced | 88 % (same) | 52 % | 11.7 | 4.2 | 0.67 | 88 % (same) | 48 % | 35.5 | 19.8 | 0.87 | 98 % (same) | 52 % | 2.0 | 1.4 | 0.88 |
| absolute trace: nn, differenced | 88 % (same) | 52 % | 12.4 | 3.3 | 0.67 | 86 % (same) | 48 % | 34.1 | 15.5 | 0.91 | 80 % (same) | 52 % | 2.9 | 1.7 | 0.69 |
| ensemble: trace models | 86 % (same) | 52 % | 11.5 | 3.4 | 0.71 | 94 % (same) | 48 % | 31.8 | 11.7 | 0.90 | 94 % (same) | 52 % | 1.6 | 0.9 | 0.94 |
| ensemble: settings + trace | 90 % (same) | 52 % | 10.5 | 3.2 | 0.78 | 98 % (same) | 48 % | 18.4 | 9.1 | 0.97 | 100 % (same) | 52 % | 1.1 | 0.6 | 0.96 |

## Relative, trained amplifiers, session E, n -> n+2

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 36 clear of 61, true change RMS 15.4 %; lambda: 48 clear of 61, true change RMS 62.5 %; gain: 43 clear of 61, true change RMS 3.3 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 53 % | 15.4 | 7.4 | - | 50 % (worse) | 54 % | 62.5 | 35.1 | - | 50 % (worse) | 58 % | 3.3 | 2.2 | - |
| settings change: gp | 81 % (same) | 53 % | 14.4 | 4.4 | 0.56 | 100 % (same) | 54 % | 29.9 | 24.2 | 0.91 | 95 % (same) | 58 % | 1.9 | 1.0 | 0.79 |
| settings change: trees | 81 % (same) | 53 % | 13.0 | 4.4 | 0.71 | 100 % (same) | 54 % | 31.0 | 21.2 | 0.90 | 95 % (same) | 58 % | 1.6 | 1.1 | 0.88 |
| settings + amplifier: gp | 83 % (reference) | 53 % | 13.6 | 4.4 | 0.68 | 100 % (reference) | 54 % | 17.6 | 8.8 | 0.97 | 95 % (reference) | 58 % | 1.8 | 1.0 | 0.86 |
| settings + amplifier: trees | 86 % (same) | 53 % | 12.5 | 4.0 | 0.79 | 100 % (same) | 54 % | 15.0 | 8.9 | 0.98 | 95 % (same) | 58 % | 1.3 | 0.8 | 0.91 |
| trace change: trees | 78 % (same) | 53 % | 12.8 | 4.8 | 0.61 | 90 % (same) | 54 % | 48.4 | 20.0 | 0.74 | 93 % (same) | 58 % | 1.9 | 1.0 | 0.84 |
| trace: combined trees [A, B, B-A] | 89 % (same) | 53 % | 11.8 | 5.2 | 0.66 | 96 % (same) | 54 % | 31.0 | 10.1 | 0.90 | 91 % (same) | 58 % | 2.0 | 1.0 | 0.80 |
| trace change: difference curves PCA + trees | 81 % (same) | 53 % | 14.5 | 6.3 | 0.57 | 77 % (same) | 54 % | 42.7 | 24.5 | 0.74 | 81 % (same) | 58 % | 2.5 | 1.4 | 0.61 |
| trace: physics difference (sigma_e) | 81 % (same) | 53 % | 13.6 | 7.1 | 0.52 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 89 % (same) | 53 % | 10.2 | 4.6 | 0.77 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 92 % (same) | 53 % | 11.7 | 3.2 | 0.87 | 100 % (same) | 54 % | 14.9 | 7.5 | 0.98 | 100 % (same) | 58 % | 1.0 | 0.6 | 0.96 |
| settings + trace: residual trees | 86 % (same) | 53 % | 11.4 | 4.7 | 0.76 | 100 % (same) | 54 % | 20.2 | 11.3 | 0.96 | 91 % (same) | 58 % | 1.7 | 0.9 | 0.86 |
| settings + trace: direction classifier | 94 % (same) | 53 % | - | - | 0.79 | 100 % (same) | 54 % | - | - | 0.90 | 98 % (same) | 58 % | - | - | 0.91 |
| absolute settings: gp, differenced | 92 % (same) | 53 % | 13.2 | 3.9 | 0.73 | 100 % (same) | 54 % | 29.6 | 18.6 | 0.92 | 95 % (same) | 58 % | 1.5 | 0.9 | 0.91 |
| absolute settings: trees, differenced | 83 % (same) | 53 % | 13.4 | 4.2 | 0.68 | 100 % (same) | 54 % | 29.4 | 17.3 | 0.92 | 88 % (same) | 58 % | 1.6 | 1.1 | 0.87 |
| absolute settings + amplifier: gp, differenced | 81 % (same) | 53 % | 14.1 | 4.0 | 0.66 | 98 % (same) | 54 % | 19.4 | 7.5 | 0.96 | 98 % (same) | 58 % | 0.8 | 0.6 | 0.97 |
| absolute settings + amplifier: trees, differenced | 97 % (same) | 53 % | 11.8 | 3.7 | 0.82 | 100 % (same) | 54 % | 14.6 | 5.6 | 0.99 | 95 % (same) | 58 % | 1.2 | 1.0 | 0.92 |
| absolute trace: physics, differenced | 89 % (same) | 53 % | 12.1 | 3.2 | 0.77 | 94 % (same) | 54 % | 34.2 | 14.2 | 0.88 | 88 % (same) | 58 % | 1.5 | 0.9 | 0.87 |
| absolute trace: trees, differenced | 86 % (same) | 53 % | 11.0 | 4.0 | 0.77 | 94 % (same) | 54 % | 37.0 | 13.0 | 0.80 | 86 % (same) | 58 % | 2.0 | 1.6 | 0.79 |
| absolute trace: nn, differenced | 89 % (same) | 53 % | 12.7 | 4.8 | 0.74 | 85 % (same) | 54 % | 36.6 | 16.6 | 0.81 | 70 % (same) | 58 % | 3.0 | 2.3 | 0.49 |
| ensemble: trace models | 83 % (same) | 53 % | 10.4 | 3.3 | 0.77 | 92 % (same) | 54 % | 35.1 | 13.4 | 0.85 | 95 % (same) | 58 % | 1.6 | 0.8 | 0.87 |
| ensemble: settings + trace | 94 % (same) | 53 % | 10.4 | 3.0 | 0.87 | 100 % (same) | 54 % | 19.9 | 7.4 | 0.98 | 98 % (same) | 58 % | 1.2 | 0.8 | 0.94 |

## Relative, trained amplifiers, session E, random pairs

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 387 clear of 715, true change RMS 14.0 %; lambda: 510 clear of 715, true change RMS 63.5 %; gain: 529 clear of 715, true change RMS 3.4 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 63 % | 14.0 | 5.5 | - | 50 % (worse) | 57 % | 63.5 | 31.8 | - | 50 % (worse) | 58 % | 3.4 | 2.3 | - |
| settings change: gp | 83 % (same) | 63 % | 12.8 | 4.9 | 0.61 | 100 % (same) | 57 % | 28.5 | 18.3 | 0.94 | 95 % (same) | 58 % | 1.8 | 1.1 | 0.87 |
| settings change: trees | 89 % (better) | 63 % | 12.1 | 4.2 | 0.69 | 99 % (same) | 57 % | 29.9 | 19.7 | 0.93 | 97 % (same) | 58 % | 1.6 | 0.9 | 0.90 |
| settings + amplifier: gp | 81 % (reference) | 63 % | 12.6 | 4.0 | 0.63 | 98 % (reference) | 57 % | 16.7 | 8.7 | 0.96 | 94 % (reference) | 58 % | 1.8 | 1.0 | 0.89 |
| settings + amplifier: trees | 89 % (better) | 63 % | 11.6 | 3.2 | 0.75 | 99 % (same) | 57 % | 15.3 | 7.1 | 0.97 | 97 % (same) | 58 % | 1.4 | 0.7 | 0.92 |
| trace change: trees | 84 % (same) | 63 % | 12.4 | 3.7 | 0.62 | 87 % (worse) | 57 % | 44.1 | 14.8 | 0.75 | 94 % (same) | 58 % | 1.7 | 0.9 | 0.88 |
| trace: combined trees [A, B, B-A] | 80 % (same) | 63 % | 12.2 | 5.0 | 0.61 | 94 % (same) | 57 % | 31.2 | 11.0 | 0.89 | 92 % (same) | 58 % | 1.8 | 0.9 | 0.85 |
| trace change: difference curves PCA + trees | 74 % (same) | 63 % | 13.6 | 5.4 | 0.53 | 78 % (worse) | 57 % | 43.3 | 23.2 | 0.69 | 84 % (worse) | 58 % | 2.5 | 1.4 | 0.67 |
| trace: physics difference (sigma_e) | 82 % (same) | 63 % | 12.2 | 4.8 | 0.58 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 90 % (same) | 63 % | 10.0 | 3.5 | 0.76 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 91 % (same) | 63 % | 11.4 | 3.5 | 0.78 | 100 % (same) | 57 % | 15.3 | 7.2 | 0.98 | 100 % (better) | 58 % | 0.9 | 0.5 | 0.97 |
| settings + trace: residual trees | 88 % (same) | 63 % | 11.1 | 4.0 | 0.72 | 99 % (same) | 57 % | 19.0 | 8.9 | 0.96 | 95 % (same) | 58 % | 1.7 | 0.9 | 0.89 |
| settings + trace: direction classifier | 90 % (same) | 63 % | - | - | 0.75 | 99 % (same) | 57 % | - | - | 0.95 | 99 % (better) | 58 % | - | - | 0.93 |
| absolute settings: gp, differenced | 89 % (same) | 63 % | 12.5 | 3.8 | 0.68 | 99 % (same) | 57 % | 29.1 | 17.3 | 0.94 | 97 % (same) | 58 % | 1.4 | 0.8 | 0.92 |
| absolute settings: trees, differenced | 87 % (same) | 63 % | 12.2 | 4.6 | 0.66 | 99 % (same) | 57 % | 28.9 | 17.4 | 0.93 | 93 % (same) | 58 % | 1.6 | 0.9 | 0.88 |
| absolute settings + amplifier: gp, differenced | 87 % (same) | 63 % | 13.5 | 4.3 | 0.69 | 97 % (same) | 57 % | 19.3 | 8.3 | 0.94 | 99 % (better) | 58 % | 0.8 | 0.5 | 0.97 |
| absolute settings + amplifier: trees, differenced | 91 % (same) | 63 % | 11.0 | 3.4 | 0.77 | 98 % (same) | 57 % | 14.6 | 7.2 | 0.97 | 96 % (same) | 58 % | 1.3 | 0.8 | 0.93 |
| absolute trace: physics, differenced | 91 % (same) | 63 % | 10.7 | 3.4 | 0.76 | 92 % (same) | 57 % | 33.4 | 14.0 | 0.85 | 95 % (same) | 58 % | 1.6 | 0.9 | 0.90 |
| absolute trace: trees, differenced | 85 % (same) | 63 % | 11.1 | 3.9 | 0.69 | 88 % (worse) | 57 % | 38.2 | 16.2 | 0.78 | 91 % (same) | 58 % | 1.9 | 1.2 | 0.85 |
| absolute trace: nn, differenced | 87 % (same) | 63 % | 11.9 | 3.3 | 0.68 | 81 % (worse) | 57 % | 36.9 | 15.8 | 0.78 | 80 % (worse) | 58 % | 2.8 | 1.6 | 0.61 |
| ensemble: trace models | 87 % (same) | 63 % | 10.3 | 3.2 | 0.75 | 92 % (same) | 57 % | 33.6 | 11.1 | 0.86 | 94 % (same) | 58 % | 1.5 | 0.8 | 0.90 |
| ensemble: settings + trace | 94 % (better) | 63 % | 10.0 | 3.1 | 0.82 | 99 % (same) | 57 % | 19.8 | 7.4 | 0.96 | 99 % (better) | 58 % | 1.1 | 0.6 | 0.96 |

## Relative, new amplifier, session E, within packet (image)

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 28 clear of 52, true change RMS 14.2 %; lambda: 38 clear of 52, true change RMS 58.2 %; gain: 40 clear of 52, true change RMS 2.8 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 100 % | 14.2 | 5.6 | - | 50 % (worse) | 84 % | 58.2 | 31.2 | - | 50 % (worse) | 45 % | 2.8 | 2.2 | - |
| settings change: gp | 96 % (reference) | 100 % | 12.4 | 3.9 | 0.38 | 100 % (reference) | 84 % | 34.1 | 20.8 | 0.85 | 100 % (reference) | 45 % | 1.4 | 0.9 | 0.85 |
| settings change: trees | 100 % (same) | 100 % | 11.7 | 3.7 | 0.32 | 100 % (same) | 84 % | 32.7 | 26.1 | 0.82 | 100 % (same) | 45 % | 1.5 | 0.9 | 0.85 |
| trace change: trees | 89 % (same) | 100 % | 12.9 | 3.9 | 0.41 | 82 % (worse) | 84 % | 38.3 | 20.6 | 0.65 | 90 % (worse) | 45 % | 1.8 | 1.1 | 0.76 |
| trace: combined trees [A, B, B-A] | 89 % (same) | 100 % | 11.6 | 5.1 | 0.37 | 76 % (worse) | 84 % | 38.8 | 20.2 | 0.68 | 88 % (worse) | 45 % | 1.8 | 1.7 | 0.80 |
| trace change: difference curves PCA + trees | 86 % (same) | 100 % | 14.1 | 5.3 | 0.35 | 76 % (worse) | 84 % | 36.0 | 21.4 | 0.65 | 75 % (worse) | 45 % | 2.8 | 1.5 | 0.52 |
| trace: physics difference (sigma_e) | 82 % (same) | 100 % | 11.9 | 5.1 | 0.45 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 89 % (same) | 100 % | 10.8 | 4.1 | 0.39 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 100 % (same) | 100 % | 10.0 | 3.6 | 0.53 | 97 % (same) | 84 % | 27.3 | 10.6 | 0.88 | 100 % (same) | 45 % | 1.6 | 0.8 | 0.84 |
| settings + trace: residual trees | 96 % (same) | 100 % | 12.4 | 4.3 | 0.27 | 97 % (same) | 84 % | 29.7 | 21.6 | 0.85 | 100 % (same) | 45 % | 1.7 | 1.1 | 0.81 |
| settings + trace: direction classifier | 96 % (same) | 100 % | - | - | 0.40 | 100 % (same) | 84 % | - | - | 0.62 | 98 % (same) | 45 % | - | - | 0.84 |
| ensemble: trace models | 93 % (same) | 100 % | 11.1 | 3.9 | 0.40 | 76 % (worse) | 84 % | 36.1 | 23.2 | 0.69 | 88 % (worse) | 45 % | 1.7 | 1.4 | 0.79 |
| ensemble: settings + trace | 100 % (same) | 100 % | 10.3 | 4.1 | 0.46 | 97 % (same) | 84 % | 27.3 | 15.9 | 0.87 | 100 % (same) | 45 % | 1.6 | 0.8 | 0.83 |

## Relative, new amplifier, session E, within packet (strip)

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 23 clear of 35, true change RMS 20.9 %; lambda: 31 clear of 35, true change RMS 62.6 %; gain: 29 clear of 35, true change RMS 16.1 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 78 % | 20.9 | 8.0 | - | 50 % (worse) | 84 % | 62.6 | 41.4 | - | 50 % (same) | 62 % | 16.1 | 2.6 | - |
| settings change: gp | 96 % (reference) | 78 % | 15.2 | 5.2 | 0.69 | 94 % (reference) | 84 % | 54.8 | 46.0 | 0.67 | 66 % (reference) | 62 % | 16.1 | 2.6 | 0.25 |
| settings change: trees | 78 % (same) | 78 % | 16.3 | 7.2 | 0.60 | 94 % (same) | 84 % | 55.6 | 44.2 | 0.61 | 62 % (same) | 62 % | 16.5 | 2.7 | 0.23 |
| trace change: trees | 87 % (same) | 78 % | 17.4 | 5.7 | 0.55 | 90 % (same) | 84 % | 43.7 | 24.0 | 0.69 | 66 % (same) | 62 % | 16.9 | 2.6 | 0.16 |
| trace: combined trees [A, B, B-A] | 87 % (same) | 78 % | 17.3 | 7.4 | 0.55 | 90 % (same) | 84 % | 42.3 | 25.6 | 0.78 | 41 % (same) | 62 % | 18.0 | 4.3 | -0.16 |
| trace change: difference curves PCA + trees | 61 % (worse) | 78 % | 21.2 | 12.9 | 0.57 | 52 % (worse) | 84 % | 72.4 | 66.1 | 0.60 | 62 % (same) | 62 % | 15.8 | 3.2 | 0.29 |
| settings + trace: combined trees | 87 % (same) | 78 % | 16.5 | 6.9 | 0.63 | 94 % (same) | 84 % | 44.0 | 25.0 | 0.75 | 55 % (same) | 62 % | 17.8 | 5.2 | 0.01 |
| settings + trace: residual trees | 87 % (same) | 78 % | 16.4 | 6.8 | 0.51 | 94 % (same) | 84 % | 58.0 | 47.1 | 0.66 | 62 % (same) | 62 % | 17.6 | 4.1 | 0.05 |
| settings + trace: direction classifier | 91 % (same) | 78 % | - | - | 0.74 | 94 % (same) | 84 % | - | - | 0.74 | 79 % (better) | 62 % | - | - | 0.51 |
| ensemble: trace models | 87 % (same) | 78 % | 17.3 | 6.4 | 0.57 | 90 % (same) | 84 % | 41.4 | 19.9 | 0.77 | 45 % (same) | 62 % | 17.4 | 3.0 | -0.05 |
| ensemble: settings + trace | 87 % (same) | 78 % | 16.2 | 5.4 | 0.63 | 94 % (same) | 84 % | 49.6 | 36.4 | 0.72 | 55 % (same) | 62 % | 17.6 | 4.0 | -0.01 |

## Relative, new amplifier, session E, n -> n+1

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 42 clear of 64, true change RMS 14.0 %; lambda: 50 clear of 64, true change RMS 72.9 %; gain: 54 clear of 64, true change RMS 3.8 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (same) | 52 % | 14.0 | 7.6 | - | 50 % (worse) | 48 % | 72.9 | 42.5 | - | 50 % (worse) | 52 % | 3.8 | 3.4 | - |
| settings change: gp | 74 % (reference) | 52 % | 13.0 | 5.9 | 0.66 | 100 % (reference) | 48 % | 47.3 | 31.1 | 0.88 | 98 % (reference) | 52 % | 2.1 | 1.1 | 0.87 |
| settings change: trees | 86 % (same) | 52 % | 12.5 | 6.3 | 0.63 | 100 % (same) | 48 % | 44.1 | 34.0 | 0.89 | 98 % (same) | 52 % | 2.0 | 1.0 | 0.87 |
| trace change: trees | 83 % (same) | 52 % | 15.1 | 5.3 | 0.57 | 90 % (same) | 48 % | 47.1 | 23.7 | 0.80 | 93 % (same) | 52 % | 2.1 | 1.5 | 0.84 |
| trace: combined trees [A, B, B-A] | 74 % (same) | 52 % | 14.1 | 7.2 | 0.50 | 84 % (same) | 48 % | 49.6 | 20.3 | 0.80 | 93 % (same) | 52 % | 2.5 | 1.7 | 0.76 |
| trace change: difference curves PCA + trees | 67 % (same) | 52 % | 13.5 | 5.8 | 0.51 | 90 % (same) | 48 % | 45.8 | 22.9 | 0.85 | 87 % (same) | 52 % | 2.6 | 1.8 | 0.74 |
| trace: physics difference (sigma_e) | 83 % (same) | 52 % | 11.6 | 4.9 | 0.64 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 88 % (same) | 52 % | 12.1 | 4.5 | 0.66 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 76 % (same) | 52 % | 13.1 | 4.2 | 0.57 | 98 % (same) | 48 % | 31.6 | 13.7 | 0.94 | 96 % (same) | 52 % | 2.1 | 1.1 | 0.87 |
| settings + trace: residual trees | 86 % (same) | 52 % | 12.8 | 5.6 | 0.66 | 98 % (same) | 48 % | 37.7 | 25.0 | 0.91 | 98 % (same) | 52 % | 2.1 | 1.2 | 0.85 |
| settings + trace: direction classifier | 74 % (same) | 52 % | - | - | 0.50 | 100 % (same) | 48 % | - | - | 0.85 | 93 % (same) | 52 % | - | - | 0.87 |
| absolute settings: gp, differenced | 79 % (same) | 52 % | 14.2 | 8.9 | 0.43 | 90 % (same) | 48 % | 63.1 | 47.0 | 0.73 | 91 % (same) | 52 % | 2.9 | 1.6 | 0.72 |
| absolute settings: trees, differenced | 81 % (same) | 52 % | 13.6 | 6.9 | 0.59 | 98 % (same) | 48 % | 46.8 | 34.4 | 0.88 | 93 % (same) | 52 % | 2.1 | 1.1 | 0.83 |
| absolute trace: physics, differenced | 86 % (same) | 52 % | 11.9 | 4.9 | 0.67 | 70 % (worse) | 48 % | 65.2 | 22.0 | 0.41 | 94 % (same) | 52 % | 3.8 | 3.2 | 0.76 |
| absolute trace: trees, differenced | 81 % (same) | 52 % | 11.7 | 4.7 | 0.56 | 62 % (worse) | 48 % | 62.1 | 37.8 | 0.51 | 69 % (worse) | 52 % | 3.8 | 3.0 | 0.37 |
| absolute trace: nn, differenced | 76 % (same) | 52 % | 12.7 | 4.6 | 0.60 | 86 % (same) | 48 % | 47.6 | 25.9 | 0.78 | 85 % (same) | 52 % | 2.8 | 1.8 | 0.70 |
| ensemble: trace models | 83 % (same) | 52 % | 11.9 | 4.1 | 0.66 | 86 % (same) | 48 % | 50.2 | 20.5 | 0.80 | 93 % (same) | 52 % | 2.5 | 1.8 | 0.82 |
| ensemble: settings + trace | 83 % (same) | 52 % | 11.2 | 4.1 | 0.74 | 96 % (same) | 48 % | 36.6 | 25.6 | 0.92 | 98 % (same) | 52 % | 2.2 | 1.5 | 0.86 |

## Relative, new amplifier, session E, n -> n+2

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 36 clear of 61, true change RMS 15.4 %; lambda: 48 clear of 61, true change RMS 62.5 %; gain: 43 clear of 61, true change RMS 3.3 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 53 % | 15.4 | 7.4 | - | 50 % (worse) | 54 % | 62.5 | 35.1 | - | 50 % (worse) | 58 % | 3.3 | 2.2 | - |
| settings change: gp | 86 % (reference) | 53 % | 14.4 | 6.1 | 0.65 | 100 % (reference) | 54 % | 39.2 | 28.5 | 0.84 | 98 % (reference) | 58 % | 1.9 | 1.2 | 0.85 |
| settings change: trees | 89 % (same) | 53 % | 14.5 | 5.8 | 0.58 | 100 % (same) | 54 % | 39.5 | 28.0 | 0.82 | 95 % (same) | 58 % | 1.8 | 1.0 | 0.86 |
| trace change: trees | 81 % (same) | 53 % | 15.2 | 5.6 | 0.61 | 85 % (same) | 54 % | 50.8 | 20.8 | 0.67 | 91 % (same) | 58 % | 2.1 | 0.9 | 0.80 |
| trace: combined trees [A, B, B-A] | 75 % (same) | 53 % | 14.9 | 8.3 | 0.46 | 88 % (same) | 54 % | 44.4 | 19.9 | 0.75 | 81 % (same) | 58 % | 2.2 | 1.3 | 0.72 |
| trace change: difference curves PCA + trees | 81 % (same) | 53 % | 14.4 | 4.4 | 0.57 | 79 % (same) | 54 % | 44.3 | 26.1 | 0.71 | 86 % (same) | 58 % | 2.3 | 1.2 | 0.76 |
| trace: physics difference (sigma_e) | 83 % (same) | 53 % | 13.4 | 6.7 | 0.57 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 89 % (same) | 53 % | 12.2 | 5.5 | 0.64 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 81 % (same) | 53 % | 13.9 | 5.7 | 0.59 | 100 % (same) | 54 % | 29.7 | 16.0 | 0.92 | 93 % (same) | 58 % | 1.9 | 0.8 | 0.83 |
| settings + trace: residual trees | 92 % (same) | 53 % | 14.6 | 4.5 | 0.62 | 100 % (same) | 54 % | 32.5 | 25.2 | 0.89 | 98 % (same) | 58 % | 1.9 | 1.2 | 0.86 |
| settings + trace: direction classifier | 89 % (same) | 53 % | - | - | 0.67 | 100 % (same) | 54 % | - | - | 0.74 | 95 % (same) | 58 % | - | - | 0.83 |
| absolute settings: gp, differenced | 83 % (same) | 53 % | 15.1 | 7.8 | 0.50 | 88 % (same) | 54 % | 53.1 | 35.7 | 0.67 | 79 % (same) | 58 % | 2.6 | 1.5 | 0.63 |
| absolute settings: trees, differenced | 83 % (same) | 53 % | 13.9 | 4.7 | 0.63 | 96 % (same) | 54 % | 40.0 | 30.2 | 0.83 | 88 % (same) | 58 % | 2.1 | 1.0 | 0.77 |
| absolute trace: physics, differenced | 83 % (same) | 53 % | 13.6 | 5.9 | 0.64 | 67 % (worse) | 54 % | 58.7 | 22.1 | 0.27 | 86 % (same) | 58 % | 3.2 | 2.2 | 0.69 |
| absolute trace: trees, differenced | 78 % (same) | 53 % | 12.9 | 5.7 | 0.49 | 56 % (worse) | 54 % | 58.6 | 31.4 | 0.32 | 72 % (worse) | 58 % | 3.5 | 2.1 | 0.15 |
| absolute trace: nn, differenced | 83 % (same) | 53 % | 13.3 | 5.5 | 0.64 | 81 % (same) | 54 % | 46.6 | 30.4 | 0.67 | 86 % (same) | 58 % | 2.8 | 1.4 | 0.63 |
| ensemble: trace models | 83 % (same) | 53 % | 12.4 | 5.1 | 0.64 | 77 % (same) | 54 % | 47.5 | 22.5 | 0.67 | 86 % (same) | 58 % | 2.3 | 1.3 | 0.78 |
| ensemble: settings + trace | 92 % (same) | 53 % | 12.4 | 5.1 | 0.75 | 100 % (same) | 54 % | 34.6 | 20.9 | 0.88 | 93 % (same) | 58 % | 2.0 | 1.1 | 0.85 |

## Relative, new amplifier, session E, random pairs

Per target: direction right on clear changes (verdict vs settings) | always-up rate | size error % | median error % | Spearman. sigma_e: 387 clear of 715, true change RMS 14.0 %; lambda: 510 clear of 715, true change RMS 63.5 %; gain: 529 clear of 715, true change RMS 3.4 %

| model | sigma_e dir | up | size | med | rho | lambda dir | up | size | med | rho | gain dir | up | size | med | rho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no change | 50 % (worse) | 63 % | 14.0 | 5.5 | - | 50 % (worse) | 57 % | 63.5 | 31.8 | - | 50 % (worse) | 58 % | 3.4 | 2.3 | - |
| settings change: gp | 85 % (reference) | 63 % | 13.2 | 4.8 | 0.63 | 99 % (reference) | 57 % | 40.0 | 27.2 | 0.89 | 96 % (reference) | 58 % | 1.9 | 1.1 | 0.87 |
| settings change: trees | 88 % (same) | 63 % | 12.9 | 4.8 | 0.64 | 99 % (same) | 57 % | 38.9 | 28.6 | 0.89 | 96 % (same) | 58 % | 1.8 | 1.0 | 0.87 |
| trace change: trees | 84 % (same) | 63 % | 14.1 | 4.3 | 0.59 | 86 % (worse) | 57 % | 46.8 | 19.1 | 0.72 | 91 % (same) | 58 % | 2.0 | 1.1 | 0.84 |
| trace: combined trees [A, B, B-A] | 75 % (same) | 63 % | 13.9 | 6.4 | 0.48 | 84 % (worse) | 57 % | 45.1 | 19.1 | 0.73 | 89 % (same) | 58 % | 2.2 | 1.4 | 0.78 |
| trace change: difference curves PCA + trees | 71 % (worse) | 63 % | 14.2 | 5.6 | 0.46 | 75 % (worse) | 57 % | 47.2 | 24.1 | 0.65 | 85 % (worse) | 58 % | 2.4 | 1.4 | 0.73 |
| trace: physics difference (sigma_e) | 83 % (same) | 63 % | 12.0 | 4.7 | 0.61 | - | - | - | - | - | - | - | - | - | - |
| trace: physics difference + correction (sigma_e) | 90 % (same) | 63 % | 11.5 | 4.8 | 0.65 | - | - | - | - | - | - | - | - | - | - |
| settings + trace: combined trees | 84 % (same) | 63 % | 12.3 | 4.5 | 0.62 | 99 % (same) | 57 % | 29.9 | 12.8 | 0.94 | 95 % (same) | 58 % | 1.9 | 1.0 | 0.86 |
| settings + trace: residual trees | 87 % (same) | 63 % | 13.2 | 5.1 | 0.63 | 98 % (same) | 57 % | 33.5 | 21.3 | 0.91 | 96 % (same) | 58 % | 2.0 | 1.1 | 0.85 |
| settings + trace: direction classifier | 83 % (same) | 63 % | - | - | 0.61 | 99 % (same) | 57 % | - | - | 0.84 | 94 % (same) | 58 % | - | - | 0.86 |
| absolute settings: gp, differenced | 82 % (same) | 63 % | 13.8 | 7.0 | 0.48 | 89 % (worse) | 57 % | 54.4 | 32.2 | 0.73 | 85 % (worse) | 58 % | 2.7 | 1.4 | 0.70 |
| absolute settings: trees, differenced | 83 % (same) | 63 % | 13.1 | 5.6 | 0.60 | 97 % (same) | 57 % | 40.9 | 26.6 | 0.88 | 93 % (same) | 58 % | 2.0 | 1.0 | 0.84 |
| absolute trace: physics, differenced | 81 % (same) | 63 % | 12.2 | 5.4 | 0.58 | 64 % (worse) | 57 % | 58.9 | 17.4 | 0.31 | 87 % (worse) | 58 % | 3.4 | 2.3 | 0.72 |
| absolute trace: trees, differenced | 73 % (same) | 63 % | 12.0 | 4.8 | 0.51 | 58 % (worse) | 57 % | 59.0 | 33.2 | 0.33 | 68 % (worse) | 58 % | 3.4 | 2.3 | 0.32 |
| absolute trace: nn, differenced | 82 % (same) | 63 % | 12.6 | 5.4 | 0.58 | 81 % (worse) | 57 % | 47.0 | 27.7 | 0.69 | 84 % (worse) | 58 % | 2.7 | 1.4 | 0.66 |
| ensemble: trace models | 83 % (same) | 63 % | 11.5 | 4.2 | 0.63 | 82 % (worse) | 57 % | 46.7 | 18.2 | 0.71 | 90 % (same) | 58 % | 2.3 | 1.4 | 0.82 |
| ensemble: settings + trace | 89 % (same) | 63 % | 11.2 | 4.1 | 0.73 | 97 % (same) | 57 % | 34.3 | 19.5 | 0.91 | 95 % (same) | 58 % | 2.1 | 1.2 | 0.87 |

## (5) Ranking: shortlist the settings from the traces, joint objective J(w) = w dlog sigma_e + (1-w) dlog lambda

Each setting (non-nominal packet) applies to the whole board, so its objective is the mean over the 3 amplifiers (predicted and true; truth = the setting's effect relative to nominal, drift removed). Weight w:(1-w) of sigma_e:lambda. Shortlist rules (lvl1/shortlist.py): weighted = by J(w); pareto = the predicted Pareto front first. Cells: top-5 regret (%) = how much worse the best of the 5 shortlisted settings truly is than the truly best setting / whether the truly best is on the 10-setting shortlist (1 = yes). The 3 amplifiers share their settings, so each column is ONE trial: read the pattern, not single numbers.

### trained amplifiers, session E (16 settings)

| model | 1:0 weighted | 0.75:0.25 weighted | 0.5:0.5 weighted | 0.25:0.75 weighted | 0:1 weighted | 1:0 pareto | 0.75:0.25 pareto | 0.5:0.5 pareto | 0.25:0.75 pareto | 0:1 pareto |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 1.6 / 0.61 | 7.6 / 0.61 | 12.3 / 0.62 | 16.3 / 0.63 | 20.7 / 0.63 | 1.5 / 0.62 | 7.6 / 0.63 | 13.2 / 0.62 | 16.8 / 0.61 | 20.5 / 0.62 |
| no change | 1.2 / 1 | 6.9 / 1 | 12.6 / 1 | 18.0 / 1 | 17.1 / 1 | 1.6 / 1 | 5.7 / 1 | 12.5 / 1 | 14.9 / 1 | 17.4 / 1 |
| settings change: gp | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings change: trees | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings + amplifier: gp | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings + amplifier: trees | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| trace change: trees | 1.7 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.7 / 1 | 5.4 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| trace: combined trees [A, B, B-A] | 1.7 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.7 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| trace change: difference curves PCA + trees | 3.8 / 0 | 16.6 / 1 | 16.0 / 1 | 0.0 / 1 | 0.0 / 1 | 3.8 / 0 | 16.6 / 1 | 16.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| trace: physics difference (sigma_e) | 0.0 / 1 | 5.4 / 1 | 10.0 / 1 | 14.7 / 1 | 21.0 / 1 | 0.0 / 1 | 5.4 / 1 | 10.0 / 1 | 14.7 / 1 | 24.9 / 1 |
| trace: physics difference + correction (sigma_e) | 0.0 / 1 | 5.4 / 1 | 10.0 / 1 | 14.7 / 1 | 26.0 / 1 | 0.0 / 1 | 5.4 / 1 | 10.0 / 1 | 14.7 / 1 | 22.3 / 0 |
| settings + trace: combined trees | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings + trace: residual trees | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings + trace: direction classifier | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| ensemble: trace models | 1.7 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.7 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| ensemble: settings + trace | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute settings: gp | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute settings: trees | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute settings + amplifier: gp | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.7 / 1 | 5.4 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute settings + amplifier: trees | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute trace: physics | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute trace: trees | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute trace: nn | 0.9 / 1 | 0.0 / 1 | 10.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 1 | 0.0 / 1 | 10.0 / 1 | 13.6 / 1 | 11.1 / 1 |
| hybrid: sigma_e absolute trees, lambda change trees | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |

### new amplifier, session E (16 settings)

| model | 1:0 weighted | 0.75:0.25 weighted | 0.5:0.5 weighted | 0.25:0.75 weighted | 0:1 weighted | 1:0 pareto | 0.75:0.25 pareto | 0.5:0.5 pareto | 0.25:0.75 pareto | 0:1 pareto |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 1.6 / 0.61 | 7.6 / 0.61 | 12.3 / 0.62 | 16.3 / 0.63 | 20.7 / 0.63 | 1.5 / 0.62 | 7.6 / 0.63 | 13.2 / 0.62 | 16.8 / 0.61 | 20.5 / 0.62 |
| no change | 1.2 / 1 | 6.9 / 1 | 12.6 / 1 | 18.0 / 1 | 17.1 / 1 | 1.6 / 1 | 5.7 / 1 | 12.5 / 1 | 14.9 / 1 | 17.4 / 1 |
| settings change: gp | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings change: trees | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| trace change: trees | 1.7 / 1 | 5.4 / 1 | 10.0 / 1 | 13.6 / 1 | 11.1 / 1 | 1.7 / 1 | 5.4 / 1 | 10.0 / 1 | 13.6 / 1 | 11.1 / 1 |
| trace: combined trees [A, B, B-A] | 1.7 / 1 | 5.4 / 1 | 10.0 / 1 | 0.0 / 1 | 0.0 / 1 | 1.7 / 1 | 5.4 / 1 | 10.0 / 1 | 13.6 / 1 | 11.1 / 1 |
| trace change: difference curves PCA + trees | 2.1 / 1 | 16.6 / 0 | 10.0 / 1 | 13.6 / 1 | 11.1 / 1 | 3.8 / 1 | 16.6 / 0 | 10.0 / 1 | 13.6 / 1 | 11.1 / 1 |
| trace: physics difference (sigma_e) | 0.0 / 1 | 5.4 / 1 | 10.0 / 1 | 14.7 / 1 | 24.8 / 1 | 0.0 / 1 | 5.4 / 1 | 10.0 / 1 | 14.7 / 1 | 19.8 / 1 |
| trace: physics difference + correction (sigma_e) | 0.0 / 1 | 11.2 / 0 | 23.3 / 0 | 35.3 / 0 | 14.3 / 1 | 0.0 / 1 | 11.2 / 0 | 23.3 / 0 | 35.3 / 0 | 20.3 / 1 |
| settings + trace: combined trees | 0.9 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 0 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings + trace: residual trees | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| settings + trace: direction classifier | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| ensemble: trace models | 1.7 / 1 | 5.4 / 1 | 10.0 / 1 | 13.6 / 1 | 11.1 / 1 | 1.7 / 1 | 5.4 / 1 | 10.0 / 1 | 13.6 / 1 | 11.1 / 1 |
| ensemble: settings + trace | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute settings: gp | 1.7 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.9 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute settings: trees | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 | 0.0 / 1 |
| absolute trace: physics | 1.7 / 1 | 5.4 / 1 | 23.3 / 1 | 35.3 / 1 | 47.4 / 1 | 1.7 / 1 | 11.2 / 1 | 23.3 / 1 | 35.3 / 1 | 47.4 / 1 |
| absolute trace: trees | 0.0 / 1 | 11.2 / 1 | 23.3 / 1 | 35.3 / 1 | 57.8 / 1 | 0.0 / 1 | 11.2 / 1 | 23.3 / 1 | 35.3 / 1 | 57.8 / 1 |
| absolute trace: nn | 2.1 / 1 | 11.4 / 1 | 16.0 / 1 | 13.6 / 1 | 11.1 / 1 | 2.1 / 1 | 11.4 / 1 | 16.0 / 1 | 13.6 / 1 | 11.1 / 1 |
| hybrid: sigma_e absolute trees, lambda change trees | 0.0 / 1 | 5.4 / 1 | 10.0 / 1 | 13.6 / 1 | 11.1 / 1 | 0.0 / 1 | 5.4 / 1 | 10.0 / 1 | 13.6 / 1 | 11.1 / 1 |
