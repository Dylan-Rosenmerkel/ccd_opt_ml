# Level-1 scorecard

Board 101 ch0/1/3, 120 rows from 40 packets, 5 contiguous time blocks. Trace models never see the settings. vs settings: the cell's settings-only reference (absolute: `settings + amplifier: gp` / `settings: gp`; relative: `settings + amplifier: gp` / `settings change: gp`), packet bootstrap 95 %: better / same / worse.

Label noise (floor): absolute sigma_e 0.8 %, lambda 2.0 %, gain 0.3 %; of a change sigma_e 1.9 %, lambda 12.0 %, gain 0.8 %.

## (1) Absolute, trained amplifiers: typical error (%)

| model | sigma_e | lambda | gain |
|---|---|---|---|
| settings: gp | 12.7 (worse) | 40.3 (worse) | 2.5 (worse) |
| settings: trees | 14.8 (worse) | 42.0 (worse) | 2.5 (worse) |
| settings + amplifier: gp | 10.7 (reference) | 37.0 (reference) | 1.4 (reference) |
| settings + amplifier: trees | 12.0 (same) | 34.8 (same) | 1.0 (better) |
| mean | 17.5 (worse) | 42.5 (worse) | 2.5 (worse) |
| amplifier mean | 15.2 (worse) | 35.3 (same) | 1.0 (better) |
| trace: physics | 11.1 (same) | 28.0 (better) | 1.2 (same) |
| trace: gp | 13.2 (same) | 28.8 (better) | 0.9 (better) |
| trace: trees | 10.1 (same) | 28.4 (better) | 1.0 (better) |
| trace: nn | 10.2 (same) | 29.4 (better) | 1.0 (better) |
| trace: cnn | 10.4 (same) | 28.5 (better) | 1.1 (better) |
| trace: pretrained cnn | 10.5 (same) | 28.6 (better) | 1.0 (better) |
| trace: deep kernel | 12.0 (same) | 29.1 (better) | 1.2 (same) |

## (2) Absolute, new amplifier: typical error (%)

| model | sigma_e | lambda | gain |
|---|---|---|---|
| settings: gp | 15.7 (reference) | 49.0 (reference) | 3.5 (reference) |
| settings: trees | 17.5 (worse) | 48.5 (same) | 3.5 (worse) |
| mean | 20.1 (worse) | 50.1 (same) | 3.5 (same) |
| trace: physics | 14.7 (same) | 55.5 (worse) | 3.5 (same) |
| trace: gp | 19.7 (worse) | 50.0 (same) | 3.5 (same) |
| trace: trees | 16.4 (same) | 49.3 (same) | 4.5 (worse) |
| trace: nn | 21.6 (worse) | 56.4 (worse) | 3.7 (worse) |
| trace: cnn | 24.3 (worse) | 41.2 (better) | 3.4 (same) |
| trace: pretrained cnn | 18.5 (worse) | 50.3 (same) | 3.4 (same) |
| trace: deep kernel | 23.3 (worse) | 89.7 (worse) | 3.2 (better) |

## (3) Relative, trained amplifiers, within packet: direction right on clear changes / size error (%)

clear changes: sigma_e 49 of 105, lambda 42 of 105, gain 22 of 105; true change RMS: sigma_e 17.4 %, lambda 34.2 %, gain 1.0 %; clear changes that go UP (= the score of always saying up): sigma_e 96 %, lambda 76 %, gain 45 %

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| no change | 50 % (same) | 17.4 (same) | 50 % (worse) | 34.2 (same) | 50 % (worse) | 1.0 (same) |
| settings change: gp | 31 % (same) | 18.5 (same) | 81 % (same) | 38.5 (better) | 68 % (same) | 0.9 (better) |
| settings change: trees | 35 % (same) | 21.4 (same) | 88 % (same) | 40.4 (same) | 95 % (same) | 0.8 (better) |
| settings + amplifier: gp | 33 % (reference) | 18.4 (reference) | 76 % (reference) | 40.1 (reference) | 86 % (reference) | 1.1 (reference) |
| settings + amplifier: trees | 37 % (same) | 21.7 (same) | 88 % (same) | 40.4 (same) | 95 % (same) | 0.9 (better) |
| trace change: trees | 76 % (better) | 11.6 (better) | 93 % (same) | 35.7 (same) | 86 % (same) | 0.9 (same) |
| trace change: gp | 53 % (better) | 26.0 (same) | 81 % (same) | 37.0 (same) | 64 % (same) | 1.1 (same) |
| trace change: shared-encoder net | 57 % (better) | 28.5 (same) | 86 % (same) | 43.1 (same) | 86 % (same) | 2.8 (worse) |
| trace change: hand-picked bands + gp | 73 % (better) | 15.0 (same) | 83 % (same) | 34.2 (same) | 86 % (same) | 1.0 (same) |

## (3) Relative, trained amplifiers, n -> n+1: direction right on clear changes / size error (%)

clear changes: sigma_e 50 of 105, lambda 61 of 105, gain 36 of 105; true change RMS: sigma_e 15.1 %, lambda 46.7 %, gain 1.5 %; clear changes that go UP (= the score of always saying up): sigma_e 52 %, lambda 51 %, gain 50 %

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| no change | 50 % (same) | 15.1 (same) | 50 % (same) | 46.7 (same) | 50 % (same) | 1.5 (same) |
| settings change: gp | 70 % (same) | 13.8 (same) | 79 % (same) | 52.2 (same) | 56 % (same) | 1.3 (same) |
| settings change: trees | 70 % (same) | 16.9 (same) | 89 % (same) | 52.9 (same) | 94 % (same) | 0.9 (same) |
| settings + amplifier: gp | 72 % (reference) | 13.5 (reference) | 79 % (reference) | 52.4 (reference) | 78 % (reference) | 1.4 (reference) |
| settings + amplifier: trees | 72 % (same) | 17.5 (same) | 89 % (same) | 52.8 (same) | 89 % (same) | 0.9 (same) |
| trace change: trees | 78 % (same) | 11.1 (same) | 89 % (same) | 44.6 (same) | 86 % (same) | 1.2 (same) |
| trace change: gp | 68 % (same) | 24.8 (same) | 74 % (same) | 43.6 (same) | 61 % (same) | 1.3 (same) |
| trace change: shared-encoder net | 64 % (same) | 34.2 (same) | 85 % (same) | 52.8 (same) | 72 % (same) | 3.9 (worse) |
| trace change: hand-picked bands + gp | 82 % (same) | 11.5 (same) | 77 % (same) | 46.7 (same) | 78 % (same) | 1.4 (same) |
| absolute settings: gp, differenced | 61 % (same) | 10.5 (same) | 79 % (same) | 40.3 (same) | 44 % (same) | 1.5 (same) |
| absolute settings: trees, differenced | 65 % (same) | 13.4 (same) | 83 % (same) | 40.3 (same) | 67 % (same) | 1.4 (same) |
| absolute settings + amplifier: gp, differenced | 70 % (same) | 13.0 (same) | 67 % (same) | 46.1 (same) | 61 % (same) | 1.7 (same) |
| absolute settings + amplifier: trees, differenced | 67 % (same) | 13.5 (same) | 83 % (same) | 40.9 (same) | 64 % (same) | 1.4 (same) |
| absolute trace: physics, differenced | 74 % (same) | 12.8 (same) | 90 % (same) | 42.5 (same) | 69 % (same) | 1.8 (same) |
| absolute trace: gp, differenced | 64 % (same) | 14.5 (same) | 75 % (same) | 45.5 (same) | 100 % (same) | 1.3 (same) |
| absolute trace: trees, differenced | 86 % (same) | 10.0 (same) | 85 % (same) | 40.5 (same) | 81 % (same) | 1.5 (same) |
| absolute trace: nn, differenced | 82 % (same) | 12.9 (same) | 77 % (same) | 42.1 (same) | 81 % (same) | 1.5 (same) |
| absolute trace: cnn, differenced | 82 % (same) | 13.4 (same) | 87 % (same) | 40.7 (same) | 56 % (same) | 1.6 (same) |
| absolute trace: pretrained cnn, differenced | 92 % (same) | 11.4 (same) | 89 % (same) | 39.3 (same) | 83 % (same) | 1.3 (same) |
| absolute trace: deep kernel, differenced | 76 % (same) | 12.5 (same) | 87 % (same) | 43.8 (same) | 44 % (same) | 2.0 (same) |

## (3) Relative, trained amplifiers, n -> n+2: direction right on clear changes / size error (%)

clear changes: sigma_e 51 of 90, lambda 53 of 90, gain 28 of 90; true change RMS: sigma_e 19.1 %, lambda 42.2 %, gain 1.3 %; clear changes that go UP (= the score of always saying up): sigma_e 49 %, lambda 40 %, gain 36 %

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| no change | 50 % (same) | 19.1 (same) | 50 % (same) | 42.2 (same) | 50 % (same) | 1.3 (same) |
| settings change: gp | 57 % (same) | 19.0 (same) | 72 % (same) | 52.0 (same) | 71 % (same) | 1.2 (same) |
| settings change: trees | 67 % (same) | 20.2 (same) | 79 % (same) | 53.4 (same) | 79 % (same) | 1.0 (better) |
| settings + amplifier: gp | 57 % (reference) | 19.0 (reference) | 66 % (reference) | 54.2 (reference) | 71 % (reference) | 1.4 (reference) |
| settings + amplifier: trees | 69 % (same) | 20.4 (same) | 85 % (same) | 53.6 (same) | 79 % (same) | 1.0 (better) |
| trace change: trees | 76 % (same) | 13.5 (same) | 87 % (same) | 45.2 (same) | 82 % (same) | 1.2 (same) |
| trace change: gp | 73 % (same) | 22.0 (same) | 74 % (same) | 40.0 (same) | 61 % (same) | 1.2 (same) |
| trace change: shared-encoder net | 69 % (same) | 34.3 (same) | 75 % (same) | 56.0 (same) | 75 % (same) | 3.6 (worse) |
| trace change: hand-picked bands + gp | 71 % (same) | 20.7 (same) | 51 % (same) | 42.2 (same) | 82 % (same) | 1.3 (same) |
| absolute settings: gp, differenced | 83 % (same) | 10.4 (same) | 80 % (same) | 35.3 (same) | 71 % (same) | 1.3 (same) |
| absolute settings: trees, differenced | 74 % (same) | 14.1 (same) | 83 % (same) | 33.8 (same) | 54 % (same) | 1.2 (same) |
| absolute settings + amplifier: gp, differenced | 80 % (same) | 12.0 (same) | 68 % (same) | 36.1 (same) | 61 % (same) | 1.6 (same) |
| absolute settings + amplifier: trees, differenced | 75 % (same) | 14.2 (same) | 83 % (same) | 34.3 (same) | 66 % (same) | 1.2 (same) |
| absolute trace: physics, differenced | 78 % (same) | 12.8 (same) | 91 % (same) | 33.5 (same) | 75 % (same) | 1.5 (same) |
| absolute trace: gp, differenced | 73 % (same) | 14.6 (same) | 81 % (same) | 34.8 (same) | 89 % (same) | 1.2 (same) |
| absolute trace: trees, differenced | 78 % (same) | 12.4 (same) | 85 % (same) | 32.5 (same) | 75 % (same) | 1.3 (same) |
| absolute trace: nn, differenced | 75 % (same) | 14.8 (same) | 81 % (same) | 35.6 (same) | 75 % (same) | 1.2 (same) |
| absolute trace: cnn, differenced | 86 % (same) | 14.2 (same) | 85 % (same) | 33.3 (same) | 64 % (same) | 1.4 (same) |
| absolute trace: pretrained cnn, differenced | 90 % (same) | 12.8 (same) | 77 % (same) | 33.4 (same) | 82 % (same) | 1.2 (same) |
| absolute trace: deep kernel, differenced | 80 % (same) | 14.6 (same) | 87 % (same) | 39.7 (same) | 68 % (same) | 1.6 (same) |

## (4) Relative, new amplifier, within packet: direction right on clear changes / size error (%)

clear changes: sigma_e 49 of 105, lambda 42 of 105, gain 22 of 105; true change RMS: sigma_e 17.4 %, lambda 34.2 %, gain 1.0 %; clear changes that go UP (= the score of always saying up): sigma_e 96 %, lambda 76 %, gain 45 %

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| no change | 50 % (same) | 17.4 (same) | 50 % (worse) | 34.2 (same) | 50 % (same) | 1.0 (same) |
| settings change: gp | 45 % (reference) | 18.3 (reference) | 79 % (reference) | 39.0 (reference) | 77 % (reference) | 1.0 (reference) |
| settings change: trees | 37 % (same) | 22.3 (worse) | 86 % (same) | 42.5 (same) | 95 % (same) | 0.9 (same) |
| trace change: trees | 65 % (better) | 12.6 (better) | 93 % (same) | 31.6 (same) | 91 % (same) | 1.0 (same) |
| trace change: gp | 59 % (same) | 21.9 (same) | 64 % (same) | 43.2 (same) | 73 % (same) | 1.3 (worse) |
| trace change: shared-encoder net | 47 % (same) | 24.9 (worse) | 60 % (same) | 51.1 (same) | 64 % (same) | 2.4 (worse) |
| trace change: hand-picked bands + gp | 76 % (better) | 15.4 (same) | 81 % (same) | 33.3 (same) | 86 % (same) | 1.4 (worse) |

## (4) Relative, new amplifier, n -> n+1: direction right on clear changes / size error (%)

clear changes: sigma_e 50 of 105, lambda 61 of 105, gain 36 of 105; true change RMS: sigma_e 15.1 %, lambda 46.7 %, gain 1.5 %; clear changes that go UP (= the score of always saying up): sigma_e 52 %, lambda 51 %, gain 50 %

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| no change | 50 % (same) | 15.1 (same) | 50 % (worse) | 46.7 (same) | 50 % (same) | 1.5 (same) |
| settings change: gp | 72 % (reference) | 14.0 (reference) | 82 % (reference) | 51.2 (reference) | 69 % (reference) | 1.4 (reference) |
| settings change: trees | 72 % (same) | 18.6 (same) | 85 % (same) | 55.9 (same) | 94 % (same) | 1.0 (same) |
| trace change: trees | 72 % (same) | 12.9 (same) | 82 % (same) | 42.9 (same) | 89 % (same) | 1.2 (same) |
| trace change: gp | 60 % (same) | 23.4 (same) | 64 % (same) | 58.0 (same) | 58 % (same) | 1.5 (same) |
| trace change: shared-encoder net | 62 % (same) | 24.6 (same) | 74 % (same) | 56.3 (same) | 61 % (same) | 3.3 (worse) |
| trace change: hand-picked bands + gp | 74 % (same) | 12.6 (same) | 79 % (same) | 42.9 (same) | 69 % (same) | 2.0 (same) |
| absolute settings: gp, differenced | 65 % (same) | 10.7 (same) | 85 % (same) | 41.9 (same) | 53 % (same) | 1.5 (same) |
| absolute settings: trees, differenced | 51 % (same) | 13.2 (same) | 73 % (same) | 42.8 (same) | 47 % (same) | 1.5 (same) |
| absolute trace: physics, differenced | 68 % (same) | 14.0 (same) | 84 % (same) | 43.1 (same) | 72 % (same) | 1.5 (same) |
| absolute trace: gp, differenced | 49 % (same) | 15.2 (same) | 66 % (same) | 46.7 (same) | 44 % (same) | 1.6 (same) |
| absolute trace: trees, differenced | 78 % (same) | 10.9 (same) | 90 % (same) | 39.4 (same) | 86 % (same) | 1.4 (same) |
| absolute trace: nn, differenced | 82 % (same) | 13.4 (same) | 80 % (same) | 42.1 (same) | 75 % (same) | 1.5 (same) |
| absolute trace: cnn, differenced | 78 % (same) | 12.3 (same) | 82 % (same) | 41.7 (same) | 56 % (same) | 1.6 (same) |
| absolute trace: pretrained cnn, differenced | 86 % (same) | 10.7 (same) | 80 % (same) | 43.9 (same) | 72 % (same) | 1.6 (same) |
| absolute trace: deep kernel, differenced | 72 % (same) | 13.4 (same) | 80 % (same) | 47.1 (same) | 57 % (same) | 1.5 (same) |

## (4) Relative, new amplifier, n -> n+2: direction right on clear changes / size error (%)

clear changes: sigma_e 51 of 90, lambda 53 of 90, gain 28 of 90; true change RMS: sigma_e 19.1 %, lambda 42.2 %, gain 1.3 %; clear changes that go UP (= the score of always saying up): sigma_e 49 %, lambda 40 %, gain 36 %

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| no change | 50 % (same) | 19.1 (same) | 50 % (same) | 42.2 (same) | 50 % (same) | 1.3 (same) |
| settings change: gp | 65 % (reference) | 18.5 (reference) | 66 % (reference) | 54.2 (reference) | 79 % (reference) | 1.2 (reference) |
| settings change: trees | 67 % (same) | 21.6 (same) | 79 % (same) | 55.0 (same) | 79 % (same) | 1.1 (same) |
| trace change: trees | 69 % (same) | 14.1 (same) | 94 % (same) | 37.3 (same) | 82 % (same) | 1.2 (same) |
| trace change: gp | 71 % (same) | 24.8 (same) | 66 % (same) | 50.8 (same) | 64 % (same) | 1.5 (same) |
| trace change: shared-encoder net | 63 % (same) | 27.9 (same) | 74 % (same) | 56.3 (same) | 79 % (same) | 3.1 (worse) |
| trace change: hand-picked bands + gp | 65 % (same) | 20.7 (same) | 79 % (same) | 38.7 (same) | 82 % (same) | 1.5 (same) |
| absolute settings: gp, differenced | 84 % (same) | 11.3 (same) | 78 % (same) | 37.0 (same) | 79 % (same) | 1.2 (same) |
| absolute settings: trees, differenced | 66 % (same) | 14.2 (same) | 71 % (same) | 36.0 (same) | 59 % (same) | 1.2 (same) |
| absolute trace: physics, differenced | 69 % (same) | 14.4 (same) | 81 % (same) | 35.5 (same) | 71 % (same) | 1.2 (same) |
| absolute trace: gp, differenced | 46 % (same) | 19.2 (same) | 70 % (same) | 42.0 (same) | 29 % (worse) | 1.3 (same) |
| absolute trace: trees, differenced | 80 % (same) | 13.5 (same) | 87 % (same) | 33.0 (same) | 71 % (same) | 1.2 (same) |
| absolute trace: nn, differenced | 82 % (same) | 14.6 (same) | 81 % (same) | 35.8 (same) | 79 % (same) | 1.3 (same) |
| absolute trace: cnn, differenced | 82 % (same) | 14.1 (same) | 77 % (same) | 35.3 (same) | 68 % (same) | 1.3 (same) |
| absolute trace: pretrained cnn, differenced | 86 % (same) | 12.5 (same) | 75 % (same) | 36.9 (same) | 75 % (same) | 1.4 (same) |
| absolute trace: deep kernel, differenced | 69 % (same) | 17.2 (same) | 74 % (same) | 43.6 (same) | 70 % (same) | 1.2 (same) |

## (5) Ranking: shortlist the settings from the traces, joint objective J(w) = w dlog sigma_e + (1-w) dlog lambda

Each setting (non-nominal packet) applies to the whole board, so its objective is the mean over the 3 amplifiers (predicted and true; truth = the setting's effect relative to nominal, drift removed). Weight w:(1-w) of sigma_e:lambda. Shortlist rules (lvl1/shortlist.py): weighted = by J(w); pareto = the predicted Pareto front first. Cells: top-5 regret (%) = how much worse the best of the 5 shortlisted settings truly is than the truly best setting / whether the truly best is on the 10-setting shortlist (1 = yes). The 3 amplifiers share their settings, so each column is ONE trial: read the pattern, not single numbers.

### trained amplifiers (35 settings)

| model | 1:0 weighted | 0.75:0.25 weighted | 0.5:0.5 weighted | 0.25:0.75 weighted | 0:1 weighted | 1:0 pareto | 0.75:0.25 pareto | 0.5:0.5 pareto | 0.25:0.75 pareto | 0:1 pareto |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 2.9 / 0.29 | 3.1 / 0.29 | 6.3 / 0.28 | 8.0 / 0.28 | 21.2 / 0.26 | 2.9 / 0.27 | 3.1 / 0.28 | 6.4 / 0.29 | 7.8 / 0.27 | 20.5 / 0.31 |
| no change | 3.3 / 0 | 3.5 / 0 | 6.6 / 0 | 8.0 / 0 | 18.5 / 0 | 3.5 / 0 | 2.6 / 0 | 6.4 / 0 | 7.8 / 0 | 20.3 / 0 |
| settings change: gp | 14.5 / 0 | 2.7 / 0 | 7.1 / 0 | 6.5 / 1 | 0.0 / 1 | 2.9 / 0 | 2.7 / 0 | 7.1 / 0 | 6.5 / 1 | 18.2 / 1 |
| settings change: trees | 0.0 / 1 | 2.7 / 0 | 7.1 / 1 | 6.5 / 0 | 18.2 / 1 | 9.7 / 1 | 2.7 / 0 | 7.1 / 0 | 6.5 / 0 | 18.2 / 0 |
| settings + amplifier: gp | 14.5 / 0 | 2.7 / 0 | 7.1 / 0 | 6.5 / 1 | 0.0 / 1 | 2.9 / 0 | 2.7 / 0 | 7.1 / 0 | 12.7 / 1 | 32.1 / 1 |
| settings + amplifier: trees | 0.0 / 1 | 2.7 / 0 | 7.1 / 1 | 6.5 / 0 | 18.2 / 1 | 9.7 / 1 | 2.7 / 0 | 7.1 / 1 | 6.5 / 0 | 18.2 / 0 |
| trace change: trees | 2.8 / 0 | 2.7 / 0 | 7.1 / 0 | 6.5 / 1 | 0.0 / 1 | 2.8 / 0 | 2.7 / 0 | 7.1 / 0 | 6.5 / 0 | 18.0 / 1 |
| trace change: gp | 6.7 / 0 | 2.7 / 0 | 7.1 / 0 | 11.2 / 0 | 18.0 / 0 | 3.6 / 0 | 2.7 / 0 | 7.1 / 0 | 10.7 / 0 | 28.7 / 0 |
| trace change: shared-encoder net | 3.5 / 0 | 2.7 / 0 | 7.1 / 0 | 6.5 / 0 | 18.2 / 0 | 2.9 / 0 | 2.7 / 0 | 7.1 / 0 | 6.5 / 0 | 18.2 / 0 |
| trace change: hand-picked bands + gp | 0.0 / 1 | 11.7 / 0 | 21.4 / 0 | 32.3 / 0 | 28.3 / 1 | 2.7 / 1 | 2.3 / 0 | 6.5 / 0 | 12.0 / 0 | 28.3 / 1 |
| absolute settings: gp | 2.8 / 0 | 0.0 / 1 | 0.0 / 1 | 1.2 / 0 | 18.0 / 0 | 2.8 / 0 | 3.2 / 1 | 7.1 / 1 | 11.2 / 0 | 18.0 / 0 |
| absolute settings: trees | 3.5 / 0 | 3.2 / 1 | 2.8 / 1 | 11.2 / 0 | 18.0 / 0 | 3.5 / 0 | 3.5 / 0 | 18.2 / 0 | 11.2 / 0 | 18.0 / 0 |
| absolute settings + amplifier: gp | 2.8 / 0 | 2.3 / 1 | 6.5 / 1 | 6.5 / 0 | 18.0 / 0 | 2.8 / 0 | 2.3 / 1 | 6.5 / 1 | 6.5 / 0 | 18.0 / 0 |
| absolute settings + amplifier: trees | 3.5 / 0 | 3.3 / 0 | 4.8 / 1 | 11.2 / 0 | 18.0 / 0 | 3.5 / 0 | 3.5 / 0 | 18.2 / 1 | 11.2 / 0 | 18.0 / 0 |
| absolute trace: physics | 2.8 / 0 | 3.5 / 1 | 6.1 / 0 | 10.3 / 0 | 0.0 / 1 | 3.5 / 0 | 3.5 / 0 | 8.1 / 0 | 11.2 / 0 | 18.0 / 1 |
| absolute trace: gp | 2.7 / 0 | 2.7 / 0 | 6.5 / 0 | 6.5 / 0 | 18.0 / 0 | 3.6 / 0 | 3.3 / 0 | 6.5 / 0 | 6.5 / 0 | 18.2 / 0 |
| absolute trace: trees | 2.4 / 0 | 2.6 / 1 | 7.1 / 1 | 12.7 / 0 | 18.0 / 0 | 2.4 / 0 | 2.6 / 1 | 6.5 / 1 | 10.7 / 0 | 18.0 / 0 |
| absolute trace: nn | 2.9 / 0 | 2.7 / 0 | 6.1 / 0 | 10.3 / 0 | 18.0 / 0 | 2.9 / 0 | 2.7 / 0 | 6.1 / 0 | 11.2 / 0 | 25.3 / 0 |
| absolute trace: cnn | 2.9 / 0 | 2.7 / 0 | 6.1 / 0 | 10.3 / 0 | 25.3 / 1 | 2.9 / 0 | 2.7 / 0 | 7.1 / 0 | 12.7 / 0 | 25.3 / 1 |
| absolute trace: pretrained cnn | 0.0 / 1 | 2.7 / 0 | 6.1 / 0 | 10.3 / 0 | 25.3 / 0 | 2.9 / 1 | 2.7 / 0 | 6.1 / 0 | 10.3 / 0 | 25.3 / 0 |
| absolute trace: deep kernel | 2.8 / 1 | 2.7 / 0 | 7.1 / 0 | 12.7 / 0 | 30.8 / 1 | 2.8 / 1 | 2.7 / 0 | 7.1 / 0 | 12.7 / 0 | 30.8 / 1 |
| hybrid: sigma_e absolute trees, lambda change trees | 2.4 / 0 | 2.6 / 0 | 6.5 / 1 | 6.5 / 1 | 0.0 / 1 | 3.5 / 0 | 3.5 / 0 | 6.5 / 0 | 6.5 / 1 | 18.2 / 1 |

### new amplifier (35 settings)

| model | 1:0 weighted | 0.75:0.25 weighted | 0.5:0.5 weighted | 0.25:0.75 weighted | 0:1 weighted | 1:0 pareto | 0.75:0.25 pareto | 0.5:0.5 pareto | 0.25:0.75 pareto | 0:1 pareto |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 3.0 / 0.27 | 3.1 / 0.27 | 6.3 / 0.27 | 7.8 / 0.28 | 20.5 / 0.27 | 2.9 / 0.28 | 3.1 / 0.28 | 6.4 / 0.29 | 7.8 / 0.29 | 20.2 / 0.28 |
| no change | 2.7 / 0 | 3.6 / 0 | 6.3 / 0 | 7.7 / 0 | 19.4 / 0 | 2.9 / 0 | 3.2 / 0 | 6.1 / 0 | 8.7 / 0 | 18.9 / 0 |
| settings change: gp | 14.5 / 0 | 2.7 / 0 | 7.1 / 0 | 6.5 / 1 | 0.0 / 1 | 2.9 / 0 | 2.7 / 0 | 7.1 / 0 | 12.7 / 1 | 32.1 / 1 |
| settings change: trees | 0.0 / 1 | 2.7 / 0 | 7.1 / 0 | 12.7 / 0 | 25.3 / 0 | 6.7 / 1 | 2.7 / 0 | 7.1 / 0 | 12.7 / 0 | 30.8 / 0 |
| trace change: trees | 2.8 / 0 | 3.2 / 0 | 6.1 / 0 | 6.5 / 1 | 0.0 / 1 | 2.8 / 0 | 3.2 / 0 | 6.1 / 0 | 6.5 / 1 | 18.0 / 1 |
| trace change: gp | 3.6 / 0 | 3.3 / 0 | 7.6 / 0 | 6.5 / 0 | 18.4 / 0 | 3.6 / 0 | 3.3 / 0 | 7.6 / 0 | 6.5 / 0 | 18.2 / 0 |
| trace change: shared-encoder net | 2.8 / 0 | 11.7 / 0 | 13.6 / 0 | 11.2 / 0 | 18.0 / 1 | 9.7 / 0 | 13.1 / 0 | 13.6 / 0 | 11.2 / 0 | 18.0 / 1 |
| trace change: hand-picked bands + gp | 2.8 / 1 | 3.2 / 1 | 6.5 / 0 | 10.7 / 0 | 18.0 / 1 | 4.0 / 0 | 0.0 / 1 | 6.5 / 1 | 10.7 / 0 | 18.0 / 0 |
| absolute settings: gp | 2.4 / 1 | 0.0 / 1 | 0.0 / 1 | 11.2 / 0 | 18.0 / 0 | 2.8 / 0 | 3.2 / 1 | 7.1 / 1 | 11.2 / 0 | 18.0 / 0 |
| absolute settings: trees | 3.0 / 0 | 2.9 / 1 | 8.1 / 1 | 11.2 / 0 | 10.1 / 1 | 3.5 / 0 | 3.5 / 1 | 8.1 / 1 | 11.2 / 0 | 18.0 / 1 |
| absolute trace: physics | 8.9 / 0 | 0.0 / 1 | 6.1 / 1 | 10.3 / 0 | 18.0 / 0 | 3.5 / 0 | 0.0 / 1 | 0.0 / 1 | 1.2 / 0 | 16.2 / 0 |
| absolute trace: gp | 2.8 / 0 | 18.8 / 0 | 18.2 / 1 | 11.2 / 0 | 18.0 / 0 | 2.8 / 0 | 18.8 / 0 | 18.2 / 1 | 11.2 / 0 | 18.0 / 0 |
| absolute trace: trees | 2.9 / 0 | 0.0 / 1 | 0.0 / 1 | 1.2 / 0 | 16.2 / 0 | 4.0 / 0 | 0.0 / 1 | 0.0 / 1 | 1.2 / 0 | 16.2 / 0 |
| absolute trace: nn | 2.9 / 0 | 2.7 / 0 | 7.1 / 0 | 11.2 / 0 | 18.0 / 1 | 2.9 / 0 | 2.7 / 0 | 7.1 / 0 | 12.7 / 0 | 30.8 / 1 |
| absolute trace: cnn | 2.9 / 0 | 2.7 / 0 | 7.1 / 0 | 12.7 / 0 | 25.3 / 0 | 2.9 / 0 | 2.7 / 0 | 7.1 / 0 | 12.7 / 0 | 25.3 / 0 |
| absolute trace: pretrained cnn | 0.0 / 1 | 2.7 / 0 | 6.1 / 0 | 10.3 / 0 | 18.0 / 0 | 0.0 / 1 | 3.2 / 0 | 6.1 / 0 | 10.3 / 0 | 25.3 / 0 |
| absolute trace: deep kernel | 2.4 / 0 | 2.3 / 0 | 0.0 / 1 | 1.2 / 0 | 16.2 / 0 | 0.0 / 1 | 2.6 / 1 | 0.0 / 1 | 1.2 / 0 | 16.2 / 0 |
| hybrid: sigma_e absolute trees, lambda change trees | 2.9 / 0 | 2.7 / 1 | 6.1 / 1 | 0.0 / 1 | 0.0 / 1 | 3.5 / 0 | 3.2 / 0 | 6.1 / 0 | 6.5 / 1 | 18.2 / 1 |
