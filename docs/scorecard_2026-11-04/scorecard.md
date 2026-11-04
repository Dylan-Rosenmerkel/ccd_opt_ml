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

## (5) Ranking: shortlist each amplifier's settings from the traces

Per amplifier, its non-nominal settings (one per packet) are ranked by the prediction; the truth is the setting's effect relative to nominal (drift removed). Mean over the 3 amplifiers. Spearman: rank agreement. Top 5: how many of the 5 truly best are on the 5-setting shortlist, and how much worse (%) the best shortlisted setting is than the true best (regret). Top 10: is the single best on the 10-setting shortlist (fraction of amplifiers).

### trained amplifiers, best = lowest sigma_e (35 settings per amplifier)

| model | Spearman | top 5: truly best found | top 5: regret (%) | top 10: regret (%) | top 10: best caught |
|---|---|---|---|---|---|
| random | 0.00 | 0.7 of 5 | 4.0 | 2.8 | 0.29 |
| no change | 0.00 | 0.6 of 5 | 4.1 | 2.8 | 0.31 |
| settings change: gp | 0.06 | 0.0 of 5 | 14.5 | 2.4 | 0.33 |
| settings change: trees | 0.04 | 0.7 of 5 | 1.6 | 1.6 | 0.33 |
| settings + amplifier: gp | 0.07 | 0.0 of 5 | 14.5 | 2.4 | 0.33 |
| settings + amplifier: trees | 0.04 | 0.7 of 5 | 1.6 | 1.6 | 0.33 |
| trace change: trees | 0.36 | 1.0 of 5 | 2.7 | 2.0 | 0.67 |
| trace change: gp | 0.16 | 0.0 of 5 | 6.7 | 2.8 | 0.33 |
| trace change: shared-encoder net | 0.22 | 0.3 of 5 | 4.8 | 3.8 | 0.00 |
| trace change: hand-picked bands + gp | 0.25 | 1.0 of 5 | 2.6 | 0.0 | 0.67 |
| absolute settings: gp | 0.47 | 1.0 of 5 | 2.3 | 1.7 | 0.67 |
| absolute settings: trees | 0.24 | 0.0 of 5 | 5.1 | 3.6 | 0.33 |
| absolute settings + amplifier: gp | 0.37 | 1.0 of 5 | 3.4 | 2.5 | 0.33 |
| absolute settings + amplifier: trees | 0.22 | 0.2 of 5 | 4.7 | 1.7 | 0.67 |
| absolute trace: physics | 0.45 | 0.7 of 5 | 2.8 | 0.0 | 1.00 |
| absolute trace: gp | 0.31 | 1.3 of 5 | 3.8 | 3.4 | 0.00 |
| absolute trace: trees | 0.66 | 1.7 of 5 | 2.1 | 2.1 | 0.67 |
| absolute trace: nn | 0.66 | 1.0 of 5 | 4.2 | 3.7 | 0.33 |
| absolute trace: cnn | 0.61 | 0.7 of 5 | 2.3 | 2.1 | 0.33 |
| absolute trace: pretrained cnn | 0.65 | 1.3 of 5 | 2.1 | 1.7 | 0.67 |
| absolute trace: deep kernel | 0.64 | 1.3 of 5 | 4.1 | 2.4 | 0.33 |
| hybrid: sigma_e absolute trees, lambda change trees | 0.66 | 1.7 of 5 | 2.1 | 2.1 | 0.67 |

### trained amplifiers, best = lowest lambda (35 settings per amplifier)

| model | Spearman | top 5: truly best found | top 5: regret (%) | top 10: regret (%) | top 10: best caught |
|---|---|---|---|---|---|
| random | 0.00 | 0.7 of 5 | 19.7 | 13.1 | 0.29 |
| no change | 0.00 | 0.7 of 5 | 19.0 | 13.6 | 0.24 |
| settings change: gp | 0.48 | 2.0 of 5 | 0.0 | 0.0 | 1.00 |
| settings change: trees | 0.58 | 1.0 of 5 | 18.2 | 0.0 | 1.00 |
| settings + amplifier: gp | 0.36 | 1.3 of 5 | 0.0 | 0.0 | 1.00 |
| settings + amplifier: trees | 0.58 | 0.7 of 5 | 19.5 | 0.0 | 1.00 |
| trace change: trees | 0.69 | 2.3 of 5 | 3.4 | 0.0 | 1.00 |
| trace change: gp | 0.55 | 2.0 of 5 | 14.9 | 14.9 | 0.00 |
| trace change: shared-encoder net | 0.50 | 1.0 of 5 | 21.6 | 14.9 | 0.00 |
| trace change: hand-picked bands + gp | 0.23 | 1.3 of 5 | 7.3 | 0.0 | 1.00 |
| absolute settings: gp | 0.31 | 2.0 of 5 | 14.9 | 12.4 | 0.00 |
| absolute settings: trees | 0.37 | 1.0 of 5 | 18.0 | 16.5 | 0.00 |
| absolute settings + amplifier: gp | 0.31 | 1.3 of 5 | 17.6 | 14.9 | 0.00 |
| absolute settings + amplifier: trees | 0.43 | 1.3 of 5 | 15.7 | 12.1 | 0.33 |
| absolute trace: physics | 0.53 | 1.7 of 5 | 8.7 | 0.0 | 1.00 |
| absolute trace: gp | 0.53 | 2.3 of 5 | 14.9 | 9.0 | 0.33 |
| absolute trace: trees | 0.54 | 1.0 of 5 | 15.8 | 13.2 | 0.33 |
| absolute trace: nn | 0.54 | 1.0 of 5 | 12.3 | 9.0 | 0.33 |
| absolute trace: cnn | 0.50 | 0.7 of 5 | 16.5 | 12.2 | 0.33 |
| absolute trace: pretrained cnn | 0.49 | 0.0 of 5 | 23.7 | 18.0 | 0.00 |
| absolute trace: deep kernel | 0.51 | 0.7 of 5 | 14.8 | 5.9 | 0.67 |
| hybrid: sigma_e absolute trees, lambda change trees | 0.69 | 2.3 of 5 | 3.4 | 0.0 | 1.00 |

### trained amplifiers, best = lowest sigma_e + lambda (35 settings per amplifier)

| model | Spearman | top 5: truly best found | top 5: regret (%) | top 10: regret (%) | top 10: best caught |
|---|---|---|---|---|---|
| random | 0.00 | 0.7 of 5 | 12.5 | 8.8 | 0.29 |
| no change | 0.00 | 0.7 of 5 | 13.3 | 8.7 | 0.29 |
| settings change: gp | 0.32 | 0.7 of 5 | 13.9 | 13.1 | 0.33 |
| settings change: trees | 0.13 | 0.3 of 5 | 14.3 | 0.3 | 0.67 |
| settings + amplifier: gp | 0.11 | 0.0 of 5 | 14.5 | 13.1 | 0.00 |
| settings + amplifier: trees | 0.13 | 0.3 of 5 | 14.3 | 0.7 | 0.67 |
| trace change: trees | 0.56 | 0.7 of 5 | 13.9 | 7.7 | 0.33 |
| trace change: gp | 0.40 | 0.0 of 5 | 14.6 | 13.2 | 0.00 |
| trace change: shared-encoder net | 0.23 | 0.7 of 5 | 12.0 | 12.0 | 0.00 |
| trace change: hand-picked bands + gp | -0.30 | 0.0 of 5 | 41.1 | 10.3 | 0.33 |
| absolute settings: gp | 0.30 | 1.3 of 5 | 0.5 | 0.0 | 1.00 |
| absolute settings: trees | 0.17 | 0.8 of 5 | 9.5 | 0.4 | 0.67 |
| absolute settings + amplifier: gp | 0.27 | 0.7 of 5 | 13.9 | 7.4 | 0.67 |
| absolute settings + amplifier: trees | 0.18 | 0.6 of 5 | 10.6 | 0.5 | 0.67 |
| absolute trace: physics | 0.48 | 0.7 of 5 | 15.0 | 4.2 | 0.33 |
| absolute trace: gp | 0.47 | 1.3 of 5 | 11.6 | 11.6 | 0.00 |
| absolute trace: trees | 0.63 | 1.3 of 5 | 4.9 | 0.0 | 1.00 |
| absolute trace: nn | 0.58 | 0.3 of 5 | 13.8 | 10.3 | 0.33 |
| absolute trace: cnn | 0.63 | 0.3 of 5 | 13.7 | 4.2 | 0.33 |
| absolute trace: pretrained cnn | 0.63 | 1.7 of 5 | 10.9 | 10.6 | 0.00 |
| absolute trace: deep kernel | 0.69 | 0.7 of 5 | 12.9 | 6.4 | 0.67 |
| hybrid: sigma_e absolute trees, lambda change trees | 0.60 | 1.0 of 5 | 12.5 | 7.0 | 0.33 |

### new amplifier, best = lowest sigma_e (35 settings per amplifier)

| model | Spearman | top 5: truly best found | top 5: regret (%) | top 10: regret (%) | top 10: best caught |
|---|---|---|---|---|---|
| random | 0.00 | 0.7 of 5 | 4.1 | 2.8 | 0.28 |
| no change | 0.00 | 0.6 of 5 | 4.0 | 2.8 | 0.30 |
| settings change: gp | 0.05 | 0.0 of 5 | 8.7 | 2.4 | 0.33 |
| settings change: trees | -0.01 | 0.7 of 5 | 1.6 | 1.6 | 0.33 |
| trace change: trees | 0.31 | 0.7 of 5 | 2.9 | 2.0 | 0.67 |
| trace change: gp | 0.20 | 0.0 of 5 | 5.4 | 2.5 | 0.33 |
| trace change: shared-encoder net | 0.03 | 0.0 of 5 | 5.3 | 2.4 | 0.33 |
| trace change: hand-picked bands + gp | 0.32 | 0.7 of 5 | 2.7 | 0.5 | 0.67 |
| absolute settings: gp | 0.35 | 1.0 of 5 | 2.0 | 2.0 | 0.67 |
| absolute settings: trees | 0.26 | 0.5 of 5 | 4.1 | 3.6 | 0.33 |
| absolute trace: physics | 0.10 | 0.0 of 5 | 5.9 | 3.9 | 0.00 |
| absolute trace: gp | -0.57 | 0.6 of 5 | 4.9 | 2.5 | 0.33 |
| absolute trace: trees | 0.57 | 0.3 of 5 | 4.1 | 3.8 | 0.33 |
| absolute trace: nn | 0.46 | 0.3 of 5 | 2.5 | 1.7 | 0.33 |
| absolute trace: cnn | 0.42 | 1.0 of 5 | 4.2 | 3.8 | 0.00 |
| absolute trace: pretrained cnn | 0.43 | 1.0 of 5 | 3.3 | 2.8 | 0.33 |
| absolute trace: deep kernel | 0.08 | 1.3 of 5 | 3.5 | 3.4 | 0.33 |
| hybrid: sigma_e absolute trees, lambda change trees | 0.57 | 0.3 of 5 | 4.1 | 3.8 | 0.33 |

### new amplifier, best = lowest lambda (35 settings per amplifier)

| model | Spearman | top 5: truly best found | top 5: regret (%) | top 10: regret (%) | top 10: best caught |
|---|---|---|---|---|---|
| random | 0.00 | 0.7 of 5 | 19.7 | 13.5 | 0.28 |
| no change | 0.00 | 0.8 of 5 | 19.5 | 13.6 | 0.27 |
| settings change: gp | 0.42 | 1.7 of 5 | 0.0 | 0.0 | 1.00 |
| settings change: trees | 0.52 | 0.0 of 5 | 25.3 | 9.3 | 0.33 |
| trace change: trees | 0.69 | 2.3 of 5 | 9.3 | 0.0 | 1.00 |
| trace change: gp | 0.30 | 1.7 of 5 | 14.9 | 11.5 | 0.33 |
| trace change: shared-encoder net | 0.32 | 1.3 of 5 | 11.8 | 9.3 | 0.33 |
| trace change: hand-picked bands + gp | 0.37 | 0.7 of 5 | 17.8 | 16.3 | 0.33 |
| absolute settings: gp | 0.24 | 2.0 of 5 | 9.3 | 3.4 | 0.67 |
| absolute settings: trees | 0.33 | 2.0 of 5 | 7.4 | 4.1 | 0.67 |
| absolute trace: physics | 0.28 | 1.3 of 5 | 9.3 | 9.3 | 0.33 |
| absolute trace: gp | -0.08 | 1.0 of 5 | 18.0 | 14.6 | 0.33 |
| absolute trace: trees | 0.48 | 1.3 of 5 | 15.5 | 6.8 | 0.33 |
| absolute trace: nn | 0.55 | 1.0 of 5 | 18.0 | 5.9 | 0.67 |
| absolute trace: cnn | 0.34 | 0.3 of 5 | 22.7 | 12.3 | 0.33 |
| absolute trace: pretrained cnn | 0.35 | 1.0 of 5 | 18.0 | 18.0 | 0.00 |
| absolute trace: deep kernel | 0.03 | 1.0 of 5 | 16.0 | 12.1 | 0.33 |
| hybrid: sigma_e absolute trees, lambda change trees | 0.69 | 2.3 of 5 | 9.3 | 0.0 | 1.00 |

### new amplifier, best = lowest sigma_e + lambda (35 settings per amplifier)

| model | Spearman | top 5: truly best found | top 5: regret (%) | top 10: regret (%) | top 10: best caught |
|---|---|---|---|---|---|
| random | 0.00 | 0.7 of 5 | 12.5 | 8.9 | 0.28 |
| no change | 0.00 | 0.7 of 5 | 12.8 | 8.0 | 0.38 |
| settings change: gp | 0.20 | 0.7 of 5 | 13.9 | 13.1 | 0.33 |
| settings change: trees | 0.15 | 0.0 of 5 | 14.7 | 13.9 | 0.00 |
| trace change: trees | 0.55 | 1.3 of 5 | 10.6 | 6.7 | 0.33 |
| trace change: gp | 0.28 | 0.3 of 5 | 15.0 | 11.6 | 0.00 |
| trace change: shared-encoder net | -0.02 | 1.0 of 5 | 18.4 | 15.2 | 0.00 |
| trace change: hand-picked bands + gp | 0.00 | 0.3 of 5 | 19.4 | 5.9 | 0.33 |
| absolute settings: gp | 0.24 | 1.3 of 5 | 0.0 | 0.0 | 1.00 |
| absolute settings: trees | 0.05 | 0.3 of 5 | 16.6 | 2.9 | 0.52 |
| absolute trace: physics | 0.30 | 0.7 of 5 | 18.2 | 10.6 | 0.00 |
| absolute trace: gp | -0.42 | 0.0 of 5 | 42.7 | 13.5 | 0.00 |
| absolute trace: trees | 0.53 | 2.0 of 5 | 0.3 | 0.0 | 1.00 |
| absolute trace: nn | 0.44 | 0.3 of 5 | 16.0 | 11.6 | 0.00 |
| absolute trace: cnn | 0.47 | 1.7 of 5 | 5.2 | 4.2 | 0.33 |
| absolute trace: pretrained cnn | 0.37 | 2.0 of 5 | 10.8 | 10.8 | 0.00 |
| absolute trace: deep kernel | -0.12 | 1.3 of 5 | 8.0 | 0.4 | 0.67 |
| hybrid: sigma_e absolute trees, lambda change trees | 0.50 | 1.3 of 5 | 6.7 | 6.7 | 0.33 |
