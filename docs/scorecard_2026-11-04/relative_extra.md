# Relative models: a combined model, and every kind of change

Direction right on clear changes (%, number of clear changes) and size error (RMS %) per pair kind; board 101 ch0/1/3, same blocks and splits as the scorecard. Within-packet pairs are nominal -> one kind of change (their sigma_e changes almost all go up: the base rate is given); between-packet pairs (n -> n+1 and n -> n+2) compare two settings directly.

## trained amplifiers

### all within-packet pairs (105 pairs; clear changes sigma_e 49 (96 % up), lambda 42 (76 % up), gain 22 (45 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 76 % | 11.6 | 93 % | 35.7 | 86 % | 0.9 |
| combined: A, B - A | 92 % | 11.3 | 95 % | 33.6 | 91 % | 0.9 |
| combined: A, B - A, hand-picked bands | 82 % | 12.1 | 83 % | 35.3 | 91 % | 0.9 |
| combined: A, B, B - A | 96 % | 10.8 | 90 % | 29.6 | 82 % | 1.0 |
| combined: A, B, B - A, hand-picked bands | 92 % | 11.0 | 88 % | 34.8 | 86 % | 1.0 |
| settings only (settings + amplifier: gp) | 33 % | 18.4 | 76 % | 40.1 | 86 % | 1.1 |

### all between-packet pairs (n+1, n+2) (231 pairs; clear changes sigma_e 120 (52 % up), lambda 126 (48 % up), gain 75 (49 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 77 % | 12.3 | 88 % | 44.9 | 84 % | 1.2 |
| combined: A, B - A | 77 % | 12.4 | 87 % | 39.7 | 84 % | 1.2 |
| combined: A, B - A, hand-picked bands | 76 % | 12.7 | 81 % | 45.7 | 86 % | 1.2 |
| combined: A, B, B - A | 80 % | 13.1 | 86 % | 37.5 | 83 % | 1.3 |
| combined: A, B, B - A, hand-picked bands | 77 % | 13.3 | 81 % | 46.4 | 86 % | 1.2 |
| settings only (settings + amplifier: gp) | 64 % | 16.3 | 73 % | 53.3 | 75 % | 1.4 |
| absolute trace: pretrained cnn, differenced | 89 % | 12.0 | 84 % | 37.7 | 77 % | 1.3 |
| absolute trace: trees, differenced | 79 % | 11.9 | 86 % | 38.5 | 73 % | 1.5 |

### nominal vs VSUB (6 pairs; clear changes sigma_e 2 (100 % up), lambda 2 (0 % up), gain 0 (nan % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 100 % | 4.9 | 50 % | 14.2 | - | 0.7 |
| combined: A, B - A | 50 % | 5.6 | 50 % | 14.9 | - | 0.7 |
| combined: A, B - A, hand-picked bands | 50 % | 5.3 | 0 % | 14.3 | - | 0.8 |
| combined: A, B, B - A | 100 % | 4.8 | 50 % | 14.8 | - | 0.8 |
| combined: A, B, B - A, hand-picked bands | 100 % | 4.1 | 50 % | 15.1 | - | 0.7 |
| settings only (settings + amplifier: gp) | 0 % | 5.6 | 0 % | 14.0 | - | 1.0 |

### nominal vs all at once (21 pairs; clear changes sigma_e 13 (100 % up), lambda 9 (100 % up), gain 11 (45 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 100 % | 6.8 | 100 % | 49.7 | 82 % | 0.8 |
| combined: A, B - A | 100 % | 10.7 | 100 % | 41.4 | 91 % | 0.8 |
| combined: A, B - A, hand-picked bands | 100 % | 11.4 | 100 % | 40.7 | 91 % | 0.8 |
| combined: A, B, B - A | 100 % | 10.7 | 100 % | 22.9 | 82 % | 0.8 |
| combined: A, B, B - A, hand-picked bands | 100 % | 10.2 | 100 % | 22.3 | 91 % | 0.8 |
| settings only (settings + amplifier: gp) | 8 % | 11.9 | 67 % | 20.2 | 82 % | 1.4 |

### nominal vs another packet's change (48 pairs; clear changes sigma_e 29 (55 % up), lambda 24 (42 % up), gain 17 (53 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 61 % | 16.0 | 96 % | 31.2 | 100 % | 0.8 |
| combined: A, B - A | 87 % | 13.5 | 88 % | 28.7 | 100 % | 0.8 |
| combined: A, B - A, hand-picked bands | 78 % | 14.7 | 79 % | 29.3 | 100 % | 0.8 |
| combined: A, B, B - A | 100 % | 12.5 | 96 % | 24.3 | 100 % | 0.9 |
| combined: A, B, B - A, hand-picked bands | 87 % | 14.1 | 100 % | 25.2 | 100 % | 0.9 |
| settings only (settings + amplifier: gp) | 48 % | 23.6 | 75 % | 29.9 | 80 % | 1.1 |
| absolute trace: pretrained cnn, differenced | 97 % | 15.5 | 96 % | 25.3 | 71 % | 1.1 |
| absolute trace: trees, differenced | 90 % | 13.6 | 88 % | 19.0 | 76 % | 1.2 |

### nominal vs clock rails (H, V, TG) (15 pairs; clear changes sigma_e 0 (nan % up), lambda 2 (100 % up), gain 0 (nan % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | - | 3.6 | 100 % | 11.1 | - | 0.5 |
| combined: A, B - A | - | 3.2 | 100 % | 11.0 | - | 0.6 |
| combined: A, B - A, hand-picked bands | - | 3.0 | 100 % | 11.5 | - | 0.5 |
| combined: A, B, B - A | - | 2.7 | 100 % | 11.7 | - | 0.6 |
| combined: A, B, B - A, hand-picked bands | - | 2.6 | 100 % | 11.6 | - | 0.5 |
| settings only (settings + amplifier: gp) | - | 4.4 | 0 % | 16.0 | - | 0.7 |

### nominal vs output stage (VDD, VR, RG, SW) (27 pairs; clear changes sigma_e 1 (100 % up), lambda 3 (100 % up), gain 9 (44 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 0 % | 3.4 | 33 % | 45.7 | 100 % | 1.1 |
| combined: A, B - A | 0 % | 2.9 | 67 % | 45.4 | 100 % | 1.1 |
| combined: A, B - A, hand-picked bands | 100 % | 2.2 | 67 % | 45.5 | 100 % | 1.1 |
| combined: A, B, B - A | 100 % | 2.3 | 33 % | 45.9 | 89 % | 1.3 |
| combined: A, B, B - A, hand-picked bands | 100 % | 2.2 | 33 % | 45.4 | 89 % | 1.2 |
| settings only (settings + amplifier: gp) | 100 % | 2.7 | 100 % | 65.4 | 100 % | 0.6 |

### nominal vs timing (36 pairs; clear changes sigma_e 33 (94 % up), lambda 26 (69 % up), gain 2 (50 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 67 % | 18.7 | 100 % | 24.8 | 50 % | 0.9 |
| combined: A, B - A | 94 % | 17.0 | 100 % | 25.6 | 50 % | 0.9 |
| combined: A, B - A, hand-picked bands | 76 % | 18.4 | 85 % | 31.9 | 50 % | 1.0 |
| combined: A, B, B - A | 94 % | 16.2 | 96 % | 24.1 | 50 % | 0.9 |
| combined: A, B, B - A, hand-picked bands | 88 % | 16.7 | 92 % | 40.0 | 50 % | 1.0 |
| settings only (settings + amplifier: gp) | 42 % | 29.8 | 88 % | 33.4 | 50 % | 1.3 |

### same kind: all at once (33 pairs; clear changes sigma_e 26 (54 % up), lambda 19 (63 % up), gain 21 (43 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 81 % | 11.3 | 68 % | 65.4 | 76 % | 1.8 |
| combined: A, B - A | 85 % | 11.9 | 100 % | 39.0 | 76 % | 1.8 |
| combined: A, B - A, hand-picked bands | 85 % | 11.5 | 100 % | 39.4 | 81 % | 1.7 |
| combined: A, B, B - A | 81 % | 10.0 | 100 % | 27.7 | 76 % | 1.7 |
| combined: A, B, B - A, hand-picked bands | 88 % | 10.1 | 100 % | 28.1 | 86 % | 1.6 |
| settings only (settings + amplifier: gp) | 54 % | 11.4 | 74 % | 28.7 | 81 % | 2.2 |
| absolute trace: pretrained cnn, differenced | 81 % | 11.3 | 95 % | 20.4 | 95 % | 1.6 |
| absolute trace: trees, differenced | 85 % | 10.1 | 89 % | 19.7 | 81 % | 2.1 |

### same kind: clock rails (H, V, TG) (12 pairs; clear changes sigma_e 0 (nan % up), lambda 2 (0 % up), gain 0 (nan % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | - | 2.7 | 100 % | 13.3 | - | 0.4 |
| combined: A, B - A | - | 4.4 | 100 % | 13.1 | - | 0.3 |
| combined: A, B - A, hand-picked bands | - | 4.3 | 100 % | 14.5 | - | 0.3 |
| combined: A, B, B - A | - | 4.0 | 100 % | 13.1 | - | 0.3 |
| combined: A, B, B - A, hand-picked bands | - | 4.1 | 100 % | 13.8 | - | 0.3 |
| settings only (settings + amplifier: gp) | - | 3.5 | 0 % | 23.4 | - | 0.4 |
| absolute trace: pretrained cnn, differenced | - | 2.8 | 50 % | 16.2 | - | 0.6 |
| absolute trace: trees, differenced | - | 4.8 | 100 % | 15.7 | - | 0.4 |

### same kind: output stage (VDD, VR, RG, SW) (36 pairs; clear changes sigma_e 2 (100 % up), lambda 9 (67 % up), gain 19 (42 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 0 % | 3.4 | 50 % | 67.8 | 100 % | 1.3 |
| combined: A, B - A | 0 % | 3.1 | 50 % | 69.5 | 100 % | 1.4 |
| combined: A, B - A, hand-picked bands | 100 % | 2.6 | 50 % | 67.4 | 100 % | 1.3 |
| combined: A, B, B - A | 100 % | 2.7 | 50 % | 66.8 | 100 % | 1.5 |
| combined: A, B, B - A, hand-picked bands | 100 % | 2.4 | 33 % | 66.8 | 100 % | 1.5 |
| settings only (settings + amplifier: gp) | 100 % | 2.7 | 100 % | 93.0 | 100 % | 0.3 |
| absolute trace: pretrained cnn, differenced | 100 % | 8.7 | 78 % | 69.1 | 68 % | 1.6 |
| absolute trace: trees, differenced | 100 % | 3.3 | 67 % | 72.7 | 74 % | 1.7 |

### same kind: timing (51 pairs; clear changes sigma_e 44 (50 % up), lambda 47 (43 % up), gain 13 (46 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 90 % | 17.0 | 95 % | 25.4 | 62 % | 1.3 |
| combined: A, B - A | 80 % | 18.9 | 88 % | 27.3 | 62 % | 1.3 |
| combined: A, B - A, hand-picked bands | 78 % | 19.4 | 73 % | 55.1 | 62 % | 1.4 |
| combined: A, B, B - A | 72 % | 22.5 | 80 % | 30.1 | 62 % | 1.4 |
| combined: A, B, B - A, hand-picked bands | 70 % | 22.0 | 66 % | 63.7 | 62 % | 1.4 |
| settings only (settings + amplifier: gp) | 85 % | 23.3 | 78 % | 40.8 | 38 % | 1.7 |
| absolute trace: pretrained cnn, differenced | 93 % | 15.0 | 83 % | 30.8 | 77 % | 1.3 |
| absolute trace: trees, differenced | 82 % | 17.5 | 85 % | 28.8 | 62 % | 1.3 |

### two different kinds of change (48 pairs; clear changes sigma_e 17 (35 % up), lambda 23 (48 % up), gain 5 (100 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 56 % | 6.6 | 90 % | 39.8 | 100 % | 0.8 |
| combined: A, B - A | 44 % | 6.1 | 85 % | 40.2 | 100 % | 0.9 |
| combined: A, B - A, hand-picked bands | 44 % | 6.0 | 85 % | 40.4 | 100 % | 0.9 |
| combined: A, B, B - A | 56 % | 5.7 | 85 % | 41.2 | 50 % | 0.9 |
| combined: A, B, B - A, hand-picked bands | 44 % | 5.2 | 85 % | 41.5 | 50 % | 0.9 |
| settings only (settings + amplifier: gp) | 56 % | 5.1 | 65 % | 69.7 | 50 % | 0.9 |
| absolute trace: pretrained cnn, differenced | 82 % | 8.1 | 74 % | 35.2 | 60 % | 1.2 |
| absolute trace: trees, differenced | 47 % | 8.9 | 87 % | 39.6 | 60 % | 1.3 |

## new amplifier

### all within-packet pairs (105 pairs; clear changes sigma_e 49 (96 % up), lambda 42 (76 % up), gain 22 (45 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 65 % | 12.6 | 93 % | 31.6 | 91 % | 1.0 |
| combined: A, B - A | 80 % | 12.1 | 74 % | 54.8 | 91 % | 0.9 |
| combined: A, B - A, hand-picked bands | 69 % | 12.4 | 74 % | 49.2 | 95 % | 0.9 |
| combined: A, B, B - A | 90 % | 11.9 | 88 % | 33.6 | 86 % | 1.0 |
| combined: A, B, B - A, hand-picked bands | 86 % | 12.2 | 88 % | 32.6 | 91 % | 1.0 |
| settings only (settings change: gp) | 45 % | 18.3 | 79 % | 39.0 | 77 % | 1.0 |

### all between-packet pairs (n+1, n+2) (231 pairs; clear changes sigma_e 120 (52 % up), lambda 126 (48 % up), gain 75 (49 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 70 % | 13.5 | 88 % | 40.4 | 86 % | 1.2 |
| combined: A, B - A | 65 % | 13.6 | 85 % | 53.0 | 86 % | 1.3 |
| combined: A, B - A, hand-picked bands | 65 % | 13.4 | 82 % | 50.9 | 88 % | 1.3 |
| combined: A, B, B - A | 75 % | 13.3 | 89 % | 40.6 | 88 % | 1.2 |
| combined: A, B, B - A, hand-picked bands | 69 % | 13.7 | 89 % | 40.5 | 89 % | 1.3 |
| settings only (settings change: gp) | 68 % | 16.3 | 75 % | 52.6 | 73 % | 1.3 |
| absolute trace: pretrained cnn, differenced | 83 % | 11.8 | 79 % | 41.9 | 69 % | 1.5 |
| absolute trace: trees, differenced | 76 % | 12.5 | 88 % | 39.2 | 75 % | 1.3 |

### nominal vs VSUB (6 pairs; clear changes sigma_e 2 (100 % up), lambda 2 (0 % up), gain 0 (nan % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 0 % | 5.5 | 100 % | 12.5 | - | 1.0 |
| combined: A, B - A | 50 % | 5.7 | 50 % | 13.4 | - | 0.7 |
| combined: A, B - A, hand-picked bands | 0 % | 6.0 | 100 % | 12.1 | - | 0.9 |
| combined: A, B, B - A | 50 % | 5.4 | 100 % | 12.7 | - | 0.8 |
| combined: A, B, B - A, hand-picked bands | 50 % | 5.6 | 100 % | 12.6 | - | 0.9 |
| settings only (settings change: gp) | 0 % | 5.6 | 50 % | 14.0 | - | 0.9 |

### nominal vs all at once (21 pairs; clear changes sigma_e 13 (100 % up), lambda 9 (100 % up), gain 11 (45 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 77 % | 10.2 | 100 % | 28.5 | 91 % | 0.8 |
| combined: A, B - A | 92 % | 11.1 | 67 % | 55.4 | 100 % | 0.8 |
| combined: A, B - A, hand-picked bands | 100 % | 11.4 | 78 % | 28.2 | 100 % | 0.8 |
| combined: A, B, B - A | 100 % | 13.8 | 67 % | 25.3 | 91 % | 0.8 |
| combined: A, B, B - A, hand-picked bands | 100 % | 14.2 | 89 % | 14.3 | 91 % | 0.9 |
| settings only (settings change: gp) | 69 % | 11.9 | 44 % | 20.1 | 73 % | 1.4 |

### nominal vs another packet's change (48 pairs; clear changes sigma_e 29 (55 % up), lambda 24 (42 % up), gain 17 (53 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 52 % | 18.4 | 92 % | 27.6 | 100 % | 0.8 |
| combined: A, B - A | 57 % | 16.1 | 88 % | 56.0 | 93 % | 0.9 |
| combined: A, B - A, hand-picked bands | 57 % | 15.8 | 92 % | 52.9 | 100 % | 1.0 |
| combined: A, B, B - A | 83 % | 13.8 | 88 % | 32.2 | 100 % | 0.8 |
| combined: A, B, B - A, hand-picked bands | 70 % | 14.9 | 92 % | 31.7 | 93 % | 0.9 |
| settings only (settings change: gp) | 35 % | 22.1 | 79 % | 28.9 | 80 % | 1.0 |
| absolute trace: pretrained cnn, differenced | 79 % | 15.9 | 96 % | 26.7 | 65 % | 1.2 |
| absolute trace: trees, differenced | 86 % | 14.8 | 88 % | 27.1 | 71 % | 1.2 |

### nominal vs clock rails (H, V, TG) (15 pairs; clear changes sigma_e 0 (nan % up), lambda 2 (100 % up), gain 0 (nan % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | - | 3.3 | 100 % | 10.6 | - | 0.6 |
| combined: A, B - A | - | 3.8 | 100 % | 26.5 | - | 0.4 |
| combined: A, B - A, hand-picked bands | - | 3.5 | 100 % | 24.5 | - | 0.4 |
| combined: A, B, B - A | - | 3.2 | 100 % | 29.9 | - | 0.6 |
| combined: A, B, B - A, hand-picked bands | - | 3.1 | 100 % | 27.8 | - | 0.7 |
| settings only (settings change: gp) | - | 4.3 | 100 % | 13.9 | - | 0.6 |

### nominal vs output stage (VDD, VR, RG, SW) (27 pairs; clear changes sigma_e 1 (100 % up), lambda 3 (100 % up), gain 9 (44 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 100 % | 2.9 | 33 % | 46.9 | 100 % | 1.3 |
| combined: A, B - A | 100 % | 3.2 | 0 % | 53.1 | 89 % | 1.2 |
| combined: A, B - A, hand-picked bands | 100 % | 3.1 | 0 % | 53.1 | 100 % | 1.1 |
| combined: A, B, B - A | 100 % | 4.4 | 33 % | 50.4 | 89 % | 1.3 |
| combined: A, B, B - A, hand-picked bands | 100 % | 4.1 | 33 % | 51.3 | 100 % | 1.3 |
| settings only (settings change: gp) | 100 % | 2.6 | 100 % | 64.5 | 100 % | 0.7 |

### nominal vs timing (36 pairs; clear changes sigma_e 33 (94 % up), lambda 26 (69 % up), gain 2 (50 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 64 % | 19.7 | 96 % | 26.9 | 50 % | 1.0 |
| combined: A, B - A | 76 % | 18.3 | 85 % | 67.3 | 50 % | 0.9 |
| combined: A, B - A, hand-picked bands | 61 % | 18.8 | 77 % | 64.9 | 50 % | 0.9 |
| combined: A, B, B - A | 88 % | 16.6 | 100 % | 24.8 | 50 % | 1.0 |
| combined: A, B, B - A, hand-picked bands | 82 % | 17.2 | 92 % | 25.6 | 50 % | 1.0 |
| settings only (settings change: gp) | 36 % | 29.6 | 88 % | 31.3 | 0 % | 0.9 |

### same kind: all at once (33 pairs; clear changes sigma_e 26 (54 % up), lambda 19 (63 % up), gain 21 (43 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 73 % | 12.9 | 95 % | 41.5 | 81 % | 1.7 |
| combined: A, B - A | 77 % | 13.9 | 95 % | 36.8 | 81 % | 1.7 |
| combined: A, B - A, hand-picked bands | 81 % | 12.6 | 100 % | 25.2 | 86 % | 1.6 |
| combined: A, B, B - A | 77 % | 12.9 | 100 % | 21.1 | 81 % | 1.6 |
| combined: A, B, B - A, hand-picked bands | 77 % | 12.8 | 100 % | 17.3 | 90 % | 1.6 |
| settings only (settings change: gp) | 62 % | 11.4 | 53 % | 28.6 | 71 % | 2.2 |
| absolute trace: pretrained cnn, differenced | 85 % | 8.1 | 95 % | 22.2 | 81 % | 2.0 |
| absolute trace: trees, differenced | 69 % | 11.0 | 95 % | 23.0 | 81 % | 2.0 |

### same kind: clock rails (H, V, TG) (12 pairs; clear changes sigma_e 0 (nan % up), lambda 2 (0 % up), gain 0 (nan % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | - | 2.3 | 100 % | 11.3 | - | 0.3 |
| combined: A, B - A | - | 3.1 | 100 % | 34.0 | - | 0.5 |
| combined: A, B - A, hand-picked bands | - | 3.6 | 100 % | 31.8 | - | 0.6 |
| combined: A, B, B - A | - | 3.8 | 100 % | 29.9 | - | 0.3 |
| combined: A, B, B - A, hand-picked bands | - | 3.6 | 100 % | 28.6 | - | 0.4 |
| settings only (settings change: gp) | - | 3.3 | 0 % | 20.6 | - | 0.4 |
| absolute trace: pretrained cnn, differenced | - | 3.1 | 0 % | 14.8 | - | 0.6 |
| absolute trace: trees, differenced | - | 4.0 | 100 % | 10.8 | - | 0.5 |

### same kind: output stage (VDD, VR, RG, SW) (36 pairs; clear changes sigma_e 2 (100 % up), lambda 9 (67 % up), gain 19 (42 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 100 % | 2.5 | 17 % | 68.6 | 100 % | 1.4 |
| combined: A, B - A | 100 % | 2.9 | 67 % | 72.8 | 100 % | 1.4 |
| combined: A, B - A, hand-picked bands | 100 % | 3.1 | 50 % | 71.7 | 100 % | 1.3 |
| combined: A, B, B - A | 100 % | 4.7 | 50 % | 71.9 | 100 % | 1.5 |
| combined: A, B, B - A, hand-picked bands | 100 % | 4.3 | 50 % | 72.0 | 100 % | 1.5 |
| settings only (settings change: gp) | 100 % | 2.6 | 100 % | 92.4 | 100 % | 0.8 |
| absolute trace: pretrained cnn, differenced | 100 % | 6.6 | 33 % | 79.0 | 63 % | 2.0 |
| absolute trace: trees, differenced | 100 % | 4.9 | 67 % | 71.6 | 79 % | 1.5 |

### same kind: timing (51 pairs; clear changes sigma_e 44 (50 % up), lambda 47 (43 % up), gain 13 (46 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 88 % | 18.0 | 90 % | 30.4 | 62 % | 1.3 |
| combined: A, B - A | 68 % | 19.5 | 78 % | 57.4 | 69 % | 1.4 |
| combined: A, B - A, hand-picked bands | 68 % | 20.0 | 68 % | 58.5 | 69 % | 1.4 |
| combined: A, B, B - A | 80 % | 20.7 | 88 % | 32.1 | 69 % | 1.4 |
| combined: A, B, B - A, hand-picked bands | 70 % | 21.2 | 85 % | 33.7 | 69 % | 1.4 |
| settings only (settings change: gp) | 92 % | 24.3 | 78 % | 40.0 | 38 % | 1.3 |
| absolute trace: pretrained cnn, differenced | 93 % | 15.7 | 83 % | 31.5 | 69 % | 1.3 |
| absolute trace: trees, differenced | 89 % | 17.2 | 89 % | 27.6 | 77 % | 1.0 |

### two different kinds of change (48 pairs; clear changes sigma_e 17 (35 % up), lambda 23 (48 % up), gain 5 (100 % up))

| model | sigma_e direction | sigma_e size | lambda direction | lambda size | gain direction | gain size |
|---|---|---|---|---|---|---|
| change trees (B - A) | 44 % | 7.0 | 90 % | 41.2 | 100 % | 1.0 |
| combined: A, B - A | 44 % | 6.9 | 90 % | 45.2 | 100 % | 1.2 |
| combined: A, B - A, hand-picked bands | 44 % | 6.6 | 85 % | 44.2 | 50 % | 1.2 |
| combined: A, B, B - A | 44 % | 6.7 | 90 % | 43.4 | 100 % | 1.0 |
| combined: A, B, B - A, hand-picked bands | 44 % | 6.4 | 90 % | 43.6 | 100 % | 1.1 |
| settings only (settings change: gp) | 67 % | 5.7 | 85 % | 69.1 | 100 % | 0.7 |
| absolute trace: pretrained cnn, differenced | 65 % | 8.4 | 65 % | 39.7 | 60 % | 1.4 |
| absolute trace: trees, differenced | 35 % | 10.8 | 87 % | 39.7 | 40 % | 1.2 |
