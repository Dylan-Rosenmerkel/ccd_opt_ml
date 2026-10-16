# Level-1 rehearsal (2026-10-16 15:16)

6 simulated hand-taken packets (2 reference + 3 trace sets x 4 channels, board 101; image 16 x 1000 x 500, 1x1, NCOLPRE 5300); truth model in lvl1/simulate_manual_run.py.

## 1. Simulation

```
m0001: delayInt 3000, VSUB 44.9, ΔH -0.47 -> ch0 σe 0.187 λ 0.0304, ch1 σe 0.187 λ 0.0304, ch2 σe 0.604 λ 0.0304, ch3 σe 0.189 λ 0.0304
m0002: delayInt 3000, VSUB 66.5, ΔH +0.20 -> ch0 σe 0.187 λ 0.0290, ch1 σe 0.187 λ 0.0290, ch2 σe 0.604 λ 0.0290, ch3 σe 0.189 λ 0.0290
m0003: delayInt 1000, VSUB 48.3, ΔH -0.04 -> ch0 σe 0.271 λ 0.0201, ch1 σe 0.271 λ 0.0201, ch2 σe 1.032 λ 0.0201, ch3 σe 0.276 λ 0.0201
m0004: delayInt 1000, VSUB 41.3, ΔH -0.06 -> ch0 σe 0.271 λ 0.0177, ch1 σe 0.271 λ 0.0177, ch2 σe 1.032 λ 0.0177, ch3 σe 0.276 λ 0.0177
m0005: delayInt 500, VSUB 62.3, ΔH -0.23 -> ch0 σe 0.373 λ 0.0326, ch1 σe 0.373 λ 0.0326, ch2 σe 1.456 λ 0.0326, ch3 σe 0.379 λ 0.0326
m0006: delayInt 500, VSUB 68.0, ΔH +0.09 -> ch0 σe 0.373 λ 0.0318, ch1 σe 0.373 λ 0.0318, ch2 σe 1.456 λ 0.0318, ch3 σe 0.379 λ 0.0318
```

## 2. Manifests

```
manifests/m0001.json: status ok, 8 reference + 12 traces, images on boards [101]
manifests/m0002.json: status ok, 8 reference + 12 traces, images on boards [101]
manifests/m0003.json: status ok, 8 reference + 12 traces, images on boards [101]
manifests/m0004.json: status ok, 8 reference + 12 traces, images on boards [101]
manifests/m0005.json: status ok, 8 reference + 12 traces, images on boards [101]
manifests/m0006.json: status ok, 8 reference + 12 traces, images on boards [101]
```

## 3–5. Route `run_WADERs`

Processing (226 s, return code 0):

```
m0001: 20/20 traces usable, 3/4 labels without flags
m0002: 20/20 traces usable, 3/4 labels without flags
m0003: 20/20 traces usable, 3/4 labels without flags
m0004: 20/20 traces usable, 3/4 labels without flags
m0005: 20/20 traces usable, 3/4 labels without flags
m0006: 20/20 traces usable, 3/4 labels without flags
dataset: processed_run_WADERs/dataset.csv
```

Labels against the truth (`check_rehearsal.py`; full table in `4_check_run_WADERs.md`):

24 labels (18 without flags)

| quantity | median rel. error (unflagged) | RMS rel. error | median |pull| | max |pull| |
|---|---|---|---|---|
| sigma_e | +1.6 % | 3.5 % | 1.10 | 5.36 |
| lambda | +4.1 % | 10.0 % | 0.66 | 2.48 |
| gain | +0.0 % | 3.0 % | 0.89 | 4.19 |
| dark | +0.5 % | 20.7 % | 0.90 | 2.70 |
| CIC | +5.4 % | 43.0 % | 1.09 | 2.09 |

Flagged labels:
* m0001 ch2: not fitted by run_WADERs (flooded or unresolved peaks)
* m0002 ch2: not fitted by run_WADERs (flooded or unresolved peaks)
* m0003 ch2: not fitted by run_WADERs (flooded or unresolved peaks)
* m0004 ch2: not fitted by run_WADERs (flooded or unresolved peaks)
* m0005 ch2: not fitted by run_WADERs (flooded or unresolved peaks)
* m0006 ch2: not fitted by run_WADERs (flooded or unresolved peaks)

Training (`train_lvl1.py --quick`, return code 0; with 6 packets the scores only show that the step runs):

```
18 examples from 6 packets, boards [101], 3 folds (grouped by packet)
trace_physics  sigma_e:  28.9 % (rank 0.98, cover 0.00)  lambda:  39.3 % (rank -0.80, cover 0.44)  gain:   2.3 % (rank -0.04, cover 0.67)       0 s
physics        sigma_e:  35.0 % (rank 1.00, cover 0.22)  lambda:  39.8 % (rank -0.62, cover 0.11)  gain:   2.2 % (rank 0.04, cover 0.72)       2 s
physics_joint  sigma_e:  35.0 % (rank 1.00, cover 0.22)  lambda:  36.8 % (rank -0.66, cover 0.11)  gain:   2.2 % (rank 0.04, cover 0.72)       1 s
ridge          sigma_e:  21.4 % (rank 0.68, cover 0.61)  lambda:  73.1 % (rank -0.81, cover 0.33)  gain:   2.6 % (rank 0.05, cover 0.89)       0 s
gp             sigma_e:  42.1 % (rank -0.92, cover 0.33)  lambda:  39.3 % (rank -0.75, cover 0.44)  gain:   2.3 % (rank 0.02, cover 0.67)       0 s

wrote 5_train_run_WADERs/report.md, report.json, predictions.csv
```

## 3–5. Route `json`

Processing (165 s, return code 0):

```
m0001: 20/20 traces usable, 3/4 labels without flags
m0002: 20/20 traces usable, 2/4 labels without flags
m0003: 20/20 traces usable, 3/4 labels without flags
m0004: 20/20 traces usable, 3/4 labels without flags
m0005: 20/20 traces usable, 3/4 labels without flags
m0006: 20/20 traces usable, 3/4 labels without flags
dataset: processed_json/dataset.csv
```

Labels against the truth (`check_rehearsal.py`; full table in `4_check_json.md`):

24 labels (17 without flags)

| quantity | median rel. error (unflagged) | RMS rel. error | median |pull| | max |pull| |
|---|---|---|---|---|
| sigma_e | -8.0 % | 12.4 % | 1.86 | 5.97 |
| lambda | -12.0 % | 19.8 % | 1.96 | 5.43 |
| gain | +10.6 % | 16.7 % | 2.13 | 4.85 |
| dark | -12.6 % | 28.3 % | 0.85 | 5.31 |
| CIC | -4.1 % | 37.0 % | 0.53 | 3.39 |

Flagged labels:
* m0001 ch2: sigma at fit limit (0.5; limits 0.001..0.5)
* m0002 ch0: dc = 0.021 +/- 0.012 is not constrained
* m0002 ch2: sigma at fit limit (0.5; limits 0.001..0.5)
* m0003 ch2: sigma at fit limit (0.5; limits 0.001..0.5)
* m0004 ch2: sigma at fit limit (0.5; limits 0.001..0.5)
* m0005 ch2: sigma at fit limit (0.5; limits 0.001..0.5)
* m0006 ch2: sigma at fit limit (0.5; limits 0.001..0.5)

Training (`train_lvl1.py --quick`, return code 0; with 6 packets the scores only show that the step runs):

```
17 examples from 6 packets, boards [101], 3 folds (grouped by packet)
trace_physics  sigma_e:  38.4 % (rank 0.93, cover 0.06)  lambda:  27.7 % (rank -0.86, cover 0.35)  gain:  12.8 % (rank -0.95, cover 0.35)       0 s
physics        sigma_e:  44.4 % (rank 0.93, cover 0.35)  lambda:  27.9 % (rank -0.71, cover 0.35)  gain:  12.7 % (rank -0.91, cover 0.35)       2 s
physics_joint  sigma_e:  44.4 % (rank 0.93, cover 0.35)  lambda:  27.8 % (rank -0.35, cover 0.41)  gain:  12.7 % (rank -0.91, cover 0.35)       1 s
ridge          sigma_e:  19.1 % (rank 0.63, cover 0.59)  lambda:  47.4 % (rank -0.51, cover 0.41)  gain:   5.1 % (rank 0.88, cover 0.94)       0 s
gp             sigma_e:  28.1 % (rank -0.93, cover 0.35)  lambda:  27.7 % (rank -0.86, cover 0.35)  gain:  12.8 % (rank -0.94, cover 0.35)       0 s

wrote 5_train_json/report.md, report.json, predictions.csv
```

Total time 423 s.
