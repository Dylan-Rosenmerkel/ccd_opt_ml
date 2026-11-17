# Test 2: physics models for gain and dark charge (train G, sealed H, board 101 ch0/1/3)

## gain, new amplifier

| model                     |   typical error % (RMS log) |   median |error| % |   change rho (all H pairs) |   change size error % |
|:--------------------------|----------------------------:|-------------------:|---------------------------:|----------------------:|
| settings: trees           |                       3.607 |              3.483 |                      0.88  |                 1.945 |
| physics, no ML (reset)    |                       8.678 |              3.868 |                      0.366 |                10.83  |
| physics, no ML (SW)       |                      69.432 |             69.393 |                      0.124 |                30.419 |
| physics features: trees   |                       3.528 |              3.13  |                      0.766 |                 2.36  |
| physics + settings: trees |                       3.421 |              2.953 |                      0.858 |                 1.945 |

## lambda, new amplifier

| model                           |   typical error % (RMS log) |   median |error| % |   change rho (all H pairs) |   change size error % |
|:--------------------------------|----------------------------:|-------------------:|---------------------------:|----------------------:|
| settings: trees                 |                      69.034 |             53.672 |                      0.915 |                41.432 |
| physics form, no ML             |                      71.493 |             54.754 |                      0.805 |                50.445 |
| components (dark + CIC): trees  |                      83.169 |             54.699 |                      0.443 |                57.048 |
| settings + trace + setup: trees |                      48.79  |             34.654 |                      0.844 |                37.348 |

(fitted V* of the reset formula: [9.34 9.27 9.38] V)

## gain, trained

| model                     |   typical error % (RMS log) |   median |error| % |   change rho (all H pairs) |   change size error % |
|:--------------------------|----------------------------:|-------------------:|---------------------------:|----------------------:|
| settings: trees           |                       2.547 |              2.304 |                      0.914 |                 1.682 |
| physics, no ML (reset)    |                       8.248 |              3.195 |                      0.373 |                10.882 |
| physics, no ML (SW)       |                      50.359 |             43.205 |                      0.124 |                30.419 |
| physics features: trees   |                       2.005 |              0.87  |                      0.751 |                 2.323 |
| physics + settings: trees |                       1.75  |              1.112 |                      0.872 |                 1.717 |

## lambda, trained

| model                           |   typical error % (RMS log) |   median |error| % |   change rho (all H pairs) |   change size error % |
|:--------------------------------|----------------------------:|-------------------:|---------------------------:|----------------------:|
| settings: trees                 |                      49.086 |             36.963 |                      0.921 |                32.417 |
| physics form, no ML             |                      53.897 |             38.677 |                      0.797 |                45.305 |
| components (dark + CIC): trees  |                      73.02  |             39.211 |                      0.423 |                54.318 |
| settings + trace + setup: trees |                      21.49  |              8.226 |                      0.827 |                26.65  |

(fitted V* of the reset formula: [9.32 9.32 9.32] V)

Scorecard v3 on H, new amplifier: gain settings gp 3.9 %, trees 3.6 %; lambda settings gp 71.7 %, trees 66.4 %,
trace physics 50.9 %, trace trees 49.5 %.  Trained: gain settings + amplifier gp 0.6 %; lambda settings + amplifier
trees 10.5 %.  Change rho (new amp): gain settings 0.87; lambda settings + trace combined trees 0.94.
