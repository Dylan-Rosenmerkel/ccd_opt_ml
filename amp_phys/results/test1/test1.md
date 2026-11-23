# Test 1: robust trace physics for sigma_e (train G, sealed H, board 101 ch0/1/3)

## new amplifier

| model                     |   typical error % (RMS log) |   median |error| % |   change rho (all H pairs) |   change size error % |
|:--------------------------|----------------------------:|-------------------:|---------------------------:|----------------------:|
| settings: trees           |                      18.22  |             14.684 |                      0.646 |                12.296 |
| physics, no ML            |                      11.882 |              7.447 |                      0.626 |                10.276 |
| physics features: trees   |                       9.569 |              5.476 |                      0.544 |                 9.976 |
| physics + settings: trees |                       8.55  |              5.712 |                      0.686 |                 9.629 |

## trained

| model                     |   typical error % (RMS log) |   median |error| % |   change rho (all H pairs) |   change size error % |
|:--------------------------|----------------------------:|-------------------:|---------------------------:|----------------------:|
| settings: trees           |                      13.777 |             10.823 |                      0.674 |                11.341 |
| physics, no ML            |                       9.991 |              5.258 |                      0.648 |                10.184 |
| physics features: trees   |                       7.242 |              3.289 |                      0.622 |                 9.431 |
| physics + settings: trees |                       6.528 |              2.299 |                      0.807 |                 8.165 |

Scorecard v3 on H, new amplifier: trace physics 13.5 %, trace trees 12.7 %, settings gp 18.8 %;
change rho: ensemble settings + trace 0.73, settings trees 0.64, physics difference + correction 0.65.
Trained: trace physics 7.6 %, settings + amplifier trees 7.7 %; change rho ensemble settings + trace 0.82.
