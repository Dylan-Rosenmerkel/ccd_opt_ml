# Physics equations, leave one amplifier out (board 101, v3 sessions G+H)

Error = RMS of log(predicted/true), %, on the held-out amplifier (its calibration image left out).

|                                                 | ch0                 | ch1                  | ch2                  | ch3                  |
|:------------------------------------------------|:--------------------|:---------------------|:---------------------|:---------------------|
| rows                                            | 100                 | 98                   | 98                   | 98                   |
| noise N0 (trace, no fit, no image)              | 8.799909066864101   | 15.364739125367965   | 33.97522218168627    | 22.186737264574973   |
| noise N1 (trace + 2 global constants, no image) | 11.253848721980555  | 13.800034520821491   | 35.26622403621946    | 21.034600584598973   |
| noise N2 (N1 + one image)                       | 9.559496905761982   | 13.43637172513707    | 33.96197846210197    | 19.103683528558477   |
| noise, true lambda and gain (limit)             | 11.749551883978791  | 11.565805996880965   | 33.42873649436321    | 19.051447972235756   |
| gain G0 (shared slopes, no image)               | 3.968921055238006   | 3.022458829495408    | 3.680719894776028    | 3.213896557282301    |
| gain G1 (shared slopes + one image)             | 3.3389051746125644  | 4.3376070828717      | 3.7261959039047317   | 3.603562796919003    |
| lambda L0 (shared slopes, no image)             | 81.7715907892421    | 27.19249464854506    | 62.859449692778455   | 57.25136180195509    |
| lambda L1 (shared slopes + one image)           | 20.588219958786595  | 27.938121774104914   | 48.103462069551064   | 32.9084960060472     |
| fitted scale s                                  | 0.7489794695617348  | 0.872816142639191    | 0.8422841607348772   | 0.8216327688745364   |
| fitted kappa                                    | 0.02552133546261231 | 0.017810795382537936 | 0.019041407608930472 | 0.022060704647519167 |
| gain slope VDD (%/V)                            | -1.617385063490956  | -1.6140195603764607  | -1.5541831095099279  | -1.5838263016836232  |
| gain slope VR (%/V)                             | -1.4405322376285554 | -2.2680397033489728  | -2.117880265831088   | -1.4394408651899175  |
| lambda slope VSUB below / above 56 V (%/V)      | 1.9 / 4.5           | 1.0 / 3.3            | 2.6 / 4.8            | 0.7 / 3.3            |

## Mean over the four held-out amplifiers vs the ML models (new amplifier)

| target | physics, no image | physics, one image | best ML, new amplifier (session G) |
|---|---|---|---|
| noise sigma_e | 20.3 (N0 20.1) | 19.0 | 11.9 (trace trees) |
| gain | 3.5 | 3.8 | 3.7 (settings trees) |
| dark charge lambda | 57.3 | 32.4 | 39.9 (trace trees) |
