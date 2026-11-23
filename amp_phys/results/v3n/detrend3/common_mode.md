# Common-mode (setup) noise vs each amplifier's own

475 simultaneous trace sets.

## Mean correlation of the per-skip noise

|         |   101 ch0 |   101 ch1 |   101 ch2 |   101 ch3 |   102 ch0 |   102 ch1 |   102 ch2 |   102 ch3 |
|:--------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| 101 ch0 |         1 |         0 |         0 |         0 |        -0 |        -0 |        -0 |        -0 |
| 101 ch1 |         0 |         1 |         0 |        -0 |        -0 |        -0 |        -0 |        -0 |
| 101 ch2 |         0 |         0 |         1 |        -0 |        -0 |        -0 |        -0 |        -0 |
| 101 ch3 |         0 |        -0 |        -0 |         1 |        -0 |        -0 |        -0 |         0 |
| 102 ch0 |        -0 |        -0 |        -0 |        -0 |         1 |         0 |         0 |         0 |
| 102 ch1 |        -0 |        -0 |        -0 |        -0 |         0 |         1 |         0 |         0 |
| 102 ch2 |        -0 |        -0 |        -0 |        -0 |         0 |         0 |         1 |         0 |
| 102 ch3 |        -0 |        -0 |        -0 |         0 |         0 |         0 |         0 |         1 |

## Share of each channel's per-skip noise variance explained by the others

| channel | by its board's other channels | by the other board |
|---|---|---|
| 101 ch0 | 0.0 % | 0.0 % |
| 101 ch1 | 0.0 % | 0.0 % |
| 101 ch2 | 0.0 % | 0.0 % |
| 101 ch3 | 0.0 % | 0.0 % |
| 102 ch0 | 0.0 % | 0.0 % |
| 102 ch1 | 0.0 % | 0.0 % |
| 102 ch2 | 0.0 % | 0.0 % |
| 102 ch3 | 0.0 % | 0.0 % |

Noise thermometry (board 102): temperature spread 4.73 K (RMS), range -6.93 to 8.56 K (5-95 %), drift -0.143 K/h.
