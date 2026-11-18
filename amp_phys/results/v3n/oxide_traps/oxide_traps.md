# Oxide trap density from each amplifier's own 1/f noise (L1 strips, shared part removed)

|                                |          0 |          1 |          2 |          3 |
|:-------------------------------|-----------:|-----------:|-----------:|-----------:|
| sigma_1 own (ADU)              |   4.81     |   4.89     |   5.37     |   5.41     |
| sigma_240 own (ADU)            |   0.409    |   0.411    |   0.426    |   0.548    |
| white-only sigma_240 (ADU)     |   0.311    |   0.315    |   0.346    |   0.349    |
| K_v output (ADU^2)             |  -2.17     |  -2.39     |   1.77     | -13.8      |
| 1/f share at N=240             |  -0.157    |  -0.171    |   0.118    |  -0.553    |
| RTS steps per 1000 skips       |   4.34e-05 |   4.34e-05 |   4.34e-05 |   4.34e-05 |
| median step (ADU)              |  66.1      |  70        |  65.6      | 342        |
| median spacing (skips)         | nan        | nan        | nan        | nan        |
| N_t A=0.85 C_ox (eV^-1 cm^-3)  |   0        |   0        |   7.9e+15  |   0        |
| one-charge step C_ox (ADU)     |   0.761    |   0.761    |   0.761    |   0.761    |
| N_t A=0.85 C_eff (eV^-1 cm^-3) |   0        |   0        |   5.99e+14 |   0        |
| one-charge step C_eff (ADU)    |   2.76     |   2.76     |   2.76     |   2.76     |
| N_t A=0.44 C_ox (eV^-1 cm^-3)  |   0        |   0        |   2.95e+16 |   0        |
| N_t A=0.44 C_eff (eV^-1 cm^-3) |   0        |   0        |   2.24e+15 |   0        |

N_t is an upper limit (any own slow noise that is not the transistor's 1/f also counts).  Typical thermal-oxide
values: 10^16-10^17 eV^-1 cm^-3 (good) to 10^18 (poor).
