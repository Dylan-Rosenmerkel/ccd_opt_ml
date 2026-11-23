# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: nominal. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 0.5

| strategy | images to 0 % | images to 0.5 % | images to 1 % | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|---|---|---|
| expert: expand out from nominal (closest first) | 33.8 | 33.8 | 30.7 | 30.7 | 30.0 | 6.2 |
| expert: hill-climb (closest to the best so far) | 30.0 | 30.0 | 26.8 | 26.8 | 26.3 | 4.3 |
| expert: hill-climb from a random start | 35.2 | 31.5 | 29.0 | 23.7 | 19.2 | 2.9 |
| expert: hill-climb from the fleet model's predicted best | 6.8 | 6.8 | 6.8 | 5.3 | 5.2 | 0.3 |
| TRACE-DRIVEN: rank by trace noise + fleet dark charge, gate, image to confirm | 9.7 | 9.5 | 9.5 | 8.3 | 5.3 | 0.0 |
| HYBRID: trace-driven first image, then fleet BO + gate | 4.2 | 4.0 | 4.0 | 2.8 | 2.3 | 0.0 |
| random order | 63.5 | 48.5 | 40.3 | 31.0 | 18.2 | 4.7 |
| fleet model ranking, no updates | 10.2 | 10.2 | 10.2 | 9.2 | 8.5 | 1.0 |
| BO, fleet as prior (multi-task GP) | 3.8 | 3.8 | 3.8 | 2.5 | 2.0 | 0.0 |
| BO, fleet + trace PHYSICS features | 11.2 | 11.2 | 11.2 | 4.3 | 3.8 | 1.8 |
| BO, fleet + physics + trace components | 18.7 | 14.5 | 14.5 | 7.5 | 5.8 | 4.2 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 4.3 | 4.3 | 4.3 | 3.2 | 2.3 | 0.0 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 5.7 | 5.7 | 5.7 | 2.5 | 2.0 | 0.0 |
| BO, fleet + physics + trace components + VALIDITY classifier | 12.8 | 8.8 | 8.8 | 5.2 | 3.2 | 1.7 |
| BO, fleet as prior (multi-task GP) + validity GATE (P<0.5 never imaged) | 3.5 | 3.5 | 3.5 | 2.5 | 2.0 | 0.0 |
| BO, fleet + trace PHYSICS features + validity GATE (P<0.5 never imaged) | 4.2 | 4.2 | 4.2 | 2.5 | 2.0 | 0.0 |
| BO, fleet + physics + trace components + validity GATE (P<0.5 never imaged) | 8.3 | 5.8 | 5.8 | 3.3 | 2.2 | 0.0 |
| BO, no fleet data | 8.5 | 8.5 | 8.5 | 4.8 | 4.5 | 0.7 |
