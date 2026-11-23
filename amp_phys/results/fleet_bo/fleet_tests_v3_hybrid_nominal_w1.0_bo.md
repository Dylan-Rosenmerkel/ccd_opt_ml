# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: nominal. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 1.0

| strategy | images to 0 % | images to 0.5 % | images to 1 % | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|---|---|---|
| expert: expand out from nominal (closest first) | 5.5 | 4.0 | 2.5 | 1.5 | 1.3 | 0.0 |
| expert: hill-climb (closest to the best so far) | 5.2 | 3.7 | 2.2 | 1.5 | 1.3 | 0.0 |
| expert: hill-climb from a random start | 49.7 | 36.7 | 30.7 | 11.2 | 5.3 | 4.5 |
| expert: hill-climb from the fleet model's predicted best | 22.8 | 21.3 | 19.8 | 12.8 | 1.2 | 3.0 |
| TRACE-DRIVEN: rank by trace noise + fleet dark charge, gate, image to confirm | 5.5 | 4.7 | 2.3 | 1.7 | 1.3 | 0.0 |
| HYBRID: trace-driven first image, then fleet BO + gate | 11.8 | 9.8 | 8.0 | 3.0 | 1.3 | 0.0 |
| random order | 42.8 | 15.5 | 9.5 | 5.0 | 1.5 | 0.9 |
| fleet model ranking, no updates | 10.2 | 8.8 | 5.3 | 3.7 | 1.3 | 0.2 |
| BO, fleet as prior (multi-task GP) | 13.3 | 11.3 | 9.2 | 4.0 | 1.3 | 0.3 |
| BO, fleet + trace PHYSICS features | 28.0 | 12.2 | 12.2 | 3.8 | 3.0 | 1.7 |
| BO, fleet + physics + trace components | 31.5 | 18.5 | 11.8 | 5.8 | 2.3 | 2.2 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 9.2 | 7.7 | 5.5 | 2.5 | 1.8 | 0.0 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 26.2 | 10.0 | 10.0 | 2.2 | 1.3 | 0.0 |
| BO, fleet + physics + trace components + VALIDITY classifier | 30.5 | 18.2 | 10.7 | 4.0 | 1.5 | 0.2 |
| BO, fleet as prior (multi-task GP) + validity GATE (P<0.5 never imaged) | 12.0 | 10.0 | 7.8 | 3.7 | 1.3 | 0.0 |
| BO, fleet + trace PHYSICS features + validity GATE (P<0.5 never imaged) | 20.2 | 8.2 | 8.2 | 2.2 | 1.3 | 0.0 |
| BO, fleet + physics + trace components + validity GATE (P<0.5 never imaged) | 23.5 | 14.7 | 9.0 | 3.8 | 1.7 | 0.2 |
| BO, no fleet data | 27.8 | 14.3 | 12.8 | 5.8 | 1.8 | 1.0 |
