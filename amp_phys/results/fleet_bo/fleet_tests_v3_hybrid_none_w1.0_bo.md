# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: none. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 1.0

| strategy | images to 0 % | images to 0.5 % | images to 1 % | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|---|---|---|
| expert: hill-climb from a random start | 49.7 | 37.2 | 29.8 | 11.5 | 5.5 | 4.4 |
| expert: hill-climb from the fleet model's predicted best | 22.8 | 21.3 | 19.8 | 12.8 | 1.2 | 3.0 |
| TRACE-DRIVEN: rank by trace noise + fleet dark charge, gate, image to confirm | 5.2 | 4.5 | 2.2 | 1.3 | 1.0 | 0.0 |
| HYBRID: trace-driven first image, then fleet BO + gate | 12.3 | 10.5 | 8.3 | 2.8 | 1.0 | 0.0 |
| random order | 63.5 | 26.3 | 15.2 | 6.5 | 3.5 | 1.5 |
| fleet model ranking, no updates | 11.5 | 10.2 | 6.7 | 4.3 | 1.2 | 0.5 |
| BO, fleet as prior (multi-task GP) | 13.7 | 11.8 | 9.2 | 3.5 | 1.2 | 0.2 |
| BO, fleet + trace PHYSICS features | 36.2 | 23.8 | 21.7 | 10.0 | 3.7 | 6.3 |
| BO, fleet + physics + trace components | 35.5 | 23.8 | 19.7 | 7.2 | 2.7 | 2.2 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 9.2 | 7.7 | 5.3 | 2.3 | 1.8 | 0.0 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 31.3 | 17.3 | 15.7 | 6.2 | 1.0 | 2.2 |
| BO, fleet + physics + trace components + VALIDITY classifier | 34.0 | 21.2 | 18.0 | 5.0 | 1.7 | 0.2 |
| BO, fleet as prior (multi-task GP) + validity GATE (P<0.5 never imaged) | 12.7 | 10.8 | 8.2 | 3.5 | 1.2 | 0.2 |
| BO, fleet + trace PHYSICS features + validity GATE (P<0.5 never imaged) | 24.7 | 14.2 | 12.2 | 4.0 | 1.2 | 0.3 |
| BO, fleet + physics + trace components + validity GATE (P<0.5 never imaged) | 26.7 | 17.2 | 15.0 | 5.0 | 1.7 | 0.0 |
| BO, no fleet data | 36.2 | 17.0 | 15.2 | 8.2 | 2.5 | 2.2 |
