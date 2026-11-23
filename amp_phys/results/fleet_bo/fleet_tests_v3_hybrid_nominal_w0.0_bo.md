# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: nominal. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 0.0

| strategy | images to 0 % | images to 0.5 % | images to 1 % | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|---|---|---|
| expert: expand out from nominal (closest first) | 38.2 | 35.3 | 33.8 | 31.2 | 30.7 | 6.2 |
| expert: hill-climb (closest to the best so far) | 34.7 | 31.3 | 28.7 | 26.0 | 25.5 | 4.5 |
| expert: hill-climb from a random start | 38.0 | 31.7 | 28.7 | 26.7 | 19.8 | 3.2 |
| expert: hill-climb from the fleet model's predicted best | 17.7 | 10.0 | 6.8 | 5.3 | 5.3 | 0.3 |
| TRACE-DRIVEN: rank by trace noise + fleet dark charge, gate, image to confirm | 8.8 | 8.7 | 8.7 | 8.2 | 8.2 | 0.0 |
| HYBRID: trace-driven first image, then fleet BO + gate | 3.3 | 3.2 | 3.2 | 2.7 | 2.7 | 0.0 |
| random order | 63.5 | 52.2 | 46.7 | 38.0 | 25.7 | 5.7 |
| fleet model ranking, no updates | 10.2 | 10.0 | 10.0 | 9.2 | 9.2 | 1.0 |
| BO, fleet as prior (multi-task GP) | 3.7 | 3.5 | 3.5 | 2.7 | 2.7 | 0.0 |
| BO, fleet + trace PHYSICS features | 11.7 | 11.7 | 8.7 | 8.5 | 4.0 | 5.5 |
| BO, fleet + physics + trace components | 11.3 | 11.0 | 10.0 | 7.0 | 2.5 | 3.8 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 4.2 | 4.0 | 4.0 | 3.0 | 3.0 | 0.0 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 8.7 | 8.7 | 5.2 | 5.0 | 2.2 | 2.2 |
| BO, fleet + physics + trace components + VALIDITY classifier | 9.8 | 9.5 | 7.5 | 5.5 | 2.5 | 2.3 |
| BO, fleet as prior (multi-task GP) + validity GATE (P<0.5 never imaged) | 3.3 | 3.2 | 3.2 | 2.7 | 2.7 | 0.0 |
| BO, fleet + trace PHYSICS features + validity GATE (P<0.5 never imaged) | 5.5 | 5.5 | 3.5 | 3.3 | 2.3 | 0.3 |
| BO, fleet + physics + trace components + validity GATE (P<0.5 never imaged) | 6.2 | 5.8 | 4.8 | 3.5 | 2.5 | 0.3 |
| BO, no fleet data | 8.7 | 8.5 | 7.8 | 4.8 | 4.8 | 0.7 |
