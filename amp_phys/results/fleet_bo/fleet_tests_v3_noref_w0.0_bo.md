# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: none. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 0.0

| strategy | images to 0 % | images to 0.5 % | images to 1 % | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|---|---|---|
| expert: hill-climb from a random start | 39.2 | 37.7 | 31.8 | 29.8 | 25.0 | 5.2 |
| expert: hill-climb from the fleet model's predicted best | 21.5 | 21.5 | 16.7 | 14.8 | 7.0 | 3.0 |
| random order | 63.5 | 50.8 | 45.8 | 38.2 | 25.7 | 6.0 |
| fleet model ranking, no updates | 10.3 | 9.5 | 8.5 | 8.5 | 7.8 | 1.0 |
| BO, fleet as prior (multi-task GP) | 4.2 | 3.0 | 2.0 | 2.0 | 1.3 | 0.0 |
| BO, fleet + trace PHYSICS features | 10.0 | 8.7 | 7.2 | 7.2 | 1.0 | 4.8 |
| BO, fleet + physics + trace components | 8.7 | 7.0 | 5.3 | 5.3 | 1.2 | 3.3 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 4.8 | 4.0 | 2.8 | 2.8 | 1.7 | 0.0 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 8.0 | 6.5 | 5.2 | 5.2 | 1.0 | 2.8 |
| BO, fleet + physics + trace components + VALIDITY classifier | 8.0 | 5.8 | 3.8 | 3.8 | 1.2 | 1.8 |
| BO, fleet as prior (multi-task GP) + validity GATE (P<0.5 never imaged) | 3.8 | 2.7 | 2.0 | 2.0 | 1.3 | 0.0 |
| BO, fleet + trace PHYSICS features + validity GATE (P<0.5 never imaged) | 4.5 | 3.5 | 2.7 | 2.7 | 1.0 | 0.3 |
| BO, fleet + physics + trace components + validity GATE (P<0.5 never imaged) | 4.8 | 3.3 | 2.2 | 2.2 | 1.2 | 0.2 |
| BO, no fleet data | 7.8 | 4.3 | 3.0 | 3.0 | 2.8 | 0.2 |
