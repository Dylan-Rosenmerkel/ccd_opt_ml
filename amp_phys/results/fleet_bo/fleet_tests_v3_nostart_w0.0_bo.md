# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: none. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 0.0

| strategy | images to 0 % | images to 0.5 % | images to 1 % | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|---|---|---|
| expert: hill-climb from a random start | 38.0 | 31.7 | 28.7 | 26.8 | 20.3 | 3.4 |
| expert: hill-climb from the fleet model's predicted best | 17.7 | 10.0 | 6.8 | 5.3 | 5.3 | 0.3 |
| random order | 63.5 | 52.2 | 45.8 | 38.5 | 26.2 | 5.7 |
| fleet model ranking, no updates | 9.3 | 9.2 | 9.2 | 8.3 | 8.3 | 1.0 |
| BO, fleet as prior (multi-task GP) | 2.8 | 2.7 | 2.7 | 1.8 | 1.8 | 0.0 |
| BO, fleet + trace PHYSICS features | 10.3 | 10.3 | 8.0 | 7.3 | 2.2 | 4.8 |
| BO, fleet + physics + trace components | 11.2 | 11.0 | 9.8 | 8.0 | 2.5 | 5.7 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 3.8 | 3.7 | 3.7 | 2.7 | 2.5 | 0.0 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 8.7 | 8.7 | 5.3 | 4.8 | 1.2 | 2.3 |
| BO, fleet + physics + trace components + VALIDITY classifier | 8.3 | 8.2 | 6.2 | 4.2 | 1.3 | 1.8 |
| BO, fleet as prior (multi-task GP) + validity GATE (P<0.5 never imaged) | 2.5 | 2.3 | 2.3 | 1.8 | 1.8 | 0.0 |
| BO, fleet + trace PHYSICS features + validity GATE (P<0.5 never imaged) | 5.0 | 5.0 | 3.2 | 2.8 | 1.2 | 0.3 |
| BO, fleet + physics + trace components + validity GATE (P<0.5 never imaged) | 5.2 | 5.0 | 3.8 | 2.7 | 1.3 | 0.3 |
| BO, no fleet data | 6.5 | 6.3 | 5.2 | 4.0 | 2.8 | 0.5 |
