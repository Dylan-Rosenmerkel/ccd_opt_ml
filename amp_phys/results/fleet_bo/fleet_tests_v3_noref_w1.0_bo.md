# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: none. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 1.0

| strategy | images to 0 % | images to 0.5 % | images to 1 % | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|---|---|---|
| expert: hill-climb from a random start | 51.0 | 39.7 | 24.5 | 15.0 | 5.5 | 3.3 |
| expert: hill-climb from the fleet model's predicted best | 14.2 | 12.3 | 7.7 | 2.2 | 1.3 | 0.3 |
| random order | 63.5 | 31.2 | 19.3 | 7.7 | 3.5 | 2.1 |
| fleet model ranking, no updates | 11.8 | 10.3 | 7.5 | 5.7 | 1.2 | 0.5 |
| BO, fleet as prior (multi-task GP) | 16.0 | 14.2 | 8.0 | 5.5 | 1.2 | 0.5 |
| BO, fleet + trace PHYSICS features | 31.7 | 30.2 | 14.7 | 9.7 | 5.8 | 4.7 |
| BO, fleet + physics + trace components | 29.7 | 24.2 | 15.7 | 13.3 | 6.7 | 7.7 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 12.8 | 11.0 | 6.5 | 4.5 | 1.0 | 0.2 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 32.8 | 30.8 | 20.3 | 11.2 | 1.5 | 3.0 |
| BO, fleet + physics + trace components + VALIDITY classifier | 29.0 | 21.8 | 14.2 | 7.3 | 2.0 | 2.2 |
| BO, fleet as prior (multi-task GP) + validity GATE (P<0.5 never imaged) | 14.7 | 12.8 | 7.3 | 5.3 | 1.0 | 0.3 |
| BO, fleet + trace PHYSICS features + validity GATE (P<0.5 never imaged) | 21.7 | 20.3 | 8.2 | 5.2 | 2.0 | 0.2 |
| BO, fleet + physics + trace components + validity GATE (P<0.5 never imaged) | 21.7 | 16.2 | 8.7 | 6.3 | 2.3 | 0.7 |
| BO, no fleet data | 31.2 | 30.0 | 13.7 | 5.8 | 2.5 | 1.5 |
