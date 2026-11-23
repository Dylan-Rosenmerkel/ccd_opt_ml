# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: none. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 0.5

| strategy | images to 0 % | images to 0.5 % | images to 1 % | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|---|---|---|
| expert: hill-climb from a random start | 42.0 | 39.2 | 32.0 | 26.3 | 19.2 | 2.9 |
| expert: hill-climb from the fleet model's predicted best | 21.7 | 21.7 | 21.7 | 5.3 | 3.0 | 0.3 |
| random order | 63.5 | 47.8 | 39.3 | 29.8 | 18.2 | 4.4 |
| fleet model ranking, no updates | 10.5 | 10.5 | 10.5 | 8.5 | 6.2 | 1.0 |
| BO, fleet as prior (multi-task GP) | 4.2 | 4.2 | 4.2 | 2.2 | 1.3 | 0.0 |
| BO, fleet + trace PHYSICS features | 15.3 | 15.3 | 9.0 | 4.8 | 2.2 | 2.7 |
| BO, fleet + physics + trace components | 15.3 | 15.3 | 15.3 | 7.8 | 3.3 | 4.7 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 4.8 | 4.8 | 4.8 | 2.8 | 2.0 | 0.0 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 13.7 | 13.7 | 5.7 | 1.8 | 1.0 | 0.0 |
| BO, fleet + physics + trace components + VALIDITY classifier | 11.2 | 11.2 | 11.2 | 5.0 | 1.7 | 2.0 |
| BO, fleet as prior (multi-task GP) + validity GATE (P<0.5 never imaged) | 3.8 | 3.8 | 3.8 | 2.2 | 1.3 | 0.0 |
| BO, fleet + trace PHYSICS features + validity GATE (P<0.5 never imaged) | 10.0 | 10.0 | 4.0 | 2.2 | 1.0 | 0.0 |
| BO, fleet + physics + trace components + validity GATE (P<0.5 never imaged) | 7.3 | 7.3 | 7.3 | 3.2 | 1.5 | 0.0 |
| BO, no fleet data | 3.3 | 3.3 | 3.3 | 3.2 | 3.2 | 0.3 |
