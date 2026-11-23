# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: none. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 0.5

| strategy | images to 0 % | images to 0.5 % | images to 1 % | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|---|---|---|
| expert: hill-climb from a random start | 35.2 | 31.5 | 29.3 | 24.0 | 19.5 | 3.1 |
| expert: hill-climb from the fleet model's predicted best | 6.8 | 6.8 | 6.8 | 5.3 | 5.2 | 0.3 |
| TRACE-DRIVEN: rank by trace noise + fleet dark charge, gate, image to confirm | 9.0 | 8.8 | 8.8 | 7.7 | 4.7 | 0.0 |
| random order | 63.5 | 48.7 | 40.2 | 31.0 | 18.8 | 4.8 |
| fleet model ranking, no updates | 9.3 | 9.3 | 9.3 | 8.3 | 7.8 | 1.0 |
| BO, fleet as prior (multi-task GP) | 3.0 | 3.0 | 3.0 | 1.7 | 1.3 | 0.0 |
| BO, fleet + trace PHYSICS features | 10.3 | 10.3 | 10.3 | 5.0 | 3.5 | 3.2 |
| BO, fleet + physics + trace components | 17.5 | 13.2 | 13.2 | 6.0 | 3.7 | 3.3 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 3.8 | 3.8 | 3.8 | 2.7 | 2.0 | 0.0 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 5.0 | 5.0 | 5.0 | 1.7 | 1.3 | 0.0 |
| BO, fleet + physics + trace components + VALIDITY classifier | 11.7 | 7.5 | 7.5 | 4.3 | 2.0 | 1.7 |
| BO, fleet as prior (multi-task GP) + validity GATE (P<0.5 never imaged) | 2.7 | 2.7 | 2.7 | 1.7 | 1.3 | 0.0 |
| BO, fleet + trace PHYSICS features + validity GATE (P<0.5 never imaged) | 3.7 | 3.7 | 3.7 | 1.8 | 1.5 | 0.0 |
| BO, fleet + physics + trace components + validity GATE (P<0.5 never imaged) | 7.5 | 4.8 | 4.8 | 2.5 | 1.8 | 0.0 |
| BO, no fleet data | 6.8 | 6.8 | 3.2 | 3.2 | 3.0 | 0.3 |
