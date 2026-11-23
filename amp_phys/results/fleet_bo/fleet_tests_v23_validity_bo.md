# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: nominal. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 1.0

| strategy | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|
| random order | 19.8 | 8.8 | 1.5 |
| fleet model ranking, no updates | 18.7 | 18.0 | 1.2 |
| BO, fleet as prior (multi-task GP) | 4.0 | 3.2 | 0.2 |
| BO, fleet + trace PHYSICS features | 15.8 | 8.3 | 6.7 |
| BO, fleet + physics + trace components | 10.2 | 6.7 | 2.7 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 4.3 | 3.7 | 0.0 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 11.7 | 7.7 | 2.8 |
| BO, fleet + physics + trace components + VALIDITY classifier | 5.2 | 3.5 | 0.0 |
| BO, no fleet data | 10.2 | 7.5 | 0.7 |

## w = 0.5

| strategy | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|
| random order | 38.2 | 20.7 | 4.5 |
| fleet model ranking, no updates | 13.2 | 12.7 | 0.2 |
| BO, fleet as prior (multi-task GP) | 16.5 | 14.8 | 0.2 |
| BO, fleet + trace PHYSICS features | 18.2 | 17.2 | 5.2 |
| BO, fleet + physics + trace components | 17.5 | 17.0 | 1.3 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 18.5 | 15.3 | 0.3 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 16.3 | 13.3 | 2.2 |
| BO, fleet + physics + trace components + VALIDITY classifier | 18.3 | 15.8 | 0.2 |
| BO, no fleet data | 10.5 | 10.5 | 0.5 |

## w = 0.0

| strategy | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|
| random order | 61.7 | 41.0 | 6.5 |
| fleet model ranking, no updates | 18.8 | 18.8 | 1.0 |
| BO, fleet as prior (multi-task GP) | 5.3 | 5.3 | 0.0 |
| BO, fleet + trace PHYSICS features | 8.0 | 8.0 | 3.2 |
| BO, fleet + physics + trace components | 5.7 | 4.8 | 0.2 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 5.7 | 5.7 | 0.0 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 10.0 | 10.0 | 1.7 |
| BO, fleet + physics + trace components + VALIDITY classifier | 6.8 | 6.3 | 0.0 |
| BO, no fleet data | 7.0 | 7.0 | 0.3 |
