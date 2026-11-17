# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: nominal. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 1.0

| strategy | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|
| random order | 5.0 | 1.5 | 0.8 |
| fleet model ranking, no updates | 3.7 | 1.3 | 0.2 |
| BO, fleet as prior (multi-task GP) | 4.0 | 1.3 | 0.3 |
| BO, fleet + trace PHYSICS features | 3.8 | 3.0 | 1.7 |
| BO, fleet + physics + trace components | 5.8 | 2.3 | 2.2 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 2.5 | 1.8 | 0.0 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 2.2 | 1.3 | 0.0 |
| BO, fleet + physics + trace components + VALIDITY classifier | 4.0 | 1.5 | 0.2 |
| BO, no fleet data | 7.2 | 1.3 | 1.5 |

## w = 0.5

| strategy | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|
| random order | 31.8 | 18.8 | 4.8 |
| fleet model ranking, no updates | 9.2 | 8.5 | 1.0 |
| BO, fleet as prior (multi-task GP) | 2.5 | 2.0 | 0.0 |
| BO, fleet + trace PHYSICS features | 4.3 | 3.8 | 1.8 |
| BO, fleet + physics + trace components | 7.5 | 5.8 | 4.2 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 3.2 | 2.3 | 0.0 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 2.5 | 2.0 | 0.0 |
| BO, fleet + physics + trace components + VALIDITY classifier | 5.2 | 3.2 | 1.7 |
| BO, no fleet data | 3.5 | 3.3 | 0.2 |

## w = 0.0

| strategy | images to 2 % | images to 5 % | unusable images before 2 % |
|---|---|---|---|
| random order | 39.3 | 26.3 | 5.8 |
| fleet model ranking, no updates | 9.2 | 9.2 | 1.0 |
| BO, fleet as prior (multi-task GP) | 2.7 | 2.7 | 0.0 |
| BO, fleet + trace PHYSICS features | 8.5 | 4.0 | 5.5 |
| BO, fleet + physics + trace components | 7.0 | 2.5 | 3.8 |
| BO, fleet as prior (multi-task GP) + VALIDITY classifier | 3.0 | 3.0 | 0.0 |
| BO, fleet + trace PHYSICS features + VALIDITY classifier | 5.0 | 2.2 | 2.2 |
| BO, fleet + physics + trace components + VALIDITY classifier | 5.5 | 2.5 | 2.3 |
| BO, no fleet data | 4.8 | 3.8 | 0.7 |
