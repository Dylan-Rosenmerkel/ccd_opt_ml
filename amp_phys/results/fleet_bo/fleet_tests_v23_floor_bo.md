# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: nominal. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 1.0

| strategy | images to 2 % | images to 5 % |
|---|---|---|
| random order | 16.3 | 7.7 |
| fleet model ranking, no updates | 15.8 | 14.5 |
| BO, fleet as prior (multi-task GP) | 2.5 | 2.2 |
| BO, fleet + traces of every candidate | 4.0 | 3.7 |
| BO, fleet + trace PHYSICS features | 6.8 | 4.7 |
| BO, fleet + physics + trace components | 7.0 | 5.3 |
| BO, fleet + top-8 features learned on the fleet | 6.2 | 4.3 |
| BO, no fleet data | 6.2 | 5.0 |

## w = 0.5

| strategy | images to 2 % | images to 5 % |
|---|---|---|
| random order | 33.3 | 18.2 |
| fleet model ranking, no updates | 12.3 | 3.8 |
| BO, fleet as prior (multi-task GP) | 17.0 | 13.0 |
| BO, fleet + traces of every candidate | 4.2 | 3.3 |
| BO, fleet + trace PHYSICS features | 7.0 | 5.8 |
| BO, fleet + physics + trace components | 10.7 | 2.8 |
| BO, fleet + top-8 features learned on the fleet | 9.3 | 3.3 |
| BO, no fleet data | 16.8 | 13.2 |

## w = 0.0

| strategy | images to 2 % | images to 5 % |
|---|---|---|
| random order | 53.8 | 36.0 |
| fleet model ranking, no updates | 15.5 | 15.5 |
| BO, fleet as prior (multi-task GP) | 5.2 | 5.2 |
| BO, fleet + traces of every candidate | 6.5 | 5.8 |
| BO, fleet + trace PHYSICS features | 3.8 | 3.8 |
| BO, fleet + physics + trace components | 6.5 | 6.3 |
| BO, fleet + top-8 features learned on the fleet | 6.3 | 6.2 |
| BO, no fleet data | 6.7 | 6.5 |
