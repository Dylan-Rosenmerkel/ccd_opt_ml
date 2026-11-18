# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: nominal. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 1.0

| strategy | images to 2 % | images to 5 % |
|---|---|---|
| random order | 4.2 | 1.3 |
| fleet model ranking, no updates | 3.8 | 1.3 |
| BO, fleet as prior (multi-task GP) | 4.3 | 1.3 |
| BO, fleet + traces of every candidate | 3.0 | 1.8 |
| BO, fleet + trace PHYSICS features | 1.5 | 1.3 |
| BO, fleet + physics + trace components | 2.5 | 1.3 |
| BO, fleet + top-8 features learned on the fleet | 3.2 | 1.5 |
| BO, no fleet data | 5.5 | 2.0 |

## w = 0.5

| strategy | images to 2 % | images to 5 % |
|---|---|---|
| random order | 26.0 | 15.5 |
| fleet model ranking, no updates | 8.2 | 7.5 |
| BO, fleet as prior (multi-task GP) | 2.5 | 2.0 |
| BO, fleet + traces of every candidate | 3.3 | 2.5 |
| BO, fleet + trace PHYSICS features | 2.8 | 2.2 |
| BO, fleet + physics + trace components | 2.8 | 2.5 |
| BO, fleet + top-8 features learned on the fleet | 2.0 | 1.8 |
| BO, no fleet data | 4.5 | 3.5 |

## w = 0.0

| strategy | images to 2 % | images to 5 % |
|---|---|---|
| random order | 30.2 | 21.3 |
| fleet model ranking, no updates | 8.2 | 8.2 |
| BO, fleet as prior (multi-task GP) | 2.7 | 2.7 |
| BO, fleet + traces of every candidate | 3.0 | 2.3 |
| BO, fleet + trace PHYSICS features | 3.3 | 2.7 |
| BO, fleet + physics + trace components | 3.2 | 2.3 |
| BO, fleet + top-8 features learned on the fleet | 3.0 | 2.5 |
| BO, no fleet data | 3.5 | 3.3 |
