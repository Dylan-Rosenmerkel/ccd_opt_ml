# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: nominal. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 1.0

| strategy | images to 2 % | images to 5 % |
|---|---|---|
| random order | 4.5 | 1.5 |
| fleet model ranking, no updates | 4.5 | 1.3 |
| BO, fleet as prior (multi-task GP) | 4.7 | 1.3 |
| BO, fleet + traces of every candidate | 4.7 | 1.3 |
| BO, fleet + trace PHYSICS features | 3.2 | 1.3 |
| BO, no fleet data | 4.3 | 1.3 |

## w = 0.5

| strategy | images to 2 % | images to 5 % |
|---|---|---|
| random order | 39.5 | 34.0 |
| fleet model ranking, no updates | 15.3 | 15.3 |
| BO, fleet as prior (multi-task GP) | 14.0 | 14.0 |
| BO, fleet + traces of every candidate | 3.0 | 3.0 |
| BO, fleet + trace PHYSICS features | 7.8 | 7.8 |
| BO, no fleet data | 10.8 | 10.0 |

## w = 0.0

| strategy | images to 2 % | images to 5 % |
|---|---|---|
| random order | 42.3 | 38.5 |
| fleet model ranking, no updates | 15.0 | 15.0 |
| BO, fleet as prior (multi-task GP) | 14.7 | 13.8 |
| BO, fleet + traces of every candidate | 2.7 | 2.7 |
| BO, fleet + trace PHYSICS features | 5.0 | 3.7 |
| BO, no fleet data | 19.0 | 17.5 |
