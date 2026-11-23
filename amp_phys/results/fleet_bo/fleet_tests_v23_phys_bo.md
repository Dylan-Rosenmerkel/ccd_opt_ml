# T2: Bayesian optimisation replayed on the measured packets

Each amplifier in turn is the new CCD; its measured settings in one of 2 time blocks are the candidates, the fleet is the other amplifiers in the other blocks. Start: nominal. Images needed (mean over amplifiers) until the best image so far is within 2 % / 5 % of the best in the pool (J = w log sigma_e + (1 - w) log lambda).

## w = 1.0

| strategy | images to 2 % | images to 5 % |
|---|---|---|
| random order | 15.8 | 7.3 |
| fleet model ranking, no updates | 13.3 | 11.3 |
| BO, fleet as prior (multi-task GP) | 4.7 | 3.3 |
| BO, fleet + traces of every candidate | 11.0 | 9.3 |
| BO, fleet + trace PHYSICS features | 7.8 | 5.5 |
| BO, no fleet data | 7.5 | 5.8 |

## w = 0.5

| strategy | images to 2 % | images to 5 % |
|---|---|---|
| random order | 56.2 | 48.7 |
| fleet model ranking, no updates | 28.8 | 28.2 |
| BO, fleet as prior (multi-task GP) | 33.8 | 33.3 |
| BO, fleet + traces of every candidate | 10.2 | 10.2 |
| BO, fleet + trace PHYSICS features | 13.3 | 13.3 |
| BO, no fleet data | 21.0 | 19.3 |

## w = 0.0

| strategy | images to 2 % | images to 5 % |
|---|---|---|
| random order | 65.0 | 56.5 |
| fleet model ranking, no updates | 48.0 | 48.0 |
| BO, fleet as prior (multi-task GP) | 25.8 | 25.7 |
| BO, fleet + traces of every candidate | 6.3 | 6.2 |
| BO, fleet + trace PHYSICS features | 8.5 | 5.5 |
| BO, no fleet data | 13.5 | 12.5 |
