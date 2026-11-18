# T1: images of a new amplifier needed (leave one amplifier out)

Amplifiers: 101:0, 101:1, 101:3; packets in 5 time blocks. The left-out amplifier is scored on one block; the fleet (the other amplifiers) trains on the other blocks; its k images come from the other blocks. Error = RMS of the log error x 100 (%), mean over draws, blocks and amplifiers. Regret = how much worse (%) the setting the model ranks best in the block is than the best in the block.

Label noise at nominal (%): sigma_e 1.8, lambda 4.1, gain 1.0

## sigma_e

| model | k=0 error / regret | k=1 error / regret | k=2 error / regret | k=4 error / regret | k=8 error / regret |
|---|---|---|---|---|---|
| fleet mean | 20.5 / – | 20.5 / – | 20.5 / – | 20.5 / – | 20.5 / – |
| shared GP (+ offset) | 17.4 / 14.3 | 11.2 / 14.3 | 10.3 / 14.3 | 9.8 / 14.3 | 9.5 / 14.3 |
| multi-task GP | 15.5 / 14.4 | 12.0 / 11.4 | 11.1 / 11.4 | 10.3 / 11.8 | 9.7 / 11.6 |
| trace trees (+ offset) | 13.5 / 5.4 | 12.2 / 5.4 | 10.9 / 5.4 | 10.3 / 5.4 | 9.9 / 5.4 |
| mean of the k images | – | 17.9 / – | 16.0 / – | 15.2 / – | 14.2 / – |

## lambda

| model | k=0 error / regret | k=1 error / regret | k=2 error / regret | k=4 error / regret | k=8 error / regret |
|---|---|---|---|---|---|
| fleet mean | 63.1 / – | 63.1 / – | 63.1 / – | 63.1 / – | 63.1 / – |
| shared GP (+ offset) | 51.3 / 4.3 | 29.9 / 4.3 | 25.9 / 4.3 | 24.8 / 4.3 | 24.0 / 4.3 |
| multi-task GP | 53.0 / 4.3 | 32.9 / 4.4 | 27.9 / 4.4 | 25.5 / 4.5 | 22.5 / 4.6 |
| trace trees (+ offset) | 52.2 / 35.8 | 39.1 / 35.8 | 36.3 / 35.8 | 33.7 / 35.8 | 32.3 / 35.8 |
| mean of the k images | – | 48.7 / – | 43.6 / – | 40.9 / – | 39.6 / – |

## gain

| model | k=0 error / regret | k=1 error / regret | k=2 error / regret | k=4 error / regret | k=8 error / regret |
|---|---|---|---|---|---|
| fleet mean | 3.7 / – | 3.7 / – | 3.7 / – | 3.7 / – | 3.7 / – |
| shared GP (+ offset) | 3.3 / 1.9 | 1.7 / 1.9 | 1.4 / 1.9 | 1.3 / 1.9 | 1.3 / 1.9 |
| multi-task GP | 3.2 / 1.9 | 1.7 / 1.9 | 1.5 / 1.9 | 1.4 / 2.0 | 1.3 / 1.9 |
| trace trees (+ offset) | 5.0 / 4.2 | 2.7 / 4.2 | 2.3 / 4.2 | 2.2 / 4.2 | 2.1 / 4.2 |
| mean of the k images | – | 2.7 / – | 2.3 / – | 2.2 / – | 2.1 / – |

## joint objective (w = 0.5), regret %

| model | k=0 | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| shared GP (+ offset) | 4.1 | 4.1 | 4.1 | 4.1 | 4.1 |
| multi-task GP | 4.1 | 4.5 | 4.9 | 5.0 | 4.7 |
| trace trees (+ offset) | 17.2 | 17.2 | 17.2 | 17.2 | 17.2 |

Multi-task GP, learned on the fleet (per left-out amplifier and block): rho = how alike the amplifiers' responses are (1 = the same shape), offset sd = how far apart their levels are (log).

* sigma_e: 101:0 block 0 rho 1.00, offset sd 0.17; 101:0 block 1 rho 1.00, offset sd 0.15; 101:0 block 2 rho 1.00, offset sd 0.15; 101:0 block 3 rho 1.00, offset sd 0.14; 101:0 block 4 rho 1.00, offset sd 0.16; 101:1 block 0 rho 0.90, offset sd 0.14; 101:1 block 1 rho 0.97, offset sd 0.18; 101:1 block 2 rho 1.00, offset sd 0.16; 101:1 block 3 rho 1.00, offset sd 0.16; 101:1 block 4 rho 1.00, offset sd 0.16; 101:3 block 0 rho 0.90, offset sd 0.09; 101:3 block 1 rho 0.90, offset sd 0.00; 101:3 block 2 rho 0.97, offset sd 0.00; 101:3 block 3 rho 0.97, offset sd 0.00; 101:3 block 4 rho 0.97, offset sd 0.09
* lambda: 101:0 block 0 rho 1.00, offset sd 0.34; 101:0 block 1 rho 1.00, offset sd 0.33; 101:0 block 2 rho 1.00, offset sd 0.40; 101:0 block 3 rho 1.00, offset sd 0.32; 101:0 block 4 rho 1.00, offset sd 0.39; 101:1 block 0 rho 0.50, offset sd 0.00; 101:1 block 1 rho 0.50, offset sd 0.00; 101:1 block 2 rho 0.50, offset sd 0.00; 101:1 block 3 rho 0.50, offset sd 0.00; 101:1 block 4 rho 0.50, offset sd 0.00; 101:3 block 0 rho 0.75, offset sd 0.00; 101:3 block 1 rho 0.75, offset sd 0.00; 101:3 block 2 rho 0.90, offset sd 0.00; 101:3 block 3 rho 0.75, offset sd 0.00; 101:3 block 4 rho 0.90, offset sd 0.00
* gain: 101:0 block 0 rho 0.90, offset sd 0.02; 101:0 block 1 rho 0.75, offset sd 0.02; 101:0 block 2 rho 0.90, offset sd 0.02; 101:0 block 3 rho 0.90, offset sd 0.02; 101:0 block 4 rho 0.90, offset sd 0.02; 101:1 block 0 rho 1.00, offset sd 0.03; 101:1 block 1 rho 1.00, offset sd 0.03; 101:1 block 2 rho 1.00, offset sd 0.03; 101:1 block 3 rho 1.00, offset sd 0.03; 101:1 block 4 rho 1.00, offset sd 0.03; 101:3 block 0 rho 0.50, offset sd 0.00; 101:3 block 1 rho 0.50, offset sd 0.00; 101:3 block 2 rho 0.50, offset sd 0.00; 101:3 block 3 rho 0.75, offset sd 0.00; 101:3 block 4 rho 0.50, offset sd 0.00
