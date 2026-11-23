# T1: images of a new amplifier needed (leave one amplifier out)

Amplifiers: 101:0, 101:1, 101:3; packets in 5 time blocks. The left-out amplifier is scored on one block; the fleet (the other amplifiers) trains on the other blocks; its k images come from the other blocks. Error = RMS of the log error x 100 (%), mean over draws, blocks and amplifiers. Regret = how much worse (%) the setting the model ranks best in the block is than the best in the block.

Label noise at nominal (%): sigma_e 1.4, lambda 3.8, gain 1.1

## sigma_e

| model | k=0 error / regret | k=1 error / regret | k=2 error / regret | k=4 error / regret | k=8 error / regret |
|---|---|---|---|---|---|
| fleet mean | 17.6 / – | 17.6 / – | 17.6 / – | 17.6 / – | 17.6 / – |
| shared GP (+ offset) | 16.9 / 3.8 | 10.1 / 3.8 | 8.9 / 3.8 | 8.4 / 3.8 | 8.1 / 3.8 |
| multi-task GP | 16.4 / 3.5 | 11.2 / 4.0 | 9.8 / 3.8 | 8.7 / 3.3 | 7.9 / 3.0 |
| trace trees (+ offset) | 8.6 / 2.4 | 9.3 / 2.4 | 8.1 / 2.4 | 7.6 / 2.4 | 7.3 / 2.4 |
| mean of the k images | – | 10.8 / – | 9.7 / – | 9.0 / – | 8.6 / – |

## lambda

| model | k=0 error / regret | k=1 error / regret | k=2 error / regret | k=4 error / regret | k=8 error / regret |
|---|---|---|---|---|---|
| fleet mean | 69.3 / – | 69.3 / – | 69.3 / – | 69.3 / – | 69.3 / – |
| shared GP (+ offset) | 57.8 / 1.2 | 31.7 / 1.2 | 28.2 / 1.2 | 26.6 / 1.2 | 25.5 / 1.2 |
| multi-task GP | 66.5 / 15.3 | 44.2 / 8.9 | 38.0 / 5.2 | 31.1 / 2.1 | 25.4 / 1.2 |
| trace trees (+ offset) | 43.8 / 41.2 | 36.8 / 41.2 | 32.8 / 41.2 | 30.6 / 41.2 | 29.5 / 41.2 |
| mean of the k images | – | 47.5 / – | 42.5 / – | 39.8 / – | 38.3 / – |

## gain

| model | k=0 error / regret | k=1 error / regret | k=2 error / regret | k=4 error / regret | k=8 error / regret |
|---|---|---|---|---|---|
| fleet mean | 3.8 / – | 3.8 / – | 3.8 / – | 3.8 / – | 3.8 / – |
| shared GP (+ offset) | 3.4 / 1.8 | 1.9 / 1.8 | 1.7 / 1.8 | 1.5 / 1.8 | 1.5 / 1.8 |
| multi-task GP | 3.3 / 1.8 | 1.9 / 1.9 | 1.7 / 1.9 | 1.5 / 1.9 | 1.5 / 1.9 |
| trace trees (+ offset) | 4.6 / 2.8 | 3.2 / 2.8 | 2.8 / 2.8 | 2.7 / 2.8 | 2.5 / 2.8 |
| mean of the k images | – | 3.1 / – | 2.7 / – | 2.5 / – | 2.4 / – |

## joint objective (w = 0.5), regret %

| model | k=0 | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| shared GP (+ offset) | 2.9 | 2.9 | 2.9 | 2.9 | 2.9 |
| multi-task GP | 5.9 | 5.0 | 3.9 | 1.8 | 1.2 |
| trace trees (+ offset) | 20.8 | 20.8 | 20.8 | 20.8 | 20.8 |

Multi-task GP, learned on the fleet (per left-out amplifier and block): rho = how alike the amplifiers' responses are (1 = the same shape), offset sd = how far apart their levels are (log).

* sigma_e: 101:0 block 0 rho 1.00, offset sd 0.11; 101:0 block 1 rho 0.90, offset sd 0.10; 101:0 block 2 rho 0.90, offset sd 0.10; 101:0 block 3 rho 0.90, offset sd 0.12; 101:0 block 4 rho 1.00, offset sd 0.10; 101:1 block 0 rho 0.50, offset sd 0.14; 101:1 block 1 rho 0.50, offset sd 0.15; 101:1 block 2 rho 0.50, offset sd 0.11; 101:1 block 3 rho 0.50, offset sd 0.16; 101:1 block 4 rho 0.50, offset sd 0.12; 101:3 block 0 rho 0.75, offset sd 0.07; 101:3 block 1 rho 0.90, offset sd 0.05; 101:3 block 2 rho 0.90, offset sd 0.06; 101:3 block 3 rho 0.50, offset sd 0.03; 101:3 block 4 rho 0.97, offset sd 0.07
* lambda: 101:0 block 0 rho 1.00, offset sd 0.28; 101:0 block 1 rho 1.00, offset sd 0.28; 101:0 block 2 rho 1.00, offset sd 0.26; 101:0 block 3 rho 1.00, offset sd 0.34; 101:0 block 4 rho 1.00, offset sd 0.24; 101:1 block 0 rho 0.00, offset sd 0.00; 101:1 block 1 rho 0.00, offset sd 0.00; 101:1 block 2 rho 0.00, offset sd 0.00; 101:1 block 3 rho 0.00, offset sd 0.16; 101:1 block 4 rho 0.50, offset sd 0.54; 101:3 block 0 rho 0.50, offset sd 0.00; 101:3 block 1 rho 0.50, offset sd 0.00; 101:3 block 2 rho 0.50, offset sd 0.00; 101:3 block 3 rho 0.50, offset sd 0.20; 101:3 block 4 rho 0.50, offset sd 0.18
* gain: 101:0 block 0 rho 0.75, offset sd 0.02; 101:0 block 1 rho 0.75, offset sd 0.02; 101:0 block 2 rho 0.75, offset sd 0.02; 101:0 block 3 rho 0.75, offset sd 0.02; 101:0 block 4 rho 0.75, offset sd 0.02; 101:1 block 0 rho 1.00, offset sd 0.04; 101:1 block 1 rho 1.00, offset sd 0.03; 101:1 block 2 rho 1.00, offset sd 0.03; 101:1 block 3 rho 1.00, offset sd 0.05; 101:1 block 4 rho 1.00, offset sd 0.03; 101:3 block 0 rho 0.50, offset sd 0.00; 101:3 block 1 rho 0.75, offset sd 0.00; 101:3 block 2 rho 0.75, offset sd 0.00; 101:3 block 3 rho 0.75, offset sd 0.00; 101:3 block 4 rho 0.75, offset sd 0.00
