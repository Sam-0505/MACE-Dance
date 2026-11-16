**Identifying the original music among 4 candidates (original + 3 tempo-matched mismatches), real dance**

| Score | FineDance top-1 | FineDance MRR | Motorica top-1 | Motorica MRR |
|---|---:|---:|---:|---:|
| Chance | 0.250 | 0.521 | 0.250 | 0.521 |
| BAS | 0.190 [0.105, 0.294] | 0.485 [0.420, 0.562] | 0.298 [0.242, 0.355] | 0.555 [0.517, 0.594] |
| BPL | 0.393 [0.287, 0.507] | 0.619 [0.554, 0.690] | 0.349 [0.293, 0.408] | 0.589 [0.550, 0.628] |
| IMC | 0.429 [0.323, 0.544] | 0.637 [0.569, 0.710] | 0.384 [0.332, 0.438] | 0.620 [0.585, 0.654] |
| BPL+IMC | 0.452 [0.345, 0.580] | 0.653 [0.583, 0.734] | 0.426 [0.368, 0.483] | 0.641 [0.603, 0.679] |
| BAS+BPL+IMC | 0.393 [0.299, 0.493] | 0.623 [0.562, 0.688] | 0.426 [0.369, 0.484] | 0.643 [0.607, 0.681] |

**Paired differences (held-out)**

| Comparison | FineDance Δ top-1 | p | FineDance Δ MRR | p | Motorica Δ top-1 | p | Motorica Δ MRR | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BAS+BPL+IMC − BAS | +0.202 [+0.064, +0.342] | 0.022 | +0.138 [+0.037, +0.239] | 0.023 | +0.128 [+0.054, +0.201] | 0.002 | +0.088 [+0.039, +0.137] | 0.002 |
| BPL+IMC − BAS | +0.262 [+0.119, +0.416] | 0.005 | +0.168 [+0.070, +0.269] | 0.006 | +0.128 [+0.057, +0.201] | 0.003 | +0.087 [+0.037, +0.135] | 0.001 |
| BAS+BPL+IMC − BPL+IMC | -0.060 [-0.121, -0.019] | 0.062 | -0.030 [-0.065, -0.003] | 0.061 | +0.000 [-0.010, +0.010] | 1.000 | +0.002 [-0.003, +0.007] | 0.626 |
| BPL+IMC − IMC | +0.024 [-0.057, +0.110] | 0.779 | +0.016 [-0.030, +0.067] | 0.559 | +0.042 [-0.011, +0.096] | 0.184 | +0.022 [-0.010, +0.053] | 0.191 |
| BPL+IMC − BPL | +0.060 [-0.045, +0.188] | 0.448 | +0.034 [-0.028, +0.105] | 0.343 | +0.076 [+0.021, +0.133] | 0.015 | +0.052 [+0.014, +0.091] | 0.011 |
| BPL − BAS | +0.202 [+0.080, +0.339] | 0.014 | +0.134 [+0.040, +0.229] | 0.016 | +0.052 [-0.026, +0.129] | 0.234 | +0.034 [-0.021, +0.088] | 0.230 |
| IMC − BAS | +0.238 [+0.116, +0.364] | 0.005 | +0.152 [+0.064, +0.240] | 0.006 | +0.087 [+0.004, +0.169] | 0.052 | +0.065 [+0.015, +0.117] | 0.019 |

**Learned weights (z-scored features; mean [min, max] over leave-one-song-out folds)**

| Model | Dataset | BAS | BPL | IMC |
|---|---|---:|---:|---:|
| BPL+IMC | FineDance | – | +0.78 [+0.36, +1.53] | +0.46 [+0.38, +0.58] |
| BPL+IMC | Motorica | – | +0.41 [+0.31, +0.45] | +0.49 [+0.43, +0.53] |
| BAS+BPL+IMC | FineDance | -0.23 [-0.35, -0.03] | +0.75 [+0.08, +1.89] | +0.44 [+0.18, +0.58] |
| BAS+BPL+IMC | Motorica | -0.02 [-0.03, +0.05] | +0.36 [+0.29, +0.45] | +0.46 [+0.42, +0.52] |

Real-dance slices (FineDance: 84 slices, 19 songs · Motorica: 289 slices, 52 songs) ranked against their own music and the 3 tempo-matched mismatched slices of the validity table. For BPL and BAS the mismatched beat lists are rescaled to the own mean beat interval (validity-table protocol), so the task is distinguishing the original music from tempo-controlled mismatches, not unrestricted music retrieval. Single scores rank by the raw value (higher = better, no fitting). Combinations: conditional-logit ranking model (softmax over the 4 candidates, L2 penalty chosen by inner song-grouped 5-fold CV), features z-scored on the training fold only; outer CV leave-one-song-out (FineDance song / Motorica music piece), so no song is in both training and test. Ties broken uniformly at random (in expectation). 95 % CIs: song-cluster bootstrap (10,000). p: two-sided song-level sign-flip permutation (10,000).
