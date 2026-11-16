# Beat-band share diagnostic for BPL

BPL keeps only the phase of the band-passed movement speed, so a weak but phase-stable beat-band component can score high (Experiment 2: a bounce with 0.02-0.03 % of its speed variance in the band reaches BPL 0.77 / 0.69). r_band = Var(q_b) / Var(q): share of the QoM (mean joint speed) variance inside the BPL band (beat rate ± 7.5 %), over the BPL window (trim 4 s). A diagnostic reported next to BPL, not a replacement.

## Table 1. BPL and r_band per model

Unit mean = mean over songs / pieces of the slice means [95 % bootstrap CI]; median [IQR] over clips. "< real p5 / p25" = share of clips whose r_band is below the 5th / 25th percentile of real dance.

| Dataset | Model | BPL | r_band % (unit mean [CI]) | r_band % median [IQR] | < real p5 | < real p25 | Spearman BPL vs r_band (clips) |
|---|---|---:|---:|---:|---:|---:|---:|
| FineDance (19 songs, seed 1) | Real dance | 0.459 | 5.52 [3.31, 7.99] | 2.31 [1.04, 7.70] | 6 % | 25 % | +0.57 |
| FineDance (19 songs, seed 1) | Lodge | 0.289 | 5.20 [3.99, 6.60] | 4.69 [2.93, 7.47] | 0 % | 1 % | -0.39 |
| FineDance (19 songs, seed 1) | MEGA | 0.279 | 3.42 [2.64, 4.32] | 2.80 [1.62, 4.46] | 0 % | 13 % | -0.02 |
| FineDance (19 songs, seed 1) | MACE-Dance | 0.302 | 3.70 [2.99, 4.41] | 3.49 [1.76, 4.54] | 0 % | 12 % | +0.08 |
| FineDance (19 songs, seed 1) | FlowerDance | 0.380 | 2.71 [2.03, 3.51] | 2.18 [1.15, 3.81] | 1 % | 24 % | +0.39 |
| FineDance (19 songs, seed 1) | Ours (critic v5 + gain) | 0.399 | 3.97 [3.16, 4.79] | 3.75 [1.92, 5.33] | 0 % | 10 % | +0.05 |
| FineDance (19 songs, seed 1) | *synthetic A3 bounce (BPL 0.772)* | – | 0.02 | – | – | – | – |
| FineDance (19 songs, seed 1) | *synthetic A1 sway (BPL 0.989)* | – | 92.14 | – | – | – | – |
| Motorica (52 pieces, 3 samples per slice) | Real dance | 0.420 | 5.39 [4.51, 6.31] | 4.18 [1.87, 8.21] | 5 % | 25 % | +0.40 |
| Motorica (52 pieces, 3 samples per slice) | Lodge | 0.290 | 4.79 [4.28, 5.33] | 4.17 [3.09, 5.84] | 0 % | 4 % | -0.27 |
| Motorica (52 pieces, 3 samples per slice) | MEGA | 0.285 | 5.03 [4.44, 5.64] | 4.40 [2.60, 6.65] | 1 % | 12 % | +0.06 |
| Motorica (52 pieces, 3 samples per slice) | MACE-Dance | 0.296 | 4.13 [3.76, 4.53] | 3.76 [2.88, 5.06] | 0 % | 8 % | +0.12 |
| Motorica (52 pieces, 3 samples per slice) | Ours (critic v5 + gain) | 0.395 | 4.25 [3.78, 4.83] | 3.59 [2.54, 5.01] | 0 % | 9 % | +0.19 |
| Motorica (52 pieces, 3 samples per slice) | *synthetic A3 bounce (BPL 0.685)* | – | 0.03 | – | – | – | – |
| Motorica (52 pieces, 3 samples per slice) | *synthetic A1 sway (BPL 0.974)* | – | 87.45 | – | – | – | – |

## Table 2. Paired unit-level differences

Δ = A − B over songs / pieces [95 % bootstrap CI], two-sided sign-flip p (10,000).

| Dataset | A − B | Δ BPL | p | Δ r_band (pp) | p |
|---|---|---:|---:|---:|---:|
| FineDance | Ours (critic v5 + gain) − Real dance | -0.059 [-0.148, +0.027] | 0.203 | -1.54 [-3.55, +0.18] | 0.132 |
| FineDance | Ours (critic v5 + gain) − MACE-Dance | +0.097 [+0.048, +0.140] | <0.001 | +0.27 [-0.11, +0.63] | 0.194 |
| FineDance | MACE-Dance − Real dance | -0.156 [-0.246, -0.072] | 0.004 | -1.81 [-3.81, -0.03] | 0.089 |
| FineDance | Lodge − Real dance | -0.170 [-0.253, -0.087] | 0.002 | -0.32 [-3.08, +2.45] | 0.824 |
| FineDance | MEGA − Real dance | -0.180 [-0.274, -0.091] | 0.001 | -2.10 [-3.96, -0.40] | 0.042 |
| FineDance | FlowerDance − Real dance | -0.079 [-0.148, -0.018] | 0.025 | -2.81 [-4.65, -1.12] | 0.006 |
| Motorica | Ours (critic v5 + gain) − Real dance | -0.025 [-0.060, +0.011] | 0.172 | -1.14 [-2.25, -0.01] | 0.054 |
| Motorica | Ours (critic v5 + gain) − MACE-Dance | +0.099 [+0.079, +0.120] | <0.001 | +0.12 [-0.28, +0.56] | 0.580 |
| Motorica | MACE-Dance − Real dance | -0.125 [-0.154, -0.096] | <0.001 | -1.26 [-2.28, -0.29] | 0.014 |
| Motorica | Lodge − Real dance | -0.130 [-0.161, -0.100] | <0.001 | -0.60 [-1.62, +0.44] | 0.259 |
| Motorica | MEGA − Real dance | -0.136 [-0.167, -0.104] | <0.001 | -0.36 [-1.39, +0.60] | 0.502 |

## Supplementary: amplitude-weighted variants (not validated; not replacements for BPL)

BPL_aw = |mean |A| e^{iΔφ}| / mean |A| weights frames by the band amplitude but stays scale-free, so it does not penalise a weak band. r_lock = |mean A e^{-iθ}|² / (2 Var q) = share of the speed variance carried by the beat-LOCKED sinusoid (≈ BPL_aw² × r_band): high only when the beat component is both strong and phase-stable.

| Dataset | Model | BPL_aw | r_lock % (unit mean [CI]) | r_lock % median | Δ r_lock vs MACE-Dance (pp), p | Δ r_lock vs Real (pp), p |
|---|---|---:|---:|---:|---:|---:|
| FineDance | Real dance | 0.550 | 2.49 [1.15, 4.11] | 0.37 | – | – |
| FineDance | Lodge | 0.362 | 0.53 [0.43, 0.64] | 0.42 | – | -1.96 [-3.50, -0.70], 0.007 |
| FineDance | MEGA | 0.344 | 0.36 [0.25, 0.48] | 0.23 | – | -2.13 [-3.68, -0.83], 0.002 |
| FineDance | MACE-Dance | 0.385 | 0.51 [0.36, 0.67] | 0.31 | – | -1.98 [-3.54, -0.72], 0.005 |
| FineDance | FlowerDance | 0.440 | 0.62 [0.37, 0.91] | 0.23 | – | -1.87 [-3.32, -0.63], 0.004 |
| FineDance | Ours (critic v5 + gain) | 0.490 | 0.86 [0.63, 1.12] | 0.55 | +0.35 [+0.17, +0.54], 0.002 | -1.62 [-3.14, -0.36], 0.034 |
| Motorica | Real dance | 0.528 | 1.59 [1.25, 1.96] | 0.73 | – | – |
| Motorica | Lodge | 0.348 | 0.53 [0.45, 0.61] | 0.42 | – | -1.07 [-1.43, -0.72], <0.001 |
| Motorica | MEGA | 0.345 | 0.62 [0.49, 0.76] | 0.31 | – | -0.98 [-1.36, -0.61], <0.001 |
| Motorica | MACE-Dance | 0.353 | 0.53 [0.46, 0.61] | 0.40 | – | -1.06 [-1.44, -0.72], <0.001 |
| Motorica | Ours (critic v5 + gain) | 0.466 | 0.92 [0.77, 1.07] | 0.65 | +0.38 [+0.23, +0.55], <0.001 | -0.68 [-1.11, -0.25], 0.002 |
