# Experiment 2: can simple non-expressive motion score high BPL or IMC?

Metric specificity and limitations, not an attempt to break the metrics. Synthetic motions are whole-body translations of a fixed pose built per real slice (same slices, beats, RMS, trim and mismatch pairs as the validity table); both metrics read only QoM (mean joint speed). Unit = song (FineDance) / piece (Motorica); values are unit means. * = differs from Real dance on own music (two-sided sign-flip p < 0.05 over units). r_band = Var(band-passed QoM) / Var(QoM) over the BPL window (band = beat rate ± 7.5 %; band_share.py). Chance floor of BPL ≈ 0.30 (random-cut shuffles, C2).

## Table 1. Scores on the own music

| Motion | Construction | FineDance (19 songs) BPL | FineDance IMC | FineDance r_band | Motorica (52 pieces) BPL | Motorica IMC | Motorica r_band |
|---|---|---:|---:|---:|---:|---:|---:|
| Real dance | reference | 0.459 | 0.219 | 5.52 % | 0.420 | 0.180 | 5.39 % |
| A1 sway, follows tracked beats | one speed peak per beat | 0.989* | -0.004* | 92.14 % | 0.974* | -0.009* | 87.45 % |
| A2 sway, metronome at own tempo | one speed peak per beat, no tempo fluctuation | 0.913* | 0.023* | 95.48 % | 0.784* | 0.009* | 95.84 % |
| A3 bounce, follows tracked beats | two speed peaks per beat | 0.772* | -0.008* | 0.02 % | 0.685* | -0.011* | 0.03 % |
| A4 bounce, metronome at own tempo | two speed peaks per beat, no tempo fluctuation | 0.103* | 0.022* | 0.00 % | 0.140* | -0.003* | 0.01 % |
| B1 speed = raw RMS | speed follows frame-level loudness | 0.644* | 1.000* | 10.59 % | 0.625* | 1.000* | 10.45 % |
| B2 speed = smoothed RMS (sigma 15) | speed follows the slow loudness envelope | 0.076* | 0.985* | 0.02 % | 0.105* | 0.977* | 0.01 % |
| MACE-Dance (main table) | generated | 0.302 | 0.038 | 3.70 % | 0.296 | 0.048 | 4.13 % |
| Ours (main table) | generated | 0.399 | 0.235 | 3.97 % | 0.395 | 0.235 | 4.25 % |

Model rows are copied from the main tables (FineDance seed 1; Motorica 3 samples per slice) for scale; their r_band is from band_share.py (same samples).

## Table 2. Own vs same-tempo mismatched music (3 tempo-matched songs, beats rescaled to the own tempo)

Does the score depend on WHICH song plays, or only on its tempo? Δ = own − mismatched, 95 % bootstrap CI over units, two-sided sign-flip p. "vs Real gap" = this row's Δ minus the Real-dance Δ.

| Metric | Motion | Dataset | own | mismatched | Δ [95 % CI] | p | vs Real gap |
|---|---|---|---:|---:|---:|---:|---:|
| BPL | Real dance | FineDance | 0.459 | 0.420 | +0.039 [+0.013, +0.065] | 0.011 | – |
| BPL | Real dance | Motorica | 0.420 | 0.384 | +0.037 [+0.018, +0.055] | <0.001 | – |
| BPL | A1 sway, follows tracked beats | FineDance | 0.989 | 0.854 | +0.135 [+0.094, +0.180] | <0.001 | +0.096 [+0.057, +0.140] |
| BPL | A1 sway, follows tracked beats | Motorica | 0.974 | 0.664 | +0.310 [+0.270, +0.350] | <0.001 | +0.274 [+0.229, +0.319] |
| BPL | A2 sway, metronome at own tempo | FineDance | 0.913 | 0.926 | -0.013 [-0.076, +0.045] | 0.695 | -0.052 [-0.115, +0.002] |
| BPL | A2 sway, metronome at own tempo | Motorica | 0.784 | 0.785 | -0.001 [-0.055, +0.051] | 0.958 | -0.038 [-0.090, +0.014] |
| BPL | A3 bounce, follows tracked beats | FineDance | 0.772 | 0.652 | +0.120 [+0.077, +0.168] | <0.001 | +0.081 [+0.035, +0.132] |
| BPL | A3 bounce, follows tracked beats | Motorica | 0.685 | 0.478 | +0.207 [+0.169, +0.247] | <0.001 | +0.170 [+0.130, +0.211] |
| BPL | A4 bounce, metronome at own tempo | FineDance | 0.103 | 0.101 | +0.002 [-0.011, +0.017] | 0.812 | -0.037 [-0.071, +0.000] |
| BPL | A4 bounce, metronome at own tempo | Motorica | 0.140 | 0.133 | +0.007 [-0.008, +0.024] | 0.434 | -0.030 [-0.057, -0.002] |
| BPL | B1 speed = raw RMS | FineDance | 0.644 | 0.606 | +0.038 [+0.002, +0.074] | 0.058 | -0.001 [-0.038, +0.035] |
| BPL | B1 speed = raw RMS | Motorica | 0.625 | 0.468 | +0.157 [+0.125, +0.191] | <0.001 | +0.120 [+0.086, +0.156] |
| IMC | Real dance | FineDance | 0.219 | 0.092 | +0.127 [+0.015, +0.230] | 0.038 | – |
| IMC | Real dance | Motorica | 0.180 | 0.011 | +0.169 [+0.124, +0.215] | <0.001 | – |
| IMC | B1 speed = raw RMS | FineDance | 1.000 | 0.092 | +0.908 [+0.835, +0.966] | <0.001 | +0.781 [+0.657, +0.909] |
| IMC | B1 speed = raw RMS | Motorica | 1.000 | 0.017 | +0.983 [+0.966, +0.999] | <0.001 | +0.814 [+0.768, +0.859] |
| IMC | B2 speed = smoothed RMS (sigma 15) | FineDance | 0.985 | 0.097 | +0.888 [+0.818, +0.952] | <0.001 | +0.762 [+0.635, +0.891] |
| IMC | B2 speed = smoothed RMS (sigma 15) | Motorica | 0.977 | 0.018 | +0.959 [+0.941, +0.978] | <0.001 | +0.790 [+0.744, +0.837] |

## Table 3. Temporal shuffling of real motion (200 permutations per slice)

C1 cuts at every k-th beat (segments keep their phase relative to the beat); C2 makes the same number of cuts at random frames (phase scrambled). Head and tail outside the cut region stay in place. Δ = shuffled − original real dance [95 % CI], p two-sided sign-flip.

| Shuffle | k beats | Dataset | BPL | Δ BPL | p | IMC | Δ IMC | p |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| none (Real dance) | – | FineDance | 0.459 | – | – | 0.219 | – | – |
| none (Real dance) | – | Motorica | 0.420 | – | – | 0.180 | – | – |
| C1 beat-aligned | 1 | FineDance | 0.439 | -0.020 [-0.048, +0.008] | 0.200 | 0.048 | -0.170 [-0.274, -0.062] | 0.006 |
| C1 beat-aligned | 1 | Motorica | 0.399 | -0.022 [-0.042, -0.002] | 0.049 | 0.082 | -0.097 [-0.135, -0.058] | <0.001 |
| C1 beat-aligned | 2 | FineDance | 0.450 | -0.008 [-0.036, +0.019] | 0.568 | 0.039 | -0.180 [-0.286, -0.072] | 0.005 |
| C1 beat-aligned | 2 | Motorica | 0.403 | -0.017 [-0.037, +0.003] | 0.103 | 0.079 | -0.101 [-0.140, -0.062] | <0.001 |
| C1 beat-aligned | 4 | FineDance | 0.451 | -0.008 [-0.038, +0.022] | 0.626 | 0.040 | -0.179 [-0.284, -0.076] | 0.004 |
| C1 beat-aligned | 4 | Motorica | 0.393 | -0.027 [-0.046, -0.008] | 0.009 | 0.073 | -0.106 [-0.144, -0.069] | <0.001 |
| C1 beat-aligned | 8 | FineDance | 0.455 | -0.004 [-0.030, +0.020] | 0.773 | 0.055 | -0.164 [-0.252, -0.068] | 0.003 |
| C1 beat-aligned | 8 | Motorica | 0.386 | -0.034 [-0.052, -0.016] | <0.001 | 0.081 | -0.098 [-0.133, -0.064] | <0.001 |
| C2 random cuts | 1 | FineDance | 0.304 | -0.155 [-0.238, -0.078] | <0.001 | 0.040 | -0.179 [-0.290, -0.069] | 0.007 |
| C2 random cuts | 1 | Motorica | 0.305 | -0.116 [-0.144, -0.088] | <0.001 | 0.077 | -0.102 [-0.141, -0.063] | <0.001 |
| C2 random cuts | 2 | FineDance | 0.311 | -0.148 [-0.225, -0.072] | 0.001 | 0.038 | -0.181 [-0.289, -0.071] | 0.006 |
| C2 random cuts | 2 | Motorica | 0.307 | -0.113 [-0.141, -0.086] | <0.001 | 0.076 | -0.103 [-0.142, -0.064] | <0.001 |
| C2 random cuts | 4 | FineDance | 0.319 | -0.140 [-0.212, -0.071] | 0.001 | 0.044 | -0.175 [-0.274, -0.070] | 0.005 |
| C2 random cuts | 4 | Motorica | 0.317 | -0.103 [-0.130, -0.078] | <0.001 | 0.070 | -0.109 [-0.149, -0.071] | <0.001 |
| C2 random cuts | 8 | FineDance | 0.353 | -0.106 [-0.163, -0.052] | 0.002 | 0.067 | -0.152 [-0.237, -0.062] | 0.005 |
| C2 random cuts | 8 | Motorica | 0.340 | -0.080 [-0.103, -0.058] | <0.001 | 0.087 | -0.093 [-0.126, -0.060] | <0.001 |
