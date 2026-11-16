| Control | Preserved | Randomized | Dataset | Original | Control | Δ [95% CI] | p (units) | p (MC) | RMS / max drift (beats) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Metronome grid (own tempo and mean phase) | tempo, mean phase | all local timing removed | FineDance | 0.459 | 0.455 | +0.003 [-0.014, +0.020] | 0.3476 | – | 0.06 / 0.17 |
|  |  |  | Motorica | 0.420 | 0.412 | +0.009 [-0.009, +0.027] | 0.1804 | – | 0.11 / 0.26 |
| IBI shuffle, global (no rescale) | interval set, first/last beat, mean interval | interval order -> drift | FineDance | 0.459 | 0.428 | +0.031 [+0.007, +0.054] | 0.0127 | 0.0045 | 0.16 / 0.33 |
|  |  |  | Motorica | 0.420 | 0.368 | +0.053 [+0.036, +0.070] | 0.0001 | 0.0005 | 0.23 / 0.47 |
| IBI shuffle, global, rescaled (= committed control) | interval set, mean interval | interval order -> drift | FineDance | 0.459 | 0.436 | +0.022 [+0.006, +0.038] | 0.0073 | 0.0095 | 0.16 / 0.33 |
|  |  |  | Motorica | 0.420 | 0.380 | +0.040 [+0.026, +0.054] | 0.0001 | 0.0005 | 0.23 / 0.47 |
| IBI shuffle within 4-beat blocks | interval set, every 4th beat position | interval order inside a bar | FineDance | 0.459 | 0.461 | -0.002 [-0.006, +0.001] | 0.8656 | 0.7811 | 0.04 / 0.12 |
|  |  |  | Motorica | 0.420 | 0.417 | +0.003 [+0.000, +0.006] | 0.0198 | 0.0460 | 0.05 / 0.17 |
| Deviation shuffle | line fit, deviation distribution | deviation order | FineDance | 0.459 | 0.433 | +0.026 [+0.005, +0.051] | 0.0112 | 0.0005 | 0.09 / 0.23 |
|  |  |  | Motorica | 0.420 | 0.364 | +0.057 [+0.040, +0.074] | 0.0001 | 0.0005 | 0.15 / 0.37 |
| Phase-randomized deviations (IAAFT) | line fit, deviation distribution and spectrum | deviation phases | FineDance | 0.459 | 0.441 | +0.017 [+0.000, +0.040] | 0.0443 | 0.0170 | 0.09 / 0.21 |
|  |  |  | Motorica | 0.420 | 0.387 | +0.033 [+0.020, +0.046] | 0.0001 | 0.0005 | 0.15 / 0.34 |
| Phase-randomized onset envelope -> tracker (= committed) | envelope spectrum, mean interval | onset timing | FineDance | 0.459 | 0.416 | +0.043 [+0.019, +0.068] | 0.0012 | 0.0005 | – |
|  |  |  | Motorica | 0.420 | 0.375 | +0.045 [+0.030, +0.060] | 0.0001 | 0.0005 | – |
| … only tracks without skipped/doubled beats | as above | as above | FineDance | 0.459 | 0.416 | +0.043 [+0.019, +0.068] | 0.0012 | 0.0005 | – |
|  |  |  | Motorica | 0.420 | 0.375 | +0.045 [+0.030, +0.060] | 0.0001 | 0.0005 | – |

Motion, band-pass (15 % around the own beat rate), Hilbert phase and 4 s trim are the BPL definition in every row; only the beat reference changes. Δ = Original − mean control per unit (FineDance 19 songs, Motorica 52 pieces); p (units) = one-sided sign-flip permutation (10,000); p (MC) = observed mean over units against 2,000 draws of one surrogate per slice. Drift = RMS and max |surrogate beat − own beat| over the clip, in beats. No control rolls the motion, so none has a seam. FineDance: own-beat deviation RMS 0.93 frames, lag-1 IBI autocorr -0.19 · Motorica: own-beat deviation RMS 1.66 frames, lag-1 IBI autocorr -0.02. Tracker tracks with a skipped/doubled beat: FineDance 0.2% · Motorica 0.8%.
