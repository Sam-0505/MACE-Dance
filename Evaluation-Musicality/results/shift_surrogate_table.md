| Dataset | Condition | Original | Control | Δ | 95% CI | p (units) | p (MC null) | Slices > null 95th pct | Mean percentile of Original | Detected? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Beat-Phase Locking (BPL, offset-free)** | | | | | | | | | | |
| FineDance | Circular shift | 0.459 | 0.405 | +0.053 | [+0.003, +0.110] | 0.0343 | 0.0010 | 17% | 0.58 | Yes |
|  | Random shift (same song) | 0.451 | 0.444 | +0.007 | [-0.013, +0.029] | 0.3005 | 0.2114 | 6% | 0.49 | No |
|  | Phase-randomized | 0.459 | 0.418 | +0.041 | [+0.017, +0.066] | 0.0018 | 0.0025 | 5% | 0.61 | Yes |
|  | IBI-shuffled beats (extra) | 0.459 | 0.437 | +0.022 | [+0.006, +0.038] | 0.0092 | 0.0080 | 6% | 0.59 | Yes |
|  | Metronome grid, own phase (extra) | 0.459 | 0.455 | +0.003 | [-0.014, +0.020] | 0.3476 | – | – | – | No |
|  | Mismatched song | 0.459 | 0.420 | +0.039 | [+0.013, +0.065] | 0.0063 | – | – | – | Yes |
| Motorica | Circular shift | 0.420 | 0.369 | +0.051 | [+0.027, +0.075] | 0.0001 | 0.0005 | 20% | 0.60 | Yes |
|  | Random shift (same song) | 0.406 | 0.390 | +0.016 | [-0.001, +0.034] | 0.0422 | 0.0040 | 12% | 0.55 | Yes |
|  | Phase-randomized | 0.420 | 0.377 | +0.044 | [+0.029, +0.059] | 0.0001 | 0.0005 | 10% | 0.59 | Yes |
|  | IBI-shuffled beats (extra) | 0.420 | 0.381 | +0.039 | [+0.025, +0.054] | 0.0001 | 0.0005 | 11% | 0.60 | Yes |
|  | Metronome grid, own phase (extra) | 0.420 | 0.412 | +0.009 | [-0.009, +0.027] | 0.1804 | – | – | – | No |
|  | Mismatched song | 0.420 | 0.384 | +0.037 | [+0.018, +0.055] | 0.0002 | – | – | – | Yes |
| **BPL-fixed: offset-sensitive, one dataset-wide offset (leave-one-unit-out)** | | | | | | | | | | |
| FineDance | Circular shift | 0.052 | 0.000 | +0.052 | [-0.056, +0.152] | 0.1722 | 0.1449 | 12% | 0.56 | No |
|  | Random shift (same song) | 0.017 | 0.007 | +0.010 | [-0.093, +0.101] | 0.4309 | 0.4058 | 10% | 0.54 | No |
|  | Phase-randomized | 0.052 | -0.014 | +0.067 | [-0.033, +0.161] | 0.1029 | 0.0825 | 11% | 0.58 | No |
|  | IBI-shuffled beats (extra) | 0.052 | 0.051 | +0.001 | [-0.037, +0.040] | 0.4747 | 0.4683 | 11% | 0.52 | No |
|  | Metronome grid, own phase (extra) | 0.052 | 0.037 | +0.015 | [-0.007, +0.040] | 0.1191 | – | – | – | No |
|  | Mismatched song | 0.052 | 0.024 | +0.029 | [-0.055, +0.106] | 0.2585 | – | – | – | No |
|  | Metronome grid, random phase (extra) | 0.052 | -0.004 | +0.056 | [-0.048, +0.152] | 0.1488 | 0.1494 | 20% | 0.58 | No |
| Motorica | Circular shift | 0.064 | 0.000 | +0.064 | [+0.015, +0.112] | 0.0069 | 0.0005 | 11% | 0.56 | Yes |
|  | Random shift (same song) | 0.041 | 0.000 | +0.041 | [-0.009, +0.088] | 0.0528 | 0.0115 | 7% | 0.53 | No |
|  | Phase-randomized | 0.064 | -0.005 | +0.069 | [+0.020, +0.116] | 0.0034 | 0.0005 | 11% | 0.56 | Yes |
|  | IBI-shuffled beats (extra) | 0.064 | 0.026 | +0.037 | [+0.004, +0.070] | 0.0170 | 0.0070 | 10% | 0.54 | Yes |
|  | Metronome grid, own phase (extra) | 0.064 | 0.044 | +0.020 | [+0.004, +0.038] | 0.0095 | – | – | – | Yes |
|  | Mismatched song | 0.064 | -0.008 | +0.072 | [+0.018, +0.122] | 0.0032 | – | – | – | Yes |
|  | Metronome grid, random phase (extra) | 0.064 | -0.004 | +0.068 | [+0.019, +0.115] | 0.0039 | 0.0005 | 13% | 0.56 | Yes |
| **BPL-within: offset-sensitive, offset from other slices of the same take** | | | | | | | | | | |
| FineDance | Circular shift | 0.059 | -0.000 | +0.059 | [-0.106, +0.213] | 0.2442 | 0.0455 | 15% | 0.60 | No |
|  | Random shift (same song) | 0.031 | -0.001 | +0.032 | [-0.118, +0.175] | 0.3401 | 0.2149 | 8% | 0.60 | No |
|  | Phase-randomized | 0.059 | 0.001 | +0.058 | [-0.109, +0.218] | 0.2541 | 0.0535 | 11% | 0.61 | No |
|  | IBI-shuffled beats (extra) | 0.059 | 0.084 | -0.025 | [-0.150, +0.066] | 0.5990 | 0.8721 | 5% | 0.54 | No |
|  | Metronome grid, own phase (extra) | 0.059 | 0.044 | +0.015 | [+0.001, +0.029] | 0.0348 | – | – | – | Yes |
|  | Mismatched song | 0.059 | -0.010 | +0.069 | [-0.078, +0.201] | 0.1805 | – | – | – | No |
|  | Metronome grid, random phase (extra) | 0.059 | 0.004 | +0.054 | [-0.112, +0.210] | 0.2618 | 0.0860 | 13% | 0.61 | No |
| Motorica | Circular shift | 0.133 | -0.000 | +0.133 | [+0.057, +0.206] | 0.0005 | 0.0005 | 20% | 0.62 | Yes |
|  | Random shift (same song) | 0.120 | 0.000 | +0.119 | [+0.053, +0.183] | 0.0005 | 0.0005 | 12% | 0.60 | Yes |
|  | Phase-randomized | 0.133 | -0.003 | +0.136 | [+0.060, +0.207] | 0.0003 | 0.0005 | 17% | 0.62 | Yes |
|  | IBI-shuffled beats (extra) | 0.133 | 0.060 | +0.073 | [+0.022, +0.122] | 0.0031 | 0.0005 | 11% | 0.59 | Yes |
|  | Metronome grid, own phase (extra) | 0.133 | 0.132 | +0.001 | [-0.023, +0.026] | 0.4563 | – | – | – | No |
|  | Mismatched song | 0.133 | 0.005 | +0.129 | [+0.049, +0.205] | 0.0012 | – | – | – | Yes |
|  | Metronome grid, random phase (extra) | 0.133 | 0.002 | +0.132 | [+0.055, +0.205] | 0.0006 | 0.0005 | 18% | 0.62 | Yes |
| **Event PLV (beat_suite.py; offset-free, accents inside their beat interval)** | | | | | | | | | | |
| FineDance | Circular shift | 0.289 | 0.240 | +0.049 | [+0.018, +0.084] | 0.0020 | 0.0005 | 26% | 0.61 | Yes |
|  | Random shift (same song) | 0.290 | 0.277 | +0.013 | [-0.004, +0.030] | 0.0794 | 0.0125 | 13% | 0.55 | No |
|  | Phase-randomized | 0.289 | 0.256 | +0.033 | [+0.015, +0.052] | 0.0007 | 0.0005 | 6% | 0.59 | Yes |
|  | IBI-shuffled beats (extra) | 0.289 | 0.267 | +0.022 | [+0.006, +0.038] | 0.0082 | 0.0020 | 8% | 0.57 | Yes |
|  | Metronome grid, own phase (extra) | 0.289 | 0.285 | +0.005 | [-0.009, +0.018] | 0.2596 | – | – | – | No |
|  | Mismatched song | 0.289 | 0.268 | +0.021 | [+0.000, +0.040] | 0.0318 | – | – | – | Yes |
| Motorica | Circular shift | 0.257 | 0.233 | +0.024 | [+0.011, +0.037] | 0.0004 | 0.0005 | 14% | 0.57 | Yes |
|  | Random shift (same song) | 0.256 | 0.249 | +0.007 | [-0.001, +0.015] | 0.0463 | 0.0570 | 8% | 0.54 | Yes |
|  | Phase-randomized | 0.257 | 0.237 | +0.020 | [+0.009, +0.031] | 0.0004 | 0.0005 | 9% | 0.56 | Yes |
|  | IBI-shuffled beats (extra) | 0.257 | 0.243 | +0.014 | [+0.004, +0.024] | 0.0037 | 0.0020 | 8% | 0.55 | Yes |
|  | Metronome grid, own phase (extra) | 0.257 | 0.257 | -0.000 | [-0.011, +0.011] | 0.5079 | – | – | – | No |
|  | Mismatched song | 0.257 | 0.251 | +0.006 | [-0.007, +0.020] | 0.1820 | – | – | – | No |
| **BAS (authors' Beat Alignment Score; offset-sensitive)** | | | | | | | | | | |
| FineDance | Circular shift | 0.211 | 0.219 | -0.008 | [-0.031, +0.015] | 0.7398 | 0.8461 | 6% | 0.43 | No |
|  | Random shift (same song) | 0.211 | 0.214 | -0.003 | [-0.031, +0.031] | 0.5594 | 0.6452 | 5% | 0.45 | No |
|  | Phase-randomized | 0.211 | 0.221 | -0.010 | [-0.034, +0.013] | 0.7951 | 0.9230 | 5% | 0.43 | No |
|  | IBI-shuffled beats (extra) | 0.211 | 0.211 | -0.000 | [-0.007, +0.007] | 0.5274 | 0.5142 | 5% | 0.49 | No |
|  | Metronome grid, own phase (extra) | 0.211 | 0.206 | +0.005 | [+0.001, +0.009] | 0.0158 | – | – | – | Yes |
|  | Mismatched song | 0.211 | 0.214 | -0.002 | [-0.023, +0.018] | 0.5856 | – | – | – | No |
|  | Metronome grid, random phase (extra) | 0.211 | 0.220 | -0.008 | [-0.032, +0.014] | 0.7529 | 0.8421 | 8% | 0.45 | No |
| Motorica | Circular shift | 0.240 | 0.237 | +0.004 | [-0.006, +0.013] | 0.2419 | 0.1249 | 8% | 0.54 | No |
|  | Random shift (same song) | 0.236 | 0.237 | -0.001 | [-0.009, +0.008] | 0.5654 | 0.5882 | 7% | 0.51 | No |
|  | Phase-randomized | 0.240 | 0.239 | +0.001 | [-0.008, +0.010] | 0.4132 | 0.3313 | 8% | 0.52 | No |
|  | IBI-shuffled beats (extra) | 0.240 | 0.240 | +0.000 | [-0.005, +0.005] | 0.4733 | 0.4538 | 7% | 0.52 | No |
|  | Metronome grid, own phase (extra) | 0.240 | 0.235 | +0.005 | [+0.001, +0.009] | 0.0093 | – | – | – | Yes |
|  | Mismatched song | 0.240 | 0.238 | +0.002 | [-0.007, +0.011] | 0.3148 | – | – | – | No |
|  | Metronome grid, random phase (extra) | 0.240 | 0.237 | +0.003 | [-0.006, +0.012] | 0.2567 | 0.1619 | 14% | 0.54 | No |
| **Intensity–Motion Coupling (IMC)** | | | | | | | | | | |
| FineDance | Circular shift | 0.219 | -0.031 | +0.250 | [+0.097, +0.402] | 0.0030 | 0.0005 | 37% | 0.68 | Yes |
|  | Random shift (same song) | 0.219 | 0.076 | +0.143 | [+0.022, +0.246] | 0.0136 | 0.0005 | 23% | 0.67 | Yes |
|  | Phase-randomized | 0.219 | -0.008 | +0.227 | [+0.102, +0.356] | 0.0020 | 0.0005 | 27% | 0.67 | Yes |
|  | Mismatched song | 0.219 | 0.092 | +0.127 | [+0.013, +0.229] | 0.0192 | – | – | – | Yes |
| Motorica | Circular shift | 0.180 | -0.032 | +0.211 | [+0.161, +0.261] | 0.0001 | 0.0005 | 30% | 0.64 | Yes |
|  | Random shift (same song) | 0.180 | 0.006 | +0.174 | [+0.132, +0.215] | 0.0001 | 0.0005 | 23% | 0.63 | Yes |
|  | Phase-randomized | 0.180 | 0.000 | +0.179 | [+0.139, +0.220] | 0.0001 | 0.0005 | 23% | 0.63 | Yes |
|  | Mismatched song | 0.180 | 0.011 | +0.169 | [+0.125, +0.214] | 0.0001 | – | – | – | Yes |

Original = own music (BPL: for the random-shift row the own window of the same full-song beat track, so both sides share one pipeline). Control = mean of the per-slice null (circular: all 781 integer shifts of 4 s..T−4 s; random shift: 200 same-song windows ≥ 5 s away; phase-randomized: 200 IAAFT RMS surrogates for IMC, 100 beat-tracked phase-randomized onset envelopes for BPL; IBI shuffle: 200; metronome: one regular grid). BPL controls keep the own mean beat frequency (beats rescaled to the own tempo). Δ = Original − Control per unit (song 19 / piece 52); p (units) = one-sided sign-flip permutation (10,000); p (MC null) = observed mean over units against 2,000 draws of the same statistic with one random null draw per slice; "Slices > null 95th pct" = share of slices whose Original exceeds the 95th percentile of their own null (5% expected by chance); BPL-fixed / BPL-within score = Re(mean exp(i(φ_QoM − θ_beats − c))); c = circular mean phase of the real-dance own-beat locking value over all other units (fixed) or over the other non-overlapping slices of the same take (within; slices without a partner dropped: FineDance 75/84, Motorica 287/289); controls never enter c. mean percentile = mean of 1 − empirical one-sided p of the Original within its null (0.5 = chance). Detected = p (units) < 0.05 and Δ > 0.
