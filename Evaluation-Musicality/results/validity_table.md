| Dataset (unit) | Metric | r own | r wrong | Δ [95% CI] | Δ>0 | p | q |
|---|---|---:|---:|---:|---:|---:|---:|
| FineDance (19 songs) | Loud | 0.219 | 0.092 | +0.127 [+0.013, +0.229] | 15/19 | 0.019 | 0.028 |
| FineDance (19 songs, trim 4 s) | Lock | 0.459 | 0.420 | +0.039 [+0.013, +0.065] | 16/19 | 0.006 | 0.017 |
| FineDance (19 songs, trim 4 s) | Lock, raw null | 0.459 | 0.204 | +0.254 [+0.159, +0.352] | 18/19 | <0.001 | – |
| Motorica (52 pieces) | Loud | 0.180 | 0.011 | +0.169 [+0.125, +0.214] | 43/52 | <0.001 | <0.001 |
| Motorica (52 pieces, trim 4 s) | Lock | 0.420 | 0.384 | +0.037 [+0.018, +0.055] | 36/52 | <0.001 | <0.001 |
| Motorica (52 pieces, trim 4 s) | Lock, raw null | 0.420 | 0.246 | +0.174 [+0.139, +0.212] | 48/52 | <0.001 | – |
| AIST++ (advanced) (10 pieces) | Loud | 0.045 | 0.001 | +0.044 [-0.025, +0.110] | 7/10 | 0.127 | 0.145 |
| AIST++ (advanced) (10 pieces, trim 4 s) | Lock | 0.474 | 0.447 | +0.026 [+0.008, +0.047] | 9/10 | 0.011 | 0.021 |
| AIST++ (advanced) (10 pieces, trim 4 s) | Lock, raw null | 0.474 | 0.337 | +0.137 [+0.054, +0.230] | 10/10 | 0.001 | – |
| PopDanceSet (clips >= 16 s) (132 videos) | Loud | 0.064 | 0.009 | +0.056 [+0.004, +0.108] | 76/132 | 0.021 | 0.028 |
| PopDanceSet (clips >= 16 s) (132 videos, trim 4 s) | Lock | 0.468 | 0.463 | +0.005 [-0.006, +0.016] | 72/132 | 0.181 | 0.181 |
| PopDanceSet (clips >= 16 s) (132 videos, trim 4 s) | Lock, raw null | 0.468 | 0.421 | +0.047 [+0.023, +0.071] | 94/132 | <0.001 | – |

Reference: older beat metrics, tempo-matched null (one-sided p from the two-sided sign-flip; no CI, no q)

| Dataset | Metric | score | Δ | Δ>0 | p |
|---|---|---:|---:|---:|---:|
| FineDance (19 groups) | BAS | 0.211 | -0.010 | 7/19 | 0.790 |
| FineDance (19 groups) | Event PLV | 0.289 | +0.110 | 16/19 | <0.001 |
| Motorica (62 groups) | BAS | 0.242 | +0.000 | 38/62 | 0.476 |
| Motorica (62 groups) | Event PLV | 0.262 | +0.032 | 46/62 | <0.001 |
| AIST++ (basic + advanced, same BPM) (70 groups) | BAS | 0.282 | +0.007 | 38/63 | 0.134 |
| AIST++ (basic + advanced, same BPM) (70 groups) | Event PLV | 0.548 | -0.016 | 42/63 | 0.799 |
| PopDanceSet (clips >= 16 s) (132 groups) | BAS | 0.264 | +0.022 | 91/132 | <0.001 |
| PopDanceSet (clips >= 16 s) (132 groups) | Event PLV | 0.228 | +0.003 | 62/132 | 0.300 |
