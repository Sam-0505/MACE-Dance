| Dataset (unit) | Metric | r own | r wrong | Δ [95% CI] | Δ>0 | p | q |
|---|---|---:|---:|---:|---:|---:|---:|
| FineDance (19 songs) | Loud | 0.219 | 0.092 | +0.127 [+0.013, +0.229] | 15/19 | 0.019 | 0.028 |
| FineDance (19 songs, trim 4 s) | Lock | 0.459 | 0.420 | +0.039 [+0.013, +0.065] | 16/19 | 0.006 | 0.013 |
| Motorica (52 pieces) | Loud | 0.180 | 0.011 | +0.169 [+0.125, +0.214] | 43/52 | <0.001 | <0.001 |
| Motorica (52 pieces, trim 4 s) | Lock | 0.420 | 0.384 | +0.037 [+0.018, +0.055] | 36/52 | <0.001 | <0.001 |
| AIST++ (70 pieces) | Loud | 0.027 | 0.072 | -0.045 [-0.114, +0.024] | 32/70 | 0.893 | 0.893 |
| AIST++ (70 pieces, trim 1/4 s) | Lock | 0.759 | 0.751 | +0.009 [+0.003, +0.015] | 50/70 | 0.002 | 0.006 |
| PopDanceSet (clips >= 16 s) (132 videos) | Loud | 0.064 | 0.009 | +0.056 [+0.004, +0.108] | 76/132 | 0.021 | 0.028 |
| PopDanceSet (clips >= 16 s) (132 videos, trim 4 s) | Lock | 0.468 | 0.463 | +0.005 [-0.006, +0.016] | 72/132 | 0.181 | 0.207 |

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
