**Values** (10-seed mean)

| Row | FID_k ↓ | FID_g ↓ | FID_s ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | DIV_s ↑ | BAS ↑ | Lock ↑ | Loud ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MACE released | 17.83‡ | 25.09‡ | 1.546 | 16.5 | 10.30‡ | 8.09‡ | 6.235 | 0.229‡ | 0.304 | 0.055 |
| + mask only | 17.60 | 21.73 | 1.444 | 16.8 | 8.97 | 6.74 | 6.102 | 0.222 | 0.304 | 0.224 |
| Full - gain | 19.91 | 26.35 | 1.444 | 15.2 | 8.72 | 6.41 | 6.102 | 0.230 | 0.385 | 0.189 |
| Full - mask | 16.17 | 23.21 | 1.546 | 12.2 | 10.05 | 8.07 | 6.235 | 0.230 | 0.404 | 0.055 |
| Full - critic | 19.07 | 17.58 | 1.443 | 11.8 | 8.93 | 6.81 | 6.102 | 0.218 | 0.305 | 0.221 |
| Full - foot term | 17.10 | 25.72 | 1.444 | 21.2 | 9.00 | 6.69 | 6.102 | 0.224 | 0.369 | 0.225 |
| Full - frozen root | 27.14 | 22.59 | 1.443 | 15.1 | 9.51 | 6.72 | 6.102 | 0.219 | 0.358 | 0.216 |
| **Full (headline)** | 17.34 | 19.84 | 1.444 | 13.4 | 9.02 | 6.74 | 6.102 | 0.222 | 0.392 | 0.223 |

**Difference from Full** (row − Full; `*` = beyond noise; ▲ better than Full, ▼ worse)

| Row | FID_k ↓ | FID_g ↓ | FID_s ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | DIV_s ↑ | BAS ↑ | Lock ↑ | Loud ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MACE released | +0.49‡ | +5.25‡ | +0.102 [+0.074, +0.130]* ▼ | +3.1 | +1.28‡ | +1.35‡ | +0.133 [+0.124, +0.142]* ▲ | +0.007‡ | -0.088 [-0.116, -0.061]* ▼ | -0.167 [-0.267, -0.069]* ▼ |
| + mask only | +0.26 [+0.19, +0.34]* ▼ | +1.89 [+1.71, +2.06]* ▼ | +0.000 [-0.000, +0.000] | +3.4 | -0.05 [-0.07, -0.02]* ▼ | -0.00 [-0.03, +0.03] | -0.000 [-0.000, -0.000] | -0.000 [-0.002, +0.002] | -0.089 [-0.116, -0.061]* ▼ | +0.001 [-0.005, +0.008] |
| Full - gain | +2.57 [+2.45, +2.70]* ▼ | +6.51 [+6.36, +6.65]* ▼ | +0.000 [+0.000, +0.000] | +1.8 | -0.30 [-0.32, -0.28]* ▼ | -0.33 [-0.35, -0.32]* ▼ | -0.000 [-0.000, -0.000] | +0.008 [+0.007, +0.010]* ▲ | -0.007 [-0.015, +0.002] | -0.034 [-0.052, -0.018]* ▼ |
| Full - mask | -1.16 [-1.86, -0.46]* ▲ | +3.37 [+2.73, +4.00]* ▼ | +0.102 [+0.074, +0.130]* ▼ | -1.1 | +1.03 [+0.87, +1.19]* ▲ | +1.33 [+1.24, +1.43]* ▲ | +0.133 [+0.124, +0.142]* ▲ | +0.009 [+0.005, +0.012]* ▲ | +0.012 [-0.011, +0.035] | -0.167 [-0.271, -0.068]* ▼ |
| Full - critic | +1.73 [+1.58, +1.87]* ▼ | -2.26 [-2.41, -2.12]* ▲ | -0.000 [-0.000, +0.000] | -1.6 | -0.09 [-0.10, -0.08]* ▼ | +0.07 [+0.06, +0.08]* ▲ | -0.000 [-0.000, +0.000] | -0.003 [-0.005, -0.001]* ▼ | -0.087 [-0.112, -0.064]* ▼ | -0.002 [-0.007, +0.004] |
| Full - foot term | -0.24 [-0.31, -0.17]* ▲ | +5.88 [+5.73, +6.04]* ▼ | +0.000 [+0.000, +0.000] | +7.8 | -0.03 [-0.05, +0.00] | -0.05 [-0.08, -0.03]* ▼ | -0.000 [-0.000, -0.000] | +0.002 [-0.001, +0.005] | -0.023 [-0.037, -0.010]* ▼ | +0.002 [-0.003, +0.007] |
| Full - frozen root | +9.80 [+9.19, +10.40]* ▼ | +2.75 [+2.51, +2.98]* ▼ | -0.000 [-0.000, +0.000] | +1.7 | +0.49 [+0.46, +0.52]* ▲ | -0.02 [-0.08, +0.03] | +0.000 [-0.000, +0.000] | -0.003 [-0.004, -0.001]* ▼ | -0.034 [-0.047, -0.022]* ▼ | -0.006 [-0.019, +0.007] |

‡ = MACE released value copied from the MACE paper (Table 2), as in the main table; it has no per-seed runs, so its difference from Full has no interval and is never starred. The MACE row's FID_s, FSR, DIV_s, Lock and Loud are ours.

10 seeds per row; seed k uses the same noise in every row, so differences from Full are paired by seed. FID_k/FID_g/FID_s/DIV_k/DIV_g/DIV_s/BAS: 95% t-interval (9 df) of the mean of the 10 per-seed differences; `*` if it excludes 0 (FID_s/DIV_s also need |difference| >= 0.05). FID_s / DIV_s: MEGADance genre-classifier style metrics (genre_metric_ablation.py, one value per seed, averaged). Lock/Loud: 95% bootstrap interval over the 19 songs of the 10-seed per-song means, paired by song. FSR pooled over the 10 seeds; it has no noise estimate, so it is never starred. FSR in %. "Full - critic" keeps the foot term with a vanishing guidance scale (gain and frozen root act only on the guidance push, so they have no effect there).
