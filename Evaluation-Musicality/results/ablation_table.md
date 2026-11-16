**Values** (seed 1)

| Row | FID_k ↓ | FID_g ↓ | FID_s ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | DIV_s ↑ | BAS ↑ | Lock ↑ | Loud ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MACE released | 15.47 | 25.76 | 1.523 | 16.1 | 9.75 | 8.10 | 6.242 | 0.234 | 0.302 | 0.038 |
| + mask only | 17.82 | 21.47 | 1.483 | 17.5 | 8.87 | 6.93 | 6.101 | 0.223 | 0.313 | 0.238 |
| Full - gain | 19.99 | 25.83 | 1.482 | 15.5 | 8.66 | 6.56 | 6.101 | 0.228 | 0.406 | 0.197 |
| Full - mask | 16.51 | 23.21 | 1.523 | 12.1 | 9.76 | 8.08 | 6.242 | 0.236 | 0.455 | 0.037 |
| Full - critic | 19.45 | 17.43 | 1.482 | 12.4 | 8.84 | 6.95 | 6.101 | 0.221 | 0.332 | 0.233 |
| Full - foot term | 17.26 | 25.42 | 1.482 | 21.9 | 8.90 | 6.82 | 6.101 | 0.223 | 0.385 | 0.240 |
| Full - frozen root | 26.51 | 22.63 | 1.482 | 16.0 | 9.50 | 6.92 | 6.101 | 0.217 | 0.354 | 0.227 |
| **Full (headline)** | 17.50 | 19.20 | 1.482 | 13.8 | 8.96 | 6.89 | 6.101 | 0.222 | 0.399 | 0.235 |
| *Full seed SD (10 seeds)* | 0.449 | 0.536 | – | – | 0.176 | 0.085 | – | 0.003 | – | – |

**Difference from Full** (row − Full; `*` = beyond noise; ▲ better than Full, ▼ worse)

| Row | FID_k ↓ | FID_g ↓ | FID_s ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | DIV_s ↑ | BAS ↑ | Lock ↑ | Loud ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MACE released | -2.03* ▲ | +6.57* ▼ | +0.041 [-0.141, +0.438] | +2.3 | +0.80* ▲ | +1.21* ▲ | +0.141 [+0.086, +0.201]* ▲ | +0.012* ▲ | -0.097 [-0.140, -0.049]* ▼ | -0.197 [-0.304, -0.094]* ▼ |
| + mask only | +0.32 | +2.27* ▼ | +0.000 [+0.000, +0.001] | +3.7 | -0.08 | +0.04 | -0.000 [-0.000, +0.000] | +0.000 | -0.086 [-0.120, -0.052]* ▼ | +0.004 [-0.007, +0.013] |
| Full - gain | +2.49* ▼ | +6.63* ▼ | +0.000 [-0.000, +0.000] | +1.7 | -0.30 | -0.33* ▼ | -0.000 [-0.000, -0.000] | +0.005 | +0.007 [-0.018, +0.036] | -0.037 [-0.057, -0.020]* ▼ |
| Full - mask | -0.99 | +4.01* ▼ | +0.041 [-0.140, +0.439] | -1.7 | +0.80* ▲ | +1.19* ▲ | +0.141 [+0.086, +0.201]* ▲ | +0.013* ▲ | +0.055 [-0.008, +0.131] | -0.197 [-0.304, -0.097]* ▼ |
| Full - critic | +1.95* ▼ | -1.77* ▲ | +0.000 [-0.000, +0.001] | -1.5 | -0.12 | +0.06 | -0.000 [-0.000, +0.000] | -0.002 | -0.067 [-0.098, -0.038]* ▼ | -0.002 [-0.009, +0.005] |
| Full - foot term | -0.24 | +6.22* ▼ | +0.000 [+0.000, +0.000] | +8.1 | -0.06 | -0.07 | -0.000 [-0.000, -0.000] | +0.000 | -0.015 [-0.042, +0.012] | +0.005 [-0.003, +0.012] |
| Full - frozen root | +9.01* ▼ | +3.43* ▼ | +0.000 [-0.000, +0.000] | +2.1 | +0.54* ▲ | +0.03 | -0.000 [-0.000, +0.000] | -0.006 | -0.046 [-0.087, -0.008]* ▼ | -0.008 [-0.026, +0.011] |

FID_s / DIV_s: MEGADance genre-classifier style metrics (genre_metric_ablation.py); `*` needs the 95% slice-bootstrap interval to exclude 0 AND |difference| >= 0.05. They respond only to the mask: every variant that keeps the mask matches Full to within 0.002 (classifier outputs differ by ~0.002 against a spread of 1.4 across clips), so the guidance components neither help nor hurt genre style by this measure.

One seed (the first sample per slice; every row shares its noise). FID_k/FID_g/DIV_k/DIV_g/BAS: `*` if |difference| > 2.83 x the Full model's seed SD over its 10 stored seeds (shown in the italic row). Lock/Loud: 95% bootstrap interval over the 19 songs, paired by song. FSR has no noise estimate, so it is never starred. FSR in %. "Full - critic" keeps the foot term with a vanishing guidance scale (gain and frozen root act only on the guidance push, so they have no effect there).
