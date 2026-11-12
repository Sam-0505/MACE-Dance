| Model | FID_k ↓ | FID_g ↓ | FID_s ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | DIV_s ↑ | BAS ↑ | Lock ↑ | Loud ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Real dance | – | – | – | 21.6 | 9.94 | 7.54 | 6.11† | 0.201 | 0.459† | 0.219† |
| Lodge | 50.00 | 35.52 | 10.79† | **2.8** | 5.67 | 4.96 | 4.62† | <u>0.226</u> | 0.289† | <u>0.210†</u> |
| MEGA | 50.00 | **13.02** | 2.50† | 24.3 | 6.23 | 6.27 | 5.90† | <u>0.226</u> | 0.279† | 0.045† |
| MACE | <u>17.83</u> | 25.09 | <u>1.52†</u> | 21.0 | **10.30** | **8.09** | **6.24†** | **0.229** | <u>0.302†</u> | 0.038† |
| Ours (critic v5 + gain) | **17.50†** | <u>19.20†</u> | **1.48†** | <u>13.8†</u> | <u>8.96†</u> | <u>6.89†</u> | <u>6.10†</u> | 0.222† | **0.399†** | **0.235†** |

Bold = best, underline = second best among the four models (Real dance excluded). FSR in %. † = our value for seed 1 (first sample per slice; MEGA is deterministic, one dance per slice); unmarked = MACE Table 2. Lock, Loud, FID_s and DIV_s are ours for every row (the MACE paper has no value for them). FID_s / DIV_s: MEGADance's genre-classifier style metrics on our 84 slices (see results/genre_metric_seed0.json); Lodge's FID_s differs a lot from MEGADance's own report (5.22), so treat it with caution.


FSR note: MACE Table 2 copies the foot-skating numbers for the baselines, but its Real dance (0.216) and MACE (0.210) cells equal our PFC values, so the column mixes two measures. The Ours cell is our foot-skating ratio (Lodge formula). Measured the same way for all rows (pooled over the 10 seeds): Real 6.4, Lodge 3.2, MEGA 10.1, MACE 16.5, Ours 13.4 (%).
