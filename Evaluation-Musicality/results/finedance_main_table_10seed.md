| Model | FID_k ↓ | FID_g ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | BAS ↑ | Lock ↑ | Loud ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Real dance | – | – | 21.6 | 9.94 | 7.54 | 0.201 | 0.459† | 0.219† |
| Lodge | 50.00 | 35.52 | **2.8** | 5.67 | 4.96 | <u>0.226</u> | 0.298† | <u>0.206†</u> |
| MEGA | 50.00 | **13.02** | 24.3 | 6.23 | 6.27 | <u>0.226</u> | 0.279† | 0.045† |
| MACE | <u>17.83</u> | 25.09 | 21.0 | **10.30** | **8.09** | **0.229** | <u>0.304†</u> | 0.055† |
| Ours (critic v5 + gain) | **17.34†** | <u>19.84†</u> | <u>13.4†</u> | <u>9.02†</u> | <u>6.74†</u> | 0.222† | **0.392†** | **0.223†** |

Bold = best, underline = second best among the four models (Real dance excluded). FSR in %. † = our 10-seed mean; unmarked = MACE Table 2. Lock/Loud are ours for every row.


FSR note: MACE Table 2 copies the foot-skating numbers for the baselines, but its Real dance (0.216) and MACE (0.210) cells equal our PFC values, so the column mixes two measures. The Ours cell is our foot-skating ratio (Lodge formula). Measured the same way for all rows (pooled over the 10 seeds): Real 6.4, Lodge 3.2, MEGA 10.1, MACE 16.5, Ours 13.4 (%).
