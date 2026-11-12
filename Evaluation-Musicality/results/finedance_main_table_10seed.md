| Model | FID_k ↓ | FID_g ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | BAS ↑ | Lock ↑ | Loud ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Real dance | – | – | 6.4† | 9.94 | 7.54 | 0.201 | 0.459† | 0.219† |
| Lodge | 50.00 | 35.52 | **3.2†** | 5.67 | 4.96 | 0.226 | 0.298† | <u>0.206†</u> |
| MEGA | 50.00 | **13.02** | <u>10.1†</u> | 6.23 | 6.27 | 0.226 | 0.279† | 0.045† |
| MACE | <u>17.83</u> | 25.09 | 16.5† | **10.30** | **8.09** | <u>0.229</u> | <u>0.304†</u> | 0.055† |
| FlowerDance | 29.73 | <u>19.59</u> | – | 8.42 | <u>7.18</u> | **0.232** | – | – |
| Ours (critic v5 + gain) | **17.34†** | 19.84† | 13.4† | <u>9.02†</u> | 6.74† | 0.222† | **0.392†** | **0.223†** |

Bold = best, underline = second best among the model rows (Real dance excluded). † = our 10-seed mean (FSR: pooled over the 10 seeds); unmarked = MACE Table 2. FSR, Lock, Loud are ours for every row.


FSR (%) is our foot-skating ratio: a port of Lodge's official foot-skating code (fsr_table.py), computed on the generated joints of every row (84 slices; real test dance for Real dance), so all rows share one definition. The papers' FSR columns were not used because they mix two measures: the MACE paper's Real dance (0.216) and MACE (0.210) cells equal our PFC values, while its Lodge, EDGE and Bailando cells are Lodge's own foot-skating numbers (MEGA's cell is from the MEGADance paper; FlowerDance's Table 1 repeats the same column). As a cross-check, our real dance gives 6.4 % and Lodge 3.2 % (10 seeds), next to the 6.22 % and 2.76 % that the Lodge paper reports.
