| Model | FID_k ↓ | FID_g ↓ | FID_s ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | DIV_s ↑ | BAS ↑ | Lock ↑ | Loud ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Real dance | – | – | – | 6.4† | 9.94 | 7.54 | 6.11† | 0.201 | 0.459† | 0.219† |
| Lodge | 50.00 | 35.52 | 10.70† | **3.2†** | 5.67 | 4.96 | 4.64† | 0.226 | 0.298† | 0.206† |
| MEGA | 50.00 | **13.02** | 2.50† | 10.1† | 6.23 | 6.27 | 5.90† | 0.226 | 0.279† | 0.045† |
| MACE | <u>17.83</u> | 25.09 | <u>1.55†</u> | 16.5† | **10.30** | **8.09** | **6.24†** | <u>0.229</u> | 0.304† | 0.055† |
| FlowerDance | 29.73 | <u>19.59</u> | 1.99† | <u>8.9†</u> | 8.42 | <u>7.18</u> | 5.99† | **0.232** | <u>0.371†</u> | <u>0.219†</u> |
| Ours (critic v5 + gain) | **17.34†** | 19.84† | **1.44†** | 13.4† | <u>9.02†</u> | 6.74† | <u>6.10†</u> | 0.222† | **0.392†** | **0.223†** |

Bold = best, underline = second best among the model rows (Real dance excluded). † = our 10-seed mean (FSR: pooled over the 10 seeds; MEGA is deterministic, one dance per slice); unmarked = copied from the MACE paper (Lodge, MEGA, MACE, Real dance) or the FlowerDance paper (FlowerDance). FSR, Lock, Loud, FID_s and DIV_s are ours for every row. FID_s / DIV_s: MEGADance's genre-classifier style metrics on our 84 slices, mean of the per-seed values (results/genre_metric_seed0..9.json).


FlowerDance: FID_k, FID_g, DIV_k, DIV_g, BAS copied from its paper; FSR, FID_s, DIV_s, Lock, Loud are ours from its released FineDance checkpoint (train-3700.pt, 21 Euler steps, shipped test features, 10 seeds). Re-scoring that checkpoint on our 84 slices gives FID_k 40.6, FID_g 18.2, DIV_k 7.85, DIV_g 7.30, BAS 0.230 (paper: 29.73 / 19.59 / 8.42 / 7.18 / 0.232).


FSR (%) is our foot-skating ratio: a port of Lodge's official foot-skating code (fsr_table.py), computed on the generated joints of every row (84 slices; real test dance for Real dance), so all rows share one definition. The papers' FSR columns were not used because they mix two measures: the MACE paper's Real dance (0.216) and MACE (0.210) cells equal our PFC values, while its Lodge, EDGE and Bailando cells are Lodge's own foot-skating numbers (MEGA's cell is from the MEGADance paper; FlowerDance's Table 1 repeats the same column). As a cross-check, our real dance gives 6.4 % and Lodge 3.2 % (10 seeds), next to the 6.22 % and 2.76 % that the Lodge paper reports.
