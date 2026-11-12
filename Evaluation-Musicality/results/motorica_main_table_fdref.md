| Model | FID_k ↓ | FID_g ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | BAS ↑ | Lock ↑ | Loud ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Real dance | – | – | 8.3 | – | – | 0.242 | 0.420 | 0.180 |
| Lodge | 78.97 | 46.85 | **1.5** | 4.15 | 4.58 | <u>0.256</u> | 0.290 | <u>0.149</u> |
| MEGA | **44.92** | 48.41 | 9.5 | 6.15 | 5.78 | **0.259** | 0.285 | 0.026 |
| MACE | 48.70 | <u>39.42</u> | 10.7 | **10.97** | **8.25** | 0.243 | <u>0.296</u> | 0.048 |
| Ours (critic v5 + gain) | <u>45.34</u> | **32.53** | <u>6.5</u> | <u>9.31</u> | <u>7.10</u> | 0.236 | **0.395** | **0.235** |

Bold = best, underline = second best among the four models (Real dance excluded). FSR in %. All cells are ours (3 samples per slice, 289 slices); MACE's paper has no Motorica numbers. FID/DIV use the authors' protocol against their FineDance test reference (how FineDance-like the Motorica dances are), not a Motorica reference.
