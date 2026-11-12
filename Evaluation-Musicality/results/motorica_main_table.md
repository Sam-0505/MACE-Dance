| Model | FID_k ↓ | FID_g ↓ | FSR ↓ | DIV_k ↑ | DIV_g ↑ | BAS ↑ | Lock ↑ | Loud ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Real dance | – | – | 8.3 | 10.26 | – | 0.242 | 0.420 | 0.180 |
| Lodge | 115.26 | – | **1.5** | 3.46 | – | <u>0.256</u> | 0.290 | <u>0.149</u> |
| MEGA | 85.82 | – | 9.5 | 4.76 | – | **0.259** | 0.285 | 0.026 |
| MACE | **72.85** | – | 10.7 | **7.71** | – | 0.243 | <u>0.296</u> | 0.048 |
| Ours (critic v5 + gain) | <u>81.06</u> | – | <u>6.5</u> | <u>6.58</u> | – | 0.236 | **0.395** | **0.235** |

Bold = best, underline = second best among the four models (Real dance excluded). FSR in %. All cells are ours (3 samples per slice, 289 slices); MACE's paper has no Motorica numbers. FID_k / DIV_k: Bailando kinetic features on the shared 21-joint body, against real Motorica (Real dance DIV_k is the real slices themselves; no FID for the real row). FID_g / DIV_g are not reported: the geometric features need SMPL-24 joints that real Motorica does not have.
