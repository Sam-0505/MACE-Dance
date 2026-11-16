# MLLM VA alignment: matched vs mismatched music, FineDance (19 clips, 1 wrong song(s) per clip)

Same judge, prompt and 24 FPS as the main table. The mismatched video is the matched v3 video with only the audio replaced by a tempo-matched wrong
song (same wrong audio for both models). Delta = matched - mismatched; CI = 95% bootstrap over clips; p = sign-flip permutation (Wilcoxon in the JSON).

| Model | Matched | Mismatched | Delta [95% CI] | clips Delta>0 / =0 / <0 | p (one-sided, Delta>0) |
|---|---:|---:|---|---|---:|
| Released MACE-Dance | 3.21 | 3.32 | -0.10 [-0.79, +0.53] | 6/6/7 | 0.6762 |
| MMC-Dance (ours) | 3.11 | 2.74 | +0.37 [-0.58, +1.26] | 9/4/6 | 0.2682 |

MMC-Dance minus MACE-Dance (paired over the same 19 clips, two-sided):

| Quantity | Released | Ours | Ours - released [95% CI] | p |
|---|---:|---:|---|---:|
| Matched score | 3.21 | 3.11 | -0.10 [-0.84, +0.63] | 0.9001 |
| Delta (matched - mismatched) | -0.10 | +0.37 | +0.47 [-0.58, +1.47] | 0.4625 |
