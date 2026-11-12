# TMD-Bench Visual Audio Alignment (MLLM judge), FineDance, full 84-slice test set

Judge: Gemini 2.5 Pro (`protected.gemini-2.5-pro`, TAMU AI API), TMD-Bench Visual Audio Alignment prompt (Appendix E, Figure 6) unmodified,
one fresh request per video, requested sampling 24 FPS (confirmed from token counts and a frame-counter test), 34.0 s clips (30 FPS source),
1280x720 mesh renders with the song's audio, identical avatar/camera/background/audio for both models, one sample (seed 0) per clip.
Scores are the model's raw 1-5 integers; there is no 0-1 rescaling. No confidence intervals or significance tests (as in TMD-Bench Table 4).

| Model | Mean VA Alignment (1-5), 84 clips (per-clip mean) | Song-level mean, 19 songs (mean of per-song means) |
|---|---:|---:|
| Released MACE-Dance | 3.08 | 3.30 |
| MMC-Dance (ours: mask + critic v5 + foot + frozen root + gain) | 3.31 | 3.17 |

Score distribution (84 clips each), scores 1/2/3/4/5: released 22/11/15/10/26, ours 13/21/11/5/34.

Counts: 168 videos, all with a valid score (0 invalid, 0 video-not-seen); 84 of 84 clips have both scores; 19 songs, 1-10 slices per song.
Requests: 168 valid + 2 transient server errors (HTTP 500), both retried once unchanged and scored. 22 earlier scores from the first
(portrait, height-only camera) renders were superseded after a framing flaw was found and are excluded (`status = superseded_framing_v1`).
Judge-set subset of the first 19 songs (slice 0 only), n = 19: released 3.26, ours 3.16.

Caveat: the per-clip mean weights songs by their number of slices (143 has 10 slices, 130 has 9, 198 and 204 have 1), and the two ways of
averaging disagree on which model is higher, so the difference is not robust. The judge's score also moved by up to 3 points when only the
framing of the same motion changed, so single-sample 1-5 scores are noisy.

Data: `results/mllm_judge.csv` (one row per request), raw responses in `results/tamu_judge_raw/`, summary in `results/mllm_finedance_summary.json`.
