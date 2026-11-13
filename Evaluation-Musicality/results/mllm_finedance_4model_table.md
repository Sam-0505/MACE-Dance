# TMD-Bench Visual Audio Alignment (MLLM judge), FineDance

Judge: Gemini 2.5 Pro (`protected.gemini-2.5-pro`, TAMU AI API), TMD-Bench VA prompt (Appendix E, Figure 6) unmodified, one fresh request per
video, 24 FPS, 34.0 s clips, 1280x720 mesh renders with the song's audio, same avatar/background/lighting/audio for every model; the camera of
each clip is fitted to the released+ours pair. One sample (seed 0) per clip. Raw 1-5 integers, no rescaling.

| Model | Mean VA Alignment (1-5), 63 clips |
|---|---:|
| Released MACE-Dance | 2.83 |
| MMC-Dance (ours) | 3.08 |
| MEGADance | 2.86 |
| Lodge | 2.94 |

Clips used: 63 (only clips where all four models have a valid score). On all 84 clips, released MACE 3.08, ours 3.31.
Rendered / scored clips per model: Released MACE-Dance 84/84, MMC-Dance (ours) 84/84, MEGADance 68/68, Lodge 83/76.
Effective judge FPS from tokens: {'24.0': 312}.
MEGADance and Lodge were rendered only where they stay in the shared camera's frame (< 1% of frames out of view); MEGADance travels most, so it is
missing on the clips where it leaves the frame. Lodge's world places the dancer 1.26 m higher than MACE's, so it is shifted rigidly (constant
floor shift, start on the released clip's start position); its motion is unchanged. Single-sample 1-5 scores are noisy.
