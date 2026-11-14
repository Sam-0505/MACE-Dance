# TMD-Bench Visual Audio Alignment (MLLM judge), FineDance

Judge: Gemini 2.5 Pro (`protected.gemini-2.5-pro`, TAMU AI API), TMD-Bench VA prompt (Appendix E, Figure 6) unmodified, one fresh request per
video, 24 FPS, 34.0 s clips, 1280x720 mesh renders with the song's audio, same avatar/background/lighting/audio for every model; the camera of
each clip is fitted to the released+ours pair. One sample (seed 0) per clip. Raw 1-5 integers, no rescaling.

| Model | Mean VA Alignment (1-5), 60 clips |
|---|---:|
| Released MACE-Dance | 2.98 |
| MMC-Dance (ours) | 3.07 |
| MEGADance | 2.88 |
| Lodge | 2.97 |
| FlowerDance | 2.50 |

Clips used: 60 (only clips where all five models have a valid score). On all 84 clips, released MACE 3.08, ours 3.31.
Rendered / scored clips per model: Released MACE-Dance 84/84, MMC-Dance (ours) 84/84, MEGADance 68/68, Lodge 83/82, FlowerDance 75/75.
Effective judge FPS from tokens: {'24.0': 393}.
MEGADance, Lodge and FlowerDance were rendered only where they stay in the shared camera's frame (< 1% of frames out of view); the models that travel most (MEGADance, FlowerDance) are
missing on the clips where they leave the frame. Lodge's world places the dancer 1.26 m higher than MACE's, so it is shifted rigidly (constant
floor shift, start on the released clip's start position); its motion is unchanged. Single-sample 1-5 scores are noisy.
