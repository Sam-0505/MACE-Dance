# Other public models on Motorica (2026-11-06)

Copies of the generation scripts that run in their own environments under
`/home/sam0505/Documents/Code/dance_baselines/{EDGE,LODGE}` (official repos, official checkpoints, each model's own
music features), on the 289 non-overlapping 34 s Motorica slices cut to `data/motorica/slices_wav/` (same clips as
MACE). Output: `data/motorica/<model>/joints.npy` (289, 3, 1020, 24, 3), SMPL joints, m, 30 fps.

- EDGE (Stanford-TML/EDGE, AIST++, Jukebox features; ckpt SHA-256 28ca4ce1...02c7): `edge_gen_motorica.py`.
  1 s of silence appended so the 5 s / 2.5 s-stride slicer covers 34 s; seeds 1000*s + i.
- Lodge (li-ronghui/LODGE, FineDance; global epoch 2999, local FineTuneV2 epoch 299): `lodge_gen_motorica.py`
  (features padded 1021 -> 1024 frames; genre from the MACE style label; `--no_fix_root` = official concatenation),
  `lodge_check_motorica.py` (sanity numbers).
- `robust_variants.py`: lodge_seamfix, edge_despike.
- Scoring: `Expert-Motion/run_baselines_motorica.sh` -> `results/beat_suite_motorica_baselines.json`.

## FineDance test (2026-11-06)

`edge_gen_finedance.py`, `lodge_gen_finedance.py`: the same pipelines on the 84 FineDance test slices
(`data/finedance_test_wav/`, symlinks to Expert-Motion/data/test/wavs_sliced, 44.1 or 48 kHz), 10 seeds, Lodge genre =
FineDance style2 label. Output `data/mace_test_{edge,lodge}/`. Scoring: `Expert-Motion/run_baselines_finedance.sh`
(our music metrics); FID / Div / PFC / BAS for EDGE and Lodge are taken from the MACE paper, Table 2.

## Bailando and MEGADance (2026-11-06)

Generation scripts run in their own environments under `/home/sam0505/Documents/Code/dance_baselines/{Bailando,MEGADance}`
(copies here: `bailando_gen_mace_eval.py`, `mega_gen_mace_eval.py`), on both the 84 FineDance test slices and the 289
Motorica slices. Output `data/mace_test_{bailando,mega}/` and `data/motorica/{bailando,mega}/`.

- Bailando (lisiyao21/Bailando, AIST++; VQ-VAE sep_vqvae_root epoch 500 + actor-critic GPT epoch 10, the paper's final
  model): its 7.5 fps librosa/essentia features, demo-mode start codes 423 / 12, 60 -> 30 fps, Y-up -> Z-up.
- MEGADance (XulongT/MEGADance, FineDance; cls 30, fsq 120, gpt 90): the repo's demo path with its 30 fps features
  (checked against its own precomputed FineDance features: correlation 1.000), genre = the slice's style label,
  start pose Pretrained/Init1.pkl, features padded 1021 -> 1024 rows, Y-up -> Z-up.
- Both pick the arg-max code at each step, so they are deterministic: one dance per slice, copied to every seed slot.
- Env notes: MEGA needs torch 2.7.1+cu128 with the prebuilt mamba_ssm 2.3.0 / causal_conv1d 1.6.2.post1 wheels
  (2.3.2 pulls a kernel package that upgrades torch); its Google Drive archives are 7-zip (py7zr).
- `fix_lodge_axes.py`: the Lodge arrays had up = -x (heading rotation applied to Y-up FK output); fixed in the
  generation scripts and in the stored arrays (originals kept as joints_upx_bug.npy).
- Scoring: `Expert-Motion/run_baselines2.sh`.
