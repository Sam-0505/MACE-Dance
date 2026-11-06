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
