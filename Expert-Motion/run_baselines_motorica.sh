#!/bin/bash
# Beat locking of other public models on Motorica (2026-11-06): EDGE (AIST++, official ckpt) and Lodge (FineDance,
# official ckpt), 289 non-overlapping 34 s slices x 3 samples, generated in /home/sam0505/Documents/Code/dance_baselines
# (EDGE/gen_motorica.py, LODGE/gen_motorica.py). Robustness variants: lodge_official (plain window concatenation),
# lodge_seamfix (frames k-3..k+3 around the 256/512/768 window joins interpolated), edge_despike (3-frame median).
cd "$(dirname "$0")"
E=../Evaluation-Musicality
.venv/bin/python $E/beat_suite.py --dataset motorica --tag _baselines --mot_models mask_2s=mace_local2s \
  guide_s0.003=mace_guide_s0.003 mask_guide_s0.003=mace_local2s_guide_s0.003 edge=edge edge_despike=edge_despike \
  lodge=lodge lodge_official=lodge_official lodge_seamfix=lodge_seamfix > logs/beat_suite_motorica_baselines.log 2>&1
echo "exit $?" >> logs/beat_suite_motorica_baselines.log
