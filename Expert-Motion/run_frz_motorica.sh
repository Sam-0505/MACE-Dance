#!/bin/bash
# Final model on Motorica (2026-11-06): mask + guidance 0.003 with the root rotation frozen + foot term (10 steps),
# 289 non-overlapping slices x 3 samples; scored like slide 8 (Lock / BAS, Loud / Perc + FID_k / PFC / Div_k, FSR,
# VBCS / ABHS) next to mask + guidance and mask + guidance + foot.
cd "$(dirname "$0")"
E=../Evaluation-Musicality
P=.venv/bin/python
L=logs/frz_motorica; mkdir -p $L
M=mace_local2s_guide_s0.003_foot10_frz
$P $E/gen_mace_motorica.py --seeds 3 --local_attn 2 --lock_guide 0.003 --lock_foot 10 --lock_freeze_root --out_name $M > $L/gen.log 2>&1 || echo "gen failed" >> $L/status.log
echo "generated" >> $L/status.log
$P $E/beat_suite.py --dataset motorica --tag _frz --mot_models mask_guide=mace_local2s_guide_s0.003 \
  foot=mace_local2s_guide_s0.003_foot10 final=$M > $L/beat_suite.log 2>&1
$P $E/motorica_compare.py --tag _frz --models released=mace foot=mace_local2s_guide_s0.003_foot10 final=$M > $L/motorica_compare.log 2>&1
$P $E/fsr_table.py --tag _frz_mot --mot final=$M > $L/fsr.log 2>&1
$P $E/tmd_bench_table.py --tag _frz_mot --mot final=$M > $L/tmd.log 2>&1
echo "all done" >> $L/status.log
