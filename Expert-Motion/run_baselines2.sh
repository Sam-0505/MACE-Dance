#!/bin/bash
# Bailando (official AIST++ actor-critic ckpt) and MEGADance (official FineDance ckpt) next to EDGE, Lodge, MACE and
# ours (2026-11-06). Generation: dance_baselines/{Bailando,MEGADance}/gen_mace_eval.py (copies in
# Evaluation-Musicality/baselines). Both models are deterministic (arg-max codes): one dance per slice, copied to every
# seed slot. Lodge arrays were re-oriented (baselines/fix_lodge_axes.py: up was -x), so its Motorica PFC / FID_k /
# Div_k are recomputed here; speed-based measures (Lock, Loud, Perc, BAS) are unchanged by the fix.
#   FineDance: Lock + BAS (beat_suite), Loud / Perc (finetune_eval); FID / Div / PFC / BAS of Bailando and MEGA are
#              taken from the MACE paper, Table 2.
#   Motorica:  Lock + BAS (beat_suite), Loud / Perc + FID_k / PFC / Div_k (motorica_compare).
#   Both:      TMD-Bench VBCS / CSD / ABHS / HSD (tmd_bench_table).
cd "$(dirname "$0")"
E=../Evaluation-Musicality
L=logs/baselines2
mkdir -p $L
for M in bailando mega; do
  .venv/bin/python $E/finetune_eval.py --label fd_$M --mace_dir $E/data/mace_test_$M > $L/ftEval_fd_$M.log 2>&1
  echo "ftEval $M exit $?" >> $L/status.log
done
.venv/bin/python $E/beat_suite.py --dataset finedance --tag _baselines2 --fd_models edge=mace_test_edge \
  lodge=mace_test_lodge bailando=mace_test_bailando mega=mace_test_mega mask_guide_s0.003=mace_test_mask2s_guide10_s0.003 \
  > $L/beat_suite_finedance.log 2>&1
echo "beat_suite finedance exit $?" >> $L/status.log
.venv/bin/python $E/beat_suite.py --dataset motorica --tag _baselines2 --mot_models mask_guide_s0.003=mace_local2s_guide_s0.003 \
  edge=edge lodge=lodge bailando=bailando mega=mega > $L/beat_suite_motorica.log 2>&1
echo "beat_suite motorica exit $?" >> $L/status.log
.venv/bin/python $E/motorica_compare.py --tag _baselines2 --models released=mace \
  mask_guide_s0.003=mace_local2s_guide_s0.003 edge=edge lodge=lodge bailando=bailando mega=mega > $L/motorica_compare.log 2>&1
echo "motorica_compare exit $?" >> $L/status.log
.venv/bin/python $E/tmd_bench_table.py > $L/tmd_bench_table.log 2>&1
echo "tmd_bench_table exit $?" >> $L/status.log
echo done >> $L/status.log
