#!/bin/bash
# Our music metrics for EDGE (official AIST++ ckpt) and Lodge (official FineDance ckpt) on the 84 FineDance test
# slices, 10 seeds (2026-11-06; generation: dance_baselines/{EDGE,LODGE}/gen_finedance.py, copies in
# Evaluation-Musicality/baselines). FID / Div / PFC / BAS for these models are taken from the MACE paper, Table 2.
cd "$(dirname "$0")"
E=../Evaluation-Musicality
for M in edge lodge; do
  cp $E/data/mace_test/names.json $E/data/mace_test_$M/names.json
  .venv/bin/python $E/finetune_eval.py --label fd_$M --mace_dir $E/data/mace_test_$M > logs/ftEval_fd_$M.log 2>&1
done
.venv/bin/python $E/beat_suite.py --dataset finedance --tag _baselines --fd_models edge=mace_test_edge \
  lodge=mace_test_lodge mask_guide_s0.003=mace_test_mask2s_guide10_s0.003 > logs/beat_suite_finedance_baselines.log 2>&1
echo "exit $?" >> logs/beat_suite_finedance_baselines.log
