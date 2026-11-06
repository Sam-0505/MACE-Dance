#!/bin/bash
# +-2 s cross-attention mask + beat-locking guidance together (2026-11-06), released weights, training-free.
# FineDance: 10 seeds, guidance 0.003 and 0.002 with the mask; coupling (finetune_eval), authors' metrics, motion
# quality, beat suite. Motorica: mask + guidance 0.003, 3 seeds, non-overlapping slices; beat suite.
cd "$(dirname "$0")"
E=../Evaluation-Musicality
P=.venv/bin/python
for S in 0.003 0.002; do
  R=mask2s_guide10_s$S
  $P $E/gen_mace_test.py --seeds 10 --local_attn 2 --lock_guide $S --out_dir $E/data/mace_test_$R > logs/gen_$R.log 2>&1 || echo "gen $R failed" >> logs/run_combined.log
  $P $E/authors_metrics.py --mace_dir $E/data/mace_test_$R > logs/authors_$R.log 2>&1
  $P $E/finetune_eval.py --label $R --mace_dir $E/data/mace_test_$R > logs/ftEval_$R.log 2>&1
  echo "$R done" >> logs/run_combined.log
done
$P $E/beat_suite.py --dataset finedance --tag _combined --fd_models mask_2s=mace_test_local2s \
  guide_s0.003=mace_test_guide10_s0.003 mask_guide_s0.003=mace_test_mask2s_guide10_s0.003 \
  mask_guide_s0.002=mace_test_mask2s_guide10_s0.002 > logs/beat_suite_combined.log 2>&1
$P $E/motion_quality_extra.py --tag _combined --models mask_2s=mace_test_local2s guide_s0.003=mace_test_guide10_s0.003 \
  mask_guide_s0.003=mace_test_mask2s_guide10_s0.003 mask_guide_s0.002=mace_test_mask2s_guide10_s0.002 > logs/mq_combined.log 2>&1
echo "finedance done" >> logs/run_combined.log
$P $E/gen_mace_motorica.py --seeds 3 --local_attn 2 --lock_guide 0.003 --out_name mace_local2s_guide_s0.003 > logs/gen_motorica_mask_guide.log 2>&1
$P $E/beat_suite.py --dataset motorica --tag _combined --mot_models mask_2s=mace_local2s guide_s0.003=mace_guide_s0.003 \
  mask_guide_s0.003=mace_local2s_guide_s0.003 > logs/beat_suite_motorica_combined.log 2>&1
echo "run_combined finished" >> logs/run_combined.log
