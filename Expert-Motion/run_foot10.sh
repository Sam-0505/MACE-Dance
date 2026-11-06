#!/bin/bash
# Mask + guidance 0.003 + foot-contact term (10 steps, mu 0.1) (2026-11-06): FineDance 10 seeds, then Motorica
# (289 non-overlapping slices x 3 samples), scored on everything in the comparison tables.
cd "$(dirname "$0")"
E=../Evaluation-Musicality
P=.venv/bin/python
L=logs/foot10; mkdir -p $L
R=mask2s_guide10_s0.003_foot10_10seeds
$P $E/gen_mace_test.py --seeds 10 --local_attn 2 --lock_guide 0.003 --lock_foot 10 --out_dir $E/data/mace_test_$R > $L/gen_fd.log 2>&1 || echo "gen fd failed" >> $L/status.log
echo "fd generated" >> $L/status.log
$P $E/authors_metrics.py --mace_dir $E/data/mace_test_$R > $L/authors_fd.log 2>&1
$P $E/finetune_eval.py --label $R --mace_dir $E/data/mace_test_$R > $L/ftEval_fd.log 2>&1
$P $E/beat_suite.py --dataset finedance --tag _foot10 --fd_models mask_guide=mace_test_mask2s_guide10_s0.003 \
  mask_guide_foot10=mace_test_$R > $L/beat_suite_fd.log 2>&1
$P $E/fsr_table.py --tag _foot10_fd --fd mask_guide_foot10=mace_test_$R > $L/fsr_fd.log 2>&1
$P $E/tmd_bench_table.py --tag _foot10_fd --fd mask_guide_foot10=mace_test_$R > $L/tmd_fd.log 2>&1
echo "fd scored" >> $L/status.log
M=mace_local2s_guide_s0.003_foot10
$P $E/gen_mace_motorica.py --seeds 3 --local_attn 2 --lock_guide 0.003 --lock_foot 10 --out_name $M > $L/gen_mot.log 2>&1 || echo "gen mot failed" >> $L/status.log
echo "mot generated" >> $L/status.log
$P $E/beat_suite.py --dataset motorica --tag _foot10 --mot_models mask_guide=mace_local2s_guide_s0.003 \
  mask_guide_foot10=$M > $L/beat_suite_mot.log 2>&1
$P $E/motorica_compare.py --tag _foot10 --models released=mace mask_guide=mace_local2s_guide_s0.003 \
  mask_guide_foot10=$M > $L/motorica_compare.log 2>&1
$P $E/fsr_table.py --tag _foot10_mot --mot mask_guide_foot10=$M > $L/fsr_mot.log 2>&1
$P $E/tmd_bench_table.py --tag _foot10_mot --mot mask_guide_foot10=$M > $L/tmd_mot.log 2>&1
echo "all done" >> $L/status.log
