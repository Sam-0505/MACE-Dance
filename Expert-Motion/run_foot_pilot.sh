#!/bin/bash
# Foot-contact term in the beat-locking guidance (2026-11-06): pilot on FineDance, 3 seeds, mask + guidance 0.003 with
# 10 or 30 foot-correction steps per guidance step (sample_guided foot_iters, mu 0.1), against the current
# mask + guidance (same seeds 0-2: paired). Scores: Lock / BAS (beat_suite), Loud / Perc (finetune_eval), authors'
# FID / Div / PFC (authors_metrics), Lodge FSR (fsr_table).
cd "$(dirname "$0")"
E=../Evaluation-Musicality
P=.venv/bin/python
L=logs/foot_pilot; mkdir -p $L
for K in 10 30; do
  R=mask2s_guide10_s0.003_foot$K
  $P $E/gen_mace_test.py --seeds 3 --local_attn 2 --lock_guide 0.003 --lock_foot $K --out_dir $E/data/mace_test_$R > $L/gen_$R.log 2>&1 || echo "gen $R failed" >> $L/status.log
  $P $E/authors_metrics.py --mace_dir $E/data/mace_test_$R > $L/authors_$R.log 2>&1
  $P $E/finetune_eval.py --label $R --mace_dir $E/data/mace_test_$R > $L/ftEval_$R.log 2>&1
  echo "$R done" >> $L/status.log
done
# baseline restricted to seeds 0-2 for the paired comparison
$P - <<'PY'
import numpy as np, shutil, os
src = '../Evaluation-Musicality/data/mace_test_mask2s_guide10_s0.003'; dst = src + '_seeds3'
os.makedirs(dst, exist_ok=True)
np.save(dst + '/joints.npy', np.load(src + '/joints.npy', mmap_mode='r')[:, :3])
shutil.copy(src + '/names.json', dst + '/names.json')
PY
$P $E/authors_metrics.py --mace_dir $E/data/mace_test_mask2s_guide10_s0.003_seeds3 > $L/authors_base.log 2>&1
$P $E/finetune_eval.py --label mask2s_guide10_s0.003_seeds3 --mace_dir $E/data/mace_test_mask2s_guide10_s0.003_seeds3 > $L/ftEval_base.log 2>&1
$P $E/beat_suite.py --dataset finedance --tag _foot_pilot --fd_models mask_guide=mace_test_mask2s_guide10_s0.003_seeds3 \
  foot10=mace_test_mask2s_guide10_s0.003_foot10 foot30=mace_test_mask2s_guide10_s0.003_foot30 > $L/beat_suite.log 2>&1
$P $E/fsr_table.py --tag _foot_pilot --fd released=mace_test mask_guide=mace_test_mask2s_guide10_s0.003 \
  foot10=mace_test_mask2s_guide10_s0.003_foot10 foot30=mace_test_mask2s_guide10_s0.003_foot30 --seeds 3 > $L/fsr.log 2>&1
echo "pilot finished" >> $L/status.log
