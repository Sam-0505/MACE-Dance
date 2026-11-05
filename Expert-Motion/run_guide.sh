#!/bin/bash
# Beat-locking guidance sweep (2026-11-05): released model, 34 s one-shot, 3 seeds (same noise as released seeds 0-2),
# guidance at t <= 600; then authors' metrics, jitter and the beat suite.
cd "$(dirname "$0")"
E=../Evaluation-Musicality
TAG=${TAG:-}  # e.g. TAG=rr for root-relative runs
M=()
for S in ${@:-0.003 0.01 0.03}; do
  R=guide${TAG}_s$S
  .venv/bin/python $E/gen_mace_test.py --seeds 3 --lock_guide $S $GUIDE_ARGS --out_dir $E/data/mace_test_$R > logs/gen_$R.log 2>&1
  .venv/bin/python $E/authors_metrics.py --mace_dir $E/data/mace_test_$R > logs/authors_$R.log 2>&1
  .venv/bin/python $E/finetune_eval.py --label $R --mace_dir $E/data/mace_test_$R > logs/ftEval_$R.log 2>&1
  M+=("$R=mace_test_$R")
done
.venv/bin/python $E/beat_suite.py --dataset finedance --tag _guide$TAG --fd_models "${M[@]}" > logs/beat_suite_guide$TAG.log 2>&1
echo "run_guide finished, exit $?" >> logs/run_guide.log
