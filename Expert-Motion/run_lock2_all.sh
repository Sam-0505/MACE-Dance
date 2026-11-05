#!/bin/bash
# Pilot 2 end to end: order_lock2.sh, then per checkpoint 34 s one-shot test generation (10 seeds), authors' metrics,
# jitter (finetune_eval), and the beat suite over all pilot-2 runs plus the pilot-1 control lock_l0.
cd "$(dirname "$0")"
E=../Evaluation-Musicality
./order_lock2.sh || { echo "order_lock2 failed $?" >> logs/run_lock2_all.log; exit 1; }
M=("lock_l0=mace_test_lock_l0")
for R in armA_l3e-3 armA_l1e-2 armB_l0 armB_l3e-3; do
  CK=runs/train/$R/weights/train-5.pt
  .venv/bin/python $E/gen_mace_test.py --checkpoint $CK --out_dir $E/data/mace_test_$R > logs/gen_$R.log 2>&1
  .venv/bin/python $E/authors_metrics.py --mace_dir $E/data/mace_test_$R > logs/authors_$R.log 2>&1
  .venv/bin/python $E/finetune_eval.py --label $R --mace_dir $E/data/mace_test_$R > logs/ftEval_$R.log 2>&1
  M+=("$R=mace_test_$R")
done
.venv/bin/python $E/beat_suite.py --dataset finedance --tag _lock2 --fd_models "${M[@]}" > logs/beat_suite_lock2.log 2>&1
.venv/bin/python $E/beatlock_validate.py --tag _lock2 --models "${M[@]}" > logs/beatlock_validate_lock2.log 2>&1
echo "run_lock2_all finished, exit $?" >> logs/run_lock2_all.log
