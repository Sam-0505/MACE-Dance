#!/bin/bash
# Beat-locking loss pilot end to end (2026-11-05): order_lock.sh, then per checkpoint one-shot 34 s generation on the
# 84 test slices (10 seeds), authors' metrics (FID, diversity, BAS), jitter/coupling (finetune_eval), and the beat suite
# (BAS / beat locking PLV / BPS vs other-song chance) over all lambdas. Detached-safe (setsid nohup).
cd "$(dirname "$0")"
E=../Evaluation-Musicality
./order_lock.sh || { echo "order_lock failed $?" >> logs/run_lock_all.log; exit 1; }
M=()
for L in 0 1e-3 3e-3 1e-2; do
  CK=runs/train/lock_l$L/weights/train-5.pt
  .venv/bin/python $E/gen_mace_test.py --checkpoint $CK --out_dir $E/data/mace_test_lock_l$L > logs/gen_lock_l$L.log 2>&1
  .venv/bin/python $E/authors_metrics.py --mace_dir $E/data/mace_test_lock_l$L > logs/authors_lock_l$L.log 2>&1
  .venv/bin/python $E/finetune_eval.py --label lock_l$L --mace_dir $E/data/mace_test_lock_l$L > logs/ftEval_lock_l$L.log 2>&1
  M+=("lock_l$L=mace_test_lock_l$L")
done
.venv/bin/python $E/beat_suite.py --dataset finedance --tag _lock --fd_models "${M[@]}" > logs/beat_suite_lock.log 2>&1
echo "run_lock_all finished, exit $?" >> logs/run_lock_all.log
