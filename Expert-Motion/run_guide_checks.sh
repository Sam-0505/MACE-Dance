#!/bin/bash
# Checks of beat-locking guidance (2026-11-05), released model, root-relative guidance:
# 1. placebo: guide towards another song's beats (batch shifted by 10 clips), score against the true beats and
#    against the guide beats; 2. 10 seeds at 0.002 / 0.003 and 0.003 at t <= 300; 3. offset spread across songs;
# 4. Motorica replication (3 seeds, non-overlapping slices).
cd "$(dirname "$0")"
E=../Evaluation-Musicality
P=.venv/bin/python
gen_eval() {  # name, gen args
  R=$1; shift
  $P $E/gen_mace_test.py "$@" --out_dir $E/data/mace_test_$R > logs/gen_$R.log 2>&1 || echo "gen $R failed"
  $P $E/authors_metrics.py --mace_dir $E/data/mace_test_$R > logs/authors_$R.log 2>&1
  $P $E/finetune_eval.py --label $R --mace_dir $E/data/mace_test_$R > logs/ftEval_$R.log 2>&1
}
gen_eval guiderr_placebo_s0.003 --seeds 3 --lock_guide 0.003 --lock_guide_shift 10
$P $E/beat_suite.py --dataset finedance --tag _placebo \
  --fd_models guide_s0.003=mace_test_guiderr_s0.003 placebo_s0.003=mace_test_guiderr_placebo_s0.003 > logs/beat_suite_placebo.log 2>&1
$P $E/beat_suite.py --dataset finedance --tag _placebo_shift10 --beats_shift 10 \
  --fd_models guide_s0.003=mace_test_guiderr_s0.003 placebo_s0.003=mace_test_guiderr_placebo_s0.003 > logs/beat_suite_placebo_shift10.log 2>&1
echo "placebo done" >> logs/run_guide_checks.log
gen_eval guide10_s0.002 --seeds 10 --lock_guide 0.002
gen_eval guide10_s0.003 --seeds 10 --lock_guide 0.003
gen_eval guide10_s0.003_t300 --seeds 10 --lock_guide 0.003 --lock_guide_tmax 300
$P $E/beat_suite.py --dataset finedance --tag _guide10 --fd_models guide_s0.002=mace_test_guide10_s0.002 \
  guide_s0.003=mace_test_guide10_s0.003 guide_s0.003_t300=mace_test_guide10_s0.003_t300 > logs/beat_suite_guide10.log 2>&1
$P $E/offset_spread.py --models guide_s0.002=mace_test_guide10_s0.002 guide_s0.003=mace_test_guide10_s0.003 \
  guide_s0.003_t300=mace_test_guide10_s0.003_t300 placebo_s0.003=mace_test_guiderr_placebo_s0.003 > logs/offset_spread.log 2>&1
echo "10 seeds done" >> logs/run_guide_checks.log
$P $E/gen_mace_motorica.py --seeds 3 --lock_guide 0.003 > logs/gen_motorica_guide_s0.003.log 2>&1
$P $E/beat_suite.py --dataset motorica --tag _guide --mot_models guide_s0.003=mace_guide_s0.003 > logs/beat_suite_motorica_guide.log 2>&1
echo "run_guide_checks finished" >> logs/run_guide_checks.log
