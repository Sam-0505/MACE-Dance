#!/bin/bash
# Our training-free MACE fixes on Lodge (2026-11-06): beat-locking guidance 0.003 (t <= 600, root rotation frozen) +
# foot-contact term (10 steps, mu 0.1) on Lodge's local stage; no attention mask (Lodge's local windows are its
# training length). Generation: Evaluation-Musicality/baselines/lodge_gen_finedance_guided.py (Lodge's venv), 84
# FineDance test slices x 10 seeds, same seeds as mace_test_lodge. Scores next to unguided Lodge, released MACE and our
# final MACE (mask + guidance + frozen root rotation + foot term). Unguided Lodge also gets the authors' FID / Div / PFC
# here (the comparison tables used the MACE paper's numbers for it).
cd "$(dirname "$0")"
E=../Evaluation-Musicality
P=.venv/bin/python
LP=/home/sam0505/Documents/Code/dance_baselines/LODGE/.venv/bin/python
L=logs/lodge_guided; mkdir -p $L
T=guide_frz_foot
G=mace_test_lodge_$T
FINAL=mace_test_mask2s_guide10_s0.003_foot10_frz
$LP -W ignore $E/baselines/lodge_gen_finedance_guided.py --stage gen --tag $T --lock_guide 0.003 --lock_freeze_root \
  --lock_foot 10 --seeds 0-9 > $L/gen.log 2>&1 || { echo "gen failed" >> $L/status.log; exit 1; }
$LP -W ignore $E/baselines/lodge_gen_finedance_guided.py --stage joints --tag $T --seeds 0-9 >> $L/gen.log 2>&1 || { echo "joints failed" >> $L/status.log; exit 1; }
echo "generated" >> $L/status.log
for D in mace_test_lodge $G; do
  $P $E/authors_metrics.py --mace_dir $E/data/$D > $L/authors_$D.log 2>&1; echo "authors $D exit $?" >> $L/status.log
done
$P $E/finetune_eval.py --label fd_lodge_$T --mace_dir $E/data/$G > $L/ftEval.log 2>&1; echo "ftEval exit $?" >> $L/status.log
$P $E/beat_suite.py --dataset finedance --tempo_null --tag _lodge_guided --fd_models lodge=mace_test_lodge \
  lodge_$T=$G final=$FINAL --pairs lodge_$T=lodge > $L/beat_suite.log 2>&1; echo "beat_suite exit $?" >> $L/status.log
$P $E/fsr_table.py --tag _lodge_guided --fd lodge=mace_test_lodge lodge_$T=$G released=mace_test final=$FINAL \
  > $L/fsr.log 2>&1; echo "fsr exit $?" >> $L/status.log
$P $E/tmd_bench_table.py --tag _lodge_guided --fd lodge=mace_test_lodge lodge_$T=$G released=mace_test final=$FINAL \
  > $L/tmd.log 2>&1; echo "tmd exit $?" >> $L/status.log
$P $E/rotation_jitter.py --tag _lodge_guided --seeds 3 --fd released=mace_test final=$FINAL lodge=mace_test_lodge \
  lodge_$T=$G > $L/jitter.log 2>&1; echo "jitter exit $?" >> $L/status.log
$P $E/lodge_guided_table.py > $L/table.log 2>&1
echo "all done" >> $L/status.log
