#!/bin/bash
# Evaluate the long-clip fine-tunes (order_long.sh) like the released model: one-shot 34 s on the 84 test slices,
# 10 seeds -> lag-max own-song Delta + jitter (finetune_eval), authors' metrics, and the length curve.
# finetune_eval runs twice: raw body energy and energy from ~2 Hz low-passed joints (jitter cannot count).
set -e
E=../Evaluation-Musicality
for L in ${@:-long_l0 long_l3e-3}; do
  CK=runs/train/$L/weights/train-30.pt
  .venv/bin/python $E/gen_mace_test.py --checkpoint $CK --out_dir $E/data/mace_test_$L
  .venv/bin/python $E/finetune_eval.py --label $L --mace_dir $E/data/mace_test_$L
  .venv/bin/python $E/finetune_eval.py --label $L --mace_dir $E/data/mace_test_$L --energy_sigma 2
  .venv/bin/python $E/authors_metrics.py --mace_dir $E/data/mace_test_$L
  .venv/bin/python $E/length_curve.py --checkpoint $CK --label $L
done
