#!/bin/bash
# Motion quality of the training-free local cross-attention mask (released weights, +-2 s and +-4 s): one-shot 34 s
# on the 84 test slices, 10 seeds -> lag-max own-song Delta + jitter (raw and jitter-free energy), authors' metrics.
set -e
E=../Evaluation-Musicality
for W in ${@:-2 4}; do
  L=local${W}s
  .venv/bin/python $E/gen_mace_test.py --local_attn $W --out_dir $E/data/mace_test_$L > logs/gen_$L.log 2>&1
  .venv/bin/python $E/finetune_eval.py --label $L --mace_dir $E/data/mace_test_$L >> logs/gen_$L.log 2>&1
  .venv/bin/python $E/finetune_eval.py --label $L --mace_dir $E/data/mace_test_$L --energy_sigma 2 >> logs/gen_$L.log 2>&1
  .venv/bin/python $E/authors_metrics.py --mace_dir $E/data/mace_test_$L >> logs/gen_$L.log 2>&1
done
echo "run_local_quality finished" >> logs/run_local_quality.log
