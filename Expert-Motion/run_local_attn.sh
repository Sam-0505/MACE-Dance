#!/bin/bash
# Causal test of the attention finding (2026-11-03): released model, no training, cross-attention restricted to
# music within +-2 s / +-4 s of each dance frame at generation time; length curve 8-34 s one-shot.
set -e
for W in 2 4; do
  .venv/bin/python ../Evaluation-Musicality/length_curve.py --local_attn $W --label released_local${W}s \
    > logs/length_curve_released_local${W}s.log 2>&1
  echo "+-$W s"; grep " s  whole" logs/length_curve_released_local${W}s.log
done
echo "run_local_attn finished"
