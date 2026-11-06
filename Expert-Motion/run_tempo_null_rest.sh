#!/bin/bash
# Lock with the tempo-matched null for the remaining deck numbers (gap, mask, placebo slides), 2026-11-06
P=.venv/bin/python; E=../Evaluation-Musicality
$P $E/beat_suite.py --dataset finedance --tempo_null --tag _tempo_null_checks \
  --fd_models mask_2s=mace_test_local2s guide_s0.003=mace_test_guiderr_s0.003 placebo_s0.003=mace_test_guiderr_placebo_s0.003 \
  guide10_s0.003=mace_test_guide10_s0.003 > $E/results/tempo_null_checks.log 2>&1
$P $E/beat_suite.py --dataset finedance --tempo_null --tag _tempo_null_placebo_shift10 --beats_shift 10 \
  --fd_models guide_s0.003=mace_test_guiderr_s0.003 placebo_s0.003=mace_test_guiderr_placebo_s0.003 > $E/results/tempo_null_placebo_shift10.log 2>&1
echo done
