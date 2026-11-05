#!/bin/bash
# Beat-input ablation and guidance sweep on the FineDance test slices (2026-11-04).
set -e
cd "$(dirname "$0")"
G=../Evaluation-Musicality/gen_mace_test.py
D=../Evaluation-Musicality/data
.venv/bin/python $G --zero_cols 34 --out_dir $D/mace_test_nobeat
.venv/bin/python $G --zero_cols 0 33 34 --out_dir $D/mace_test_norhythm
.venv/bin/python $G --beta 1.0 --out_dir $D/mace_test_beta1
.venv/bin/python $G --beta 0.5 --out_dir $D/mace_test_beta05
.venv/bin/python $G --local_attn 2 --beta 1.0 --out_dir $D/mace_test_local2s_beta1
echo ALL DONE
