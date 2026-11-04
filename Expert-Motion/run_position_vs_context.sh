#!/bin/bash
# Position vs attention interference (2026-11-04): same excerpts, early/late x blocked/full cross-attention.
set -e
E=../Evaluation-Musicality/position_vs_context.py
.venv/bin/python $E --label lora_l0 --checkpoint runs/train/lora_l0/weights/train-30.pt
.venv/bin/python $E --label released
echo "run_position_vs_context finished"
