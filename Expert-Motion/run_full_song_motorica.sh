#!/bin/bash
# Full-take coupling curves on Motorica (2026-11-04): one-shot generation over whole takes (40-290 s), 5 samples.
set -e
E="../Evaluation-Musicality/full_song_curve.py --dataset motorica"
.venv/bin/python $E --label released
.venv/bin/python $E --label released_local2s --local_attn 2
.venv/bin/python $E --label lora_l0 --checkpoint runs/train/lora_l0/weights/train-30.pt
.venv/bin/python $E --label released_stitched --stitch
echo "run_full_song_motorica finished"
