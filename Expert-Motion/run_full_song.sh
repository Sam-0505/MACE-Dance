#!/bin/bash
# Full-song coupling curves (2026-11-04): one-shot generation over whole FineDance test songs (39-190 s).
set -e
E=../Evaluation-Musicality/full_song_curve.py
while pgrep -f "full_song_curve.py --label released$" > /dev/null; do sleep 20; done
.venv/bin/python $E --label released_local2s --local_attn 2
.venv/bin/python $E --label lora_l0 --checkpoint runs/train/lora_l0/weights/train-30.pt
.venv/bin/python $E --label released_stitched --stitch
.venv/bin/python $E --label lora_ab_l3e-3 --checkpoint runs/train/lora_ab_l3e-3/weights/train-30.pt
.venv/bin/python $E --label lora_l0_local2s --checkpoint runs/train/lora_l0/weights/train-30.pt --local_attn 2
echo "run_full_song finished"
