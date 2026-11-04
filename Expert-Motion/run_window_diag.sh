#!/bin/bash
# Window-length diagnostic (2026-11-03): is one-shot MACE's loss of music coupling past ~17 s a length-generalisation
# failure tied to the 8 s training window? LoRA fine-tunes (lambda 0, the lora_l0 recipe) on 12 s and 16 s clips,
# then the length curve. Prediction if it is: coupling holds to roughly twice the training window (~24 s / ~32 s).
# Steps matched to lora_l0 (2788 clips / batch 8 = 348 steps x 30 epochs = 10,440 steps).
set -e
for F in 360 480; do
  [ -f data/dataset_backups/train_long$F.npz ] || .venv/bin/python build_long_clips.py --frames $F > logs/build_long_clips_$F.log 2>&1
  N=$(.venv/bin/python -c "import numpy as np; print(len(np.load('data/dataset_backups/train_long$F.npz')['names']))")
  EP=$(( (10440 + (N + 7) / 8 / 2) / ((N + 7) / 8) ))
  L=lora_w$((F / 30))_l0
  echo "$F frames: $N clips, $EP epochs -> $L"
  WANDB_MODE=offline .venv/bin/python train.py --feature_type baseline --batch_size 8 \
    --checkpoint runs/train/exp/weights/train-3750.pt --epochs $EP --save_interval $EP --learning_rate 1e-4 \
    --long_clips data/dataset_backups/train_long$F.npz --music_signals data/dataset_backups/train_long$F.npz \
    --lora_rank 16 --slim_ckpt --exp_name $L --lambda_dyn 0 > logs/$L.log 2>&1
  .venv/bin/python ../Evaluation-Musicality/length_curve.py --checkpoint runs/train/$L/weights/train-$EP.pt \
    --label $L > logs/length_curve_$L.log 2>&1
  grep " s  whole" logs/length_curve_$L.log
done
echo "run_window_diag finished"
