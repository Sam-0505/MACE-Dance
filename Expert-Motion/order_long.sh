#!/bin/bash
# Long-clip fine-tunes (2026-11-03): same recipe, lambda 0 (long clips alone) then lambda 3e-3 (the user's loss).
set -e
for L in 0 3e-3; do
  WANDB_MODE=offline .venv/bin/python train.py --feature_type baseline --batch_size 8 \
    --checkpoint runs/train/exp/weights/train-3750.pt --exp_name long_l$L --epochs 30 --save_interval 10 \
    --learning_rate 1e-4 --long_clips data/dataset_backups/train_long1020.npz \
    --music_signals data/dataset_backups/train_long1020.npz --lambda_dyn $L --slim_ckpt > logs/long_l$L.log 2>&1
done
