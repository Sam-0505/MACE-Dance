#!/bin/bash
# Loss changes A + B (2026-11-03), same long-clip recipe as order_long.sh so it compares directly with long_l3e-3:
# A = body energy from joints low-passed with sigma 2 frames (~2 Hz); B = loss max(0, r_real - r_pred).
set -e
L=3e-3
WANDB_MODE=offline .venv/bin/python train.py --feature_type baseline --batch_size 8 \
  --checkpoint runs/train/exp/weights/train-3750.pt --exp_name long_ab_l$L --epochs 30 --save_interval 10 \
  --learning_rate 1e-4 --long_clips data/dataset_backups/train_long1020.npz \
  --music_signals data/dataset_backups/train_long1020.npz --lambda_dyn $L --dyn_energy_sigma 2 --dyn_hinge \
  --slim_ckpt > logs/long_ab_l$L.log 2>&1
