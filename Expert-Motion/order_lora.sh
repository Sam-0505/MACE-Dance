#!/bin/bash
# LoRA fine-tunes (2026-11-03): rank-16 adapters (model/lora.py), released weights frozen, same long-clip recipe as
# order_long.sh / order_ab.sh. lora_l0 = long clips alone; lora_ab_l3e-3 = with the A + B musicality loss.
set -e
COMMON="--feature_type baseline --batch_size 8 --checkpoint runs/train/exp/weights/train-3750.pt --epochs 30
  --save_interval 10 --learning_rate 1e-4 --long_clips data/dataset_backups/train_long1020.npz
  --music_signals data/dataset_backups/train_long1020.npz --lora_rank 16 --slim_ckpt"
WANDB_MODE=offline .venv/bin/python train.py $COMMON --exp_name lora_l0 --lambda_dyn 0 > logs/lora_l0.log 2>&1
WANDB_MODE=offline .venv/bin/python train.py $COMMON --exp_name lora_ab_l3e-3 --lambda_dyn 3e-3 \
  --dyn_energy_sigma 2 --dyn_hinge > logs/lora_ab_l3e-3.log 2>&1
