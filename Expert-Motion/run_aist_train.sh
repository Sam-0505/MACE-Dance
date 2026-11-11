#!/bin/bash
# MACE trained from scratch on AIST++ (EDGE's split: 952 train sequences, 5 s slices, 0.5 s stride) for the in-domain
# comparison with EDGE and Bailando. Batch 64 (128 runs out of memory on a 4090), cap 400 epochs (~70 s each),
# validation loss every 10 epochs on 48 held-out training sequences, early stop after 10 evaluations without improvement.
# Outputs: runs/aist/mace/weights/{train-N.pt every 50, best.pt, last.pt (full state; resume with --checkpoint + --start_epoch)}.
cd "$(dirname "$0")"
export WANDB_MODE=offline
.venv/bin/python train.py --data_path data/aist/ --processed_data_dir data/aist_backups/ --project runs/aist --exp_name mace \
  --feature_type baseline --epochs 400 --batch_size 64 --save_interval 50 --slim_ckpt --skip_sample \
  --val_every 10 --patience 10 --last_every 10 > logs/aist/train.log 2>&1
