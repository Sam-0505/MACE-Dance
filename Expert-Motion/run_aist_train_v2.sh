#!/bin/bash
# AIST++ retrain (v2): the first run (run_aist_train.sh) went unstable after epoch ~100 at a constant lr of 4e-4.
# v2: linear warm-up (5 epochs) + cosine decay to 2.5% over 300 epochs, gradient-norm clip 1.0, no early stopping
# (validation loss did not track sample quality: epoch 20 had the best loss but FID_k 60 vs 38 at epoch 100).
# Pick the checkpoint by FID with Evaluation-Musicality/aist_eval.py. train-N.pt every 25 epochs, last.pt every 10.
cd "$(dirname "$0")"
export WANDB_MODE=offline
.venv/bin/python train.py --data_path data/aist/ --processed_data_dir data/aist_backups/ --project runs/aist --exp_name mace_v2 \
  --feature_type baseline --epochs 300 --batch_size 64 --save_interval 25 --slim_ckpt --skip_sample \
  --lr_schedule cosine --warmup_epochs 5 --grad_clip 1.0 --val_every 10 --patience 0 --last_every 10 > logs/aist/train_v2.log 2>&1
