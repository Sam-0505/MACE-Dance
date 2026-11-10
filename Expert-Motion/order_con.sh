#!/bin/bash
# Contrastive beat-locking fine-tune (2026-11-07): rank-16 LoRA + a new zero-initialised beat-phase block, from the
# released checkpoint, 8 s slices, batch 64, 30 epochs. Loss: diffusion losses + lambda * InfoNCE of the locking score of
# the predicted x0 against its own beats vs 7 tempo-matched other-song clips' beats (model/beatlock.py:contrastive_locking),
# t < 500, joints low-passed (sigma 2). lambda from the gradient-size check (|g_con| / |g_EDGE| ~ 2000 per unit lambda,
# t 50..900): 1e-4 ~ 0.2x, 3e-4 ~ 0.6x, 1e-3 ~ 2x. lambda 0 = control (terms logged only): fine-tune damage alone.
# Waits until the GPU is free of the Appearance jobs (one process per csv row, short gaps between rows).
set -e
cd "$(dirname "$0")"
ok=0
while [ $ok -lt 5 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if ! pgrep -f appearance_infer >/dev/null && [ "$used" -lt 4000 ]; then ok=$((ok+1)); else ok=0; fi
  sleep 20
done
for L in 1e-3 3e-3; do
  WANDB_MODE=offline .venv/bin/python train.py --feature_type baseline --batch_size 64 \
    --checkpoint runs/train/exp/weights/train-3750.pt --exp_name con_l$L --epochs 30 --save_interval 10 \
    --learning_rate 1e-4 --lora_rank 16 --beat_phase_input --lambda_con $L --con_log --lock_joint_sigma 2 \
    --con_t_max 500 --slim_ckpt > logs/con_l$L.log 2>&1
done
