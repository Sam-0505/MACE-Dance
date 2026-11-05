#!/bin/bash
# Beat-locking loss pilot (model/beatlock.py; DRAFT, not run yet, 2026-11-05): fine-tune from the release on the 8 s
# slices, same recipe as the slow-coupling lambda sweep (batch 64, 5 epochs, ~2 min/epoch). lambda = 0 is the control
# (terms logged only). The gradient of the term at lambda = 1 is ~200x that of the EDGE losses, hence 1e-3 .. 1e-2.
# Evaluate with gen_mace_test.py --checkpoint ... and beat_suite.py (PLV delta must rise; FID/jitter must not).
set -e
cd "$(dirname "$0")"
for L in 0 1e-3 3e-3 1e-2; do
  WANDB_MODE=offline .venv/bin/python train.py --feature_type baseline --batch_size 64 \
    --checkpoint runs/train/exp/weights/train-3750.pt --exp_name lock_l$L --epochs 5 --save_interval 5 \
    --lambda_lock $L --lock_log --lock_batch --slim_ckpt > logs/lock_l$L.log 2>&1
done
