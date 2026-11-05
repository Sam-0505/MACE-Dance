#!/bin/bash
# Beat-locking pilot 2 (2026-11-05), same recipe as order_lock.sh (8 s slices, batch 64, 5 epochs, batch hinge,
# chance-corrected). Pilot 1 met the loss by sharpening the noisy input's own timing at low t and left high t unlocked.
#   arm A: loss only for t >= 600, joints low-passed (sigma 2 frames) before the speed; control = lock_l0 (pilot 1,
#          identical training since the loss flags do nothing at lambda 0)
#   arm B: arm A + beat-phase input (zero-initialised cos/sin projection, model.beat_phase_features); lambda 0 = input only
set -e
cd "$(dirname "$0")"
LOCK="--lock_log --lock_batch --lock_t_min 600 --lock_joint_sigma 2"
run() {  # name lambda extra...
  WANDB_MODE=offline .venv/bin/python train.py --feature_type baseline --batch_size 64 \
    --checkpoint runs/train/exp/weights/train-3750.pt --exp_name $1 --epochs 5 --save_interval 5 \
    --lambda_lock $2 $LOCK --slim_ckpt "${@:3}" > logs/$1.log 2>&1
}
run armA_l3e-3 3e-3
run armA_l1e-2 1e-2
run armB_l0 0 --beat_phase_input
run armB_l3e-3 3e-3 --beat_phase_input
