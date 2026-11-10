#!/bin/bash
# EDGE (original transformer denoiser + classifier-free guidance) trained on the MACE FineDance cache.
# Chained, resumable: each job continues from the newest checkpoint and stops at the walltime limit; resubmit with
#   sbatch --dependency=afterany:<jobid> hprc/train_edge_finedance.slurm
# until the last epoch is reached. CLUSTER-SPECIFIC LINES ARE MARKED "EDIT" and must be set from the cluster's docs.
#SBATCH --job-name=edge_fd
#SBATCH --time=24:00:00                 # EDIT: partition walltime limit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8               # EDIT: data loader uses 0.75 x cores
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1              # EDIT: check `maxconfig`; for N GPUs on one node: sbatch --gres=gpu:a100:N --cpus-per-task=8N ... with GPUS=N
#SBATCH --partition=gpu                # EDIT: Grace's GPU partition (check `maxconfig`)
##SBATCH --account=<project id>        # EDIT
#SBATCH --output=logs/edge_fd_%j.log

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"          # submit from Expert-Motion/
EXP=${EXP:-edge_fd}
EPOCHS=${EPOCHS:-1000}
SAVE=${SAVE:-25}
BATCH=${BATCH:-128}                     # EFFECTIVE batch: with accelerate each process loads BATCH/GPUS, so it stays 128
GPUS=${GPUS:-1}
PY=${PY:-${SCRATCH:-$HOME}/envs/edge/bin/python}   # built by hprc/setup_env.sh (env lives in $SCRATCH)
# the venv's python is the module's python: load the same modules used when the env was built
module purge
module load GCCcore/12.2.0 Python/3.10.8

export WANDB_MODE=offline

# newest checkpoint of this experiment, if any
LAST=$(ls runs/train/$EXP/weights/train-*.pt 2>/dev/null | sed 's/.*train-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
RESUME=""
START=1
if [ -n "${LAST:-}" ]; then
  RESUME="--checkpoint runs/train/$EXP/weights/train-$LAST.pt"
  START=$((LAST + 1))
fi
if [ "$START" -gt "$EPOCHS" ]; then echo "already at epoch $LAST >= $EPOCHS"; exit 0; fi
echo "start epoch $START of $EPOCHS, resume: ${RESUME:-none}"

# --skip_sample: no render / metric at checkpoints (no display or ffmpeg on compute nodes)
# --project/--exp_name: the folder is reused when --start_epoch > 1
if [ "$GPUS" -gt 1 ]; then
  LAUNCH="$PY -m accelerate.commands.launch --multi_gpu --num_processes $GPUS --mixed_precision no"
else
  LAUNCH="$PY"
fi
$LAUNCH train.py --feature_type baseline --arch edge --batch_size $((BATCH / GPUS)) --epochs $EPOCHS --save_interval $SAVE \
  --skip_sample --exp_name $EXP --start_epoch $START $RESUME
