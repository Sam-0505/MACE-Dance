# EDGE trained on FineDance (HPRC)

Original EDGE transformer denoiser (8 layers, 512 wide, 8 heads, classifier-free guidance: condition dropped with
probability 0.25, guidance weight 2) trained with MACE's data: the cached FineDance training set (183 songs, 19,892
slices of 240 frames), its normalizer, the 35-d baseline music features, Adan 4e-4 / weight decay 0.02, EMA 0.9999,
the same losses as MACE (reconstruction, joint, velocity, FK, foot contact). Only the denoiser and the guidance scheme
differ (`train.py --arch edge`, `model/edge_model.py`).

## What to copy to the cluster

From `Expert-Motion/`: the code (`*.py`, `model/`, `dataset/`, `hprc/`, `requirements.txt`) and, in
`data/dataset_backups/`, only `train_tensor_dataset.pkl` (3.5 GB), `test_tensor_dataset.pkl` and `normalizer.pkl`.
Not needed: `processed_*_data.pkl`, `runs/`, `eval/`, `wandb/`, `logs/`, `.venv/`, `graphify-out/`.

## Environment

`bash hprc/setup_env.sh` on the login node (load the cluster's Python 3.10 module first). It builds a venv that mirrors
the local one: torch 2.4.0+cu121 and the prebuilt pytorch3d 0.7.8 wheel for py310 / cu121 / torch 2.4.0 (the wheel index
resolves; install on the cluster itself is untested). `--arch edge` needs no Mamba (`mamba_ssm` is imported optionally).

## Copy (from this machine)

```bash
rsync -avh --progress --exclude /runs --exclude /eval --exclude /wandb --exclude /logs --exclude /.venv --exclude /graphify-out --exclude __pycache__ --exclude /data/test --exclude '/data/dataset_backups/processed_*' --exclude '/data/dataset_backups/*.npz' /home/sam0505/Documents/Code/MACE-Dance/Expert-Motion/ <netid>@grace.hprc.tamu.edu:/scratch/user/<netid>/edge_fd/
```

Runs from any directory (absolute source path, excludes anchored to the top level; keep the trailing `/` on the source).

Large files go to `$SCRATCH`, not `$HOME`. `train_tensor_dataset.pkl` is 3.4 GB.

## Pilot, then the run

1. Pilot (cost, not quality): 2-3 epochs on 1 GPU, then on 2 GPUs (Grace has at most a100:2 per node):
   `EPOCHS=3 SAVE=1 EXP=edge_pilot1 sbatch hprc/train_edge_finedance.slurm` and
   `GPUS=2 EPOCHS=3 SAVE=1 EXP=edge_pilot2 sbatch --gres=gpu:a100:2 --cpus-per-task=16 hprc/train_edge_finedance.slurm`.
   Read seconds per epoch from the log and the SU charge from the scheduler's accounting; decide the epoch count and
   GPU count from that. Multi-GPU is untested (the local machine has one GPU). The effective batch stays 128.
2. `EPOCHS=1000 SAVE=25 EXP=edge_fd sbatch hprc/train_edge_finedance.slurm`; if the walltime ends first,
   `sbatch --dependency=afterany:<jobid> hprc/train_edge_finedance.slurm` continues from the newest checkpoint.
3. Copy `runs/train/edge_fd/weights/train-<n>.pt` back and generate locally:
   `cd Expert-Motion && .venv/bin/python ../Evaluation-Musicality/gen_mace_test.py --stitch --seeds 10 --checkpoint
   runs/train/edge_fd/weights/train-<n>.pt --out_dir ../Evaluation-Musicality/data/mace_test_edge_fd`.
