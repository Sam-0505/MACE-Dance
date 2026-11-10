#!/bin/bash
# One-off environment for training on the cluster login node (needs internet). Mirrors the local .venv:
# Python 3.10, torch 2.4.0+cu121, pytorch3d 0.7.8 from the prebuilt wheel for py310 / cu121 / torch 2.4.0.
# EDIT: load the cluster's Python 3.10 module first, e.g.  module load <python 3.10 module>  (see `module spider python`)
set -e
ENV=${ENV:-$HOME/envs/edge}
python3 --version | grep -q "3.10" || { echo "need python 3.10, got $(python3 --version)"; exit 1; }
python3 -m venv $ENV
. $ENV/bin/activate
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 scipy==1.11.2 einops==0.8.1 accelerate==1.10.1 wandb==0.22.2 tqdm==4.66.1 matplotlib==3.8.0 \
            p_tqdm==1.4.2 librosa==0.11.0 soundfile==0.13.1
pip install pytorch3d==0.7.8 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html
python -c "import torch, pytorch3d, accelerate, einops, wandb; print('imports ok, torch', torch.__version__)"
echo "env at $ENV  (python: $ENV/bin/python)"
