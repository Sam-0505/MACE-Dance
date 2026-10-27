"""Thin wrapper around the Motion Expert for generating SMPL joints from 35-dim baseline features.
Import from inside Expert-Motion (its modules and checkpoint paths are relative to it)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from EDGE import EDGE
from model.metric import style2s
from dataset.quaternion import ax_from_6v


class MaceSampler:
    def __init__(self, checkpoint='./runs/train/exp/weights/train-3750.pt'):
        self.model = EDGE('baseline', checkpoint_path=checkpoint)
        self.model.eval()
        self.device = self.model.accelerator.device

    @torch.no_grad()
    def sample(self, feats, genres, seed):
        """feats: (B, T, 35) raw baseline features; genres: list of style names. Returns (B, T, 24, 3) joints.
        The same seed gives the same noise for the same batch shape, so runs differing only in
        conditioning are paired."""
        cond = torch.as_tensor(np.asarray(feats), dtype=torch.float32, device=self.device)
        genre = torch.tensor([style2s.index(g) for g in genres], dtype=torch.long, device=self.device)
        torch.manual_seed(seed)
        d = self.model.diffusion
        x = d.ddim_sample((cond.shape[0], cond.shape[1], self.model.repr_dim), cond, genre)
        x = self.model.normalizer.unnormalize(x.detach().cpu())
        _, x = torch.split(x, (4, x.shape[2] - 4), dim=2)
        q = ax_from_6v(x[:, :, 3:].reshape(x.shape[0], x.shape[1], 24, 6)).to(self.device)
        return d.smpl.forward(q, x[:, :, :3].to(self.device)).cpu().numpy()
