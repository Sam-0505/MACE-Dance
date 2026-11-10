"""Critic loading and guidance glue (shared by critic4.py, critic_gate4.py, mace_sampler.py, gen_mace_*.py).

The critic in use is v5 (critic4.py --lite): a beat-synchronous, phase-marginalised critic trained on REAL FineDance dance
only, then frozen, so the guidance never contains our beat-locking formula. Earlier critics (v1 contrastive, v2, v3, v4
full-capacity) were removed on 2026-11-08; their results stay in results/ and logs/, their checkpoints no longer load.
"""
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
BEAT_COL = 34  # beat one-hot column of the 35 baseline music features
SIGMA = 2  # Gaussian low-pass of the joints (frames), as the guidance
CKPT = os.path.join(HERE, 'data', 'critic', 'critic_v5.pt')


def load_critic(device='cuda', path=None):
    """Frozen critic v5 (critic4.Critic4); comma-separated paths give critic4.Ensemble (mean score)."""
    from critic4 import Critic4, Ensemble
    if path and ',' in path:
        return Ensemble([load_critic(device, p) for p in path.split(',')]).eval()
    ck = torch.load(path or CKPT, map_location=device)
    if ck.get('kind') != 'd' or not ck['cfg'].get('lite'):
        raise ValueError(f'{path or CKPT}: not a critic v5 checkpoint (old critics were removed 2026-11-08)')
    m = Critic4(ck['cfg']).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def windowed_critic(critic, joints, feats, null_feats, win=240, stride=120):
    """Guidance score: critic on the own beat grid - mean over other songs' grids, averaged over windows of win frames.
    joints (B, T, 24, 3) world joints (differentiable), feats (B, T, 35), null_feats list of (B, T, 35). Returns (B,)."""
    return critic.windowed(joints, feats, null_feats, win=win, stride=stride)


def neg_mask(songs, nbeats):
    """True where j is a valid negative of i: another song, beat count within 1."""
    s = torch.as_tensor(songs, device=nbeats.device)
    return (s.view(-1, 1) != s.view(1, -1)) & ((nbeats.view(-1, 1) - nbeats.view(1, -1)).abs() <= 1)


def load_train():
    """Training clips: normalised pose (N, 240, 151), features (N, 240, 35), song id (N,), the normalizer."""
    ds = pickle.load(open('data/dataset_backups/train_tensor_dataset.pkl', 'rb'))
    pose = torch.as_tensor(np.asarray(ds.data['pose']), dtype=torch.float32)
    feats = torch.as_tensor(np.asarray(ds.data['features']), dtype=torch.float32)
    names = [os.path.basename(n).split('_')[0] for n in ds.data['filenames']]
    ids = {s: i for i, s in enumerate(dict.fromkeys(names))}
    return pose, feats, np.array([ids[n] for n in names]), ds.normalizer
