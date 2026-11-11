"""AIST++ 10-genre dance classifier for the style metrics FID_s / DIV_s (MEGADance's FineDance classifier cannot score AIST++).

Architecture = MEGADance's Dance_Encoder (models/CLS/classification.py: 8 x [1-layer transformer encoder, AvgPool stride 2],
hidden 512, 8 heads, dropout .25, mean over time, MLP 512-128-out) with a 10-way output instead of 20.
Input per frame = root translation relative to the clip's first frame (3, metres) + the 24 joints relative to the root (72)
= 75, from JOINT POSITIONS (Y-up, 30 fps) so that every model can be scored (Bailando outputs positions only; 2026-11-11),
standardised with train mean / std. Clips are 5 s (150 frames).
Data: Expert-Motion/data/aist/{train,test}/motions_sliced (EDGE split); validation = every 20th training sequence
(held out), as in train.py. Label = genre from the file name (gBR gPO gLO gWA gMH gLH gHO gKR gJS gJB).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/aist_genre_cls.py --train      # -> Evaluation-Musicality/data/aist/genre_cls.pt
Use: aist_table.py calls `embed(load_model(), joints)`.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from aist_common import GENRES  # noqa: E402

CKPT = os.path.join(HERE, 'data', 'aist', 'genre_cls.pt')
DATA = os.path.join(HERE, '..', 'Expert-Motion', 'data', 'aist')


class TTransformer(nn.Module):
    def __init__(self, hidden=512, heads=8):
        super().__init__()
        self.layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dropout=0.25), num_layers=1)

    def forward(self, x):
        return rearrange(self.layer(rearrange(x, 'b t c -> t b c')), 't b c -> b t c')


class DownSample(nn.Module):
    def __init__(self, rate=2):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=3, stride=rate, padding=1)

    def forward(self, x):
        return rearrange(self.pool(rearrange(x, 'b t c -> b c t')), 'b c t -> b t c')


class DanceEncoder(nn.Module):
    def __init__(self, n_out=10, d_in=75, hidden=512, heads=8, layers=8):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.Linear(hidden, hidden))
        self.out = nn.Sequential(nn.Linear(hidden, 128), nn.Linear(128, n_out))
        blocks = nn.Sequential()
        for i in range(layers):
            blocks.add_module(f'former{i}', TTransformer(hidden, heads))
            blocks.add_module(f'down{i}', DownSample(2))
        self.blocks = blocks

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)).mean(1))


def features(j):
    """j (B, T, 24, 3) Y-up joints, metres -> (B, T, 75)."""
    root = j[:, :, 0]
    return torch.cat([root - root[:, :1], (j - j[:, :, :1]).flatten(2)], -1)


def fk(pos, q):
    from vis import SMPLSkeleton
    sk = SMPLSkeleton(device='cpu')
    out = []
    with torch.no_grad():
        for i in range(0, len(q), 512):
            out.append(sk.forward(q[i:i + 512].float().reshape(-1, q.shape[1], 24, 3), pos[i:i + 512].float()))
    return torch.cat(out)


def load(split, val_every=20):
    files = sorted(glob.glob(f'{DATA}/{split}/motions_sliced/*.pkl'))
    pos, q, y, seq = [], [], [], []
    for f in files:
        d = pickle.load(open(f, 'rb'))
        n = os.path.basename(f)[:-4]
        pos.append(d['pos'])
        q.append(d['q'])
        y.append(GENRES.index(n[:3]))
        seq.append(n.rsplit('_slice', 1)[0])
    return torch.tensor(np.stack(pos)), torch.tensor(np.stack(q)), torch.tensor(y), np.array(seq)


def load_model(path=CKPT, device='cuda'):
    c = torch.load(path, map_location=device, weights_only=False)
    m = DanceEncoder().to(device)
    m.load_state_dict(c['model'])
    return m.eval(), c['mean'].to(device), c['std'].to(device)


@torch.no_grad()
def embed(model_pack, joints, device='cuda', bs=256):
    """joints (N, T, 24, 3) torch, Y-up, 30 fps -> (N, 10) classifier output (the 'style feature')."""
    m, mean, std = model_pack
    out = []
    for i in range(0, len(joints), bs):
        x = features(joints[i:i + bs].float().to(device))
        out.append(m((x - mean) / std).cpu())
    return torch.cat(out)


def train(args):
    dev = 'cuda'
    torch.manual_seed(0)
    pos, q, y, seq = load('train')
    seqs = sorted(set(seq))
    val_seqs = set(seqs[::20])
    isv = np.array([s in val_seqs for s in seq])
    x = features(fk(pos, q))
    mean, std = x[~isv].mean((0, 1)), x[~isv].std((0, 1)) + 1e-6
    x = (x - mean) / std
    tp, tq, ty, _ = load('test')
    xt = (features(fk(tp, tq)) - mean) / std
    xtr, ytr = x[~isv], y[~isv]
    xva, yva = x[isv], y[isv]
    m = DanceEncoder().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr, betas=(0.5, 0.999))
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.33)
    best = -1
    for ep in range(1, args.epochs + 1):
        m.train()
        perm = torch.randperm(len(xtr))
        tot = 0
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            j = perm[i:i + args.batch]
            loss = nn.functional.cross_entropy(m(xtr[j].to(dev)), ytr[j].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        sch.step()
        m.eval()
        with torch.no_grad():
            def acc(a, b):
                hit = sum(int((m(a[i:i + 256].to(dev)).argmax(1).cpu() == b[i:i + 256]).sum()) for i in range(0, len(a), 256))
                return hit / len(a)
            va, te = acc(xva, yva), acc(xt, ty)
        print(f'epoch {ep} loss {tot / (len(perm) // args.batch):.4f} val acc {va:.3f} test acc {te:.3f}', flush=True)
        if va > best:
            best = va
            os.makedirs(os.path.dirname(CKPT), exist_ok=True)
            torch.save({'model': m.state_dict(), 'mean': mean, 'std': std, 'epoch': ep, 'val_acc': va, 'test_acc': te}, CKPT)
    print(f'best val acc {best:.3f}; saved {CKPT}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    a = ap.parse_args()
    if a.train:
        train(a)
