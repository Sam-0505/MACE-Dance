"""Generate Motion Expert dances on the ImperialDance short-clip music, for a same-song comparison.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/gen_mace_imperial.py --n 40

Writes Evaluation-Musicality/data/imperial/mace/<choreo>.npy: (n, T, 24, 3) SMPL joints at 30 fps,
one sample per noise seed, generated at the music clip's own length.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from EDGE import EDGE
from model.metric import style2s
from dataset.quaternion import ax_from_6v
import audio_features as af
from imperial_subset import SHORT_CLIPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='./runs/train/exp/weights/train-3750.pt')
    ap.add_argument('--audio_dir', default=os.path.join(HERE, 'data', 'imperial', 'short_audio'))
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'data', 'imperial', 'mace'))
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = EDGE('baseline', checkpoint_path=args.checkpoint)
    model.eval()
    diffusion = model.diffusion
    device = model.accelerator.device

    for c, src in SHORT_CLIPS.items():
        feat = af.extract_file(os.path.join(args.audio_dir, c + '.wav'))
        cond = torch.from_numpy(feat).float().unsqueeze(0).repeat(args.n, 1, 1).to(device)
        genre = torch.full((args.n,), style2s.index(src['genre']), dtype=torch.long, device=device)
        torch.manual_seed(args.seed)
        with torch.no_grad():
            x = diffusion.ddim_sample((args.n, cond.shape[1], model.repr_dim), cond, genre)
            x = model.normalizer.unnormalize(x.detach().cpu())
            _, x = torch.split(x, (4, x.shape[2] - 4), dim=2)          # drop foot contacts
            pos = x[:, :, :3].to(device)
            q = ax_from_6v(x[:, :, 3:].reshape(args.n, x.shape[1], 24, 6)).to(device)
            joints = diffusion.smpl.forward(q, pos).cpu().numpy()
        np.save(os.path.join(args.out_dir, c + '.npy'), joints.astype(np.float32))
        print(c, src['genre'], joints.shape)


if __name__ == '__main__':
    main()
