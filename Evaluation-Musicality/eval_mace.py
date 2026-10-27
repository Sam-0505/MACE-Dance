"""Musicality metrics for Motion Expert outputs vs. ground truth.

Run from Expert-Motion with its venv, after test.py has written predictions:
    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/eval_mace.py --pred_dir eval/train_exp2/0
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from einops import rearrange

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())  # Expert-Motion, for its SMPL FK and axis helpers

from model.metric import SMPLSkeleton, align_axis, rotate_pred_like_gt
import audio_features as af
import metrics as M


def fk(smpl, poses, trans, device, from_pred):
    """Joints exactly as the authors' Metric builds them: (N, T, 24, 3)."""
    poses = rearrange(torch.from_numpy(np.asarray(poses)).to(device), 'b t (j c) -> b t j c', c=3)
    trans = torch.from_numpy(np.asarray(trans)).to(device)
    if from_pred:
        poses, trans = rotate_pred_like_gt(poses, trans)
    return align_axis(smpl.forward(poses, trans).detach().cpu()).numpy()


def report(name, rows):
    s = M.summarize(rows)
    line = ' | '.join(f'{k} {m:.3f}±{sd:.3f}' for k, (m, sd) in s.items() if not k.endswith('_lag'))
    print(f'{name:<22} n={len(rows):<3} {line}')
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_dir', default='data/test')
    ap.add_argument('--pred_dir', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'mace.json'))
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(device)

    names = sorted(f[:-4] for f in os.listdir(args.pred_dir) if f.endswith('.pkl'))
    musics, rmss, gt_poses, gt_trans, pr_poses, pr_trans = [], [], [], [], [], []
    for n in names:
        musics.append(np.load(os.path.join(args.gt_dir, 'baseline_feats', n + '.npy')))
        rmss.append(af.rms_envelope(af.load_audio(os.path.join(args.gt_dir, 'wavs_sliced', n + '.wav'))))
        with open(os.path.join(args.gt_dir, 'motions_sliced', n + '.pkl'), 'rb') as f:
            m = pickle.load(f)
        gt_poses.append(m['q']); gt_trans.append(m['pos'])
        with open(os.path.join(args.pred_dir, n + '.pkl'), 'rb') as f:
            p = pickle.load(f)
        pr_poses.append(p['smpl_poses']); pr_trans.append(p['smpl_trans'])

    gt = fk(smpl, gt_poses, gt_trans, device, from_pred=False)
    pred = fk(smpl, pr_poses, pr_trans, device, from_pred=True)
    songs = [n.split('_')[0] for n in names]

    rows = {
        'GT': [M.clip_scores(m, r, j) for m, r, j in zip(musics, rmss, gt)],
        'Pred': [M.clip_scores(m, r, j) for m, r, j in zip(musics, rmss, pred)],
        'GT (mismatched)': M.mismatched_scores(musics, rmss, list(gt), songs),
        'Pred (mismatched)': M.mismatched_scores(musics, rmss, list(pred), songs),
    }
    print(f'{len(names)} clips from {len(set(songs))} songs')
    out = {k: report(k, v) for k, v in rows.items()}
    for k in ('GT', 'Pred'):
        out[k + '_ci95'] = {m: M.bootstrap_ci([r[m] for r in rows[k]]) for m in ('BAS', 'VBCS', 'ABHS', 'AAS', 'EAS')}
    out['per_clip'] = {k: dict(zip(names, v)) for k, v in rows.items() if '(' not in k}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print('saved', args.out)


if __name__ == '__main__':
    main()
