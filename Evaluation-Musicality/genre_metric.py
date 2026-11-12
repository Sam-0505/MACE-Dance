"""MEGADance style metrics FID_s / DIV_s (+ genre accuracy) for real dance, MACE released and ours (2026-11-10).

Uses the MEGADance repo's own genre classifier (output/cls/30) and its metric functions (utils/metric.py: normalize by the
real features, calc_fid, calculate_avg_distance, compute_acc), on the 20-dim classifier output of 1024-frame clips.
Inputs: data/rot_export/genre_inputs_seed<k>.npz from prep_genre_inputs.py (FineDance Y-up axis-angle poses + root
translation in metres; 84 test slices of 1020 frames, padded to 1024 by repeating the last frame).

Pose convention: MEGADance's dataloader turns the stored (T, 72) poses into 6D with rotation_matrix_to_rotation_6d, which
reads each joint's 3 numbers as EULER angles. FineDance's raw poses are axis-angle, so this is a mismatch in their pipeline
but it is what the classifier was trained on. We report both conventions ('euler' = their transform, 'axis' = proper
Rodrigues); the one on which REAL dance is classified better is the one the classifier expects.

    cd /home/sam0505/Documents/Code/dance_baselines/MEGADance
    .venv/bin/python /home/sam0505/Documents/Code/MACE-Dance/Evaluation-Musicality/genre_metric.py --seed 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MEGA = '/home/sam0505/Documents/Code/dance_baselines/MEGADance'
os.chdir(MEGA)
sys.path.insert(0, MEGA)

from models.CLS.classification import CLS  # noqa: E402
from utils.load_model import load_model  # noqa: E402
from utils.metric import calc_fid, calculate_avg_distance, compute_acc  # noqa: E402
from utils.utils import matrix_to_rotation_6d, rotation_matrix_to_rotation_6d  # noqa: E402

STYLE2S = ['Dai', 'ShenYun', 'Wei', 'Korean', 'Urban', 'Hiphop', 'Popping', 'Miao', 'HanTang', 'Breaking', 'Kun',
           'Locking', 'Jazz', 'Choreography', 'Chinese', 'DunHuang']      # MEGADance's label order (models/CLS/dataloader.py)
SETS = {'real': 'Real dance', 'lodge': 'Lodge', 'mega': 'MEGA', 'flower': 'FlowerDance', 'oneshot': 'MACE released',
        'gain': 'Ours (critic v5 + gain)'}
L = 1024


def pad(x):
    return np.concatenate([x, np.repeat(x[:, -1:], L - x.shape[1], axis=1)], axis=1)


def axis_angle_to_matrix(aa):
    """Rodrigues, (..., 3) -> (..., 3, 3)."""
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / th
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = torch.zeros_like(kx)
    K = torch.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], dim=-1).reshape(*k.shape[:-1], 3, 3)
    th = th[..., None]
    return torch.eye(3, dtype=aa.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def six_d(q, conv):
    q = torch.as_tensor(q, dtype=torch.float32)
    if conv == 'euler':
        return rotation_matrix_to_rotation_6d(q)                       # MEGADance's own transform
    return matrix_to_rotation_6d(axis_angle_to_matrix(q.reshape(*q.shape[:2], 24, 3))).reshape(*q.shape[:2], 144)


@torch.no_grad()
def logits(cls, q, pos, conv, device, native6d=False):
    out = []
    for i in range(0, len(q), 16):
        p6 = (torch.as_tensor(q[i:i + 16], dtype=torch.float32) if native6d else six_d(q[i:i + 16], conv)).to(device)
        tr = torch.as_tensor(pos[i:i + 16], dtype=torch.float32).to(device)
        out.append(cls.dance_encode(tr, p6).cpu())
    return torch.cat(out).numpy()


def norm(gt, pred):
    m, s = gt.mean(0), gt.std(0)
    return (gt - m) / (s + 1e-10), (pred - m) / (s + 1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    d = np.load(os.path.join(HERE, 'data', 'rot_export', f'genre_inputs_seed{a.seed}.npz'))
    labels = torch.tensor([STYLE2S.index(g) for g in d['genres']])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cls = CLS(device).to(device)
    cls = load_model(cls, 'cls', 30)
    cls.eval()
    songs = np.array([n.split('_')[0] for n in d['names']])
    half = np.isin(songs, sorted(set(songs))[::2])                       # song-disjoint split of the real slices

    res = {}
    for conv in ('euler', 'axis'):
        f = {k: logits(cls, pad(d[f'{k}_q']) if k != 'mega' else pad(d['mega_p6']), pad(d[f'{k}_pos']), conv, device, k == 'mega')
             for k in SETS if f'{k}_pos' in d}                               # MEGA: native 6D, no convention
        r = {}
        for k in f:
            row = {'acc': compute_acc(torch.as_tensor(f[k]), labels)}
            if k != 'real':
                g, p = norm(f['real'], f[k])
                row.update({'FID_s': float(calc_fid(p, g)), 'DIV_s': float(calculate_avg_distance(p))})
            else:
                g, _ = norm(f['real'], f['real'])
                row['DIV_s'] = float(calculate_avg_distance(g))
            r[k] = row
        g, p = norm(f['real'][half], f['real'][~half])                   # floor: real vs real of other songs
        r['real_split_floor'] = {'FID_s': float(calc_fid(p, g)), 'n': [int(half.sum()), int((~half).sum())]}
        rng = np.random.default_rng(0)                                   # bootstrap over slices (paired across models)
        boot = {k: {'FID_s': [], 'DIV_s': []} for k in f if k != 'real'}
        for _ in range(300):
            idx = rng.integers(0, len(labels), len(labels))
            for k in boot:
                g, p = norm(f['real'][idx], f[k][idx])
                boot[k]['FID_s'].append(float(calc_fid(p, g))); boot[k]['DIV_s'].append(float(calculate_avg_distance(p)))
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
        r['bootstrap_ci95'] = {k: {m: ci(v[m]) for m in v} for k, v in boot.items()}
        r['bootstrap_ours_minus'] = {k: {m: {'mean': float(np.mean(np.array(boot['gain'][m]) - np.array(boot[k][m]))),
                                             'ci95': ci(np.array(boot['gain'][m]) - np.array(boot[k][m]))}
                                         for m in ('FID_s', 'DIV_s')} for k in boot if k != 'gain'}
        res[conv] = r
        print(f'\n== convention: {conv}')
        for k, row in r.items():
            print(f'{SETS.get(k, k):28s}', {m: (round(v, 3) if isinstance(v, float) else v) for m, v in row.items()})
    json.dump(res, open(os.path.join(HERE, 'results', f'genre_metric_seed{a.seed}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
