"""Prepare FineDance Y-up pose/translation arrays for the MEGADance genre classifier (2026-11-10).

Reads the Z-up exports of export_rot_all.py (oneshot = MACE released, gain = ours) and the real test slices
(Expert-Motion/data/test/motions_sliced, already Y-up), converts the exports back to Y-up with the exact inverse of
export_smpl_examples.yup_to_zup (checked by a round trip on the real slices), and writes one npz for genre_metric.py.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/prep_genre_inputs.py --seed 0
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion, quaternion_multiply,
                                  quaternion_to_axis_angle)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from export_smpl_examples import yup_to_zup  # noqa: E402


def zup_to_yup(q, pos):
    """Exact inverse of yup_to_zup. q (T, 72), pos (T, 3)."""
    q = torch.as_tensor(q, dtype=torch.float32).reshape(1, -1, 24, 3).clone()
    pos = torch.as_tensor(pos, dtype=torch.float32).reshape(1, -1, 3)
    rot_inv = torch.tensor([0.7071068, -0.7071068, 0, 0]).unsqueeze(0)
    q[:, :, :1] = quaternion_to_axis_angle(quaternion_multiply(rot_inv, axis_angle_to_quaternion(q[:, :, :1])))
    pos = RotateAxisAngle(-90, axis='X', degrees=True).transform_points(pos)
    return q[0].reshape(-1, 72).numpy(), pos[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    d = os.path.join(HERE, 'data', 'rot_export')
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    genres = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['genres']
    real = [pickle.load(open(f'data/test/motions_sliced/{n}.pkl', 'rb')) for n in names]
    rq, rp = np.stack([m['q'] for m in real]).astype(np.float32), np.stack([m['pos'] for m in real]).astype(np.float32)

    # round trip on the real slices: Y-up -> Z-up -> Y-up must give the input back (rotations compared as matrices)
    errs = []
    for i in range(0, len(names), 12):
        q2, p2 = zup_to_yup(*yup_to_zup(rq[i], rp[i]))
        errs.append((float(np.abs(p2 - rp[i]).max()),
                     float(np.abs(axis_angle_to_quaternion(torch.tensor(q2).reshape(-1, 3)).abs()
                                  - axis_angle_to_quaternion(torch.tensor(rq[i]).reshape(-1, 3)).abs()).max())))
    print('round trip max |pos| error, max |quat| error:', max(e[0] for e in errs), max(e[1] for e in errs), flush=True)
    assert max(e[0] for e in errs) < 1e-4 and max(e[1] for e in errs) < 1e-4

    out = {'names': np.array(names), 'genres': np.array(genres), 'real_q': rq, 'real_pos': rp}
    for v in ('oneshot', 'gain'):
        z = np.load(os.path.join(d, f'{v}_seed{a.seed}.npz'))
        qs, ps = zip(*[zup_to_yup(z['poses'][i], z['trans'][i]) for i in range(len(names))])
        out[f'{v}_q'], out[f'{v}_pos'] = np.stack(qs), np.stack(ps)
    lz = os.path.join(d, f'lodge_seed{a.seed}.npz')          # Lodge: already FineDance Y-up axis-angle
    if os.path.exists(lz):
        z = np.load(lz)
        out['lodge_q'], out['lodge_pos'] = z['poses'], z['trans']
    mz = os.path.join(d, 'mega_native.npz')                  # MEGADance: native decoded 6D, Y-up (deterministic)
    if os.path.exists(mz):
        z = np.load(mz)
        out['mega_p6'], out['mega_pos'] = z['p6'], z['trans']
    print('real pos global mean/std (classifier trained with mean -0.0101, std 0.6421):', rp.mean(), rp.std())
    for v in ('oneshot', 'gain'):
        print(v, 'pos mean/std', out[f'{v}_pos'].mean(), out[f'{v}_pos'].std())
    np.savez(os.path.join(d, f'genre_inputs_seed{a.seed}.npz'), **out)
    print('wrote', os.path.join(d, f'genre_inputs_seed{a.seed}.npz'))


if __name__ == '__main__':
    main()
