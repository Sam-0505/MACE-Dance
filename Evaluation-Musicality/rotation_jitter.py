"""Whole-body rotation jitter about the root (2026-11-06): the twitch beat-locking guidance produced by rotating the whole
skeleton about the pelvis (69 % of the guidance step went to the root's global rotation).

Per frame, the rigid rotation (Kabsch) of a joint group, positions relative to the pelvis, against the clip's first
frame; its frame-to-frame angle gives a turning speed (deg/s). Jitter = RMS of the turning speed minus its 0.1 s
Gaussian-smoothed version: fast back-and-forth rotation, not turning as such.
  pelvis : pelvis, hips, spine1 (SMPL 0, 1, 2, 3)  -> rotation of the hip block, i.e. the root rotation
  torso  : pelvis, hips, spines, neck, collars, head, shoulders -> rotation of the whole upper body block
FineDance real dance (motion_quality_extra.real_joints) is the reference.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/rotation_jitter.py --fd released=mace_test ours=mace_test_... --seeds 3
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d as g1

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
GROUPS = {'pelvis': [0, 1, 2, 3], 'torso': [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17]}


def turn_speed(j, idx):
    x = j[:, idx] - j[:, :1]
    h = np.einsum('ji,tjk->tik', x[0], x)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum('tji,tkj->tik', vt, u)))
    vt[:, -1] *= d[:, None]
    r = np.einsum('tji,tkj->tik', vt, u)
    rel = np.einsum('tij,tkj->tik', r[1:], r[:-1])
    return np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))) * 30


def clip_stats(j):
    out = {}
    for g, idx in GROUPS.items():
        a = turn_speed(j, idx)
        out[f'{g}_speed'] = float(a.mean())
        out[f'{g}_jitter'] = float(np.sqrt(((a - g1(a, 3)) ** 2).mean()))
    return out


def main():
    from motion_quality_extra import real_joints
    ap = argparse.ArgumentParser()
    ap.add_argument('--fd', nargs='+', required=True, help='label=dir under data/')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    res = {}
    sets = [('real', real_joints(names)[:, None])]
    for lab, d in (x.split('=', 1) for x in a.fd):
        sets.append((lab, np.load(os.path.join(HERE, 'data', d, 'joints.npy'), mmap_mode='r')))
    for lab, J in sets:
        rows = [clip_stats(np.asarray(J[i, s], np.float64)) for i in range(len(names)) for s in range(min(a.seeds, J.shape[1]))]
        res[lab] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(f'{lab:28s} ' + '  '.join(f'{k} {v:6.1f}' for k, v in res[lab].items()), flush=True)
    json.dump(res, open(os.path.join(HERE, 'results', f'rotation_jitter{a.tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
