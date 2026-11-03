"""Musicality and jitter of a fine-tuned Motion Expert on the 19 FineDance test songs (lambda sweep, 2026-11-02).

Input: gen_mace_test.py output for one checkpoint (10 seeds x 84 slices). Musicality uses the test-set pipeline of
perc_distinct.py / unique_parts.py unchanged (tempo-first wrong songs, lag-max r within 300 ms, per-song units):
the trained pair (slow treble -> body energy) and five pairs not in the loss. Jitter: mean joint speed, mean joint
acceleration, and the share of joint-velocity power above 5 Hz, averaged over slices and seeds, next to real dance
(release motion).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/finetune_eval.py --label released --mace_dir ../Evaluation-Musicality/data/mace_test
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from model.metric import SMPLSkeleton  # noqa: E402
from multicomponent import UPPER  # noqa: E402
from finedance_common import SLICE_LEN, SLICE_STRIDE, release_joints  # noqa: E402
from accent_overlap import full_music  # noqa: E402
import perc_distinct as PD  # noqa: E402
import unique_parts as UP  # noqa: E402

OUT = os.path.join(HERE, 'results', 'finetune_sweep.json')
PAIRS = {'trained: treble slow -> energy': ('treble_slow', 'energy_slow'),
         'treble fast -> energy': ('treble_fast', 'energy_fast'),
         'treble unique slow -> energy': ('treble_u_slow', 'energy_slow'),
         'perc slow -> energy': ('perc_slow', 'energy_slow'),
         'perc fast -> upper accel': ('perc_fast', 'uacc'),
         'loud slow -> energy': ('loud_slow', 'energy_slow')}


def motion_views(j, src):
    return {**UP.motion_views(j, src), **PD.motion_views(j, UPPER)}


def jitter(j):
    """j: (T, 24, 3) -> (mean speed m/s, mean acceleration m/s^2, share of velocity power above 5 Hz)."""
    j = np.asarray(j, dtype=np.float64)
    v = np.diff(j, axis=0) * 30
    a = np.diff(j, 2, axis=0) * 900
    p = np.abs(np.fft.rfft(v - v.mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(len(v), 1 / 30)
    return np.linalg.norm(v, axis=2).mean(), np.linalg.norm(a, axis=2).mean(), p[f > 5].sum() / p.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--mace_dir', required=True)
    args = ap.parse_args()
    PD.MACE_DIR = args.mace_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(device)

    units, own, wrong, real, model = PD.load_dataset('test', smpl, device, full_music, UP.music_views, motion_views)
    stats, _ = PD.score_pairs(units, own, wrong, real, model, PAIRS)

    names = json.load(open(os.path.join(args.mace_dir, 'names.json')))['names']
    gen = np.load(os.path.join(args.mace_dir, 'joints.npy'))
    jm = np.array([[jitter(x) for x in row] for row in gen]).mean((0, 1))
    full = {}
    jr = []
    for n in names:
        s, k = n.split('_slice')
        if s not in full:
            full[s] = release_joints(s, smpl, device)
        jr.append(jitter(full[s][SLICE_STRIDE * int(k):SLICE_STRIDE * int(k) + SLICE_LEN]))
    jr = np.mean(jr, 0)
    jit = {'model': dict(zip(('speed', 'accel', 'hf_share'), map(float, jm))),
           'real': dict(zip(('speed', 'accel', 'hf_share'), map(float, jr)))}
    print(f"jitter  model: speed {jm[0]:.2f} m/s, accel {jm[1]:.1f} m/s^2, >5 Hz {jm[2]:.3f} | "
          f"real: {jr[0]:.2f}, {jr[1]:.1f}, {jr[2]:.3f}")

    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    results[args.label] = {'mace_dir': args.mace_dir, 'pairs': stats, 'jitter': jit}
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
