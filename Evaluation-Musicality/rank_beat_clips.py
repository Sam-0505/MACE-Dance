"""Rank test clips for the beat-bead / beat-photo videos (2026-11-09).

Per clip and dancer (real, MACE released x10 seeds, ours x10 seeds) two numbers from the beat grid:
  pose ratio : RMS pose spread over 8 consecutive beats (24 joints, pelvis-relative, screen plane) on the grid, divided by the
               same spread averaged over the grid shifted by 0..0.95 beat. < 1 = the dancer lands in the same place at the
               beats more than at other times. Computed on the true beat (shift 0) and at the dancer's own best shift
               (own = shift maximising the 4-limb z-speed at the grid).
  speed depth: (max - min)/2 of the 4-limb z-speed at the grid over shifts (the beads signal), and the shift of its max.
Ranking uses the MEDIAN over seeds of (released - ours), so a clip is not picked on one lucky seed. Nothing here uses the critic
score or the locking formula. Writes results/rank_beat_clips.json.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/rank_beat_clips.py [--ours_dir DIR]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import render_fix_compare as R  # noqa: E402
from eval_mace import fk  # noqa: E402
from model.metric import SMPLSkeleton  # noqa: E402
from render_eas_compare import view_axes  # noqa: E402

FPS, K = 30, 8
LIMBS = [7, 8, 20, 21]
SHIFTS = np.linspace(0, 0.95, 20)
OURS_DIR = os.path.join(HERE, 'data', 'mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10')


def analyse(j, A, iv, T):
    h, up, s = view_axes(j[:T])
    rel = j[:T] - j[:T, :1]
    xy = np.stack([rel[..., h], s * rel[..., up]], -1)
    v = gaussian_filter1d(np.linalg.norm(np.gradient(xy[:, LIMBS], axis=0) * FPS, axis=-1), 1.0, axis=0).mean(1)
    z = (v - v.mean()) / (v.std() + 1e-8)
    grids = [np.clip(np.round(A + s_ * iv).astype(int), 0, T - 1) for s_ in SHIFTS]
    prof = np.array([z[g].mean() for g in grids])
    spr = np.array([np.mean([np.sqrt(((xy[g[k:k + K]] - xy[g[k:k + K]].mean(0)) ** 2).sum(-1).mean())
                             for k in range(0, len(g) - K, 2)]) for g in grids])
    b = int(prof.argmax())
    return {'ratio0': float(spr[0] / spr.mean()), 'ratio_own': float(spr[b] / spr.mean()), 'own_shift': float(SHIFTS[b]),
            'depth': float((prof.max() - prof.min()) / 2), 'z0': float(prof[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours_dir', default=OURS_DIR)
    a = ap.parse_args()
    names = json.load(open(os.path.join(R.ONESHOT, 'names.json')))['names']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(dev)
    one_all = np.load(os.path.join(R.ONESHOT, 'joints.npy'), mmap_mode='r')
    ours_all = np.load(os.path.join(a.ours_dir, 'joints.npy'), mmap_mode='r')
    res = {}
    for i, n in enumerate(names):
        A = np.flatnonzero(np.load(os.path.join('data/test/baseline_feats', n + '.npy'))[:, 34])
        if len(A) < 20:
            continue
        m = pickle.load(open(os.path.join('data/test/motions_sliced', n + '.pkl'), 'rb'))
        gt = fk(smpl, [m['q']], [m['pos']], dev, from_pred=False)[0]
        one, ours = np.asarray(one_all[i], np.float32), np.asarray(ours_all[i], np.float32)
        T = min(len(gt), one.shape[1], ours.shape[1])
        A = A[A < T - 1]
        iv = np.polyfit(np.arange(len(A)), A, 1)[0]
        res[n] = {'bpm': float(60 * FPS / iv), 'real': analyse(gt, A, iv, T),
                  'one': [analyse(j, A, iv, T) for j in one], 'ours': [analyse(j, A, iv, T) for j in ours]}
        print(n, f'{res[n]["bpm"]:.0f} BPM', flush=True)
    json.dump(res, open(os.path.join(HERE, 'results', 'rank_beat_clips.json'), 'w'))
    summarise(res)


def summarise(res):
    rows = []
    for n, r in res.items():
        g = lambda key, who: np.array([x[key] for x in r[who]])  # noqa: E731
        rows.append({'n': n, 'bpm': r['bpm'],
                     'real_r0': r['real']['ratio0'], 'one_r0': np.median(g('ratio0', 'one')), 'our_r0': np.median(g('ratio0', 'ours')),
                     'real_ro': r['real']['ratio_own'], 'one_ro': np.median(g('ratio_own', 'one')), 'our_ro': np.median(g('ratio_own', 'ours')),
                     'real_d': r['real']['depth'], 'one_d': np.median(g('depth', 'one')), 'our_d': np.median(g('depth', 'ours')),
                     'real_s': r['real']['own_shift'], 'our_s': float(np.median(g('own_shift', 'ours'))),
                     'gap0': float(np.median(g('ratio0', 'one') - g('ratio0', 'ours'))),
                     'gapo': float(np.median(g('ratio_own', 'one') - g('ratio_own', 'ours'))),
                     'gapd': float(np.median(g('depth', 'ours') - g('depth', 'one')))})
    print(f'\n{len(rows)} clips. medians over 10 seeds.  r0 / ro = pose ratio on true beat / own phase (<1 = lands in the same place)')
    for key, title in (('gap0', 'true-beat pose gap (released - ours)'), ('gapo', 'own-phase pose gap'), ('gapd', 'speed-depth gap (ours - released)')):
        print(f'\n== top 8 by {title}')
        print(f'{"clip":12s} {"bpm":>4s} | r0 real/rel/ours | ro real/rel/ours | depth real/rel/ours | shift real/ours | gap')
        for x in sorted(rows, key=lambda x: -x[key])[:8]:
            print(f'{x["n"]:12s} {x["bpm"]:4.0f} | {x["real_r0"]:.2f} {x["one_r0"]:.2f} {x["our_r0"]:.2f} | {x["real_ro"]:.2f} {x["one_ro"]:.2f} {x["our_ro"]:.2f} '
                  f'| {x["real_d"]:.2f} {x["one_d"]:.2f} {x["our_d"]:.2f} | {x["real_s"]:.2f} {x["our_s"]:.2f} | {x[key]:+.3f}')
    print('\nall clips: median over clips  r0 real/rel/ours', *(f'{np.median([x[k] for x in rows]):.3f}' for k in ('real_r0', 'one_r0', 'our_r0')),
          ' ro', *(f'{np.median([x[k] for x in rows]):.3f}' for k in ('real_ro', 'one_ro', 'our_ro')),
          ' depth', *(f'{np.median([x[k] for x in rows]):.2f}' for k in ('real_d', 'one_d', 'our_d')))


if __name__ == '__main__':
    main()
