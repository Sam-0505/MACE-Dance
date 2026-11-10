"""Viewer-facing beat score and clip ranking, FineDance and Motorica (2026-11-09).

hit rate = share of motion accents (peaks of the mean joint speed, the accents of beat_suite.py) that land within +-W frames
of a beat of the song (W = 2 -> +-67 ms, about where people begin to hear/see an offset). Reported as "hits on the beat out
of 10" together with the share a uniformly random accent would get ((2W+1)/beat interval), so no chance correction is needed
to read it. Per clip: real, MACE released (all seeds), ours (critic v5 + gain, all seeds); ranking on the median over seeds of
(ours - released), so a clip is not picked on one seed. Also the speed-at-beat depth used by render_beads.py.
Writes results/rank_hits_on_beat_<dataset>.json.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/rank_hits_on_beat.py [--W 2]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import beat_suite as BS  # noqa: E402
import render_fix_compare as R  # noqa: E402
from eval_mace import fk  # noqa: E402
from model.metric import SMPLSkeleton  # noqa: E402
from tmd_beat_metrics import accents, velocity  # noqa: E402

FPS = 30
FD_OURS = os.path.join(HERE, 'data', 'mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10')
MOT_OURS = 'mace_local2s_criticv5_s0.003_foot10_frz_gain'


def hit(j, A, W):
    """(hits within +-W frames of a beat, number of accents) over accents inside the beat range, and the chance share."""
    M = accents(velocity(j, '3d'))
    M = M[(M >= A[0]) & (M <= A[-1])]
    if len(M) < 5:
        return None
    d = np.abs(M[:, None] - A[None]).min(1)
    iv = np.median(np.diff(A))
    return {'rate': float((d <= W).mean()), 'chance': float(min(1.0, (2 * W + 1) / iv)), 'n': int(len(M))}


def finedance(W):
    names = json.load(open(os.path.join(R.ONESHOT, 'names.json')))['names']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(dev)
    one = np.load(os.path.join(R.ONESHOT, 'joints.npy'), mmap_mode='r')
    ours = np.load(os.path.join(FD_OURS, 'joints.npy'), mmap_mode='r')
    out = {}
    for i, n in enumerate(names):
        A = np.flatnonzero(np.load(os.path.join('data/test/baseline_feats', n + '.npy'))[:, 34])
        if len(A) < 20:
            continue
        m = pickle.load(open(os.path.join('data/test/motions_sliced', n + '.pkl'), 'rb'))
        gt = fk(smpl, [m['q']], [m['pos']], dev, from_pred=False)[0]
        T = min(len(gt), one.shape[2], ours.shape[2])
        A = A[A < T - 1]
        out[n] = {'bpm': float(60 * FPS / np.median(np.diff(A))), 'real': [hit(gt[:T], A, W)],
                  'one': [hit(np.asarray(j[:T], np.float32), A, W) for j in one[i]],
                  'ours': [hit(np.asarray(j[:T], np.float32), A, W) for j in ours[i]]}
    return out


def motorica(W):
    BS.MOT_MODELS.clear()
    BS.MOT_MODELS.update({'released': 'mace', 'ours': MOT_OURS})
    group, beats, sets, _ = BS.load_motorica()
    out = {}
    for k, (g, A) in enumerate(zip(group, beats)):
        A = np.asarray(A)
        if len(A) < 20:
            continue
        T = 1020
        out[f'{g}#{k}'] = {'bpm': float(60 * FPS / np.median(np.diff(A))),
                           'real': [hit(np.asarray(sets['real'][k][0])[:T], A, W)],
                           'one': [hit(np.asarray(j)[:T], A, W) for j in sets['released'][k]],
                           'ours': [hit(np.asarray(j)[:T], A, W) for j in sets['ours'][k]]}
    return out


def summarise(name, res):
    rows = []
    for n, r in res.items():
        f = lambda who: np.array([x['rate'] for x in r[who] if x])  # noqa: E731
        if min(len(f(w)) for w in ('real', 'one', 'ours')) == 0:
            continue
        ch = np.mean([x['chance'] for x in r['real'] if x])
        rows.append({'n': n, 'bpm': r['bpm'], 'real': float(f('real')[0]), 'one': float(np.median(f('one'))),
                     'ours': float(np.median(f('ours'))), 'chance': float(ch),
                     'gap': float(np.median(f('ours')) - np.median(f('one'))),
                     'gap_seeds_pos': float(np.mean(np.median(f('one')) < f('ours')))})
    a = lambda k: np.mean([x[k] for x in rows])  # noqa: E731
    print(f'\n=== {name}: {len(rows)} clips. hits on the beat out of 10 (mean over clips): real {10 * a("real"):.2f}  '
          f'released {10 * a("one"):.2f}  ours {10 * a("ours"):.2f}  | a random accent: {10 * a("chance"):.2f}')
    print(f'    clips where ours > released: {np.mean([x["gap"] > 0 for x in rows]) * 100:.0f} %, real > released: '
          f'{np.mean([x["real"] > x["one"] for x in rows]) * 100:.0f} %')
    print(f'{"clip":12s} {"bpm":>4s} | real  rel  ours  chance | gap(ours-rel)  seeds ours>rel')
    for x in sorted(rows, key=lambda x: -x['gap'])[:10]:
        print(f'{x["n"]:12s} {x["bpm"]:4.0f} | {10 * x["real"]:4.1f} {10 * x["one"]:4.1f} {10 * x["ours"]:4.1f}  {10 * x["chance"]:4.1f}  | '
              f'{10 * x["gap"]:+5.2f}  {x["gap_seeds_pos"] * 100:3.0f}%')
    json.dump(rows, open(os.path.join(HERE, 'results', f'rank_hits_on_beat_{name}.json'), 'w'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', type=int, default=2)
    a = ap.parse_args()
    summarise('finedance', finedance(a.W))
    summarise('motorica', motorica(a.W))
