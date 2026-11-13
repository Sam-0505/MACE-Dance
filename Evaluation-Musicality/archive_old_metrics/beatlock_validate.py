"""Does the differentiable beat-locking measure of the draft loss (Expert-Motion/model/beatlock.py) show the same gap as
the evaluation's PLV, on clips as long as the training clips (8 s = 240 frames)? (2026-11-05)

FineDance test: the 84 slices of 34 s are cut into four non-overlapping 8 s windows. Per window: soft R (the loss's
measure, for several widths of the local-mean filter) and the hard PLV (beat_suite.py), against the window's own beats
and against the same window of 12 slices of other songs. Delta = own - other; unit = song (mean over its windows; MACE:
over its 10 seeds); sign-flip tests over the 19 songs. Also: across real windows, how well soft R tracks hard PLV.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/beatlock_validate.py
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from tmd_beat_metrics import accents, signflip, velocity  # noqa: E402
from tmd_beat_phase import phases, plv  # noqa: E402
from model.beatlock import locking, speed  # noqa: E402

WIN, N_NULL = 240, 12
WIDE = (5, 10, 20)
OUT = os.path.join(HERE, 'results', 'beatlock_validate.json')


def soft(joints, beats, wide):
    """joints (N, T, 24, 3) numpy, beats (N, T) one-hot -> R (N,)."""
    with torch.no_grad():
        v = speed(torch.from_numpy(np.ascontiguousarray(joints, dtype=np.float32)))
        return locking(v, torch.from_numpy(beats.astype(np.float32)), wide)[0].numpy()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=[], help='extra label=data_dir pairs (gen_mace_test.py outputs)')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    from motion_quality_extra import real_joints
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    song = np.array([n.split('_')[0] for n in names])
    songs = list(dict.fromkeys(song))
    onehot = np.stack([np.load(f'data/test/baseline_feats/{n}.npy')[:1020, 34] for n in names])  # (84, 1020)
    rng = np.random.default_rng(0)
    null = [rng.choice(np.flatnonzero(song != song[i]), N_NULL, replace=False) for i in range(len(names))]
    real = np.stack(real_joints(names))[:, None]  # (84, 1, 1020, 24, 3)
    mace = np.load(os.path.join(HERE, 'data', 'mace_test', 'joints.npy'), mmap_mode='r')
    sets = {'real': real, 'released': mace}
    for x in args.models:
        lab, d = x.split('=', 1)
        sets[lab] = np.load(os.path.join(HERE, 'data', d, 'joints.npy'), mmap_mode='r')
    res, tests, track = {}, {}, {}
    per_song = {}
    for label, J in sets.items():
        S = J.shape[1]
        own = {k: np.zeros((len(names), S, 4)) for k in [*map(str, WIDE), 'plv']}
        oth = {k: np.zeros((len(names), S, 4)) for k in own}
        for i in range(len(names)):
            Ji = np.asarray(J[i], dtype=np.float32)
            for q in range(4):
                lo = q * WIN
                jw = Ji[:, lo:lo + WIN]
                bw_own = np.repeat(onehot[i:i + 1, lo:lo + WIN], S, 0)
                for wd in WIDE:
                    own[str(wd)][i, :, q] = soft(jw, bw_own, wd)
                    oth[str(wd)][i, :, q] = np.mean([soft(jw, np.repeat(onehot[k:k + 1, lo:lo + WIN], S, 0), wd)
                                                     for k in null[i]], 0)
                for s in range(S):
                    m = accents(velocity(jw[s], '3d'))
                    own['plv'][i, s, q] = plv(phases(m, np.flatnonzero(onehot[i, lo:lo + WIN])))
                    oth['plv'][i, s, q] = np.nanmean([plv(phases(m, np.flatnonzero(onehot[k, lo:lo + WIN])))
                                                      for k in null[i]])
        res[label], per_song[label] = {}, {}
        for k in own:
            d = np.nanmean(own[k] - oth[k], (1, 2))  # per slice
            ds = np.array([d[song == g].mean() for g in songs])
            per_song[label][k] = ds
            res[label][k] = {'own': float(np.nanmean(own[k])), 'other': float(np.nanmean(oth[k])),
                             'delta': float(ds.mean()), 'p': signflip(ds, rng),
                             'songs_pos': f'{int((ds > 0).sum())}/{len(ds)}'}
        if label == 'real':
            for wd in WIDE:
                a, b = own[str(wd)].ravel(), own['plv'].ravel()
                ok = ~np.isnan(b)
                track[str(wd)] = float(np.corrcoef(a[ok], b[ok])[0, 1])
        print(label, json.dumps(res[label]), flush=True)
    for lab in [x for x in sets if x != 'real']:
        for k in per_song['real']:
            d = per_song[lab][k] - per_song['real'][k]
            tests[f'{lab} - real {k}'] = {'diff': float(d.mean()), 'p': signflip(d, rng),
                                          'songs_real_higher': int((d < 0).sum())}
    print('corr(soft R, hard PLV) over real windows', track)
    print(json.dumps(tests, indent=1))
    with open(OUT.replace('.json', f'{args.tag}.json'), 'w') as f:
        json.dump({'table': res, 'tests': tests, 'corr_soft_vs_plv_real': track}, f, indent=1)


if __name__ == '__main__':
    main()
