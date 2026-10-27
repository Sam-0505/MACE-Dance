"""Song-level EAS for GT vs multiple Motion Expert samples per test clip.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/gen_mace_test.py --seeds 10
    .venv/bin/python ../Evaluation-Musicality/eas_multiseed.py

Mismatch baseline: GT uses 10 random other-song pairings; MACE sample seed s uses pairing seed s.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from model.metric import SMPLSkeleton
import audio_features as af
import metrics as M
from eval_mace import fk


def per_song(values, songs, order):
    values, songs = np.asarray(values), np.asarray(songs)
    return np.array([values[songs == s].mean() for s in order])


def paired(a, b):
    return {'a_mean': float(a.mean()), 'b_mean': float(b.mean()),
            'n_a_gt_b': int((a > b).sum()), 'n': len(a), 'wilcoxon_p': float(wilcoxon(a, b).pvalue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_dir', default='data/test')
    ap.add_argument('--mace_dir', default=os.path.join(HERE, 'data', 'mace_test'))
    ap.add_argument('--n_mismatch', type=int, default=10)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'eas_multiseed.json'))
    args = ap.parse_args()

    with open(os.path.join(args.mace_dir, 'names.json')) as f:
        names = json.load(f)['names']
    mace = np.load(os.path.join(args.mace_dir, 'joints.npy')).astype(np.float32)  # (N, S, T, 24, 3)
    N, S = mace.shape[:2]
    songs = [n.split('_')[0] for n in names]
    order = sorted(set(songs))

    musics, rmss, poses, trans = [], [], [], []
    for n in names:
        musics.append(np.load(os.path.join(args.gt_dir, 'baseline_feats', n + '.npy')))
        rmss.append(af.rms_envelope(af.load_audio(os.path.join(args.gt_dir, 'wavs_sliced', n + '.wav'))))
        with open(os.path.join(args.gt_dir, 'motions_sliced', n + '.pkl'), 'rb') as f:
            m = pickle.load(f)
        poses.append(m['q']); trans.append(m['pos'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gt = fk(SMPLSkeleton(device), poses, trans, device, from_pred=False)

    eas = lambda rows: np.array([r['EAS'] for r in rows])

    # GT: own + mean of n_mismatch other-song pairings
    gt_own = eas([M.clip_scores(m, r, j) for m, r, j in zip(musics, rmss, gt)])
    gt_other = np.mean([eas(M.mismatched_scores(musics, rmss, list(gt), songs, seed=k))
                        for k in range(args.n_mismatch)], axis=0)

    # MACE: own and other per sample seed
    own_rows, other = [], []
    for s in range(S):
        js = list(mace[:, s])
        own_rows.append([M.clip_scores(m, r, j) for m, r, j in zip(musics, rmss, js)])
        other.append(eas(M.mismatched_scores(musics, rmss, js, songs, seed=s)))
    mace_own_seed = np.stack([eas(r) for r in own_rows])          # (S, N)
    mace_own, mace_other = mace_own_seed.mean(0), np.mean(other, axis=0)

    seed_means = mace_own_seed.mean(1)
    bas0 = float(np.mean([r['BAS'] for r in own_rows[0]]))
    bas_all = [float(np.mean([r['BAS'] for r in rows])) for rows in own_rows]
    gt_bas = float(np.mean([M.bas(m[:, 34], j) for m, j in zip(musics, gt)]))

    gs_own, gs_other = per_song(gt_own, songs, order), per_song(gt_other, songs, order)
    ms_own, ms_other = per_song(mace_own, songs, order), per_song(mace_other, songs, order)
    # per-seed song-level own vs other for MACE (how often a single sample looks music-following)
    per_seed_song = []
    for s in range(S):
        a, b = per_song(mace_own_seed[s], songs, order), per_song(other[s], songs, order)
        per_seed_song.append(paired(a, b))

    out = {
        'n_clips': N, 'n_seeds': S, 'n_songs': len(order),
        'a_mace_eas_per_seed': seed_means.tolist(),
        'a_mace_eas_seed_mean_sd': [float(seed_means.mean()), float(seed_means.std())],
        'b_gt_own_vs_other': paired(gs_own, gs_other),
        'b_mace_own_vs_other': paired(ms_own, ms_other),
        'b_mace_own_vs_other_per_seed': per_seed_song,
        'c_gt_vs_mace': paired(gs_own, ms_own),
        'd_bas': {'mace_seed0': bas0, 'mace_per_seed': bas_all, 'gt': gt_bas},
        'per_song': {s: {'gt_own': float(a), 'gt_other': float(b), 'mace_own': float(c), 'mace_other': float(d)}
                     for s, a, b, c, d in zip(order, gs_own, gs_other, ms_own, ms_other)},
    }
    print(json.dumps({k: v for k, v in out.items() if k not in ('per_song', 'b_mace_own_vs_other_per_seed')},
                     indent=1))
    print('per-seed MACE own>other (songs):', [p['n_a_gt_b'] for p in per_seed_song],
          'p:', [round(p['wilcoxon_p'], 3) for p in per_seed_song])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', args.out)


if __name__ == '__main__':
    main()
