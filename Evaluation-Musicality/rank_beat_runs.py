"""Runs of steady beat hits (2026-11-09): stretches of N consecutive beats where the dancer's accents (peaks of mean joint speed)
keep the same place in the beat, i.e. a consistent step on the beat for a while. For each window of N beats (stride 2):
  R   = length of the mean phase vector of the accents in the window (1 = every accent at the same place of its beat)
  mu  = mean phase (0 = on the beat; +-0.1 beat ~ +-50 ms at 120 BPM)
  run = R >= 0.75 (chance for ~10 accents: p ~ 0.003);  on-beat run = run and |mu| <= 0.1.
Reported per dancer: share of windows that are runs / on-beat runs, so ours can be compared with released and real on how OFTEN
such a stretch happens, not only on the best one; then the best (clip, seed, window) for ours with the same window of released
(same seed) and real. Needs rank_hits_on_beat.py in the same folder. Writes results/rank_beat_runs_<dataset>.json.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/rank_beat_runs.py [--N 8]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import rank_hits_on_beat as H  # noqa: E402
from tmd_beat_metrics import accents, velocity  # noqa: E402
from tmd_beat_phase import phases  # noqa: E402

FPS = 30


def windows(j, A, N, stride=2):
    """list of (first beat index, n accents, R, mu) for windows of N beats, stride 2."""
    M = accents(velocity(j, '3d'))
    M = M[(M >= A[0]) & (M < A[-1])]
    ph = phases(M, A)
    M = M[:len(ph)] if len(ph) == len(M) else M[(M >= A[0]) & (M < A[-1])]
    beat_of = np.searchsorted(A, M, side='right') - 1
    out = []
    for k in range(0, len(A) - N, stride):
        s = (beat_of >= k) & (beat_of < k + N)
        if s.sum() < 6:
            out.append((k, int(s.sum()), np.nan, np.nan))
            continue
        z = np.exp(2j * np.pi * ph[s]).mean()
        out.append((k, int(s.sum()), float(abs(z)), float(np.angle(z) / (2 * np.pi))))
    return out


def shares(w, rmin=0.75, mu_max=0.1):
    R = np.array([x[2] for x in w])
    mu = np.array([x[3] for x in w])
    ok = ~np.isnan(R)
    if not ok.any():
        return np.nan, np.nan
    return float(np.mean(ok & (R >= rmin))), float(np.mean(ok & (R >= rmin) & (np.abs(mu) <= mu_max)))


def clips(name):
    """yield (clip, bpm, A, real_j, [released_j], [ours_j]) for the dataset."""
    import pickle
    import torch
    import render_fix_compare as R
    from eval_mace import fk
    from model.metric import SMPLSkeleton
    if name == 'finedance':
        names = json.load(open(os.path.join(R.ONESHOT, 'names.json')))['names']
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        smpl = SMPLSkeleton(dev)
        one = np.load(os.path.join(R.ONESHOT, 'joints.npy'), mmap_mode='r')
        ours = np.load(os.path.join(H.FD_OURS, 'joints.npy'), mmap_mode='r')
        for i, n in enumerate(names):
            A = np.flatnonzero(np.load(os.path.join('data/test/baseline_feats', n + '.npy'))[:, 34])
            if len(A) < 20:
                continue
            m = pickle.load(open(os.path.join('data/test/motions_sliced', n + '.pkl'), 'rb'))
            gt = fk(smpl, [m['q']], [m['pos']], dev, from_pred=False)[0]
            T = min(len(gt), one.shape[2], ours.shape[2])
            A = A[A < T - 1]
            yield (n, float(60 * FPS / np.median(np.diff(A))), A, gt[:T], [np.asarray(j[:T], np.float32) for j in one[i]],
                   [np.asarray(j[:T], np.float32) for j in ours[i]])
    else:
        H.BS.MOT_MODELS.clear()
        H.BS.MOT_MODELS.update({'released': 'mace', 'ours': H.MOT_OURS})
        group, beats, sets, _ = H.BS.load_motorica()
        for k, (g, A) in enumerate(zip(group, beats)):
            A = np.asarray(A)
            if len(A) < 20:
                continue
            yield (f'{g}#{k}', float(60 * FPS / np.median(np.diff(A))), A, np.asarray(sets['real'][k][0])[:1020],
                   [np.asarray(j)[:1020] for j in sets['released'][k]], [np.asarray(j)[:1020] for j in sets['ours'][k]])


def run(name, N):
    rows, best = [], []
    for n, bpm, A, gt, one, ours in clips(name):
        wr = windows(gt, A, N)
        wo = [windows(j, A, N) for j in one]
        wu = [windows(j, A, N) for j in ours]
        sr, so, su = shares(wr), np.array([shares(w) for w in wo]), np.array([shares(w) for w in wu])
        rows.append({'n': n, 'bpm': bpm, 'real': sr, 'one': np.nanmean(so, 0).tolist(), 'ours': np.nanmean(su, 0).tolist()})
        for si, w in enumerate(wu):  # candidate windows for ours: an on-beat run
            for q, (k, na, R, mu) in enumerate(w):
                if not np.isnan(R) and R >= 0.75 and abs(mu) <= 0.1:
                    ro = wo[min(si, len(wo) - 1)][q]
                    best.append({'n': n, 'seed': si, 'beat0': int(k), 't': float(A[k] / FPS), 'bpm': bpm, 'ours': (R, mu, na),
                                 'one_same_seed': ro[2:], 'real': wr[q][2:]})
    m = lambda key, c: np.nanmean([x[key][c] if isinstance(x[key], list) else x[key][c] for x in rows])  # noqa: E731
    print(f'\n=== {name}: {len(rows)} clips, windows of {N} beats. share of windows that are a steady run (R >= 0.75) / an on-beat run')
    print(f'  real {100 * m("real", 0):4.1f}% / {100 * m("real", 1):4.1f}%   released {100 * m("one", 0):4.1f}% / {100 * m("one", 1):4.1f}%   '
          f'ours {100 * m("ours", 0):4.1f}% / {100 * m("ours", 1):4.1f}%')
    # clips ranked by ours - released on-beat run share (median over seeds is in the mean here; seeds are averaged per clip)
    rk = sorted(rows, key=lambda x: -(x['ours'][1] - x['one'][1]))[:8]
    print(f'  {"clip":58s} {"bpm":>4s} | on-beat-run share: real  rel  ours')
    for x in rk:
        print(f'  {x["n"]:58s} {x["bpm"]:4.0f} |   {100 * x["real"][1]:4.0f}% {100 * x["one"][1]:4.0f}% {100 * x["ours"][1]:4.0f}%')
    # best single windows for ours: highest R with released and real in the same window
    best.sort(key=lambda b: -(b['ours'][0] - (0 if np.isnan(b['one_same_seed'][0]) else b['one_same_seed'][0])))
    print('  best on-beat windows for ours (R, mu, n accents) | released same seed/window | real same window')
    for b in best[:6]:
        f = lambda t: ' '.join('nan' if (v is None or (isinstance(v, float) and np.isnan(v))) else f'{v:+.2f}' for v in t)  # noqa: E731
        print(f'    {b["n"]:58s} seed {b["seed"]} {b["t"]:5.1f}s | {f(b["ours"])} | {f(b["one_same_seed"])} | {f(b["real"])}')
    json.dump({'rows': rows, 'best': best[:40]}, open(os.path.join(HERE, 'results', f'rank_beat_runs_{name}.json'), 'w'), default=float)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=8)
    a = ap.parse_args()
    for name in ('finedance', 'motorica'):
        run(name, a.N)
