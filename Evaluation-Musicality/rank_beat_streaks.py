"""Streaks of beats with a hit (2026-11-09): for each dancer, the longest run of consecutive beats that each have a motion accent
(peak of mean joint speed) within +-W frames (default 2, +-67 ms). A viewer can follow "hit, hit, hit ..." on the beat.
Per clip: real, released and ours (all seeds). Prints how often a streak of >= S beats happens (share of clip x seed), the mean
longest streak, and the clips where ours has the longest median streak, next to released and real on the same clip. Chance for
this is printed as the longest streak a random accent set of equal size gives (shuffled phase, 20 draws).
Needs rank_beat_runs.py / rank_hits_on_beat.py in the same folder. Writes results/rank_beat_streaks_<dataset>.json.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/rank_beat_streaks.py [--W 2] [--S 8]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import rank_beat_runs as RB  # noqa: E402
from tmd_beat_metrics import accents, velocity  # noqa: E402


def longest(hit):
    best = cur = 0
    start = bs = 0
    for k, h in enumerate(hit):
        if h:
            if cur == 0:
                start = k
            cur += 1
            if cur > best:
                best, bs = cur, start
        else:
            cur = 0
    return best, bs


def streak(j, A, W, rng=None):
    M = accents(velocity(j, '3d'))
    M = M[(M >= A[0] - W) & (M <= A[-1] + W)]
    if rng is not None:  # chance: same number of accents, uniform positions
        M = np.sort(rng.uniform(A[0], A[-1], len(M)))
    d = np.abs(A[:, None] - M[None]).min(1) if len(M) else np.full(len(A), 99)
    return longest(d <= W)


def run(name, W, S):
    rng = np.random.default_rng(0)
    rows = []
    for n, bpm, A, gt, one, ours in RB.clips(name):
        r = streak(gt, A, W)
        o = [streak(j, A, W) for j in one]
        u = [streak(j, A, W) for j in ours]
        ch = [streak(gt, A, W, rng)[0] for _ in range(20)]
        rows.append({'n': n, 'bpm': bpm, 'real': r, 'one': o, 'ours': u, 'chance': float(np.mean(ch)),
                     'chance_ge': float(np.mean(np.array(ch) >= S))})
    L = lambda k: np.array([x[0] for r in rows for x in r[k]] if k != 'real' else [r['real'][0] for r in rows])  # noqa: E731
    print(f'\n=== {name}: {len(rows)} clips. longest streak of consecutive beats with a hit within +-{W} frames')
    for lab, k in (('real', 'real'), ('released', 'one'), ('ours', 'ours')):
        v = L(k)
        print(f'  {lab:9s} mean longest {v.mean():.2f}   share with a streak >= {S}: {100 * np.mean(v >= S):4.1f} %   >= 12: {100 * np.mean(v >= 12):4.1f} %')
    print(f'  random accents (same count): mean longest {np.mean([r["chance"] for r in rows]):.2f}   share >= {S}: '
          f'{100 * np.mean([r["chance_ge"] for r in rows]):4.1f} %')
    key = lambda x: np.median([s[0] for s in x['ours']]) - np.median([s[0] for s in x['one']])  # noqa: E731
    print(f'  top clips by median-seed streak (ours - released)')
    print(f'  {"clip":58s} {"bpm":>4s} | longest streak: real  released(median)  ours(median)  ours best seed (beat0)')
    for x in sorted(rows, key=lambda x: -key(x))[:8]:
        ub = max(x['ours'], key=lambda s: s[0])
        print(f'  {x["n"]:58s} {x["bpm"]:4.0f} |   {x["real"][0]:2d}   {np.median([s[0] for s in x["one"]]):5.1f}   '
              f'{np.median([s[0] for s in x["ours"]]):5.1f}   {ub[0]} (beat {ub[1]})')
    json.dump(rows, open(os.path.join(HERE, 'results', f'rank_beat_streaks_{name}.json'), 'w'), default=float)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', type=int, default=2)
    ap.add_argument('--S', type=int, default=8)
    a = ap.parse_args()
    for name in ('finedance', 'motorica'):
        run(name, a.W, a.S)
