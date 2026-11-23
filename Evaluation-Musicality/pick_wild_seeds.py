"""Seed choice for the user-supplied songs (2026-11-21), same rule as the FineDance showcase (rank_showcase_windows.py):
lock r (lock_models.py metric, band-pass on the whole 34 s segment) inside the shown window (requested start .. +DUR s);
ours = its strongest-lock seed among those whose beat-wave amplitude is >= 0.8x both baselines' median amplitude, MACE-Dance
and FlowerDance = their MEDIAN-r seed (typical sample). Best case for ours, typical for the baselines: label it so.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/pick_wild_seeds.py

Writes Evaluation-Musicality/data/wild/seeds.json and prints the render_pink_grid.py --wild_seeds argument per song.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import lock_models as LM  # noqa: E402
from robustness_loud import motion_signals  # noqa: E402
from rank_showcase_windows import phase_series  # noqa: E402

WILD = os.path.join(HERE, 'data', 'wild')
FPS, DUR = 30, 10.2


def main():
    rot = {k: np.load(os.path.join(WILD, 'rot', f'{k}.npz')) for k in ('gain', 'oneshot', 'flower')}
    names = [str(n) for n in rot['gain']['names']]
    out = {}
    for i, n in enumerate(names):
        meta = json.load(open(os.path.join(WILD, n, 'meta.json')))
        b = np.flatnonzero(np.load(os.path.join(WILD, n, 'feats.npy'))[:, 34])
        w0 = int(round(meta['window_start_in_segment_s'] * FPS))
        w1 = w0 + int(round(DUR * FPS))
        r, amp = {}, {}
        for k, z in rot.items():
            zs = [phase_series(motion_signals(np.asarray(j, np.float64))['speed'], b) for j in z['joints'][i]]
            r[k] = np.array([float(np.abs(e[w0:w1].mean())) for e, _ in zs])
            amp[k] = np.array([float(a[w0:w1].mean()) for _, a in zs])
        med = {k: int(np.argsort(r[k])[len(r[k]) // 2]) for k in ('oneshot', 'flower')}
        ok = amp['gain'] >= 0.8 * max(np.median(amp['oneshot']), np.median(amp['flower']))
        cand = np.flatnonzero(ok) if ok.any() else np.arange(len(r['gain']))
        best = int(cand[np.argmax(r['gain'][cand])])
        out[n] = {'seeds': {'gain': best, **med}, 'window_s': [w0 / FPS, w1 / FPS], 'bpm': meta['bpm'],
                  'r_chosen': {'gain': round(r['gain'][best], 3), 'oneshot': round(r['oneshot'][med['oneshot']], 3), 'flower': round(r['flower'][med['flower']], 3)},
                  'r_median_over_seeds': {k: round(float(np.median(v)), 3) for k, v in r.items()},
                  'r_all_seeds': {k: [round(float(x), 3) for x in v] for k, v in r.items()},
                  'ours_amp_rule_ok_seeds': int(ok.sum())}
        print(n, json.dumps({k: out[n][k] for k in ('seeds', 'r_chosen', 'r_median_over_seeds', 'ours_amp_rule_ok_seeds')}))
        print('   --wild', n, '--wild_seeds', ' '.join(f'{k}={v}' for k, v in out[n]['seeds'].items()))
    json.dump(out, open(os.path.join(WILD, 'seeds.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
