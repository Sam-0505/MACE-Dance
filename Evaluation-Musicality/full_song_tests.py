"""Paired per-song tests on full_song_curve.py results (2026-11-04).

For every window size and start-time bin: real - model for each model, and the +-2 s mask against the released,
34 s LoRA and stitched models. Songs (takes) are the units; only songs present in the bin for both sides enter.
Sign-flip p (20,000) and bootstrap 95 % CI (4,000).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/full_song_tests.py --dataset motorica
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MASK = 'released_local2s'


def test(x, rng):
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return None
    s = rng.choice([-1, 1], (20000, len(x)))
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
    return {'diff': float(x.mean()), 'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            'p': float((np.abs((s * x).mean(1)) >= abs(x.mean())).mean()), 'n': int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('finedance', 'motorica'), default='finedance')
    a = ap.parse_args()
    suffix = '' if a.dataset == 'finedance' else f'_{a.dataset}'
    res = json.load(open(os.path.join(HERE, 'results', f'full_song_curve{suffix}.json')))
    rng = np.random.default_rng(0)
    models = [m for m in res if m != 'real']
    out = {}
    for wn in res['real']:
        out[wn] = {}
        print(f'\n=== window {wn}')
        for b in res['real'][wn]:
            P = lambda m: np.array(res[m][wn][b]['per_song'], float)  # noqa: E731
            pairs = [(f'real - {m}', P('real') - P(m)) for m in models]
            pairs += [(f'{MASK} - {m}', P(MASK) - P(m)) for m in models if m != MASK and MASK in res]
            out[wn][b] = {'real': res['real'][wn][b]['mean'], 'n_real': res['real'][wn][b]['n_songs'],
                          'models': {m: res[m][wn][b]['mean'] for m in models},
                          'tests': {name: test(x, rng) for name, x in pairs}}
            r = out[wn][b]
            print(f"  {b:10s} (n {r['n_real']:2d})  real {r['real']:+.3f}  " +
                  '  '.join(f'{m} {v:+.3f}' for m, v in r['models'].items() if v is not None))
            for name, t in r['tests'].items():
                if t:
                    print(f"      {name:38s} {t['diff']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] p {t['p']:.3f}")
    with open(os.path.join(HERE, 'results', f'full_song_tests{suffix}.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
