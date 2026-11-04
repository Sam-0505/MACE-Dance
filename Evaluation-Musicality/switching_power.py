"""Power check for switching_test.py (2026-11-04): would the test detect switching if it were there?

Planted data on the real FineDance music: per song, every SEG seconds the energy follows one randomly chosen
component (z-scored, smoothed), plus the real residual's noise (phase-randomised real energy) at coupling strength
r_plant. Runs the same near-far decay statistic and the same fixed-relationship surrogate null (fewer surrogates).
Also reports the real dance's whole-song fixed-fit r, so r_plant can be compared with what real dancers show.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/switching_power.py
    .venv/bin/python ../Evaluation-Musicality/switching_power.py --dataset motorica   # -> switching_power_motorica.json
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import switching_test as S  # noqa: E402

N_SUR = 10


def run(data, keep, seg_s, r_plant, rng):
    real, sur = [], []
    for s in keep:
        X = np.stack([gaussian_filter1d(data[s][c].astype(np.float64), S.SIGMA) for c in S.COMPS], 1)
        y = gaussian_filter1d(data[s]['energy'].astype(np.float64), S.SIGMA)
        T, seg = len(y), int(seg_s * S.FPS)
        sig = np.zeros(T)
        for a in range(0, T, seg):  # each segment: one component, z-scored within the segment
            k = rng.integers(len(S.COMPS))
            x = X[a:a + seg, k]
            sig[a:a + seg] = (x - x.mean()) / (x.std() + 1e-12)
        noise = S.phase_randomise(S.zs(y), rng)
        noise = noise / noise.std()
        ys = r_plant * sig + np.sqrt(1 - r_plant ** 2) * noise
        real.append(S.decay_curve(S.transfer(*S.windowed(X, ys)))[1])
        Xz, yz = S.zs(X), S.zs(ys)
        b = np.linalg.solve(Xz.T @ Xz + S.RIDGE * np.eye(len(S.COMPS)), Xz.T @ yz)
        fit = Xz @ b
        sur.append([S.decay_curve(S.transfer(*S.windowed(X, fit + S.phase_randomise(yz - fit, rng))))[1]
                    for _ in range(N_SUR)])
    agg = float(np.nanmean(real))
    agg_sur = np.nanmean(np.array(sur), 0)
    return agg, float(agg_sur.mean()), float((agg_sur >= agg).mean())


def main():
    data = S.build()
    keep = [s for s in data if len(data[s]['energy']) // S.WIN >= S.FAR + 2]
    rng = np.random.default_rng(0)
    # what coupling strength do real dancers show? whole-song fixed fit and 10 s window single-component max r
    fixed = []
    for s in keep:
        X = S.zs(np.stack([gaussian_filter1d(data[s][c].astype(np.float64), S.SIGMA) for c in S.COMPS], 1))
        y = S.zs(gaussian_filter1d(data[s]['energy'].astype(np.float64), S.SIGMA))
        b = np.linalg.solve(X.T @ X + S.RIDGE * np.eye(len(S.COMPS)), X.T @ y)
        fixed.append(np.corrcoef(X @ b, y)[0, 1])
    print(f'real dance whole-song fixed-fit r: median {np.median(fixed):.3f}, mean {np.mean(fixed):.3f}', flush=True)
    out = {'real_fixed_fit_r_median': float(np.median(fixed)), 'runs': []}
    for seg_s in (30, 60):
        for r in (0.2, 0.3, 0.5):
            agg, sur, p = run(data, keep, seg_s, r, rng)
            out['runs'].append({'segment_s': seg_s, 'r_plant': r, 'decay': agg, 'decay_surrogate': sur, 'p': p})
            print(f'segment {seg_s} s, r_plant {r}: decay {agg:+.3f} vs surrogate {sur:+.3f}, p {p:.2f}', flush=True)
    name = 'switching_power.json' if S.DATASET == 'finedance' else f'switching_power_{S.DATASET}.json'
    with open(os.path.join(HERE, 'results', name), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('finedance', 'motorica'), default='finedance')
    S.configure(ap.parse_args().dataset)
    main()
