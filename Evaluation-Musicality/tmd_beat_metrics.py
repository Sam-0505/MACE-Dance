"""TMD-Bench beat metrics (Yang et al., ICML 2026, arXiv 2605.01809, Sec. 3.3) on MACE's 34 s test dances (2026-11-04).

The paper's definitions, applied to joints instead of 2D pose-estimator keypoints:
  V(t) = mean over keypoints of |P_{t+1} - P_t|, smoothed with a 1-D Gaussian; motion accents M = its local maxima.
  VBCS = 1/|M| sum_{m in M} exp(-min_a |m - a|^2 / (2 sigma^2))       (precision side: accents land near a beat)
  ABHS = 1/|A| sum_{a in A} 1[min_m |a - m| < tau]                    (coverage side: beats get an accent)
  CSD / HSD = their spread over clips. The paper gives no values for sigma, tau, the smoothing or the beat tracker.
Choices here: music beats A = the beat one-hot of the shipped test features (librosa beat_track, the beats the
authors' BAS uses); smoothing 5 frames and sigma = 3 frames (0.1 s), both as in the authors' BAS; tau = 3 frames.
Sensitivity: sigma = tau in {2, 3, 5} frames. The authors' BAS (velocity minima, averaged over music beats) is
recomputed with the same code for reference.

Two keypoint sets: 3D (24 SMPL joints, metres) and 2D (orthographic front view: the model's Z-up x/z plane, the
closest joint-level stand-in for the paper's video keypoints).

Chance: VBCS and ABHS are not zero for motion unrelated to the music (beats every ~0.7 s, accents every ~0.5 s).
So each clip is also scored against the beats of every slice of the OTHER 18 songs (cropped/tiled to length),
and Delta = own - other is the music-specific part, as for the coupling metric. Accent rate (per s) is reported
because more accents raise ABHS by themselves.

Position: the 34 s clips are split into four 8.5 s quarters (one-shot generation; does beat alignment fade?).
Units: per clip mean over the 10 seeds; per song mean over its slices; paired sign-flip tests over the 19 songs.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/tmd_beat_metrics.py
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

FPS, SMOOTH = 30, 5
SIGMAS = (2, 3, 5)  # frames; sigma = tau
MAIN = 3
QUARTERS = 4
MODELS = {'released': 'mace_test', 'mask_2s': 'mace_test_local2s', 'mask_4s': 'mace_test_local4s',
          'stitched': 'mace_test_stitched', 'lora_34s': 'mace_test_lora_l0'}
OUT = os.path.join(HERE, 'results', 'tmd_beat_metrics.json')


def velocity(j, view):
    """j (T, 24, 3) Z-up -> smoothed mean keypoint speed (T-1,)."""
    p = j if view == '3d' else j[..., [0, 2]]
    return gaussian_filter1d(np.linalg.norm(np.diff(p, axis=0), axis=-1).mean(1), SMOOTH)


def accents(v):
    return argrelextrema(v, np.greater)[0]


def kin_beats(v):
    return argrelextrema(v, np.less)[0]


def vbcs(M, A, s):
    if len(M) == 0 or len(A) == 0:
        return np.nan
    d = np.abs(M[:, None] - A[None]).min(1)
    return float(np.exp(-d ** 2 / (2 * s ** 2)).mean())


def abhs(M, A, tau):
    if len(M) == 0 or len(A) == 0:
        return np.nan
    return float((np.abs(A[:, None] - M[None]).min(1) < tau).mean())


def bas(K, A):
    """The authors' BA (model/metric.py): music beats, kinematic beats = velocity minima, sigma^2 = 9."""
    if len(A) == 0 or len(K) == 0:
        return np.nan
    return float(np.exp(-np.abs(A[:, None] - K[None]).min(1) ** 2 / 18).mean())


def in_range(x, lo, hi):
    return x[(x >= lo) & (x < hi)]


def clip_scores(v, A, others):
    """v velocity, A own beats, others: list of beat arrays of other-song slices. Returns dict of scores."""
    M, K = accents(v), kin_beats(v)
    T = len(v)
    r = {'rate': len(M) / (T / FPS), 'BAS': bas(K, A), 'BAS_other': np.nanmean([bas(K, o) for o in others])}
    for s in SIGMAS:
        r[f'VBCS_{s}'] = vbcs(M, A, s)
        r[f'ABHS_{s}'] = abhs(M, A, s)
        r[f'VBCS_{s}_other'] = np.nanmean([vbcs(M, o, s) for o in others])
        r[f'ABHS_{s}_other'] = np.nanmean([abhs(M, o, s) for o in others])
    q = T // QUARTERS
    for k in range(QUARTERS):
        lo, hi = k * q, (k + 1) * q
        Mk, Ak = in_range(M, lo, hi), in_range(A, lo, hi)
        r[f'VBCS_q{k}'] = vbcs(Mk, A, MAIN)  # accents in the quarter, nearest beat anywhere
        r[f'ABHS_q{k}'] = abhs(M, Ak, MAIN)
        r[f'VBCS_q{k}_other'] = np.nanmean([vbcs(Mk, o, MAIN) for o in others])
        r[f'ABHS_q{k}_other'] = np.nanmean([abhs(M, in_range(o, lo, hi), MAIN) for o in others])
    return r


def signflip(x, rng):
    x = x[~np.isnan(x)]
    s = rng.choice([-1, 1], (20000, len(x)))
    return float((np.abs((s * x).mean(1)) >= abs(x.mean()) - 1e-12).mean())


def main():
    from motion_quality_extra import real_joints
    rng = np.random.default_rng(0)
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    song = np.array([n.split('_')[0] for n in names])
    songs = np.unique(song)
    beats = [np.flatnonzero(np.load(f'data/test/baseline_feats/{n}.npy')[:, 34]) for n in names]
    others = [[beats[k] for k in range(len(names)) if song[k] != song[i]] for i in range(len(names))]
    per_song = lambda x: np.array([np.nanmean(x[song == s]) for s in songs])  # noqa: E731
    real = real_joints(names)
    table, songwise = {}, {}

    def run(label, clips):  # clips: per slice, list of (T, 24, 3) samples
        out = {}
        for view in ('3d', '2d'):
            rows = []
            for i, js in enumerate(clips):
                rs = [clip_scores(velocity(j, view), beats[i], others[i]) for j in js]
                rows.append({k: np.nanmean([r[k] for r in rs]) for k in rs[0]})
            keys = list(rows[0])
            sw = {k: per_song(np.array([r[k] for r in rows])) for k in keys}
            for k in list(keys):
                if k + '_other' in sw:
                    sw[k + '_delta'] = sw[k] - sw[k + '_other']
            songwise[(label, view)] = sw
            out[view] = {k: float(np.nanmean(v)) for k, v in sw.items()}
        table[label] = out
        print(label, {k: round(out['3d'][k], 3) for k in ('rate', 'BAS', f'VBCS_{MAIN}', f'VBCS_{MAIN}_delta',
                                                            f'ABHS_{MAIN}', f'ABHS_{MAIN}_delta')}, flush=True)

    run('real', [[j] for j in real])
    for label, d in MODELS.items():
        J = np.load(os.path.join(HERE, 'data', d, 'joints.npy'), mmap_mode='r')
        run(label, [np.asarray(J[i], dtype=np.float32) for i in range(len(names))])
    tests = {}
    for view in ('3d', '2d'):
        for a, b in [(m, 'real') for m in MODELS] + [('mask_2s', 'released'), ('mask_4s', 'released'),
                                                      ('mask_2s', 'stitched')]:
            A, B = songwise[(a, view)], songwise[(b, view)]
            tests[f'{view} {a} - {b}'] = {k: {'diff': float(np.nanmean(A[k] - B[k])), 'p': signflip(A[k] - B[k], rng)}
                                          for k in A if not k.endswith('_other')}
    for label in table:  # music-specific part > 0?
        for view in ('3d', '2d'):
            sw = songwise[(label, view)]
            tests[f'{view} {label} delta>0'] = {k: {'mean': float(np.nanmean(sw[k])), 'p': signflip(sw[k], rng)}
                                                for k in sw if k.endswith('_delta')}
    # CSD / HSD as in the paper: spread over clips of VBCS / ABHS (reported as SD over songs here)
    for label in table:
        for view in ('3d', '2d'):
            sw = songwise[(label, view)]
            table[label][view]['CSD'] = float(np.nanstd(sw[f'VBCS_{MAIN}']))
            table[label][view]['HSD'] = float(np.nanstd(sw[f'ABHS_{MAIN}']))
    with open(OUT, 'w') as f:
        json.dump({'settings': {'sigma_tau_frames': SIGMAS, 'main': MAIN, 'smooth': SMOOTH, 'fps': FPS},
                   'table': table, 'tests': tests}, f, indent=1)
    for view in ('3d', '2d'):
        print(f'\n[{view}]  sigma = tau = {MAIN} frames')
        cols = ['rate', 'BAS', 'BAS_delta', f'VBCS_{MAIN}', f'VBCS_{MAIN}_delta', f'ABHS_{MAIN}', f'ABHS_{MAIN}_delta',
                'CSD', 'HSD']
        print(f'{"":10s}' + ''.join(f'{c:>12s}' for c in cols))
        for label in table:
            print(f'{label:10s}' + ''.join(f'{table[label][view].get(c, np.nan):12.3f}' for c in cols))
        print('quarters (VBCS delta / ABHS delta):')
        for label in table:
            t = table[label][view]
            print(f'{label:10s}' + ''.join(f"  q{k} {t[f'VBCS_q{k}_delta']:+.3f}/{t[f'ABHS_q{k}_delta']:+.3f}"
                                           for k in range(QUARTERS)))
        for s in SIGMAS:
            print(f'sigma {s}: ' + '  '.join(f"{lab} V {table[lab][view][f'VBCS_{s}']:.3f} ({table[lab][view][f'VBCS_{s}_delta']:+.3f})"
                                            f" H {table[lab][view][f'ABHS_{s}']:.3f} ({table[lab][view][f'ABHS_{s}_delta']:+.3f})"
                                            for lab in table))
    key = [f'VBCS_{MAIN}', f'VBCS_{MAIN}_delta', f'ABHS_{MAIN}', f'ABHS_{MAIN}_delta', 'rate', 'BAS']
    for t, v in tests.items():
        if 'delta>0' in t:
            print(t, '  '.join(f"{k} {v[k]['mean']:+.3f} (p {v[k]['p']:.3f})" for k in v if MAIN == 3 and '_3_' in k or k == 'BAS_delta'))
        else:
            print(t, '  '.join(f"{k} {v[k]['diff']:+.3f} (p {v[k]['p']:.3f})" for k in key if k in v))


if __name__ == '__main__':
    main()
