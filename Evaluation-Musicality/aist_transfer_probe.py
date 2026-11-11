"""Why does the AIST++-trained MACE make hyperactive dances on Motorica music? (2026-11-11)

1. Feature shift: per-group statistics of the 35-dim baseline features (env 0, MFCC 1-20, chroma 21-32, onset peaks 33, beat 34) of AIST++
   training windows vs Motorica windows.
2. Group ablation: sample 5 s windows (150 frames, one window per take) with the AIST++ checkpoint, replacing ONE feature group of the
   Motorica clips by (a) the AIST++ per-dimension mean, (b) the group of a random AIST++ training window; plus (c) all groups moment-matched
   to the AIST++ statistics. Speed = mean joint speed, root-relative speed (m/s); real AIST++ median root-relative speed 0.77.
    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/aist_transfer_probe.py [--ckpt runs/aist/mace/weights/train-100.pt]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from mace_sampler import MaceSampler  # noqa: E402
import audio_features as af  # noqa: E402
from motorica_common import STYLE_TO_MACE, pool, slice_starts, style, wav_path  # noqa: E402

G = {'env': [0], 'mfcc': list(range(1, 21)), 'chroma': list(range(21, 33)), 'peak': [33], 'beat': [34]}


def speeds(j):
    sp = np.linalg.norm(j[:, 1:] - j[:, :-1], axis=-1).mean(-1) * 30
    rel = j - j[:, :, :1]
    rs = np.linalg.norm(rel[:, 1:] - rel[:, :-1], axis=-1).mean(-1) * 30
    return float(sp.mean()), float(rs.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/aist/mace/weights/train-100.pt')
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'aist_transfer_probe.json'))
    a = ap.parse_args()
    D = os.path.join(HERE, '..', 'Expert-Motion', 'data', 'aist', 'train', 'baseline_feats')
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(f'{D}/*.npy'))
    A = np.stack([np.load(f) for f in rng.choice(files, 1500, replace=False)])           # (1500, 150, 35)
    takes = pool()[:a.n]
    M, st = [], []
    for t in takes:
        y = af.load_audio(wav_path(t))
        o = slice_starts(t)[0]
        M.append(af.extract_baseline(y[o * af.HOP_LENGTH:(o + 151) * af.HOP_LENGTH])[:150])
        st.append(STYLE_TO_MACE[style(t)])
    M = np.stack(M)
    out = {'feature_stats': {}, 'speed': {}}
    print('group   AIST mean±sd (over windows of the per-window mean) | Motorica | std over time within a window (AIST / Motorica)')
    for g, ix in G.items():
        am, mm = A[:, :, ix].mean((1, 2)), M[:, :, ix].mean((1, 2))
        asd, msd = A[:, :, ix].std(1).mean(), M[:, :, ix].std(1).mean()
        out['feature_stats'][g] = {'aist_mean': float(am.mean()), 'aist_sd': float(am.std()), 'mot_mean': float(mm.mean()), 'mot_sd': float(mm.std()),
                                   'aist_within_sd': float(asd), 'mot_within_sd': float(msd),
                                   'mot_outside_aist_p5_p95': float(np.mean((mm < np.percentile(am, 5)) | (mm > np.percentile(am, 95))))}
        print(f'{g:7s} {am.mean():8.3f}±{am.std():.3f} | {mm.mean():8.3f}±{mm.std():.3f} | {asd:.3f} / {msd:.3f}   '
              f'(Motorica windows outside AIST 5-95%: {out["feature_stats"][g]["mot_outside_aist_p5_p95"]:.0%})', flush=True)
    sm = MaceSampler(a.ckpt)
    sty = list(st)

    def run(name, feats, styles):
        j = sm.sample(feats, styles, seed=0)
        s, r = speeds(j)
        out['speed'][name] = {'speed': s, 'root_rel': r}
        print(f'{name:38s} speed {s:5.2f} root-rel {r:5.2f}', flush=True)

    ai = rng.choice(len(A), len(M), replace=False)
    from aist_common import GENRE_TO_STYLE
    run('AIST++ windows (reference)', A[ai], [GENRE_TO_STYLE[os.path.basename(files[0])[:3]]] * len(ai))
    run('Motorica as is', M, sty)
    for g, ix in G.items():
        X = M.copy()
        X[:, :, ix] = A[:, :, ix].mean((0, 1))
        run(f'Motorica, {g} -> AIST mean', X, sty)
    for g, ix in G.items():
        X = M.copy()
        X[:, :, ix] = A[ai][:, :, ix]
        run(f'Motorica, {g} -> random AIST window', X, sty)
    Mn = (M - M.mean((0, 1))) / (M.std((0, 1)) + 1e-6) * A.std((0, 1)) + A.mean((0, 1))
    run('Motorica, all moment-matched to AIST', Mn, sty)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
