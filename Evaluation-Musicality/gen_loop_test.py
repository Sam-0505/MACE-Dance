"""Looped-song test (2026-11-09): does a model repeat its steps when the music repeats exactly?

A bar of BEATS beats of a test clip's baseline features (onset envelope, beat flags, ... all 35 columns) is tiled to the
clip length, so the music is exactly periodic with period P frames. MACE released (one shot, as data/mace_test) and ours
(mask 2 s + critic v5 + foot + frozen root + gain 0.49, as in export_smpl_gain.py) generate SEEDS samples each.
Repeat score per dancer: pose distance between frames t and t + lag (pelvis-relative 24 joints, RMS, metres), averaged over t,
as a function of lag. A dancer that repeats its steps has a dip at lag = P, 2P ...; the ratio D(P) / mean D(other lags in
0.5P..3.5P, not within 0.15P of a multiple of P) says how much (1 = no repeat, < 1 = repeat). The same is computed on the
speed profile (4-limb speed correlation at lag P).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/gen_loop_test.py --clips 161_slice2 036_slice2 130_slice2 --beats 8 --seeds 10

Writes Evaluation-Musicality/data/loop_test/<clip>_b<beats>_<model>.npy (seeds, T, 24, 3) and results/loop_test.json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
OUT = os.path.join(HERE, 'data', 'loop_test')
LIMBS = [7, 8, 20, 21]


def loop_feats(name, beats, start_beat):
    f = np.load(f'data/test/baseline_feats/{name}.npy')
    A = np.flatnonzero(f[:, 34])
    a, b = A[start_beat], A[start_beat + beats]
    P = int(b - a)
    seg = f[a:b]
    reps = int(np.ceil(len(f) / P)) + 1
    return np.tile(seg, (reps, 1))[:len(f)], P, a


def pose_dist(j, lag):
    r = j - j[:, :1]
    return float(np.sqrt(((r[lag:] - r[:-lag]) ** 2).sum(-1).mean(-1)).mean())


def repeat_scores(j, P):
    """j: (T, 24, 3). Returns D(P)/null, D(2P)/null, speed corr at P, and the whole D(lag) curve."""
    lags = np.arange(int(0.5 * P), int(3.5 * P))
    D = np.array([pose_dist(j, l) for l in lags])
    near = np.min(np.abs(lags[:, None] - P * np.arange(1, 4)[None]), 1) <= 0.15 * P
    null = D[~near].mean()
    sp = np.linalg.norm(np.gradient(j[:, LIMBS], axis=0), axis=-1).mean(1)
    sp = (sp - sp.mean()) / (sp.std() + 1e-8)
    return {'ratio_P': pose_dist(j, P) / null, 'ratio_2P': pose_dist(j, 2 * P) / null,
            'speed_corr_P': float(np.mean(sp[P:] * sp[:-P])), 'D': D.tolist(), 'null': float(null)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', nargs='+', default=['161_slice2', '036_slice2', '130_slice2'])
    ap.add_argument('--beats', type=int, default=8)
    ap.add_argument('--start_beat', type=int, default=8)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--models', nargs='+', default=['released', 'ours'])
    ap.add_argument('--analyse_only', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    feats, Ps = zip(*[loop_feats(n, a.beats, a.start_beat)[:2] for n in a.clips])
    tag = lambda n, m: os.path.join(OUT, f'{n}_b{a.beats}_{m}.npy')  # noqa: E731

    if not a.analyse_only:
        from critic import load_critic
        from length_curve import local_cross_attention
        from mace_sampler import MaceSampler
        genres = [json.load(open(f'data/test/label/{n.split("_")[0]}.json'))['style2'] for n in a.clips]
        F = np.stack(feats)
        for m in a.models:
            s = MaceSampler()
            if m == 'ours':
                local_cross_attention(s.model.diffusion.model, 60)
                run = lambda seed: s.sample_guided(F, genres, seed=seed, scale=0.003, t_max=600, root_relative=True,  # noqa: E731
                                                   guide_shift=0, foot_iters=10, foot_mu=0.1, freeze_root_rot=True,
                                                   gain_ref=0.49, critic=load_critic(),
                                                   critic_groups=[n.split('_')[0] for n in a.clips])
            else:
                run = lambda seed: s.sample(F, genres, seed=seed)  # noqa: E731
            out = np.stack([np.asarray(run(sd), np.float16) for sd in range(a.seeds)], 1)  # (clips, seeds, T, 24, 3)
            for c, n in enumerate(a.clips):
                np.save(tag(n, m), out[c])
            print(m, out.shape, flush=True)
            del s
            import torch
            torch.cuda.empty_cache()

    res = {}
    print(f'\nloop of {a.beats} beats; ratio = pose distance one period apart / other lags (1 = no repeat, < 1 = repeats)')
    print(f'{"clip":12s} {"P":>4s} {"model":9s} {"D(P)/null":>10s} {"D(2P)/null":>11s} {"speed corr P":>13s}  seeds with ratio<0.9')
    for n, P in zip(a.clips, Ps):
        for m in a.models:
            J = np.load(tag(n, m)).astype(np.float32)
            sc = [repeat_scores(j, P) for j in J]
            r = np.array([x['ratio_P'] for x in sc])
            res[f'{n}_{m}'] = {'P': P, 'ratio_P': r.tolist(), 'ratio_2P': [x['ratio_2P'] for x in sc],
                               'speed_corr_P': [x['speed_corr_P'] for x in sc]}
            print(f'{n:12s} {P:4d} {m:9s} {r.mean():10.3f} {np.mean([x["ratio_2P"] for x in sc]):11.3f} '
                  f'{np.mean([x["speed_corr_P"] for x in sc]):13.3f}  {(r < 0.9).sum()}/{len(r)}')
    json.dump(res, open(os.path.join(HERE, 'results', 'loop_test.json'), 'w'))


if __name__ == '__main__':
    main()
