"""Export the three final comparison dances as SMPL motion .pkl files for the Motion Projector / Appearance Expert
(2026-11-07): real dancer, MACE released, and ours (mask + guidance + foot term, root rotation frozen).

Examples (the clips of the *_real_released_final.mp4 videos):
  loudness : 012_slice3   perc : 161_slice1   beat : 161_slice3
Seed of each example = the median-scoring one-shot seed of the example's metric, exactly as render_fix_compare.py
picks it, so the motions are the ones shown in the skeleton videos. Both MACE variants are regenerated for all 84 test
slices in one batch (same seed -> same noise) and checked against the stored joints.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/export_smpl_final.py

Writes Evaluation-Musicality/data/appearance_final/<example>_<slice>_{real,released,ours}.pkl
(smpl_poses (T, 72) axis-angle, smpl_trans (T, 3), Z-up, 30 fps: the Motion Expert's own output format).
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import render_fix_compare as R  # noqa: E402
from dataset.quaternion import ax_from_6v  # noqa: E402
from eval_mace import fk  # noqa: E402
from export_smpl_examples import yup_to_zup  # noqa: E402
from model.metric import SMPLSkeleton  # noqa: E402

OURS_DIR = os.path.join(HERE, 'data', 'mace_test_mask2s_guide10_s0.003_foot10_frz')
OUT = os.path.join(HERE, 'data', 'appearance_final')
CLIPS = {'loudness': '012_slice3', 'perc': '161_slice1', 'beat': '161_slice3'}
MAKE = {'loudness': R.loudness_example, 'perc': R.perc_example, 'beat': R.beat_example}


def pick_seeds(names, smpl):
    seeds = {}
    for key, n in CLIPS.items():
        i = names.index(n)
        m = pickle.load(open(f'data/test/motions_sliced/{n}.pkl', 'rb'))
        gt = fk(smpl, [m['q']], [m['pos']], 'cuda', from_pred=False)[0]
        one, fix = R.load(names, R.ONESHOT, i), R.load(names, OURS_DIR, i)
        ex = MAKE[key](n, gt, one, fix)
        T = ex['T']
        s_one = [ex['score'](j) for j in one[:, :T]]
        seeds[key] = int(np.argsort(s_one)[len(s_one) // 2])
        print(f'{key} {n}: seed {seeds[key]} ({ex["metric"]} released {s_one[seeds[key]]:+.2f})', flush=True)
    return seeds


def main():
    from mace_sampler import MaceSampler
    from length_curve import local_cross_attention
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(R.ONESHOT, 'names.json')))
    genres, names = meta['genres'], meta['names']
    feats = np.stack([np.load(f'data/test/baseline_feats/{n}.npy') for n in names])
    smpl = SMPLSkeleton('cuda')
    seeds = pick_seeds(names, smpl)

    stored = {'released': os.path.join(R.ONESHOT, 'joints.npy'), 'ours': os.path.join(OURS_DIR, 'joints.npy')}
    for variant in ('released', 'ours'):
        sampler = MaceSampler()
        if variant == 'ours':
            local_cross_attention(sampler.model.diffusion.model, 60)
        keep = {}
        orig = sampler.to_joints

        def to_joints(x, keep=keep, orig=orig):
            keep['x'] = x.detach().cpu()
            return orig(x)
        sampler.to_joints = to_joints
        for seed in sorted(set(seeds.values())):
            if variant == 'released':
                J = sampler.sample(feats, genres, seed=seed)
            else:  # same settings as gen_mace_test.py --local_attn 2 --lock_guide 0.003 --lock_foot 10 --lock_freeze_root
                J = sampler.sample_guided(feats, genres, seed=seed, scale=0.003, t_max=600, root_relative=True,
                                          guide_shift=0, foot_iters=10, foot_mu=0.1, freeze_root_rot=True)
            for key, n in CLIPS.items():
                if seeds[key] != seed:
                    continue
                i = names.index(n)
                ref = np.asarray(np.load(stored[variant], mmap_mode='r')[i, seed], dtype=np.float32)
                err = float(np.abs(J[i] - ref).max())
                x = sampler.model.normalizer.unnormalize(keep['x'][i:i + 1])
                _, x = torch.split(x, (4, x.shape[2] - 4), dim=2)
                q = ax_from_6v(x[0, :, 3:].reshape(-1, 24, 6)).reshape(-1, 72).numpy()
                pos = x[0, :, :3].numpy()
                path = os.path.join(OUT, f'{key}_{n}_{variant}.pkl')
                pickle.dump({'smpl_poses': q, 'smpl_trans': pos, 'full_pose': J[i]}, open(path, 'wb'))
                print(f'{path}  seed {seed}  max |joints - stored| {err:.4f} m', flush=True)
                assert err < 0.02, 'regenerated sample differs from the stored one'
        del sampler
        torch.cuda.empty_cache()
    for key, n in CLIPS.items():
        m = pickle.load(open(f'data/test/motions_sliced/{n}.pkl', 'rb'))
        q, pos = yup_to_zup(m['q'], m['pos'])
        err = float(np.abs(fk(smpl, [q], [pos], 'cuda', from_pred=True)
                           - fk(smpl, [m['q']], [m['pos']], 'cuda', from_pred=False)).max())
        path = os.path.join(OUT, f'{key}_{n}_real.pkl')
        pickle.dump({'smpl_poses': q, 'smpl_trans': pos}, open(path, 'wb'))
        print(f'{path}  joints vs original GT {err:.2e} m', flush=True)
        assert err < 1e-3
    json.dump(seeds, open(os.path.join(OUT, 'seeds.json'), 'w'))


if __name__ == '__main__':
    main()
