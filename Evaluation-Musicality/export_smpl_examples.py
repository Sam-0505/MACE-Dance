"""Export the render_fix_compare.py examples as SMPL motion .pkl files for the Motion Projector / Appearance Expert
(2026-11-04).

For each example slice: the real dance, the one-shot MACE sample and the +-2 s mask sample, both MACE samples with the
seed shown in the videos. MACE samples are regenerated exactly as gen_mace_test.py made them (all 84 test slices in
one batch, same seed, so the same noise) and checked against the stored joints. Format = the Motion Expert's own
output (diffusion.render_sample): smpl_poses (T, 72) axis-angle and smpl_trans (T, 3), Z-up. The real test motion is
Y-up (eval_mace.fk uses rotate_pred_like_gt to map predictions onto it), so it is rotated +90 deg about X, the inverse
of rotate_pred_like_gt; the check is that eval_mace.fk gives the same joints for the converted motion read as a
prediction (from_pred=True) as for the original read as ground truth.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/export_smpl_examples.py

Writes Evaluation-Musicality/data/appearance_examples/<example>_<slice>_{real,oneshot,fixed}.pkl.
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
from dataset.quaternion import ax_from_6v  # noqa: E402
from model.metric import SMPLSkeleton  # noqa: E402
from eval_mace import fk  # noqa: E402
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion, quaternion_multiply,  # noqa: E402
                                  quaternion_to_axis_angle)

EXAMPLES = {'loudness': ('012_slice3', 9), 'perc': ('198_slice0', 3)}  # slice, seed shown in the videos
OUT = os.path.join(HERE, 'data', 'appearance_examples')


def yup_to_zup(q, pos):
    """Inverse of rotate_pred_like_gt. q (T, 72), pos (T, 3) -> same, Z-up."""
    q = torch.as_tensor(q, dtype=torch.float32).reshape(1, -1, 24, 3).clone()
    pos = torch.as_tensor(pos, dtype=torch.float32).reshape(1, -1, 3)
    rot = torch.tensor([0.7071068, 0.7071068, 0, 0]).unsqueeze(0)
    q[:, :, :1] = quaternion_to_axis_angle(quaternion_multiply(rot, axis_angle_to_quaternion(q[:, :, :1])))
    pos = RotateAxisAngle(90, axis='X', degrees=True).transform_points(pos)
    return q[0].reshape(-1, 72).numpy(), pos[0].numpy()


def main():
    from mace_sampler import MaceSampler
    from length_curve import local_cross_attention
    os.makedirs(OUT, exist_ok=True)
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))
    genres, names = names['genres'], names['names']
    feats = np.stack([np.load(f'data/test/baseline_feats/{n}.npy') for n in names])
    stored = {'oneshot': os.path.join(HERE, 'data', 'mace_test', 'joints.npy'),
              'fixed': os.path.join(HERE, 'data', 'mace_test_local2s', 'joints.npy')}
    for variant in ('oneshot', 'fixed'):
        sampler = MaceSampler()
        if variant == 'fixed':
            local_cross_attention(sampler.model.diffusion.model, 60)
        keep = {}
        orig = sampler.to_joints

        def to_joints(x, keep=keep, orig=orig):
            keep['x'] = x.detach().cpu()
            return orig(x)
        sampler.to_joints = to_joints
        for key, (n, seed) in EXAMPLES.items():
            i = names.index(n)
            j = sampler.sample(feats, genres, seed=seed)[i]
            ref = np.asarray(np.load(stored[variant], mmap_mode='r')[i, seed], dtype=np.float32)
            err = float(np.abs(j - ref).max())
            x = sampler.model.normalizer.unnormalize(keep['x'][i:i + 1])
            _, x = torch.split(x, (4, x.shape[2] - 4), dim=2)
            q = ax_from_6v(x[0, :, 3:].reshape(-1, 24, 6)).reshape(-1, 72).numpy()
            pos = x[0, :, :3].numpy()
            path = os.path.join(OUT, f'{key}_{n}_{variant}.pkl')
            pickle.dump({'smpl_poses': q, 'smpl_trans': pos, 'full_pose': j}, open(path, 'wb'))
            print(f'{path}  max |joints - stored| {err:.4f} m', flush=True)
            assert err < 0.02, 'regenerated sample differs from the one in the videos'
        del sampler
        torch.cuda.empty_cache()
    for key, (n, _) in EXAMPLES.items():
        m = pickle.load(open(f'data/test/motions_sliced/{n}.pkl', 'rb'))
        q, pos = yup_to_zup(m['q'], m['pos'])
        smpl = SMPLSkeleton('cuda')
        err = float(np.abs(fk(smpl, [q], [pos], 'cuda', from_pred=True)
                           - fk(smpl, [m['q']], [m['pos']], 'cuda', from_pred=False)).max())
        path = os.path.join(OUT, f'{key}_{n}_real.pkl')
        pickle.dump({'smpl_poses': q, 'smpl_trans': pos}, open(path, 'wb'))
        print(f'{path}  joints vs original GT {err:.2e} m', flush=True)
        assert err < 1e-3


if __name__ == '__main__':
    main()
