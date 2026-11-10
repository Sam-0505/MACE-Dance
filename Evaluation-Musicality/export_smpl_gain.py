"""Export one test slice of the render_fix_compare.py videos as SMPL motion .pkl files for the Motion Projector / Appearance
Expert / pink mannequin renders (2026-11-08): the real dancer (real), MACE released (oneshot) and ours = critic v5 + gain
with mask, frozen root rotation and foot term (gain).

Defaults = the loudness_198_slice0 videos (seed 3). Lock example: --slice 161_slice2 --seed 6 --tag lock.

The MACE motions are the stored samples (mace_test, mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10), regenerated for all 84
test slices in one batch (same seed -> same noise) because the stored runs keep joints only, not the SMPL rotations, and
checked against the stored joints (max error < 2 cm). The settings of the gain10 run are not recorded anywhere, so the
candidates below are tried in order and the first that reproduces the stored joints is used (the check makes the guess safe).
The real motion is rotated to Z-up (export_smpl_examples.yup_to_zup) and checked against the original through eval_mace.fk.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/export_smpl_gain.py [--slice S --seed K --tag T --variants real oneshot gain]

Writes Evaluation-Musicality/data/appearance_examples/<tag>_<slice>_<variant>.pkl (smpl_poses (T, 72), smpl_trans (T, 3), Z-up).
"""
import argparse
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
from critic import load_critic  # noqa: E402
from eval_mace import fk  # noqa: E402
from export_smpl_examples import yup_to_zup  # noqa: E402
from model.metric import SMPLSkeleton  # noqa: E402

STORED = {'oneshot': os.path.join(HERE, 'data', 'mace_test', 'joints.npy'),
          'gain': os.path.join(HERE, 'data', 'mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10', 'joints.npy')}
OUT = os.path.join(HERE, 'data', 'appearance_examples')
CANDIDATES = [dict(grad_smooth=0.0, gain_ref=0.49), dict(grad_smooth=4.0, gain_ref=0.49)]


def save(path, sampler, keep, J, i):
    x = sampler.model.normalizer.unnormalize(keep['x'][i:i + 1])
    _, x = torch.split(x, (4, x.shape[2] - 4), dim=2)
    q = ax_from_6v(x[0, :, 3:].reshape(-1, 24, 6)).reshape(-1, 72).numpy()
    pickle.dump({'smpl_poses': q, 'smpl_trans': x[0, :, :3].numpy(), 'full_pose': J[i]}, open(path, 'wb'))
    print(f'wrote {path}', flush=True)


def make_sampler(local):
    from mace_sampler import MaceSampler
    from length_curve import local_cross_attention
    sampler = MaceSampler()
    if local:
        local_cross_attention(sampler.model.diffusion.model, 60)
    keep = {}
    orig = sampler.to_joints

    def to_joints(x):
        keep['x'] = x.detach().cpu()
        return orig(x)
    sampler.to_joints = to_joints
    return sampler, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slice', default='198_slice0')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--tag', default='loudness')
    ap.add_argument('--variants', nargs='+', default=['gain'], choices=['real', 'oneshot', 'gain'])
    a = ap.parse_args()
    meta = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))
    genres, names = meta['genres'], meta['names']
    feats = np.stack([np.load(f'data/test/baseline_feats/{n}.npy') for n in names])
    i = names.index(a.slice)
    path = lambda v: os.path.join(OUT, f'{a.tag}_{a.slice}_{v}.pkl')  # noqa: E731

    if 'real' in a.variants:
        m = pickle.load(open(f'data/test/motions_sliced/{a.slice}.pkl', 'rb'))
        q, pos = yup_to_zup(m['q'], m['pos'])
        smpl = SMPLSkeleton('cuda')
        err = float(np.abs(fk(smpl, [q], [pos], 'cuda', from_pred=True) - fk(smpl, [m['q']], [m['pos']], 'cuda', from_pred=False)).max())
        assert err < 1e-3, err
        pickle.dump({'smpl_poses': q, 'smpl_trans': pos}, open(path('real'), 'wb'))
        print(f'wrote {path("real")}  joints vs original GT {err:.2e} m', flush=True)

    if 'oneshot' in a.variants:
        sampler, keep = make_sampler(False)
        J = sampler.sample(feats, genres, seed=a.seed)
        ref = np.asarray(np.load(STORED['oneshot'], mmap_mode='r')[i, a.seed], dtype=np.float32)
        err = float(np.abs(J[i] - ref).max())
        print(f'oneshot: max |joints - stored| {err:.4f} m', flush=True)
        assert err < 0.02, 'regenerated MACE released sample differs from the stored one'
        save(path('oneshot'), sampler, keep, J, i)
        del sampler
        torch.cuda.empty_cache()

    if 'gain' in a.variants:
        ref = np.asarray(np.load(STORED['gain'], mmap_mode='r')[i, a.seed], dtype=np.float32)
        for cand in CANDIDATES:
            sampler, keep = make_sampler(True)
            J = sampler.sample_guided(feats, genres, seed=a.seed, scale=0.003, t_max=600, root_relative=True, guide_shift=0,
                                      foot_iters=10, foot_mu=0.1, freeze_root_rot=True, critic=load_critic(),
                                      critic_groups=[n.split('_')[0] for n in names], **cand)
            err = float(np.abs(J[i] - ref).max())
            print(f'gain {cand}: max |joints - stored| {err:.4f} m', flush=True)
            if err < 0.02:
                save(path('gain'), sampler, keep, J, i)
                break
            del sampler
            torch.cuda.empty_cache()
        else:
            raise SystemExit('no candidate reproduces the stored gain10 sample; settings need to be found')


if __name__ == '__main__':
    main()
