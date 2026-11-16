"""Rotations of the ablation variants (seed 1 = seed index 0) for the FID_s / DIV_s genre metric (2026-11-15).

Regenerates each leave-one-out / ladder variant of the headline model with the exact flags of its stored run (same seed ->
same noise), checks every one of the 84 slices against the stored joints (max error < 2 cm), keeps the SMPL rotations and root
translation, converts them back to FineDance Y-up (prep_genre_inputs.zup_to_yup, the exact inverse of yup_to_zup) and writes
data/rot_export/ablation_seed0_yup.npz with <variant>_q (84, 1020, 72 axis-angle) and <variant>_pos (84, 1020, 3).
MACE released and the full model are already in genre_inputs_seed0.npz (export_rot_all.py), so they are not repeated.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/export_rot_ablation.py
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from critic import load_critic  # noqa: E402
from export_rot_all import rotations  # noqa: E402
from export_smpl_gain import make_sampler  # noqa: E402
from prep_genre_inputs import zup_to_yup  # noqa: E402

D = os.path.join(HERE, 'data')
FULL = dict(scale=0.003, t_max=600, root_relative=True, guide_shift=0, foot_iters=10, foot_mu=0.1, freeze_root_rot=True,
            grad_smooth=0.0, gain_ref=0.49)
VARIANTS = {   # name: (local attention 2 s, guided?, overrides of FULL, stored dir for the check)
    'mask': (True, False, {}, 'mace_test_local2s'),
    'nogain': (True, True, dict(gain_ref=0.0), 'mace_test_mask2s_criticv5_s0.003_foot10_frz'),
    'nomask': (False, True, {}, 'mace_test_abl_nomask'),
    'nocritic': (True, True, dict(scale=1e-9), 'mace_test_abl_nocritic'),
    'nofoot': (True, True, dict(foot_iters=0), 'mace_test_abl_nofoot'),
    'nofrz': (True, True, dict(freeze_root_rot=False), 'mace_test_abl_nofrz'),
}


def main():
    meta = json.load(open(os.path.join(D, 'mace_test', 'names.json')))
    genres, names = meta['genres'], meta['names']
    feats = np.stack([np.load(f'data/test/baseline_feats/{n}.npy') for n in names])
    out = {}
    for v, (local, guided, over, stored) in VARIANTS.items():
        sampler, keep = make_sampler(local)
        if guided:
            J = sampler.sample_guided(feats, genres, seed=0, critic=load_critic(),
                                      critic_groups=[n.split('_')[0] for n in names], **(FULL | over))
        else:
            J = sampler.sample(feats, genres, seed=0)
        ref = np.asarray(np.load(os.path.join(D, stored, 'joints.npy'), mmap_mode='r')[:, 0], dtype=np.float32)
        err = float(np.abs(J - ref).max())
        print(f'{v}: max |joints - stored {stored}| {err:.4f} m', flush=True)
        assert err < 0.02, f'{v}: regenerated samples differ from the stored run'
        q, tr = rotations(sampler, keep)
        qs, ps = zip(*[zup_to_yup(q[i], tr[i]) for i in range(len(names))])
        out[f'{v}_q'], out[f'{v}_pos'] = np.stack(qs), np.stack(ps)
        del sampler
        torch.cuda.empty_cache()
    np.savez(os.path.join(D, 'rot_export', 'ablation_seed0_yup.npz'), names=np.array(names), **out)
    print('wrote data/rot_export/ablation_seed0_yup.npz', {k: v.shape for k, v in out.items() if k.endswith('_q')})


if __name__ == '__main__':
    main()
