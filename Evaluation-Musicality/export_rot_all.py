"""Export SMPL rotations + root translation of MACE released ('oneshot') and ours (critic v5 + gain, 'gain') for all 84
FineDance test slices, one seed (2026-11-10). Input to genre_metric.py (MEGADance FID_s / DIV_s).

Same regeneration path as export_smpl_gain.py (same seed -> same noise as the stored runs), but all 84 slices are saved
and ALL of them are checked against the stored joints (max error < 2 cm), so the exported rotations are the ones behind
the stored joints / tables. Output is Z-up like the rest of our pipeline; genre_metric.py converts to FineDance Y-up.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/export_rot_all.py --seed 0 --variants oneshot gain

Writes Evaluation-Musicality/data/rot_export/<variant>_seed<k>.npz: poses (84, T, 72) axis-angle, trans (84, T, 3), names.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from critic import load_critic  # noqa: E402
from dataset.quaternion import ax_from_6v  # noqa: E402
from export_smpl_gain import CANDIDATES, STORED, make_sampler  # noqa: E402

OUT = os.path.join(HERE, 'data', 'rot_export')


def rotations(sampler, keep):
    x = sampler.model.normalizer.unnormalize(keep['x'])
    _, x = torch.split(x, (4, x.shape[2] - 4), dim=2)
    b, t = x.shape[:2]
    q = ax_from_6v(x[:, :, 3:].reshape(b, t, 24, 6)).reshape(b, t, 72).numpy()
    return q, x[:, :, :3].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variants', nargs='+', default=['oneshot', 'gain'], choices=['oneshot', 'gain'])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))
    genres, names = meta['genres'], meta['names']
    feats = np.stack([np.load(f'data/test/baseline_feats/{n}.npy') for n in names])

    def check(variant, J):
        ref = np.asarray(np.load(STORED[variant], mmap_mode='r')[:, a.seed], dtype=np.float32)
        return float(np.abs(J - ref).max()), float(np.abs(J - ref).mean())

    def write(variant, keep, err):
        q, tr = rotations(*keep)
        path = os.path.join(OUT, f'{variant}_seed{a.seed}.npz')
        np.savez(path, poses=q, trans=tr, names=np.array(names), genres=np.array(genres), max_err_m=err)
        print(f'wrote {path} poses {q.shape} trans {tr.shape} (max |joints - stored| {err:.4f} m)', flush=True)

    if 'oneshot' in a.variants:
        sampler, keep = make_sampler(False)
        J = sampler.sample(feats, genres, seed=a.seed)
        mx, mean = check('oneshot', J)
        print(f'oneshot: max {mx:.4f} m  mean {mean:.5f} m', flush=True)
        assert mx < 0.02, 'regenerated MACE released differs from the stored samples'
        write('oneshot', (sampler, keep), mx)
        del sampler
        torch.cuda.empty_cache()

    if 'gain' in a.variants:
        for cand in CANDIDATES:
            sampler, keep = make_sampler(True)
            J = sampler.sample_guided(feats, genres, seed=a.seed, scale=0.003, t_max=600, root_relative=True, guide_shift=0,
                                      foot_iters=10, foot_mu=0.1, freeze_root_rot=True, critic=load_critic(),
                                      critic_groups=[n.split('_')[0] for n in names], **cand)
            mx, mean = check('gain', J)
            print(f'gain {cand}: max {mx:.4f} m  mean {mean:.5f} m', flush=True)
            if mx < 0.02:
                write('gain', (sampler, keep), mx)
                break
            del sampler
            torch.cuda.empty_cache()
        else:
            raise SystemExit('no candidate reproduces the stored gain10 samples; settings need to be found')


if __name__ == '__main__':
    main()
