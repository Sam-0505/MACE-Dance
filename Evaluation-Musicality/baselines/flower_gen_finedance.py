"""FlowerDance (ECCV'26, XulongT/FlowerDance, FineDance checkpoint train-3700.pt) on the 84 FineDance test slices (2026-11-12).

Uses the repo's own inference.py pieces (load_model, normalize_features, GENRES; 21 Euler MeanFlow steps = its default and
what test_loop uses) and mirrors flow_matching.render_sample (unnormalize -> 6D -> axis-angle -> SMPLSkeleton FK, Z-up like
MACE, since both build on EDGE). Music: the SHIPPED baseline features of the test slices (the same files our MACE runs use),
normalized with FlowerDance's embedded training bounds. Genre: the slice's FineDance style2 label. One sample per slice
per seed; seed s uses torch.manual_seed(s) at the start of each chunk (84 slices = 2 chunks of 42).

    cd /home/sam0505/Documents/Code/dance_baselines/FlowerDance
    WANDB_MODE=offline /home/sam0505/Documents/Code/MACE-Dance/Expert-Motion/.venv/bin/python \
        /home/sam0505/Documents/Code/MACE-Dance/Evaluation-Musicality/baselines/flower_gen_finedance.py --seeds 1

Writes Evaluation-Musicality/data/mace_test_flower/joints.npy (84, S, 1020, 24, 3) float16 (+ names.json) and
data/rot_export/flower_seed<s>.npz: poses (84, 1020, 72) axis-angle, trans (84, 1020, 3), Z-up (convert with
prep_genre_inputs.zup_to_yup for the genre classifier).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

FLOWER = '/home/sam0505/Documents/Code/dance_baselines/FlowerDance'
EVAL = '/home/sam0505/Documents/Code/MACE-Dance/Evaluation-Musicality'
FEATS = '/home/sam0505/Documents/Code/MACE-Dance/Expert-Motion/data/test/baseline_feats'
os.chdir(FLOWER)
sys.path.insert(0, FLOWER)
import inference as inf  # noqa: E402

CHUNK, STEPS = 42, 21


@torch.no_grad()
def sample(model, feats_norm, genre_ids, seed):
    from dataset.quaternion import ax_from_6v
    dev = model.accelerator.device
    cond = torch.from_numpy(feats_norm).to(dev)
    genre = torch.tensor(genre_ids, dtype=torch.long, device=dev)
    torch.manual_seed(seed)
    x = model.flow_matching.euler_sample((len(cond), cond.shape[1], inf.MOTION_DIM), cond, genre, n_steps=STEPS).cpu()
    x = model.normalizer.unnormalize(x)
    _, x = torch.split(x, (4, x.shape[2] - 4), dim=2)
    pos = x[:, :, :3].to(dev)
    q = ax_from_6v(x[:, :, 3:].reshape(len(cond), cond.shape[1], 24, 6)).to(dev)
    joints = model.flow_matching.smpl.forward(q, pos)
    return joints.cpu().numpy(), q.reshape(len(cond), cond.shape[1], 72).cpu().numpy(), pos.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--checkpoint', default=os.path.join(FLOWER, 'ckpt', 'train-3700.pt'))
    a = ap.parse_args()
    meta = json.load(open(os.path.join(EVAL, 'data', 'mace_test', 'names.json')))
    names, genres = meta['names'], meta['genres']
    raw = np.stack([np.load(os.path.join(FEATS, f'{n}.npy')) for n in names]).astype(np.float32)
    feats = inf.normalize_features(raw)
    gid = [inf.GENRES.index(g) for g in genres]
    print('features', raw.shape, 'normalized range', float(feats.min()), float(feats.max()), flush=True)
    model = inf.load_model(inf.resolve_checkpoint(a.checkpoint))
    out_j, out_dir = [], os.path.join(EVAL, 'data', 'rot_export')
    os.makedirs(out_dir, exist_ok=True)
    for s in range(a.seeds):
        J, Q, P = [], [], []
        for c in range(0, len(names), CHUNK):
            j, q, p = sample(model, feats[c:c + CHUNK], gid[c:c + CHUNK], s)
            J.append(j); Q.append(q); P.append(p)
            print(f'seed {s} slices {c}-{c + len(j) - 1} joints {j.shape} finite {np.isfinite(j).all()}', flush=True)
        J, Q, P = np.concatenate(J), np.concatenate(Q), np.concatenate(P)
        out_j.append(J.astype(np.float16))
        np.savez(os.path.join(out_dir, f'flower_seed{s}.npz'), poses=Q, trans=P)
    dest = os.path.join(EVAL, 'data', 'mace_test_flower')
    os.makedirs(dest, exist_ok=True)
    np.save(os.path.join(dest, 'joints.npy'), np.stack(out_j, 1))
    shutil.copy(os.path.join(EVAL, 'data', 'mace_test', 'names.json'), os.path.join(dest, 'names.json'))
    print('saved', dest, np.stack(out_j, 1).shape)


if __name__ == '__main__':
    main()
