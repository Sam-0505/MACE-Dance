"""Generate several Motion Expert samples per test clip (shipped baseline features).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/gen_mace_test.py --seeds 10
    .venv/bin/python ../Evaluation-Musicality/gen_mace_test.py --checkpoint runs/train/<exp>/weights/train-<n>.pt \
        --out_dir ../Evaluation-Musicality/data/mace_test_<exp>
    .venv/bin/python ../Evaluation-Musicality/gen_mace_test.py --stitch --out_dir ../Evaluation-Musicality/data/mace_test_stitched
    .venv/bin/python ../Evaluation-Musicality/gen_mace_test.py --local_attn 2 --out_dir ../Evaluation-Musicality/data/mace_test_local2s

Beat-input tests (2026-11-04): --zero_cols 34 removes the beat one-hot, --zero_cols 0 33 34 the onset envelope,
onset peaks and beats; --beta sets the GFT guidance mix at inference (released 0.75; trained on U(0, 1)).
Same seeds as the released run, so differences are paired.

Writes Evaluation-Musicality/data/mace_test/joints.npy (clips, seeds, T, 24, 3) float16 + names.json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mace_sampler import MaceSampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_dir', default='data/test')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'data', 'mace_test'))
    ap.add_argument('--checkpoint', default='./runs/train/exp/weights/train-3750.pt')
    ap.add_argument('--stitch', action='store_true', help='overlapping 8 s windows (MaceSampler.sample_stitched)')
    ap.add_argument('--local_attn', type=float,
                    help='seconds: restrict cross-attention to +- this around each frame (length_curve.local_cross_attention)')
    ap.add_argument('--zero_cols', type=int, nargs='*', default=[], help='feature columns set to 0')
    ap.add_argument('--beta', type=float, help='GFT guidance mix at inference (default 0.75)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    names = sorted(f[:-4] for f in os.listdir(os.path.join(args.gt_dir, 'baseline_feats')))
    feats = np.stack([np.load(os.path.join(args.gt_dir, 'baseline_feats', n + '.npy')) for n in names])
    feats[..., args.zero_cols] = 0
    genres = []
    for n in names:
        with open(os.path.join(args.gt_dir, 'label', n.split('_')[0] + '.json')) as f:
            genres.append(json.load(f)['style2'])

    sampler = MaceSampler(args.checkpoint)
    if args.local_attn:
        from length_curve import local_cross_attention
        local_cross_attention(sampler.model.diffusion.model, round(args.local_attn * 30))
    if args.beta is not None:
        import functools
        net = sampler.model.diffusion.model
        net.infer_pred = functools.partial(type(net).infer_pred, net, beta=args.beta)
    sample = sampler.sample_stitched if args.stitch else sampler.sample
    out = np.stack([sample(feats, genres, seed=s).astype(np.float16) for s in range(args.seeds)], axis=1)
    np.save(os.path.join(args.out_dir, 'joints.npy'), out)
    with open(os.path.join(args.out_dir, 'names.json'), 'w') as f:
        json.dump({'names': names, 'genres': genres}, f)
    print('saved', out.shape)


if __name__ == '__main__':
    main()
