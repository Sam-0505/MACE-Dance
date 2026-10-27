"""Generate several Motion Expert samples per test clip (shipped baseline features).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/gen_mace_test.py --seeds 10

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
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    names = sorted(f[:-4] for f in os.listdir(os.path.join(args.gt_dir, 'baseline_feats')))
    feats = np.stack([np.load(os.path.join(args.gt_dir, 'baseline_feats', n + '.npy')) for n in names])
    genres = []
    for n in names:
        with open(os.path.join(args.gt_dir, 'label', n.split('_')[0] + '.json')) as f:
            genres.append(json.load(f)['style2'])

    sampler = MaceSampler()
    out = np.stack([sampler.sample(feats, genres, seed=s).astype(np.float16) for s in range(args.seeds)], axis=1)
    np.save(os.path.join(args.out_dir, 'joints.npy'), out)
    with open(os.path.join(args.out_dir, 'names.json'), 'w') as f:
        json.dump({'names': names, 'genres': genres}, f)
    print('saved', out.shape)


if __name__ == '__main__':
    main()
