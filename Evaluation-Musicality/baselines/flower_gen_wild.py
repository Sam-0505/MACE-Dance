"""FlowerDance on the user-supplied song segments of gen_wild_songs.py (2026-11-21), 10 seeds, same sampler as flower_gen_finedance.py
(21 Euler MeanFlow steps, its embedded feature bounds, torch.manual_seed(seed) per batch). Genre = the one gen_wild_songs used.

    cd /home/sam0505/Documents/Code/dance_baselines/FlowerDance
    WANDB_MODE=offline /home/sam0505/Documents/Code/MACE-Dance/Expert-Motion/.venv/bin/python \
        /home/sam0505/Documents/Code/MACE-Dance/Evaluation-Musicality/baselines/flower_gen_wild.py

Reads data/wild/<name>/feats.npy (names and genre from data/wild/rot/gain.npz); writes data/wild/rot/flower.npz with the same keys.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flower_gen_finedance as FG  # noqa: E402  (chdirs into FlowerDance and imports its inference module)

WILD = os.path.join(FG.EVAL, 'data', 'wild')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--checkpoint', default=os.path.join(FG.FLOWER, 'ckpt', 'train-3700.pt'))
    a = ap.parse_args()
    ref = np.load(os.path.join(WILD, 'rot', 'gain.npz'))
    names, genre = [str(n) for n in ref['names']], str(ref['genre'])
    raw = np.stack([np.load(os.path.join(WILD, n, 'feats.npy')) for n in names]).astype(np.float32)
    feats = FG.inf.normalize_features(raw)
    gid = [FG.inf.GENRES.index(genre)] * len(names)
    model = FG.inf.load_model(FG.inf.resolve_checkpoint(a.checkpoint))
    P, X, J = [], [], []
    for s in range(a.seeds):
        j, q, p = FG.sample(model, feats, gid, s)
        J.append(j.astype(np.float16)); P.append(q); X.append(p)
        print(f'flower seed {s} finite {np.isfinite(j).all()}', flush=True)
    np.savez(os.path.join(WILD, 'rot', 'flower.npz'), poses=np.stack(P, 1), trans=np.stack(X, 1), joints=np.stack(J, 1),
             names=np.array(names), genre=genre)
    print('saved flower', np.stack(J, 1).shape)


if __name__ == '__main__':
    main()
