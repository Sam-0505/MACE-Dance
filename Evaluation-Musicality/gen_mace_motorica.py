"""Generate Motion Expert dances on every Motorica analysis slice (motorica_common.pool / slice_starts).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/gen_mace_motorica.py --seeds 10

Each slice is 34 s of the take's own wav; baseline features are extracted from that slice audio (as for the
FineDance test slices). Style labels: motorica_common.STYLE_TO_MACE. Own- and wrong-music scores use the same MACE
dance, so the label does not enter Delta.
Writes data/motorica/mace/joints.npy (slices, seeds, 1020, 24, 3) float16 and slices.json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mace_sampler import MaceSampler  # noqa: E402
import audio_features as af  # noqa: E402
from motorica_common import ROOT, STYLE_TO_MACE, SLICE_LEN, pool, slice_starts, style, wav_path  # noqa: E402

BATCH = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    args = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(ROOT), 'mace')
    os.makedirs(out_dir, exist_ok=True)

    slices, feats = [], []
    for t in pool():
        y = af.load_audio(wav_path(t))
        for o in slice_starts(t):
            f = af.extract_baseline(y[o * af.HOP_LENGTH:(o + SLICE_LEN + 1) * af.HOP_LENGTH])[:SLICE_LEN]
            assert len(f) == SLICE_LEN, (t, o, len(f))
            slices.append((t, o)); feats.append(f)
    feats = np.stack(feats)
    styles = [STYLE_TO_MACE[style(t)] for t, _ in slices]
    print(len(slices), 'slices')

    sampler = MaceSampler()
    out = np.zeros((len(slices), args.seeds, SLICE_LEN, 24, 3), dtype=np.float16)
    for s in range(args.seeds):
        for b in range(0, len(slices), BATCH):
            out[b:b + BATCH, s] = sampler.sample(feats[b:b + BATCH], styles[b:b + BATCH], seed=s * 1000 + b)
        print('seed', s, 'done')
    np.save(os.path.join(out_dir, 'joints.npy'), out)
    with open(os.path.join(out_dir, 'slices.json'), 'w') as f:
        json.dump({'slices': slices, 'styles': styles}, f)
    print('saved', out.shape)


if __name__ == '__main__':
    main()
