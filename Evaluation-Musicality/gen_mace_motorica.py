"""Generate Motion Expert dances on every Motorica analysis slice (motorica_common.pool / slice_starts).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/gen_mace_motorica.py --seeds 10

Each slice is 34 s of the take's own wav; baseline features are extracted from that slice audio (as for the
FineDance test slices). Style labels: motorica_common.STYLE_TO_MACE. Own- and wrong-music scores use the same MACE
dance, so the label does not enter Delta.
Writes data/motorica/mace/joints.npy (slices, seeds, 1020, 24, 3) float16 and slices.json.
--local_attn 2 (2026-11-04): +-2 s cross-attention mask (length_curve.local_cross_attention), written to
data/motorica/<--out_name, default mace_local2s>; seeds are the same as the released run (paired noise).
--lock_guide <scale> (2026-11-05): beat-locking guidance (MaceSampler.sample_guided), only on the non-overlapping
slices that beat_suite.py scores (offset % 1020 == 0); null beats = slices of other takes. Written to
data/motorica/<--out_name, default mace_guide_s<scale>>; slices.json lists the slices kept.
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
    ap.add_argument('--local_attn', type=float, help='half-width (s) of the local cross-attention mask')
    ap.add_argument('--out_name', help="output folder under data/motorica (default 'mace', or 'mace_local<w>s')")
    ap.add_argument('--lock_guide', type=float, default=0)
    ap.add_argument('--lock_guide_tmax', type=int, default=600)
    args = ap.parse_args()
    name = args.out_name or ('mace' if not args.local_attn else f'mace_local{args.local_attn:g}s')
    if args.lock_guide and not args.out_name:
        name = f'mace_guide_s{args.lock_guide:g}' + ('' if args.lock_guide_tmax == 600 else f'_t{args.lock_guide_tmax}')
    out_dir = os.path.join(os.path.dirname(ROOT), name)
    os.makedirs(out_dir, exist_ok=True)

    slices, feats = [], []
    for t in pool():
        y = af.load_audio(wav_path(t))
        for o in slice_starts(t):
            f = af.extract_baseline(y[o * af.HOP_LENGTH:(o + SLICE_LEN + 1) * af.HOP_LENGTH])[:SLICE_LEN]
            assert len(f) == SLICE_LEN, (t, o, len(f))
            slices.append((t, o)); feats.append(f)
    if args.lock_guide:
        keep = [k for k, (t, o) in enumerate(slices) if o % 1020 == 0]
        slices, feats = [slices[k] for k in keep], [feats[k] for k in keep]
    feats = np.stack(feats)
    styles = [STYLE_TO_MACE[style(t)] for t, _ in slices]
    print(len(slices), 'slices')

    sampler = MaceSampler()
    if args.local_attn:
        from length_curve import local_cross_attention
        local_cross_attention(sampler.model.diffusion.model, round(args.local_attn * 30))
    out = np.zeros((len(slices), args.seeds, SLICE_LEN, 24, 3), dtype=np.float16)
    for s in range(args.seeds):
        for b in range(0, len(slices), BATCH):
            if args.lock_guide:
                out[b:b + BATCH, s] = sampler.sample_guided(
                    feats[b:b + BATCH], styles[b:b + BATCH], seed=s * 1000 + b, scale=args.lock_guide,
                    t_max=args.lock_guide_tmax, groups=[t for t, _ in slices[b:b + BATCH]])
            else:
                out[b:b + BATCH, s] = sampler.sample(feats[b:b + BATCH], styles[b:b + BATCH], seed=s * 1000 + b)
        print('seed', s, 'done')
    np.save(os.path.join(out_dir, 'joints.npy'), out)
    with open(os.path.join(out_dir, 'slices.json'), 'w') as f:
        json.dump({'slices': slices, 'styles': styles}, f)
    print('saved', out.shape)


if __name__ == '__main__':
    main()
