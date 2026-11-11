"""Generate the AIST++-trained MACE (critic + gain variant, see aist_eval.py) on the Motorica slices (2026-11-11).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/gen_aist_model_motorica.py --seeds 3 [--limit 8]

Slices: the non-overlapping 34 s slices scored by beat_suite.py (offset % 1020 == 0, 289 slices), 3 seeds. The model was trained on
5 s windows, so a slice is made like EDGE's long sampling: 150-frame windows, stride 75, crossfaded (MaceSampler.sample_stitched_guided).
34 s = 1020 frames is not 150 + 75k, so 30 frames of the last feature row are appended (1050 = 150 + 12 * 75) and the dance is cut
back to 1020. Critic guidance 0.003 + foot term 10 + frozen root rotation + motion gain 0.77 m/s (real AIST++ median), the AIST++ critic;
null music = windows of the other takes. Styles: motorica_common.STYLE_TO_MACE (all inside the six labels the model was trained on).
Writes data/motorica/<--out_name>/joints.npy (slices, seeds, 1020, 24, 3) float16 (MACE's Z-up frame) and slices.json.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from mace_sampler import MaceSampler  # noqa: E402
import audio_features as af  # noqa: E402
from motorica_common import ROOT, STYLE_TO_MACE, SLICE_LEN, pool, slice_starts, style, wav_path  # noqa: E402

WIN, STRIDE = 150, 75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--ckpt', default='runs/aist/mace/weights/train-100.pt')
    ap.add_argument('--critic', default=os.path.join(HERE, 'data', 'critic', 'critic_aist.pt'))
    ap.add_argument('--out_name', default='ours_aist')
    ap.add_argument('--scale', type=float, default=0.003)
    ap.add_argument('--foot', type=int, default=10)
    ap.add_argument('--gain_ref', type=float, default=0.77)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--plain', action='store_true', help='no guidance (scale 0)')
    args = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(ROOT), args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    slices, feats = [], []
    for t in pool():
        y = af.load_audio(wav_path(t))
        for o in slice_starts(t):
            if o % 1020:
                continue
            f = af.extract_baseline(y[o * af.HOP_LENGTH:(o + SLICE_LEN + 1) * af.HOP_LENGTH])[:SLICE_LEN]
            assert len(f) == SLICE_LEN, (t, o, len(f))
            slices.append((t, o))
            feats.append(np.concatenate([f, np.repeat(f[-1:], 30, 0)]))
    if args.limit:
        slices, feats = slices[:args.limit], feats[:args.limit]
    feats = np.stack(feats)
    styles = [STYLE_TO_MACE[style(t)] for t, _ in slices]
    takes = [t for t, _ in slices]
    print(len(slices), 'slices', flush=True)
    bank = np.concatenate([[f[k:k + WIN] for k in range(0, f.shape[0] - WIN + 1, STRIDE)] for f in feats])
    bank_g = np.concatenate([[t] * ((f.shape[0] - WIN) // STRIDE + 1) for t, f in zip(takes, feats)])
    from critic import load_critic
    critic = None if args.plain else load_critic(path=args.critic)
    sampler = MaceSampler(args.ckpt)
    out = np.zeros((len(slices), args.seeds, SLICE_LEN, 24, 3), dtype=np.float16)
    for s in range(args.seeds):
        for b in range(0, len(slices), args.batch):
            sl = slice(b, b + args.batch)
            if args.plain:
                j = sampler.sample_stitched(feats[sl], styles[sl], seed=s * 1000 + b, window=WIN, stride=STRIDE)
            else:
                j = sampler.sample_stitched_guided(feats[sl], styles[sl], s * 1000 + b, bank, bank_g, takes[sl], args.scale,
                                                   WIN, STRIDE, foot_iters=args.foot, freeze_root_rot=True, critic=critic,
                                                   gain_ref=args.gain_ref)
            out[sl, s] = j[:, :SLICE_LEN]
            print('seed', s, 'clips', b + len(j), flush=True)
    np.save(os.path.join(out_dir, 'joints.npy'), out)
    with open(os.path.join(out_dir, 'slices.json'), 'w') as f:
        json.dump({'slices': slices, 'styles': styles}, f)
    print('saved', out.shape)


if __name__ == '__main__':
    main()
