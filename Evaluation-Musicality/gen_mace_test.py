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
Beat-locking guidance (2026-11-05): --lock_guide <scale> [--lock_guide_tmax 600] [--lock_guide_shift 10 (placebo)].
Critic guidance (2026-11-07): add --critic_guide to --lock_guide <scale> (score = frozen critic.py critic, not the locking formula).

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
from critic import load_critic


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
    ap.add_argument('--lock_guide', type=float, default=0, help='beat-locking guidance scale (MaceSampler.sample_guided)')
    ap.add_argument('--lock_guide_tmax', type=int, default=600, help='guide only at steps t <= this')
    ap.add_argument('--lock_guide_world', action='store_true', help='score world joints (default: root-relative)')
    ap.add_argument('--lock_guide_shift', type=int, default=0,
                    help='placebo: guide towards the beats of the clip this many places earlier (another song)')
    ap.add_argument('--lock_foot', type=int, default=0,
                    help='foot-contact correction steps after each guidance step (sample_guided foot_iters; 0 = off)')
    ap.add_argument('--lock_foot_mu', type=float, default=0.1, help='fraction of the Polyak step per foot step')
    ap.add_argument('--lock_freeze_root', action='store_true', help='no guidance on the root rotation (sample_guided)')
    ap.add_argument('--grad_smooth', type=float, default=0.0, help='Gaussian sigma (frames) low-pass of the guidance gradient in time (0 = off)')
    ap.add_argument('--gain_ref', type=float, default=0.0, help='motion-proportional guidance gain: reference speed m/s (0 = off; 0.49 = real FineDance median)')
    ap.add_argument('--critic_guide', action='store_true',
                    help='guide with the frozen music-motion critic (critic.py) instead of the beat-locking score')
    ap.add_argument('--critic_ckpt', default=None, help='critic checkpoint for --critic_guide (default data/critic/critic_v5.pt)')
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
    if args.lock_guide:
        import functools
        sample = functools.partial(sampler.sample_guided, scale=args.lock_guide, t_max=args.lock_guide_tmax,
                                   root_relative=not args.lock_guide_world, guide_shift=args.lock_guide_shift,
                                   foot_iters=args.lock_foot, foot_mu=args.lock_foot_mu,
                                   freeze_root_rot=args.lock_freeze_root, grad_smooth=args.grad_smooth, gain_ref=args.gain_ref,
                                   critic=load_critic(path=args.critic_ckpt) if args.critic_guide else None,
                                   critic_groups=[n.split('_')[0] for n in names])
    out = np.stack([sample(feats, genres, seed=s).astype(np.float16) for s in range(args.seeds)], axis=1)
    np.save(os.path.join(args.out_dir, 'joints.npy'), out)
    with open(os.path.join(args.out_dir, 'names.json'), 'w') as f:
        json.dump({'names': names, 'genres': genres}, f)
    print('saved', out.shape)


if __name__ == '__main__':
    main()
