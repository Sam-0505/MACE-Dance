"""Validity gate for critic v4 (critic4.py), decided 2026-11-08 BEFORE training. FineDance test (84 slices of 34 s, songs never
seen by the critic), 8 s windows (stride 240), unit = song (mean over its clips, windows, seeds), sign-flip tests.

  delta   per clip: S(own grid) - mean S(grid of the 12 other-song clips with the closest beat count) (tempo-matched null)
  G1 offset sanity   real dance, the grid shifted by 32 constant fractions of a beat: range of the mean own score
                     < 10 % of delta_real (the critic must ignore a constant offset)
  G2 drift           real dance, AR(1) drift of the grid (rho 0.8; stationary std 0, 0.05, 0.1, 0.2, 0.3 beat; 4 draws): the mean
                     own score falls at every step, and S(0) - S(0.3) >= 50 % of delta_real
  G3 own song        delta_real > 0 with p < 0.05, and delta_real - delta_released > 0 per song with p < 0.05
Pass = G1 and G2 and G3. Other sets (mask, final) are reported, not gated. Writes results/critic_gate4.json.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/critic_gate4.py --critic ../Evaluation-Musicality/data/critic/critic_v5.pt
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
import critic4 as K  # noqa: E402
from critic import BEAT_COL, load_critic  # noqa: E402
from tmd_beat_metrics import signflip  # noqa: E402

N_NULL = 12
SETS = {'released': 'mace_test', 'mask': 'mace_test_local2s', 'final': 'mace_test_mask2s_guide10_s0.003_foot10_frz'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--critic', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--sets', nargs='*', help='label=data_dir to score instead of the default sets (released is always kept)')
    args = ap.parse_args()
    if args.sets:
        SETS.clear()
        SETS.update({'released': 'mace_test', **dict(x.split('=', 1) for x in args.sets)})
    from motion_quality_extra import real_joints
    dev = 'cuda'
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    song = np.array([n.split('_')[0] for n in names])
    feats = torch.as_tensor(np.stack([np.load(f'data/test/baseline_feats/{n}.npy') for n in names]), dtype=torch.float32, device=dev)
    nb = (feats[:, :, BEAT_COL] > 0.5).sum(1).float().cpu().numpy()
    null = []
    for i in range(len(names)):
        cand = np.flatnonzero(song != song[i])
        null.append(cand[np.argsort(np.abs(nb[cand] - nb[i]), kind='stable')[:N_NULL]])
    null_idx = torch.as_tensor(np.stack(null), device=dev)
    critic = load_critic(dev, args.critic)
    T = feats.shape[1]
    starts = list(range(0, T - K.WIN + 1, K.WIN))
    grids = [K.grid(feats[:, s:s + K.WIN, BEAT_COL]) for s in starts]
    rng = np.random.default_rng(0)

    def per_song(x):
        return np.array([x[song == g].mean() for g in np.unique(song)])

    @torch.no_grad()
    def events(joints):
        j = torch.as_tensor(np.asarray(joints), dtype=torch.float32, device=dev)
        return [critic.ev(j[:, s:s + K.WIN]) for s in starts]

    @torch.no_grad()
    def clip_delta(A):
        d = torch.zeros(len(names), device=dev)
        for a, (pos, mask) in zip(A, grids):
            own = critic.score(a, pos, mask)
            nul = torch.stack([critic.score(a, pos[null_idx[:, k]], mask[null_idx[:, k]]) for k in range(N_NULL)]).mean(0)
            d += (own - nul) / len(starts)
        return d.cpu().numpy()

    @torch.no_grad()
    def own_score(A, transform):
        return float(np.mean([critic.score(a, *transform(pos, mask)).mean().item() for a, (pos, mask) in zip(A, grids)]))

    out = {}
    A_real = events(real_joints(names))
    deltas = {'real': per_song(clip_delta(A_real))}
    for lab, d in SETS.items():
        J = np.load(os.path.join(HERE, 'data', d, 'joints.npy'), mmap_mode='r')
        deltas[lab] = per_song(np.mean([clip_delta(events(np.asarray(J[:, k], dtype=np.float32))) for k in range(J.shape[1])], 0))
    for lab, x in deltas.items():
        out[lab] = {'delta_mean': float(x.mean()), 'p': signflip(x, rng), 'songs_positive': f'{(x > 0).sum()}/{len(x)}'}
        print(f"{lab:9s} own - tempo-matched other-song {x.mean():+.4f}  p {out[lab]['p']:.4f}  songs>0 {out[lab]['songs_positive']}", flush=True)
    dr = deltas['real']
    diff = dr - deltas['released']
    out['real_minus_released'] = {'mean': float(diff.mean()), 'p': signflip(diff, rng), 'songs_positive': f'{(diff > 0).sum()}/{len(diff)}'}

    # G1 constant offset
    fr = np.arange(32) / 32
    curve = [own_score(A_real, lambda p, m, f=f: K.shift_frac(p, m, float(f))) for f in fr]
    g1_range = float(max(curve) - min(curve))
    # G2 drift
    amps = [0.0, 0.05, 0.1, 0.2, 0.3]
    dcurve = []
    for a in amps:
        gen = torch.Generator(device=dev).manual_seed(7)
        vals = []
        for _ in range(4):
            vals.append(own_score(A_real, lambda p, m: K.drift(p, m, torch.full((p.shape[0],), a, device=dev),
                                                               torch.full((p.shape[0],), 0.8, device=dev), gen)))
        dcurve.append(float(np.mean(vals)))
    dreal = float(dr.mean())
    g1 = dreal > 0 and g1_range < 0.10 * dreal
    g2 = all(dcurve[i + 1] < dcurve[i] for i in range(len(dcurve) - 1)) and dreal > 0 and (dcurve[0] - dcurve[-1]) >= 0.5 * dreal
    g3 = dreal > 0 and out['real']['p'] < 0.05 and out['real_minus_released']['mean'] > 0 and out['real_minus_released']['p'] < 0.05
    out['G1_offset'] = {'curve': curve, 'range': g1_range, 'limit': 0.10 * dreal, 'pass': bool(g1)}
    out['G2_drift'] = {'amps': amps, 'own_score': dcurve, 'drop_0.3': dcurve[0] - dcurve[-1], 'need': 0.5 * dreal, 'pass': bool(g2)}
    out['G3_own_song'] = {'pass': bool(g3)}
    out['PASS'] = bool(g1 and g2 and g3)
    print(f"real - released per song {diff.mean():+.4f}  p {out['real_minus_released']['p']:.4f}  songs>0 {out['real_minus_released']['songs_positive']}")
    print(f"G1 offset: score range over 32 constant shifts {g1_range:.4f} (limit {0.10 * dreal:.4f}) -> {'PASS' if g1 else 'FAIL'}")
    print('G2 drift: own score at drift ' + ', '.join(f'{a}: {v:.4f}' for a, v in zip(amps, dcurve))
          + f"; drop {dcurve[0] - dcurve[-1]:.4f} (need {0.5 * dreal:.4f}) -> {'PASS' if g2 else 'FAIL'}")
    print(f"G3 own song (real p < 0.05, real > released p < 0.05) -> {'PASS' if g3 else 'FAIL'}")
    print('GATE ' + ('PASSED' if out['PASS'] else 'FAILED'))
    json.dump(out, open(os.path.join(HERE, 'results', f'critic_gate4{args.tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
