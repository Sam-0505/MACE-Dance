"""Validity gate for the AIST++ critic (critic4.py --data aist), criteria fixed 2026-11-11 BEFORE running, copied from
critic_gate4.py (FineDance). Data: the 42 AIST++ test slices (5 s = 150 frames, EDGE split; music pieces never used for the
critic's training clips of the same dance are not guaranteed unseen, so the unit is the music piece), one window per clip.

  delta   per clip: S(own grid) - mean S(grid of the 12 other-piece clips with the closest beat count)
  G1 offset sanity   real dance, the grid shifted by 32 constant fractions of a beat: range of the mean own score < 10 % of delta_real
  G2 drift           real dance, AR(1) drift (rho 0.8; stationary std 0, .05, .1, .2, .3 beat; 4 draws): the mean own score falls at
                     every step and S(0) - S(0.3) >= 50 % of delta_real
  G3 own piece       delta_real > 0 with p < 0.05 (sign-flip over pieces), and delta_real - delta_model > 0 with p < 0.05, where
                     "model" = MACE trained on AIST++ (epoch-100 checkpoint, 3 seeds, same slices)
Pass = G1 and G2 and G3. Writes results/critic_gate_aist.json.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/critic_gate_aist.py --critic ../Evaluation-Musicality/data/critic/critic_aist.pt
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import critic4 as K  # noqa: E402
from critic import BEAT_COL, load_critic  # noqa: E402
from tmd_beat_metrics import signflip  # noqa: E402
from aist_common import GENRE_TO_STYLE  # noqa: E402

N_NULL = 12
D = os.path.join(HERE, '..', 'Expert-Motion', 'data', 'aist', 'test')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--critic', required=True)
    ap.add_argument('--model_ckpt', default='runs/aist/mace/weights/train-100.pt')
    args = ap.parse_args()
    from vis import SMPLSkeleton
    from mace_sampler import MaceSampler
    dev = 'cuda'
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{D}/motions_sliced/*.pkl'))
    song = np.array([n.split('_')[4] for n in names])
    print(len(names), 'slices,', len(set(song)), 'pieces', flush=True)
    feats_np = np.stack([np.load(f'{D}/baseline_feats/{n}.npy') for n in names])
    feats = torch.as_tensor(feats_np, dtype=torch.float32, device=dev)
    nb = (feats[:, :, BEAT_COL] > 0.5).sum(1).float().cpu().numpy()
    null = []
    for i in range(len(names)):
        cand = np.flatnonzero(song != song[i])
        null.append(cand[np.argsort(np.abs(nb[cand] - nb[i]), kind='stable')[:N_NULL]])
    null_idx = torch.as_tensor(np.stack(null), device=dev)
    critic = load_critic(dev, args.critic)
    pos_g, mask_g = K.grid(feats[:, :, BEAT_COL])
    rng = np.random.default_rng(0)
    pieces = np.unique(song)

    def per_song(x):
        return np.array([x[song == g].mean() for g in pieces])

    sk = SMPLSkeleton(device='cpu')
    ds = [pickle.load(open(f'{D}/motions_sliced/{n}.pkl', 'rb')) for n in names]
    with torch.no_grad():
        real = sk.forward(torch.tensor(np.stack([d['q'] for d in ds])).float().reshape(len(ds), -1, 24, 3),
                          torch.tensor(np.stack([d['pos'] for d in ds])).float()).numpy()

    @torch.no_grad()
    def events(j):
        return critic.ev(torch.as_tensor(np.asarray(j), dtype=torch.float32, device=dev))

    @torch.no_grad()
    def clip_delta(A):
        own = critic.score(A, pos_g, mask_g)
        nul = torch.stack([critic.score(A, pos_g[null_idx[:, k]], mask_g[null_idx[:, k]]) for k in range(N_NULL)]).mean(0)
        return (own - nul).cpu().numpy()

    @torch.no_grad()
    def own_score(A, transform):
        return float(critic.score(A, *transform(pos_g, mask_g)).mean().item())

    sampler = MaceSampler(args.model_ckpt)
    gens = []
    for seed in range(3):
        gens.append(sampler.sample(feats_np, [GENRE_TO_STYLE[n[:3]] for n in names], seed))
    A_real = events(real)
    deltas = {'real': per_song(clip_delta(A_real)),
              'model': per_song(np.mean([clip_delta(events(g)) for g in gens], 0))}
    out = {}
    for lab, x in deltas.items():
        out[lab] = {'delta_mean': float(x.mean()), 'p': signflip(x, rng), 'pieces_positive': f'{(x > 0).sum()}/{len(x)}'}
        print(f"{lab:6s} own - tempo-matched other-piece {x.mean():+.4f}  p {out[lab]['p']:.4f}  pieces>0 {out[lab]['pieces_positive']}", flush=True)
    dr = deltas['real']
    diff = dr - deltas['model']
    out['real_minus_model'] = {'mean': float(diff.mean()), 'p': signflip(diff, rng), 'pieces_positive': f'{(diff > 0).sum()}/{len(diff)}'}
    fr = np.arange(32) / 32
    curve = [own_score(A_real, lambda p, m, f=f: K.shift_frac(p, m, float(f))) for f in fr]
    g1_range = float(max(curve) - min(curve))
    amps = [0.0, 0.05, 0.1, 0.2, 0.3]
    dcurve = []
    for a in amps:
        gen = torch.Generator(device=dev).manual_seed(7)
        dcurve.append(float(np.mean([own_score(A_real, lambda p, m: K.drift(p, m, torch.full((p.shape[0],), a, device=dev),
                                                                              torch.full((p.shape[0],), 0.8, device=dev), gen))
                                     for _ in range(4)])))
    dreal = float(dr.mean())
    g1 = dreal > 0 and g1_range < 0.10 * dreal
    g2 = all(dcurve[i + 1] < dcurve[i] for i in range(len(dcurve) - 1)) and dreal > 0 and (dcurve[0] - dcurve[-1]) >= 0.5 * dreal
    g3 = dreal > 0 and out['real']['p'] < 0.05 and out['real_minus_model']['mean'] > 0 and out['real_minus_model']['p'] < 0.05
    out['G1_offset'] = {'curve': curve, 'range': g1_range, 'limit': 0.10 * dreal, 'pass': bool(g1)}
    out['G2_drift'] = {'amps': amps, 'own_score': dcurve, 'drop_0.3': dcurve[0] - dcurve[-1], 'need': 0.5 * dreal, 'pass': bool(g2)}
    out['G3_own_piece'] = {'pass': bool(g3)}
    out['PASS'] = bool(g1 and g2 and g3)
    print(f"real - model per piece {diff.mean():+.4f}  p {out['real_minus_model']['p']:.4f}  pieces>0 {out['real_minus_model']['pieces_positive']}")
    print(f"G1 offset: range {g1_range:.4f} (limit {0.10 * dreal:.4f}) -> {'PASS' if g1 else 'FAIL'}")
    print('G2 drift: ' + ', '.join(f'{a}: {v:.4f}' for a, v in zip(amps, dcurve)) + f"; drop {dcurve[0] - dcurve[-1]:.4f} (need {0.5 * dreal:.4f}) -> {'PASS' if g2 else 'FAIL'}")
    print(f"G3 own piece -> {'PASS' if g3 else 'FAIL'}")
    print('GATE ' + ('PASSED' if out['PASS'] else 'FAILED'))
    json.dump(out, open(os.path.join(HERE, 'results', 'critic_gate_aist.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
