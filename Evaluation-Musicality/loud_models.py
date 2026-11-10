"""Loudness <-> QoM coupling for real dance and every model, ONE seed (FineDance test, 19 songs).

Metric (one clip):  r = Pearson( Gaussian_sigma15(audio RMS), Gaussian_sigma15(QoM) ), zero lag.
  QoM(t) = mean over the 24 joints of the 3D frame-to-frame displacement norm (unsquared speed, global joints).
  Audio RMS: linear, 15360 Hz, hop 512 (30 fps).  No lag search, no Savitzky-Golay pre-filter.
Specificity (contrast): Delta = r(own song) - mean r(3 wrong songs), tempo-first matching as in robustness_loud.py
  (full-song BPM, half/double allowed, 3 % tie tolerance, same-genre first among ties; one seeded random slice of each
  wrong song per clip, the same wrong slices for real dance and every model). Unit = song (19), mean over its slices.
Model dance = seed 0 of data/<dir>/joints.npy (Bailando and MEGADance are deterministic: one dance per slice).

Tests (sign-flip permutation, 10,000, seed 0; Wilcoxon reported next to it):
  real   H1: Delta_real > 0 (one-sided)
  model  H2: Delta_model > 0 (one-sided)                      -- "does the model show the coupling?"
  G      two-sided, (Delta_real - Delta_model) paired by song  -- "is the model different from real dance?"
q-values: Benjamini-Hochberg over the models, separately for the H2 family and the G family (permutation p).
  The real-dance H1 is a single test, so q = p.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/loud_models.py
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from model.metric import SMPLSkeleton  # noqa: E402
import metrics as M  # noqa: E402
from eval_mace import fk  # noqa: E402
from multicomponent import (T, SEED, GT_DIR, MACE_DIR, smooth, boot_ci, wil, signflip_p, bh)  # noqa: E402
from robustness_loud import (music_signals, motion_signals, song_bpm, matches, TEMPO_TOL)  # noqa: E402

SIGMA = 15
# seeds 0..N_SEEDS-1 of every model (same number for all); --seeds N, default 1. Real dance has one dance.
N_SEEDS = int(sys.argv[sys.argv.index('--seeds') + 1]) if '--seeds' in sys.argv else 1
MODELS = {   # label -> data dir (FineDance test generations, 84 slices)
    'MACE released': 'mace_test',
    'MACE mask2s': 'mace_test_local2s',
    'Ours final': 'mace_test_mask2s_guide10_s0.003_foot10_frz',
    'Ours critic v5': 'mace_test_mask2s_criticv5_s0.003_foot10_frz',
    'Ours critic v5 + gain': 'mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10',
    'Ours mask + guidance': 'mace_test_mask2s_guide10_s0.003',
    'Ours + foot term': 'mace_test_mask2s_guide10_s0.003_foot10_10seeds',
    'EDGE': 'mace_test_edge',
    'Lodge': 'mace_test_lodge',
    'Bailando': 'mace_test_bailando',
    'MEGADance': 'mace_test_mega',
}
OUT = os.path.join(HERE, 'results', f'loud_models_{N_SEEDS}seed.json')


def main():
    names = json.load(open(os.path.join(MACE_DIR, 'names.json')))['names']
    N = len(names)
    songs = np.array([n.split('_')[0] for n in names])
    order = sorted(set(songs))
    genre = {s: json.load(open(os.path.join(GT_DIR, 'label', s + '.json')))['style2'] for s in order}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    poses, trans = [], []
    for n in names:
        with open(os.path.join(GT_DIR, 'motions_sliced', n + '.pkl'), 'rb') as f:
            m = pickle.load(f)
        poses.append(m['q']); trans.append(m['pos'])
    real = fk(SMPLSkeleton(device), poses, trans, device, from_pred=False)

    mus = [smooth(music_signals(n)['rms'], SIGMA) for n in names]
    qom = lambda j: smooth(motion_signals(j)['speed'], SIGMA)  # noqa: E731
    sets = {'Real dance': [[qom(real[i])] for i in range(N)]}   # per slice: list over seeds
    for label, d in MODELS.items():
        p = os.path.join(HERE, 'data', d, 'joints.npy')
        if not os.path.exists(p):
            print('missing', label, p)
            continue
        J = np.load(p, mmap_mode='r')
        assert J.shape[0] == N and J.shape[2] == T and J.shape[1] >= N_SEEDS, (label, J.shape)
        sets[label] = [[qom(np.asarray(J[i, s], dtype=np.float64)) for s in range(N_SEEDS)] for i in range(N)]
    print('signals done:', list(sets), flush=True)

    bpm = {s: song_bpm(s) for s in order}
    mt = matches(order, genre, bpm, 'tempo_first')
    rng = np.random.default_rng(SEED)
    pair = [[int(rng.choice(np.flatnonzero(songs == o))) for o in mt[songs[i]]] for i in range(N)]

    def song_level(v):
        return np.array([v[songs == s].mean() for s in order])

    r0 = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731  zero-lag Pearson
    per = {}
    for label, mot in sets.items():
        ns = len(mot[0])
        o_s = np.array([[r0(mus[i], x) for x in mot[i]] for i in range(N)])                       # (slices, seeds)
        w_s = np.array([[np.mean([r0(mus[j], x) for j in pair[i]]) for x in mot[i]] for i in range(N)])
        own, wrong = song_level(o_s.mean(1)), song_level(w_s.mean(1))
        per[label] = {'own': own, 'wrong': wrong, 'delta': own - wrong,
                      'seed_delta': [float((song_level(o_s[:, s]) - song_level(w_s[:, s])).mean()) for s in range(ns)]}

    rng = np.random.default_rng(SEED)
    res = {}
    for label, v in per.items():
        d = v['delta']
        res[label] = {'r_own': float(v['own'].mean()), 'r_wrong': float(v['wrong'].mean()),
                      'delta': float(d.mean()), 'delta_ci': boot_ci(d, rng),
                      'p_perm': signflip_p(d, rng, 'greater'), 'p_wilcoxon': wil(d, 'greater'),
                      'n_pos': int((d > 0).sum()), 'n': len(d), 'seed_delta': v['seed_delta']}
        if label != 'Real dance':
            g = per['Real dance']['delta'] - d
            res[label].update({'G': float(g.mean()), 'G_ci': boot_ci(g, rng), 'G_p_perm': signflip_p(g, rng),
                               'G_p_wilcoxon': wil(g, 'two-sided'), 'n_G_pos': int((g > 0).sum())})
    ml = [k for k in res if k != 'Real dance']
    for key, q in (('p_perm', 'q_perm'), ('p_wilcoxon', 'q_wilcoxon'), ('G_p_perm', 'G_q_perm'),
                   ('G_p_wilcoxon', 'G_q_wilcoxon')):
        for k, qv in zip(ml, bh([res[k][key] for k in ml])):
            res[k][q] = qv
    res['Real dance']['q_perm'] = res['Real dance']['p_perm']
    res['Real dance']['q_wilcoxon'] = res['Real dance']['p_wilcoxon']

    paired = {}
    for a, b in (('Ours critic v5', 'Lodge'), ('Ours final', 'Lodge'), ('Ours critic v5', 'Ours final')):
        if a in per and b in per:
            dd = per[a]['delta'] - per[b]['delta']
            paired[f'{a} - {b}'] = {'diff': float(dd.mean()), 'ci': boot_ci(dd, rng), 'p_perm': signflip_p(dd, rng),
                                    'n_a_higher': int((dd > 0).sum()), 'n': len(dd)}

    print(f"\n{N_SEEDS} seed(s) per model")
    print(f"{'model':22s} {'r_own':>6s} {'r_wrong':>7s} {'Delta':>6s} {'CI95':>15s} {'pos':>5s} {'p':>7s} {'q':>7s} "
          f"{'seed Delta min..max':>20s}")
    for k, v in res.items():
        sd = v['seed_delta']
        print(f"{k:22s} {v['r_own']:6.3f} {v['r_wrong']:7.3f} {v['delta']:6.3f} "
              f"[{v['delta_ci'][0]:+.3f},{v['delta_ci'][1]:+.3f}] {v['n_pos']:2d}/{v['n']} {v['p_perm']:7.4f} "
              f"{v['q_perm']:7.4f} {min(sd):+.3f}..{max(sd):+.3f}")
    print('\npaired (Delta difference, two-sided):')
    for k, v in paired.items():
        print(f"  {k}: {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] p {v['p_perm']:.3f} "
              f"first higher in {v['n_a_higher']}/{v['n']}")
    out = {'settings': {'sigma': SIGMA, 'lag': 0, 'motion': 'QoM (mean unsquared joint speed, global)',
                        'music': 'linear RMS', 'match': 'tempo_first', 'tempo_tol': TEMPO_TOL, 'n_seeds': N_SEEDS,
                        'n_songs': len(order), 'models': MODELS, 'n_perm': 10000, 'rng_seed': SEED,
                        'q': 'BH over models per family (H2, G); real H1 single test'},
           'bpm_full_song': bpm, 'results': res, 'paired': paired,
           'per_song': {k: {kk: dict(zip(order, map(float, vv))) for kk, vv in v.items() if kk != 'seed_delta'}
                        for k, v in per.items()}}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
