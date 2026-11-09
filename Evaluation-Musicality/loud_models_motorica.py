"""Loudness <-> QoM coupling on Motorica for real dance and every model, ONE seed (same metric as loud_models.py).

Metric (one clip): r = Pearson( Gaussian_sigma15(audio RMS), Gaussian_sigma15(QoM) ), zero lag, QoM = mean unsquared
  3D joint speed (global joints). Audio RMS linear, 15360 Hz, hop 512. Slices are cut BEFORE smoothing.
Slices: the 289 non-overlapping 34 s slices (offset % 1020 == 0) of 62 takes, as in beat_suite.load_motorica.
Units: 52 music pieces (takes that share one recording are merged, finedance_release_loud.music_groups; a piece is never
  its own wrong song). Own music = the take's wav; wrong music = 3 tempo-first matched pieces (robustness_loud.matches,
  full-piece BPM), one seeded random slice of each; the same wrong slices for real dance and every model.
Real dance = 21 BVH body joints (m); models = 24 SMPL joints, seed index 0 of data/motorica/<dir>/joints.npy.
Tests, p and q: as loud_models.py (one-sided sign-flip permutation of Delta > 0; BH over models for the q of the
  "model shows coupling" family; real dance is a single test). Paired Delta differences between models are also reported.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/loud_models_motorica.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

import audio_features as af  # noqa: E402
from multicomponent import SEED, smooth, boot_ci, wil, signflip_p, bh  # noqa: E402
from robustness_loud import loudness_signals, motion_signals, bpm_from_wav, matches, TEMPO_TOL  # noqa: E402
from finedance_release_loud import music_groups  # noqa: E402
from motorica_common import ROOT, SLICE_LEN, body_joints, style, wav_path  # noqa: E402

SIGMA = 15
# seeds 0..N_SEEDS-1 of every model (same number for all; Motorica has 3 for most models); --seeds N, default 1.
N_SEEDS = int(sys.argv[sys.argv.index('--seeds') + 1]) if '--seeds' in sys.argv else 1
MODELS = {
    'MACE released': 'mace',
    'MACE mask2s': 'mace_local2s',
    'Ours final': 'mace_local2s_guide_s0.003_foot10_frz',
    'Ours critic v5': 'mace_local2s_criticv5_s0.003_foot10_frz',
    'EDGE': 'edge',
    'Lodge': 'lodge',
    'Bailando': 'bailando',
    'MEGADance': 'mega',
}
OUT = os.path.join(HERE, 'results', f'loud_models_motorica_{N_SEEDS}seed.json')


def main():
    mot = os.path.dirname(ROOT)
    meta = json.load(open(os.path.join(mot, 'mace', 'slices.json')))
    keep = [k for k, (t, o) in enumerate(meta['slices']) if o % SLICE_LEN == 0]
    slices = [tuple(meta['slices'][k]) for k in keep]
    takes = sorted({t for t, _ in slices})
    group = music_groups(takes, path=wav_path)
    pieces = sorted(set(group.values()))
    genre = {p: style(p) for p in pieces}
    slice_piece = np.array([group[t] for t, _ in slices])
    print(f'{len(slices)} slices, {len(takes)} takes, {len(pieces)} pieces', flush=True)

    music = {t: loudness_signals(af.load_audio(wav_path(t)))['rms'] for t in takes}
    mus = [smooth(music[t][o:o + SLICE_LEN], SIGMA) for t, o in slices]
    qom = lambda j: smooth(motion_signals(np.asarray(j, dtype=np.float64))['speed'], SIGMA)  # noqa: E731
    cache, real = {}, []
    for t, o in slices:
        if t not in cache:
            cache = {t: body_joints(t)}
        real.append([qom(cache[t][o:o + SLICE_LEN])])   # per slice: list over seeds
    sets = {'Real dance': real}
    for label, d in MODELS.items():
        p = os.path.join(mot, d, 'joints.npy')
        if not os.path.exists(p):
            print('missing', label, p)
            continue
        J = np.load(p, mmap_mode='r')
        if J.shape[0] == len(keep):
            assert [tuple(x) for x in json.load(open(os.path.join(mot, d, 'slices.json')))['slices']] == slices
            idx = list(range(len(keep)))
        else:
            idx = keep
        assert J.shape[1] >= N_SEEDS, (label, J.shape)
        sets[label] = [[qom(J[k, s]) for s in range(N_SEEDS)] for k in idx]
    print('signals done:', list(sets), flush=True)

    bpm = {p: bpm_from_wav(wav_path(p)) for p in pieces}
    mt = matches(pieces, genre, bpm, 'tempo_first')
    piece_slices = {p: [i for i, q in enumerate(slice_piece) if q == p] for p in pieces}
    rng = np.random.default_rng(SEED)
    pair = [[piece_slices[w][int(rng.integers(len(piece_slices[w])))] for w in mt[group[t]]] for t, _ in slices]

    def piece_level(v):
        return np.array([np.mean(v[slice_piece == p]) for p in pieces])

    r0 = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
    per = {}
    for label, mo in sets.items():
        ns, S = len(mo[0]), len(slices)
        o_s = np.array([[r0(mus[i], x) for x in mo[i]] for i in range(S)])                        # (slices, seeds)
        w_s = np.array([[np.mean([r0(mus[j], x) for j in pair[i]]) for x in mo[i]] for i in range(S)])
        own, wrong = piece_level(o_s.mean(1)), piece_level(w_s.mean(1))
        per[label] = {'own': own, 'wrong': wrong, 'delta': own - wrong,
                      'seed_delta': [float((piece_level(o_s[:, s]) - piece_level(w_s[:, s])).mean()) for s in range(ns)]}

    rng = np.random.default_rng(SEED)
    res = {}
    for label, v in per.items():
        d = v['delta']
        res[label] = {'r_own': float(v['own'].mean()), 'r_wrong': float(v['wrong'].mean()), 'delta': float(d.mean()),
                      'delta_ci': boot_ci(d, rng), 'p_perm': signflip_p(d, rng, 'greater'),
                      'p_wilcoxon': wil(d, 'greater'), 'n_pos': int((d > 0).sum()), 'n': len(d),
                      'seed_delta': v['seed_delta']}
        if label != 'Real dance':
            g = per['Real dance']['delta'] - d
            res[label].update({'G': float(g.mean()), 'G_ci': boot_ci(g, rng), 'G_p_perm': signflip_p(g, rng),
                               'G_p_wilcoxon': wil(g, 'two-sided')})
    ml = [k for k in res if k != 'Real dance']
    for key, q in (('p_perm', 'q_perm'), ('p_wilcoxon', 'q_wilcoxon'), ('G_p_perm', 'G_q_perm')):
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
    print(f"{'model':16s} {'r_own':>6s} {'r_wrong':>7s} {'Delta':>6s} {'CI95':>15s} {'pos':>6s} {'p':>7s} {'q':>7s} "
          f"{'seed Delta min..max':>20s}")
    for k, v in res.items():
        sd = v['seed_delta']
        print(f"{k:16s} {v['r_own']:6.3f} {v['r_wrong']:7.3f} {v['delta']:6.3f} "
              f"[{v['delta_ci'][0]:+.3f},{v['delta_ci'][1]:+.3f}] {v['n_pos']:2d}/{v['n']} {v['p_perm']:7.4f} "
              f"{v['q_perm']:7.4f} {min(sd):+.3f}..{max(sd):+.3f}")
    print('\npaired (Delta difference, two-sided):')
    for k, v in paired.items():
        print(f"  {k}: {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] p {v['p_perm']:.3f} "
              f"first higher in {v['n_a_higher']}/{v['n']}")
    out = {'settings': {'sigma': SIGMA, 'lag': 0, 'motion': 'QoM (unsquared mean joint speed, global)', 'music': 'linear RMS',
                        'match': 'tempo_first', 'tempo_tol': TEMPO_TOL, 'n_seeds': N_SEEDS, 'n_pieces': len(pieces),
                        'n_slices': len(slices), 'models': MODELS, 'n_perm': 10000, 'rng_seed': SEED,
                        'q': 'BH over models per family; real H1 single test'},
           'bpm': bpm, 'results': res, 'paired': paired,
           'per_piece': {k: {kk: dict(zip(pieces, map(float, vv))) for kk, vv in v.items() if kk != 'seed_delta'}
                         for k, v in per.items()}}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
