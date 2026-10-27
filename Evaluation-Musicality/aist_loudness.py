"""Loudness <-> body energy on AIST++: the FineDance evaluation (multicomponent.py 'loud' + robustness_loud.py)
repeated on a second dataset, real dance vs MACE.

Units are music pieces, analysed separately for
  basic  the 16-beat basic dances (60 pieces, ~6 dances each), and
  adv    the 64-beat advanced dances, first 34 s (10 pieces, 1-4 dances each).
For each piece the music is its first T frames (motion frame 0 = music time 0), T = shortest real dance on it.
MACE: 10 samples per piece generated on the same segment (gen_mace_aist.py).

Wrong songs: 3 pieces from OTHER genres at the closest tempo (robustness_loud.tempo_dist, half/double allowed),
ties broken at random with a fixed seed, each long enough for T frames; their first T frames are used.
Every genre uses the same nominal tempi, so most pieces get three exact-tempo matches. The same wrong pieces are
used for the real dances and for MACE.

Variants: the robustness_loud.py list, with 'match_genre_first' replaced by 'match_random' (3 random other-genre
pieces, any tempo). Stats as in multicomponent.component_stats, plus real own - MACE own.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/aist_loudness.py
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from vis import SMPLSkeleton  # noqa: E402
import audio_features as af  # noqa: E402
import metrics as M  # noqa: E402
from multicomponent import N_MATCH, SEED, smooth, component_stats, boot_ci, wil, signflip_p  # noqa: E402
from robustness_loud import PRIMARY, VARIANTS, TEMPO_TOL, loudness_signals, motion_signals, tempo_dist  # noqa: E402
from aist_common import DATA, GENRE_NAMES, nominal_bpm, sequences, segments, joints  # noqa: E402

OUT = os.path.join(HERE, 'results', 'aist_loudness.json')
AIST_VARIANTS = {('match_random' if k == 'match_genre_first' else k): ({'match': 'random'} if k == 'match_genre_first'
                                                                         else v) for k, v in VARIANTS.items()}


def wrong_pieces(key, seg_T, music_len, how, rng):
    piece, kind = key
    T = seg_T[key]
    cand = [p for p in music_len if p[1:3] != piece[1:3] and music_len[p] >= T]
    if how == 'random':
        return [str(p) for p in rng.choice(cand, N_MATCH, replace=False)]
    tie = rng.random(len(cand))
    order = sorted(range(len(cand)), key=lambda i: (tempo_dist(nominal_bpm(cand[i]), nominal_bpm(piece)), tie[i]))
    return [cand[i] for i in order[:N_MATCH]]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skel = SMPLSkeleton(device)
    seqs = sequences()
    seg_T = segments()
    pieces = sorted({p for p, _ in seqs.values()})
    music = {p: loudness_signals(af.load_audio(os.path.join(DATA, 'music', p + '.wav'))) for p in pieces}
    music_len = {p: len(m['rms']) for p, m in music.items()}

    real = {key: [] for key in seg_T}
    for n, key in seqs.items():
        real[key].append(motion_signals(joints(n, skel, device)[:seg_T[key]]))
    mace = {key: [motion_signals(j.astype(np.float32)) for j in np.load(os.path.join(DATA, 'mace', f'{key[0]}_{key[1]}.npy'))]
            for key in seg_T}
    print('signals done:', {k: sum(1 for x in seg_T if x[1] == k) for k in ('basic', 'adv')}, 'pieces')

    matches = {}
    for how in ('tempo_first', 'random'):
        rng = np.random.default_rng(SEED)
        matches[how] = {key: wrong_pieces(key, seg_T, music_len, how, rng) for key in sorted(seg_T)}

    results = {}
    for kind in ('basic', 'adv'):
        keys = sorted(k for k in seg_T if k[1] == kind)
        results[kind] = {}
        for vname, change in AIST_VARIANTS.items():
            cfg = {**PRIMARY, **change}
            sig, lag = cfg['sigma'], cfg['lag']
            sc = lambda a, b: M.lagged_corr(a, b, lag)  # noqa: E731
            ms = lambda p, T: smooth(music[p][cfg['music']][:T], sig)  # noqa: E731
            ro, rx, mo, mx, lags = [], [], [], [], []
            for key in keys:
                T = seg_T[key]
                own = ms(key[0], T)
                wrong = [ms(p, T) for p in matches[cfg['match']][key]]
                rd = [smooth(d[cfg['motion']], sig) for d in real[key]]
                md = [smooth(d[cfg['motion']], sig) for d in mace[key]]
                ro.append(np.mean([sc(own, d)[1] for d in rd]))
                rx.append(np.mean([sc(w, d)[1] for d in rd for w in wrong]))
                mo.append(np.mean([sc(own, d)[1] for d in md]))
                mx.append(np.mean([sc(w, d)[1] for d in md for w in wrong]))
                lags += [sc(own, d)[2] for d in rd]
            ro, rx, mo, mx = map(np.array, (ro, rx, mo, mx))
            rng = np.random.default_rng(SEED)
            st = component_stats(ro, rx, mo, mx, rng)
            oo = ro - mo
            st.update({'own_diff': float(oo.mean()), 'own_diff_ci': boot_ci(oo, rng),
                       'own_diff_wilcoxon_p': wil(oo, 'two-sided'), 'own_diff_perm_p': signflip_p(oo, rng),
                       'n_own_real_gt_mace': int((oo > 0).sum()),
                       'real_lag_median': float(np.median(lags)),
                       'real_frac_at_lag_edge': float(np.mean(np.abs(lags) == lag)) if lag else 0.0,
                       'config': cfg})
            if vname == 'primary':
                st['per_piece'] = {k[0]: {'real_own': float(a), 'real_other': float(b), 'mace_own': float(c),
                                          'mace_other': float(d), 'n_real': len(real[k])}
                                   for k, a, b, c, d in zip(keys, ro, rx, mo, mx)}
            results[kind][vname] = st
            print(f"{kind:5s} {vname:18s} R {st['real_own']:.3f}/{st['real_other']:.3f} "
                  f"M {st['mace_own']:.3f}/{st['mace_other']:.3f} dR {st['d_real']:+.3f} (p {st['H1_perm_p']:.3f}) "
                  f"dM {st['d_mace']:+.3f} (p {st['H2_perm_p']:.3f}) G {st['G']:+.3f} "
                  f"[{st['G_ci'][0]:+.3f},{st['G_ci'][1]:+.3f}] p {st['G_perm_p']:.3f} | "
                  f"own R-M {st['own_diff']:+.3f} p {st['own_diff_perm_p']:.3f} | n+ R{st['n_real_pos']} "
                  f"M{st['n_mace_pos']}/{st['n']} edge {st['real_frac_at_lag_edge']:.2f}")

    td = [tempo_dist(nominal_bpm(w), nominal_bpm(k[0])) for k, ws in matches['tempo_first'].items() for w in ws]
    out = {'results': results, 'segments_T': {f'{p}_{k}': t for (p, k), t in seg_T.items()},
           'matching': {how: {f'{p}_{k}': ws for (p, k), ws in m.items()} for how, m in matches.items()},
           'tempo_match_summary': {'n_exact': int(sum(d == 0 for d in td)), 'n': len(td),
                                   'max_rel_dist': float(max(td)), 'tol': TEMPO_TOL},
           'genre_names': GENRE_NAMES}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print('tempo matching:', out['tempo_match_summary'])
    print('saved', OUT)


if __name__ == '__main__':
    main()
