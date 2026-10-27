"""Loudness <-> body energy on the Motorica Dance dataset: real dance vs MACE (which never saw this data).

Same test as the FineDance evaluation (robustness_loud.py / finedance_release_loud.py):
  * takes: motorica_common.pool() (audio available, motion/audio lengths within 1 s), 34 s slices every 17 s;
  * units: music pieces; takes sharing one recording (same song for two dancers or two takes, incl. small offsets
    or gain changes) are merged with finedance_release_loud.music_groups and never used as each other's wrong song;
  * own music = the take's own wav; wrong songs = 3 tempo-first matched pieces, one seeded-random slice each;
  * real motion = 21 body joints from the BVH (fingers dropped), m, 30 fps; MACE = 10 samples per slice
    (gen_mace_motorica.py) on the same slice audio, scored against the same wrong slices;
  * variants: robustness_loud.VARIANTS with genre-first matching replaced by 3 random pieces of any tempo;
  * stats: multicomponent.component_stats on pieces, plus real own - MACE own and a per-style breakdown.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/motorica_loudness.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

import audio_features as af  # noqa: E402
import metrics as M  # noqa: E402
from multicomponent import SEED, smooth, component_stats, boot_ci, wil, signflip_p  # noqa: E402
from robustness_loud import PRIMARY, VARIANTS, loudness_signals, motion_signals, bpm_from_wav, matches  # noqa: E402
from finedance_release_loud import music_groups  # noqa: E402
from motorica_common import ROOT, SLICE_LEN, body_joints, pool, slice_starts, style, wav_path  # noqa: E402

OUT = os.path.join(HERE, 'results', 'motorica_loudness.json')
MOT_VARIANTS = {('match_random' if k == 'match_genre_first' else k): ({'match': 'random'} if k == 'match_genre_first'
                                                                       else v) for k, v in VARIANTS.items()}


def main():
    takes = pool()
    group = music_groups(takes, path=wav_path)
    pieces = sorted(set(group.values()))
    genre = {p: style(p) for p in pieces}
    print(f'{len(takes)} takes in {len(pieces)} pieces')

    meta = json.load(open(os.path.join(os.path.dirname(ROOT), 'mace', 'slices.json')))
    mace_all = np.load(os.path.join(os.path.dirname(ROOT), 'mace', 'joints.npy'))
    mace_idx = {(t, o): i for i, (t, o) in enumerate(meta['slices'])}

    music = {t: loudness_signals(af.load_audio(wav_path(t))) for t in takes}
    slices, real, mace = [], [], []
    for t in takes:
        j = body_joints(t)
        for o in slice_starts(t):
            slices.append((t, o))
            real.append(motion_signals(j[o:o + SLICE_LEN]))
            mace.append([motion_signals(x.astype(np.float32)) for x in mace_all[mace_idx[(t, o)]]])
    slice_piece = np.array([group[t] for t, _ in slices])
    print(len(slices), 'slices')

    bpm = {p: bpm_from_wav(wav_path(p)) for p in pieces}
    mt = {'tempo_first': matches(pieces, genre, bpm, 'tempo_first')}
    rng = np.random.default_rng(SEED)
    mt['random'] = {p: [str(x) for x in rng.choice([q for q in pieces if q != p], 3, replace=False)] for p in pieces}
    piece_slices = {p: [s for s, q in zip(slices, slice_piece) if q == p] for p in pieces}
    pairing = {}
    for how, m in mt.items():
        rng = np.random.default_rng(SEED)
        pairing[how] = [[piece_slices[w][int(rng.integers(len(piece_slices[w])))] for w in m[group[t]]]
                        for t, _ in slices]

    results = {}
    for vname, change in MOT_VARIANTS.items():
        cfg = {**PRIMARY, **change}
        sig, lag = cfg['sigma'], cfg['lag']
        ms = lambda t, o: smooth(music[t][cfg['music']][o:o + SLICE_LEN], sig)  # noqa: E731
        sc = lambda a, b: M.lagged_corr(a, b, lag)[1]  # noqa: E731
        ro, rx, mo, mx = [], [], [], []
        for i, (t, o) in enumerate(slices):
            own = ms(t, o)
            wrong = [ms(wt, wo) for wt, wo in pairing[cfg['match']][i]]
            r = smooth(real[i][cfg['motion']], sig)
            md = [smooth(x[cfg['motion']], sig) for x in mace[i]]
            ro.append(sc(own, r)); rx.append(np.mean([sc(w, r) for w in wrong]))
            mo.append(np.mean([sc(own, x) for x in md])); mx.append(np.mean([sc(w, x) for x in md for w in wrong]))
        v = {k: np.array([np.mean(np.array(a)[slice_piece == p]) for p in pieces])
             for k, a in (('ro', ro), ('rx', rx), ('mo', mo), ('mx', mx))}
        rng = np.random.default_rng(SEED)
        st = component_stats(v['ro'], v['rx'], v['mo'], v['mx'], rng)
        oo = v['ro'] - v['mo']
        st.update({'own_diff': float(oo.mean()), 'own_diff_ci': boot_ci(oo, rng),
                   'own_diff_wilcoxon_p': wil(oo, 'two-sided'), 'own_diff_perm_p': signflip_p(oo, rng),
                   'n_own_real_gt_mace': int((oo > 0).sum()), 'config': cfg})
        if vname == 'primary':
            st['per_piece'] = {p: {k: float(a[n]) for k, a in v.items()} for n, p in enumerate(pieces)}
            st['per_style'] = {}
            for g in sorted(set(genre.values())):
                sel = np.array([genre[p] == g for p in pieces])
                dr, dm = (v['ro'] - v['rx'])[sel], (v['mo'] - v['mx'])[sel]
                st['per_style'][g] = {'n': int(sel.sum()), 'd_real': float(dr.mean()), 'd_mace': float(dm.mean()),
                                      'n_real_pos': int((dr > 0).sum()), 'n_mace_pos': int((dm > 0).sum())}
        results[vname] = st
        print(f"{vname:18s} R {st['real_own']:.3f}/{st['real_other']:.3f} M {st['mace_own']:.3f}/{st['mace_other']:.3f} "
              f"dR {st['d_real']:+.3f} [{st['d_real_ci'][0]:+.3f},{st['d_real_ci'][1]:+.3f}] (p {st['H1_perm_p']:.4f}) "
              f"dM {st['d_mace']:+.3f} (p {st['H2_perm_p']:.3f}) G {st['G']:+.3f} [{st['G_ci'][0]:+.3f},"
              f"{st['G_ci'][1]:+.3f}] p {st['G_perm_p']:.4f} | own R-M {st['own_diff']:+.3f} p {st['own_diff_perm_p']:.4f}"
              f" | n+ R{st['n_real_pos']} M{st['n_mace_pos']}/{st['n']}")

    out = {'results': results, 'takes': takes, 'pieces': {p: [t for t in takes if group[t] == p] for p in pieces},
           'bpm': bpm, 'style': genre, 'matching': mt, 'n_slices': len(slices)}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print('per style (primary):', json.dumps(results['primary']['per_style']))
    print('saved', OUT)


if __name__ == '__main__':
    main()
