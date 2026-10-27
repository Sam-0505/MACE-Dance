"""Is percussion <-> upper-body acceleration distinct from loudness <-> energy?

Fixed before running (2026-10-27). Same Delta / G procedure as the loudness analyses (lag-max Pearson r within
+-9 frames, own vs 3 tempo-first matched wrong pieces, units = songs/pieces), on three datasets with the pools,
matching and wrong-slice pairing of the loudness runs (read from their results JSONs, same seeded draws):
  test      FineDance test, 19 songs, clean release motion (robustness_loud.py --gt release), real + MACE
  release   FineDance non-test songs, 150 pieces (finedance_release_loud.py), real only
  motorica  Motorica, 52 pieces (motorica_loudness.py), real + MACE
Signals per 34 s slice, smoothed after slicing:
  music  perc = RMS of the HPSS percussive part, loud = RMS (both 30 fps)
  motion uacc = mean |2nd difference| of the upper-body joints, energy = whole-body motion energy
Variants (music <-> motion):
  primary            perc s2         <-> uacc s2           (the multicomponent.py component)
  loud_same_scale    loud s2         <-> uacc s2           (loudness in place of percussion)
  slow               perc s15        <-> uacc s15
  fast               perc s2 - s15   <-> uacc s2 - s15     (slow part removed from both)
  fast_loud          loud s2 - s15   <-> uacc s2 - s15
  perc_partial_loud  perc s2 with loud s2 regressed out (per slice)      <-> uacc s2
  both_partial       perc s2 with loud s2 regressed out <-> uacc s2 with energy s2 regressed out
Head-to-head (per piece, two-sided sign-flip + Wilcoxon): Delta(primary) - Delta(loud_same_scale) and
Delta(fast) - Delta(fast_loud).
Reading rule: percussion is a distinct finding only if Delta_real stays > 0 (p < .05) in perc_partial_loud or
fast on the datasets with enough units (release, motorica); otherwise report it as part of the loudness effect.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/perc_distinct.py [test|release|motorica ...]
"""
import json
import os
import sys

import librosa
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from model.metric import SMPLSkeleton  # noqa: E402
import audio_features as af  # noqa: E402
import metrics as M  # noqa: E402
from multicomponent import SEED, GT_DIR, MACE_DIR, UPPER, smooth, component_stats, boot_ci, wil, signflip_p  # noqa: E402
from finedance_common import RELEASE, SLICE_LEN, SLICE_STRIDE, release_joints  # noqa: E402
import motorica_common as MC  # noqa: E402

OUT = os.path.join(HERE, 'results', 'perc_distinct.json')
LAG, FAST, SLOW = 9, 2, 15
UPPER_MOTORICA = [MC.BODY.index(b) for b in ('Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm',
                                             'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand')]
VARIANTS = {'primary': ('perc', 'uacc'), 'loud_same_scale': ('loud', 'uacc'), 'slow': ('perc_slow', 'uacc_slow'),
            'fast': ('perc_fast', 'uacc_fast'), 'fast_loud': ('loud_fast', 'uacc_fast'),
            'perc_partial_loud': ('perc_x_loud', 'uacc'), 'both_partial': ('perc_x_loud', 'uacc_x_energy')}
HEAD_TO_HEAD = {'perc_vs_loud': ('primary', 'loud_same_scale'), 'perc_vs_loud_fast': ('fast', 'fast_loud')}


def resid(y, x):
    X = np.stack([x, np.ones_like(x)], 1)
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def full_music(path):
    y = af.load_audio(path)
    return {'perc': librosa.feature.rms(y=librosa.effects.percussive(y), hop_length=af.HOP_LENGTH)[0],
            'loud': af.rms_envelope(y)}


def music_views(m, o):
    p, l = (np.asarray(m[k][o:o + SLICE_LEN], dtype=np.float64) for k in ('perc', 'loud'))
    p2, l2 = smooth(p, FAST), smooth(l, FAST)
    p15, l15 = smooth(p, SLOW), smooth(l, SLOW)
    return {'perc': p2, 'loud': l2, 'perc_slow': p15, 'perc_fast': p2 - p15, 'loud_fast': l2 - l15,
            'perc_x_loud': resid(p2, l2)}


def motion_views(j, upper):
    j = np.asarray(j[:SLICE_LEN], dtype=np.float64)
    a = np.linalg.norm(j[2:, upper] - 2 * j[1:-1, upper] + j[:-2, upper], axis=2).mean(1)
    a2, a15 = smooth(np.pad(a, (1, 1), mode='edge'), FAST), smooth(np.pad(a, (1, 1), mode='edge'), SLOW)
    return {'uacc': a2, 'uacc_slow': a15, 'uacc_fast': a2 - a15,
            'uacc_x_energy': resid(a2, smooth(M.motion_energy(j), FAST))}


# ----------------------------------------------------------------------------- datasets
# Each loader returns units (per slice), own music view, wrong music views (3 per slice), real and MACE motion views.
# fm(path) -> full-song music arrays; mv(arrays, offset) -> music view; dv(joints, source) -> motion view, where
# source is 'release' (FineDance release FK), 'mace' (MACE output) or 'motorica' (BVH body joints).
def load_test(smpl, device, fm, mv, dv):
    names = json.load(open(os.path.join(MACE_DIR, 'names.json')))['names']
    mace = np.load(os.path.join(MACE_DIR, 'joints.npy')).astype(np.float32)
    songs = np.array([n.split('_')[0] for n in names])
    mt = json.load(open(os.path.join(HERE, 'results', 'robustness_loud_release.json')))['matching']['tempo_first']
    rng = np.random.default_rng(SEED)
    pair = [[int(rng.choice(np.flatnonzero(songs == o['song']))) for o in mt[songs[i]]] for i in range(len(names))]
    mus = [mv(fm(os.path.join(GT_DIR, 'wavs_sliced', n + '.wav')), 0) for n in names]
    full = {s: release_joints(s, smpl, device) for s in sorted(set(songs))}
    real = [dv(full[s][SLICE_STRIDE * int(n.split('_slice')[1]):], 'release') for s, n in zip(songs, names)]
    mace_v = [[dv(x, 'mace') for x in row] for row in mace]
    return songs, mus, [[mus[k] for k in p] for p in pair], real, mace_v


def load_release(smpl, device, fm, mv, dv):
    rel = json.load(open(os.path.join(HERE, 'results', 'finedance_release_loud.json')))
    piece_of = {s: p for p, ss in rel['pieces'].items() for s in ss}
    music = {p: fm(os.path.join(RELEASE, 'music_wav', p + '.wav')) for p in rel['pieces']}
    slices, real = [], []
    for s in rel['pool']:
        j = release_joints(s, smpl, device)
        for o in range(0, min(len(j), len(music[piece_of[s]]['loud'])) - SLICE_LEN + 1, SLICE_STRIDE):
            slices.append((s, o))
            real.append(dv(j[o:o + SLICE_LEN], 'release'))
    assert len(slices) == rel['n_slices'], (len(slices), rel['n_slices'])
    units = np.array([piece_of[s] for s, _ in slices])
    offs = {p: sorted({o for (s, o), q in zip(slices, units) if q == p}) for p in rel['pieces']}
    rng = np.random.default_rng(SEED)
    pair = [[(w, int(rng.choice(offs[w]))) for w in rel['matching']['tempo_first'][piece_of[s]]] for s, _ in slices]
    cache = {}

    def view(p, o):
        if (p, o) not in cache:
            cache[(p, o)] = mv(music[p], o)
        return cache[(p, o)]
    return units, [view(piece_of[s], o) for s, o in slices], [[view(*w) for w in pr] for pr in pair], real, None


def load_motorica(fm, mv, dv):
    mot = json.load(open(os.path.join(HERE, 'results', 'motorica_loudness.json')))
    group = {t: p for p, ts in mot['pieces'].items() for t in ts}
    meta = json.load(open(os.path.join(os.path.dirname(MC.ROOT), 'mace', 'slices.json')))
    mace_all = np.load(os.path.join(os.path.dirname(MC.ROOT), 'mace', 'joints.npy'), mmap_mode='r')
    mace_idx = {(t, o): i for i, (t, o) in enumerate(meta['slices'])}
    music = {t: fm(MC.wav_path(t)) for t in mot['takes']}
    slices, real, mace_v = [], [], []
    for t in mot['takes']:
        j = MC.body_joints(t)
        for o in MC.slice_starts(t):
            slices.append((t, o))
            real.append(dv(j[o:o + SLICE_LEN], 'motorica'))
            mace_v.append([dv(x.astype(np.float32), 'mace') for x in mace_all[mace_idx[(t, o)]]])
    assert len(slices) == mot['n_slices']
    units = np.array([group[t] for t, _ in slices])
    piece_slices = {p: [s for s, q in zip(slices, units) if q == p] for p in mot['pieces']}
    rng = np.random.default_rng(SEED)
    pair = [[piece_slices[w][int(rng.integers(len(piece_slices[w])))] for w in mot['matching']['tempo_first'][group[t]]]
            for t, _ in slices]
    cache = {}

    def view(t, o):
        if (t, o) not in cache:
            cache[(t, o)] = mv(music[t], o)
        return cache[(t, o)]
    return units, [view(t, o) for t, o in slices], [[view(*w) for w in pr] for pr in pair], real, mace_v


# ----------------------------------------------------------------------------- analysis
def score_pairs(units, own, wrong, real, mace, pairs):
    """pairs: {name: (music key, motion key)}. Returns per-pair stats and per-piece Deltas ({'real', 'mace'})."""
    pieces = sorted(set(units))
    per_piece = lambda a: np.array([np.mean(np.asarray(a)[units == p]) for p in pieces])  # noqa: E731
    sc = lambda a, b: M.lagged_corr(a, b, LAG)[1]  # noqa: E731
    out, deltas = {}, {}
    for vname, (mk, dk) in pairs.items():
        ro = [sc(own[i][mk], real[i][dk]) for i in range(len(own))]
        rx = [np.mean([sc(w[mk], real[i][dk]) for w in wrong[i]]) for i in range(len(own))]
        v = {'ro': per_piece(ro), 'rx': per_piece(rx)}
        rng = np.random.default_rng(SEED)
        if mace is not None:
            mo = [np.mean([sc(own[i][mk], x[dk]) for x in mace[i]]) for i in range(len(own))]
            mx = [np.mean([sc(w[mk], x[dk]) for x in mace[i] for w in wrong[i]]) for i in range(len(own))]
            v.update(mo=per_piece(mo), mx=per_piece(mx))
            st = component_stats(v['ro'], v['rx'], v['mo'], v['mx'], rng)
            deltas[vname] = {'real': v['ro'] - v['rx'], 'mace': v['mo'] - v['mx']}
        else:
            dr = v['ro'] - v['rx']
            st = {'real_own': float(v['ro'].mean()), 'real_other': float(v['rx'].mean()), 'd_real': float(dr.mean()),
                  'd_real_ci': boot_ci(dr, rng), 'H1_wilcoxon_p': wil(dr, 'greater'),
                  'H1_perm_p': signflip_p(dr, rng, 'greater'), 'n_real_pos': int((dr > 0).sum()), 'n': len(dr)}
            deltas[vname] = {'real': dr}
        st['signals'] = [mk, dk]
        out[vname] = st
        line = (f"  {vname:18s} R own {st['real_own']:.3f} other {st['real_other']:.3f} dR {st['d_real']:+.3f} "
                f"[{st['d_real_ci'][0]:+.3f},{st['d_real_ci'][1]:+.3f}] p {st['H1_perm_p']:.4f} n+ {st['n_real_pos']}/{st['n']}")
        if mace is not None:
            line += (f" | dM {st['d_mace']:+.3f} p {st['H2_perm_p']:.3f} | G {st['G']:+.3f} "
                     f"[{st['G_ci'][0]:+.3f},{st['G_ci'][1]:+.3f}] p {st['G_perm_p']:.4f}")
        print(line, flush=True)
    return out, deltas


def analyse(units, own, wrong, real, mace):
    out, deltas = score_pairs(units, own, wrong, real, mace, VARIANTS)
    h2h = {}
    for hname, (a, b) in HEAD_TO_HEAD.items():
        h2h[hname] = {}
        for who in deltas[a]:
            d = deltas[a][who] - deltas[b][who]
            rng = np.random.default_rng(SEED)
            h2h[hname][who] = {'mean': float(d.mean()), 'ci': boot_ci(d, rng), 'perm_p': signflip_p(d, rng),
                               'wilcoxon_p': wil(d, 'two-sided'), 'n_pos': int((d > 0).sum()), 'n': len(d)}
            print(f"  {hname:18s} {who:4s} dDelta {d.mean():+.3f} [{h2h[hname][who]['ci'][0]:+.3f},"
                  f"{h2h[hname][who]['ci'][1]:+.3f}] p {h2h[hname][who]['perm_p']:.4f} n+ {(d > 0).sum()}/{len(d)}")
    return {'variants': out, 'head_to_head': h2h}


def load_dataset(name, smpl, device, fm, mv, dv):
    return {'test': lambda: load_test(smpl, device, fm, mv, dv), 'release': lambda: load_release(smpl, device, fm, mv, dv),
            'motorica': lambda: load_motorica(fm, mv, dv)}[name]()


def main():
    which = [a for a in sys.argv[1:] if a in ('test', 'release', 'motorica')] or ['test', 'release', 'motorica']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(device)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in which:
        print(name, flush=True)
        dv = lambda j, src: motion_views(j, UPPER_MOTORICA if src == 'motorica' else UPPER)  # noqa: E731
        data = load_dataset(name, smpl, device, full_music, music_views, dv)
        results[name] = analyse(*data)
        with open(OUT, 'w') as f:
            json.dump(results, f, indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
