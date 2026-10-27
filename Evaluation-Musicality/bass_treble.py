"""Bass and treble: does each drive its own body part? A 2x2 crossover on top of the Delta procedure.

Fixed before running (2026-10-27). Datasets, pools, tempo-first wrong pieces and slice pairing as in
perc_distinct.py (test with MACE, release real only, motorica with MACE). Per slice, smoothed after slicing (s = 2):
  music  bass   = half-wave-rectified spectral flux 50-200 Hz      (multicomponent.py lowflux)
         treble = half-wave-rectified spectral flux 6.4-12.8 kHz   (multicomponent.py highflux)
  motion legs   = mean speed of pelvis, hips, knees, ankles, feet  } in travel-free coordinates: the pelvis's
         hands  = mean speed of head, wrists (and SMPL hands)      } horizontal position is subtracted from every
                                                                     joint, vertical movement (bounce) is kept
Cells: Delta(music, motion) = S_own - S_matched_wrong, per piece. Crossover statistics per piece:
  bass_pref   = Delta(bass, legs)    - Delta(bass, hands)
  treble_pref = Delta(treble, hands) - Delta(treble, legs)
  I           = bass_pref + treble_pref           (general intensity cancels: it raises all four cells)
Body-part specificity counts only if I_real > 0 (one-sided sign-flip p < .05) AND both bass_pref and treble_pref
are > 0 on average (a uniformly noisier body part can fake I, but not both preferences at once).
Variants: world (world coordinates, the multicomponent.py definitions), legs_vertical (legs = vertical speed only,
Burger et al. 2018), bass_50_100 (Burger et al. 2013 sub-band 2), partial_loud (loudness regressed out of both flux
signals per slice).
Literature (read 2026-10-27): Burger et al. 2013, 30 clips, song level: 50-100 Hz flux ~ head speed (r = .73; feet
.31 n.s.), 6.4-12.8 kHz flux ~ hand (.65) and head (.64) speed. Burger et al. 2018: strong low-band flux tightens
beat-level synchronisation of vertical hip and foot movement. Neither tests within-song coupling.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/bass_treble.py [test|release|motorica ...]
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
import motorica_common as MC  # noqa: E402
from multicomponent import (SEED, FLUX_SR, FLUX_HOP, FLUX_NFFT, LOW_BAND, HIGH_BAND, LOWER, HEADHAND,  # noqa: E402
                            band_flux, smooth, boot_ci, wil, signflip_p)
from finedance_common import SLICE_LEN  # noqa: E402
from perc_distinct import FAST, resid, score_pairs, load_dataset  # noqa: E402

OUT = os.path.join(HERE, 'results', 'bass_treble.json')
BASS_SB2 = (50, 100)
# (axis, sign) of "up" per motion source, checked on every slice with head - pelvis
UP = {'release': (0, -1), 'mace': (2, 1), 'motorica': (1, 1)}
JOINTS = {'smpl': {'legs': LOWER, 'hands': HEADHAND, 'head': 15, 'pelvis': 0},
          'motorica': {'legs': [MC.BODY.index(b) for b in ('Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
                                                           'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase')],
                       'hands': [MC.BODY.index(b) for b in ('Head', 'LeftHand', 'RightHand')],
                       'head': MC.BODY.index('Head'), 'pelvis': MC.BODY.index('Hips')}}
VARIANTS = {'primary': ('bass', 'treble', 'legs', 'hands'),
            'world': ('bass', 'treble', 'legs_world', 'hands_world'),
            'legs_vertical': ('bass', 'treble', 'legs_vert', 'hands'),
            'bass_50_100': ('bass_sb2', 'treble', 'legs', 'hands'),
            'partial_loud': ('bass_x_loud', 'treble_x_loud', 'legs', 'hands')}
UPRIGHT = {}                                       # per source: per-slice mean (head - pelvis) . up (orientation check)


def full_music(path):
    y48, _ = librosa.load(path, sr=FLUX_SR)
    S = np.abs(librosa.stft(y48, n_fft=FLUX_NFFT, hop_length=FLUX_HOP))
    freqs = librosa.fft_frequencies(sr=FLUX_SR, n_fft=FLUX_NFFT)
    return {'bass': band_flux(S, freqs, LOW_BAND), 'treble': band_flux(S, freqs, HIGH_BAND),
            'bass_sb2': band_flux(S, freqs, BASS_SB2), 'loud': af.rms_envelope(af.load_audio(path))}


def music_views(m, o):
    v = {k: smooth(np.asarray(a[o:o + SLICE_LEN], dtype=np.float64), FAST) for k, a in m.items()}
    assert all(len(a) == SLICE_LEN for a in v.values()), {k: len(a) for k, a in v.items()}
    v['bass_x_loud'], v['treble_x_loud'] = resid(v['bass'], v['loud']), resid(v['treble'], v['loud'])
    return v


def speed(j, idx):
    return smooth(np.pad(np.linalg.norm(j[1:, idx] - j[:-1, idx], axis=2).mean(1), (0, 1), mode='edge'), FAST)


def motion_views(j, src):
    J = JOINTS['motorica' if src == 'motorica' else 'smpl']
    ax, sign = UP[src]
    j = np.asarray(j[:SLICE_LEN], dtype=np.float64)
    UPRIGHT.setdefault(src, []).append(float(sign * (j[:, J['head'], ax] - j[:, J['pelvis'], ax]).mean()))
    horiz = [a for a in range(3) if a != ax]
    tf = j.copy()
    tf[:, :, horiz] -= j[:, J['pelvis']:J['pelvis'] + 1, horiz]
    vert = np.abs(np.diff(j[:, J['legs'], ax], axis=0)).mean(1)
    return {'legs': speed(tf, J['legs']), 'hands': speed(tf, J['hands']),
            'legs_world': speed(j, J['legs']), 'hands_world': speed(j, J['hands']),
            'legs_vert': smooth(np.pad(vert, (0, 1), mode='edge'), FAST)}


def one_sided(d, rng):
    return {'mean': float(d.mean()), 'ci': boot_ci(d, rng), 'perm_p': signflip_p(d, rng, 'greater'),
            'wilcoxon_p': wil(d, 'greater'), 'n_pos': int((d > 0).sum()), 'n': len(d)}


def analyse(units, own, wrong, real, mace):
    cells = {f'{m}|{d}': (m, d) for b, t, l, h in VARIANTS.values() for m in (b, t) for d in (l, h)}
    stats, deltas = score_pairs(units, own, wrong, real, mace, cells)
    out = {}
    for vname, (b, t, l, h) in VARIANTS.items():
        out[vname] = {'cells': {f'{mk}|{dk}': {k: stats[f'{m}|{d}'][k] for k in ('d_real', 'd_mace') if k in stats[f'{m}|{d}']}
                                for mk, m in (('bass', b), ('treble', t)) for dk, d in (('legs', l), ('hands', h))}}
        for who in deltas[f'{b}|{l}']:
            D = lambda m, d: deltas[f'{m}|{d}'][who]  # noqa: E731
            bp, tp = D(b, l) - D(b, h), D(t, h) - D(t, l)
            rng = np.random.default_rng(SEED)
            out[vname][who] = {'I': one_sided(bp + tp, rng), 'bass_pref': one_sided(bp, rng),
                               'treble_pref': one_sided(tp, rng)}
            r = out[vname][who]
            r['specific'] = bool(r['I']['perm_p'] < .05 and bp.mean() > 0 and tp.mean() > 0)
        if 'mace' in out[vname]:
            g = (deltas[f'{b}|{l}']['real'] - deltas[f'{b}|{h}']['real'] - deltas[f'{t}|{l}']['real'] + deltas[f'{t}|{h}']['real']
                 - deltas[f'{b}|{l}']['mace'] + deltas[f'{b}|{h}']['mace'] + deltas[f'{t}|{l}']['mace'] - deltas[f'{t}|{h}']['mace'])
            rng = np.random.default_rng(SEED)
            out[vname]['I_real_minus_mace'] = {'mean': float(g.mean()), 'ci': boot_ci(g, rng),
                                               'perm_p': signflip_p(g, rng), 'wilcoxon_p': wil(g, 'two-sided')}
        for who in ('real', 'mace'):
            if who in out[vname]:
                r = out[vname][who]
                print(f"  {vname:14s} {who:4s} I {r['I']['mean']:+.3f} [{r['I']['ci'][0]:+.3f},{r['I']['ci'][1]:+.3f}] "
                      f"p {r['I']['perm_p']:.4f} | bass_pref {r['bass_pref']['mean']:+.3f} (p {r['bass_pref']['perm_p']:.3f})"
                      f" | treble_pref {r['treble_pref']['mean']:+.3f} (p {r['treble_pref']['perm_p']:.3f}) | "
                      f"specific {r['specific']}", flush=True)
    return {'variants': out, 'cell_stats': stats}


def main():
    which = [a for a in sys.argv[1:] if a in ('test', 'release', 'motorica')] or ['test', 'release', 'motorica']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(device)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in which:
        print(name, flush=True)
        UPRIGHT.clear()
        data = load_dataset(name, smpl, device, full_music, music_views, motion_views)
        upright = {}
        for src, up in UPRIGHT.items():
            up = np.array(up)
            upright[src] = {'median': float(np.median(up)), 'n_below_0.2': int((up < .2).sum()), 'n': len(up)}
            print(f'  upright check {src}: median head-pelvis along up {np.median(up):.2f} m, '
                  f'slices < 0.2 m: {(up < .2).sum()}/{len(up)}')
            assert np.median(up) > 0.4, f'wrong up axis for {src}'
        results[name] = analyse(*data)
        results[name]['upright'] = upright
        with open(OUT, 'w') as f:
            json.dump(results, f, indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
