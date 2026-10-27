"""EAS on ImperialDance short clips: expertise levels vs. MACE on the same music.

    cd Expert-Motion && .venv/bin/python ../Evaluation-Musicality/gen_mace_imperial.py
    cd ../Evaluation-Musicality && ../Expert-Motion/.venv/bin/python eval_imperial_eas.py

The music start inside each recording is unknown to within ~1-2 s, so every group gets the
same treatment. Each group is one of: Expert, Intermediate, Beginner, MACE, or a control where
dances are paired with another song. The group's music offset is searched within +/- SEARCH s of
the nominal start, on even-indexed repeats; EAS is then reported on odd-indexed repeats at that
offset. Any inflation from picking the best offset is therefore shared by all groups, and the
control shows its size.

Every music window lies fully inside the music (no silence padding: the music/silence step
correlated with almost any dance and inflated the controls), and all songs use one common
window length so that groups and controls are comparable.
"""
import argparse
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter as G
from scipy.stats import mannwhitneyu

import audio_features as af
import metrics as M
from eval_imperial import load_motion
from imperial_subset import SHORT_CLIPS, LEVELS

SEARCH = 2.0    # +/- seconds searched around the nominal start
TRIM = 2.0      # seconds dropped from the start of every dance (capture start-up movement)


def energy_curves(joints_list):
    """z-scored, 0.5 s-smoothed motion energy per dance: (N, T)."""
    e = np.stack([G(M.motion_energy(j), M.ENERGY_SMOOTH) for j in joints_list])
    return (e - e.mean(1, keepdims=True)) / (e.std(1, keepdims=True) + 1e-8)


def eas_by_offset(rms_s, e_z, offsets):
    """Zero-lag Pearson r between music loudness windows and each dance: (N, n_offsets)."""
    L = e_z.shape[1]
    w = np.stack([rms_s[o:o + L] for o in offsets])
    w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-8)
    return e_z @ w.T / L


def searched_eas(rms_s, joints, nominal, length):
    """Pick the offset on even dances, score odd dances there. Returns (scores, offset_s, curve)."""
    e_z = energy_curves(joints[:, :length])
    s = int(SEARCH * M.FPS)
    offsets = [o for o in range(nominal - s, nominal + s + 1) if 0 <= o <= len(rms_s) - length]
    r = eas_by_offset(rms_s, e_z, offsets)
    best = int(np.argmax(r[0::2].mean(0)))
    return r[1::2, best], (offsets[best] - nominal) / M.FPS, r[1::2].mean(0)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--motion_dir', default=os.path.join(here, 'data', 'imperial', 'motion'))
    ap.add_argument('--audio_dir', default=os.path.join(here, 'data', 'imperial', 'short_audio'))
    ap.add_argument('--mace_dir', default=os.path.join(here, 'data', 'imperial', 'mace'))
    ap.add_argument('--out', default=os.path.join(here, 'results', 'imperial_eas.json'))
    args = ap.parse_args()

    trim, s = int(TRIM * M.FPS), int(SEARCH * M.FPS)
    rms, groups = {}, {}
    for c in SHORT_CLIPS:
        y = af.load_audio(os.path.join(args.audio_dir, c + '.wav'))
        rms[c] = G(af.rms_envelope(y), M.ENERGY_SMOOTH)
        groups[c] = {LEVELS[l]: load_motion(args.motion_dir, l, c)[:, trim:] for l in LEVELS}
        groups[c]['MACE'] = np.load(os.path.join(args.mace_dir, c + '.npy'))[:, trim:]

    names = list(LEVELS.values()) + ['MACE']
    nominal = trim   # dance frame 0 (after trim) sits here in the music when the recording starts with it
    # common window: fits inside every song for every searched offset, and inside every dance
    length = min(min(len(r) for r in rms.values()) - nominal - s,
                 min(g.shape[1] for gs in groups.values() for g in gs.values()))
    results = {}
    print(f'EAS (zero-lag r, held-out half) over a {length / M.FPS:.1f} s window, music offset searched within '
          f'+/-{SEARCH:g} s; control = same dances vs the other songs\n')
    for c in SHORT_CLIPS:
        clip = rms[c]
        cv = float(clip.std() / clip.mean())
        res = {'loudness_cv': cv}
        print(f'{c} ({SHORT_CLIPS[c]["genre"]}), loudness variation {cv:.2f}' + ('  <- nearly flat' if cv < 0.1 else ''))
        for g in names:
            own, off, _ = searched_eas(rms[c], groups[c][g], nominal, length)
            ctrl = np.concatenate([searched_eas(rms[o], groups[c][g], nominal, length)[0] for o in SHORT_CLIPS if o != c])
            res[g] = {'eas': own.tolist(), 'mean': float(own.mean()), 'offset_s': off,
                      'control_mean': float(np.nanmean(ctrl)), 'n': int(len(own))}
            print(f'  {g:<13} EAS {own.mean():+.3f} ± {own.std():.3f} (n={len(own):<3}) offset {off:+.2f}s '
                  f'| control {np.nanmean(ctrl):+.3f} | gain {own.mean() - np.nanmean(ctrl):+.3f}')
        for g in names[1:]:
            p = mannwhitneyu(res['Expert']['eas'], res[g]['eas'], alternative='greater').pvalue
            res[f'p_Expert_gt_{g}'] = float(p)
        print('  Expert > ' + ' | '.join(f'{g} p={res[f"p_Expert_gt_{g}"]:.2g}' for g in names[1:]) + '\n')
        results[c] = res

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print('saved', args.out)


if __name__ == '__main__':
    main()
