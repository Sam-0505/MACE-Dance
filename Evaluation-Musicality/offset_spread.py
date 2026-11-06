"""Where inside the beat do dances lock, and does that offset vary across songs as in real dance? (2026-11-05)

Beat-locking guidance (mace_sampler.sample_guided) rewards locking at any offset. Check that guided dances do not all
lock at one fixed phase (e.g. accents right on the beat) while real dancers' offsets vary from song to song.

Per dance (slice x seed; FineDance test): phases of its accents (tmd_beat_metrics.accents of the 3D speed) against the
slice's beats; offset = angle of mean exp(2 pi i phase), strength = its length minus the mean length against 12 other
songs' beats (as beat_suite.py). Concentration of offsets = length of the strength-weighted mean unit vector (1 = all
at one offset, ~0 = spread), over all dances, within a song (one seed at a time; real: the same dancer) and within a slice across seeds.
(Pooling the accents of a song's slices and seeds before taking the PLV cancels the guided locking: each guided
dance locks at its own offset.)

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/offset_spread.py --models guide=mace_test_guiderr_s0.003
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from tmd_beat_metrics import accents, velocity  # noqa: E402
from tmd_beat_phase import phases  # noqa: E402

N_NULL = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=[], help='label=data_dir pairs (gen_mace_test.py outputs)')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    from motion_quality_extra import real_joints
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    song = np.array([n.split('_')[0] for n in names])
    songs = list(dict.fromkeys(song))
    beats = [np.flatnonzero(np.load(f'data/test/baseline_feats/{n}.npy')[:, 34]) for n in names]
    rng = np.random.default_rng(0)
    null = [rng.choice(np.flatnonzero(song != song[i]), N_NULL, replace=False) for i in range(len(names))]
    sets = {'real': [[j] for j in real_joints(names)]}
    for lab, d in [('released', 'mace_test'), *(x.split('=', 1) for x in args.models)]:
        J = np.load(os.path.join(HERE, 'data', d, 'joints.npy'), mmap_mode='r')
        sets[lab] = [np.asarray(J[i], dtype=np.float32) for i in range(len(names))]
    out = {}
    for lab, clips in sets.items():
        rows = []  # per dance (slice, seed): song, slice, seed, offset, strength
        for i, js in enumerate(clips):
            for s_, j in enumerate(js):
                m = accents(velocity(j, '3d'))
                z = np.exp(2j * np.pi * phases(m, beats[i])).mean()
                zo = np.mean([abs(np.exp(2j * np.pi * phases(m, beats[k])).mean()) for k in null[i]])
                rows.append((song[i], i, s_, np.angle(z) / (2 * np.pi) % 1, abs(z) - zo))
        off = np.array([r[3] for r in rows])
        st = np.array([r[4] for r in rows])
        u = np.exp(2j * np.pi * off)
        w = np.clip(st, 0, None)

        def conc(mask):
            return float(abs((w[mask] * u[mask]).sum()) / w[mask].sum()) if w[mask].sum() > 0 else np.nan

        allm = np.ones(len(rows), bool)
        sg = np.array([r[0] for r in rows])
        sl = np.array([r[1] for r in rows])
        se = np.array([r[2] for r in rows])
        # one seed at a time, so a song has as many dances as in real data (the resultant of more vectors is smaller)
        within_song = np.nanmean([conc((sg == g) & (se == k)) for g in songs for k in range(len(clips[0]))])
        within_slice = np.nanmean([conc(sl == i) for i in range(len(names))]) if len(clips[0]) > 1 else np.nan
        hist = np.histogram(off, bins=8, range=(0, 1), weights=w)[0]
        out[lab] = {'concentration_all': conc(allm), 'concentration_within_song': float(within_song),
                    'concentration_within_slice_across_seeds': float(within_slice),
                    'mean_offset': float(np.angle((w * u).sum()) / (2 * np.pi) % 1),
                    'share_locked': float((st > 0).mean()),
                    'hist8_weighted': (hist / hist.sum()).round(3).tolist()}
        r = out[lab]
        print(f"{lab:18s} dances above chance {r['share_locked']:.2f} | offset concentration (locking-weighted): "
              f"all {r['concentration_all']:.2f}, within song {r['concentration_within_song']:.2f}, "
              f"within slice across seeds {r['concentration_within_slice_across_seeds']:.2f} | "
              f"mean offset {r['mean_offset']:.2f} | hist8 {r['hist8_weighted']}", flush=True)
    with open(os.path.join(HERE, 'results', f'offset_spread{args.tag}.json'), 'w') as f:
        json.dump({'songs': songs, 'sets': out}, f, indent=1, default=float)


if __name__ == '__main__':
    main()
