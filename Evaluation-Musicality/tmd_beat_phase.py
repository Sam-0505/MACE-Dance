"""Follow-up to tmd_beat_metrics.py (2026-11-04): beat locking at ANY constant offset, and tempo following.

VBCS / ABHS ask whether motion accents fall ON the beat (zero offset). On FineDance the real dance scores at chance
there (own-song minus other-song ~ 0). A dancer can still be locked to the beat with a fixed offset (accent = peak
speed between two hits, the stop on the beat), so:
  phase  : every motion event between beats a_k <= m < a_{k+1} gets phase phi = (m - a_k) / (a_{k+1} - a_k);
           PLV = |mean exp(2 pi i phi)| over all events of one song (pooled over its slices; for a model, one seed
           at a time, then averaged over seeds, so real and model PLVs pool the same number of clips).
           Events: velocity maxima (TMD accents) and velocity minima (the authors' BAS kinematic beats).
           Null: the same events against the beats of other-song slices; Delta = own - other.
  tempo  : across the 84 slices, r(beat rate, accent rate), real vs model.
  spread : CSD / HSD as in TMD-Bench, the SD over clips of VBCS / ABHS, computed per seed (one dance per clip, like
           the real data) and averaged over seeds.
Same settings as tmd_beat_metrics.py (smoothing 5 frames, sigma = tau = 3 frames, 3D joints).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/tmd_beat_phase.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from tmd_beat_metrics import FPS, MAIN, MODELS, abhs, accents, kin_beats, signflip, vbcs, velocity  # noqa: E402

OUT = os.path.join(HERE, 'results', 'tmd_beat_phase.json')


def phases(M, A):
    if len(A) < 2:
        return np.empty(0)
    k = np.searchsorted(A, M, side='right') - 1
    ok = (k >= 0) & (k < len(A) - 1)
    k, m = k[ok], M[ok]
    return (m - A[k]) / (A[k + 1] - A[k])


def plv(ph):
    return float(np.abs(np.exp(2j * np.pi * ph).mean())) if len(ph) else np.nan


def song_plv(events, beats, idx, other_idx):
    """events/beats per slice; idx: slices of the song; other_idx: slices of other songs (beats cropped to length)."""
    own = np.concatenate([phases(events[i], beats[i]) for i in idx])
    oth = [plv(np.concatenate([phases(events[i], beats[o]) for i in idx])) for o in other_idx]
    return plv(own), float(np.mean(oth)), float(np.angle(np.exp(2j * np.pi * own).mean()) / (2 * np.pi) % 1)


def main():
    from motion_quality_extra import real_joints
    rng = np.random.default_rng(0)
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    song = np.array([n.split('_')[0] for n in names])
    songs = np.unique(song)
    beats = [np.flatnonzero(np.load(f'data/test/baseline_feats/{n}.npy')[:, 34]) for n in names]
    beat_rate = np.array([len(b) / (1020 / FPS) for b in beats])
    real = real_joints(names)
    sets = {'real': [real]}
    for label, d in MODELS.items():
        J = np.load(os.path.join(HERE, 'data', d, 'joints.npy'), mmap_mode='r')
        sets[label] = [np.asarray(J[:, s], dtype=np.float32) for s in range(J.shape[1])]
    res, songwise = {}, {}
    for label, seeds in sets.items():
        acc = {'max': [], 'min': []}
        per_seed = []
        for js in seeds:  # js (84, T, 24, 3): one dance per slice
            v = [velocity(j, '3d') for j in js]
            ev = {'max': [accents(x) for x in v], 'min': [kin_beats(x) for x in v]}
            row = {}
            for kind in ('max', 'min'):
                own, oth, pref = [], [], []
                for s in songs:
                    idx = np.flatnonzero(song == s)
                    # 12 other-song slices per song keeps the null cheap; fixed choice for all labels
                    o = np.random.default_rng(int(s)).choice(np.flatnonzero(song != s), 12, replace=False)
                    a, b, c = song_plv(ev[kind], beats, idx, o)
                    own.append(a); oth.append(b); pref.append(c)
                row[f'plv_{kind}'] = np.array(own)
                row[f'plv_{kind}_delta'] = np.array(own) - np.array(oth)
                row[f'phase_{kind}'] = np.array(pref)
            rate = np.array([len(e) / (1020 / FPS) for e in ev['max']])
            row['tempo_r'] = float(np.corrcoef(beat_rate, rate)[0, 1])
            vb = np.array([vbcs(ev['max'][i], beats[i], MAIN) for i in range(len(names))])
            ab = np.array([abhs(ev['max'][i], beats[i], MAIN) for i in range(len(names))])
            row['CSD'], row['HSD'] = float(np.nanstd(vb)), float(np.nanstd(ab))
            per_seed.append(row)
        sw = {k: np.mean([r[k] for r in per_seed], 0) for k in per_seed[0] if k.startswith('plv')}
        songwise[label] = sw
        res[label] = {k: float(np.mean(v)) for k, v in sw.items()}
        for k in ('tempo_r', 'CSD', 'HSD'):
            res[label][k] = float(np.mean([r[k] for r in per_seed]))
        # preferred phase: circular mean over songs of seed 0
        for kind in ('max', 'min'):
            ph = per_seed[0][f'phase_{kind}']
            res[label][f'phase_{kind}_circmean'] = float(np.angle(np.exp(2j * np.pi * ph).mean()) / (2 * np.pi) % 1)
            res[label][f'phase_{kind}_consistency'] = float(np.abs(np.exp(2j * np.pi * ph).mean()))
        print(label, {k: round(v, 3) for k, v in res[label].items()}, flush=True)
    tests = {}
    for label in sets:
        for k in ('plv_max_delta', 'plv_min_delta'):
            tests[f'{label} {k} > 0'] = {'mean': float(songwise[label][k].mean()), 'p': signflip(songwise[label][k], rng)}
    for label in MODELS:
        for k in ('plv_max', 'plv_max_delta', 'plv_min', 'plv_min_delta'):
            d = songwise[label][k] - songwise['real'][k]
            tests[f'{label} - real {k}'] = {'diff': float(d.mean()), 'p': signflip(d, rng)}
    for t, v in tests.items():
        print(t, v)
    with open(OUT, 'w') as f:
        json.dump({'table': res, 'tests': tests,
                   'per_song': {lab: {k: v.tolist() for k, v in sw.items()} for lab, sw in songwise.items()}}, f,
                  indent=1)


if __name__ == '__main__':
    main()
