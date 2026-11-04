"""Beat locking with an offset that may change along the song (2026-11-04), follow-up to tmd_beat_phase.py.

tmd_beat_phase.py pools the accent phases of a whole song, so it only credits one constant hit-to-beat offset. Here
the PLV is computed in windows of W seconds (34 s = one test slice, 17 s, 8.5 s, 4.25 s); each window has its own
offset. Delta = own-song PLV - PLV against other-song beats in the same window (short windows hold few accents,
so their chance PLV is high; the null absorbs that). Per song: mean over its windows; model: per seed, then mean.
Offset drift: for windows with own-song PLV above their own chance level, the circular distance between the
preferred phases of neighbouring windows (0 = same offset, 0.5 = opposite), real dance only (a model's phases are
not locked, so its drift is noise).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/tmd_beat_phase_local.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from tmd_beat_metrics import FPS, MODELS, accents, kin_beats, signflip, velocity  # noqa: E402
from tmd_beat_phase import phases  # noqa: E402

WINDOWS = {'34s': 1020, '17s': 510, '8.5s': 255, '4.25s': 128}
N_OTHER = 12
OUT = os.path.join(HERE, 'results', 'tmd_beat_phase_local.json')


def vec(ph):
    return np.exp(2j * np.pi * ph).mean() if len(ph) else np.nan


def window_scores(ev, A, others, w):
    """Per window: own PLV, mean other-song PLV, own preferred phase."""
    out = []
    for lo in range(0, 1020 - w + 1, w):
        m = ev[(ev >= lo) & (ev < lo + w)]
        if len(m) < 3:
            continue
        own = vec(phases(m, A))
        if np.isnan(own):
            continue
        oth = [abs(vec(phases(m, o))) for o in others]
        out.append((abs(own), float(np.nanmean(oth)), np.angle(own) / (2 * np.pi) % 1))
    return out


def main():
    from motion_quality_extra import real_joints
    rng = np.random.default_rng(0)
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    song = np.array([n.split('_')[0] for n in names])
    songs = np.unique(song)
    beats = [np.flatnonzero(np.load(f'data/test/baseline_feats/{n}.npy')[:, 34]) for n in names]
    oth_idx = {s: np.random.default_rng(int(s)).choice(np.flatnonzero(song != s), N_OTHER, replace=False)
               for s in songs}
    sets = {'real': [real_joints(names)]}
    for label, d in MODELS.items():
        J = np.load(os.path.join(HERE, 'data', d, 'joints.npy'), mmap_mode='r')
        sets[label] = [np.asarray(J[:, s], dtype=np.float32) for s in range(J.shape[1])]
    res, songwise, drift = {}, {}, {}
    for label, seeds in sets.items():
        acc = {f'{wn}_{k}': [] for wn in WINDOWS for k in ('max', 'min')}
        for js in seeds:
            v = [velocity(j, '3d') for j in js]
            ev = {'max': [accents(x) for x in v], 'min': [kin_beats(x) for x in v]}
            for wn, w in WINDOWS.items():
                for kind in ('max', 'min'):
                    per = []
                    for s in songs:
                        rows = []
                        for i in np.flatnonzero(song == s):
                            ws = window_scores(ev[kind][i], beats[i], [beats[o] for o in oth_idx[s]], w)
                            rows += ws
                            if label == 'real' and kind == 'max' and len(ws) > 1:
                                ok = [r[2] if r[0] > r[1] else None for r in ws]
                                for a, b in zip(ok[:-1], ok[1:]):
                                    if a is not None and b is not None:
                                        d = abs(a - b) % 1
                                        drift.setdefault(wn, []).append(min(d, 1 - d))
                        per.append(np.mean([r[0] - r[1] for r in rows]) if rows else np.nan)
                    acc[f'{wn}_{kind}'].append(per)
        songwise[label] = {k: np.nanmean(np.array(v), 0) for k, v in acc.items()}
        res[label] = {k: float(np.nanmean(v)) for k, v in songwise[label].items()}
        print(label, {k: round(v, 3) for k, v in res[label].items()}, flush=True)
    tests = {}
    for k in songwise['real']:
        tests[f'real {k} > 0'] = {'mean': res['real'][k], 'p': signflip(songwise['real'][k], rng),
                                  'songs_pos': int((songwise['real'][k] > 0).sum())}
        for label in MODELS:
            d = songwise[label][k] - songwise['real'][k]
            tests[f'{label} - real {k}'] = {'diff': float(np.nanmean(d)), 'p': signflip(d, rng),
                                            'songs_real_higher': int((d < 0).sum())}
            tests[f'{label} {k} > 0'] = {'mean': res[label][k], 'p': signflip(songwise[label][k], rng)}
    dr = {wn: {'median_shift': float(np.median(v)), 'share_within_0.125': float(np.mean(np.array(v) < 0.125)),
               'n_pairs': len(v), 'uniform_expectation_median': 0.25, 'uniform_share_within_0.125': 0.25}
          for wn, v in drift.items()}
    print(json.dumps(dr, indent=1))
    for t, v in tests.items():
        if ' max' in t or '_max' in t:
            print(t, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
    with open(OUT, 'w') as f:
        json.dump({'table': res, 'tests': tests, 'drift_real': dr}, f, indent=1)


if __name__ == '__main__':
    main()
