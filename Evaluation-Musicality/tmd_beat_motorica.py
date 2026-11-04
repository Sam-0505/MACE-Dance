"""Beat locking on Motorica (2026-11-04): does the FineDance result (real dance phase-locks to its own beats, MACE
does not) replicate on a second dataset with another capture setup and improvised dance?

Slices: the gen_mace_motorica.py slices (34 s), keeping only non-overlapping ones (start % 1020 == 0) so no
motion is counted twice. Real dance: the 21 BVH body joints (motorica_common.body_joints; accents use only the
timing of speed peaks, so the joint set does not matter). MACE: data/motorica/mace (released, 10 seeds) and
data/motorica/mace_local2s (+-2 s mask, 5 seeds, same noise as released seeds 0-4) if present.
Beats: the beat one-hot of af.extract_baseline on the slice audio (the beats MACE was given, as on FineDance).
Null: the beats of slices of other takes (N_OTHER per take, fixed); takes that reuse the same music would make
this null conservative.
Measures (tmd_beat_metrics / tmd_beat_phase / tmd_beat_phase_local):
  VBCS, ABHS, BAS (zero offset), own - other
  PLV whole take (one offset per take), own - other
  PLV in 34 / 17 / 8.5 / 4.25 s windows (offset free per window), own - other
Unit: take (mean over its slices; model: per seed, then mean over seeds); sign-flip tests over takes.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/tmd_beat_motorica.py
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from tmd_beat_metrics import MAIN, abhs, accents, bas, kin_beats, signflip, vbcs, velocity  # noqa: E402
from tmd_beat_phase import phases, plv  # noqa: E402
from tmd_beat_phase_local import WINDOWS, window_scores  # noqa: E402

MOT = os.path.join(HERE, 'data', 'motorica')
N_OTHER = 12
CACHE = os.path.join(MOT, 'beats_slices.npz')
OUT = os.path.join(HERE, 'results', 'tmd_beat_motorica.json')
MODELS = {'released': 'mace', 'mask_2s': 'mace_local2s'}


def slice_beats(arg):
    import audio_features as af
    from motorica_common import SLICE_LEN, wav_path
    t, o = arg
    y = af.load_audio(wav_path(t))
    f = af.extract_baseline(y[o * af.HOP_LENGTH:(o + SLICE_LEN + 1) * af.HOP_LENGTH])[:SLICE_LEN]
    return np.flatnonzero(f[:, 34])


def take_scores(ev_max, ev_min, beats, idx, oth):
    """Scores of one take: idx its slices, oth other-take slice indices."""
    r = {}
    ob = [beats[o] for o in oth]
    r['VBCS_delta'] = np.nanmean([vbcs(ev_max[i], beats[i], MAIN) - np.nanmean([vbcs(ev_max[i], b, MAIN) for b in ob])
                                  for i in idx])
    r['ABHS_delta'] = np.nanmean([abhs(ev_max[i], beats[i], MAIN) - np.nanmean([abhs(ev_max[i], b, MAIN) for b in ob])
                                  for i in idx])
    r['BAS'] = np.nanmean([bas(ev_min[i], beats[i]) for i in idx])
    r['BAS_delta'] = np.nanmean([bas(ev_min[i], beats[i]) - np.nanmean([bas(ev_min[i], b) for b in ob]) for i in idx])
    for kind, ev in (('max', ev_max), ('min', ev_min)):
        own = plv(np.concatenate([phases(ev[i], beats[i]) for i in idx]))
        other = np.mean([plv(np.concatenate([phases(ev[i], beats[o]) for i in idx])) for o in oth])
        r[f'plv_take_{kind}'] = own
        r[f'plv_take_{kind}_delta'] = own - other
    for wn, w in WINDOWS.items():
        rows = sum((window_scores(ev_max[i], beats[i], ob, w) for i in idx), [])
        r[f'plv_{wn}_delta'] = np.mean([a - b for a, b, _ in rows]) if rows else np.nan
    return r


def main():
    from motorica_common import body_joints
    rng = np.random.default_rng(0)
    meta = json.load(open(os.path.join(MOT, 'mace', 'slices.json')))
    keep = [k for k, (t, o) in enumerate(meta['slices']) if o % 1020 == 0]
    slices = [tuple(meta['slices'][k]) for k in keep]
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        assert [tuple(s) for s in z['slices'].tolist()] == slices
        beats = list(z['beats'])
    else:
        with Pool(8) as p:
            beats = p.map(slice_beats, slices)
        np.savez(CACHE, slices=np.array(slices, dtype=object), beats=np.array(beats, dtype=object))
    take = np.array([t for t, _ in slices])
    takes = list(dict.fromkeys(take))
    oth = {t: np.random.default_rng(k).choice(np.flatnonzero(take != t), N_OTHER, replace=False)
           for k, t in enumerate(takes)}
    print(len(slices), 'non-overlapping slices,', len(takes), 'takes,',
          f'{np.mean([len(b) for b in beats]) / 34:.2f} beats/s', flush=True)
    cache_j = {}
    real = []
    for t, o in slices:
        if t not in cache_j:
            cache_j = {t: body_joints(t)}
        real.append(cache_j[t][o:o + 1020].astype(np.float32))
    sets = {'real': [real]}
    for label, d in MODELS.items():
        p = os.path.join(MOT, d, 'joints.npy')
        if not os.path.exists(p):
            print('missing', p)
            continue
        J = np.load(p, mmap_mode='r')
        assert json.load(open(os.path.join(MOT, d, 'slices.json')))['slices'] == meta['slices']
        sets[label] = [[np.asarray(J[k, s], dtype=np.float32) for k in keep] for s in range(J.shape[1])]
    res, takewise = {}, {}
    for label, seeds in sets.items():
        per_seed = []
        for js in seeds:
            v = [velocity(j, '3d') for j in js]
            emax, emin = [accents(x) for x in v], [kin_beats(x) for x in v]
            rows = [take_scores(emax, emin, beats, np.flatnonzero(take == t), oth[t]) for t in takes]
            per_seed.append({k: np.array([r[k] for r in rows]) for k in rows[0]})
        takewise[label] = {k: np.nanmean([p[k] for p in per_seed], 0) for k in per_seed[0]}
        res[label] = {k: float(np.nanmean(v)) for k, v in takewise[label].items()}
        res[label]['accent_rate'] = float(np.mean([len(accents(velocity(j, '3d'))) / 34 for j in seeds[0]]))
        print(label, f'({len(seeds)} seeds)', {k: round(v, 3) for k, v in res[label].items()}, flush=True)
    tests = {}
    for label in sets:
        for k, v in takewise[label].items():
            if k.endswith('delta'):
                tests[f'{label} {k} > 0'] = {'mean': float(np.nanmean(v)), 'p': signflip(v, rng),
                                            'takes_pos': int((v > 0).sum()), 'n': int((~np.isnan(v)).sum())}
    for label in [m for m in MODELS if m in sets]:
        for k in takewise[label]:
            d = takewise[label][k] - takewise['real'][k]
            tests[f'{label} - real {k}'] = {'diff': float(np.nanmean(d)), 'p': signflip(d, rng),
                                            'takes_real_higher': int((d < 0).sum())}
    if 'mask_2s' in sets:
        for k in takewise['mask_2s']:
            d = takewise['mask_2s'][k] - takewise['released'][k]
            tests[f'mask_2s - released {k}'] = {'diff': float(np.nanmean(d)), 'p': signflip(d, rng)}
    for t, v in tests.items():
        print(t, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
    with open(OUT, 'w') as f:
        json.dump({'n_slices': len(slices), 'n_takes': len(takes), 'table': res, 'tests': tests,
                   'per_take': {lab: {k: v.tolist() for k, v in tw.items()} for lab, tw in takewise.items()},
                   'takes': takes}, f, indent=1)


if __name__ == '__main__':
    main()
