"""Beat metrics on three datasets with a mismatched-music chance baseline (2026-11-04).

Per clip and dance, three scores against the clip's own beats and against beats of clips of OTHER songs:
  BAS   the authors' Beat Alignment Score (model/metric.py: velocity minima, music beats, sigma^2 = 9 frames^2)
  PLV   beat locking at any constant offset: |mean exp(2 pi i phase)| of the motion accents (velocity maxima),
        phase = position of the accent inside its beat interval (tmd_beat_phase.py), one offset per clip
  BPS   PhaseDance's Beat Periodicity Score (Kim et al., NeurIPS 2026, Sec. 5.1.2): BPM from the median interval of
        velocity minima vs the median beat interval, best of harmonic ratios {1, 2, 1/2, 4, 1/4, 3, 1/3},
        BPS = exp(-e^2 / (2 * 0.2^2)) with e the relative tempo error. (They use BeatNet beats; librosa here.)
Delta = own - mean over null clips. Unit = song / piece / take: mean over its clips (and seeds); sign-flip tests.
Datasets:
  finedance  84 test slices (34 s), real = test motion, models = gen_mace_test.py outputs present on disk
  motorica   non-overlapping 34 s slices of 62 takes (tmd_beat_motorica.py), real = BVH body joints
  aist       AIST++ (aist_common): one music segment per (piece, basic/adv); every real dance on it vs the 10 MACE
             samples (gen_mace_aist.py). Two nulls: any other piece of the same kind, and other pieces with the
             SAME nominal BPM (only the phase of the beat grid differs, so a tempo score cannot tell them apart).
Influence (finedance): mean root-relative joint distance of a variant to the released sample with the same seed,
divided by the distance between two released seeds; ~0 means the change of input does nothing.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/beat_suite.py --dataset aist     # finedance | motorica | aist
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from tmd_beat_metrics import FPS, accents, bas, kin_beats, signflip, velocity  # noqa: E402
from tmd_beat_phase import phases, plv  # noqa: E402

N_NULL = 12
HARM = (1, 2, 0.5, 4, 0.25, 3, 1 / 3)
FD_MODELS = {'released': 'mace_test', 'mask_2s': 'mace_test_local2s', 'nobeat': 'mace_test_nobeat',
             'norhythm': 'mace_test_norhythm', 'beta1': 'mace_test_beta1', 'beta05': 'mace_test_beta05',
             'mask_2s_beta1': 'mace_test_local2s_beta1'}
MOT_MODELS = {'released': 'mace', 'mask_2s': 'mace_local2s'}


def bps(minima, beats):
    if len(minima) < 3 or len(beats) < 3:
        return np.nan
    bm, bb = 60 * FPS / np.median(np.diff(minima)), 60 * FPS / np.median(np.diff(beats))
    e = min(abs(bm / (h * bb) - 1) for h in HARM)
    return float(np.exp(-e ** 2 / (2 * 0.2 ** 2)))


def scores(v, A):
    M, K = accents(v), kin_beats(v)
    return np.array([bas(K, A), plv(phases(M, A)), bps(K, A)])


NAMES = ('BAS', 'PLV', 'BPS')


# ---------------------------------------------------------------- loaders
def load_finedance():
    from motion_quality_extra import real_joints
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    group = [n.split('_')[0] for n in names]
    beats = [np.flatnonzero(np.load(f'data/test/baseline_feats/{n}.npy')[:, 34]) for n in names]
    sets = {'real': [[j] for j in real_joints(names)]}
    for label, d in FD_MODELS.items():
        p = os.path.join(HERE, 'data', d, 'joints.npy')
        if os.path.exists(p):
            J = np.load(p, mmap_mode='r')
            sets[label] = [np.asarray(J[i], dtype=np.float32) for i in range(len(names))]
    return group, beats, sets, {'other': None}


def load_motorica():
    from motorica_common import body_joints
    mot = os.path.join(HERE, 'data', 'motorica')
    meta = json.load(open(os.path.join(mot, 'mace', 'slices.json')))
    keep = [k for k, (t, o) in enumerate(meta['slices']) if o % 1020 == 0]
    z = np.load(os.path.join(mot, 'beats_slices.npz'), allow_pickle=True)  # tmd_beat_motorica.py cache
    beats = list(z['beats'])
    slices = [tuple(meta['slices'][k]) for k in keep]
    assert [tuple(s) for s in z['slices'].tolist()] == slices
    group = [t for t, _ in slices]
    real, cache = [], {}
    for t, o in slices:
        if t not in cache:
            cache = {t: body_joints(t)}
        real.append([cache[t][o:o + 1020].astype(np.float32)])
    sets = {'real': real}
    for label, d in MOT_MODELS.items():
        p = os.path.join(mot, d, 'joints.npy')
        if os.path.exists(p):
            J = np.load(p, mmap_mode='r')
            if J.shape[0] == len(keep):  # generated on the kept slices only (gen_mace_motorica.py --lock_guide)
                assert [tuple(x) for x in json.load(open(os.path.join(mot, d, 'slices.json')))['slices']] == slices
                sets[label] = [np.asarray(J[i], dtype=np.float32) for i in range(len(keep))]
            else:
                sets[label] = [np.asarray(J[k], dtype=np.float32) for k in keep]
    return group, beats, sets, {'other': None}


def load_aist():
    import torch
    import audio_features as af
    from aist_common import DATA, joints, nominal_bpm, segments, sequences
    from model.metric import SMPLSkeleton
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    skel = SMPLSkeleton(dev)
    seg = segments()
    keys = sorted(seg)
    beats = {}
    for piece, kind in keys:
        T = seg[(piece, kind)]
        y = af.load_audio(os.path.join(DATA, 'music', piece + '.wav'))
        f = af.extract_baseline(y[:(T + 1) * af.HOP_LENGTH], start_bpm=nominal_bpm(piece))[:T]
        beats[(piece, kind)] = np.flatnonzero(f[:, 34])
    by_key = {}
    for n, key in sequences().items():
        by_key.setdefault(key, []).append(n)
    # one "clip" per (piece, kind); real = every dance on it (cropped to the segment), model = its 10 samples
    group, B, sets = [], [], {'real': [], 'released': []}
    for key in keys:
        T = seg[key]
        group.append(key[0] + '_' + key[1])
        B.append(beats[key])
        sets['real'].append([joints(n, skel, dev)[:T].astype(np.float32)
                             for n in by_key[key]])
        sets['released'].append(np.asarray(np.load(os.path.join(DATA, 'mace', f'{key[0]}_{key[1]}.npy')),
                                           dtype=np.float32))
    kind = [k[1] for k in keys]
    bpm = [nominal_bpm(k[0]) for k in keys]
    return group, B, sets, {'other': ('kind', kind), 'same_bpm': ('bpm', kind, bpm)}


# ---------------------------------------------------------------- scoring
def song_of(g):
    """AIST groups are '<piece>_<kind>': the null must exclude the same piece in the other kind."""
    return g.split('_')[0] if g.endswith(('_basic', '_adv')) else g


def null_sets(group, nulls, rng):
    n = len(group)
    out = {}
    for name, spec in nulls.items():
        cand = []
        for i in range(n):
            ok = [k for k in range(n) if song_of(group[k]) != song_of(group[i])]
            if spec is not None and spec[0] == 'kind':
                ok = [k for k in ok if spec[1][k] == spec[1][i]]
            if spec is not None and spec[0] == 'bpm':
                ok = [k for k in ok if spec[1][k] == spec[1][i] and spec[2][k] == spec[2][i]]
            cand.append(list(rng.choice(ok, min(N_NULL, len(ok)), replace=False)) if ok else [])
        out[name] = cand
    return out


def evaluate(group, beats, sets, nulls):
    rng = np.random.default_rng(0)
    nidx = null_sets(group, nulls, rng)
    groups = list(dict.fromkeys(group))
    gidx = {g: [i for i, x in enumerate(group) if x == g] for g in groups}
    res, per_group = {}, {}
    for label, clips in sets.items():
        own = np.full((len(clips), 3), np.nan)
        delta = {nm: np.full((len(clips), 3), np.nan) for nm in nidx}
        for i, js in enumerate(clips):
            vs = [velocity(j, '3d') for j in js]
            o = np.nanmean([scores(v, beats[i]) for v in vs], 0)
            own[i] = o
            for nm, cand in nidx.items():
                if cand[i]:
                    nl = np.nanmean([[scores(v, beats[k]) for k in cand[i]] for v in vs], (0, 1))
                    delta[nm][i] = o - nl
        g_own = np.array([np.nanmean(own[gidx[g]], 0) for g in groups])
        g_delta = {nm: np.array([np.nanmean(d[gidx[g]], 0) for g in groups]) for nm, d in delta.items()}
        per_group[label] = {'own': g_own, **{f'delta_{nm}': d for nm, d in g_delta.items()}}
        res[label] = {}
        for m, name in enumerate(NAMES):
            res[label][name] = float(np.nanmean(g_own[:, m]))
            for nm, d in g_delta.items():
                res[label][f'{name}_delta_{nm}'] = float(np.nanmean(d[:, m]))
                res[label][f'{name}_delta_{nm}_p'] = signflip(d[:, m], rng)
                res[label][f'{name}_delta_{nm}_pos'] = f'{int((d[:, m] > 0).sum())}/{int((~np.isnan(d[:, m])).sum())}'
        print(label, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[label].items()}, flush=True)
    tests = {}
    for label in sets:
        if label == 'real':
            continue
        for key in per_group[label]:
            d = per_group[label][key] - per_group['real'][key]
            for m, name in enumerate(NAMES):
                tests[f'{label} - real {name} {key}'] = {'diff': float(np.nanmean(d[:, m])), 'p': signflip(d[:, m], rng),
                                                        'groups_real_higher': int((d[:, m] < 0).sum())}
        if label != 'released' and 'released' in per_group:
            for key in per_group[label]:
                d = per_group[label][key] - per_group['released'][key]
                for m, name in enumerate(NAMES):
                    tests[f'{label} - released {name} {key}'] = {'diff': float(np.nanmean(d[:, m])),
                                                                'p': signflip(d[:, m], rng)}
    return res, tests, len(groups)


def influence(sets, n_clips=84):
    """finedance only: how much does each variant change the dance, relative to changing the seed?"""
    if 'released' not in sets:
        return {}
    R = sets['released']
    rel = lambda j: j - j[:, :, :1]  # noqa: E731
    base = np.mean([np.linalg.norm(rel(R[i])[0] - rel(R[i])[1], axis=-1).mean() for i in range(n_clips)])
    out = {}
    for label, clips in sets.items():
        if label in ('real', 'released'):
            continue
        S = min(len(clips[0]), len(R[0]))
        d = np.mean([np.linalg.norm(rel(clips[i][:S]) - rel(R[i][:S]), axis=-1).mean() for i in range(n_clips)])
        out[label] = {'dist_to_released_same_seed_m': float(d), 'dist_between_seeds_m': float(base),
                      'ratio': float(d / base)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('finedance', 'motorica', 'aist'), required=True)
    ap.add_argument('--fd_models', nargs='*', help='finedance: label=data_dir pairs to score instead of FD_MODELS '
                    '(released is always kept)')
    ap.add_argument('--mot_models', nargs='*', help='motorica: label=dir (under data/motorica) pairs to score instead '
                    'of MOT_MODELS (released is always kept)')
    ap.add_argument('--beats_shift', type=int, default=0,
                    help='placebo check: score every clip against the beats (and song) of the clip this many places '
                    'earlier (gen_mace_test.py --lock_guide_shift)')
    ap.add_argument('--tag', default='', help='suffix of the results file')
    a = ap.parse_args()
    if a.fd_models:
        FD_MODELS.clear()
        FD_MODELS.update({'released': 'mace_test', **dict(x.split('=', 1) for x in a.fd_models)})
    if a.mot_models:
        MOT_MODELS.clear()
        MOT_MODELS.update({'released': 'mace', **dict(x.split('=', 1) for x in a.mot_models)})
    group, beats, sets, nulls = {'finedance': load_finedance, 'motorica': load_motorica, 'aist': load_aist}[a.dataset]()
    if a.beats_shift:
        n, k = len(group), a.beats_shift
        group, beats = [group[(i - k) % n] for i in range(n)], [beats[(i - k) % n] for i in range(n)]
    print(a.dataset, len(group), 'clips,', len(set(group)), 'groups,', {k: len(v[0]) for k, v in sets.items()},
          'dances in first clip', flush=True)
    res, tests, ng = evaluate(group, beats, sets, nulls)
    out = {'n_groups': ng, 'table': res, 'tests': tests}
    if a.dataset == 'finedance':
        out['influence'] = influence(sets)
        print('influence', json.dumps(out['influence'], indent=1))
    for t, v in tests.items():
        if ' - real ' in t and 'delta_other' in t or ' - released ' in t and 'delta_other' in t:
            print(t, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
    with open(os.path.join(HERE, 'results', f'beat_suite_{a.dataset}{a.tag}.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
