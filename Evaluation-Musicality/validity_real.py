"""Metric validity on REAL dance: does the metric score a dance higher with its own music than with other music?

For every unit (music piece) and every real dance on it:
  Loud  r = Pearson(Gaussian_sigma15(audio RMS), Gaussian_sigma15(QoM)), zero lag      (loud_models.py)
  Lock  r = |mean exp(i(phi_QoM - theta_beats))| over the clip without `--trim` seconds at both ends  (lock_models.py)
          wrong beats either rescaled to the own tempo ('rescaled', the agreed null) or raw ('raw')
Delta = r(own music) - mean r(wrong music); wrong music = the 3 other-genre pieces closest in tempo (ties at random, seeded),
the first T frames of each (motion frame 0 = music time 0).  Unit = piece (basic and advanced AIST++ dances are separate
sets: 16 beats ~ 7-12 s vs 64 beats ~ 28-34 s).  Tests: sign-flip permutation (10,000, seed 0) of Delta > 0, Wilcoxon next to it.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/validity_real.py --dataset aist --trim_basic 1 --trim_adv 4
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from multicomponent import N_MATCH, SEED, smooth, boot_ci, wil, signflip_p  # noqa: E402
from robustness_loud import loudness_signals, motion_signals, tempo_dist  # noqa: E402
from lock_models import interval, lock_r, FPS  # noqa: E402

SIGMA = 15


def r0(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def unit_scores(u, trim):
    """per unit: mean over its real dances of Loud and Lock (own, wrong rescaled, wrong raw)."""
    T = u['T']
    own_iv = interval(u['beats'])
    rms_own = smooth(u['rms'][:T], SIGMA)
    rms_w = [smooth(w['rms'][:T], SIGMA) for w in u['wrong']]
    out = {k: [] for k in ('loud_own', 'loud_wrong', 'lock_own', 'lock_wrong_rescaled', 'lock_wrong_raw')}
    for j in u['real']:
        q = motion_signals(j[:T])['speed']
        qs = smooth(q, SIGMA)
        out['loud_own'].append(r0(rms_own, qs))
        out['loud_wrong'].append(np.mean([r0(w, qs) for w in rms_w]))
        out['lock_own'].append(lock_r(q, own_iv, u['beats'], 1.0, trim))
        for key, resc in (('lock_wrong_rescaled', True), ('lock_wrong_raw', False)):
            out[key].append(np.nanmean([lock_r(q, own_iv, w['beats'], 1.0, trim, resc) for w in u['wrong']]))
    return {k: float(np.nanmean(v)) for k, v in out.items()}


def stats(own, wrong, rng):
    d = np.asarray(own) - np.asarray(wrong)
    return {'r_own': float(np.mean(own)), 'r_wrong': float(np.mean(wrong)), 'delta': float(d.mean()),
            'delta_ci': boot_ci(d, rng), 'p_perm': signflip_p(d, rng, 'greater'), 'p_wilcoxon': wil(d, 'greater'),
            'n_pos': int((d > 0).sum()), 'n': int(len(d))}


# ------------------------------------------------------------------------------------------------ AIST++
def load_aist(kind):
    import torch
    import audio_features as af
    from aist_common import DATA, nominal_bpm, segments, sequences, joints
    from vis import SMPLSkeleton
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    skel = SMPLSkeleton(dev)
    seg = segments()
    seqs = sequences()
    keys = sorted(k for k in seg if k[1] == kind)
    pieces = sorted({p for p, _ in seg})
    audio = {p: af.load_audio(os.path.join(DATA, 'music', p + '.wav')) for p in pieces}
    rms = {p: loudness_signals(audio[p])['rms'] for p in pieces}

    def beats(p, T):
        f = af.extract_baseline(audio[p][:(T + 1) * af.HOP_LENGTH], start_bpm=nominal_bpm(p))[:T]
        return np.flatnonzero(f[:, 34])

    rng = np.random.default_rng(SEED)
    units = {}
    for key in keys:
        piece, _ = key
        T = seg[key]
        cand = [p for p in pieces if p[1:3] != piece[1:3] and len(rms[p]) >= T]
        tie = rng.random(len(cand))
        order = sorted(range(len(cand)), key=lambda i: (tempo_dist(nominal_bpm(cand[i]), nominal_bpm(piece)), tie[i]))
        wrong = [cand[i] for i in order[:N_MATCH]]
        units[piece] = {'T': T, 'rms': rms[piece], 'beats': beats(piece, T),
                        'wrong': [{'piece': w, 'rms': rms[w], 'beats': beats(w, T)} for w in wrong],
                        'real': [joints(n, skel, dev)[:T] for n, k in seqs.items() if k == key]}
    return units


# ------------------------------------------------------------------------------------------------ PopDanceSet
POP = os.path.join(os.getcwd(), 'data', 'popdance_raw', 'x', 'popdanceset')


def safe_load(path):
    """PopDanceSet ships pickles; only numpy arrays are allowed through (anything else raises)."""
    import io
    import pickle
    allow = {('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', '_reconstruct'),
             ('numpy._core.multiarray', '_reconstruct'), ('numpy.core.multiarray', 'scalar')}

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in allow:
                raise pickle.UnpicklingError(f'blocked {module}.{name}')
            return super().find_class(module, name)
    return U(io.BytesIO(open(path, 'rb').read())).load()


def load_popdance(min_s, rootrel):
    """one unit per clip (motion 30 fps, its own wav); video = name before '_Clip'. Joints = HybrIK 24 SMPL joints
    (pred_xyz_24_struct, root-relative) + root translation `transl` (global) unless rootrel."""
    import glob
    import audio_features as af
    from robustness_loud import bpm_from_wav
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(POP, 'motions', '*.pkl')))
    clips = {}
    for n in names:
        d = safe_load(os.path.join(POP, 'motions', n + '.pkl'))
        y = af.load_audio(os.path.join(POP, 'wavs', n + '.wav'))
        T = min(len(d['transl']), len(af.rms_envelope(y)), 1020)   # 1020 = motion_signals' cap (34 s, as FineDance)
        if T < min_s * FPS:
            continue
        j = d['pred_xyz_24_struct'].astype(np.float64)
        if not rootrel:
            j = j + d['transl'].astype(np.float64)[:, None, :]
        clips[n] = {'T': T, 'y': y, 'joints': j[:T]}
    for n, c in clips.items():
        c['rms'] = loudness_signals(c['y'])['rms']
        c['bpm'] = bpm_from_wav(os.path.join(POP, 'wavs', n + '.wav'))
    vid = {n: n.split('_Clip')[0] for n in clips}
    rng = np.random.default_rng(SEED)
    units = {}
    for n, c in clips.items():
        T = c['T']
        cand = [m for m in clips if vid[m] != vid[n] and clips[m]['T'] >= T]
        tie = rng.random(len(cand))
        order = sorted(range(len(cand)), key=lambda i: (tempo_dist(clips[cand[i]]['bpm'], c['bpm']), tie[i]))
        wrong, seen = [], set()
        for i in order:                       # three wrong clips from three different videos
            if vid[cand[i]] not in seen:
                seen.add(vid[cand[i]])
                wrong.append(cand[i])
            if len(wrong) == N_MATCH:
                break
        units[n] = {'T': T, 'rms': c['rms'], 'beats': _beats(c['y'], T),
                    'wrong': [{'piece': w, 'rms': clips[w]['rms'], 'beats': _beats(clips[w]['y'], T)} for w in wrong],
                    'real': [c['joints']]}
    return units, vid


def _beats(y, T):
    import audio_features as af
    f = af.extract_baseline(y[:(T + 1) * af.HOP_LENGTH])[:T]
    return np.flatnonzero(f[:, 34])


def run_popdance(a):
    trim = int(round(a.trim * FPS))
    units, vid = load_popdance(a.min_s, a.rootrel)
    print(len(units), 'clips >=', a.min_s, 's;', len(set(vid[n] for n in units)), 'videos', flush=True)
    sc = {n: unit_scores(u, trim) for n, u in units.items()}
    vids = sorted({vid[n] for n in sc})
    col = lambda k: np.array([np.nanmean([sc[n][k] for n in sc if vid[n] == v]) for v in vids])  # noqa: E731
    rng = np.random.default_rng(SEED)
    res = {'trim_s': a.trim, 'min_s': a.min_s, 'rootrel': a.rootrel, 'n_clips': len(sc),
           'Loud': stats(col('loud_own'), col('loud_wrong'), rng),
           'Lock rescaled': stats(col('lock_own'), col('lock_wrong_rescaled'), rng),
           'Lock raw': stats(col('lock_own'), col('lock_wrong_raw'), rng)}
    for m, v in res.items():
        if isinstance(v, dict):
            print(f"pop {m:14s} own {v['r_own']:.3f} wrong {v['r_wrong']:.3f} Delta {v['delta']:+.3f} "
                  f"[{v['delta_ci'][0]:+.3f},{v['delta_ci'][1]:+.3f}] pos {v['n_pos']}/{v['n']} "
                  f"p {v['p_perm']:.4f} p_w {v['p_wilcoxon']:.4f}", flush=True)
    fn = os.path.join(HERE, 'results', f"validity_real_popdance_min{int(a.min_s)}_trim{a.trim:g}"
                      f"{'_rootrel' if a.rootrel else ''}.json")
    with open(fn, 'w') as f:
        json.dump({'settings': {'sigma': SIGMA, 'match': 'closest tempo, 3 different videos, ties seeded',
                                'n_wrong': N_MATCH, 'unit': 'video (mean over its clips)', 'n_perm': 10000,
                                'rng_seed': SEED}, 'results': res, 'per_clip': sc}, f, indent=1)
    print('saved', fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('aist', 'popdance'), required=True)
    ap.add_argument('--trim_basic', type=float, default=1.0, help='seconds cut at both ends, 16-beat dances (7-12 s)')
    ap.add_argument('--trim_adv', type=float, default=4.0, help='seconds cut at both ends, 64-beat dances (28-34 s)')
    ap.add_argument('--trim', type=float, default=4.0, help='popdance: seconds cut at both ends')
    ap.add_argument('--min_s', type=float, default=16.0, help='popdance: keep clips at least this long (seconds)')
    ap.add_argument('--rootrel', action='store_true', help='popdance: QoM from root-relative joints (drop root translation)')
    a = ap.parse_args()
    if a.dataset == 'popdance':
        return run_popdance(a)
    res, per = {}, {}
    for kind, trim_s in (('basic', a.trim_basic), ('adv', a.trim_adv)):
        trim = int(round(trim_s * FPS))
        units = load_aist(kind)
        sc = {p: unit_scores(u, trim) for p, u in units.items()}
        ps = sorted(sc)
        col = lambda k: np.array([sc[p][k] for p in ps])  # noqa: E731
        rng = np.random.default_rng(SEED)
        res[kind] = {'trim_s': trim_s, 'n_dances': int(sum(len(u['real']) for u in units.values())),
                     'Loud': stats(col('loud_own'), col('loud_wrong'), rng),
                     'Lock rescaled': stats(col('lock_own'), col('lock_wrong_rescaled'), rng),
                     'Lock raw': stats(col('lock_own'), col('lock_wrong_raw'), rng)}
        per[kind] = sc
        for m, v in res[kind].items():
            if isinstance(v, dict):
                print(f"{kind:5s} {m:14s} own {v['r_own']:.3f} wrong {v['r_wrong']:.3f} Delta {v['delta']:+.3f} "
                      f"[{v['delta_ci'][0]:+.3f},{v['delta_ci'][1]:+.3f}] pos {v['n_pos']}/{v['n']} "
                      f"p {v['p_perm']:.4f} p_w {v['p_wilcoxon']:.4f}", flush=True)
    fn = os.path.join(HERE, 'results', 'validity_real_aist.json')
    with open(fn, 'w') as f:
        json.dump({'settings': {'sigma': SIGMA, 'match': 'other genre, closest tempo, ties seeded', 'n_wrong': N_MATCH,
                                'n_perm': 10000, 'rng_seed': SEED}, 'results': res, 'per_unit': per}, f, indent=1)
    print('saved', fn)


if __name__ == '__main__':
    main()
