"""Energy coupling and motion statistics of several models on Motorica, next to real dance (2026-11-06).

Complements beat_suite.py --dataset motorica (beat locking). All sets on the same 289 non-overlapping 34 s slices
(offset % 1020 == 0, the beat_suite.load_motorica order) and the same 21 body joints: real = BVH BODY joints
(motorica_common.body_joints); models = SMPL joints mapped to the same body parts (SMPL_TO_BODY). First 3 samples
per slice for every model (released MACE has 10; the others 3).

Energy coupling: the motorica_loudness.py primary test (robustness_loud.PRIMARY: RMS loudness -> motion energy,
sigma 15 frames, lag-max over +-9 frames, wrong music = one slice of each of 3 tempo-matched other pieces; pieces =
finedance_release_loud.music_groups), restricted to these slices. Also with root-relative energy, so whole-body
travel cannot carry it. Delta = own - wrong per piece; sign-flip tests over pieces, model - real paired.

FID (fid_scores): kinetic FID against real Motorica on the 21-joint body, and the authors' FID_k / FID_g / Div /
PFC against their FineDance test reference.

Motion statistics per dance (motion_quality_extra.clip_stats, generalised to any up axis and the 21-joint body):
speed, freeze (< 0.15 m/s), root travel per 34 s, foot skate / contact / float (ankles + toes), acceleration, share
of joint-velocity power above 5 Hz (finetune_eval.jitter), and seed_div (mean pairwise root-relative distance
between the samples of one slice). Units: piece means; sign-flip tests vs real over pieces.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/motorica_compare.py --models released=mace mask_2s=mace_local2s ...
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

import audio_features as af  # noqa: E402
import metrics as M  # noqa: E402
from multicomponent import SEED, smooth  # noqa: E402
from robustness_loud import PRIMARY, loudness_signals, bpm_from_wav, matches  # noqa: E402
from finedance_release_loud import music_groups  # noqa: E402
from motorica_common import ROOT, BODY, SLICE_LEN, body_joints, pool, style, wav_path  # noqa: E402

MOT = os.path.dirname(ROOT)
OUT = os.path.join(HERE, 'results', 'motorica_compare.json')
FPS, S = 30, 3
# BODY: Hips Spine Spine1 Neck Head | LShoulder LArm LForeArm LHand | RShoulder RArm RForeArm RHand |
#       LUpLeg LLeg LFoot LToeBase | RUpLeg RLeg RFoot RToeBase
SMPL_TO_BODY = [0, 3, 6, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21, 1, 4, 7, 10, 2, 5, 8, 11]
FEET = [BODY.index(b) for b in ('LeftFoot', 'LeftToeBase', 'RightFoot', 'RightToeBase')]
KEYS = ['speed', 'accel', 'hf5', 'freeze', 'travel', 'skate', 'contact', 'float']


def up_axis(j):
    """Axis along which the root sits highest above the feet."""
    return int(np.argmax((j[:, 0] - j[:, FEET].mean(1)).mean(0)))


def stats(j):
    j = np.asarray(j, np.float64)
    up = up_axis(j)
    hor = [a for a in range(3) if a != up]
    v = np.diff(j, axis=0) * FPS
    sp = np.linalg.norm(v, axis=-1).mean(1)
    acc = np.linalg.norm(np.diff(j, 2, axis=0) * FPS ** 2, axis=-1).mean()
    p = np.abs(np.fft.rfft(v - v.mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(len(v), 1 / FPS)
    feet = j[:, FEET]
    h = feet[..., up] - feet[..., up].min()
    vz = np.abs(np.diff(feet[..., up], axis=0)) * FPS
    vxy = np.linalg.norm(np.diff(feet[..., hor], axis=0), axis=-1) * FPS
    contact = (h[1:] < 0.03) & (vz < 0.1)
    return {'speed': sp.mean(), 'accel': acc, 'hf5': p[f > 5].sum() / p.sum(), 'freeze': (sp < 0.15).mean(),
            'travel': np.linalg.norm(np.diff(j[:, 0, hor], axis=0), axis=-1).sum(),
            'skate': vxy[contact].mean() if contact.any() else np.nan, 'contact': contact.mean(),
            'float': np.percentile(h.min(1), 95)}


def seed_div(js):
    r = js - js[:, :, :1]
    return float(np.mean([np.linalg.norm(r[a] - r[b], axis=-1).mean()
                          for a in range(len(r)) for b in range(a + 1, len(r))]))


def signals(j):
    j = np.asarray(j, np.float64)
    return {'energy': M.motion_energy(j), 'energy_root_rel': M.motion_energy(j - j[:, :1])}


def to_y_up(x):
    """(N, T, J, 3) -> same with the up axis (detected on the first clip) moved to index 1."""
    up = up_axis(x[0])
    hor = [a for a in range(3) if a != up]
    return x[..., [hor[0], up, hor[1]]]


def fid_scores(sets, sets24):
    """(1) FID_k_mot / Div_k_mot: Bailando kinetic features (model.features.kinetic, y up) on the shared 21-joint body,
    against the real Motorica dances of the same slices, normalised by the real features (as model.metric).
    (2) FID_k / FID_g / Div_k / Div_g / PFC exactly as authors_metrics.scores (without BAS) on the full SMPL joints,
    against the authors' FineDance test reference (Metric('./data/test')): how far each model's Motorica dances are
    from real FineDance dance, which favours models trained on FineDance. Per sample index, then mean +- sd."""
    import torch
    from pytorch3d.transforms import RotateAxisAngle
    from model.features.geometric import geometric_features
    from model.features.kinetic import kinetic_features
    from model.metric import Metric, align_axis, calc_fid, calculate_avg_distance, calculate_pfc
    kin = lambda x: kinetic_features(torch.as_tensor(to_y_up(np.asarray(x)), dtype=torch.float32))  # noqa: E731
    norm = lambda gt, p: ((gt - gt.mean(0)) / (gt.std(0) + 1e-4), (p - gt.mean(0)) / (gt.std(0) + 1e-4))  # noqa: E731
    real = np.stack([c[0] for c in sets['real']])
    kr = kin(real)
    out = {'real': {'Div_k_mot': {'mean': float(calculate_avg_distance(norm(kr, kr)[0])), 'sd': 0.0}}}
    metric = Metric('./data/test', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    for lab in sets:
        if lab == 'real':
            continue
        per = []
        for s in range(S):
            x = np.stack([c[s] for c in sets[lab]])
            g, p = norm(kr, kin(x))
            r = {'FID_k_mot': float(calc_fid(p, g)), 'Div_k_mot': float(calculate_avg_distance(p))}
            j = np.stack([c[s] for c in sets24[lab]])
            assert up_axis(j[0][:, SMPL_TO_BODY]) == 2, (lab, 'expected Z-up SMPL joints as MACE')
            k = torch.as_tensor(j)
            k = RotateAxisAngle(-90, axis='X', degrees=True).transform_points(k.reshape(-1, 3)).reshape(k.shape)
            k = align_axis(k)
            geo_gt, geo = metric.normalize(metric.geo_feat_gt, geometric_features(k))
            kin_gt, kn = metric.normalize(metric.kin_feat_gt, kinetic_features(k))
            r.update({'FID_k_fd': float(calc_fid(kn, kin_gt)), 'FID_g_fd': float(calc_fid(geo, geo_gt)),
                      'Div_k': float(calculate_avg_distance(kn)), 'Div_g': float(calculate_avg_distance(geo)),
                      'PFC': float(calculate_pfc(k))})
            per.append(r)
        out[lab] = {m: {'mean': float(np.mean([q[m] for q in per])), 'sd': float(np.std([q[m] for q in per]))}
                    for m in per[0]}
        print(lab, 'fid', {m: round(v['mean'], 3) for m, v in out[lab].items()}, flush=True)
    return out


def signflip(x, rng):
    x = x[~np.isnan(x)]
    s = rng.choice([-1, 1], (20000, len(x)))
    return float((np.abs((s * x).mean(1)) >= abs(x.mean())).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True, help='label=dir under data/motorica')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    models = dict(x.split('=', 1) for x in args.models)

    meta = json.load(open(os.path.join(MOT, 'mace', 'slices.json')))
    slices = [tuple(s) for s in meta['slices'] if s[1] % 1020 == 0]
    takes = sorted({t for t, _ in slices})
    assert set(takes) <= set(pool())
    group = music_groups(takes, path=wav_path)
    pieces = sorted(set(group.values()))
    slice_piece = np.array([group[t] for t, _ in slices])
    print(len(slices), 'slices', len(takes), 'takes', len(pieces), 'pieces', flush=True)

    # joints: real (N, 1, T, 21, 3); models (N, S, T, 21, 3)
    sets, sets24 = {}, {}
    cache, real = {}, []
    for t, o in slices:
        if t not in cache:
            cache = {t: body_joints(t)}
        real.append(cache[t][o:o + SLICE_LEN][None].astype(np.float32))
    sets['real'] = real
    for lab, d in models.items():
        J = np.load(os.path.join(MOT, d, 'joints.npy'), mmap_mode='r')
        if J.shape[0] == len(slices):
            rows = range(len(slices))
        else:
            idx = {tuple(s): i for i, s in enumerate(json.load(open(os.path.join(MOT, d, 'slices.json')))['slices'])}
            rows = [idx[s] for s in slices]
        sets24[lab] = [np.asarray(J[r, :S], np.float32) for r in rows]
        sets[lab] = [x[:, :, SMPL_TO_BODY] for x in sets24[lab]]
    print({k: v[0].shape for k, v in sets.items()}, flush=True)

    # ---- energy coupling
    music = {t: loudness_signals(af.load_audio(wav_path(t))) for t in takes}
    genre = {p: style(p) for p in pieces}
    bpm = {p: bpm_from_wav(wav_path(p)) for p in pieces}
    mt = matches(pieces, genre, bpm, 'tempo_first')
    piece_slices = {p: [s for s, q in zip(slices, slice_piece) if q == p] for p in pieces}
    rng = np.random.default_rng(SEED)
    wrong = [[piece_slices[w][int(rng.integers(len(piece_slices[w])))] for w in mt[group[t]]] for t, _ in slices]
    sig, lag = PRIMARY['sigma'], PRIMARY['lag']
    ms = lambda t, o: smooth(music[t][PRIMARY['music']][o:o + SLICE_LEN], sig)  # noqa: E731
    sc = lambda a, b: M.lagged_corr(a, b, lag)[1]  # noqa: E731
    coup, per_piece = {}, {}
    for lab, clips in sets.items():
        per_piece[lab] = {}
        coup[lab] = {}
        for kind in ('energy', 'energy_root_rel'):
            own_r, wrong_r = [], []
            for i, (t, o) in enumerate(slices):
                own = ms(t, o)
                wr = [ms(wt, wo) for wt, wo in wrong[i]]
                mot = [smooth(signals(x)[kind], sig) for x in clips[i]]
                own_r.append(np.mean([sc(own, m) for m in mot]))
                wrong_r.append(np.mean([sc(w, m) for m in mot for w in wr]))
            own_r, wrong_r = np.array(own_r), np.array(wrong_r)
            d = np.array([(own_r - wrong_r)[slice_piece == p].mean() for p in pieces])
            per_piece[lab][kind] = d
            coup[lab][kind] = {'own': float(own_r.mean()), 'wrong': float(wrong_r.mean()), 'delta': float(d.mean()),
                               'p': signflip(d, rng), 'pos': f'{int((d > 0).sum())}/{len(d)}'}
        print(lab, 'coupling', json.dumps(coup[lab]), flush=True)

    # ---- motion statistics
    mstat, mpiece = {}, {}
    for lab, clips in sets.items():
        per = {k: [] for k in KEYS + ['seed_div']}
        for js in clips:
            cs = [stats(x) for x in js]
            for k in KEYS:
                per[k].append(np.nanmean([c[k] for c in cs]))
            per['seed_div'].append(seed_div(js) if len(js) > 1 else np.nan)
        mpiece[lab] = {k: np.array([np.nanmean(np.array(v)[slice_piece == p]) for p in pieces]) for k, v in per.items()}
        mstat[lab] = {k: float(np.nanmean(v)) for k, v in mpiece[lab].items()}
        print(lab, 'motion', {k: round(v, 3) for k, v in mstat[lab].items()}, flush=True)

    fid = fid_scores(sets, sets24)

    tests = {}
    for lab in sets:
        if lab == 'real':
            continue
        for kind in ('energy', 'energy_root_rel'):
            d = per_piece[lab][kind] - per_piece['real'][kind]
            tests[f'{lab} - real coupling {kind}'] = {'diff': float(d.mean()), 'p': signflip(d, rng)}
        for k in KEYS:
            d = mpiece[lab][k] - mpiece['real'][k]
            tests[f'{lab} - real {k}'] = {'diff': float(np.nanmean(d)), 'p': signflip(d, rng)}

    print('\ncoupling (own - wrong music, per piece)')
    for lab in sets:
        c = coup[lab]
        print(f"{lab:18s} energy {c['energy']['delta']:+.3f} p {c['energy']['p']:.3f} {c['energy']['pos']:>6s} | "
              f"root-rel {c['energy_root_rel']['delta']:+.3f} p {c['energy_root_rel']['p']:.3f} "
              f"{c['energy_root_rel']['pos']:>6s}")
    print('\n' + f'{"":18s}' + ''.join(f'{k:>9s}' for k in KEYS + ['seed_div']))
    for lab in sets:
        print(f'{lab:18s}' + ''.join(f'{mstat[lab][k]:9.3f}' for k in KEYS + ['seed_div']))
    print('\nFID / diversity (mean +- sd over samples)')
    for lab, r in fid.items():
        print(f'{lab:18s} ' + '  '.join(f"{k} {v['mean']:.3f}±{v['sd']:.3f}" for k, v in r.items()))
    with open(OUT.replace('.json', f'{args.tag}.json'), 'w') as f:
        json.dump({'slices': len(slices), 'pieces': pieces, 'coupling': coup, 'motion': mstat, 'fid': fid,
                   'tests': tests,
                   'smpl_to_body': SMPL_TO_BODY, 'samples_per_slice': S}, f, indent=1)


if __name__ == '__main__':
    main()
