"""Does the +-2 s cross-attention mask hurt other motion properties? (2026-11-04)

One-shot 34 s samples (gen_mace_test.py, 84 test slices x 10 seeds) of the released model, the +-2 s / +-4 s masks
and the stitched sampler, next to the real test dance. All joints in the model's own Z-up frame (the real motion is
mapped with export_smpl_examples.yup_to_zup and the same SMPL forward kinematics), metres, 30 fps.
  div_k / div_g  : the authors' Div_k / Div_g (models: results/authors_metrics.json), with the real test dance
                   scored the same way here for reference
  seed_div       : mean pairwise distance between the 10 samples of one slice (root-relative joints, m), i.e. how
                   different two dances for the same music are (no real reference: one real dance per slice)
  speed          : mean joint speed (m/s)        freeze : share of frames with mean joint speed < 0.15 m/s
  travel         : root horizontal path length per 34 s clip (m)
  skate          : mean horizontal speed of a foot while in contact (ankle/toe within 3 cm of the clip's lowest
                   foot height and vertical speed < 0.1 m/s), m/s;  contact = share of foot-frames in contact
  float          : 95th percentile of the per-frame lowest foot height above the clip's lowest (m)
Per slice: mean over seeds; per song: mean over slices; paired sign-flip tests over the 19 songs (mask - released).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/motion_quality_extra.py
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from model.metric import SMPLSkeleton, calculate_avg_distance  # noqa: E402

FPS = 30
FEET = (7, 8, 10, 11)  # ankles, toes
MODELS = {'released': 'mace_test', 'mask_2s': 'mace_test_local2s', 'mask_4s': 'mace_test_local4s',
          'stitched': 'mace_test_stitched'}
OUT = os.path.join(HERE, 'results', 'motion_quality_extra.json')


def real_joints(names):
    from export_smpl_examples import yup_to_zup
    smpl = SMPLSkeleton('cuda')
    out = []
    for n in names:
        m = pickle.load(open(f'data/test/motions_sliced/{n}.pkl', 'rb'))
        q, pos = yup_to_zup(m['q'], m['pos'])
        q = torch.as_tensor(q, device='cuda').reshape(1, -1, 24, 3)
        out.append(smpl.forward(q, torch.as_tensor(pos, device='cuda')[None]).cpu().numpy()[0])
    return np.stack(out).astype(np.float32)


def clip_stats(j):
    """j (T, 24, 3) Z-up."""
    v = np.linalg.norm(np.diff(j, axis=0), axis=-1) * FPS  # (T-1, 24)
    speed = v.mean(1)
    root = j[:, 0, :2]
    feet = j[:, FEET]
    h = feet[..., 2] - feet[..., 2].min()
    vz = np.abs(np.diff(feet[..., 2], axis=0)) * FPS
    vxy = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1) * FPS
    contact = (h[1:] < 0.03) & (vz < 0.1)
    return {'speed': float(speed.mean()), 'freeze': float((speed < 0.15).mean()),
            'travel': float(np.linalg.norm(np.diff(root, axis=0), axis=-1).sum()),
            'skate': float(vxy[contact].mean()) if contact.any() else np.nan, 'contact': float(contact.mean()),
            'float': float(np.percentile(h.min(1), 95))}


def seed_div(js):
    """js (S, T, 24, 3): mean pairwise root-relative joint distance."""
    r = js - js[:, :, :1]
    d = [np.linalg.norm(r[a] - r[b], axis=-1).mean() for a in range(len(r)) for b in range(a + 1, len(r))]
    return float(np.mean(d))


def authors_div(joints, metric):
    from authors_metrics import geometric_features, kinetic_features, align_axis, RotateAxisAngle
    k = torch.as_tensor(joints, dtype=torch.float32)
    k = RotateAxisAngle(-90, axis='X', degrees=True).transform_points(k.reshape(-1, 3)).reshape(k.shape)
    k = align_axis(k)
    _, geo = metric.normalize(metric.geo_feat_gt, geometric_features(k))
    _, kin = metric.normalize(metric.kin_feat_gt, kinetic_features(k))
    return float(calculate_avg_distance(kin)), float(calculate_avg_distance(geo))


def signflip(x, rng):
    s = rng.choice([-1, 1], (20000, len(x)))
    return float((np.abs((s * x).mean(1)) >= abs(x.mean())).mean())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', help='label=data_dir pairs to score instead of MODELS (released is kept); '
                    'tests are then each model - released')
    ap.add_argument('--tag', default='', help='suffix of the results file')
    args = ap.parse_args()
    out = OUT
    if args.models:
        MODELS.clear()
        MODELS.update({'released': 'mace_test', **dict(x.split('=', 1) for x in args.models)})
        out = OUT.replace('.json', f'{args.tag}.json')
    from model.metric import Metric
    rng = np.random.default_rng(0)
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    song = np.array([n.split('_')[0] for n in names])
    songs = np.unique(song)
    metric = Metric('./data/test', torch.device('cuda'))
    per_song = lambda x: np.array([np.nanmean(x[song == s]) for s in songs])  # noqa: E731
    res, songwise = {}, {}
    AM = json.load(open(os.path.join(HERE, 'results', 'authors_metrics.json')))
    real = real_joints(names)
    rs = [clip_stats(j) for j in real]
    dk, dg = authors_div(real, metric)
    res['real'] = {k: float(np.nanmean(per_song(np.array([r[k] for r in rs])))) for k in rs[0]}
    res['real'].update({'div_k': dk, 'div_g': dg})
    print('real', {k: round(v, 3) for k, v in res['real'].items()}, flush=True)
    for label, d in MODELS.items():
        J = np.load(os.path.join(HERE, 'data', d, 'joints.npy'), mmap_mode='r')  # (84, S, T, 24, 3)
        stats = {}
        for i in range(len(names)):
            js = np.asarray(J[i], dtype=np.float32)
            cs = [clip_stats(j) for j in js]
            for k in cs[0]:
                stats.setdefault(k, []).append(np.nanmean([c[k] for c in cs]))
            stats.setdefault('seed_div', []).append(seed_div(js))
        songwise[label] = {k: per_song(np.array(v)) for k, v in stats.items()}
        res[label] = {k: float(v.mean()) for k, v in songwise[label].items()}
        a = AM[d]  # authors_metrics.py, same Div_k / Div_g computation, mean over the 10 seeds
        res[label].update({'div_k': a['Div_k']['mean'], 'div_g': a['Div_g']['mean']})
        print(label, {k: round(v, 3) for k, v in res[label].items()}, flush=True)
    res['tests'] = {}
    for label in [m for m in MODELS if m != 'released']:
        res['tests'][f'{label} - released'] = {
            k: {'diff': float((songwise[label][k] - songwise['released'][k]).mean()),
                'p': signflip(songwise[label][k] - songwise['released'][k], rng)} for k in songwise[label]}
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    keys = ['speed', 'freeze', 'travel', 'skate', 'contact', 'float', 'seed_div', 'div_k', 'div_g']
    print('\n' + f'{"":10s}' + ''.join(f'{k:>10s}' for k in keys))
    for label in ['real'] + list(MODELS):
        print(f'{label:10s}' + ''.join(f'{res[label].get(k, np.nan):10.3f}' for k in keys))
    for t, v in res['tests'].items():
        print(t, '  '.join(f"{k} {v[k]['diff']:+.3f} (p {v[k]['p']:.3f})" for k in v))


if __name__ == '__main__':
    main()
