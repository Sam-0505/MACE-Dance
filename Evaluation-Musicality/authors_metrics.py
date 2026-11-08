"""The authors' test metrics (Expert-Motion/model/metric.py: BAS, FID_k, FID_g, Div_k, Div_g, PFC) from joint arrays.

Metric.calculate_metric rebuilds joints from saved SMPL rotations after rotate_pred_like_gt, a -90 deg rotation of
the whole body about X. FK commutes with a global rotation, so rotating the joints gives the same keypoints; that
lets samples without rotations (the crossfaded stitched samples) be scored. Check: on a test.py output folder the
two paths agree (--check).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/authors_metrics.py --check eval/train_sweep_l0_test/0
    .venv/bin/python ../Evaluation-Musicality/authors_metrics.py --mace_dir ../Evaluation-Musicality/data/mace_test_stitched

Other models (2026-11-07): the Lodge, Bailando and MEGA arrays were stored with align_axis' +90 deg heading about z
already applied (baselines/*_gen_*.py), i.e. 90 deg off MACE's model frame; --rot_z -90 undoes it before scoring.
This matters here (not for the speed-based music metrics): in the authors' frame the true up is -x, while
kinetic_features treats y and calculate_pfc z as up, so FID_k, Div_k, FID_g, Div_g and PFC depend on the heading.
--real scores the real test dance (Metric's own GT keypoints). --out writes elsewhere; entries keep the per-run values.
    .venv/bin/python ../Evaluation-Musicality/authors_metrics.py --mace_dir ../Evaluation-Musicality/data/mace_test_lodge \
        --rot_z -90 --out ../Evaluation-Musicality/results/authors_metrics_baselines.json
    .venv/bin/python ../Evaluation-Musicality/authors_metrics.py --real --out ../Evaluation-Musicality/results/authors_metrics_baselines.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from pytorch3d.transforms import RotateAxisAngle

sys.path.insert(0, os.getcwd())
from model.metric import (Metric, align_axis, calc_fid, calculate_avg_distance, calculate_beat_similarity,  # noqa: E402
                          calculate_pfc, geometric_features, kinetic_features)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'authors_metrics.json')


def scores(metric, joints, rot_z=0.):
    """joints: (84, T, 24, 3) in the model's frame (rotated by rot_z deg about z first), in test-file order."""
    k = torch.as_tensor(np.asarray(joints), dtype=torch.float32)
    if rot_z:
        k = RotateAxisAngle(rot_z, axis='Z', degrees=True).transform_points(k.reshape(-1, 3)).reshape(k.shape)
    k = RotateAxisAngle(-90, axis='X', degrees=True).transform_points(k.reshape(-1, 3)).reshape(k.shape)
    return eval_frame_scores(metric, align_axis(k))


def eval_frame_scores(metric, k):
    """k: (84, T, 24, 3) in the authors' evaluation frame (Metric.keypoint_gt's)."""
    geo_gt, geo = metric.normalize(metric.geo_feat_gt, geometric_features(k))
    kin_gt, kin = metric.normalize(metric.kin_feat_gt, kinetic_features(k))
    return {'BAS': float(calculate_beat_similarity(metric.music_beat_gt, k)), 'FID_k': float(calc_fid(kin, kin_gt)),
            'FID_g': float(calc_fid(geo, geo_gt)), 'Div_k': float(calculate_avg_distance(kin)),
            'Div_g': float(calculate_avg_distance(geo)), 'PFC': float(calculate_pfc(k))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mace_dir', help='gen_mace_test.py output; scored per seed')
    ap.add_argument('--check', help='test.py output folder of pkl files: compare with Metric.calculate_metric')
    ap.add_argument('--rot_z', type=float, default=0., help='deg about z applied to the stored joints first')
    ap.add_argument('--real', action='store_true', help='score the real test dance (one deterministic run)')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = Metric('./data/test', device)
    if args.check:
        files = sorted(os.listdir(args.check))
        joints = np.stack([pickle.load(open(os.path.join(args.check, f), 'rb'))['full_pose'] for f in files])
        print('joints path:', scores(metric, joints))
        metric.calculate_metric(args.check, device)
        return
    if args.real:
        per_seed, key = [eval_frame_scores(metric, metric.keypoint_gt)], 'real'
    else:
        per_seed, key = run_dir(metric, args.mace_dir, args.rot_z), os.path.basename(args.mace_dir.rstrip('/'))
    res = {m: {'mean': float(np.mean([p[m] for p in per_seed])), 'sd': float(np.std([p[m] for p in per_seed])),
               'runs': [p[m] for p in per_seed]} for m in per_seed[0]}
    if args.rot_z:
        res['rot_z'] = args.rot_z
    print(key, {m: f"{v['mean']:.3f} ± {v['sd']:.3f}" for m, v in res.items() if m != 'rot_z'})
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    results[key] = res
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)


def run_dir(metric, mace_dir, rot_z):
    names = json.load(open(os.path.join(mace_dir, 'names.json')))['names']
    assert names == sorted(f[:-4] for f in os.listdir('./data/test/baseline_feats'))
    gen = np.load(os.path.join(mace_dir, 'joints.npy'))
    return [scores(metric, gen[:, s].astype(np.float32), rot_z) for s in range(gen.shape[1])]


if __name__ == '__main__':
    main()
