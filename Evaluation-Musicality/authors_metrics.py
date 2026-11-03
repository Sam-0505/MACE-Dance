"""The authors' test metrics (Expert-Motion/model/metric.py: BAS, FID_k, FID_g, Div_k, Div_g, PFC) from joint arrays.

Metric.calculate_metric rebuilds joints from saved SMPL rotations after rotate_pred_like_gt, a -90 deg rotation of
the whole body about X. FK commutes with a global rotation, so rotating the joints gives the same keypoints; that
lets samples without rotations (the crossfaded stitched samples) be scored. Check: on a test.py output folder the
two paths agree (--check).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/authors_metrics.py --check eval/train_sweep_l0_test/0
    .venv/bin/python ../Evaluation-Musicality/authors_metrics.py --mace_dir ../Evaluation-Musicality/data/mace_test_stitched
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


def scores(metric, joints):
    """joints: (84, T, 24, 3) in the model's frame, in test-file order."""
    k = torch.as_tensor(np.asarray(joints), dtype=torch.float32)
    k = RotateAxisAngle(-90, axis='X', degrees=True).transform_points(k.reshape(-1, 3)).reshape(k.shape)
    k = align_axis(k)
    geo_gt, geo = metric.normalize(metric.geo_feat_gt, geometric_features(k))
    kin_gt, kin = metric.normalize(metric.kin_feat_gt, kinetic_features(k))
    return {'BAS': float(calculate_beat_similarity(metric.music_beat_gt, k)), 'FID_k': float(calc_fid(kin, kin_gt)),
            'FID_g': float(calc_fid(geo, geo_gt)), 'Div_k': float(calculate_avg_distance(kin)),
            'Div_g': float(calculate_avg_distance(geo)), 'PFC': float(calculate_pfc(k))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mace_dir', help='gen_mace_test.py output; scored per seed')
    ap.add_argument('--check', help='test.py output folder of pkl files: compare with Metric.calculate_metric')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = Metric('./data/test', device)
    if args.check:
        files = sorted(os.listdir(args.check))
        joints = np.stack([pickle.load(open(os.path.join(args.check, f), 'rb'))['full_pose'] for f in files])
        print('joints path:', scores(metric, joints))
        metric.calculate_metric(args.check, device)
        return
    names = json.load(open(os.path.join(args.mace_dir, 'names.json')))['names']
    assert names == sorted(f[:-4] for f in os.listdir('./data/test/baseline_feats'))
    gen = np.load(os.path.join(args.mace_dir, 'joints.npy'))
    per_seed = [scores(metric, gen[:, s].astype(np.float32)) for s in range(gen.shape[1])]
    res = {m: {'mean': float(np.mean([p[m] for p in per_seed])), 'sd': float(np.std([p[m] for p in per_seed]))}
           for m in per_seed[0]}
    print(os.path.basename(args.mace_dir.rstrip('/')), {m: f"{v['mean']:.3f} ± {v['sd']:.3f}" for m, v in res.items()})
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    results[os.path.basename(args.mace_dir.rstrip('/'))] = res
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
