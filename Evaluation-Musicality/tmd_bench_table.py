"""TMD-Bench MDAlign beat metrics (Yang et al., ICML 2026, arXiv 2605.01809, Sec. 3.3 / App. C) for every model of
the comparison tables, on FineDance test and Motorica (2026-11-06). Raw values, as the paper defines them.

  V(t)  = mean keypoint displacement between frames, Gaussian-smoothed; motion accents M = local maxima of V.
  VBCS  = mean over accents of exp(-d^2 / 2 sigma^2), d = distance to the nearest music beat     (accent precision)
  ABHS  = share of music beats with an accent closer than tau                                       (beat coverage)
  CSD / HSD = standard deviation of VBCS / ABHS over clips;  MDAlign = (VBCS + ABHS) / 2.
The paper gives no sigma, tau, smoothing or beat tracker. Same choices as tmd_beat_metrics.py: smoothing 5 frames and
sigma = tau = 3 frames (0.1 s) as in the authors' BAS; beats = the librosa beats the models / BAS use (FineDance:
baseline_feats column 34; Motorica: beats_slices.npz, the af.extract_baseline beats of each slice).

Keypoints: the paper uses 2D pose-estimator keypoints from video. Main view '2d' = an orthographic camera in front of
each clip's mean facing direction (facing = up x (right hip - left hip)), keeping the lateral and the up axis; this
is independent of each model's default heading. '3d' = all joints in 3D (in the notes). Real Motorica = the 21 BVH
body joints (accent timing does not depend on the joint set); models = 24 SMPL joints.

Clips: FineDance 84 slices x all samples; Motorica 289 non-overlapping slices x 3 samples. Models that sample
deterministically (Bailando, MEGA: identical seeds) are scored on one sample per slice. Mean and SD are over all
scored clips.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/tmd_bench_table.py
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from tmd_beat_metrics import MAIN, abhs, accents, vbcs  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

SMOOTH = 5
OUT = os.path.join(HERE, 'results', 'tmd_bench_table.json')
FD = {'real': None, 'edge': 'mace_test_edge', 'lodge': 'mace_test_lodge', 'bailando': 'mace_test_bailando',
      'mega': 'mace_test_mega', 'released': 'mace_test', 'ours': 'mace_test_mask2s_guide10_s0.003'}
MOT = {'real': None, 'edge': 'edge', 'lodge': 'lodge', 'bailando': 'bailando', 'mega': 'mega', 'released': 'mace',
       'ours': 'mace_local2s_guide_s0.003'}
MOT_S = 3
SMPL_HIPS, BODY_HIPS = (1, 2, 15), (13, 17, 4)  # (left hip, right hip, head)


def front_view(j, hips):
    """(T, J, 3) -> (T, J, 2): lateral and up coordinates seen from the clip's mean facing direction."""
    d = (j[:, hips[2]] - j[:, 0]).mean(0)  # pelvis -> head
    up = int(np.argmax(np.abs(d)))
    u = np.eye(3)[up] * np.sign(d[up])
    r = (j[:, hips[1]] - j[:, hips[0]]).mean(0)
    r = r - r.dot(u) * u
    r /= np.linalg.norm(r) + 1e-9
    return np.stack([j @ r, j @ u], -1)


def velocity(p):
    return gaussian_filter1d(np.linalg.norm(np.diff(p, axis=0), axis=-1).mean(1), SMOOTH)


def scores(args):
    j, A, hips = args
    out = {}
    for view, p in (('2d', front_view(j, hips)), ('3d', j)):
        M = accents(velocity(p))
        out[view] = (vbcs(M, A, MAIN), abhs(M, A, MAIN))
    return out


def summarize(rows):
    r = {}
    for view in ('2d', '3d'):
        v = np.array([x[view][0] for x in rows], float)
        h = np.array([x[view][1] for x in rows], float)
        r[view] = {'VBCS': float(np.nanmean(v)), 'CSD': float(np.nanstd(v)), 'ABHS': float(np.nanmean(h)),
                   'HSD': float(np.nanstd(h)), 'MDAlign': float((np.nanmean(v) + np.nanmean(h)) / 2), 'n': len(rows)}
    return r


def samples(J, i, s_max):
    x = np.asarray(J[i, :s_max], np.float32)
    if len(x) > 1 and np.array_equal(x[0], x[1]):
        return x[:1]  # deterministic model
    return x


def finedance(pool):
    from motion_quality_extra import real_joints
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    beats = [np.flatnonzero(np.load(f'data/test/baseline_feats/{n}.npy')[:, 34]) for n in names]
    res = {}
    for lab, d in FD.items():
        if d is None:
            clips = [(j, beats[i], SMPL_HIPS) for i, j in enumerate(real_joints(names))]
        else:
            J = np.load(os.path.join(HERE, 'data', d, 'joints.npy'), mmap_mode='r')
            assert J.shape[0] == len(names), (d, J.shape)
            clips = [(j, beats[i], SMPL_HIPS) for i in range(len(names)) for j in samples(J, i, J.shape[1])]
        res[lab] = summarize(pool.map(scores, clips, chunksize=8))
        print('finedance', lab, len(clips), res[lab]['2d'], flush=True)
    return res


def motorica(pool):
    from motorica_common import body_joints
    mot = os.path.join(HERE, 'data', 'motorica')
    meta = json.load(open(os.path.join(mot, 'mace', 'slices.json')))
    keep = [k for k, (t, o) in enumerate(meta['slices']) if o % 1020 == 0]
    slices = [tuple(meta['slices'][k]) for k in keep]
    z = np.load(os.path.join(mot, 'beats_slices.npz'), allow_pickle=True)
    assert [tuple(s) for s in z['slices'].tolist()] == slices
    beats = list(z['beats'])
    res = {}
    for lab, d in MOT.items():
        if d is None:
            clips, cache = [], {}
            for i, (t, o) in enumerate(slices):
                if t not in cache:
                    cache = {t: body_joints(t)}
                clips.append((cache[t][o:o + 1020].astype(np.float32), beats[i], BODY_HIPS))
        else:
            J = np.load(os.path.join(mot, d, 'joints.npy'), mmap_mode='r')
            if J.shape[0] == len(slices):
                rows = range(len(slices))
                assert [tuple(x) for x in json.load(open(os.path.join(mot, d, 'slices.json')))['slices']] == slices
            else:
                rows = keep
            clips = [(j, beats[i], SMPL_HIPS) for i, r in enumerate(rows) for j in samples(J, r, MOT_S)]
        res[lab] = summarize(pool.map(scores, clips, chunksize=8))
        print('motorica', lab, len(clips), res[lab]['2d'], flush=True)
    return res


def main():
    import argparse
    global FD, MOT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--fd', nargs='*', help='label=dir (under data/) FineDance models instead of FD (real always kept)')
    ap.add_argument('--mot', nargs='*', help='label=dir (under data/motorica) Motorica models instead of MOT')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    if args.fd is not None or args.mot is not None:
        FD = {'real': None, **dict(x.split('=', 1) for x in (args.fd or []))}
        MOT = {'real': None, **dict(x.split('=', 1) for x in (args.mot or []))}
    OUT = OUT.replace('.json', f'{args.tag}.json')
    with Pool(12) as pool:
        out = {'settings': {'sigma': MAIN, 'tau': MAIN, 'smooth': SMOOTH, 'fps': 30, 'motorica_samples': MOT_S},
               'finedance': finedance(pool) if len(FD) > 1 else {}, 'motorica': motorica(pool) if len(MOT) > 1 else {}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    for ds in ('finedance', 'motorica'):
        if not out[ds]:
            continue
        for view in ('2d', '3d'):
            print(f'\n{ds} {view}: {"model":10s} VBCS   CSD    ABHS   HSD    MDAlign  n')
            for lab, r in out[ds].items():
                x = r[view]
                print(f'{"":14s}{lab:10s} {x["VBCS"]:.3f}  {x["CSD"]:.3f}  {x["ABHS"]:.3f}  {x["HSD"]:.3f}  '
                      f'{x["MDAlign"]:.3f}   {x["n"]}')


if __name__ == '__main__':
    main()
