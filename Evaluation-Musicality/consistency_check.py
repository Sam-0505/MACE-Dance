"""Does stitched long-form sampling keep a 34 s dance consistent? (2026-11-02)

Real dance vs one-shot MACE vs stitched MACE (MaceSampler.sample_stitched) on the 84 FineDance test slices, 10 seeds
per model. All features are invariant to where the dancer faces: joints relative to the pelvis, rotated each frame
so the hip axis points along x (z is up in the model frame).

Window descriptor (8 s window): mean and SD of that canonical pose (plus pelvis height), mean speed of each joint,
pelvis travel speed and turning speed; standardized by the real test windows. Each slice has four windows
(0-8, 8-16, 16-24, 24-32 s).
  1 style: logistic-regression genre classifier on the descriptors of all real training slices (182 songs, no test
    songs). Per slice: share of the four windows that get the slice's majority genre, mean probability the windows
    give that genre, accuracy against the song's genre label.
  2 drift: Euclidean distance between a slice's windows 1, 2 and 3 apart (8, 16, 24 s), next to the distance between
    windows of two different slices of the same source.
  3 repetition: 1 s segments (step 1/3 s) of canonical pose; for each, the nearest other segment of the same slice
    that starts 4-8 s or more than 8 s away, divided by the median distance between segments >= 4 s apart. A segment
    recurs if that ratio is below the 25th percentile of the real ratios (both bands pooled).
Means are per slice, then per song; 95% CI by bootstrap over the 19 songs.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/consistency_check.py
"""
import json
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
from EDGE import EDGE  # noqa: E402
from model import musicality as MU  # noqa: E402

OUT = os.path.join(HERE, 'results', 'consistency_check.json')
SOURCES = {'one-shot': 'mace_test', 'stitched': 'mace_test_stitched'}
FPS, W = 30, 240
SEG, SEG_STEP, NEAR, FAR = 30, 10, 4 * FPS, 8 * FPS


def canonical(j):
    """(..., T, 24, 3) -> (..., T, 73): pelvis-relative joints with the hip axis along x, plus pelvis height."""
    r = j[..., 2, :2] - j[..., 1, :2]
    th = np.arctan2(r[..., 1], r[..., 0])[..., None]
    rel = j - j[..., :1, :]
    c, s = np.cos(-th), np.sin(-th)
    x = c * rel[..., 0] - s * rel[..., 1]
    y = s * rel[..., 0] + c * rel[..., 1]
    pose = np.stack([x, y, rel[..., 2]], -1).reshape(*j.shape[:-2], 72)
    return np.concatenate([pose, j[..., 0, 2:3]], -1), th[..., 0]


def descriptor(j):
    """(N, T, 24, 3) window -> (N, 172)."""
    pose, th = canonical(j)
    speed = np.linalg.norm(np.diff(j, axis=-3), axis=-1).mean(-2) * FPS
    travel = np.linalg.norm(np.diff(j[..., 0, :2], axis=-2), axis=-1).mean(-1, keepdims=True) * FPS
    turn = np.abs(np.angle(np.exp(1j * np.diff(th, axis=-1)))).mean(-1, keepdims=True) * FPS
    return np.concatenate([pose.mean(-2), pose.std(-2), speed, travel, turn], -1)


def windows(j):
    """(N, 1020, 24, 3) -> (N, 4, 172)."""
    return np.stack([descriptor(j[:, o:o + W]) for o in range(0, 4 * W, W)], 1)


def recurrence(j):
    """(1020, 24, 3) -> nearest-segment ratios for the 4-8 s and > 8 s bands, (S,) each."""
    pose, _ = canonical(j)
    starts = np.arange(0, len(pose) - SEG + 1, SEG_STEP)
    seg = np.stack([pose[a:a + SEG:3].ravel() for a in starts])
    d = np.sqrt(((seg[:, None] - seg[None]) ** 2).sum(-1))
    lag = np.abs(starts[:, None] - starts[None])
    norm = np.median(d[lag >= NEAR])
    near = np.where((lag >= NEAR) & (lag < FAR), d, np.inf).min(1) / norm
    far = np.where(lag >= FAR, d, np.inf).min(1) / norm
    return near, far


def song_ci(per_slice, song, rng):
    songs = np.unique(song)
    per = np.array([per_slice[song == s].mean() for s in songs])
    boot = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(2000)]
    return {'mean': float(per.mean()), 'ci': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def train_descriptors(model):
    ds = pickle.load(open('data/dataset_backups/train_tensor_dataset.pkl', 'rb')).data
    d = model.diffusion
    out = []
    with torch.no_grad():
        for a in range(0, len(ds['pose']), 512):
            j = MU.pose_to_joints(ds['pose'][a:a + 512].cuda(), model.normalizer, d.smpl).cpu().numpy()
            out.append(descriptor(j))
    return np.concatenate(out), np.asarray(ds['genres'])


def main():
    rng = np.random.default_rng(0)
    model = EDGE('baseline', checkpoint_path='runs/train/exp/weights/train-3750.pt')
    dt = pickle.load(open('data/dataset_backups/test_tensor_dataset.pkl', 'rb')).data
    names = [os.path.splitext(os.path.basename(f))[0] for f in dt['filenames']]
    song = np.array([n.split('_slice')[0] for n in names])
    label = np.asarray(dt['genres'])
    with torch.no_grad():
        real = MU.pose_to_joints(dt['pose'].cuda(), model.normalizer, model.diffusion.smpl).cpu().numpy()
    data = {'real': real[:, None]}
    for k, sub in SOURCES.items():
        assert json.load(open(os.path.join(HERE, 'data', sub, 'names.json')))['names'] == names
        data[k] = np.load(os.path.join(HERE, 'data', sub, 'joints.npy')).astype(np.float32)

    print('training the genre classifier', flush=True)
    Xtr, ytr = train_descriptors(model)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=0.1).fit(scaler.transform(Xtr), ytr)
    desc = {k: np.stack([windows(v[:, s]) for s in range(v.shape[1])], 1) for k, v in data.items()}  # (84, S, 4, 172)
    std = StandardScaler().fit(desc['real'].reshape(-1, desc['real'].shape[-1]))

    res, ratios = {}, {}
    for k, D in desc.items():
        n, S = D.shape[:2]
        p = clf.predict_proba(scaler.transform(D.reshape(-1, D.shape[-1]))).reshape(n, S, 4, -1)
        pred = clf.classes_[p.argmax(-1)]
        major = np.array([[np.bincount(r, minlength=clf.classes_.max() + 1).argmax() for r in row] for row in pred])
        ci = np.searchsorted(clf.classes_, major)
        agree = (pred == major[..., None]).mean(-1).mean(1)
        p_major = np.take_along_axis(p, np.broadcast_to(ci[..., None, None], (n, S, 4, 1)), -1)[..., 0].mean(-1).mean(1)
        acc = (pred == label[:, None, None]).mean((1, 2))

        Z = std.transform(D.reshape(-1, D.shape[-1])).reshape(D.shape)
        gap = {g: np.linalg.norm(Z[:, :, g:] - Z[:, :, :-g], axis=-1).mean((1, 2)) for g in (1, 2, 3)}
        # same window position and seed, in a random slice of another song
        other = np.array([np.linalg.norm(Z[i] - Z[rng.choice(np.flatnonzero(song != song[i]), S), np.arange(S)],
                                         axis=-1).mean() for i in range(n)])

        near, far = [], []
        for i in range(n):
            rs = [recurrence(data[k][i, s]) for s in range(S)]
            near.append(np.stack([r[0] for r in rs]))
            far.append(np.stack([r[1] for r in rs]))
        ratios[k] = (np.array(near), np.array(far))  # (84, S, segments)

        res[k] = {'style_agree': song_ci(agree, song, rng), 'style_p_major': song_ci(p_major, song, rng),
                  'genre_acc': song_ci(acc, song, rng),
                  **{f'drift_{8 * g}s': song_ci(v, song, rng) for g, v in gap.items()},
                  'drift_other_slice': song_ci(other, song, rng)}
        print(k, {m: round(v['mean'], 3) for m, v in res[k].items()}, flush=True)

    tau = np.percentile(np.concatenate([ratios['real'][0].ravel(), ratios['real'][1].ravel()]), 25)
    for k, (near, far) in ratios.items():
        res[k]['recur_4_8s'] = song_ci((near < tau).mean((1, 2)), song, rng)
        res[k]['recur_over_8s'] = song_ci((far < tau).mean((1, 2)), song, rng)
        res[k]['nn_ratio_4_8s'] = song_ci(np.median(near, (1, 2)), song, rng)
        res[k]['nn_ratio_over_8s'] = song_ci(np.median(far, (1, 2)), song, rng)
        print(k, {m: round(res[k][m]['mean'], 3) for m in ('recur_4_8s', 'recur_over_8s', 'nn_ratio_4_8s',
                                                           'nn_ratio_over_8s')}, flush=True)
    res['classifier'] = {'train_windows': int(len(ytr)), 'tau': float(tau)}
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
