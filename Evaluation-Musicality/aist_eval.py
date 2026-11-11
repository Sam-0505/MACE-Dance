"""AIST++ in-domain evaluation in Bailando's protocol (FID_k, FID_g, Div_k, Div_g, BAS), for MACE trained on AIST++.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/aist_eval.py --ckpt runs/aist/mace/weights/best.pt --name best

Protocol (Bailando README, utils/metrics_new.py): kinetic (72-d) and manual (32-d) features of the generated dances are
compared with those of ALL 1408 real AIST++ motions (root-subtracted at frame 0, 60 fps, y-up); Div is the mean pairwise
distance of the normalised features; BAS is the mean Gaussian-weighted distance of each music beat to the nearest motion beat
(kinetic-velocity local minima, Gaussian smoothing 5, sigma^2 = 9). Differences from Bailando's own run:
  - MACE makes 30 fps motion; it is upsampled by linear interpolation to 60 fps so the 60 fps feature code applies.
  - Music beats are the beat column of the training features (EDGE's librosa baseline, 30 fps), not Bailando's 53rd music column.
  - Dances are the 20 test sequences (EDGE split, ignore list removed), each cut to its first min(music, motion, 20 s)
    on a stitched-window grid; 3 seeds, each seed scored as one set of 20 dances, then mean and std over seeds.
Long dances are made like EDGE's long sampling: 5 s windows overlapping by 2.5 s, crossfaded (MaceSampler.sample_stitched).
Real rows: the same 20 test dances, same cut.
"""
import argparse
import glob
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import torch
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/sam0505/Documents/Code/dance_baselines/Bailando/utils')
from features.kinetic import extract_kinetic_features  # noqa: E402
from features.manual_new import extract_manual_features  # noqa: E402
from metrics_new import calc_fid, calculate_avg_distance, normalize  # noqa: E402

RAW = os.path.join(HERE, '..', 'Expert-Motion', 'data', 'aist_raw', 'edge_aistpp')
SPLITS = os.path.join(HERE, '..', 'Expert-Motion', 'data', 'aist', 'splits')
GT_CACHE = os.path.join(HERE, 'data', 'aist', 'gt_feats_all1408.npz')
MAX60 = 1200  # Bailando keeps the first 1200 frames of 60 fps
STYLE_WIN, STRIDE = 150, 75


def read_list(f):
    return [x.strip() for x in open(f) if x.strip()]


def feats_of(kp60):
    """kp60: (T, 24, 3) y-up, 60 fps -> (72,), (32,) Bailando features after subtracting frame-0 root."""
    kp = kp60 - kp60[:1, :1]
    return extract_kinetic_features(kp), extract_manual_features(kp)


def _gt_one(name):
    from vis import SMPLSkeleton  # noqa: F401  (FK as in the rest of the repo)
    d = pickle.load(open(f'{RAW}/motions/{name}.pkl', 'rb'))
    sk = _gt_one.sk
    with torch.no_grad():
        kp = sk.forward(torch.from_numpy(d['smpl_poses']).float().reshape(1, -1, 24, 3),
                        torch.from_numpy(d['smpl_trans'] / d['smpl_scaling']).float()[None]).squeeze(0).numpy()
    return feats_of(kp)


def _gt_init():
    from vis import SMPLSkeleton
    torch.set_num_threads(1)
    _gt_one.sk = SMPLSkeleton(device='cpu')


def gt_features():
    if os.path.exists(GT_CACHE):
        z = np.load(GT_CACHE)
        return z['k'], z['m']
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{RAW}/motions/*.pkl'))
    with Pool(20, initializer=_gt_init) as p:
        out = p.map(_gt_one, names, chunksize=8)
    k = np.stack([o[0] for o in out])
    m = np.stack([o[1] for o in out])
    os.makedirs(os.path.dirname(GT_CACHE), exist_ok=True)
    np.savez(GT_CACHE, k=k, m=m, names=np.array(names))
    return k, m


def motion_beats(kp60):
    vel = np.mean(np.sqrt(np.sum((kp60[1:] - kp60[:-1]) ** 2, axis=2)), axis=1)
    return argrelextrema(G(vel, 5), np.less)[0]


def ba(music_beats, mbeats):
    return float(np.mean([np.exp(-np.min((mbeats - b) ** 2) / 2 / 9) for b in music_beats]))


def up(joints30):
    """(T,24,3) at 30 fps in MACE's z-up frame -> 60 fps y-up (undo the dataset's +90 deg about X: (y, z) -> (-z, y))."""
    j = np.stack([joints30[..., 0], joints30[..., 2], -joints30[..., 1]], -1)  # (x, y, z) z-up -> (x, z, -y) y-up
    t = np.arange(len(j))
    t2 = np.arange(2 * len(j)) / 2.0
    t2 = t2[t2 <= t[-1]]
    return np.stack([np.stack([np.interp(t2, t, j[:, a, c]) for c in range(3)], -1) for a in range(24)], 1)


def score(gen60, beats60, gtk, gtm):
    fk = [feats_of(g[:MAX60]) for g in gen60]
    pk = np.stack([f[0] for f in fk])
    pm = np.stack([f[1] for f in fk])
    gk, pk_n = normalize(gtk, pk)
    gm, pm_n = normalize(gtm, pm)
    bas = np.mean([ba(b[b < min(len(g), MAX60)], motion_beats(g[:MAX60])) for g, b in zip(gen60, beats60)])
    return {'FID_k': float(calc_fid(pk_n, gk)), 'FID_g': float(calc_fid(pm_n, gm)),
            'Div_k': float(calculate_avg_distance(pk_n)), 'Div_g': float(calculate_avg_distance(pm_n)), 'BAS': float(bas)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None, help='MACE checkpoint; omit with --real')
    ap.add_argument('--name', required=True)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--real', action='store_true', help='score the real test dances')
    ap.add_argument('--critic', default=None, help='critic checkpoint: critic guidance + foot term + frozen root rotation + motion gain')
    ap.add_argument('--scale', type=float, default=0.003, help='guidance scale (as the FineDance main model)')
    ap.add_argument('--foot', type=int, default=10)
    ap.add_argument('--gain_ref', type=float, default=0.77, help='real AIST++ median root-relative speed (m/s), FineDance: 0.49')
    ap.add_argument('--save', default=None, help='also save each dance as <dir>/<name>.npy (S, T, 24, 3), Y-up, 30 fps')
    ap.add_argument('--direct', action='store_true', help='sample the whole clip in one pass instead of stitched windows')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'aist_eval.json'))
    args = ap.parse_args()

    gtk, gtm = gt_features()
    ignore = set(read_list(f'{SPLITS}/ignore_list.txt'))
    test = [n for n in read_list(f'{SPLITS}/crossmodal_test.txt') if n not in ignore]
    import audio_features as af
    from aist_common import GENRE_TO_STYLE, nominal_bpm  # noqa: F401
    sampler = None
    if not args.real:
        from mace_sampler import MaceSampler
        sampler = MaceSampler(args.ckpt)

    clips = []
    for n in test:
        d = pickle.load(open(f'{RAW}/motions/{n}.pkl', 'rb'))
        y = af.load_audio(f'{RAW}/wavs/{n}.wav')
        piece = n.split('_')[4]
        bpm = 110 + 5 * int(piece[3]) if piece[:3] == 'mHO' else 80 + 10 * int(piece[3])
        T = min(len(d['smpl_poses']) // 2, 600)
        feat = af.extract_baseline(y[:(T + 1) * af.HOP_LENGTH], start_bpm=bpm)[:T]
        T = len(feat)
        if not args.direct:
            T = STYLE_WIN + (T - STYLE_WIN) // STRIDE * STRIDE
        clips.append((n, d, feat[:T]))

    bank = bank_g = None
    if args.critic:
        from critic import load_critic
        critic = load_critic(path=args.critic)
        wins, bank_g = [], []
        for n, d, feat in clips:
            for k in range(0, len(feat) - STYLE_WIN + 1, STRIDE):
                wins.append(feat[k:k + STYLE_WIN])
                bank_g.append(n.split('_')[4])
        bank = np.stack(wins)
    res = {'name': args.name, 'ckpt': args.ckpt, 'critic': args.critic, 'n_dances': len(clips), 'per_seed': []}
    saved = {}
    for seed in range(1 if args.real else args.seeds):
        gens, beats = [], []
        for n, d, feat in clips:
            T = len(feat)
            if args.real:
                from vis import SMPLSkeleton
                sk = SMPLSkeleton(device='cpu')
                with torch.no_grad():
                    kp = sk.forward(torch.from_numpy(d['smpl_poses'][:2 * T]).float().reshape(1, -1, 24, 3),
                                    torch.from_numpy(d['smpl_trans'][:2 * T] / d['smpl_scaling']).float()[None]).squeeze(0).numpy()
                gens.append(kp)
            else:
                style = GENRE_TO_STYLE[n[:3]]
                f = feat[None]
                if args.critic:
                    j = sampler.sample_stitched_guided(f, [style], seed, bank, bank_g, n.split('_')[4], args.scale, STYLE_WIN, STRIDE,
                                                       foot_iters=args.foot, freeze_root_rot=True, critic=critic,
                                                       gain_ref=args.gain_ref)[0]
                else:
                    j = (sampler.sample(f, [style], seed) if args.direct else sampler.sample_stitched(f, [style], seed, STYLE_WIN, STRIDE))[0]
                gens.append(up(j))
                saved.setdefault(n, []).append(np.stack([j[..., 0], j[..., 2], -j[..., 1]], -1).astype(np.float32))
            beats.append(2 * np.where(feat[:, 34] > 0)[0])
        r = score(gens, beats, gtk, gtm)
        res['per_seed'].append(r)
        print(args.name, seed, r, flush=True)
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for n, v in saved.items():
            np.save(os.path.join(args.save, n + '.npy'), np.stack(v))
    keys = res['per_seed'][0].keys()
    res['mean'] = {k: float(np.mean([r[k] for r in res['per_seed']])) for k in keys}
    res['std'] = {k: float(np.std([r[k] for r in res['per_seed']])) for k in keys}
    allres = json.load(open(args.out)) if os.path.exists(args.out) else {}
    allres[args.name] = res
    json.dump(allres, open(args.out, 'w'), indent=1)
    print('MEAN', args.name, res['mean'])


if __name__ == '__main__':
    main()
