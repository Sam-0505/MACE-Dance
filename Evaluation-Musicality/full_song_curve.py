"""Full-song coupling curve (2026-11-04): does one-shot coupling hold over whole FineDance test songs (39-193 s),
far past the 34 s slices of length_curve.py?

Each of the 19 test songs is generated in one shot over its full length (cropped to 240 + 130 k frames so the
stitched sampler fits), 10 samples per song. EDGE's 35-dim baseline features are recomputed on the full wav
(build_long_clips.baseline_features, the recipe that reproduces the shipped test features). Real dance: the
FineDance release motion (smooth; finedance_common). Song 130's wav is 17 s longer than its motion; frame 0 =
music 0 is assumed, as for the slices.

Score: zero-lag own-song Delta between slow treble flux and slow body energy (sigma 15), as in length_curve.py, in
non-overlapping windows (8 s and 34 s) along the song: r(own music window, energy window) minus the mean r with
every same-size window of the other 18 songs. Windows are binned by start time; per song mean per bin, 95% CI by
bootstrap over the songs present in the bin. Prediction from the window-length diagnostic: the 34 s LoRA model holds
to ~34 s and then falls; the released model with the +-2 s cross-attention mask holds at every position.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/full_song_curve.py --label released
    .venv/bin/python ../Evaluation-Musicality/full_song_curve.py --label released_local2s --local_attn 2
    .venv/bin/python ../Evaluation-Musicality/full_song_curve.py --label lora_l0 \
        --checkpoint runs/train/lora_l0/weights/train-30.pt
    .venv/bin/python ../Evaluation-Musicality/full_song_curve.py --label released_stitched --stitch
    .venv/bin/python ../Evaluation-Musicality/full_song_curve.py --report    # table of all labels run so far
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from model import musicality as MU  # noqa: E402

CACHE = os.path.join(HERE, 'data', 'full_song')
OUT = os.path.join(HERE, 'results', 'full_song_curve.json')
SEEDS, CHUNK, SIGMA, FPS = 10, 5, 15, 30
WINDOWS = {'8s': 240, '34s': 1020}
BINS = [0, 34, 68, 102, 1e9]  # window start, seconds
BIN_NAMES = ['0-34 s', '34-68 s', '68-102 s', '102+ s']


def songs():
    return sorted({f.split('_')[0] for f in os.listdir('data/test/baseline_feats')})


def usable_length(song):
    from finedance_common import RELEASE
    t = np.load(os.path.join(RELEASE, 'motion', song + '.npy'), mmap_mode='r').shape[0]
    return 240 + 130 * ((t - 240) // 130)


def prepare():
    """Cache per song: baseline features, treble flux, real-dance energy, genre."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, 'inputs.npz')
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    from build_long_clips import baseline_features
    from accent_overlap import full_music
    from finedance_common import release_joints
    from model.metric import SMPLSkeleton
    smpl = SMPLSkeleton('cuda')
    out = {'songs': [], 'genres': [], 'feats': [], 'treble': [], 'real_energy': []}
    for s in songs():
        T = usable_length(s)
        f = baseline_features(f'data/test/wavs/{s}.wav')[:T].astype(np.float32)
        tr = np.asarray(full_music(f'data/test/wavs/{s}.wav')['treble'][:T], dtype=np.float32)
        j = torch.as_tensor(np.asarray(release_joints(s, smpl, 'cuda'))[:T], dtype=torch.float32)
        assert len(f) == len(tr) == len(j) == T, (s, len(f), len(tr), len(j), T)
        with open(f'data/test/label/{s}.json') as fh:
            out['genres'].append(json.load(fh)['style2'])
        out['songs'].append(s)
        out['feats'].append(f)
        out['treble'].append(tr)
        out['real_energy'].append(MU.motion_energy(j[None])[0].numpy())
        print(s, T / FPS, 's', flush=True)
    obj = {k: np.array(v, dtype=object) for k, v in out.items()}
    np.savez(path, **obj)
    return obj


def generate(label, checkpoint, local_attn, stitch, inp):
    from mace_sampler import MaceSampler
    from length_curve import local_cross_attention
    d = os.path.join(CACHE, label)
    os.makedirs(d, exist_ok=True)
    sampler = MaceSampler(checkpoint) if checkpoint else MaceSampler()
    if local_attn:
        local_cross_attention(sampler.model.diffusion.model, round(local_attn * FPS))
    sample = sampler.sample_stitched if stitch else sampler.sample
    for s, f, g in zip(inp['songs'], inp['feats'], inp['genres']):
        path = os.path.join(d, f'{s}.npy')
        if os.path.exists(path):
            continue
        e = []
        for c in range(SEEDS // CHUNK):  # CHUNK samples of one song per batch; seed c -> distinct noise per row
            j = sample(np.repeat(f[None], CHUNK, 0), [g] * CHUNK, seed=c)
            e.append(MU.motion_energy(torch.from_numpy(j)).numpy())
        np.save(path, np.concatenate(e).astype(np.float32))  # (SEEDS, T)
        print(label, s, len(f) / FPS, 's', flush=True)


def windows(x, w):
    return [(k * w, x[k * w:(k + 1) * w]) for k in range(len(x) // w)]


def z(a):
    a = MU.gaussian_smooth(torch.as_tensor(np.stack(a), dtype=torch.float64), SIGMA)
    a = a - a.mean(1, keepdim=True)
    return a / a.norm(dim=1, keepdim=True)


def score(energy, inp, w):
    """energy: list per song of (T,). Returns per-window records (song index, start s, Delta)."""
    mw, ew, idx = [], [], []
    for i, (e, m) in enumerate(zip(energy, inp['treble'])):
        for (st, mm), (_, ee) in zip(windows(m, w), windows(e, w)):
            mw.append(mm)
            ew.append(ee)
            idx.append((i, st / FPS))
    R = (z(mw) @ z(ew).T).numpy()  # R[music window, energy window]
    song = np.array([i for i, _ in idx])
    other = song[:, None] != song[None, :]
    delta = np.diag(R) - (R * other).sum(0) / other.sum(0)
    return [(i, st, dl) for (i, st), dl in zip(idx, delta)]


def binned(records, n_songs, rng):
    out = {}
    for b, name in enumerate(BIN_NAMES):
        per = np.full(n_songs, np.nan)
        for i in range(n_songs):
            v = [dl for (j, st, dl) in records if j == i and BINS[b] <= st < BINS[b + 1]]
            if v:
                per[i] = np.mean(v)
        out[name] = per
    return out


def summary(per, rng):
    ok = ~np.isnan(per)
    p = per[ok]
    if len(p) < 3:
        return {'mean': float(np.mean(p)) if len(p) else None, 'ci': None, 'n_songs': int(len(p)), 'per_song': per.tolist()}
    boot = [p[rng.integers(0, len(p), len(p))].mean() for _ in range(2000)]
    return {'mean': float(p.mean()), 'ci': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'n_songs': int(len(p)), 'per_song': per.tolist()}


def evaluate(label, inp):
    rng = np.random.default_rng(0)
    n = len(inp['songs'])
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if 'real' not in res:
        res['real'] = {wn: {b: summary(v, rng) for b, v in binned(score(list(inp['real_energy']), inp, w), n, rng).items()}
                       for wn, w in WINDOWS.items()}
    E = [np.load(os.path.join(CACHE, label, f'{s}.npy')) for s in inp['songs']]
    res[label] = {}
    for wn, w in WINDOWS.items():
        per_seed = [binned(score([e[k] for e in E], inp, w), n, rng) for k in range(SEEDS)]
        res[label][wn] = {b: summary(np.mean([p[b] for p in per_seed], 0), rng) for b in BIN_NAMES}
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    report()


def report():
    res = json.load(open(OUT))
    for wn in WINDOWS:
        print(f'\nwindow {wn}: Delta by window start (n songs)')
        print(f'{"":20s}' + ''.join(f'{b:>18s}' for b in BIN_NAMES))
        for lab in res:
            print(f'{lab:20s}' + ''.join(f"{res[lab][wn][b]['mean']:+10.3f} (n {res[lab][wn][b]['n_songs']:2d})  "
                                         if res[lab][wn][b]['mean'] is not None else f'{"-":>18s}'
                                         for b in BIN_NAMES))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--label')
    ap.add_argument('--checkpoint')
    ap.add_argument('--local_attn', type=float)
    ap.add_argument('--stitch', action='store_true')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report:
        report()
    else:
        inp = prepare()
        generate(a.label, a.checkpoint, a.local_attn, a.stitch, inp)
        evaluate(a.label, inp)
