"""Diagnostics for the lambda sweep (2026-11-02): where does the training-time r gain go?

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/sweep_diagnostics.py

Same 8 s (240-frame) windows for training songs and test songs, so the two splits are scored identically.
  train: up to 4 non-overlapping, non-silent training slices per song (slice k with k % 8 == 0)
  test : the 84 test slices cut at frames 0 and 240 (168 windows, 19 songs); treble from the sliced test wav
Score of a motion window = zero-lag Pearson r(smoothed treble, smoothed body energy), sigma 15, the loss's own
torch functions. Delta = r(own music) - mean r(music of every window from another song of the same split),
averaged per song, then over songs (95% bootstrap CI over songs).

D1  x0 prediction from q_sample(real motion, t), inference guidance (beta 0.75) and training guidance (random beta)
D2  full DDIM sampling (2 seeds), plus Delta of the x0 estimate at each sampling step
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, '../Evaluation-Musicality')
from EDGE import EDGE  # noqa: E402
from model import musicality as MU  # noqa: E402
from accent_overlap import full_music  # noqa: E402

DEV = 'cuda'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'sweep_diagnostics.json')
CKPTS = {'released': 'runs/train/exp/weights/train-3750.pt',
         'lambda_0': 'runs/train/sweep_l0/weights/train-5.pt',
         'lambda_3e-4': 'runs/train/sweep_l3e-4/weights/train-5.pt',
         'lambda_1e-3': 'runs/train/sweep_l1e-3/weights/train-5.pt'}
T_GRID = [10, 50, 100, 200, 400, 600, 800, 950]
SIGMA, W, CHUNK, SEEDS = 15, 240, 192, 2


def load_windows():
    rng = np.random.default_rng(0)
    sig = np.load('data/dataset_backups/music_signals_train.npz')
    names = list(sig['names'])
    by_song = {}
    for i, n in enumerate(names):
        s, k = n.split('_slice')
        if int(k) % 8 == 0 and sig['treble'][i].std() > 1e-8:
            by_song.setdefault(s, []).append(i)
    idx = np.sort(np.concatenate([rng.choice(v, min(4, len(v)), replace=False) for v in by_song.values()]))
    ds = pickle.load(open('data/dataset_backups/train_tensor_dataset.pkl', 'rb')).data
    train = dict(pose=torch.as_tensor(ds['pose'][idx]), cond=torch.as_tensor(ds['features'][idx]),
                 genre=torch.as_tensor(np.asarray(ds['genres'])[idx]), music=torch.from_numpy(sig['treble'][idx]),
                 song=np.array([names[i].split('_slice')[0] for i in idx]))
    del ds

    dt = pickle.load(open('data/dataset_backups/test_tensor_dataset.pkl', 'rb')).data
    pose, cond, genre, music, song = [], [], [], [], []
    for i, f in enumerate(dt['filenames']):
        n = os.path.splitext(os.path.basename(f))[0]
        tr = np.asarray(full_music(f'data/test/wavs_sliced/{n}.wav')['treble'], dtype=np.float32)
        for o in (0, W):
            pose.append(dt['pose'][i, o:o + W]); cond.append(torch.as_tensor(dt['features'][i, o:o + W]))
            genre.append(int(dt['genres'][i])); music.append(torch.from_numpy(tr[o:o + W])); song.append(n.split('_slice')[0])
    test = dict(pose=torch.stack(pose), cond=torch.stack(cond), genre=torch.tensor(genre), music=torch.stack(music),
                song=np.array(song))
    return {'train': train, 'test': test}


def zscore(x):
    x = x - x.mean(1, keepdim=True)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def score(energy_s, music_s, song, rng=np.random.default_rng(1)):
    """energy_s, music_s: (B, T) smoothed. Returns mean own r, Delta and its 95% CI over songs."""
    R = (zscore(music_s.double()) @ zscore(energy_s.double()).T).cpu().numpy()  # R[j, i] = r(music j, motion i)
    own = np.diag(R)
    other = song[:, None] != song[None, :]
    wrong = (R * other).sum(0) / other.sum(0)
    d = own - wrong
    songs = np.unique(song)
    per = np.array([d[song == s].mean() for s in songs])
    boot = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(2000)]
    return {'r_own': float(own.mean()), 'delta': float(per.mean()),
            'ci': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def energy_s(x, d):
    """Normalized pose (B, T, 151) -> smoothed body energy, clipped like Normalizer.unnormalize."""
    return MU.gaussian_smooth(MU.motion_energy(MU.pose_to_joints(x.clamp(-1, 1), d.normalizer, d.smpl)), SIGMA)


def chunks(n):
    return [slice(a, min(a + CHUNK, n)) for a in range(0, n, CHUNK)]


@torch.no_grad()
def x0_preds(d, w, t, guidance):
    torch.manual_seed(t)
    out = []
    for c in chunks(len(w['pose'])):
        x = w['pose'][c].to(DEV)
        tt = torch.full((len(x),), t, device=DEV, dtype=torch.long)
        xt = d.q_sample(x, tt, torch.randn_like(x))
        cond, genre = w['cond'][c].to(DEV), w['genre'][c].to(DEV)
        x0 = d.model.infer_pred(xt, cond, genre, tt) if guidance == 'infer' else d.model.train_pred(xt, cond, genre, tt)
        out.append(energy_s(x0, d))
    return torch.cat(out)


@torch.no_grad()
def sample(d, w, seed):
    """d.ddim_sample with the x0 estimate of every step kept (smoothed energy)."""
    torch.manual_seed(seed)
    final, steps = [], []
    times = list(reversed(torch.linspace(-1, d.n_timestep - 1, steps=51).int().tolist()))
    for c in chunks(len(w['pose'])):
        cond, genre = w['cond'][c].to(DEV), w['genre'][c].to(DEV)
        x = torch.randn((len(cond), W, 151), device=DEV)
        per_step = []
        for time, time_next in zip(times[:-1], times[1:]):
            tc = torch.full((len(x),), time, device=DEV, dtype=torch.long)
            pred_noise, x_start = d.model_predictions(x, cond, genre, tc, clip_x_start=d.clip_denoised)
            per_step.append(energy_s(x_start, d))
            if time_next < 0:
                x = x_start
                continue
            a, an = d.alphas_cumprod[time], d.alphas_cumprod[time_next]
            sigma = ((1 - a / an) * (1 - an) / (1 - a)).sqrt()
            x = x_start * an.sqrt() + (1 - an - sigma ** 2).sqrt() * pred_noise + sigma * torch.randn_like(x)
        final.append(energy_s(x, d))
        steps.append(torch.stack(per_step))
    return torch.cat(final), torch.cat(steps, 1), times[:-1]


def main():
    wins = load_windows()
    for k, w in wins.items():
        print(k, len(w['pose']), 'windows,', len(np.unique(w['song'])), 'songs', flush=True)
        w['music_s'] = MU.gaussian_smooth(w['music'].to(DEV), SIGMA)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for label, ck in CKPTS.items():
        model = EDGE('baseline', checkpoint_path=ck)
        model.eval()
        d = model.diffusion
        d.normalizer = model.normalizer
        r = {}
        for split, w in wins.items():
            sc = lambda e: score(e, w['music_s'], w['song'])  # noqa: E731
            if 'real' not in res.get(split, {}):
                res.setdefault(split, {})['real'] = sc(torch.cat([energy_s(w['pose'][c].to(DEV), d) for c in chunks(len(w['pose']))]))
            r[split] = {'x0_infer': {t: sc(x0_preds(d, w, t, 'infer')) for t in T_GRID},
                        'x0_train': {t: sc(x0_preds(d, w, t, 'train')) for t in T_GRID}}
            samp = [sample(d, w, s) for s in range(SEEDS)]
            r[split]['sampled'] = [sc(f) for f, _, _ in samp]
            times = samp[0][2]
            r[split]['sampling_steps'] = {times[i]: sc(samp[0][1][i]) for i in range(0, 50, 7)}
            print(label, split, 'real', res[split]['real']['delta'],
                  'x0 infer', {t: round(v['delta'], 3) for t, v in r[split]['x0_infer'].items()},
                  'sampled', [round(v['delta'], 3) for v in r[split]['sampled']], flush=True)
        res.setdefault('ckpt', {})[label] = r
        with open(OUT, 'w') as f:
            json.dump(res, f, indent=1)
        del model
        torch.cuda.empty_cache()
    print('saved', OUT)


if __name__ == '__main__':
    main()
