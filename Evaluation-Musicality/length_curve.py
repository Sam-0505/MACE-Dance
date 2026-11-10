"""Length curve (2026-11-03): does one-shot MACE lose song-specific musicality as the generated length grows past its
8 s training windows?

For each length L (8 to 34 s), the released model samples the first L frames of every FineDance test slice in one
shot (MaceSampler.sample on the cropped features), 10 seeds. Score: zero-lag own-song Delta between slow treble flux
and slow body energy (sigma 15; sweep_diagnostics.py, removed 2026-11-10 (tag pre-cleanup-2026-11-10), results/sweep_diagnostics*.json kept; wrong = every window of another song at the same position).
  whole : the full L frames, next to the real dance cropped to the same L
  first8: frames 0-240 only - the same music and window at every L, so a drop is caused by generating longer
  pieces8: every non-overlapping 8 s piece of the L frames
Per slice and seed, then averaged over seeds per song; 95% CI by bootstrap over the 19 songs.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/length_curve.py
    .venv/bin/python ../Evaluation-Musicality/length_curve.py --plot   # figure only, from results/length_curve.json
    .venv/bin/python ../Evaluation-Musicality/length_curve.py --checkpoint runs/train/long_l0/weights/train-30.pt \
        --label long_l0   # a fine-tuned checkpoint -> results/length_curve_long_l0.json
    .venv/bin/python ../Evaluation-Musicality/length_curve.py --local_attn 2 --label released_local2s
        # causal test (2026-11-03): cross-attention restricted to music within +-2 s of each dance frame
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from mace_sampler import MaceSampler  # noqa: E402
from model import musicality as MU  # noqa: E402
from model.metric import style2s  # noqa: E402
from accent_overlap import full_music  # noqa: E402

OUT = os.path.join(HERE, 'results', 'length_curve.json')
CACHE = os.path.join(HERE, 'data', 'length_curve')
LENGTHS = [240, 300, 360, 450, 510, 600, 750, 900, 1020]
SEEDS, W, SIGMA = 10, 240, 15


def deltas(energy, music, song):
    """energy, music: (B, T) unsmoothed. Per-window own - mean wrong r, (B,)."""
    e, m = MU.gaussian_smooth(energy.double(), SIGMA), MU.gaussian_smooth(music.double(), SIGMA)
    z = lambda x: (x - x.mean(1, keepdim=True)) / (x - x.mean(1, keepdim=True)).norm(dim=1, keepdim=True)  # noqa: E731
    R = (z(m) @ z(e).T).cpu().numpy()
    other = song[:, None] != song[None, :]
    return np.diag(R) - (R * other).sum(0) / other.sum(0)


def views(energy, music, song):
    """energy (B, L) -> {'whole', 'first8', 'pieces8'}: (B,) Delta per slice."""
    L = energy.shape[1]
    out = {'whole': deltas(energy, music[:, :L], song), 'first8': deltas(energy[:, :W], music[:, :W], song)}
    k = L // W
    p = deltas(torch.cat([energy[:, i * W:(i + 1) * W] for i in range(k)]),
               torch.cat([music[:, i * W:(i + 1) * W] for i in range(k)]), np.tile(song, k))
    out['pieces8'] = p.reshape(k, -1).mean(0)
    return out


def song_ci(per_slice, song, rng):
    songs = np.unique(song)
    per = np.array([per_slice[song == s].mean() for s in songs])
    boot = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(2000)]
    return {'mean': float(per.mean()), 'ci': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'per_song': per.tolist()}


def local_cross_attention(net, half_width):
    """Restrict every dance layer's cross-attention to music frames within +-half_width frames of the query.
    attention_check.py found that at 34 s the attention no longer stays near the current frame."""
    for layer in net.dancelayerstack:
        t = layer.transformer

        def forward(x, mem, t=t):
            q = torch.arange(x.shape[1], device=x.device)
            k = torch.arange(mem.shape[1], device=x.device)
            mask = (q[:, None] - k[None]).abs() > half_width  # True = not allowed
            kv = t.norm_music(mem)
            x = t.dropout_cross(t.cross_attn(t.norm_dance(x), kv, kv, attn_mask=mask, need_weights=False)[0])
            return t.dropout2(t.linear2(t.dropout1(t.activation(t.linear1(t.norm(x))))))

        t.forward = forward


def main(checkpoint=None, label='released', local_attn=None):
    rng = np.random.default_rng(0)
    cache = CACHE if label == 'released' else os.path.join(CACHE, label)
    out = OUT if label == 'released' else OUT.replace('.json', f'_{label}.json')
    os.makedirs(cache, exist_ok=True)
    dt = pickle.load(open('data/dataset_backups/test_tensor_dataset.pkl', 'rb')).data
    names = [os.path.splitext(os.path.basename(f))[0] for f in dt['filenames']]
    song = np.array([n.split('_slice')[0] for n in names])
    feats = np.stack([np.load(f'data/test/baseline_feats/{n}.npy') for n in names])
    genres = [style2s[int(g)] for g in dt['genres']]
    music = torch.stack([torch.from_numpy(np.asarray(full_music(f'data/test/wavs_sliced/{n}.wav')['treble'][:1020],
                                                     dtype=np.float32)) for n in names]).cuda()
    sampler = MaceSampler(checkpoint) if checkpoint else MaceSampler()
    if local_attn:
        assert label != 'released', 'give a local-attention run its own --label'
        local_cross_attention(sampler.model.diffusion.model, round(local_attn * 30))
    with torch.no_grad():
        real = MU.motion_energy(MU.pose_to_joints(dt['pose'].cuda(), sampler.model.normalizer,
                                                  sampler.model.diffusion.smpl))

    res = {'lengths_s': [L / 30 for L in LENGTHS], 'real': {}, 'model': {}}
    for L in LENGTHS:
        path = os.path.join(cache, f'energy_{L}.npy')
        if not os.path.exists(path):
            e = [MU.motion_energy(torch.from_numpy(sampler.sample(feats[:, :L], genres, seed=s))).numpy()
                 for s in range(SEEDS)]
            np.save(path, np.stack(e, 1))
        e = torch.from_numpy(np.load(path)).cuda()  # (84, seeds, L)
        per_seed = [views(e[:, s], music, song) for s in range(e.shape[1])]
        rv = views(real[:, :L], music, song)
        res['real'][L] = {v: song_ci(rv[v], song, rng) for v in rv}
        res['model'][L] = {v: song_ci(np.mean([p[v] for p in per_seed], 0), song, rng) for v in rv}
        res['model'][L]['seed_sd'] = {v: float(np.std([song_ci(p[v], song, rng)['mean'] for p in per_seed])) for v in rv}
        res['model'][L]['energy_mean'] = float(e.mean())
        res['real'][L]['energy_mean'] = float(real[:, :L].mean())
        print(f'{L / 30:5.1f} s  ' + '  '.join(f"{v}: real {res['real'][L][v]['mean']:+.3f} model "
                                                f"{res['model'][L][v]['mean']:+.3f}" for v in rv), flush=True)
        with open(out, 'w') as f:
            json.dump(res, f, indent=1)
    print('saved', out)
    if label == 'released':
        plot()


def plot():
    """results/figures/length_curve.png: (a) whole-length Delta, real vs one-shot MACE, with the stitched 34 s result;
    (b) the gap real - model per song, 95% bootstrap CI."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted, grid = '#0b0b0b', '#52514e', '#e4e3df'
    real_c, model_c, stitch_c = '#2a78d6', '#eb6834', '#1baf7a'
    res = json.load(open(OUT))
    Ls = sorted(res['model'], key=int)
    x = np.array([int(L) / 30 for L in Ls])
    rng = np.random.default_rng(0)
    stitched = json.load(open(os.path.join(HERE, 'results', 'sweep_diagnostics_scale.json')))['released_stitched']['34s']

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': muted, 'axes.labelcolor': ink, 'xtick.color': muted,
                         'ytick.color': muted, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=200)
    for ax in (a, b):
        ax.grid(axis='y', color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.axvline(8, color=muted, lw=1, ls=':')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:g}' for v in x])
        ax.set_xlabel('generated length, one shot (s)')

    for key, color, marker, label in (('real', real_c, 'o', 'real dancers'), ('model', model_c, 's', 'MACE one-shot')):
        m = np.array([res[key][L]['whole']['mean'] for L in Ls])
        lo, hi = np.array([res[key][L]['whole']['ci'] for L in Ls]).T
        a.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
        a.plot(x, m, color=color, lw=2, marker=marker, ms=5, mec='white', mew=1, label=label)
        a.annotate(label, (x[-1], m[-1]), xytext=(6, 0), textcoords='offset points', va='center', color=ink, fontsize=8)
    a.plot([34], [stitched['delta_mean']], marker='D', ms=6, color=stitch_c, mec='white', mew=1, ls='none',
           label='MACE stitched (34 s)')
    a.annotate('stitched', (34, stitched['delta_mean']), xytext=(-4, 12), textcoords='offset points', color=ink,
               fontsize=8, ha='right')
    a.text(8.3, a.get_ylim()[1], 'training length', color=muted, fontsize=7, va='top')
    a.set_ylabel('own-song Δ (slow treble ↔ body energy)')
    a.set_title('a  Music coupling of the whole dance', loc='left', color=ink, fontsize=10)
    a.legend(frameon=False, fontsize=7, loc='upper left', bbox_to_anchor=(0.0, 0.93))
    a.set_xlim(7, 38)

    g, lo, hi = [], [], []
    for L in Ls:
        d = np.array(res['real'][L]['whole']['per_song']) - np.array(res['model'][L]['whole']['per_song'])
        bt = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        g.append(d.mean()); lo.append(np.percentile(bt, 2.5)); hi.append(np.percentile(bt, 97.5))
    g, lo, hi = map(np.array, (g, lo, hi))
    b.axhline(0, color=muted, lw=1)
    b.errorbar(x, g, yerr=[g - lo, hi - g], color=model_c, lw=2, marker='s', ms=5, mec='white', mew=1, capsize=0,
               elinewidth=1.2)
    b.set_ylabel('gap: real − MACE one-shot')
    b.set_title('b  Gap to real dancers (95% CI, 19 songs)', loc='left', color=ink, fontsize=10)
    b.set_xlim(7, 35.5)
    fig.tight_layout()
    path = os.path.join(HERE, 'results', 'figures', 'length_curve.png')
    fig.savefig(path, facecolor='white')
    print('saved', path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--checkpoint')
    ap.add_argument('--label', default='released')
    ap.add_argument('--local_attn', type=float, help='seconds: restrict cross-attention to +- this around each frame')
    a = ap.parse_args()
    plot() if a.plot else main(a.checkpoint, a.label, a.local_attn)
