"""Beat photo strip (2026-11-09): one pose per beat for real | MACE released | ours, consecutive beats of one window, plus an
overlay column with all of the window's poses on top of each other (a tight overlay = the dancer lands in the same place
every beat). Same motions as render_beads.py (161_slice2, seed 6). Prints the pose spread across the window's beats and the
spread for the grid shifted by 0.0..0.95 beat, over all 8-beat windows of the clip, so the picture can be checked against a
number.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/render_beat_photos.py [--start 5.0] [--nbeats 8] [--own_phase]
"""
import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import render_fix_compare as R  # noqa: E402
from eval_mace import fk  # noqa: E402
from model.metric import SMPLSkeleton  # noqa: E402
from render_eas_compare import LEFT, PARENTS, view_axes  # noqa: E402

FPS = 30
LIMBS = [7, 8, 20, 21]
OURS_DIR = os.path.join(HERE, 'data', 'mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10')


def spread(xy, t):
    """RMS distance of the poses at frames t from their mean pose (all 24 joints, pelvis-relative, screen plane)."""
    p = xy[t]
    return float(np.sqrt(((p - p.mean(0)) ** 2).sum(-1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slice', default='161_slice2')
    ap.add_argument('--seed', type=int, default=6)
    ap.add_argument('--start', type=float, default=5.0)
    ap.add_argument('--nbeats', type=int, default=8)
    ap.add_argument('--own_phase', action='store_true', help='each dancer at its own best shift of the beat grid')
    ap.add_argument('--no_real', action='store_true')
    ap.add_argument('--ours_dir', default=OURS_DIR)
    ap.add_argument('--ours_label', default='Ours: critic v5 + gain')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'data', 'videos'))
    a = ap.parse_args()

    n = a.slice
    names = json.load(open(os.path.join(R.ONESHOT, 'names.json')))['names']
    i = names.index(n)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = pickle.load(open(os.path.join('data/test/motions_sliced', n + '.pkl'), 'rb'))
    gt = fk(SMPLSkeleton(dev), [m['q']], [m['pos']], dev, from_pred=False)[0]
    one, ours = R.load(names, R.ONESHOT, i), R.load(names, a.ours_dir, i)
    T = min(len(gt), one.shape[1], ours.shape[1])
    A = np.flatnonzero(np.load(os.path.join('data/test/baseline_feats', n + '.npy'))[:, 34])
    A = A[A < T - 1]
    iv = np.polyfit(np.arange(len(A)), A, 1)[0]
    bpm = 60 * FPS / iv

    def to_xy(j):
        h, up, s = view_axes(j[:T])
        rel = j[:T] - j[:T, :1]
        return np.stack([rel[..., h], s * rel[..., up]], -1)

    shifts = np.linspace(0, 0.95, 20)

    def grid(xy, sh):
        return np.clip(np.round(A + sh * iv).astype(int), 0, T - 1)

    def best_shift(xy):
        v = gaussian_filter1d(np.linalg.norm(np.gradient(xy[:, LIMBS], axis=0) * FPS, axis=-1), 1.0, axis=0).mean(1)
        z = (v - v.mean()) / (v.std() + 1e-8)
        return shifts[np.argmax([z[grid(xy, s)].mean() for s in shifts])]

    rows = [('Real dancer', to_xy(gt), 'tab:blue')] if not a.no_real else []
    rows += [('MACE released', to_xy(one[a.seed]), 'tab:orange'), (a.ours_label, to_xy(ours[a.seed]), 'tab:green')]
    k0 = int(np.searchsorted(A, a.start * FPS))
    K = a.nbeats
    fig, axes = plt.subplots(len(rows), K + 1, figsize=(1.9 * (K + 1), 3.0 * len(rows)))
    allxy = np.concatenate([r[1].reshape(-1, 2) for r in rows])
    xl, yl = (allxy[:, 0].min() - 0.1, allxy[:, 0].max() + 0.1), (allxy[:, 1].min() - 0.05, allxy[:, 1].max() + 0.05)
    print(f'{n} seed {a.seed}  {bpm:.0f} BPM  window beats {k0}..{k0 + K - 1} (from {A[k0] / FPS:.1f} s)')
    print(f'{"dancer":24s} {"window":>8s} {"all 8-beat windows":>20s} {"grid shifted (mean/min/max)":>30s}')
    for r, (label, xy, color) in enumerate(rows):
        sh = best_shift(xy) if a.own_phase else 0.0
        t_w = grid(xy, sh)[k0:k0 + K]
        # spread of the 8-beat windows on the chosen grid vs on every shifted grid
        def win_mean(s):
            g = grid(xy, s)
            return np.mean([spread(xy, g[k:k + K]) for k in range(0, len(g) - K, 2)])
        base = win_mean(sh)
        allv = np.array([win_mean(s) for s in shifts])
        print(f'{label:24s} {spread(xy, t_w):8.3f} {base:20.3f} {allv.mean():12.3f}/{allv.min():.3f}/{allv.max():.3f}'
              f'   -> grid {base / allv.mean():.2f}x the shifted-grid mean' + (f'   (own shift {sh:.2f})' if a.own_phase else ''))
        for c in range(K):
            ax = axes[r, c]
            f = t_w[c]
            for cc in range(1, 24):
                p = PARENTS[cc]
                ax.plot([xy[f, p, 0], xy[f, cc, 0]], [xy[f, p, 1], xy[f, cc, 1]], lw=1.8, color='tab:red' if cc in LEFT else '0.3')
            ax.plot(xy[f, LIMBS, 0], xy[f, LIMBS, 1], 'o', ms=5, color='#ffd11a', mec='#403000', mew=0.8, zorder=5)
            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'beat {c + 1}\n{f / FPS:.1f} s', fontsize=9)
        ax = axes[r, K]
        for c in range(K):
            f = t_w[c]
            for cc in range(1, 24):
                p = PARENTS[cc]
                ax.plot([xy[f, p, 0], xy[f, cc, 0]], [xy[f, p, 1], xy[f, cc, 1]], lw=1.2, color=color, alpha=0.3)
            ax.plot(xy[f, LIMBS, 0], xy[f, LIMBS, 1], 'o', ms=4, color='#ffd11a', mec='#403000', mew=0.5, alpha=0.8, zorder=5)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('#f6f6f6')
        ax.set_xlabel(f'pose spread {spread(xy, t_w):.3f} (lower = same place)', fontsize=8)
        if r == 0:
            ax.set_title('all %d beats\noverlaid' % K, fontsize=9, fontweight='bold')
        axes[r, 0].set_ylabel(label + (f'\n(+{sh:.2f} beat)' if a.own_phase else ''), fontsize=11)
    fig.suptitle(f'One pose per beat, {K} consecutive beats of song {n} ({bpm:.0f} BPM)'
                 + ('  ·  each dancer at its own phase of the beat' if a.own_phase else '  ·  at the beat of the song'), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(a.out_dir, f'beatphotos_{n}{a.suffix}{"_ownphase" if a.own_phase else ""}.png')
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == '__main__':
    main()
