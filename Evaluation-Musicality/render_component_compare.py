"""GT vs MACE stick-figure videos for the fast multi-component pairings (percussive, low-band flux,
high-band flux), using exactly the features and scores of multicomponent.py.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/render_component_compare.py --ffmpeg /path/to/ffmpeg

For each component the example is ILLUSTRATIVE, not random: the song with the largest
(Delta_real - Delta_MACE) in results/multicomponent.json, and within it the slice where the real
dancer's own-song score is highest. MACE shows the seed with the median score for that slice.
The joints a component measures are highlighted; the strip scrolls an 8 s window because these
signals change within fractions of a second.
Writes data/videos/<component>_<slice>_gt_vs_mace.mp4.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

import metrics as M
import multicomponent as MC
from eval_mace import fk
from model.metric import SMPLSkeleton
from render_eas_compare import PARENTS, view_axes, zs

COMPONENTS = {
    'perc': ('Percussive energy', 'upper-body acceleration', MC.UPPER),
    'lowflux': ('Bass changes (50–200 Hz flux)', 'hip and leg speed', MC.LOWER),
    'highflux': ('Treble changes (6.4–12.8 kHz flux)', 'head and hand speed', MC.HEADHAND),
}
WINDOW = 8.0   # seconds of strip shown around the cursor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--components', nargs='*', default=list(COMPONENTS))
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'data', 'videos'))
    ap.add_argument('--ffmpeg', default='ffmpeg')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(MC.OUT) as f:
        per_song = json.load(f)['per_song']
    with open(os.path.join(MC.MACE_DIR, 'names.json')) as f:
        names = json.load(f)['names']
    mace_all = np.load(os.path.join(MC.MACE_DIR, 'joints.npy'), mmap_mode='r')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(device)

    for comp in args.components:
        title, motion_label, joints_idx = COMPONENTS[comp]
        sigma, mkey = MC.SIGMA[comp], MC.MOTION_OF[comp]
        gaps = {s: (v['real_own'] - v['real_other']) - (v['mace_own'] - v['mace_other'])
                for s, v in per_song[comp].items()}
        song = max(gaps, key=gaps.get)

        cands = []
        for i, n in enumerate(names):
            if n.split('_')[0] != song:
                continue
            mus = MC.smooth(MC.music_features(n)[comp], sigma)
            with open(os.path.join(MC.GT_DIR, 'motions_sliced', n + '.pkl'), 'rb') as f:
                m = pickle.load(f)
            gt = fk(smpl, [m['q']], [m['pos']], device, from_pred=False)[0][:MC.T]
            seeds = np.asarray(mace_all[i], dtype=np.float32)[:, :MC.T]
            gt_mot = MC.smooth(MC.motion_features(gt)[mkey], sigma)
            mace_mots = [MC.smooth(MC.motion_features(j)[mkey], sigma) for j in seeds]
            s_gt = MC.score(mus, gt_mot)[1]
            s_mace = [MC.score(mus, x)[1] for x in mace_mots]
            k = int(np.argsort(s_mace)[len(s_mace) // 2])
            cands.append((s_gt, n, gt, seeds[k], k, s_mace[k], s_mace, mus, gt_mot, mace_mots[k]))
        s_gt, n, gt, mace, k, s_m, s_all, mus, gt_mot, mace_mot = max(cands, key=lambda c: c[0])

        wav = os.path.join(MC.GT_DIR, 'wavs_sliced', n + '.wav')
        t = np.arange(MC.T) / M.FPS
        jset = set(joints_idx)
        fig = plt.figure(figsize=(9.6, 6.4), dpi=100)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.3])
        panels = []
        for col, (joints, label) in enumerate([(gt, f'Real dancer   S = {s_gt:+.2f}'),
                                               (mace, f'MACE (seed {k}, median of 10)   S = {s_m:+.2f}')]):
            ax = fig.add_subplot(gs[0, col])
            h, up, sgn = view_axes(joints)
            rel = joints - joints[:, :1]
            ax.set_xlim(rel[..., h].min() - 0.2, rel[..., h].max() + 0.2)
            v = sgn * rel[..., up]
            ax.set_ylim(v.min() - 0.1, v.max() + 0.1)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(label, fontsize=10)
            hot = 'tab:blue' if col == 0 else 'tab:orange'
            lines = [ax.plot([], [], lw=3.5 if c in jset else 1.5,
                             color=hot if c in jset else '0.55')[0] for c in range(1, 24)]
            panels.append((rel, h, up, sgn, lines))
        strip = fig.add_subplot(gs[1, :])
        strip.plot(t, zs(mus), color='k', lw=1.2, label=title.split(' (')[0].lower())
        strip.plot(t, zs(gt_mot), color='tab:blue', lw=1.1, label=f'real: {motion_label}')
        strip.plot(t, zs(mace_mot), color='tab:orange', lw=1.1, label=f'MACE: {motion_label}')
        strip.set_yticks([])
        strip.set_xlabel('seconds (z-scored; S = correlation with the music curve, ±0.3 s lag)')
        strip.legend(loc='upper right', fontsize=7, ncol=3)
        cursor = strip.axvline(0, color='red', lw=1.5)
        fig.suptitle('')
        sup = fig.texts[0] if fig.texts else fig.suptitle('')
        fig.tight_layout()

        out = os.path.join(args.out_dir, f'{comp}_{n}_gt_vs_mace.mp4')
        proc = subprocess.Popen([args.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
                                 '-s', '960x640', '-r', str(M.FPS), '-i', '-', '-i', wav, '-c:v', 'libx264',
                                 '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', out], stdin=subprocess.PIPE)
        for fr in range(MC.T):
            for rel, h, up, sgn, lines in panels:
                for c, ln in zip(range(1, 24), lines):
                    p = PARENTS[c]
                    ln.set_data([rel[fr, p, h], rel[fr, c, h]], [sgn * rel[fr, p, up], sgn * rel[fr, c, up]])
            now = fr / M.FPS
            lo = min(max(0.0, now - WINDOW / 2), t[-1] - WINDOW)
            strip.set_xlim(lo, lo + WINDOW)
            cursor.set_xdata([now])
            sup.set_text(f'{title} ↔ {motion_label}   ·   song {n}   {now:4.1f} s   (illustrative example)')
            fig.canvas.draw()
            proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        proc.stdin.close()
        proc.wait()
        plt.close(fig)
        print(f'{out}\n  song {song} (largest real-vs-MACE gap, {gaps[song]:+.2f}); slice {n}: real S {s_gt:+.2f}, '
              f'MACE seed {k} S {s_m:+.2f} (10 seeds {min(s_all):+.2f}..{max(s_all):+.2f})')


if __name__ == '__main__':
    main()
