"""Side-by-side GT vs MACE stick-figure videos for every slice of one test song, with the music,
and a strip plotting music loudness against each dancer's motion energy (what EAS correlates).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/render_eas_compare.py --song 012 --ffmpeg /path/to/ffmpeg

MACE comes from data/mace_test/joints.npy (gen_mace_test.py); for each slice the seed with the
median EAS is shown, so the video is a typical sample rather than a hand-picked one.
Writes data/videos/<song>_sliceK_gt_vs_mace.mp4.
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
from scipy.ndimage import gaussian_filter as G

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

import audio_features as af
import metrics as M
from eval_mace import fk
from model.metric import SMPLSkeleton

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 22}
HEAD, ANKLES, SHOULDERS = 15, (7, 8), (16, 17)


def view_axes(joints):
    """(horizontal, up, up_sign) for a front view, inferred from the skeleton itself."""
    up_vec = (joints[:, HEAD] - joints[:, list(ANKLES)].mean(1)).mean(0)
    up = int(np.argmax(np.abs(up_vec)))
    side = np.abs(joints[:, SHOULDERS[0]] - joints[:, SHOULDERS[1]]).mean(0)
    side[up] = -1
    return int(np.argmax(side)), up, float(np.sign(up_vec[up]))


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--song', default='012')
    ap.add_argument('--gt_dir', default='data/test')
    ap.add_argument('--mace', default=os.path.join(HERE, 'data', 'mace_test'))
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'data', 'videos'))
    ap.add_argument('--ffmpeg', default='ffmpeg')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.mace, 'names.json')) as f:
        names = json.load(f)['names']
    mace_all = np.load(os.path.join(args.mace, 'joints.npy'), mmap_mode='r')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(device)

    for i, n in enumerate(names):
        if n.split('_')[0] != args.song:
            continue
        feat = np.load(os.path.join(args.gt_dir, 'baseline_feats', n + '.npy'))
        wav = os.path.join(args.gt_dir, 'wavs_sliced', n + '.wav')
        rms = af.rms_envelope(af.load_audio(wav))
        with open(os.path.join(args.gt_dir, 'motions_sliced', n + '.pkl'), 'rb') as f:
            m = pickle.load(f)
        gt = fk(smpl, [m['q']], [m['pos']], device, from_pred=False)[0]
        seeds = np.asarray(mace_all[i], dtype=np.float32)                       # (S, T, 24, 3)
        eas_seeds = [M.clip_scores(feat, rms, j)['EAS'] for j in seeds]
        s = int(np.argsort(eas_seeds)[len(eas_seeds) // 2])
        mace = seeds[s]
        eas_gt = M.clip_scores(feat, rms, gt)['EAS']
        T = min(len(gt), len(mace), len(rms))

        loud = zs(G(rms[:T], M.ENERGY_SMOOTH))
        e_gt = zs(G(M.motion_energy(gt[:T]), M.ENERGY_SMOOTH))
        e_mace = zs(G(M.motion_energy(mace[:T]), M.ENERGY_SMOOTH))
        t = np.arange(T) / M.FPS

        fig = plt.figure(figsize=(9.6, 6.4), dpi=100)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.3])
        panels = []
        for col, (joints, title) in enumerate([(gt, f'Real dancer (GT)   EAS {eas_gt:+.2f}'),
                                               (mace, f'MACE (seed {s}, median of 10)   EAS {eas_seeds[s]:+.2f}')]):
            ax = fig.add_subplot(gs[0, col])
            h, up, sgn = view_axes(joints)
            # centre on the pelvis each frame so travel across the floor doesn't push the figure out of view
            rel = joints - joints[:, :1]
            ax.set_xlim(rel[..., h].min() - 0.2, rel[..., h].max() + 0.2)
            v = sgn * rel[..., up]
            ax.set_ylim(v.min() - 0.1, v.max() + 0.1)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=10)
            lines = [ax.plot([], [], lw=2, color='tab:red' if c in LEFT else 'k')[0] for c in range(1, 24)]
            panels.append((rel, h, up, sgn, lines))
        strip = fig.add_subplot(gs[1, :])
        strip.plot(t, loud, color='k', lw=1.5, label='music loudness')
        strip.plot(t, e_gt, color='tab:blue', lw=1.2, label=f'GT motion energy (EAS {eas_gt:+.2f})')
        strip.plot(t, e_mace, color='tab:orange', lw=1.2, label=f'MACE motion energy (EAS {eas_seeds[s]:+.2f})')
        strip.set_xlim(0, T / M.FPS); strip.set_yticks([])
        strip.set_xlabel('seconds (curves z-scored, 0.5 s smoothing; EAS = correlation with loudness)')
        strip.legend(loc='upper right', fontsize=7, ncol=3)
        cursor = strip.axvline(0, color='red', lw=1.5)
        sup = fig.suptitle('')
        fig.tight_layout()

        out = os.path.join(args.out_dir, f'{n}_gt_vs_mace.mp4')
        proc = subprocess.Popen([args.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
                                 '-s', '960x640', '-r', str(M.FPS), '-i', '-', '-i', wav, '-c:v', 'libx264',
                                 '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', out], stdin=subprocess.PIPE)
        for f in range(T):
            for rel, h, up, sgn, lines in panels:
                for c, ln in zip(range(1, 24), lines):
                    p = PARENTS[c]
                    ln.set_data([rel[f, p, h], rel[f, c, h]], [sgn * rel[f, p, up], sgn * rel[f, c, up]])
            cursor.set_xdata([f / M.FPS])
            sup.set_text(f'song {args.song} {n.split("_")[1]}   {f / M.FPS:5.1f} s')
            fig.canvas.draw()
            proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        proc.stdin.close()
        proc.wait()
        plt.close(fig)
        print(f'{out}  GT EAS {eas_gt:+.2f} | MACE seed {s} EAS {eas_seeds[s]:+.2f} '
              f'(10 seeds: {min(eas_seeds):+.2f}..{max(eas_seeds):+.2f})')


if __name__ == '__main__':
    main()
