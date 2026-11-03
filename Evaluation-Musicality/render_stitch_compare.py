"""Real dancer vs one-shot MACE vs stitched MACE, side by side with the music (2026-11-02).

Three test slices picked at random (numpy seed 0, three different songs), model seed 0, NOT chosen by score.
Front-view stick figures; the clip's mean pelvis position is subtracted, so travel stays visible. The strip shows
slow treble flux and each dancer's slow body energy (sigma 15, z-scored); grey bands mark the stitched windows'
overlaps (where two windows are crossfaded).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/render_stitch_compare.py --ffmpeg /path/to/ffmpeg

Writes data/videos/stitch_<slice>_real_oneshot_stitched.mp4.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from EDGE import EDGE  # noqa: E402
from model import musicality as MU  # noqa: E402
from accent_overlap import full_music  # noqa: E402
from render_eas_compare import PARENTS, view_axes, zs  # noqa: E402

FPS, T, WINDOW, STRIDE, STRIP = 30, 1020, 240, 130, 8.0


def slow(x):
    return MU.gaussian_smooth(torch.as_tensor(np.asarray(x, dtype=np.float64))[None], 15)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0, help='model seed shown')
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'data', 'videos'))
    ap.add_argument('--ffmpeg', default='ffmpeg')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = EDGE('baseline', checkpoint_path='runs/train/exp/weights/train-3750.pt')
    dt = pickle.load(open('data/dataset_backups/test_tensor_dataset.pkl', 'rb')).data
    names = [os.path.splitext(os.path.basename(f))[0] for f in dt['filenames']]
    one = np.load(os.path.join(HERE, 'data', 'mace_test', 'joints.npy'), mmap_mode='r')
    sti = np.load(os.path.join(HERE, 'data', 'mace_test_stitched', 'joints.npy'), mmap_mode='r')
    assert json.load(open(os.path.join(HERE, 'data', 'mace_test_stitched', 'names.json')))['names'] == names

    rng = np.random.default_rng(0)
    songs = rng.choice(sorted({n.split('_slice')[0] for n in names}), args.n, replace=False)
    picks = [int(rng.choice([i for i, n in enumerate(names) if n.startswith(s + '_')])) for s in songs]
    overlaps = [(k * STRIDE / FPS, (WINDOW + (k - 1) * STRIDE) / FPS) for k in range(1, (T - WINDOW) // STRIDE + 1)]

    for i in picks:
        n = names[i]
        with torch.no_grad():
            real = MU.pose_to_joints(dt['pose'][i:i + 1].cuda(), model.normalizer, model.diffusion.smpl)[0].cpu().numpy()
        clips = [(real, 'Real dancer', 'tab:blue'),
                 (np.asarray(one[i, args.seed], dtype=np.float32), 'MACE one-shot', 'tab:orange'),
                 (np.asarray(sti[i, args.seed], dtype=np.float32), 'MACE stitched', 'tab:green')]
        wav = os.path.join('data/test/wavs_sliced', n + '.wav')
        music = slow(full_music(wav)['treble'][:T])
        t = np.arange(T) / FPS

        fig = plt.figure(figsize=(14.4, 6.4), dpi=100)
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.2])
        panels = []
        h, up, sgn = view_axes(real)
        for col, (j, label, color) in enumerate(clips):
            ax = fig.add_subplot(gs[0, col])
            rel = j - j[:, :1].mean(0, keepdims=True)
            ax.set_xlim(-1.6, 1.6)
            v = sgn * rel[..., up]
            ax.set_ylim(v.min() - 0.1, v.max() + 0.1)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            energy = slow(MU.motion_energy(torch.from_numpy(j[None]))[0].numpy())
            r = np.corrcoef(music, energy)[0, 1]
            ax.set_title(f'{label}   r(treble, energy) = {r:+.2f}', fontsize=10)
            lines = [ax.plot([], [], lw=2, color=color)[0] for _ in range(1, 24)]
            panels.append((rel, lines, energy, color, label))
        strip = fig.add_subplot(gs[1, :])
        for a, b in overlaps:
            strip.axvspan(a, b, color='0.9', lw=0)
        strip.plot(t, zs(music), color='k', lw=1.4, label='slow treble flux')
        for _, _, energy, color, label in panels:
            strip.plot(t, zs(energy), color=color, lw=1.1, label=f'{label}: body energy')
        strip.set_yticks([]); strip.set_xlim(0, T / FPS)
        strip.set_xlabel('seconds (z-scored, sigma 15 smoothing; grey = stitched overlaps)')
        strip.legend(loc='upper right', fontsize=7, ncol=4)
        cursor = strip.axvline(0, color='red', lw=1.5)
        sup = fig.suptitle('')
        fig.tight_layout()

        out = os.path.join(args.out_dir, f'stitch_{n}_real_oneshot_stitched.mp4')
        proc = subprocess.Popen([args.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
                                 '-s', '1440x640', '-r', str(FPS), '-i', '-', '-i', wav, '-c:v', 'libx264',
                                 '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', out], stdin=subprocess.PIPE)
        for fr in range(T):
            for rel, lines, *_ in panels:
                for c, ln in zip(range(1, 24), lines):
                    p = PARENTS[c]
                    ln.set_data([rel[fr, p, h], rel[fr, c, h]], [sgn * rel[fr, p, up], sgn * rel[fr, c, up]])
            cursor.set_xdata([fr / FPS])
            win = min(fr // STRIDE, (T - WINDOW) // STRIDE)
            sup.set_text(f'song {n}   {fr / FPS:4.1f} s   (random slice, model seed {args.seed}; '
                         f'stitched window {win + 1} of {(T - WINDOW) // STRIDE + 1})')
            fig.canvas.draw()
            proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        proc.stdin.close()
        proc.wait()
        plt.close(fig)
        print(out, flush=True)


if __name__ == '__main__':
    main()
