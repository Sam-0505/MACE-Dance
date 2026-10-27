"""Render one ImperialDance repeat as a stick figure with the music placed at the README start,
for checking by eye whether the motion and the source video line up.

    python render_imperial_check.py --choreo u_ch0 --level e --repeat 1 --ffmpeg /path/to/ffmpeg

Writes data/imperial/check/<level>_<choreo>_r<repeat>.mp4. Use --start to try another start time
(seconds into the source track) once you know where the choreography really begins.
"""
import argparse
import json
import os
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter as G

import audio_features as af
import metrics as M
from eval_imperial import load_motion

NAMES = ['RShin', 'RToe', 'RThigh', 'LThigh', 'LFoot', 'RShoulder', 'LHand', 'RHand', 'LFArm', 'LShoulder', 'RFoot',
         'Neck', 'LShin', 'Chest', 'Head', 'LToe', 'Ab', 'RUArm', 'LUArm', 'Hip', 'RFArm']
BONES = [('Hip', 'Ab'), ('Ab', 'Chest'), ('Chest', 'Neck'), ('Neck', 'Head')]
for s in 'LR':
    BONES += [('Chest', s + 'Shoulder'), (s + 'Shoulder', s + 'UArm'), (s + 'UArm', s + 'FArm'), (s + 'FArm', s + 'Hand'),
              ('Hip', s + 'Thigh'), (s + 'Thigh', s + 'Shin'), (s + 'Shin', s + 'Foot'), (s + 'Foot', s + 'Toe')]
BONES = [(NAMES.index(a), NAMES.index(b)) for a, b in BONES]
UP, LR_AXIS, FB_AXIS = 1, 2, 0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--choreo', required=True)
    ap.add_argument('--level', default='e')
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--start', type=float, default=None, help='music start in the source track (s); default README start')
    ap.add_argument('--motion_dir', default=os.path.join(here, 'data', 'imperial', 'motion'))
    ap.add_argument('--audio_dir', default=os.path.join(here, 'data', 'imperial', 'audio'))
    ap.add_argument('--out_dir', default=os.path.join(here, 'data', 'imperial', 'check'))
    ap.add_argument('--ffmpeg', default='ffmpeg')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    joints = load_motion(args.motion_dir, args.level, args.choreo)[args.repeat]          # (T, 21, 3) at 30 fps
    T = len(joints)
    with open(os.path.join(args.audio_dir, args.choreo + '.json')) as f:
        meta = json.load(f)
    start = meta['readme_start'] if args.start is None else args.start
    y = af.load_audio(os.path.join(args.audio_dir, args.choreo + '.wav'))
    off = int(round((start - meta['window_start']) * af.SR))
    y = y[off:off + int(T / M.FPS * af.SR)]
    feat = af.extract_baseline(y)[:T]
    beats = np.flatnonzero(feat[:, af.BEAT]) / M.FPS
    speed = G(np.linalg.norm(joints[1:] - joints[:-1], axis=2).mean(1), 2)
    t = np.arange(len(speed)) / M.FPS

    name = f'{args.level}_{args.choreo}_r{args.repeat}' + ('' if args.start is None else f'_start{start:g}')
    wav = os.path.join(args.out_dir, name + '.wav')
    sf.write(wav, y, af.SR)

    lo, hi = joints.reshape(-1, 3).min(0), joints.reshape(-1, 3).max(0)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    views = [(fig.add_subplot(gs[0, 0]), LR_AXIS, 'front (left-right vs up)'), (fig.add_subplot(gs[0, 1]), FB_AXIS, 'side')]
    strip = fig.add_subplot(gs[1, :])
    strip.plot(t, speed / speed.max(), lw=1, color='tab:blue', label='motion speed')
    strip.plot(np.arange(T) / M.FPS, G(feat[:, af.ENVELOPE], 2) / (G(feat[:, af.ENVELOPE], 2).max() + 1e-8),
               lw=1, color='tab:orange', alpha=0.7, label='music onsets')
    for b in beats:
        strip.axvline(b, color='gray', lw=0.5, alpha=0.5)
    strip.set_xlim(0, T / M.FPS)
    strip.set_yticks([])
    strip.set_xlabel('seconds into motion (grey lines = music beats)')
    strip.legend(loc='upper right', fontsize=7)
    cursor = strip.axvline(0, color='red', lw=1.5)
    lines = []
    for ax, h, title in views:
        ax.set_xlim(lo[h] - 0.2, hi[h] + 0.2)
        ax.set_ylim(lo[UP] - 0.05, hi[UP] + 0.1)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        lines.append([ax.plot([], [], lw=2, color='tab:red' if NAMES[b][0] == 'L' else 'k')[0] for a, b in BONES])
    label = fig.suptitle('')
    fig.tight_layout()

    video = os.path.join(args.out_dir, name + '.mp4')
    proc = subprocess.Popen([args.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
                             '-s', '800x600', '-r', str(M.FPS), '-i', '-', '-i', wav, '-c:v', 'libx264',
                             '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', video], stdin=subprocess.PIPE)
    for f in range(T):
        for (ax, h, _), ls in zip(views, lines):
            for (a, b), ln in zip(BONES, ls):
                ln.set_data([joints[f, a, h], joints[f, b, h]], [joints[f, a, UP], joints[f, b, UP]])
        cursor.set_xdata([f / M.FPS])
        label.set_text(f'{args.choreo} level={args.level} repeat {args.repeat} | motion {f / M.FPS:5.2f}s | '
                       f'source track {start + f / M.FPS:6.2f}s')
        fig.canvas.draw()
        proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    proc.stdin.close()
    proc.wait()
    os.remove(wav)
    print('wrote', video)


if __name__ == '__main__':
    main()
