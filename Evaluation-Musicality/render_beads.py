"""Beat beads (2026-11-09): beat locking made visible in the dance itself.

Real dancer | MACE released | ours, same song, same seed, skeletons centred on the pelvis. Each panel draws a fading
trail behind both wrists and both ankles, as thick as the limb is fast, and drops a bead on every trail at every BEAT of
the song (the beat grid of the baseline features, column 34). A dancer locked to the beat puts the beads at the same
place of the swing each cycle: the real dancer's beads sit on the thick (fast) part, a constant lag only moves where.
Measured on 161_slice2 (z-scored limb speed at the beat vs shift of the grid): real 0.67 depth/2, released 0.10 (noise),
ours 0.24; limb POSITION at the beats clusters only 14 % tighter for real and 0-1 % for the others, so position beads
were not used. --own_phase puts each dancer's beads at its own best shift of the grid (printed), not at the beat.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/render_beads.py --ffmpeg $FF [--start 5 --dur 10] [--still 9.0]

Writes data/videos/beads_<slice>_<suffix>.mp4 (+ a still png with --still).
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
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import render_fix_compare as R  # noqa: E402
from eval_mace import fk  # noqa: E402
from model.metric import SMPLSkeleton  # noqa: E402
from render_eas_compare import LEFT, PARENTS, view_axes  # noqa: E402

FPS = 30
JOINTS = {7: 'L ankle', 8: 'R ankle', 20: 'L wrist', 21: 'R wrist'}
GOLD, GOLD_EDGE = (1.0, 0.82, 0.1), (0.25, 0.18, 0.0)
OURS_DIR = os.path.join(HERE, 'data', 'mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slice', default='161_slice2')
    ap.add_argument('--seed', type=int, default=6)
    ap.add_argument('--start', type=float, default=5.0, help='seconds')
    ap.add_argument('--dur', type=float, default=10.0, help='seconds; 0 = to the end')
    ap.add_argument('--trail', type=float, default=1.0, help='trail length in seconds')
    ap.add_argument('--keep', type=int, default=0, help='beats kept as faint dots in the body frame (0 = none)')
    ap.add_argument('--own_phase', action='store_true', help='drop each dancer\'s beads at its own best shift of the beat grid')
    ap.add_argument('--ours_dir', default=OURS_DIR)
    ap.add_argument('--ours_label', default='Ours: critic v5 + gain')
    ap.add_argument('--oneshot_label', default='MACE released')
    ap.add_argument('--suffix', default='real_released_ours')
    ap.add_argument('--still', type=float, default=None, help='also save a png of this second')
    ap.add_argument('--no_video', action='store_true')
    ap.add_argument('--slow', type=float, default=1.0, help='slow motion factor 1 or 2: the motion is linearly interpolated to 60 fps and played at 30 fps, audio slowed to match')
    ap.add_argument('--no_real', action='store_true', help='two panels: MACE released | ours (no real dancer)')
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'data', 'videos'))
    ap.add_argument('--ffmpeg', default='ffmpeg')
    a = ap.parse_args()

    n = a.slice
    names = json.load(open(os.path.join(R.ONESHOT, 'names.json')))['names']
    assert json.load(open(os.path.join(a.ours_dir, 'names.json')))['names'] == names
    i = names.index(n)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = pickle.load(open(os.path.join('data/test/motions_sliced', n + '.pkl'), 'rb'))
    gt = fk(SMPLSkeleton(device), [m['q']], [m['pos']], device, from_pred=False)[0]
    one, ours = R.load(names, R.ONESHOT, i)[a.seed], R.load(names, a.ours_dir, i)[a.seed]
    T = min(len(gt), one.shape[0], ours.shape[0])
    A = np.flatnonzero(np.load(os.path.join('data/test/baseline_feats', n + '.npy'))[:, 34])
    A = A[A < T]
    bpm = 60 * FPS / np.polyfit(np.arange(len(A)), A, 1)[0]
    f0 = int(round(a.start * FPS))
    nfr = T - f0 if a.dur <= 0 else min(int(round(a.dur * FPS)), T - f0)
    W = int(round(a.trail * FPS))
    defs = [(gt[:T], 'Real dancer', 'tab:blue')] if not a.no_real else []
    defs += [(one[:T], a.oneshot_label, 'tab:orange'), (ours[:T], a.ours_label, 'tab:green')]
    print(f'{n} seed {a.seed}: {bpm:.0f} BPM, {len(A)} beats, frames {f0}..{f0 + nfr}', flush=True)

    # shared limits so the three panels have the same scale
    P = []
    for joints, label, color in defs:
        h, up, sgn = view_axes(joints)
        rel = joints - joints[:, :1]
        P.append((np.stack([rel[..., h], sgn * rel[..., up]], -1), label, color))  # (T, 24, 2)
    xs = np.concatenate([p[0][..., 0].ravel() for p in P])
    ys = np.concatenate([p[0][..., 1].ravel() for p in P])
    xl, yl = (xs.min() - 0.15, xs.max() + 0.15), (ys.min() - 0.1, ys.max() + 0.1)

    # limb speed on screen (z-scored per dancer over all four limbs), shared width scale
    JL = list(JOINTS)
    SP = []
    for xy, label, color in P:
        v = np.linalg.norm(np.gradient(xy[:, JL], axis=0) * FPS, axis=-1)  # (T, 4)
        SP.append(gaussian_filter1d(v, 1.0, axis=0))
    ref = np.percentile(np.concatenate([x.ravel() for x in SP]), 95)
    iv = np.polyfit(np.arange(len(A)), A, 1)[0]
    shifts = np.linspace(0, 0.95, 20)
    BEATS = []
    for k, (xy, label, color) in enumerate(P):
        z = SP[k].mean(1); z = (z - z.mean()) / (z.std() + 1e-8)
        prof = np.array([z[np.clip(np.round(A + sh * iv).astype(int), 0, T - 1)].mean() for sh in shifts])
        best = shifts[prof.argmax()]
        print(f'  {label}: speed at beat {prof[0]:+.2f}; best shift {best:.2f} beat gives {prof.max():+.2f} '
              f'(modulation depth/2 {(prof.max() - prof.min()) / 2:.2f})', flush=True)
        BEATS.append(np.clip(np.round(A + (best if a.own_phase else 0) * iv).astype(int), 0, T - 1))
        if a.own_phase:
            P[k] = (xy, f'{label}   (beads {best:.2f} beat after the beat)', color)

    U = int(round(a.slow))
    if U > 1:  # slow motion: interpolate poses and speeds to U x the frame rate, play back at 30 fps
        def up(x):
            t = np.arange(U * len(x)) / U
            i0 = np.minimum(t.astype(int), len(x) - 1); i1 = np.minimum(i0 + 1, len(x) - 1); w = (t - i0).reshape((-1,) + (1,) * (x.ndim - 1))
            return x[i0] * (1 - w) + x[i1] * w
        P = [(up(xy), label, color) for xy, label, color in P]
        SP = [up(x) for x in SP]
        BEATS = [b * U for b in BEATS]
        A = A * U; W *= U; f0 *= U; nfr *= U
    npan = len(P)
    fig = plt.figure(figsize=(4.8 * npan + 0.005, 6.605), dpi=100)  # 1440 x 660 px: x264 needs even sizes
    arts = []
    for col, (xy, label, color) in enumerate(P):
        ax = fig.add_subplot(1, npan, col + 1)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=12)
        keep = ax.scatter([], [], s=14, c=[GOLD], edgecolors='none', zorder=3)
        bones = [ax.plot([], [], lw=1.6, color='tab:red' if c in LEFT else '0.35', zorder=2)[0] for c in range(1, 24)]
        trails = {j: LineCollection([], linewidths=2.6, zorder=4, capstyle='butt') for j in JOINTS}
        for lc in trails.values():
            ax.add_collection(lc)
        beads = ax.scatter([], [], s=60, c=[GOLD], edgecolors=[GOLD_EDGE], linewidths=1.2, zorder=6)
        lamp = ax.plot([xl[0] + 0.1 * (xl[1] - xl[0])], [yl[1] - 0.06 * (yl[1] - yl[0])], 'o', ms=15, color='0.88')[0]
        ax.text(xl[0] + 0.19 * (xl[1] - xl[0]), yl[1] - 0.06 * (yl[1] - yl[0]), 'beat', va='center', fontsize=9, color='0.4')
        arts.append((xy, color, keep, bones, trails, beads, lamp, col))
    sup = fig.suptitle('')
    fig.text(0.5, 0.015, 'trails = both wrists and ankles (last 1 s), thicker = faster · gold beads = where each limb was at every beat of the song '
             + ('· faint dots = the last %d beats' % a.keep if a.keep else ''), ha='center', fontsize=9, color='0.35')
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    def draw(fr):
        for xy, color, keep, bones, trails, beads, lamp, pi in arts:
            A_ = BEATS[pi]
            for c, ln in zip(range(1, 24), bones):
                p = PARENTS[c]
                ln.set_data([xy[fr, p, 0], xy[fr, c, 0]], [xy[fr, p, 1], xy[fr, c, 1]])
            lamp.set_color('#16213a' if ((fr - A >= 0) & (fr - A < 4 * U)).any() else '0.88')
            lo = max(0, fr - W)
            rgb = np.array(matplotlib.colors.to_rgb(color))
            for jn, (j, lc) in enumerate(trails.items()):
                pts = xy[lo:fr + 1, j]
                if len(pts) < 2:
                    lc.set_segments([]); continue
                lc.set_segments(np.stack([pts[:-1], pts[1:]], 1))
                age = (np.arange(len(pts) - 1) + 1) / W  # newest segment is the most opaque
                al = (0.15 + 0.85 * age)[:, None]  # opaque fade toward white: translucent segments would overlap into blobs
                lc.set_color(rgb * al + (1 - al))
                lc.set_linewidth(1.0 + 9.0 * np.clip(SP[pi][lo + 1:fr + 1, jn] / ref, 0, 1.15))
            recent = A_[(A_ >= lo) & (A_ <= fr)]
            if len(recent):
                off = np.concatenate([xy[recent, j] for j in JOINTS])
                age = np.tile(fr - recent, len(JOINTS))
                s = 45 + 130 * np.clip(1 - age / (6 * U), 0, 1)  # a bead pops in the beat it is dropped
                al = np.clip(1.0 - 0.6 * age / W, 0, 1)
                beads.set_offsets(off); beads.set_sizes(s)
                fc = np.tile(np.array(GOLD + (1,)), (len(off), 1)); fc[:, 3] = al
                ec = np.tile(np.array(GOLD_EDGE + (1,)), (len(off), 1)); ec[:, 3] = al
                beads.set_facecolors(fc); beads.set_edgecolors(ec)
            else:
                beads.set_offsets(np.empty((0, 2)))
            past = A_[A_ <= fr][-a.keep:] if a.keep else np.array([], int)
            if len(past):
                off = np.concatenate([xy[past, j] for j in JOINTS])
                rank = np.tile(np.arange(len(past)) + 1, len(JOINTS)) / max(1, a.keep)  # newest = 1
                fc = np.tile(np.array(GOLD + (1,)), (len(off), 1)); fc[:, 3] = 0.15 + 0.5 * rank
                keep.set_offsets(off); keep.set_facecolors(fc)
            else:
                keep.set_offsets(np.empty((0, 2)))
        sup.set_text(f'Beat beads   ·   song {n} ({bpm:.0f} BPM)   {fr / (FPS * U):4.1f} s')
        fig.canvas.draw()

    if a.still is not None:
        draw(int(round(a.still * FPS)) * U)
        p = os.path.join(a.out_dir, f'beads_{n}_{a.suffix}_still.png')
        fig.savefig(p); print(p)
    if not a.no_video:
        out = os.path.join(a.out_dir, f'beads_{n}_{a.suffix}.mp4')
        Wpx, Hpx = fig.canvas.get_width_height()
        wav = os.path.join('data/test/wavs_sliced', n + '.wav')
        vf = [] if U == 1 else ['-af', f'atempo={1 / U:.4f}']
        proc = subprocess.Popen([a.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{Wpx}x{Hpx}',
                                 '-r', str(FPS), '-i', '-', '-ss', f'{f0 / (FPS * U):.3f}', '-i', wav, '-map', '0:v', '-map', '1:a'] + vf + [
                                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-c:a', 'aac', '-shortest', out],
                                stdin=subprocess.PIPE)
        for fr in range(f0, f0 + nfr):
            draw(fr)
            proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        proc.stdin.close(); proc.wait()
        print(out)


if __name__ == '__main__':
    main()
