"""Stitch the Appearance Expert videos of 198_slice0, 5.0-15.2 s, side by side (real | MACE released | ours: critic v5 + gain)
with the song and the loudness graph under them (2026-11-08).

Each video is two relay chunks (Expert-Appearance/outputs/gain198, appearance_relay.py): 77 + 76 frames at 15 fps = 10.2 s;
chunk 1's frame 0 repeats chunk 0's last frame and is dropped. The strip is the graph of loudness_198_slice0_*.mp4 (music
loudness against whole-body motion energy, z-scored, 0.5 s smoothing) cut to the window, with a cursor; the EAS in the
labels is the full 34 s clip's, as in that video. Two steps, because the venvs differ:

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/stitch_appearance_gain.py strip
    cd ../Motion-Projector   # any venv with imageio + pillow
    .venv/bin/python ../Evaluation-Musicality/stitch_appearance_gain.py stitch --ffmpeg /path/to/ffmpeg

Writes Expert-Appearance/outputs/gain198/loudness_198_slice0_stitched.mp4.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
OUT = os.path.join(ROOT, 'Expert-Appearance', 'outputs', 'gain198')
WAV = os.path.join(ROOT, 'Expert-Motion', 'data', 'test', 'wavs_sliced', '198_slice0.wav')
N, SEED, START = '198_slice0', 3, 150  # motion frame (30 fps) of the first video frame = 5.0 s
NFR, FPS = 153, 15
PH = 800
PW = PH * 720 // 1280
HEAD, STRIP_H = 150, 230
ONE_DIR = 'data/mace_test'
OURS_DIR = 'data/mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10'
COLORS = {'real': 'tab:blue', 'oneshot': 'tab:orange', 'gain': 'tab:green'}
LABELS = {'real': 'Real dancer (mocap)', 'oneshot': 'MACE released', 'gain': 'Ours: critic v5 + gain'}
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
AX = (0.045, 0.30, 0.945, 0.66)  # strip axes in figure fractions


def strip(full=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from scipy.ndimage import gaussian_filter as G
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.getcwd())
    import render_fix_compare as R
    from eval_mace import fk
    from model.metric import SMPLSkeleton
    names = json.load(open(os.path.join(HERE, ONE_DIR.replace('data/', 'data/'), 'names.json')))['names']
    i = names.index(N)
    smpl = SMPLSkeleton('cuda')
    m = pickle.load(open(f'data/test/motions_sliced/{N}.pkl', 'rb'))
    gt = fk(smpl, [m['q']], [m['pos']], 'cuda', from_pred=False)[0]
    one = R.load(names, os.path.join(HERE, ONE_DIR), i)[SEED]
    ours = R.load(names, os.path.join(HERE, OURS_DIR), i)[SEED]
    ex = R.loudness_example(N, gt, one[None], ours[None])
    T = ex['T']
    clips = {'real': gt[:T], 'oneshot': one[:T], 'gain': ours[:T]}
    t = np.arange(T) / 30
    lo, hi = (0.0, T / 30) if full else (START / 30, (START + NFR * 2) / 30)  # full: whole clip, static graph (pink mannequin video)
    fig = plt.figure(figsize=(PW * 3 / 100, STRIP_H / 100), dpi=100)
    ax = fig.add_axes(AX)
    ax.plot(t, ex['music'], color='k', lw=2, label='music loudness')
    scores = {}
    for k, j in clips.items():
        ax.plot(t, ex['curve'](j), color=COLORS[k], lw=1.4, label=LABELS[k])
        scores[k] = float(ex['score'](j))
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.set_xlabel('seconds of the clip (z-scored, 0.5 s smoothing)', fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(loc='upper center', ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.28), frameon=False)
    suffix = '_full' if full else ''
    fig.savefig(os.path.join(OUT, f'strip{suffix}.png'))
    json.dump({'scores': scores, 'xlim': [lo, hi]}, open(os.path.join(OUT, f'strip{suffix}.json'), 'w'))
    print('strip', scores)


def stitch(ffmpeg):
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    info = json.load(open(os.path.join(OUT, 'strip.json')))
    sp = Image.open(os.path.join(OUT, 'strip.png')).convert('RGB')
    lo, hi = info['xlim']
    W = PW * 3
    frames = {}
    for v in COLORS:
        a = [f for f in imageio.get_reader(os.path.join(OUT, f'loudness_198_{v}_c0.mp4'))]
        b = [f for f in imageio.get_reader(os.path.join(OUT, f'loudness_198_{v}_c1.mp4'))]
        frames[v] = a + b[1:]
        print(v, len(a), len(b), '->', len(frames[v]))
    nfr = min(min(len(x) for x in frames.values()), NFR)
    f_big, f_small = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 22)
    silent = os.path.join(OUT, 'stitched_noaudio.mp4')
    wr = imageio.get_writer(silent, fps=FPS, quality=8, macro_block_size=1)
    for k in range(nfr):
        canvas = Image.new('RGB', (W, HEAD + PH + STRIP_H), 'white')
        d = ImageDraw.Draw(canvas)
        d.text((16, 12), 'Music loudness ↔ whole-body motion energy', fill='black', font=f_big)
        d.text((16, 48), f'song {N}, seconds {START / 30:.1f}-{START / 30 + nfr / FPS:.1f} of the 34 s clip. EAS = correlation '
               'with loudness over the', fill=(90, 90, 90), font=f_small)
        d.text((16, 76), 'full clip (the same seed-3 dances as in the skeleton video).', fill=(90, 90, 90), font=f_small)
        for c, v in enumerate(COLORS):
            canvas.paste(Image.fromarray(frames[v][k]).resize((PW, PH)), (c * PW, HEAD))
            d.text((c * PW + 16, 112), f'{LABELS[v]}   EAS {info["scores"][v]:+.2f}', fill='black', font=f_small)
        canvas.paste(sp, (0, HEAD + PH))
        now = START / 30 + k / FPS
        x = int(W * (AX[0] + AX[2] * (now - lo) / (hi - lo)))
        d.line([(x, HEAD + PH + int(STRIP_H * (1 - AX[1] - AX[3]))), (x, HEAD + PH + int(STRIP_H * (1 - AX[1])))],
               fill='red', width=3)
        wr.append_data(np.asarray(canvas))
    wr.close()
    out = os.path.join(OUT, 'loudness_198_slice0_stitched.mp4')
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', silent, '-ss', f'{START / 30:.3f}', '-t', f'{nfr / FPS:.3f}',
                    '-i', WAV, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                    '-c:a', 'aac', '-b:a', '192k', '-shortest', out], check=True)
    os.remove(silent)
    print(out, f'{nfr} frames, {nfr / FPS:.2f} s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['strip', 'stitch'])
    ap.add_argument('--ffmpeg', default='ffmpeg')
    ap.add_argument('--full', action='store_true', help='strip over the whole 34 s clip (render_pink_stitch.py --full)')
    a = ap.parse_args()
    strip(a.full) if a.mode == 'strip' else stitch(a.ffmpeg)
