"""Plain pink mannequin version of the 198_slice0 loudness comparison (2026-11-08): real | MACE released | ours (critic v5 +
gain), 5.0-15.0 s of the clip at 30 fps, the loudness graph under the three figures, the song as audio.

No Appearance Expert: the SMPL mesh of each motion (the Motion Projector's own mesh render) is shown directly as a
featureless pink body on a pale background, so no person's look (face, clothes, scene) is added and nothing is generated
by Wan. Same motions as the skeleton and appearance videos (seed 3). The graph is Expert-Appearance/outputs/gain198/strip.png
(stitch_appearance_gain.py strip).

    cd Motion-Projector
    .venv/bin/python ../Evaluation-Musicality/render_pink_stitch.py --ffmpeg /path/to/ffmpeg [--full]

--full (2026-11-09): the whole 34 s clip (1020 frames; no Wan window to respect), each figure centred on its pelvis in the
ground plane so that travel cannot leave the panel; needs `stitch_appearance_gain.py strip --full` first (graph over 34 s).
Writes loudness_198_slice0_pink_full.mp4.

Writes Expert-Appearance/outputs/gain198/loudness_198_slice0_pink.mp4.
"""
import argparse
import json
import os
import subprocess
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'Motion-Projector', 'wan_preprocess'))
sys.path.insert(0, os.path.join(ROOT, 'Motion-Projector'))
import smpl_render  # noqa: E402

OUT = os.path.join(ROOT, 'Expert-Appearance', 'outputs', 'gain198')
EX = os.path.join(HERE, 'data', 'appearance_examples')
SMPL = os.path.join(ROOT, 'Motion-Projector', 'models', 'smpl', 'SMPL_NEUTRAL_chumpyfree.pkl')
WAV = os.path.join(ROOT, 'Expert-Motion', 'data', 'test', 'wavs_sliced', '198_slice0.wav')
MOTIONS = {'real': 'perc_198_slice0_real.pkl', 'oneshot': 'perc_198_slice0_oneshot.pkl', 'gain': 'loudness_198_slice0_gain.pkl'}
LABELS = {'real': 'Real dancer (mocap)', 'oneshot': 'MACE released', 'gain': 'Ours: critic v5 + gain'}
START, NFR, FPS = 150, 300, 30  # 5.0-15.0 s
PH = 800
PW = PH * 720 // 1280
HEAD, STRIP_H = 150, 230
PINK, BG = (0.96, 0.48, 0.68, 1.0), (0.95, 0.94, 0.95, 1.0)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
AX = (0.045, 0.30, 0.945, 0.66)  # strip axes in figure fractions, as in stitch_appearance_gain.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ffmpeg', default='ffmpeg')
    ap.add_argument('--full', action='store_true')
    a = ap.parse_args()
    global START, NFR
    if a.full:
        START, NFR = 0, 1020
    suffix = '_full' if a.full else ''
    info = json.load(open(os.path.join(OUT, f'strip{suffix}.json')))
    lo, hi = info['xlim']
    sp = Image.open(os.path.join(OUT, f'strip{suffix}.png')).convert('RGB')
    frames = {}
    for v, f in MOTIONS.items():
        poses, trans = smpl_render.load_motion(os.path.join(EX, f))
        if a.full:  # centre on the pelvis in the ground plane (Z is up), keep the height
            trans = trans.copy()
            trans[:, :2] = 0
        idx = np.arange(START, START + NFR)
        verts, faces = smpl_render.smpl_vertices(poses[idx], trans[idx], SMPL)
        fwd = smpl_render.facing_direction(poses[idx[0]])
        frames[v] = smpl_render.render_frames(verts, faces, fwd, 720, 1280, body_color=PINK, bg_color=BG)
        print(v, len(frames[v]), flush=True)
    f_big, f_small, f_tiny = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 18)
    W = PW * 3
    silent = os.path.join(OUT, 'pink_noaudio.mp4')
    wr = imageio.get_writer(silent, fps=FPS, quality=8, macro_block_size=1)
    for k in range(NFR):
        canvas = Image.new('RGB', (W, HEAD + PH + STRIP_H), 'white')
        d = ImageDraw.Draw(canvas)
        d.text((16, 12), 'Music loudness ↔ whole-body motion energy', fill='black', font=f_big)
        d.text((16, 48), 'song 198_slice0, ' + ('the whole 34 s clip' if a.full else 'seconds 5.0-15.0 of the 34 s clip')
               + '. EAS = correlation with loudness over the',
               fill=(90, 90, 90), font=f_small)
        d.text((16, 76), 'full clip (the same seed-3 dances as in the skeleton video).', fill=(90, 90, 90), font=f_small)
        for c, v in enumerate(MOTIONS):
            canvas.paste(Image.fromarray(frames[v][k]).resize((PW, PH)), (c * PW, HEAD))
            d.text((c * PW + 16, 112), f'{LABELS[v]}   EAS {info["scores"][v]:+.2f}', fill='black', font=f_small)
        canvas.paste(sp, (0, HEAD + PH))
        for c, (v, col) in enumerate([('real', (31, 119, 180)), ('oneshot', (255, 127, 14)), ('gain', (44, 160, 44))]):
            d.text((60 + c * 300, HEAD + PH - 26), LABELS[v], fill=col, font=f_tiny)
        d.text((60 + 3 * 300 - 20, HEAD + PH - 26), 'music loudness', fill='black', font=f_tiny)
        x = int(W * (AX[0] + AX[2] * (START / 30 + k / FPS - lo) / (hi - lo)))
        d.line([(x, HEAD + PH + int(STRIP_H * (1 - AX[1] - AX[3]))), (x, HEAD + PH + int(STRIP_H * (1 - AX[1])))],
               fill='red', width=3)
        wr.append_data(np.asarray(canvas))
    wr.close()
    out = os.path.join(OUT, f'loudness_198_slice0_pink{suffix}.mp4')
    subprocess.run([a.ffmpeg, '-y', '-loglevel', 'error', '-i', silent, '-ss', f'{START / 30:.3f}', '-t', f'{NFR / FPS:.3f}',
                    '-i', WAV, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                    '-c:a', 'aac', '-b:a', '192k', '-shortest', out], check=True)
    os.remove(silent)
    print(out, f'{NFR} frames, {NFR / FPS:.1f} s')


if __name__ == '__main__':
    main()
