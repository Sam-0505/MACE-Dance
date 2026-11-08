"""Stitch the Appearance Expert videos of one final dance side by side (real | MACE released | ours) with the song.

The audio is the song slice cut at the same 5.1 s window the motion was animated from (windows.json), so the music
is in sync with the dance. Labels carry the window's score from windows.json (the window was chosen where ours beats
released the most, which the title says).

    cd Motion-Projector   # any venv with imageio + pillow
    .venv/bin/python ../Evaluation-Musicality/stitch_appearance.py loudness --ffmpeg /path/to/ffmpeg

Writes Expert-Appearance/outputs/final/<example>_<slice>_stitched.mp4 (the 3 separate videos stay as they are).
"""
import argparse
import json
import os
import subprocess

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
FINAL = os.path.join(ROOT, 'Expert-Appearance', 'outputs', 'final')
WIN = os.path.join(ROOT, 'Evaluation-Musicality', 'data', 'appearance_final', 'windows.json')
WAVS = os.path.join(ROOT, 'Expert-Motion', 'data', 'test', 'wavs_sliced')
TITLES = {'loudness': 'Music loudness ↔ whole-body motion energy', 'perc': 'Percussive energy ↔ upper-body acceleration',
          'beat': 'Beat locking: body-speed accents vs the beats'}
FPS = 15
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('example', choices=list(TITLES))
    ap.add_argument('--ffmpeg', default='ffmpeg')
    ap.add_argument('--height', type=int, default=960, help='height of each panel (source is 720 x 1280)')
    a = ap.parse_args()
    w = json.load(open(WIN))[a.example]
    n, start, met = w['clip'], w['start'], w['metric']
    stem = f'{a.example}_{n}'
    labels = [('real', 'Real dancer (mocap)', w['best']['real']), ('released', 'MACE released', w['best']['released']),
              ('ours', 'Ours: mask + guidance + foot', w['best']['ours'])]
    readers = [imageio.get_reader(os.path.join(FINAL, f'{stem}_{v}.mp4')) for v, _, _ in labels]
    ph = a.height
    pw = ph * 720 // 1280
    head = 140
    f_big, f_small = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 22)
    silent = os.path.join(FINAL, f'{stem}_stitched_noaudio.mp4')
    out = os.path.join(FINAL, f'{stem}_stitched.mp4')
    wr = imageio.get_writer(silent, fps=FPS, quality=8, macro_block_size=1)
    nfr = min(r.count_frames() for r in readers)
    for k in range(nfr):
        canvas = Image.new('RGB', (pw * 3, ph + head), 'white')
        d = ImageDraw.Draw(canvas)
        d.text((16, 12), TITLES[a.example], fill='black', font=f_big)
        d.text((16, 48), f'{n}, seconds {start / 30:.1f}-{start / 30 + 153 / 30:.1f} of the clip. {met} below is computed '
               f'inside this window,', fill=(90, 90, 90), font=f_small)
        d.text((16, 76), 'which was chosen where ours beats MACE released the most.', fill=(90, 90, 90), font=f_small)
        for c, ((v, lab, sc), r) in enumerate(zip(labels, readers)):
            fr = Image.fromarray(r.get_data(k)).resize((pw, ph))
            canvas.paste(fr, (c * pw, head))
            d.text((c * pw + 16, 108), f'{lab}   {met} {sc:+.2f}', fill='black', font=f_small)
        wr.append_data(np.asarray(canvas))
    wr.close()
    for r in readers:
        r.close()
    dur = nfr / FPS
    subprocess.run([a.ffmpeg, '-y', '-loglevel', 'error', '-i', silent, '-ss', f'{start / 30:.3f}', '-t', f'{dur:.3f}',
                    '-i', os.path.join(WAVS, n + '.wav'), '-map', '0:v', '-map', '1:a', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-crf', '18', '-c:a', 'aac', '-b:a', '192k', '-shortest', out], check=True)
    os.remove(silent)
    print(out, f'{nfr} frames, {dur:.2f} s')


if __name__ == '__main__':
    main()
