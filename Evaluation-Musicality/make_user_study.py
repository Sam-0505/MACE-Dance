"""Blind user-study versions of the grid videos of render_pink_grid.py (2026-11-21).

Default set (FineDance showcase): data/videos/showcase_windows/grid_pink_*.mp4, 2 x 2 panels Real | FlowerDance / MACE-Dance | Ours.
--wild set (user-supplied songs, no real dance): data/videos/wild/grid_pink_*.mp4, 1 x 3 row FlowerDance | MACE-Dance | Ours;
serial numbers continue after the FineDance videos (--start_serial, default 5) in the same folder, own key file.

The panels (450 x 800) are cut out (title strips and header dropped, so no method, clip or seed is visible) and put back in a
shuffled order labelled D1, D2, ... (left to right, top to bottom). Panel order = rows of a random Latin square, so no method sits
in the same position twice (4 x 4: Klein-group square, so the methods do not keep one circular order that just rotates and spotting
one method does not give the others away); the video order is shuffled too. Audio is copied unchanged. No mesh re-render.

    cd Motion-Projector
    .venv/bin/python ../Evaluation-Musicality/make_user_study.py --ffmpeg $FF [--seed 2026]
    .venv/bin/python ../Evaluation-Musicality/make_user_study.py --ffmpeg $FF --wild

Writes data/videos/user_study/video_<NN>.mp4 (shareable, no key inside) and the answer key OUTSIDE that folder:
data/videos/user_study_key.{csv,json} (default set) or user_study_key_wild.{csv,json} (--wild).
"""
import argparse
import csv
import glob
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VID = os.path.join(HERE, 'data', 'videos')
OUT = os.path.join(VID, 'user_study')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
PW, PH, TITLE, HEAD = 450, 800, 46, 50  # geometry of render_pink_grid.py
NAME = {'real': 'Real dance', 'flower': 'FlowerDance', 'oneshot': 'MACE-Dance', 'gain': 'Ours'}
SETS = {'finedance': dict(src=os.path.join(VID, 'showcase_windows'), key=os.path.join(VID, 'user_study_key'), cols=2,
                          pos={'real': (0, 0), 'flower': (0, 1), 'oneshot': (1, 0), 'gain': (1, 1)}),
        'wild': dict(src=os.path.join(VID, 'wild'), key=os.path.join(VID, 'user_study_key_wild'), cols=3,
                     pos={'flower': (0, 0), 'oneshot': (0, 1), 'gain': (0, 2)})}


def latin_square(n, rng):
    """random n x n Latin square (rows, columns and symbols shuffled); 4 x 4 from the Klein-group square (i XOR j), else cyclic."""
    i = np.arange(n)
    L = i[:, None] ^ i[None] if n == 4 else (i[:, None] + i[None]) % n
    return rng.permutation(n)[L[rng.permutation(n)][:, rng.permutation(n)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--wild', action='store_true')
    ap.add_argument('--start_serial', type=int, default=None, help='first serial number (default 1, or 5 with --wild)')
    ap.add_argument('--ffmpeg', default='ffmpeg')
    a = ap.parse_args()
    import imageio
    from PIL import Image, ImageDraw, ImageFont

    S = SETS['wild' if a.wild else 'finedance']
    methods, cols = list(S['pos']), S['cols']
    first = a.start_serial or (5 if a.wild else 1)
    srcs = sorted(glob.glob(os.path.join(S['src'], 'grid_pink_*.mp4')))
    assert srcs, S['src']
    assert len(srcs) <= len(methods), 'more videos than Latin-square rows'
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(srcs))          # serial number -> source video
    L = latin_square(len(methods), rng)          # video k, position p -> method index
    os.makedirs(OUT, exist_ok=True)
    f_t, f_h = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT, 22)
    rows_out = -(-len(methods) // cols)
    Wc, Hc = cols * PW, HEAD + rows_out * (TITLE + PH)
    rows = []
    for k, si in enumerate(order):
        src, serial = srcs[si], f'{first + k:02d}'
        layout = [methods[m] for m in L[k]]      # D1..Dn
        out = os.path.join(OUT, f'video_{serial}.mp4')
        silent = out.replace('.mp4', '_noaudio.mp4')
        rd = imageio.get_reader(src)
        fps = rd.get_meta_data()['fps']
        wr = imageio.get_writer(silent, fps=fps, quality=8, macro_block_size=1)
        for fr in rd:
            canvas = Image.new('RGB', (Wc, Hc), 'white')
            d = ImageDraw.Draw(canvas)
            d.text((16, 12), f'Video {serial}', fill='black', font=f_h)
            for p, meth in enumerate(layout):
                sr, sc = S['pos'][meth]
                ys, xs = HEAD + sr * (TITLE + PH) + TITLE, sc * PW
                r, c = divmod(p, cols)
                x0, y0 = c * PW, HEAD + r * (TITLE + PH)
                canvas.paste(Image.fromarray(fr[ys:ys + PH, xs:xs + PW]), (x0, y0 + TITLE))
                lab = f'D{p + 1}'
                d.text((x0 + (PW - d.textlength(lab, font=f_t)) / 2, y0 + 6), lab, fill='black', font=f_t)
            for c in range(1, cols):
                d.line([(c * PW, HEAD), (c * PW, Hc)], fill='white', width=4)
            for r in range(1, rows_out):
                d.line([(0, HEAD + r * (TITLE + PH)), (Wc, HEAD + r * (TITLE + PH))], fill='white', width=4)
            wr.append_data(np.asarray(canvas))
        wr.close()
        rd.close()
        subprocess.run([a.ffmpeg, '-y', '-loglevel', 'error', '-i', silent, '-i', src, '-map', '0:v', '-map', '1:a',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-c:a', 'copy', '-shortest', out], check=True)
        os.remove(silent)
        rows.append({'video': f'video_{serial}.mp4', 'source': os.path.basename(src),
                     **{f'D{p + 1}': NAME[m] for p, m in enumerate(layout)}})
        print(rows[-1], flush=True)
    with open(S['key'] + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pos = 'D1 top left, D2 top right, D3 bottom left, D4 bottom right' if cols == 2 else 'D1 left, D2 middle, D3 right'
    json.dump({'seed': a.seed, 'positions': pos, 'videos': rows}, open(S['key'] + '.json', 'w'), indent=1)
    print('key:', S['key'] + '.csv')


if __name__ == '__main__':
    main()
