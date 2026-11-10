"""Plain pink mannequin version of the lock 161_slice2 comparison (2026-11-09): real | MACE released | ours (critic v5 + gain),
seed 6, 5.0-15.0 s of the clip at 30 fps, with the song. Same motions, scores, beat grid and graph as
lock_161_slice2_lock_real_released_critic_v5_gain.mp4 (render_fix_compare.py --examples lock); each figure carries the beat
lamp and the beat clock (each peak's place in its beat, arrow = mean phase) of that video. No Appearance Expert: the SMPL
mesh of each motion is drawn directly as a featureless pink body.

SMPL files: export_smpl_gain.py --slice 161_slice2 --seed 6 --tag lock --variants real oneshot gain. Two steps (venvs differ):

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/render_pink_lock.py strip [--full]
    cd ../Motion-Projector
    .venv/bin/python ../Evaluation-Musicality/render_pink_lock.py render --ffmpeg /path/to/ffmpeg [--full]

--full (2026-11-09): the whole 34 s clip (1020 frames; there is no 5.1 s Wan window to respect), each figure centred on its
pelvis in the ground plane (as the skeleton videos do) so that travel cannot leave the panel, and the graph scrolls: 6 s
around a fixed cursor. Writes lock_161_slice2_pink_full.mp4 (+ strip_full.png, strip_full.json).
Writes Expert-Appearance/outputs/lock161/lock_161_slice2_pink.mp4 (+ strip.png, strip.json).
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
OUT = os.path.join(ROOT, 'Expert-Appearance', 'outputs', 'lock161')
EX = os.path.join(HERE, 'data', 'appearance_examples')
SMPL = os.path.join(ROOT, 'Motion-Projector', 'models', 'smpl', 'SMPL_NEUTRAL_chumpyfree.pkl')
WAV = os.path.join(ROOT, 'Expert-Motion', 'data', 'test', 'wavs_sliced', '161_slice2.wav')
N, SEED = '161_slice2', 6
ONE_DIR = os.path.join(HERE, 'data', 'mace_test')
OURS_DIR = os.path.join(HERE, 'data', 'mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10')
KEYS = ['real', 'oneshot', 'gain']
FILES = {k: f'lock_{N}_{k}.pkl' for k in KEYS}
LABELS = {'real': 'Real dancer (mocap)', 'oneshot': 'MACE released', 'gain': 'Ours: critic v5 + gain'}
COLORS = {'real': 'tab:blue', 'oneshot': 'tab:orange', 'gain': 'tab:green'}
START, NFR, FPS = 150, 300, 30  # 5.0-15.0 s
PH = 800
PW = PH * 720 // 1280
HEAD, STRIP_H = 150, 260
PINK, BG = (0.96, 0.48, 0.68, 1.0), (0.95, 0.94, 0.95, 1.0)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
AX = (0.045, 0.27, 0.945, 0.70)  # strip axes in figure fractions


def strip(full=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.getcwd())
    import render_fix_compare as R
    from eval_mace import fk
    from model.metric import SMPLSkeleton
    names = json.load(open(os.path.join(ONE_DIR, 'names.json')))['names']
    i = names.index(N)
    m = pickle.load(open(f'data/test/motions_sliced/{N}.pkl', 'rb'))
    gt = fk(SMPLSkeleton('cuda'), [m['q']], [m['pos']], 'cuda', from_pred=False)[0]
    one = R.load(names, ONE_DIR, i)[SEED]
    ours = R.load(names, OURS_DIR, i)[SEED]
    ex = R.lock_example(N, gt, one[None], ours[None])
    T = ex['T']
    clips = {'real': gt[:T], 'oneshot': one[:T], 'gain': ours[:T]}
    B = ex['beats']
    t = np.arange(T) / 30
    lo, hi = START / 30, (START + NFR) / 30
    ax_pos = AX
    if full:  # whole clip in one wide image: 6 s = one video width, the render crops a scrolling window
        lo, hi = 0.0, T / 30
        fig = plt.figure(figsize=(round(PW * 3 * hi / 6) / 100, STRIP_H / 100), dpi=100)
        ax_pos = (0.0, AX[1], 1.0, AX[3])
    else:
        fig = plt.figure(figsize=(PW * 3 / 100, STRIP_H / 100), dpi=100)
    ax = fig.add_axes(ax_pos)
    for b in B:
        ax.axvline(b / 30, color='0.82', lw=0.9)
    data = {'beats': [int(b) for b in B], 'xlim': [lo, hi], 'scores': {}, 'acc': {}, 'phase': {}}
    for r, k in enumerate(KEYS):
        j = clips[k]
        c, acc = ex['curve'](j), ex['accents'](j)
        off = -3.2 * r
        ax.plot(t[:len(c)], c + off, color=COLORS[k], lw=1.4, label=LABELS[k])
        ax.plot(acc / 30, c[acc] + off, 'o', color=COLORS[k], ms=4)
        data['scores'][k] = float(ex['score'](j))
        acc_in = acc[(acc >= B[0]) & (acc < B[-1])]
        data['acc'][k] = [int(a) for a in acc]
        data['acc_in'] = data.get('acc_in', {})
        data['acc_in'][k] = [int(a) for a in acc_in]
        data['phase'][k] = [float(p) for p in ex['phase'](acc)]
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.tick_params(labelsize=9)
    if full:
        ax.set_xticks(np.arange(0, hi + 1, 1))
    else:
        ax.set_xlabel('seconds of the clip · grey lines = beats, dots = peaks of whole-body motion at the beat rate', fontsize=9)
        ax.legend(loc='upper center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.2), frameon=False)
    suffix = '_full' if full else ''
    fig.savefig(os.path.join(OUT, f'strip{suffix}.png'))
    data['bpm'] = 60 * 30 / np.polyfit(np.arange(len(B)), B, 1)[0]
    print('scores', data['scores'], f'bpm {data["bpm"]:.0f}')
    json.dump(data, open(os.path.join(OUT, f'strip{suffix}.json'), 'w'))


def render(ffmpeg, full=False):
    global START, NFR
    if full:
        START, NFR = 0, 1020
    suffix = '_full' if full else ''
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    sys.path.insert(0, os.path.join(ROOT, 'Motion-Projector', 'wan_preprocess'))
    sys.path.insert(0, os.path.join(ROOT, 'Motion-Projector'))
    import smpl_render
    info = json.load(open(os.path.join(OUT, f'strip{suffix}.json')))
    lo, hi = info['xlim']
    B = np.array(info['beats'])
    sp = Image.open(os.path.join(OUT, f'strip{suffix}.png')).convert('RGB')
    if full:  # pad the wide strip by half a video width each side so the cursor can sit at the centre all the time
        wide = Image.new('RGB', (sp.width + PW * 3, STRIP_H), 'white')
        wide.paste(sp, (PW * 3 // 2, 0))
        sp_wide, sp = wide, None
    frames = {}
    for k in KEYS:
        poses, trans = smpl_render.load_motion(os.path.join(EX, FILES[k]))
        if full:  # centre on the pelvis in the ground plane (Z is up), keep the height
            trans = trans.copy()
            trans[:, :2] = 0
        idx = np.arange(START, START + NFR)
        verts, faces = smpl_render.smpl_vertices(poses[idx], trans[idx], SMPL)
        fwd = smpl_render.facing_direction(poses[idx[0]])
        frames[k] = smpl_render.render_frames(verts, faces, fwd, 720, 1280, body_fraction=0.46, body_color=PINK, bg_color=BG)
        print(k, len(frames[k]), flush=True)
    f_big, f_small, f_tiny = (ImageFont.truetype(FONT, s) for s in (26, 22, 18))
    W = PW * 3
    silent = os.path.join(OUT, 'pink_noaudio.mp4')
    wr = imageio.get_writer(silent, fps=FPS, quality=8, macro_block_size=1)
    rgb = {k: tuple(int(255 * c) for c in matplotlib_rgb(COLORS[k])) for k in KEYS}
    cx, cy, cr = 0, 0, 52  # clock radius (panel-relative centre set below)
    for k_ in range(NFR):
        fr = START + k_
        canvas = Image.new('RGB', (W, HEAD + PH + STRIP_H), 'white')
        d = ImageDraw.Draw(canvas)
        d.text((16, 12), f'Beat phase locking: whole-body motion at the beat rate vs the beats ({info["bpm"]:.0f} BPM)',
               fill='black', font=f_big)
        d.text((16, 48), 'song 161_slice2, ' + ('the whole 34 s clip' if full else 'seconds 5.0-15.0 of the 34 s clip') + f' ({fr / 30:4.1f} s). Lock = phase stability vs the beats over the full', fill=(90, 90, 90), font=f_small)
        d.text((16, 76), 'clip (0.29 = noise, 1 = fixed offset); same seed-6 dances as in the skeleton video.', fill=(90, 90, 90), font=f_small)
        lit = bool(((fr - B >= 0) & (fr - B < 4)).any())
        for c, key in enumerate(KEYS):
            x0 = c * PW
            canvas.paste(Image.fromarray(frames[key][k_]).resize((PW, PH)), (x0, HEAD))
            d.text((x0 + 16, 112), f'{LABELS[key]}   Lock {info["scores"][key]:+.2f}', fill='black', font=f_small)
            # beat lamp, as in the skeleton video
            d.ellipse([x0 + 14, HEAD + 14, x0 + 44, HEAD + 44], fill=(22, 33, 58) if lit else (225, 225, 225))
            d.text((x0 + 52, HEAD + 18), 'beat', fill=(100, 100, 100), font=f_tiny)
            # beat clock: place of each peak so far in its beat interval, arrow = mean phase
            ccx, ccy = x0 + PW - 80, HEAD + 80
            d.ellipse([ccx - cr, ccy - cr, ccx + cr, ccy + cr], outline=(150, 150, 150), width=2)
            d.polygon([(ccx, ccy - cr - 4), (ccx - 6, ccy - cr - 14), (ccx + 6, ccy - cr - 14)], fill=(80, 80, 80))
            acc_in = np.array(info['acc_in'][key])
            ph = np.array(info['phase'][key])
            n_seen = int(np.searchsorted(acc_in, fr, side='right'))
            seen = ph[:n_seen]
            col = rgb[key]
            for p in seen:
                px, py = ccx + cr * np.sin(2 * np.pi * p), ccy - cr * np.cos(2 * np.pi * p)
                d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=col)
            if len(seen):
                z = np.exp(2j * np.pi * seen).mean()
                d.line([(ccx, ccy), (ccx + cr * z.imag, ccy - cr * z.real)], fill=col, width=5)
        if full:  # scrolling 6 s window, cursor fixed at the centre, legend in the figures' colours
            x0w = int(round((sp_wide.width - W) * fr / (NFR - 1)))
            canvas.paste(sp_wide.crop((x0w, 0, x0w + W, STRIP_H)), (0, HEAD + PH))
            x = W // 2
            for c, key in enumerate(KEYS):
                d.text((16 + c * 330, HEAD + PH + 4), LABELS[key], fill=rgb[key], font=f_tiny)
            d.text((W - 330, HEAD + PH + 4), 'grey = beats · dots = peaks', fill=(100, 100, 100), font=f_tiny)
        else:
            canvas.paste(sp, (0, HEAD + PH))
            x = int(W * (AX[0] + AX[2] * (fr / 30 - lo) / (hi - lo)))
        d.line([(x, HEAD + PH + int(STRIP_H * (1 - AX[1] - AX[3]))), (x, HEAD + PH + int(STRIP_H * (1 - AX[1])))],
               fill='red', width=3)
        wr.append_data(np.asarray(canvas))
    wr.close()
    out = os.path.join(OUT, f'lock_161_slice2_pink{suffix}.mp4')
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', silent, '-ss', f'{START / 30:.3f}', '-t', f'{NFR / FPS:.3f}',
                    '-i', WAV, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                    '-c:a', 'aac', '-b:a', '192k', '-shortest', out], check=True)
    os.remove(silent)
    print(out, f'{NFR} frames, {NFR / FPS:.1f} s')


def matplotlib_rgb(name):
    return {'tab:blue': (0.12, 0.47, 0.71), 'tab:orange': (1.0, 0.5, 0.05), 'tab:green': (0.17, 0.63, 0.17)}[name]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['strip', 'render'])
    ap.add_argument('--ffmpeg', default='ffmpeg')
    ap.add_argument('--full', action='store_true', help='the whole 34 s clip, scrolling graph')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    strip(a.full) if a.mode == 'strip' else render(a.ffmpeg, a.full)
