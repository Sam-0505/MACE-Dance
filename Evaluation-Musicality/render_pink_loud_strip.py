"""Music loudness vs whole-body motion as a beat time series on the pink-mannequin SMPL bodies (2026-11-10): MACE released vs Ours.

Top: graph of the music loudness (RMS) and the motion energy of both dancers (z-scored, 0.5 s smoothing, the curves of the
loudness_198_slice0 video; EAS = correlation of the two over the clip). Below: one mannequin per beat for each dancer, 8 consecutive
beats; the body colour is that dancer's motion energy at the beat (z -1.5 .. +1.5 -> blue .. red, two colours only).

Two steps (the venvs differ):

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/render_pink_loud_strip.py series --slice 198_slice0 --seed 3
    cd ../Motion-Projector
    .venv/bin/python ../Evaluation-Musicality/render_pink_loud_strip.py render --slice 198_slice0 --start 7.95 --beats 8 \\
        --mace perc_198_slice0_oneshot.pkl --ours loudness_198_slice0_gain.pkl

Writes Evaluation-Musicality/data/loud_strip/<slice>_seed<k>.npz and data/videos/loudstrip_<slice>_<tag>.png.
SMPL files: export_smpl_gain.py (the loudness_198_slice0 dances are seed 3).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SERIES = os.path.join(HERE, 'data', 'loud_strip')
FPS = 30
STORED = {'mace': os.path.join(HERE, 'data', 'mace_test', 'joints.npy'),
          'ours': os.path.join(HERE, 'data', 'mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10', 'joints.npy')}


def series(a):
    """z-scored, smoothed music loudness and whole-body motion energy of MACE released and Ours (stored joints of seed --seed)."""
    from scipy.ndimage import gaussian_filter as G
    sys.path.insert(0, os.getcwd())
    import audio_features as af
    import metrics as M
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    i = names.index(a.slice)
    feat = np.load(f'data/test/baseline_feats/{a.slice}.npy')
    rms = af.rms_envelope(af.load_audio(f'data/test/wavs_sliced/{a.slice}.wav'))
    J = {k: np.asarray(np.load(p, mmap_mode='r')[i][a.seed], np.float32) for k, p in STORED.items()}
    T = min(len(rms), len(J['mace']))
    z = lambda x: (x - x.mean()) / (x.std() + 1e-8)  # noqa: E731
    out = {'mus': z(G(rms[:T], M.ENERGY_SMOOTH)), 'rms': rms[:T], 'B': np.flatnonzero(feat[:, 34])}
    for k, j in J.items():
        out[k] = z(G(M.motion_energy(j[:T]), M.ENERGY_SMOOTH))
    os.makedirs(SERIES, exist_ok=True)
    path = os.path.join(SERIES, f'{a.slice}_seed{a.seed}.npz')
    np.savez(path, **out)
    print(path, f'EAS (corr with loudness over the clip): MACE {np.corrcoef(out["mus"], out["mace"])[0, 1]:.2f}, '
                f'Ours {np.corrcoef(out["mus"], out["ours"])[0, 1]:.2f}')


def window_scores(a):
    """For every 8-beat window starting in --scan_from..--scan_to s: loudness rise and correlation of each dancer's energy with it."""
    S = np.load(a.npz)
    B = S['B']
    for k in range(len(B) - a.beats):
        s0, s1 = int(B[k]), int(B[k + a.beats])
        if not (a.scan_from <= s0 / FPS <= a.scan_to):
            continue
        r = {d: np.corrcoef(S['mus'][s0:s1], S[d][s0:s1])[0, 1] for d in ('mace', 'ours')}
        print(f'{s0 / FPS:5.2f}-{s1 / FPS:5.2f} s  loudness rise {S["mus"][s1 - 1] - S["mus"][s0]:+.2f}  corr MACE {r["mace"]:+.2f}  Ours {r["ours"]:+.2f}')


def render(a):
    import render_pink_beads as RPB
    from PIL import Image, ImageDraw, ImageFont
    sys.path.insert(0, os.path.join(RPB.ROOT, 'Motion-Projector', 'wan_preprocess'))
    RPB.STOPS = [(0.0, (60, 120, 235)), (1.0, (225, 30, 30))]  # two colours only: little motion = blue, a lot = red
    SR, N = RPB.SR, a.slice
    S = np.load(a.npz)
    Bt = S['B']
    iv = float(np.polyfit(np.arange(len(Bt)), Bt, 1)[0])
    bpm = 60 * FPS / iv
    sel = Bt[Bt >= a.start * FPS][:a.beats]
    nb = len(sel)
    files = {'oneshot': a.mace, 'gain': a.ours}
    ser = {'oneshot': S['mace'], 'gain': S['ours']}
    labels = {'oneshot': 'MACE released', 'gain': 'Ours: critic v5 + gain'}
    CW, CH, LW, HD, GH, FT = 170, 294, 210, 100, 250, 60
    W = LW + nb * CW + 20
    canvas = Image.new('RGB', (W, HD + GH + 2 * CH + FT), 'white')
    d = ImageDraw.Draw(canvas)
    f_t, f_s, f_b = (ImageFont.truetype(RPB.FONT, z) for z in (20, 13, 16))
    bt = sel / FPS
    d.text((12, 8), f'Music loudness vs whole-body motion   ·   song {N} ({bpm:.0f} BPM)   ·   {nb} consecutive beats from {bt[0]:.1f} s', fill='black', font=f_t)
    d.text((12, 36), 'curves = music loudness (RMS) and the motion energy of each dancer, z-scored with 0.5 s smoothing · body colour = its motion energy at that beat',
           fill=(80, 80, 80), font=f_s)
    if a.note:
        d.text((12, 58), a.note, fill=(150, 40, 40), font=f_s)
    # graph: time mapped so that the beats fall on the column centres
    xs = lambda t_: LW + CW / 2 + (t_ - bt[0]) / (bt[-1] - bt[0]) * (nb - 1) * CW  # noqa: E731
    gy0, gy1 = HD + 4, HD + GH - 30
    ys = lambda v_: gy0 + (2.6 - v_) / 5.2 * (gy1 - gy0)  # noqa: E731
    d.rectangle([LW, gy0, W - 20, gy1], outline=(200, 200, 200))
    d.line([(LW, ys(0)), (W - 20, ys(0))], fill=(235, 235, 235))
    for b_ in bt:
        d.line([(xs(b_), gy0), (xs(b_), gy1)], fill=(215, 215, 215))
    fa, fb = int((bt[0] - 0.5 * iv / FPS) * FPS), int((bt[-1] + 0.5 * iv / FPS) * FPS)
    curves = [('music loudness (RMS)', S['mus'], (20, 20, 20)), (labels['oneshot'] + ': motion energy', ser['oneshot'], RPB.COLORS['oneshot']),
              ('Ours: motion energy', ser['gain'], RPB.COLORS['gain'])]
    cl = lambda v_: float(np.clip(v_, -2.6, 2.6))  # noqa: E731
    for li, (lab, cv, col) in enumerate(curves):
        d.line([(xs(i / FPS), ys(cl(cv[i]))) for i in range(max(fa, 0), min(fb, len(cv) - 1) + 1)], fill=col, width=3 if li == 0 else 2)
        d.text((12, gy0 + 6 + 20 * li), lab, fill=col, font=f_s)
        for b_ in sel:
            d.ellipse([xs(b_ / FPS) - 4, ys(cl(cv[b_])) - 4, xs(b_ / FPS) + 4, ys(cl(cv[b_])) + 4], fill=col)
    for c in range(nb):
        d.text((LW + c * CW + 8, gy1 + 6), f'beat {c + 1}  ·  {bt[c]:.1f} s', fill=(60, 60, 60), font=f_s)
    for ri, k in enumerate(('oneshot', 'gain')):
        poses, trans = SR.load_motion(os.path.join(RPB.EX, files[k]))
        p, t = poses[sel].reshape(nb, 72), trans[sel].copy()
        t[:, :2] = 0
        verts, faces = SR.smpl_vertices(p, t, RPB.SMPL)
        z = ser[k][sel]
        level = np.repeat(np.clip(0.5 + z / 3.0, 0, 1)[:, None], verts.shape[1], 1)
        frames = RPB.render_colored(verts, faces, SR.facing_direction(p[0]), level)
        y0 = HD + GH + ri * CH
        d.text((12, y0 + CH // 2 - 10), labels[k], fill=(40, 40, 40), font=f_b)
        for c in range(nb):
            im = Image.fromarray(frames[c][190:1090, 100:620]).resize((CW, CH), Image.LANCZOS)
            canvas.paste(im, (LW + c * CW, y0))
            d.rectangle([LW + c * CW, y0, LW + c * CW + CW - 1, y0 + CH - 1], outline=(215, 215, 215))
            d.text((LW + c * CW + 6, y0 + CH - 18), f'{z[c]:+.1f}', fill=(60, 60, 60), font=f_s)
        print(f'  {labels[k]:24s} motion energy z at the beats {np.round(z, 2).tolist()}', flush=True)
    print(f'  music loudness z at the beats {np.round(S["mus"][sel], 2).tolist()}', flush=True)
    lx, ly, lw = LW + 240, HD + GH + 2 * CH + 20, 250
    canvas.paste(Image.fromarray(np.tile(RPB.gradient(lw)[None], (14, 1, 1))), (lx, ly))
    d.rectangle([lx, ly, lx + lw, ly + 14], outline=(120, 120, 120))
    d.text((lx - 215, ly - 1), 'less motion energy (z -1.5)', fill=(70, 70, 70), font=f_s)
    d.text((lx + lw + 8, ly - 1), 'more (z +1.5)', fill=(70, 70, 70), font=f_s)
    out = os.path.join(HERE, 'data', 'videos', f'loudstrip_{N}_{a.tag}.png')
    canvas.save(out)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['series', 'scan', 'render'])
    ap.add_argument('--slice', default='198_slice0')
    ap.add_argument('--seed', type=int, default=3, help='series: seed of the stored joints (loudness_198_slice0 = 3)')
    ap.add_argument('--npz', default=None)
    ap.add_argument('--start', type=float, default=7.95, help='render: the series starts at the first beat after this (s)')
    ap.add_argument('--beats', type=int, default=8)
    ap.add_argument('--scan_from', type=float, default=6.0)
    ap.add_argument('--scan_to', type=float, default=12.0)
    ap.add_argument('--mace', default='perc_198_slice0_oneshot.pkl', help='SMPL file of MACE released in data/appearance_examples')
    ap.add_argument('--ours', default='loudness_198_slice0_gain.pkl', help='SMPL file of Ours')
    ap.add_argument('--note', default='')
    ap.add_argument('--tag', default='seed3')
    a = ap.parse_args()
    if a.npz is None:
        a.npz = os.path.join(SERIES, f'{a.slice}_seed{a.seed}.npz')
    {'series': series, 'scan': window_scores, 'render': render}[a.step](a)


if __name__ == '__main__':
    main()
