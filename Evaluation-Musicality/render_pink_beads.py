"""Beat beads on the pink mannequin (2026-11-09): real | MACE released | ours, the SMPL body of each motion drawn as a
featureless pink figure (as render_pink_lock.py), with the beat-beads overlay of render_beads.py on top: trails behind both hands
and both feet, as thick as the limb is fast, a gold bead dropped on every trail at every beat of the song, a beat lamp. The
trail points are the SMPL joints (hands 22/23, feet 10/11) projected with the camera that renders the figure.
--slow 2 interpolates the poses (slerp) to 60 fps and plays them at 30 fps with the audio at half speed.

SMPL files: export_smpl_gain.py --slice 161_slice2 --seed 1 --tag beads1 --variants real oneshot gain (Expert-Motion venv).

    cd Motion-Projector
    .venv/bin/python ../Evaluation-Musicality/render_pink_beads.py --ffmpeg $FF [--seed_tag beads1] [--start 13.5 --dur 5.5] [--slow 2] [--still 16.0]

Writes Evaluation-Musicality/data/videos/beads_pink_<slice>_<tag>[_slow2x].mp4 (+ _still.png).
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'Motion-Projector', 'wan_preprocess'))
sys.path.insert(0, os.path.join(ROOT, 'Motion-Projector'))
import smpl_render as SR  # noqa: E402

EX = os.path.join(HERE, 'data', 'appearance_examples')
SMPL = os.path.join(ROOT, 'Motion-Projector', 'models', 'smpl', 'SMPL_NEUTRAL_chumpyfree.pkl')
N = '161_slice2'
FPS, PH, PW, HEAD, FOOT = 30, 800, 450, 92, 60
PINK, BG = (0.96, 0.48, 0.68, 1.0), (0.95, 0.94, 0.95, 1.0)
GOLD, GOLD_EDGE = (255, 209, 26), (64, 46, 0)
KEYS = ['real', 'oneshot', 'gain']
LABELS = {'real': 'Real dancer', 'oneshot': 'MACE released', 'gain': 'Ours: critic v5 + gain'}
COLORS = {'real': (31, 119, 180), 'oneshot': (230, 92, 0), 'gain': (23, 150, 60)}
SHADE = 'linear'


def shade_u(x, lo0, vref):
    """0..1 position on the colour bar for a speed ratio x (local contrast: lo0 = 0.5, vref = 1.5, 1.0 in the middle). linear = (x - lo0) / (vref - lo0);
    log = symmetric in log(ratio) (1/1.5x .. 1.5x); exp = convex, only the fastest cells get dark; stretch = concave around 1.0 (|s|^0.6), so the
    moderate values 1.0 .. 1.3x get darker."""
    u = np.clip((np.asarray(x, float) - lo0) / (vref - lo0), 0, 1)
    if SHADE == 'log':
        return np.clip(0.5 + np.log(np.clip(x, 1e-3, None)) / (2 * np.log(vref)), 0, 1)
    if SHADE == 'exp':
        return (np.exp(2.0 * u) - 1) / (np.exp(2.0) - 1)
    if SHADE == 'stretch':
        sgn = 2 * u - 1
        return 0.5 + 0.5 * np.sign(sgn) * np.abs(sgn) ** 0.6
    return u


SHADE_LO, SHADE_HI = 0.85, 1.15  # beat-phase pattern: relative speed mapped light -> dark
TIPS = [10, 11, 22, 23]  # SMPL joints: feet and hands
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def interp(poses, trans, ts):
    """poses (T, 72) axis-angle, trans (T, 3) at frame times 0..T-1; sampled at float frame times ts (slerp / linear)."""
    T = len(poses)
    rv = poses.reshape(T, 24, 3)
    out = np.empty((len(ts), 24, 3), np.float32)
    for j in range(24):
        out[:, j] = Slerp(np.arange(T), Rot.from_rotvec(rv[:, j]))(ts).as_rotvec()
    tr = np.stack([np.interp(ts, np.arange(T), trans[:, c]) for c in range(3)], 1).astype(np.float32)
    return out.reshape(len(ts), 72), tr


def project(X, eye, center, W, H):
    """pyrender perspective camera (yfov SR.FOV_Y) looking from eye at center, z up; X (..., 3) -> pixels (..., 2)."""
    cam = SR._look_at(eye, center, np.array([0.0, 0.0, 1.0]))
    Xc = (X - cam[:3, 3]) @ cam[:3, :3]  # camera coordinates (camera looks down -z)
    f = 1 / np.tan(SR.FOV_Y / 2)
    x = Xc[..., 0] / -Xc[..., 2] * f / (W / H)
    y = Xc[..., 1] / -Xc[..., 2] * f
    return np.stack([(x + 1) / 2 * W, (1 - y) / 2 * H], -1)


STOPS = [(0.0, (60, 120, 235)), (1.0, (225, 30, 30))]  # two colours only: still = blue, fast = red


def gradient(n=256):
    """(n, 3) uint8 colour scale of the speed glow, blue -> red (two colours; a pink body next to red was hard to tell apart)."""
    g = np.linspace(0, 1, n)
    return np.stack([np.interp(g, [t for t, _ in STOPS], [c[i] for _, c in STOPS]) for i in range(3)], 1).astype(np.uint8)


LOCAL = False
SHADE_LOCK = False
LOCKPH = {}


def draw_legend(canvas, d, x, y, w, h, ref, font):
    """colour bar with 0 .. ref m/s under the speed glow (linear: colour = speed / ref, clipped)."""
    from PIL import Image
    canvas.paste(Image.fromarray(np.tile(gradient(w)[None], (h, 1, 1))), (x, y))
    d.rectangle([x, y, x + w, y + h], outline=(120, 120, 120))
    d.text((x - (178 if LOCAL else 16), y - 1), ('trough of the wave' if SHADE_LOCK else 'slower than around it') if LOCAL else '0', fill=(70, 70, 70), font=font)
    d.text((x + w + 6, y - 1), ('peak of the wave' if SHADE_LOCK else 'faster than around it (1.5x)') if LOCAL else f'{ref:.1f} m/s or more', fill=(70, 70, 70), font=font)


def make_strip(a, geo, tot, vref, Bt, f0, bpm, lo0=0.0):
    """Beat time series photo: one mannequin per beat for 3 dancers (rows), consecutive beats (columns), coloured by speed; with --mid
    a second row per dancer a half beat after each beat. Writes data/videos/beatseries_pink_<slice>_<tag>[_mid].png."""
    from PIL import Image, ImageDraw, ImageFont
    iv = float(np.median(np.diff(Bt)))
    sel = (Bt[Bt >= a.strip_start * FPS][:a.strip_beats] - f0).astype(int)
    shift = ({k: int(round(LOCKPH[k] * iv)) for k in KEYS} if a.own_lock else {k: int(round(-o * iv)) if a.own else 0 for k, o in zip(KEYS, a.offsets)})
    rows = [('own phase' if (a.own or a.own_lock) else 'on the beat', sel)] + ([('half a beat later', sel + int(round(iv / 2)))] if a.mid else [])
    if a.phase_rows:
        rows = [('at the beat' if p == 0 else f'+{p:g} beat', sel + int(round(p * iv))) for p in a.phase_rows]
    CW, CH, LW, HD, FT, GAP = 170, 294, 210, 98, 62, 16
    nb, nr = len(sel), len(rows)
    canvas = Image.new('RGB', (LW + nb * CW + (110 if a.phase_rows else 0), HD + len(KEYS) * (nr * CH + GAP) + FT), 'white')
    d = ImageDraw.Draw(canvas)
    f_t, f_s, f_b = (ImageFont.truetype(FONT, z) for z in (20, 13, 16))
    d.text((12, 8), f'Beat time series   ·   song {N} ({bpm:.0f} BPM)   ·   {nb} consecutive beats from {(sel[0] + f0) / FPS:.1f} s'
           + ('   ·   each dancer at its own peak phase of the beat' if (a.own or a.own_lock) else ''), fill='black', font=f_t)
    what = ('height of the beat-rate wave of the whole-body motion (dark = peak; number = its z-score)' if SHADE_LOCK else 'speed of the whole body relative to its average over one beat either side (number = that ratio)' if a.local else
            'total speed of the whole body') if tot else 'speed of each body part'
    d.text((12, 36), ('shade = ' if a.palette == 'grey' else 'colour = ') + what + (' (same scale for all three dancers)' if len(KEYS) == 3 else ' (same scale for both)'), fill=(80, 80, 80), font=f_s)
    if a.note:
        d.text((12, 56), a.note, fill=(150, 40, 40), font=f_s)
    for c in range(nb):
        d.text((LW + c * CW + 8, HD - 22), f'beat {c + 1}  ·  {(sel[c] + f0) / FPS:.1f} s', fill=(60, 60, 60), font=f_s)
    for di, k in enumerate(KEYS):
        idx = np.clip(np.concatenate([r[1] + shift[k] for r in rows]), 0, len(geo[k][3]) - 1)
        verts, faces, fwd, vs = geo[k]
        level = (np.repeat(shade_u(tot[k][idx], lo0, vref)[:, None], vs.shape[1], 1) if tot else np.clip(vs[idx] / vref, 0, 1))
        frames = render_colored(verts[idx], faces, fwd, level)
        y0 = HD + di * (nr * CH + GAP)
        d.text((12, y0 + nr * CH // 2 - 22), LABELS[k], fill=(40, 40, 40) if a.palette == 'grey' else tuple(COLORS[k]), font=f_b)
        for ri, (rname, _) in enumerate(rows):
            if nr > 1:
                d.text((12, y0 + ri * CH + CH // 2 + (6 if ri == 0 else -14)),
                       (f'own peak phase ({shift[k] / iv:+.2f} beat)' if ((a.own or a.own_lock) and ri == 0) else rname), fill=(90, 90, 90), font=f_s)
            for c in range(nb):
                im = Image.fromarray(frames[ri * nb + c][190:1090, 100:620]).resize((CW, CH), Image.LANCZOS)
                x, y = LW + c * CW, y0 + ri * CH
                canvas.paste(im, (x, y))
                d.rectangle([x, y, x + CW - 1, y + CH - 1], outline=(215, 215, 215))
                if tot:
                    d.text((x + 6, y + CH - 18), ((f'{(tot[k][idx[ri * nb + c]] - 1) * 3:+.1f}' if SHADE_LOCK else f'x{tot[k][idx[ri * nb + c]]:.2f}') if a.local else f'{tot[k][idx[ri * nb + c]]:.2f} m/s'), fill=(60, 60, 60), font=f_s)
            if a.phase_rows and tot:
                v = tot[k][idx[ri * nb:(ri + 1) * nb]]
                d.text((LW + nb * CW + 10, y0 + ri * CH + CH // 2 - 22), 'faster than', fill=(110, 110, 110), font=f_s)
                d.text((LW + nb * CW + 10, y0 + ri * CH + CH // 2 - 6), 'its average:', fill=(110, 110, 110), font=f_s)
                d.text((LW + nb * CW + 10, y0 + ri * CH + CH // 2 + 12), f'{int((v > 1).sum())} of {nb}', fill=(30, 30, 30), font=f_b)
    draw_legend(canvas, d, LW + 40, HD + len(KEYS) * (nr * CH + GAP) + 22, 300, 14, vref, f_s)
    out = os.path.join(HERE, 'data', 'videos', f'beatseries_pink_{N}_{a.tag}{"_own" if a.own else ""}{"_mid" if a.mid else ""}{"_phases" if a.phase_rows else ""}.png')
    canvas.save(out)
    print(out)


def make_phase_grid(a, geo, tot, vref, Bt, f0, bpm, lo0=0.0):
    """Beat phase grid: one block per dancer (side by side), rows = position inside the beat (0, 1/4, 1/2, 3/4 beat after the beat),
    columns = consecutive beats, colour = total speed. A dancer locked to the beat repeats the same shading pattern from column to column
    (at a constant offset); the number at the end of each row is its mean speed over the beats. Writes beatphases_<slice>_<tag>.png."""
    from PIL import Image, ImageDraw, ImageFont
    iv = float(np.median(np.diff(Bt)))
    sel = (Bt[Bt >= a.strip_start * FPS][:a.strip_beats] - f0).astype(int)
    phases = [0.0, 0.25, 0.5, 0.75]
    CW, CH, LW, HD, GAPX, AVG = 104, 182, 118, 134, 30, 96
    nb = len(sel)
    BW = nb * CW + AVG
    canvas = Image.new('RGB', (LW + len(KEYS) * BW + (len(KEYS) - 1) * GAPX, HD + len(phases) * CH + 78), 'white')
    d = ImageDraw.Draw(canvas)
    f_t, f_s, f_b = (ImageFont.truetype(FONT, z) for z in (22, 13, 17))
    d.text((12, 8), f'Beat phase grid   ·   song {N} ({bpm:.0f} BPM)   ·   {nb} consecutive beats from {(sel[0] + f0) / FPS:.1f} s', fill='black', font=f_t)
    d.text((12, 38), 'rows = position inside the beat (0, 1/4, 1/2, 3/4 of a beat after the beat) · columns = consecutive beats · '
           + ('shade' if a.palette == 'grey' else 'colour') + (' = height of the beat-rate wave of the whole-body motion (dark = peak) · ' if SHADE_LOCK else
                                                              ' = speed of the whole body relative to that dancer\'s own average over one beat either side · '
                                                              if a.local else ' = total speed of the whole body (same scale for all three dancers) · ')
           + 'a dancer locked to the beat repeats the same pattern from column to column', fill=(80, 80, 80), font=f_s)
    if a.note:
        d.text((12, 60), a.note, fill=(150, 40, 40), font=f_s)
    for ri, ph in enumerate(phases):
        d.text((12, HD + ri * CH + CH // 2 - 9), 'at the beat' if ph == 0 else f'+{ph:g} beat', fill=(60, 60, 60), font=f_b)
    for di, k in enumerate(KEYS):
        verts, faces, fwd, vs = geo[k]
        x0 = LW + di * (BW + GAPX)
        d.text((x0, HD - 44), LABELS[k], fill=(30, 30, 30), font=f_b)
        for c in range(nb):
            d.text((x0 + c * CW + 6, HD - 20), f'beat {c + 1}', fill=(90, 90, 90), font=f_s)
        idx = np.clip(np.concatenate([sel + int(round(p * iv)) for p in phases]), 0, len(vs) - 1)
        grid = tot[k][idx].reshape(len(phases), nb)  # (phase, beat)
        print(f'  {k:8s} mean speed by phase {np.round(grid.mean(1), 2).tolist()}; fastest phase per beat {grid.argmax(0).tolist()} '
              f'(0 = at the beat: {int((grid.argmax(0) == 0).sum())} of {nb} beats)', flush=True)
        level = np.repeat(np.clip((tot[k][idx] - lo0) / (vref - lo0), 0, 1)[:, None], vs.shape[1], 1)
        frames = render_colored(verts[idx], faces, fwd, level)
        for ri in range(len(phases)):
            for c in range(nb):
                im = Image.fromarray(frames[ri * nb + c][190:1090, 100:620]).resize((CW, CH), Image.LANCZOS)
                x, y = x0 + c * CW, HD + ri * CH
                canvas.paste(im, (x, y))
                d.rectangle([x, y, x + CW - 1, y + CH - 1], outline=(215, 215, 215))
            m = float(np.mean(tot[k][idx[ri * nb:(ri + 1) * nb]]))
            d.text((x0 + nb * CW + 8, HD + ri * CH + CH // 2 - 18), f'mean', fill=(110, 110, 110), font=f_s)
            d.text((x0 + nb * CW + 8, HD + ri * CH + CH // 2 - 2), ((f'{(m - 1) * 3:+.1f}' if SHADE_LOCK else f'x{m:.2f}') if a.local else f'{m:.2f} m/s'), fill=(30, 30, 30), font=f_b)
    draw_legend(canvas, d, LW + (200 if a.local else 40), HD + len(phases) * CH + 28, 300, 14, vref, f_s)
    out = os.path.join(HERE, 'data', 'videos', f'beatphases_{N}_{a.tag}.png')
    canvas.save(out)
    print(out)


def clip_profile(poses, trans, Bt, trim=120, phases=(0.0, 0.25, 0.5, 0.75)):
    """Whole-clip beat-phase profile of the relative speed (as rank_relspeed_clips.py): s(t) = mean over the 24 SMPL joints of the
    root-relative speed (sigma 1 frame); ratio = s / its mean over one beat either side; profile(phi) = mean of ratio at beat + phi x interval
    over the beats without 4 s at both ends. Returns (profile over `phases`, number of beats used)."""
    from scipy.ndimage import uniform_filter1d
    jreg = __import__('smplx').SMPL(SMPL, num_betas=10).J_regressor.numpy()
    t = trans.copy()
    t[:, :2] = 0
    verts, _ = SR.smpl_vertices(poses, t, SMPL)
    J = np.einsum('jv,tvc->tjc', jreg, verts)
    r = J - J[:, :1]
    sp = gaussian_filter1d(np.linalg.norm(np.gradient(r, axis=0) * FPS, axis=-1), 1.0, axis=0).mean(1)
    iv = float(np.median(np.diff(Bt)))
    ratio = sp / uniform_filter1d(sp, max(3, int(round(2 * iv))), mode='nearest')
    b = Bt[(Bt >= trim) & (Bt < len(ratio) - trim - iv)].astype(float)
    g = np.round(b[:, None] + np.array(phases)[None] * iv).astype(int)
    return ratio[g].mean(0), len(b)


def make_avg_grid(a, geo, prof, nbeats, Bt, f0, bpm):
    """Beat-phase pattern: rows = MACE released, Ours, Real (reference); columns = 0, 1/4, 1/2, 3/4 beat after the beat. The pose is one beat
    (the first beat after --strip_start); the shade of each cell is the clip-wide average relative speed at that phase of the beat
    (SHADE_LO..SHADE_HI x the dancer's own average). Writes beatavg_<slice>_<tag>.png."""
    from PIL import Image, ImageDraw, ImageFont
    iv = float(np.median(np.diff(Bt)))
    b0 = int((Bt[Bt >= a.strip_start * FPS][0] - f0))
    phases = [0.0, 0.25, 0.5, 0.75]
    rows = ['oneshot', 'gain', 'real']
    CW, CH, LW, HD, FT = 200, 346, 270, 150, 70
    canvas = Image.new('RGB', (LW + len(phases) * CW + 20, HD + len(rows) * CH + FT), 'white')
    TW = LW + len(phases) * CW  # text width
    d = ImageDraw.Draw(canvas)
    f_t, f_s, f_b = (ImageFont.truetype(FONT, z) for z in (22, 14, 18))
    d.text((12, 8), f'Speed pattern inside the beat   ·   song {N} ({bpm:.0f} BPM)', fill='black', font=f_t)
    d.text((12, 40), 'columns = position inside the beat: at the beat, then 1/4, 1/2, 3/4 of a beat after it', fill=(80, 80, 80), font=f_s)
    d.text((12, 60), 'darker = the dancer moves faster at that point of the beat (number = speed relative to its own average)', fill=(80, 80, 80), font=f_s)
    d.text((12, 80), f'averaged over all {nbeats} beats of the clip; the pose shown is from one beat ({(b0 + f0) / FPS:.1f} s)', fill=(80, 80, 80), font=f_s)
    if a.note:
        for li, line in enumerate(a.note.split('|')):
            d.text((12, 104 + 18 * li), line.strip(), fill=(150, 40, 40), font=f_s)
    for c, ph in enumerate(phases):
        d.text((LW + c * CW + 8, HD - 22), 'at the beat' if ph == 0 else f'+{ph:g} beat', fill=(60, 60, 60), font=f_b)
    for ri, k in enumerate(rows):
        verts, faces, fwd, vs = geo[k]
        idx = np.clip(np.array([b0 + int(round(p * iv)) for p in phases]), 0, len(vs) - 1)
        level = np.repeat(np.clip((prof[k] - SHADE_LO) / (SHADE_HI - SHADE_LO), 0, 1)[:, None], vs.shape[1], 1)
        wide = np.arange(0, len(vs), 4)  # camera framed over the whole window so the floor poses of the real dancer are not cropped
        frames = render_colored(verts[np.concatenate([idx, wide])], faces, fwd, np.concatenate([level, np.repeat(level[:1], len(wide), 0)]))[:len(phases)]
        d.text((12, HD + ri * CH + CH // 2 - 22), LABELS[k], fill=(40, 40, 40), font=f_b)
        if k == 'real':
            d.text((12, HD + ri * CH + CH // 2 + 2), '(reference)', fill=(110, 110, 110), font=f_s)
        print(f'  {k:8s} relative speed by phase {np.round(prof[k], 3).tolist()} (depth {(prof[k].max() - prof[k].min()) / 2:.3f})', flush=True)
        for c in range(len(phases)):
            im = Image.fromarray(frames[c][190:1090, 100:620]).resize((CW, CH), Image.LANCZOS)
            x, y = LW + c * CW, HD + ri * CH
            canvas.paste(im, (x, y))
            d.rectangle([x, y, x + CW - 1, y + CH - 1], outline=(215, 215, 215))
            d.text((x + 8, y + CH - 24), f'x{prof[k][c]:.2f}', fill=(40, 40, 40), font=f_b)
    lx, ly, lw = LW + 210, HD + len(rows) * CH + 24, 250
    canvas.paste(Image.fromarray(np.tile(gradient(lw)[None], (14, 1, 1))), (lx, ly))
    d.rectangle([lx, ly, lx + lw, ly + 14], outline=(120, 120, 120))
    d.text((lx - 190, ly - 1), f'slower ({SHADE_LO:.2f}x or less)', fill=(70, 70, 70), font=f_s)
    d.text((lx + lw + 8, ly - 1), f'faster ({SHADE_HI:.2f}x or more)', fill=(70, 70, 70), font=f_s)
    out = os.path.join(HERE, 'data', 'videos', f'beatavg_{N}_{a.tag}.png')
    canvas.save(out)
    print(out)


def lock_signals(poses, trans, Bt):
    """Beat-locking signals of one dancer (lock_models.py): QoM(t) = mean over the 24 SMPL joints of the frame-to-frame displacement (world coordinates),
    zero-phase band-pass at the beat frequency (width 15 % of it), z-scored; accents = its peaks; ph = place of each accent inside its beat
    (0 = on the beat, tmd_beat_phase.phases). Returns dict(curve, acc, ph) over the whole clip."""
    from scipy.signal import argrelextrema
    jreg = __import__('smplx').SMPL(SMPL, num_betas=10).J_regressor.numpy()
    verts, _ = SR.smpl_vertices(poses, trans, SMPL)
    J = np.einsum('jv,tvc->tjc', jreg, verts).astype(np.float64)
    q = np.pad(np.linalg.norm(J[1:] - J[:-1], axis=2).mean(1), (0, 1), mode='edge')
    iv = float(np.polyfit(np.arange(len(Bt)), np.asarray(Bt, float), 1)[0])
    fc = 1 / iv
    X = np.fft.fft(q - q.mean())
    f = np.fft.fftfreq(len(q))
    band = np.fft.ifft(2 * X * ((f > 0) & (np.abs(f - fc) <= 0.15 * fc / 2))).real
    z = (band - band.mean()) / (band.std() + 1e-8)
    acc = argrelextrema(band, np.greater)[0]
    inr = acc[(acc >= Bt[0]) & (acc < Bt[-1])]
    k = np.searchsorted(Bt, inr, side='right') - 1
    return {'curve': z, 'acc': acc, 'acc_in': inr, 'ph': (inr - Bt[k]) / (Bt[k + 1] - Bt[k])}


def render_colored(verts, faces, forward, level, width=720, height=1280, body_fraction=0.56):
    """As smpl_render.render_frames, with the body coloured blue -> red by level (T, 6890) in 0..1 (gradient()). This pyrender
    build ignores vertex colours, so the level is the u texture coordinate of a gradient texture (smooth, lit like the plain body)."""
    import pyrender
    import trimesh
    from PIL import Image
    tex = np.tile(gradient()[None], (4, 1, 1))
    # frame the whole set of shown poses (not just the first one: a crouching first pose would put the camera far too close)
    center, dist = SR.camera_target_and_distance(verts.reshape(-1, 3), body_fraction)
    eye = center + np.array([forward[0], forward[1], 0.0]) * dist
    cam_pose = SR._look_at(eye, center, np.array([0.0, 0.0, 1.0]))
    scene = pyrender.Scene(bg_color=list(BG), ambient_light=[0.45, 0.45, 0.45])
    scene.add(pyrender.PerspectiveCamera(yfov=SR.FOV_Y, aspectRatio=width / height), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, roughnessFactor=0.8, baseColorTexture=pyrender.Texture(source=tex, source_channels='RGB'))
    renderer = pyrender.OffscreenRenderer(width, height)
    frames, node = [], None
    try:
        for v, u in zip(verts, level):
            if node is not None:
                scene.remove_node(node)
            m = trimesh.Trimesh(vertices=v, faces=faces, process=False)
            m.visual = trimesh.visual.TextureVisuals(uv=np.stack([np.clip(u, 0.004, 0.996), np.full_like(u, 0.5)], 1))  # u = 1 would wrap to blue
            node = scene.add(pyrender.Mesh.from_trimesh(m, material=material, smooth=True))
            frames.append(renderer.render(scene)[0])
    finally:
        renderer.delete()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed_tag', default='beads1')
    ap.add_argument('--start', type=float, default=13.5)
    ap.add_argument('--dur', type=float, default=5.5)
    ap.add_argument('--slow', type=int, default=1, choices=[1, 2])
    ap.add_argument('--trail', type=float, default=1.0, help='trail length in seconds of motion')
    ap.add_argument('--still', type=float, default=None, help='also save a png at this second of the clip')
    ap.add_argument('--no_video', action='store_true')
    ap.add_argument('--glow', action='store_true', help='speed glow: body tinted pink -> red by the exact speed of each surface point '
                    '(relative to the pelvis, one scale for all three), gold frame flash on every beat; no trails')
    ap.add_argument('--uniform', action='store_true', help='with --glow: one colour for the whole body per frame, from its total speed '
                    '(mean speed of all surface points relative to the pelvis)')
    ap.add_argument('--strip', action='store_true', help='photo series instead of a video: one mannequin per beat for --strip_beats beats')
    ap.add_argument('--strip_beats', type=int, default=8)
    ap.add_argument('--strip_start', type=float, default=14.3, help='seconds; the series starts at the first beat after this')
    ap.add_argument('--mid', action='store_true', help='strip: add a row half a beat after each beat')
    ap.add_argument('--parts', action='store_true', help='strip: colour each body part by its own speed (default: whole body by total speed)')
    ap.add_argument('--palette', default='bluered', choices=['bluered', 'grey'], help='speed colour scale (two ends only): blue -> red, or light grey -> black')
    ap.add_argument('--local', action='store_true', help='with --uniform: colour = total speed / its mean over one beat either side (local contrast: '
                    'a speed peak inside a slow stretch counts as a peak); 0.5x -> 1.5x')
    ap.add_argument('--note', default='', help='text shown in the header (e.g. which seed this is)')
    ap.add_argument('--slice', default='161_slice2', help='test clip (SMPL files <seed_tag>_<slice>_<variant>.pkl from export_smpl_gain.py)')
    ap.add_argument('--own', action='store_true', help='strip: first row at each dancer\'s own phase (speed peak) of the beat instead of at the beat')
    ap.add_argument('--offsets', type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=('REAL', 'RELEASED', 'OURS'),
                    help='with --own: lock offsets in beats (rank_lock_clips.py: mean phase of the band-passed QoM minus beat phase); the speed peak is at -offset beats from the beat')
    ap.add_argument('--phases', action='store_true', help='photo grid instead of a video: 4 phases of the beat x consecutive beats, dancers side by side')
    ap.add_argument('--beads', action='store_true', help='with --glow: also drop the gold beads on the hands and feet at every beat')
    ap.add_argument('--phase_rows', type=float, nargs='+', default=None, help='strip: one row per dancer for each of these positions in the beat (0, 0.25, ...), with a count of beats above its own average')
    ap.add_argument('--shade_curve', default='linear', choices=['linear', 'log', 'exp', 'stretch'], help='strip: mapping of the speed ratio to the grey scale')
    ap.add_argument('--lock_graph', action='store_true', help='video: add the phase clock (top right of each panel) and the beat graph (bottom) of the beat-locking videos; no gold frame')
    ap.add_argument('--shade_lock', action='store_true', help='video (with --lock_graph): shade = the beat-rate wave of the graph (z-score -1.5 .. +1.5 -> light .. dark) instead of relative speed')
    ap.add_argument('--own_lock', action='store_true', help='strip: one row per dancer, at that dancer\'s own average peak phase of the beat-rate wave (needs --shade_lock)')
    ap.add_argument('--plain', action='store_true', help='video: plain pink mannequin, no shading, no trails or beads (use with --lock_graph for the clock and graph)')
    ap.add_argument('--no_real', action='store_true', help='leave the real dancer out (rows: MACE released, ours)')
    ap.add_argument('--avg', action='store_true', help='photo grid of the clip-wide beat-phase pattern: 3 dancers x 4 phases of one beat, shade = relative speed averaged over all beats')
    ap.add_argument('--tag', default='seed1')
    ap.add_argument('--ffmpeg', default='ffmpeg')
    a = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    global STOPS, LOCAL, N, KEYS, SHADE, SHADE_LOCK
    SHADE_LOCK = a.shade_lock
    if a.shade_lock:
        a.tag += '_shadelock'
    SHADE = a.shade_curve
    if a.shade_curve != 'linear':
        a.tag += '_' + a.shade_curve
    if a.no_real:
        KEYS = ['oneshot', 'gain']
    N = a.slice
    LOCAL = a.local
    if a.local:
        a.tag += '_local'
    if a.palette == 'grey':
        STOPS = [(0.0, (215, 215, 215)), (1.0, (20, 20, 20))]  # still = light grey, fast = black
        a.tag += '_grey'
    if a.avg:
        a.strip, a.parts, a.uniform = True, False, True
        a.palette = 'grey'
    if a.phases:
        a.strip, a.parts = True, False
    if a.strip:
        a.glow, a.uniform, a.slow, a.no_video = True, not a.parts, 1, True
        a.start, a.dur = a.strip_start - 1.3, a.strip_beats * 0.6 + 3.0  # a beat of margin each side for the local average

    U = a.slow
    f0 = int(round(a.start * FPS))
    nf = int(round(a.dur * FPS))
    ts = f0 + np.arange(nf * U) / U  # source frame time of every output frame
    feats = np.load(os.path.join(ROOT, 'Expert-Motion', 'data', 'test', 'baseline_feats', N + '.npy'))
    Bt = np.flatnonzero(feats[:, 34])
    bpm = 60 * FPS / np.polyfit(np.arange(len(Bt)), Bt, 1)[0]
    Bm = (Bt - f0) * U  # beat positions in output frames (can be outside the window)
    import smplx
    import torch
    jreg = smplx.SMPL(SMPL, num_betas=10).J_regressor.numpy()  # (24, 6890)

    imgs, pts, geo, leg, prof, lock = {}, {}, {}, {}, {}, {}
    for k in KEYS:
        poses, trans = SR.load_motion(os.path.join(EX, f'{a.seed_tag}_{N}_{k}.pkl'))
        if a.avg:
            prof[k], nbeats = clip_profile(poses, trans, Bt)
        if a.lock_graph or a.shade_lock or a.own_lock:
            lock[k] = lock_signals(poses, trans, Bt)
        lo = max(int(ts[0]) - 1, 0)
        hi = min(int(ts[-1]) + 3, len(poses))
        p, t = interp(poses[lo:hi], trans[lo:hi], ts - lo)
        t[:, :2] = 0  # centre on the pelvis in the ground plane, keep the height (as render_pink_lock --full)
        verts, faces = SR.smpl_vertices(p, t, SMPL)
        center, dist = SR.camera_target_and_distance(verts[0], 0.56)
        fwd = SR.facing_direction(p[0])
        eye = center + np.array([fwd[0], fwd[1], 0.0]) * dist
        J = np.einsum('jv,tvc->tjc', jreg, verts)[:, TIPS]  # (T, 4, 3)
        pts[k] = project(J, eye, center, 720, 1280) * (PW / 720)
        if a.glow:  # speed of every surface point relative to the pelvis (m/s), smoothed over ~1 source frame
            rel = verts - np.einsum('v,tvc->tc', jreg[0], verts)[:, None]
            vs = gaussian_filter1d(np.linalg.norm(np.gradient(rel, axis=0), axis=-1) * FPS * U, 1.0 * U, axis=0).astype(np.float32)
            geo[k] = (verts, faces, fwd, vs)
            continue
        frames = SR.render_frames(verts, faces, fwd, 720, 1280, body_fraction=0.56, body_color=PINK, bg_color=BG)
        imgs[k] = [np.asarray(Image.fromarray(f).resize((PW, PH), Image.LANCZOS)) for f in frames]
        del frames, verts
        print(k, len(imgs[k]), flush=True)
    if a.own_lock:
        for k in KEYS:
            LOCKPH[k] = float((np.angle(np.exp(2j * np.pi * lock[k]['ph']).mean()) / (2 * np.pi)) % 1.0)
            print(f'  {k:8s} average peak phase of the beat-rate wave: {LOCKPH[k]:+.2f} beat after the beat', flush=True)
    if a.glow:
        vref = float(np.percentile(np.concatenate([g[3].ravel() for g in geo.values()]), 97))
        print(f'glow scale: red = {vref:.2f} m/s (97th percentile of all surface speeds)', flush=True)
        if a.uniform:  # total speed of the body = mean over all surface points; one scale for the three dancers
            tot = {k: g[3].mean(1) for k, g in geo.items()}
            vref = float(np.percentile(np.concatenate(list(tot.values())), 97))
            print(f'total-speed glow: red = {vref:.2f} m/s (97th percentile of total body speed)', flush=True)
            for k, t in tot.items():
                at = np.array([t[b] for b in Bm if 0 <= b < len(t)])
                later = np.array([t[int(b + 8 * U)] for b in Bm if 0 <= b and b + 8 * U < len(t)])
                print(f'  {k:8s} total speed: mean {t.mean():.2f} m/s, at the beat {at.mean():.2f} ({at.mean() / t.mean():.2f}x), '
                      f'half a beat later {later.mean():.2f} ({later.mean() / t.mean():.2f}x)', flush=True)
        lo0 = 0.0
        if a.local:  # local contrast: total speed over its mean in a window of one beat either side; 0.5x -> 1.5x
            from scipy.ndimage import uniform_filter1d
            w = max(3, int(round(2 * float(np.median(np.diff(Bt))) * U)))
            ratio = {k: t / uniform_filter1d(t, w, mode='nearest') for k, t in tot.items()}
            for k, r in ratio.items():
                at = np.array([r[b] for b in Bm if 0 <= b < len(r)])
                print(f'  {k:8s} local contrast at the beats: mean {at.mean():.2f}, beats above the local mean {100 * np.mean(at > 1):.0f}%', flush=True)
            tot, vref, lo0 = ratio, 1.5, 0.5
        if a.shade_lock:  # shade = height of the beat-rate wave (z-score) at each frame: -1.5 .. +1.5 -> 0 .. 1
            ix = np.clip(f0 + np.round(np.arange(len(ts)) / U).astype(int), 0, len(lock[KEYS[0]]['curve']) - 1)
            tot = {k: 1.0 + lock[k]['curve'][ix] / 3.0 for k in KEYS}
            vref, lo0 = 1.5, 0.5
        leg['ref'] = vref
        if a.avg:
            make_avg_grid(a, geo, prof, nbeats, Bt, f0, bpm)
            return
        if a.phases:
            make_phase_grid(a, geo, tot, vref, Bt, f0, bpm, lo0)
            return
        if a.strip:
            make_strip(a, geo, tot if a.uniform else None, vref, Bt, f0, bpm, lo0)
            return
        for k, (verts, faces, fwd, vs) in geo.items():
            if a.uniform:
                frames = render_colored(verts, faces, fwd, np.repeat(np.clip((tot[k] - lo0) / (vref - lo0), 0, 1)[:, None], vs.shape[1], 1))
            else:
                frames = render_colored(verts, faces, fwd, np.clip(vs / vref, 0, 1))
            imgs[k] = [np.asarray(Image.fromarray(f).resize((PW, PH), Image.LANCZOS)) for f in frames]
            print(k, len(imgs[k]), flush=True)

    # trail width = limb speed in the picture; one scale for all three dancers
    spd = {k: gaussian_filter1d(np.linalg.norm(np.gradient(pts[k], axis=0), axis=-1) * FPS * U, 1.0 * U, axis=0) for k in KEYS}
    ref = np.percentile(np.concatenate([v.ravel() for v in spd.values()]), 95)
    Wt = int(round(a.trail * FPS * U))
    f_t, f_s = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 14)
    GH = 300 if a.lock_graph else 0
    Wc, Hc = PW * len(KEYS), HEAD + PH + FOOT + GH

    def frame(m):
        canvas = Image.new('RGB', (Wc, Hc), 'white')
        d = ImageDraw.Draw(canvas)
        d.text((16, 8), f'{"Beat locking" if a.plain else "Speed glow" if a.glow else "Beat beads"}   ·   song {N} ({bpm:.0f} BPM)   {ts[m] / FPS:4.1f} s' + ('   ·   half speed' if U > 1 else ''),
               fill='black', font=f_t)
        for c, k in enumerate(KEYS):
            x0 = c * PW
            canvas.paste(Image.fromarray(imgs[k][m]), (x0, HEAD))
            d.text((x0 + 16, 62), LABELS[k], fill='black', font=f_t)
            col = np.array(COLORS[k], float)
            lo = max(0, m - Wt)
            for j in (range(4) if not (a.glow or a.plain) else []):
                seg = pts[k][lo:m + 1, j]
                for q in range(len(seg) - 1):
                    age = (q + 1) / max(1, Wt)
                    al = 0.15 + 0.85 * age
                    cc = tuple(int(v) for v in col * al + np.array(BG[:3]) * 255 * (1 - al))
                    w = 1 + int(round(8 * min(1.15, spd[k][lo + q + 1, j] / ref)))
                    xy = [(x0 + seg[q][0], HEAD + seg[q][1]), (x0 + seg[q + 1][0], HEAD + seg[q + 1][1])]
                    d.line(xy, fill=cc, width=w)
            recent = Bm[(Bm >= lo) & (Bm <= m)]
            for b in (recent if ((not a.glow or a.beads) and not a.plain) else []):
                age = m - b
                r = 5 + 6 * max(0.0, 1 - age / (6 * U))
                for j in range(4):
                    x, y = pts[k][b, j]
                    d.ellipse([x0 + x - r, HEAD + y - r, x0 + x + r, HEAD + y + r], fill=GOLD, outline=GOLD_EDGE, width=2)
            lit = bool(((m - Bm >= 0) & (m - Bm < 4 * U)).any())
            if a.glow and lit and not a.lock_graph:
                d.rectangle([x0 + 2, HEAD + 2, x0 + PW - 3, HEAD + PH - 3], outline=GOLD, width=8)
            d.ellipse([x0 + 14, HEAD + 14, x0 + 40, HEAD + 40], fill=(22, 33, 58) if lit else (225, 225, 225))
            d.text((x0 + 48, HEAD + 18), 'beat', fill=(100, 100, 100), font=f_s)
            if a.lock_graph:  # phase clock: place of each peak of the beat-rate motion inside its beat (top = on the beat), arrow = their average
                cx, cy, R = x0 + PW - 62, HEAD + 66, 44
                col3 = tuple(COLORS[k])
                d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(150, 150, 150), width=2)
                d.polygon([(cx - 6, cy - R - 12), (cx + 6, cy - R - 12), (cx, cy - R - 2)], fill=(90, 90, 90))
                fr = f0 + m / U
                seen = lock[k]['ph'][:np.searchsorted(lock[k]['acc_in'], fr, side='right')]
                for p in seen:
                    d.ellipse([cx + R * np.sin(2 * np.pi * p) - 4, cy - R * np.cos(2 * np.pi * p) - 4, cx + R * np.sin(2 * np.pi * p) + 4, cy - R * np.cos(2 * np.pi * p) + 4], fill=col3)
                if len(seen):
                    zc = np.exp(2j * np.pi * seen).mean()
                    d.line([(cx, cy), (cx + R * zc.imag, cy - R * zc.real)], fill=col3, width=5)
                d.text((cx - 36, cy + R + 8), 'peak phase', fill=(110, 110, 110), font=f_s)
        if a.glow:
            draw_legend(canvas, d, 16 + (178 if a.local else 16), HEAD + PH + 34, 240, 14, leg['ref'], f_s)
        if a.note:
            d.text((16, 36), a.note, fill=(150, 40, 40), font=f_s)
        d.text((16, HEAD + PH + 8), ((((('shade = height of the beat-rate wave in the graph below (dark = peak)' if a.shade_lock else 'shade = body speed relative to its own average over one beat either side') if a.local else 'whole body colour = total speed of the body') if a.uniform else 'body colour = how fast each part moves')
                                      + (' (same scale for all three) · ' if len(KEYS) == 3 else ' · ') + ('lamp = a beat' if a.lock_graph else 'gold frame and lamp = a beat'))
                                     if a.glow else 'trails = both hands and both feet (last 1 s), thicker = faster · gold beads = where each limb was at every beat of the song'),
               fill=(90, 90, 90), font=f_s)
        if a.plain:
            d.rectangle([0, HEAD + PH + 4, Wc, HEAD + PH + FOOT - 2], fill='white')
            d.text((16, HEAD + PH + 8), 'plain mannequin, no shading · lamp = a beat · the clock and graph show the timing of the beat-rate motion', fill=(90, 90, 90), font=f_s)
        if a.lock_graph:
            gy0, gy1, gx0, gx1 = HEAD + PH + FOOT + 40, HEAD + PH + FOOT + GH - 62, 16, Wc - 16
            now = (f0 + m / U) / FPS
            lo = min(max(now - 3.0, f0 / FPS), (f0 + nf) / FPS - 6.0)
            xp = lambda t_: gx0 + (t_ - lo) / 6.0 * (gx1 - gx0)  # noqa: E731
            yp = lambda v_: gy0 + (2.6 - v_) / 8.6 * (gy1 - gy0)  # noqa: E731
            d.rectangle([gx0, gy0, gx1, gy1], outline=(200, 200, 200))
            for b_ in Bt:
                if lo <= b_ / FPS <= lo + 6.0:
                    d.line([(xp(b_ / FPS), gy0), (xp(b_ / FPS), gy1)], fill=(215, 215, 215), width=1)
            for sec in range(int(np.ceil(lo)), int(lo + 6.0) + 1):
                d.text((xp(sec) - 8, gy1 + 4), f'{sec}', fill=(110, 110, 110), font=f_s)
            for r_, k in enumerate(KEYS):
                cv, ac = lock[k]['curve'], lock[k]['acc']
                i0, i1 = int(np.floor(lo * FPS)), int(np.ceil((lo + 6.0) * FPS))
                pts_ = [(xp(i / FPS), yp(cv[i] - 3.2 * r_)) for i in range(max(i0, 0), min(i1, len(cv) - 1) + 1)]
                d.line(pts_, fill=tuple(COLORS[k]), width=2)
                for a_ in ac[(ac >= i0) & (ac <= i1)]:
                    d.ellipse([xp(a_ / FPS) - 4, yp(cv[a_] - 3.2 * r_) - 4, xp(a_ / FPS) + 4, yp(cv[a_] - 3.2 * r_) + 4], fill=tuple(COLORS[k]))
                d.text((gx0 + 200 * r_, gy0 - 18), LABELS[k], fill=tuple(COLORS[k]), font=f_s)
            d.line([(xp(now), gy0), (xp(now), gy1)], fill=(220, 30, 30), width=2)
            d.text((gx0, gy1 + 22), 'seconds · grey lines = beats · curve = whole-body motion at the beat rate · dots = its peaks (the accents)', fill=(90, 90, 90), font=f_s)
            d.text((gx0, gy1 + 40), 'clock: each peak\'s place in its beat (top = on it), all peaks since 0 s · arrow = average (longer = steadier)', fill=(90, 90, 90), font=f_s)
        return np.asarray(canvas)

    tag = a.tag + ('_plain' if a.plain else '') + (('_glowtotal' if a.uniform else '_glow') if a.glow else '') + ('_slow2x' if U > 1 else '')
    out = os.path.join(HERE, 'data', 'videos', f'beads_pink_{N}_{tag}.mp4')
    if a.still is not None:
        m = int(round((a.still * FPS - f0) * U))
        Image.fromarray(frame(m)).save(out.replace('.mp4', '_still.png'))
        print(out.replace('.mp4', '_still.png'))
    if not a.no_video:
        import imageio
        silent = out.replace('.mp4', '_noaudio.mp4')
        wr = imageio.get_writer(silent, fps=FPS, quality=8, macro_block_size=1)
        for m in range(len(ts)):
            wr.append_data(frame(m))
        wr.close()
        wav = os.path.join(ROOT, 'Expert-Motion', 'data', 'test', 'wavs_sliced', N + '.wav')
        af = ['-af', f'atempo={1 / U:.4f}'] if U > 1 else []
        subprocess.run([a.ffmpeg, '-y', '-loglevel', 'error', '-i', silent, '-ss', f'{a.start:.3f}', '-i', wav, '-map', '0:v', '-map', '1:a']
                       + af + ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-c:a', 'aac', '-b:a', '192k', '-shortest', out], check=True)
        os.remove(silent)
        print(out)


if __name__ == '__main__':
    main()
