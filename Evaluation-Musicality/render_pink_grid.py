"""2 x 2 comparison video on the plain pink mannequin (2026-11-21): Real dance | FlowerDance / MACE-Dance | Ours, one test clip and seed,
each panel titled, with the song as audio. No beat graph, clock or shading.

Motions: real = appearance_examples/<real_tag>_<slice>_real.pkl (export_smpl_gain.py --variants real); the generated ones come from
data/rot_export/<key>_seed<s>.npz (poses (84, 1020, 72), trans (84, 1020, 3), Z-up): flower = FlowerDance, oneshot = MACE-Dance
released, gain = ours (critic v5 + gain). Each body is centred on its pelvis in the ground plane (height kept), as render_pink_beads.py.
All four panels share one camera distance, fitted so every frame of every dancer stays in view (same scale across panels).

    cd Motion-Projector
    .venv/bin/python ../Evaluation-Musicality/render_pink_grid.py --ffmpeg $FF --slice 143_slice7 --seed 4 --start 4.30 --dur 10.2

Writes Evaluation-Musicality/data/videos/grid_pink_<slice>_seed<s>[_<start>s].mp4.

--wild NAME: a user-supplied song of gen_wild_songs.py (no real dance): 1 x 3 row FlowerDance | MACE-Dance | Ours from
data/wild/rot/{flower,oneshot,gain}.npz, each model at its own seed (--wild_seeds flower=S oneshot=S gain=S), audio = the song
segment, --start defaults to the requested song start. Writes data/videos/wild/grid_pink_<name>.mp4.

--row (2026-11-21): 1 x 4 row FlowerDance | MACE-Dance | MMC-Dance (Ours) | Real Dance, no clip header. Under each dancer a scrolling
6 s energy graph: whole-body motion energy QoM (mean joint frame-to-frame displacement, lock_models.py signal; same y scale in all
panels) with its beat-rate part (band-passed QoM around its mean) as a dark line and its peaks (the accents, the beat clock's dots) as dots, grey lines = music beats, red line = now. Top right of each dancer a beat clock (render_pink_beads.py phase clock): where each
accent so far (peak of QoM band-passed at the beat rate) falls inside its beat (top = on the beat), arrow =
their average (length = steadiness, i.e. the running lock r). Writes data/videos/showcase_windows/row_energy_<slice>_seed<s>[_<start>s].mp4.

--imc (with --row, 2026-11-21): graph for IMC instead of the beat graph: 'Motion intensity (QoM)' on top and 'Audio intensity'
(audio RMS, robustness_loud.music_signals, saved first as data/appearance_examples/rms_<slice>.npy with the Expert-Motion venv) below,
both Gaussian-smoothed sigma 15 frames and scaled to their own min..max over the clip, i.e. exactly the two curves IMC correlates
(Pearson, zero lag); the whole rendered span is shown, already played part solid, red line = now. No beat lines. The beat clock (BPL) stays.
Writes data/videos/showcase_windows/row_imc_<slice>_seed<s>[_<start>s].mp4. Prints the reported whole-clip values
(results/fullclip_bpl_imc_seeds.json): IMC at the right end of the motion strip's title line, BPL under the beat clock (2026-11-23).
--split: the same panels as one silent video per model + the song, for the review site (make_site.py, site/videos/).
--still_only: only the middle-frame png (layout check).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'Motion-Projector'))
import smpl_render as SR  # noqa: E402

EX = os.path.join(HERE, 'data', 'appearance_examples')
ROT = os.path.join(HERE, 'data', 'rot_export')
SMPL = os.path.join(ROOT, 'Motion-Projector', 'models', 'smpl', 'SMPL_NEUTRAL_chumpyfree.pkl')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FPS, PW, PH, TITLE, HEAD = 30, 450, 800, 46, 50
PINK, BG = (0.96, 0.48, 0.68, 1.0), (0.95, 0.94, 0.95, 1.0)
GRID = [['real', 'flower'], ['oneshot', 'gain']]
LABELS = {'real': 'Real dance', 'flower': 'FlowerDance', 'oneshot': 'MACE-Dance', 'gain': 'Ours'}
ROW = [['flower', 'oneshot', 'gain', 'real']]
ROW_LABELS = {'real': 'Real Dance', 'flower': 'FlowerDance', 'oneshot': 'MACE-Dance', 'gain': 'MMC-Dance (Ours)'}
COLORS = {'real': (31, 119, 180), 'flower': (148, 103, 189), 'oneshot': (230, 92, 0), 'gain': (23, 150, 60)}
GH, FOOT, GWIN = 210, 0, 6.0  # --row: energy graph height, legend strip (none: explanations go in the supplement), graph time span (s)


def load(key, slice_, seed, real_tag, wild=False):
    if wild:
        z = np.load(os.path.join(HERE, 'data', 'wild', 'rot', f'{key}.npz'))
        i = [str(n) for n in z['names']].index(slice_)
        return z['poses'][i, seed].astype(np.float32), z['trans'][i, seed].astype(np.float32)
    if key == 'real':
        return SR.load_motion(os.path.join(EX, f'{real_tag}_{slice_}_real.pkl'))
    z = np.load(os.path.join(ROT, f'{key}_seed{seed}.npz'))
    names = list(z['names']) if 'names' in z.files else json.load(open(os.path.join(HERE, 'data', 'mace_test_flower', 'names.json')))['names']
    i = names.index(slice_)
    return z['poses'][i].astype(np.float32), z['trans'][i].astype(np.float32)


def fit_camera(verts, forward, aspect, margin):
    """Aim point and the camera distance at which EVERY frame of verts (T, V, 3) fits the frame (height and width, with margin),
    seen from the horizontal direction forward (smpl_render frames the first pose only, which clips later wide poses)."""
    pts = verts.reshape(-1, 3)
    center = (pts.max(0) + pts.min(0)) / 2
    side = np.array([-forward[1], forward[0], 0.0])
    rel = pts - center
    depth = rel @ np.array([forward[0], forward[1], 0.0])  # toward the camera (+) -> appears larger
    ty = np.tan(SR.FOV_Y / 2) / margin
    tx = ty * aspect
    # a point at offset h (and depth toward the camera) is inside the half-angle when h / (dist - depth) <= tan
    need = np.maximum(np.abs(rel[:, 2]) / ty, np.abs(rel @ side) / tx) + depth
    return center, float(need.max())


def render_fixed(verts, faces, forward, center, dist, width=720, height=1280):
    """As smpl_render.render_frames, with a given aim point and distance."""
    import pyrender
    import trimesh
    eye = center + np.array([forward[0], forward[1], 0.0]) * dist
    cam_pose = SR._look_at(eye, center, np.array([0.0, 0.0, 1.0]))
    scene = pyrender.Scene(bg_color=list(BG), ambient_light=[0.45, 0.45, 0.45])
    scene.add(pyrender.PerspectiveCamera(yfov=SR.FOV_Y, aspectRatio=width / height), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, roughnessFactor=0.8, baseColorFactor=list(PINK))
    renderer = pyrender.OffscreenRenderer(width, height)
    frames, node = [], None
    try:
        for v in verts:
            if node is not None:
                scene.remove_node(node)
            node = scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=v, faces=faces, process=False), material=material, smooth=True))
            frames.append(renderer.render(scene)[0])
    finally:
        renderer.delete()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slice', default='143_slice7')
    ap.add_argument('--seed', type=int, default=4)
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--dur', type=float, default=None, help='seconds (default: to the end of the clip)')
    ap.add_argument('--real_tag', default='lockbest')
    ap.add_argument('--margin', type=float, default=1.08, help='>1 leaves this much room around the widest/tallest pose')
    ap.add_argument('--still', action='store_true', help='also save a png of the middle frame')
    ap.add_argument('--wild', default=None, help='user-supplied song name (gen_wild_songs.py); 1 x 3 layout without the real dance')
    ap.add_argument('--wild_seeds', nargs='+', default=[], help='per-model seeds for --wild, e.g. flower=3 oneshot=5 gain=2')
    ap.add_argument('--row', action='store_true', help='1 x 4 row Flower | MACE | MMC-Dance (Ours) | Real, energy graph + beat clock, no header')
    ap.add_argument('--imc', action='store_true', help='with --row: motion energy vs music energy graph (IMC) instead of the beat graph')
    ap.add_argument('--split', action='store_true', help='with --imc: one silent video per model + the song (audio.m4a) in '
                    'site/videos/<slice>_seed<s>/ for the review site (make_site.py), instead of the row video')
    ap.add_argument('--still_only', action='store_true', help='only save the middle-frame png (layout preview), no video')
    ap.add_argument('--ffmpeg', default='ffmpeg')
    a = ap.parse_args()
    a.still = a.still or a.still_only
    a.row = a.row or a.imc
    from PIL import Image, ImageDraw, ImageFont

    grid, seeds, labels = (ROW, {}, ROW_LABELS) if a.row else (GRID, {}, LABELS)
    if a.wild:
        grid = [['flower', 'oneshot', 'gain']]
        seeds = {k: int(v) for k, v in (s.split('=') for s in a.wild_seeds)}
        wmeta = json.load(open(os.path.join(HERE, 'data', 'wild', a.wild, 'meta.json')))
        if not a.start:
            a.start = wmeta['window_start_in_segment_s']
        feats = np.load(os.path.join(HERE, 'data', 'wild', a.wild, 'feats.npy'))
        wav = os.path.join(HERE, 'data', 'wild', a.wild, 'segment.wav')
    else:
        feats = np.load(os.path.join(ROOT, 'Expert-Motion', 'data', 'test', 'baseline_feats', a.slice + '.npy'))
        wav = os.path.join(ROOT, 'Expert-Motion', 'data', 'test', 'wavs_sliced', a.slice + '.wav')
    Bt = np.flatnonzero(feats[:, 34])
    bpm = 60 * FPS / np.polyfit(np.arange(len(Bt)), Bt, 1)[0]

    geo, sig, f0 = {}, {}, int(round(a.start * FPS))
    if a.row:
        from render_pink_beads import lock_signals
    for k in sum(grid, []):
        poses, trans = load(k, a.wild, seeds[k], None, wild=True) if a.wild else load(k, a.slice, a.seed, a.real_tag)
        if a.row:  # signals over the whole clip (as the lock metric), shown for the rendered window
            sig[k] = lock_signals(poses, trans, Bt)
            sig[k]['qs'] = np.convolve(np.pad(sig[k]['q'], 2, mode='edge'), np.array([1, 4, 6, 4, 1]) / 16, 'valid')  # display smoothing only
            sig[k]['wave'] = sig[k]['q'].mean() + sig[k]['band']  # beat-rate part of the energy, drawn around its mean
            print(k, 'whole-clip clock r (accent phases)', round(float(np.abs(np.exp(2j * np.pi * sig[k]['ph']).mean())), 3), flush=True)
        f1 = len(poses) if a.dur is None else min(f0 + int(round(a.dur * FPS)), len(poses))
        p, t = poses[f0:f1].copy(), trans[f0:f1].copy()
        t[:, :2] = 0
        verts, faces = SR.smpl_vertices(p, t, SMPL)
        fwd = SR.facing_direction(p[0])
        geo[k] = (verts, faces, fwd, *fit_camera(verts, fwd, PW / PH, a.margin))
    dist = max(g[4] for g in geo.values())  # one distance for all panels: same scale, nobody clipped
    print('camera distance per panel', {k: round(g[4], 2) for k, g in geo.items()}, '-> used', round(dist, 2), flush=True)
    imgs = {}
    for k, (verts, faces, fwd, center, _) in geo.items():
        frames = render_fixed(verts, faces, fwd, center, dist)
        imgs[k] = [np.asarray(Image.fromarray(f).resize((PW, PH), Image.LANCZOS)) for f in frames]
        bgc = np.array(BG[:3]) * 255
        edge = sum(bool((np.abs(np.concatenate([f[:2].reshape(-1, 3), f[-2:].reshape(-1, 3), f[:, :2].reshape(-1, 3), f[:, -2:].reshape(-1, 3)]) - bgc) > 12).any())
                   for f in imgs[k])
        print(k, len(imgs[k]), f'frames touching the panel edge: {edge}', flush=True)
        del frames
    del geo
    nf = min(len(v) for v in imgs.values())

    f_h, f_t, f_s = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 16)
    nr, nc = len(grid), len(grid[0])
    Wc, Hc = nc * PW, HEAD + nr * (TITLE + PH)
    if a.row:
        Wc, Hc = nc * PW, TITLE + PH + GH + FOOT
        f_c = ImageFont.truetype(FONT, 13)
        ymax = 1.1 * max(float(np.percentile(s_['qs'], 99.5)) for s_ in sig.values())  # one energy scale for all panels
        t_end = (f0 + nf) / FPS
        if a.imc:
            from scipy.ndimage import gaussian_filter1d
            rms = np.load(os.path.join(EX, f'rms_{a.slice}.npy'))
            zs_ = lambda x: (x - x.mean()) / (x.std() + 1e-12)  # noqa: E731
            mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)  # noqa: E731
            mus = gaussian_filter1d(rms.astype(np.float64), 15)
            for k in sig:
                sig[k]['qi'] = gaussian_filter1d(sig[k]['q'], 15)
                print(k, 'IMC (SMPL QoM vs RMS, sigma 15, whole clip)', round(float((zs_(sig[k]['qi'][:len(mus)]) * zs_(mus[:len(sig[k]['qi'])])).mean()), 3), flush=True)
            span = slice(f0, f0 + nf)
            musn = mm(mus[span])
            for k in sig:
                sig[k]['qin'] = mm(sig[k]['qi'][span])
            # printed values = the reported whole-clip metrics (results/fullclip_bpl_imc_seeds.json, same seed for all models)
            met = {}
            for r_ in json.load(open(os.path.join(HERE, 'results', 'fullclip_bpl_imc_seeds.json'))):
                if r_['clip'] == a.slice and r_['seed'] == a.seed:
                    met = {k: dict(bpl=r_['bpl'][i], imc=r_['imc'][i]) for i, k in enumerate(['gain', 'oneshot', 'flower'])}
                    met['real'] = dict(bpl=r_['real'][0], imc=r_['real'][1])
            assert met, f'{a.slice} seed {a.seed} not in fullclip_bpl_imc_seeds.json'
            print('printed metrics', met, flush=True)
            f_v = ImageFont.truetype(FONT, 17)

        def imc_graph(d, x0, col, k, m):
            gx0, gx1 = x0 + 16, x0 + PW - 16
            xp = lambda i: gx0 + i / max(nf - 1, 1) * (gx1 - gx0)  # noqa: E731
            strips = [(TITLE + PH + 22, TITLE + PH + 92, sig[k]['qin'], col, 'Motion intensity (QoM)'),
                      (TITLE + PH + 112, TITLE + PH + 182, musn, (70, 70, 70), 'Audio intensity (RMS)')]
            for y0, y1, v, c_, name in strips:
                d.rectangle([gx0, y0, gx1, y1], outline=(200, 200, 200))
                pts_ = [(xp(i), y1 - 3 - v[i] * (y1 - y0 - 6)) for i in range(nf)]
                faint = tuple(int(255 - 0.3 * (255 - u)) for u in c_)
                d.line(pts_, fill=faint, width=2)
                if m > 0:
                    past = pts_[:m + 1]
                    d.polygon([(past[0][0], y1)] + past + [(past[-1][0], y1)], fill=tuple(int(255 - 0.22 * (255 - u)) for u in c_))
                    d.line(past, fill=c_, width=3)
                d.text((gx0 + 4, y0 - 15), name, fill=c_, font=f_c)
            vt = f"IMC {met[k]['imc']:.2f}"  # whole-clip IMC, right end of the motion box's title line (never over the curve)
            d.text((gx1 - d.textlength(vt, font=f_v), strips[0][0] - 19), vt, fill=col, font=f_v)
            d.line([(xp(m), TITLE + PH + 22), (xp(m), TITLE + PH + 182)], fill=(220, 30, 30), width=2)
            for sec in range(int(np.ceil(f0 / FPS)), int(t_end) + 1, 5):
                tl = f'{sec}'
                d.text((xp(sec * FPS - f0) - d.textlength(tl, font=f_c) / 2, TITLE + PH + 184), tl, fill=(120, 120, 120), font=f_c)

        def row_frame(m):
            canvas = Image.new('RGB', (Wc, Hc), 'white')
            d = ImageDraw.Draw(canvas)
            fr = f0 + m
            now = fr / FPS
            lo = min(max(now - GWIN / 2, f0 / FPS), max(t_end - GWIN, f0 / FPS))
            for c, k in enumerate(grid[0]):
                x0, col = c * PW, COLORS[k]
                d.text((x0 + (PW - d.textlength(labels[k], font=f_t)) / 2, 8), labels[k], fill=col, font=f_t)
                canvas.paste(Image.fromarray(imgs[k][m]), (x0, TITLE))
                # beat clock
                cx, cy, R = x0 + PW - 58, TITLE + 60, 40
                d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(255, 255, 255), outline=(150, 150, 150), width=2)
                d.polygon([(cx - 6, cy - R - 12), (cx + 6, cy - R - 12), (cx, cy - R - 2)], fill=(90, 90, 90))
                seen = sig[k]['ph'][:np.searchsorted(sig[k]['acc_in'], fr, side='right')]
                for p_ in seen:
                    px, py = cx + R * np.sin(2 * np.pi * p_), cy - R * np.cos(2 * np.pi * p_)
                    d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=col)
                if len(seen):
                    zc = np.exp(2j * np.pi * seen).mean()
                    d.line([(cx, cy), (cx + R * zc.imag, cy - R * zc.real)], fill=col, width=5)
                d.text((cx - d.textlength('beat clock', font=f_c) / 2, cy + R + 6), 'beat clock', fill=(110, 110, 110), font=f_c)
                if a.imc:
                    vt = f"BPL {met[k]['bpl']:.2f}"  # whole-clip BPL under the clock
                    d.text((cx - d.textlength(vt, font=f_v) / 2, cy + R + 22), vt, fill=col, font=f_v)
                    imc_graph(d, x0, col, k, m)
                    continue
                # energy graph
                gx0, gx1, gy0, gy1 = x0 + 16, x0 + PW - 16, TITLE + PH + 14, TITLE + PH + GH - 26
                xp = lambda t_: gx0 + (t_ - lo) / GWIN * (gx1 - gx0)  # noqa: E731
                yp = lambda v_: gy1 - min(v_ / ymax, 1.0) * (gy1 - gy0)  # noqa: E731
                d.rectangle([gx0, gy0, gx1, gy1], outline=(200, 200, 200))
                for b_ in Bt:
                    if lo <= b_ / FPS <= lo + GWIN:
                        d.line([(xp(b_ / FPS), gy0), (xp(b_ / FPS), gy1)], fill=(205, 205, 205), width=1)
                qs = sig[k]['qs']
                i0, i1 = max(int(np.floor(lo * FPS)), 0), min(int(np.ceil((lo + GWIN) * FPS)), len(qs) - 1)
                pts_ = [(xp(i / FPS), yp(qs[i])) for i in range(i0, i1 + 1)]
                light = tuple(int(255 - 0.25 * (255 - v)) for v in col)
                d.polygon([(pts_[0][0], gy1)] + pts_ + [(pts_[-1][0], gy1)], fill=light)
                d.line(pts_, fill=tuple(int(255 - 0.6 * (255 - v)) for v in col), width=1)
                wv = sig[k]['wave']
                dark = tuple(int(0.75 * v) for v in col)
                d.line([(xp(i / FPS), yp(wv[i])) for i in range(i0, i1 + 1)], fill=dark, width=3)
                ac = sig[k]['acc']
                for a_ in ac[(ac >= i0) & (ac <= i1)]:
                    d.ellipse([xp(a_ / FPS) - 5, yp(wv[a_]) - 5, xp(a_ / FPS) + 5, yp(wv[a_]) + 5], fill=dark, outline='white')
                d.line([(xp(now), gy0), (xp(now), gy1)], fill=(220, 30, 30), width=2)
                for sec in range(int(np.ceil(lo)), int(lo + GWIN) + 1):
                    tl = f'{sec}'
                    d.text((xp(sec) - d.textlength(tl, font=f_c) / 2, gy1 + 4), tl, fill=(120, 120, 120), font=f_c)
            for c in range(1, nc):
                d.line([(c * PW, 0), (c * PW, TITLE + PH + GH)], fill='white', width=4)
            return np.asarray(canvas)
    if a.wild:
        header = f'{a.wild} ({bpm:.0f} BPM) · seeds ' + ' '.join(f'{LABELS[k]} {seeds[k]}' for k in grid[0])
        clock0 = wmeta['segment_start_s']  # show song time
    else:
        header, clock0 = f'FineDance {a.slice} ({bpm:.0f} BPM) · seed {a.seed}', 0.0

    def frame(m):
        canvas = Image.new('RGB', (Wc, Hc), 'white')
        d = ImageDraw.Draw(canvas)
        d.text((16, 12), header, fill='black', font=f_h if not a.wild else f_s)
        tt = f'{clock0 + (f0 + m) / FPS:4.1f} s'
        d.text((Wc - 16 - d.textlength(tt, font=f_s), 17), tt, fill=(110, 110, 110), font=f_s)
        for r, row in enumerate(grid):
            for c, k in enumerate(row):
                x0, y0 = c * PW, HEAD + r * (TITLE + PH)
                d.text((x0 + (PW - d.textlength(LABELS[k], font=f_t)) / 2, y0 + 8), LABELS[k], fill='black', font=f_t)
                canvas.paste(Image.fromarray(imgs[k][m]), (x0, y0 + TITLE))
        for c in range(1, nc):
            d.line([(c * PW, HEAD), (c * PW, Hc)], fill='white', width=4)
        for r in range(1, nr):
            d.line([(0, HEAD + r * (TITLE + PH)), (Wc, HEAD + r * (TITLE + PH))], fill='white', width=4)
        return np.asarray(canvas)

    sfx = f'_{a.start:.2f}s' if (a.start or a.dur) else ''
    if a.row:
        frame = row_frame  # noqa: F811
        out = os.path.join(HERE, 'data', 'videos', 'showcase_windows', f'row_{"imc" if a.imc else "energy"}_{a.slice}_seed{a.seed}{sfx}.mp4')
    elif a.wild:
        os.makedirs(os.path.join(HERE, 'data', 'videos', 'wild'), exist_ok=True)
        out = os.path.join(HERE, 'data', 'videos', 'wild', f'grid_pink_{a.wild}.mp4')
    else:
        out = os.path.join(HERE, 'data', 'videos', f'grid_pink_{a.slice}_seed{a.seed}{sfx}.mp4')
    if a.still:
        Image.fromarray(frame(nf // 2)).save(out.replace('.mp4', '_still.png'))
        print(out.replace('.mp4', '_still.png'))
        if a.still_only:
            return
    import imageio
    if a.split:
        assert a.imc and not a.wild, '--split is for the --imc row'
        import make_site as MS
        sdir = os.path.join(MS.VIDS, f'{a.slice}_seed{a.seed}')
        os.makedirs(sdir, exist_ok=True)
        name = dict(zip(['flower', 'oneshot', 'gain', 'real'], MS.MODELS))
        tmp = {k: os.path.join(sdir, f'{name[k]}_tmp.mp4') for k in grid[0]}
        wrs = {k: imageio.get_writer(tmp[k], fps=FPS, quality=9, macro_block_size=1) for k in grid[0]}
        for m in range(nf):
            F = frame(m)
            for c, k in enumerate(grid[0]):
                wrs[k].append_data(F[:, c * PW:(c + 1) * PW])
        for k in grid[0]:
            wrs[k].close()
            MS.encode_video(a.ffmpeg, tmp[k], os.path.join(sdir, name[k] + '.mp4'))
            os.remove(tmp[k])
        MS.encode_audio(a.ffmpeg, wav, os.path.join(sdir, 'audio.m4a'), a.start, nf / FPS)
        print(sdir, '(now run make_site.py to update clips.js)')
        return
    silent = out.replace('.mp4', '_noaudio.mp4')
    wr = imageio.get_writer(silent, fps=FPS, quality=8, macro_block_size=1)
    for m in range(nf):
        wr.append_data(frame(m))
    wr.close()
    subprocess.run([a.ffmpeg, '-y', '-loglevel', 'error', '-i', silent, '-ss', f'{a.start:.3f}', '-i', wav, '-map', '0:v', '-map', '1:a',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-c:a', 'aac', '-b:a', '192k', '-shortest', out], check=True)
    os.remove(silent)
    print(out)


if __name__ == '__main__':
    main()
