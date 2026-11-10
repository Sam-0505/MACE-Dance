"""Beat-locking video on the pink mannequin: real | MACE released | Ours (--no_real leaves the real dancer out), the SMPL body of each
motion drawn as a featureless figure, with the beat-locking overlay: a beat lamp per panel, a phase clock (top right of each panel: where
each peak of the beat-rate motion falls inside its beat, arrow = their average) and a beat graph (bottom: the band-passed whole-body
motion at the beat rate, its peaks, grey lines = beats). By default the body shade IS the beat-rate wave of the graph (z-score -1.5 .. +1.5
-> light .. dark); --plain draws an unshaded pink body instead. --slow 2 interpolates the poses (slerp) to 60 fps and plays them at 30 fps
with the audio at half speed.

SMPL files: export_smpl_gain.py --slice 161_slice2 --seed 1 --tag beads1 --variants real oneshot gain (Expert-Motion venv).

    cd Motion-Projector
    .venv/bin/python ../Evaluation-Musicality/render_pink_beads.py --ffmpeg $FF --seed_tag beads1 --slice 161_slice2 --start 13.5 --dur 10.2 [--plain] [--slow 2] [--still 16.0]

Writes Evaluation-Musicality/data/videos/beads_pink_<slice>_<tag>[_plain][_shadelock][_slow2x].mp4 (+ _still.png).
(The earlier speed-shaded, bead-trail, photo-strip and phase-grid modes were removed 2026-11-10; see NEGATIVE_RESULTS.md.)
"""
import argparse
import os
import subprocess
import sys

import numpy as np
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
KEYS = ['real', 'oneshot', 'gain']
LABELS = {'real': 'Real dancer', 'oneshot': 'MACE released', 'gain': 'Ours: critic v5 + gain'}
COLORS = {'real': (31, 119, 180), 'oneshot': (230, 92, 0), 'gain': (23, 150, 60)}


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


STOPS = [(0.0, (215, 215, 215)), (1.0, (20, 20, 20))]  # light grey = trough of the beat-rate wave, black = peak


def gradient(n=256):
    """(n, 3) uint8 colour scale of the body shade, light grey -> black."""
    g = np.linspace(0, 1, n)
    return np.stack([np.interp(g, [t for t, _ in STOPS], [c[i] for _, c in STOPS]) for i in range(3)], 1).astype(np.uint8)


def draw_legend(canvas, d, x, y, w, h, font):
    """colour bar under the panels: trough .. peak of the beat-rate wave."""
    from PIL import Image
    canvas.paste(Image.fromarray(np.tile(gradient(w)[None], (h, 1, 1))), (x, y))
    d.rectangle([x, y, x + w, y + h], outline=(120, 120, 120))
    d.text((x - 178, y - 1), 'trough of the wave', fill=(70, 70, 70), font=font)
    d.text((x + w + 6, y - 1), 'peak of the wave', fill=(70, 70, 70), font=font)


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
    ap.add_argument('--still', type=float, default=None, help='also save a png at this second of the clip')
    ap.add_argument('--no_video', action='store_true')
    ap.add_argument('--note', default='', help='text shown in the header (e.g. which seed this is)')
    ap.add_argument('--slice', default='161_slice2', help='test clip (SMPL files <seed_tag>_<slice>_<variant>.pkl from export_smpl_gain.py)')
    ap.add_argument('--plain', action='store_true', help='plain pink mannequin, no shade (the clock and graph are still drawn)')
    ap.add_argument('--no_real', action='store_true', help='leave the real dancer out (rows: MACE released, ours)')
    ap.add_argument('--tag', default='seed1')
    ap.add_argument('--ffmpeg', default='ffmpeg')
    a = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    global KEYS, N
    if not a.plain:
        a.tag += '_shadelock'
    if a.no_real:
        KEYS = ['oneshot', 'gain']
    N = a.slice

    U = a.slow
    f0 = int(round(a.start * FPS))
    nf = int(round(a.dur * FPS))
    ts = f0 + np.arange(nf * U) / U  # source frame time of every output frame
    feats = np.load(os.path.join(ROOT, 'Expert-Motion', 'data', 'test', 'baseline_feats', N + '.npy'))
    Bt = np.flatnonzero(feats[:, 34])
    bpm = 60 * FPS / np.polyfit(np.arange(len(Bt)), Bt, 1)[0]
    Bm = (Bt - f0) * U  # beat positions in output frames (can be outside the window)

    imgs, geo, lock = {}, {}, {}
    for k in KEYS:
        poses, trans = SR.load_motion(os.path.join(EX, f'{a.seed_tag}_{N}_{k}.pkl'))
        lock[k] = lock_signals(poses, trans, Bt)
        lo = max(int(ts[0]) - 1, 0)
        hi = min(int(ts[-1]) + 3, len(poses))
        p, t = interp(poses[lo:hi], trans[lo:hi], ts - lo)
        t[:, :2] = 0  # centre on the pelvis in the ground plane, keep the height (as render_pink_lock --full)
        verts, faces = SR.smpl_vertices(p, t, SMPL)
        fwd = SR.facing_direction(p[0])
        if not a.plain:
            geo[k] = (verts, faces, fwd)
            continue
        frames = SR.render_frames(verts, faces, fwd, 720, 1280, body_fraction=0.56, body_color=PINK, bg_color=BG)
        imgs[k] = [np.asarray(Image.fromarray(f).resize((PW, PH), Image.LANCZOS)) for f in frames]
        print(k, len(imgs[k]), flush=True)
        del frames, verts
    if not a.plain:
        # shade = height of the beat-rate wave (z-score) at each frame: -1.5 .. +1.5 -> 0 .. 1
        ix = np.clip(f0 + np.round(np.arange(len(ts)) / U).astype(int), 0, len(lock[KEYS[0]]['curve']) - 1)
        tot = {k: 1.0 + lock[k]['curve'][ix] / 3.0 for k in KEYS}
        vref, lo0 = 1.5, 0.5
        for k, (verts, faces, fwd) in geo.items():
            frames = render_colored(verts, faces, fwd, np.repeat(np.clip((tot[k] - lo0) / (vref - lo0), 0, 1)[:, None], verts.shape[1], 1))
            imgs[k] = [np.asarray(Image.fromarray(f).resize((PW, PH), Image.LANCZOS)) for f in frames]
            print(k, len(imgs[k]), flush=True)

    f_t, f_s = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 14)
    GH = 300
    Wc, Hc = PW * len(KEYS), HEAD + PH + FOOT + GH

    def frame(m):
        canvas = Image.new('RGB', (Wc, Hc), 'white')
        d = ImageDraw.Draw(canvas)
        d.text((16, 8), f'{"Beat locking" if a.plain else "Speed glow"}   ·   song {N} ({bpm:.0f} BPM)   {ts[m] / FPS:4.1f} s' + ('   ·   half speed' if U > 1 else ''),
               fill='black', font=f_t)
        for c, k in enumerate(KEYS):
            x0 = c * PW
            canvas.paste(Image.fromarray(imgs[k][m]), (x0, HEAD))
            d.text((x0 + 16, 62), LABELS[k], fill='black', font=f_t)
            lit = bool(((m - Bm >= 0) & (m - Bm < 4 * U)).any())
            d.ellipse([x0 + 14, HEAD + 14, x0 + 40, HEAD + 40], fill=(22, 33, 58) if lit else (225, 225, 225))
            d.text((x0 + 48, HEAD + 18), 'beat', fill=(100, 100, 100), font=f_s)
            if True:  # phase clock: place of each peak of the beat-rate motion inside its beat (top = on the beat), arrow = their average
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
        if not a.plain:
            draw_legend(canvas, d, 16 + 178, HEAD + PH + 34, 240, 14, f_s)
        if a.note:
            d.text((16, 36), a.note, fill=(150, 40, 40), font=f_s)
        d.text((16, HEAD + PH + 8), 'shade = height of the beat-rate wave in the graph below (dark = peak)'
               + (' (same scale for all three) · ' if len(KEYS) == 3 else ' · ') + 'lamp = a beat', fill=(90, 90, 90), font=f_s)
        if a.plain:
            d.rectangle([0, HEAD + PH + 4, Wc, HEAD + PH + FOOT - 2], fill='white')
            d.text((16, HEAD + PH + 8), 'plain mannequin, no shading · lamp = a beat · the clock and graph show the timing of the beat-rate motion', fill=(90, 90, 90), font=f_s)
        if True:
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

    tag = a.tag + ('_plain' if a.plain else '') + ('_slow2x' if U > 1 else '')
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
