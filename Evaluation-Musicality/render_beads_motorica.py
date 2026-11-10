"""Beat beads on a Motorica stretch where the real dancer is steady on the beat (2026-11-09).

Real dancer (BVH, 21 joints) | MACE released | ours (critic v5 + gain) (SMPL, 24 joints), one take, one stretch picked by
find_real_stretch.py (real keeps the same place in the beat for 13 beats, mean phase within 0.1 beat of the beat). Pelvis-centred
skeletons; trails behind both hands and both feet, as thick as the limb is fast; a gold bead at every beat of the song. The
generated seeds are the ones whose steadiness R in the stretch is closest to the median of their seeds (10 released, 3 ours), so
neither is a favourable or unfavourable draw. Beads at the TRUE beat for all three dancers.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/render_beads_motorica.py --ffmpeg $FF --still 10.0 [--no_video]
Writes data/videos/beads_motorica_<take>_<k>.mp4 (+ _still.png).
"""
import argparse
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import rank_hits_on_beat as H  # noqa: E402
from motorica_common import wav_path  # noqa: E402
from render_eas_compare import ANKLES, HEAD, LEFT, PARENTS, SHOULDERS  # noqa: E402
from tmd_beat_metrics import accents, velocity  # noqa: E402
from tmd_beat_phase import phases  # noqa: E402

FPS = 30
GOLD, GOLD_EDGE = (1.0, 0.82, 0.1), (0.25, 0.18, 0.0)
# Motorica BODY order (motorica_common.BODY): Hips Spine Spine1 Neck Head | L: Shoulder Arm ForeArm Hand | R: same | L: UpLeg Leg Foot Toe | R: same
MP = [-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
MLEFT = {5, 6, 7, 8, 13, 14, 15, 16}
SKEL = {'real': dict(parents=MP, left=MLEFT, head=4, feet=(15, 19), sh=(5, 9), limbs=(8, 12, 15, 19)),
        'smpl': dict(parents=PARENTS, left=LEFT, head=HEAD, feet=ANKLES, sh=SHOULDERS, limbs=(20, 21, 7, 8))}


def screen_xy(j, sk):
    """pelvis-relative (T, J, 2): horizontal and vertical axes inferred from the skeleton itself."""
    up_vec = (j[:, sk['head']] - j[:, list(sk['feet'])].mean(1)).mean(0)
    up = int(np.argmax(np.abs(up_vec)))
    side = np.abs(j[:, sk['sh'][0]] - j[:, sk['sh'][1]]).mean(0)
    side[up] = -1
    h = int(np.argmax(side))
    rel = j - j[:, :1]
    return np.stack([rel[..., h], np.sign(up_vec[up]) * rel[..., up]], -1)


def steadiness(j, A, b0, nb):
    """R of the accents' beat phases over beats b0 .. b0+nb (1 = every accent at the same place in its beat)."""
    M = accents(velocity(j, '3d'))
    M = M[(M >= A[b0]) & (M < A[b0 + nb])]
    ph = phases(M, A)
    return float(abs(np.exp(2j * np.pi * ph).mean())) if len(ph) >= 4 else np.nan


def pick(js, A, b0, nb):
    R = np.array([steadiness(j, A, b0, nb) for j in js])
    k = int(np.nanargmin(np.abs(R - np.nanmedian(R))))
    return k, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', default='kthstreet_gLH_sFM_cAll_d01_mLH_ch01_yougonnaregretit_003#181')
    ap.add_argument('--lead', type=float, default=1.5, help='seconds shown before the steady stretch')
    ap.add_argument('--tail', type=float, default=0.5)
    ap.add_argument('--trail', type=float, default=1.0)
    ap.add_argument('--seed_one', type=int, default=None)
    ap.add_argument('--seed_ours', type=int, default=None)
    ap.add_argument('--still', type=float, default=None, help='save a png of this second of the slice')
    ap.add_argument('--no_video', action='store_true')
    ap.add_argument('--sheet', type=int, default=0, help='save a contact sheet of this many consecutive beats from the stretch start')
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'data', 'videos'))
    ap.add_argument('--ffmpeg', default='ffmpeg')
    a = ap.parse_args()

    st = [x for x in json.load(open(os.path.join(HERE, 'results', 'real_stretch_motorica.json'))) if x['n'] == a.clip]
    st = max(st, key=lambda x: x['beats'])
    H.BS.MOT_MODELS.clear()
    H.BS.MOT_MODELS.update({'released': 'mace', 'ours': H.MOT_OURS})
    group, beats, sets, _ = H.BS.load_motorica()
    k = [f'{g}#{i}' for i, g in enumerate(group)].index(a.clip)
    meta = json.load(open(os.path.join(HERE, 'data', 'motorica', 'mace', 'slices.json')))
    keep = [i for i, (t, o) in enumerate(meta['slices']) if o % 1020 == 0]
    take, off = meta['slices'][keep[k]]
    A = np.asarray(beats[k])
    real = np.asarray(sets['real'][k][0])[:1020]
    one, ours = np.asarray(sets['released'][k])[:, :1020], np.asarray(sets['ours'][k])[:, :1020]
    b0, nb = st['beat0'], st['beats']
    si, Ro = pick(one, A, b0, nb) if a.seed_one is None else (a.seed_one, None)
    su, Ru = pick(ours, A, b0, nb) if a.seed_ours is None else (a.seed_ours, None)
    Rr = steadiness(real, A, b0, nb)
    bpm = 60 * FPS / np.median(np.diff(A))
    print(f'{take} slice offset {off}  {bpm:.0f} BPM  stretch beats {b0}..{b0 + nb} = {A[b0] / FPS:.1f}-{A[b0 + nb] / FPS:.1f} s')
    print(f'  steadiness R in the stretch: real {Rr:.2f}; released seeds {np.round(Ro, 2) if Ro is not None else "-"} -> seed {si}; '
          f'ours seeds {np.round(Ru, 2) if Ru is not None else "-"} -> seed {su}')
    f0 = max(0, int(round(A[b0] - a.lead * FPS)))
    f1 = min(1019, int(round(A[b0 + nb] + a.tail * FPS)))
    W = int(round(a.trail * FPS))
    defs = [(real, 'real', 'Real dancer (mocap)', 'tab:blue'), (one[si], 'smpl', f'MACE released (seed {si})', 'tab:orange'),
            (ours[su], 'smpl', f'Ours: critic v5 + gain (seed {su})', 'tab:green')]

    P, SP = [], []
    for j, kind, label, color in defs:
        sk = SKEL[kind]
        xy = screen_xy(j, sk)
        P.append((xy, sk, label, color))
        SP.append(gaussian_filter1d(np.linalg.norm(np.gradient(xy[:, list(sk['limbs'])], axis=0) * FPS, axis=-1), 1.0, axis=0))
    refs = [np.percentile(x.ravel(), 95) for x in SP]  # trail width scaled per dancer: the beads show WHERE on a swing, not who is faster
    allxy = np.concatenate([p[0].reshape(-1, 2) for p in P])
    xl, yl = (allxy[:, 0].min() - 0.15, allxy[:, 0].max() + 0.15), (allxy[:, 1].min() - 0.1, allxy[:, 1].max() + 0.1)

    fig = plt.figure(figsize=(14.405, 6.605), dpi=100)
    arts = []
    for col, (xy, sk, label, color) in enumerate(P):
        ax = fig.add_subplot(1, 3, col + 1)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=12)
        bones = [ax.plot([], [], lw=1.6, color='tab:red' if c in sk['left'] else '0.35', zorder=2)[0] for c in range(1, len(sk['parents']))]
        trails = [LineCollection([], linewidths=2.6, zorder=4, capstyle='butt') for _ in sk['limbs']]
        for lc in trails:
            ax.add_collection(lc)
        beads = ax.scatter([], [], s=60, c=[GOLD], edgecolors=[GOLD_EDGE], linewidths=1.2, zorder=6)
        lamp = ax.plot([xl[0] + 0.1 * (xl[1] - xl[0])], [yl[1] - 0.06 * (yl[1] - yl[0])], 'o', ms=15, color='0.88')[0]
        ax.text(xl[0] + 0.19 * (xl[1] - xl[0]), yl[1] - 0.06 * (yl[1] - yl[0]), 'beat', va='center', fontsize=9, color='0.4')
        arts.append((xy, sk, color, bones, trails, beads, lamp, col))
    sup = fig.suptitle('')
    fig.text(0.5, 0.015, 'trails = both hands and feet (last 1 s), thicker = faster · gold beads = where each limb was at every beat of '
             'the song · trail width is scaled per dancer · the real dancer keeps the same place in the beat through the marked stretch', ha='center', fontsize=9, color='0.35')
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    def draw(fr):
        inside = A[b0] <= fr <= A[b0 + nb]
        for xy, sk, color, bones, trails, beads, lamp, pi in arts:
            for c, ln in zip(range(1, len(sk['parents'])), bones):
                p = sk['parents'][c]
                ln.set_data([xy[fr, p, 0], xy[fr, c, 0]], [xy[fr, p, 1], xy[fr, c, 1]])
            lamp.set_color('#16213a' if ((fr - A >= 0) & (fr - A < 4)).any() else '0.88')
            lo = max(0, fr - W)
            rgb = np.array(matplotlib.colors.to_rgb(color))
            for jn, (j, lc) in enumerate(zip(sk['limbs'], trails)):
                pts = xy[lo:fr + 1, j]
                if len(pts) < 2:
                    lc.set_segments([]); continue
                lc.set_segments(np.stack([pts[:-1], pts[1:]], 1))
                age = (np.arange(len(pts) - 1) + 1) / W
                al = (0.15 + 0.85 * age)[:, None]
                lc.set_color(rgb * al + (1 - al))
                lc.set_linewidth(1.0 + 9.0 * np.clip(SP[pi][lo + 1:fr + 1, jn] / refs[pi], 0, 1.15))
            recent = A[(A >= lo) & (A <= fr)]
            if len(recent):
                off_ = np.concatenate([xy[recent, j] for j in sk['limbs']])
                age = np.tile(fr - recent, len(sk['limbs']))
                al = np.clip(1.0 - 0.6 * age / W, 0, 1)
                fc = np.tile(np.array(GOLD + (1,)), (len(off_), 1)); fc[:, 3] = al
                ec = np.tile(np.array(GOLD_EDGE + (1,)), (len(off_), 1)); ec[:, 3] = al
                beads.set_offsets(off_); beads.set_sizes(45 + 130 * np.clip(1 - age / 6, 0, 1))
                beads.set_facecolors(fc); beads.set_edgecolors(ec)
            else:
                beads.set_offsets(np.empty((0, 2)))
        sup.set_text(f'Beat beads   ·   {take.split("_", 5)[-1]}  ({bpm:.0f} BPM)   {fr / FPS:4.1f} s' +
                     ('   ·   ● real dancer steady on the beat' if inside else ''))
        sup.set_color('#16213a' if inside else '0.4')
        fig.canvas.draw()

    tag = f'{a.clip.split("#")[0].split("_")[-2]}_{a.clip.split("#")[0].split("_")[-1]}_{k}'
    if a.still is not None:
        draw(int(round(a.still * FPS)))
        p = os.path.join(a.out_dir, f'beads_motorica_{tag}_still.png')
        fig.savefig(p); print(p)
    if a.sheet:
        from PIL import Image
        ims = []
        for q in range(a.sheet):
            draw(int(A[b0 + 3 + q]) + 2)
            ims.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert('RGB'))
        w, h = ims[0].size
        sheet = Image.new('RGB', (w, h * a.sheet), 'white')
        for q, im in enumerate(ims):
            sheet.paste(im, (0, q * h))
        sheet = sheet.resize((w * 2 // 3, h * a.sheet * 2 // 3))
        p = os.path.join(a.out_dir, f'beads_motorica_{tag}_sheet.png')
        sheet.save(p); print(p)
    if not a.no_video:
        out = os.path.join(a.out_dir, f'beads_motorica_{tag}.mp4')
        Wpx, Hpx = fig.canvas.get_width_height()
        proc = subprocess.Popen([a.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{Wpx}x{Hpx}',
                                 '-r', str(FPS), '-i', '-', '-ss', f'{(off + f0) / FPS:.3f}', '-t', f'{(f1 - f0) / FPS:.3f}',
                                 '-i', wav_path(take), '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                                 '-crf', '18', '-c:a', 'aac', '-shortest', out], stdin=subprocess.PIPE)
        for fr in range(f0, f1):
            draw(fr)
            proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        proc.stdin.close(); proc.wait()
        print(out, f'{f1 - f0} frames')


if __name__ == '__main__':
    main()
