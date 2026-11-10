"""Beat photo series on the Motorica stretch where the real dancer is steady on the beat (2026-11-09).

One pose per beat for 8 consecutive beats inside the stretch of find_real_stretch.py, real | MACE released | ours (the same seeds
as render_beads_motorica.py: those whose steadiness R in the stretch is closest to the median of their seeds), and an overlay
column with the 8 poses on top of each other. Under each overlay: the pose spread (RMS distance of the poses from their mean pose,
pelvis-relative, screen plane) and the same spread for the beat grid shifted by 0..0.95 beat over the whole stretch (mean), so the
picture can be read against a number. Beats are the TRUE beats for all three dancers.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/render_beat_photos_motorica.py [--first 3] [--nbeats 8]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import rank_hits_on_beat as H  # noqa: E402
import render_beads_motorica as B  # noqa: E402

FPS = 30


def spread(xy, t):
    p = xy[t]
    return float(np.sqrt(((p - p.mean(0)) ** 2).sum(-1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', default='kthstreet_gLH_sFM_cAll_d01_mLH_ch01_yougonnaregretit_003#181')
    ap.add_argument('--first', type=int, default=3, help='first beat of the photo series, counted from the start of the stretch')
    ap.add_argument('--nbeats', type=int, default=8)
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'data', 'videos'))
    a = ap.parse_args()
    st = max([x for x in json.load(open(os.path.join(HERE, 'results', 'real_stretch_motorica.json'))) if x['n'] == a.clip],
             key=lambda x: x['beats'])
    H.BS.MOT_MODELS.clear()
    H.BS.MOT_MODELS.update({'released': 'mace', 'ours': H.MOT_OURS})
    group, beats, sets, _ = H.BS.load_motorica()
    k = [f'{g}#{i}' for i, g in enumerate(group)].index(a.clip)
    A = np.asarray(beats[k])
    real = np.asarray(sets['real'][k][0])[:1020]
    one, ours = np.asarray(sets['released'][k])[:, :1020], np.asarray(sets['ours'][k])[:, :1020]
    b0, nb = st['beat0'], st['beats']
    si, _ = B.pick(one, A, b0, nb)
    su, _ = B.pick(ours, A, b0, nb)
    rows = [('Real dancer (mocap)', real, 'real', 'tab:blue'), (f'MACE released (seed {si})', one[si], 'smpl', 'tab:orange'),
            (f'Ours: critic v5 + gain (seed {su})', ours[su], 'smpl', 'tab:green')]
    K = a.nbeats
    sel = [b0 + a.first + c for c in range(K)]
    assert sel[-1] < b0 + nb, 'photo series must stay inside the steady stretch'
    XY = [B.screen_xy(j, B.SKEL[kind]) for _, j, kind, _ in rows]
    allxy = np.concatenate([x.reshape(-1, 2) for x in XY])
    xl, yl = (allxy[:, 0].min() - 0.1, allxy[:, 0].max() + 0.1), (allxy[:, 1].min() - 0.05, allxy[:, 1].max() + 0.05)
    shifts = np.linspace(0, 0.95, 20)
    bpm = 60 * FPS / np.median(np.diff(A))
    fig, axes = plt.subplots(3, K + 1, figsize=(1.9 * (K + 1), 3.2 * 3))
    print(f'{a.clip}: {bpm:.0f} BPM, stretch beats {b0}..{b0 + nb} ({A[b0] / FPS:.1f}-{A[b0 + nb] / FPS:.1f} s), photos at beats {sel[0]}..{sel[-1]} '
          f'({A[sel[0]] / FPS:.1f}-{A[sel[-1]] / FPS:.1f} s)')
    for r, ((label, j, kind, color), xy) in enumerate(zip(rows, XY)):
        sk = B.SKEL[kind]
        t_w = A[sel].astype(int)
        # spread of every K consecutive beats of the stretch on the shifted grids (mean over shifts and windows)
        def win(sh):
            g = np.clip(np.round(A + sh * np.median(np.diff(A))).astype(int), 0, 1019)
            return np.mean([spread(xy, g[q:q + K]) for q in range(b0, b0 + nb - K + 1)])
        null = np.mean([win(s) for s in shifts])
        sp = spread(xy, t_w)
        print(f'  {label:36s} pose spread {sp:.3f}   shifted-grid mean over the stretch {null:.3f}   ratio {win(0) / null:.2f}')
        for c in range(K + 1):
            ax = axes[r, c]
            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            frames = [t_w[c]] if c < K else list(t_w)
            for f in frames:
                for cc in range(1, len(sk['parents'])):
                    p = sk['parents'][cc]
                    ax.plot([xy[f, p, 0], xy[f, cc, 0]], [xy[f, p, 1], xy[f, cc, 1]], lw=1.8 if c < K else 1.2,
                            color=('tab:red' if cc in sk['left'] else '0.3') if c < K else color, alpha=1 if c < K else 0.3)
                ax.plot(xy[f, list(sk['limbs']), 0], xy[f, list(sk['limbs']), 1], 'o', ms=5 if c < K else 4, color='#ffd11a',
                        mec='#403000', mew=0.8 if c < K else 0.5, alpha=1 if c < K else 0.8, zorder=5)
            if r == 0:
                ax.set_title(f'beat {c + 1}\n{A[sel[c]] / FPS:.1f} s' if c < K else f'all {K} beats\noverlaid', fontsize=9,
                             fontweight='normal' if c < K else 'bold')
            if c == K:
                ax.set_facecolor('#f6f6f6')
                ax.set_xlabel(f'pose spread {sp:.3f}\n(shifted-beat level {null:.3f})', fontsize=8)
        axes[r, 0].set_ylabel(label, fontsize=10)
    fig.suptitle(f'One pose per beat, {K} consecutive beats inside the real dancer\'s steady stretch ({A[b0] / FPS:.1f}-{A[b0 + nb] / FPS:.1f} s, '
                 f'{bpm:.0f} BPM) · yougonnaregretit_003', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(a.out_dir, 'beatphotos_motorica_yougonnaregretit_003_181.png')
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == '__main__':
    main()
