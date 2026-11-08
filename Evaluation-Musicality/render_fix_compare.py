"""Real dancer vs one-shot MACE vs MACE with the +-2 s local cross-attention mask (2026-11-04).

Re-renders the two earlier error examples with the fix added as a third panel, same slice, same layout:
  loudness : 012_slice3 (render_eas_compare.py)  music loudness <-> whole-body motion energy, EAS
  perc     : 198_slice0 (render_component_compare.py)  percussive energy <-> upper-body acceleration, S
The one-shot MACE panel is the same sample as in the earlier videos (median score of 10 seeds). The fixed panel uses
the SAME seed, i.e. the same starting noise (gen_mace_test.py --local_attn 2 -> data/mace_test_local2s), so the only
difference between the two MACE panels is the attention mask. The fixed model's 10-seed median is printed too.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/render_fix_compare.py --ffmpeg /path/to/ffmpeg

Writes data/videos/<example>_<slice>_real_oneshot_fixed.mp4.

Final-model version (2026-11-06): --fixed_dir / --fixed_label / --oneshot_label / --suffix put any other run in the
third panel, e.g. ours = mask + guidance + foot term (same seed again, so the panels share the starting noise):
    .venv/bin/python ../Evaluation-Musicality/render_fix_compare.py --ffmpeg $FF \
        --fixed_dir ../Evaluation-Musicality/data/mace_test_mask2s_guide10_s0.003_foot10_10seeds \
        --fixed_label 'Ours: mask + guidance + foot' --oneshot_label 'MACE released' --suffix real_released_ours
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter as G  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import audio_features as af  # noqa: E402
import metrics as M  # noqa: E402
import multicomponent as MC  # noqa: E402
from eval_mace import fk  # noqa: E402
from model.metric import SMPLSkeleton  # noqa: E402
from render_eas_compare import LEFT, PARENTS, view_axes, zs  # noqa: E402

ONESHOT = os.path.join(HERE, 'data', 'mace_test')
FIXED = os.path.join(HERE, 'data', 'mace_test_local2s')
WINDOW = 8.0  # seconds of strip shown around the cursor (percussion)


def load(names, d, i):
    return np.asarray(np.load(os.path.join(d, 'joints.npy'), mmap_mode='r')[i], dtype=np.float32)


def loudness_example(n, gt, one, fix):
    feat = np.load(os.path.join('data/test/baseline_feats', n + '.npy'))
    rms = af.rms_envelope(af.load_audio(os.path.join('data/test/wavs_sliced', n + '.wav')))
    score = lambda j: M.clip_scores(feat, rms, j)['EAS']  # noqa: E731
    T = min(len(gt), one.shape[1], len(rms))
    curve = lambda j: zs(G(M.motion_energy(j[:T]), M.ENERGY_SMOOTH))  # noqa: E731
    return {'title': 'Music loudness ↔ whole-body motion energy', 'metric': 'EAS', 'score': score, 'curve': curve,
            'music': zs(G(rms[:T], M.ENERGY_SMOOTH)), 'music_label': 'music loudness', 'T': T, 'hot': None,
            'xlabel': 'seconds (z-scored, 0.5 s smoothing; EAS = correlation with loudness)', 'window': None}


def perc_example(n, gt, one, fix):
    comp, sigma, mkey = 'perc', MC.SIGMA['perc'], MC.MOTION_OF['perc']
    mus = MC.smooth(MC.music_features(n)[comp], sigma)
    curve_raw = lambda j: MC.smooth(MC.motion_features(j[:MC.T])[mkey], sigma)  # noqa: E731
    score = lambda j: MC.score(mus, curve_raw(j))[1]  # noqa: E731
    return {'title': 'Percussive energy ↔ upper-body acceleration', 'metric': 'S', 'score': score,
            'curve': lambda j: zs(curve_raw(j)), 'music': zs(mus), 'music_label': 'percussive energy', 'T': MC.T,
            'hot': set(MC.UPPER), 'xlabel': 'seconds (z-scored; S = correlation with the music curve, ±0.3 s lag)',
            'window': WINDOW}


def beat_example(n, gt, one, fix):
    """Beat locking (2026-11-06): body-speed accents against the song's beat grid. Score = the clip's Lock Δ as in
    beat_suite.py --tempo_null: PLV of the accents' beat phases, own beats minus the mean over the 12 other-song slices
    nearest in tempo."""
    from beat_suite import N_NULL, null_sets
    from tmd_beat_metrics import accents, velocity
    from tmd_beat_phase import phases, plv
    names = json.load(open(os.path.join(ONESHOT, 'names.json')))['names']
    beats = [np.flatnonzero(np.load(os.path.join('data/test/baseline_feats', x + '.npy'))[:, 34]) for x in names]
    bpm = np.array([60 * M.FPS / np.polyfit(np.arange(len(b)), b, 1)[0] for b in beats])
    i = names.index(n)
    null = null_sets([x.split('_')[0] for x in names], {'t': ('tempo', bpm)}, np.random.default_rng(0))['t'][i]
    assert len(null) == N_NULL
    A = beats[i]

    def score(j):
        m = accents(velocity(j, '3d'))
        return plv(phases(m, A)) - np.mean([plv(phases(m, beats[k])) for k in null])
    T = min(len(gt), one.shape[1])
    return {'title': f'Beat locking: body-speed accents vs the beats ({bpm[i]:.0f} BPM)', 'metric': 'Lock', 'score': score,
            'curve': lambda j: zs(velocity(j[:T], '3d')), 'accents': lambda j: accents(velocity(j[:T], '3d')),
            'beats': A[A < T], 'phase': lambda m: phases(m, A), 'music': None, 'music_label': None, 'T': T, 'hot': None,
            'xlabel': 'seconds · grey lines = beats, dots = accents (peaks of body speed) · Lock = own beats − 12 '
                      'tempo-matched songs; clock: each accent’s place in its beat', 'window': 6.0}


EXAMPLES = {'loudness': ('012_slice3', loudness_example), 'perc': ('198_slice0', perc_example),
            'beat': ('161_slice3', beat_example)}


def render(key, names, smpl, device, ffmpeg, out_dir, fixed_dir=FIXED, fixed_label='MACE + ±2 s mask',
           oneshot_label='MACE one-shot', suffix='real_oneshot_fixed', extra=()):
    """extra: (dir, label) runs shown between the one-shot and the fixed panel (same seed)."""
    n, make = EXAMPLES[key]
    i = names.index(n)
    with open(os.path.join('data/test/motions_sliced', n + '.pkl'), 'rb') as f:
        m = pickle.load(f)
    gt = fk(smpl, [m['q']], [m['pos']], device, from_pred=False)[0]
    one, fix = load(names, ONESHOT, i), load(names, fixed_dir, i)
    ex = make(n, gt, one, fix)
    T = ex['T']
    gt, one, fix = gt[:T], one[:, :T], fix[:, :T]
    s_one = [ex['score'](j) for j in one]
    s_fix = [ex['score'](j) for j in fix]
    k = int(np.argsort(s_one)[len(s_one) // 2])  # the earlier video's sample
    mids = []
    for (d, lab), col in zip(extra, ('tab:purple', 'tab:brown', 'tab:pink')):
        x = load(names, d, i)[:, :T]
        sx = [ex['score'](j) for j in x]
        mids.append((x[k], f'{lab} (seed {k})   {ex["metric"]} {sx[k]:+.2f}', col))
        print(f'  {lab}: seed {k} {sx[k]:+.2f} (10 seeds median {np.median(sx):+.2f})', flush=True)
    s_gt = ex['score'](gt)
    met = ex['metric']
    panels_def = [(gt, f'Real dancer   {met} {s_gt:+.2f}', 'tab:blue'),
                  (one[k], f'{oneshot_label} (seed {k})   {met} {s_one[k]:+.2f}', 'tab:orange'), *mids,
                  (fix[k], f'{fixed_label} (seed {k})   {met} {s_fix[k]:+.2f}', 'tab:green')]
    npan = len(panels_def)
    W = 480 * npan

    fig = plt.figure(figsize=(W / 100, 6.4), dpi=100)
    gs = fig.add_gridspec(2, npan, height_ratios=[3, 1.3])
    panels = []
    for col, (joints, label, color) in enumerate(panels_def):
        ax = fig.add_subplot(gs[0, col])
        h, up, sgn = view_axes(joints)
        rel = joints - joints[:, :1]
        ax.set_xlim(rel[..., h].min() - 0.2, rel[..., h].max() + 0.2)
        v = sgn * rel[..., up]
        ax.set_ylim(v.min() - 0.1, v.max() + 0.1)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=10)
        if ex['hot']:
            lines = [ax.plot([], [], lw=3.5 if c in ex['hot'] else 1.5,
                             color=color if c in ex['hot'] else '0.55')[0] for c in range(1, 24)]
        else:
            lines = [ax.plot([], [], lw=2, color='tab:red' if c in LEFT else 'k')[0] for c in range(1, 24)]
        beat_art = None
        if ex.get('beats') is not None:
            # beat lamp (lit just after each beat), accent flash on the skeleton, and a beat clock of the accents so far
            xl, yl = ax.get_xlim(), ax.get_ylim()
            lamp = ax.plot([xl[0] + 0.08 * (xl[1] - xl[0])], [yl[1] - 0.06 * (yl[1] - yl[0])], 'o', ms=16,
                           color='0.88')[0]
            ax.text(xl[0] + 0.15 * (xl[1] - xl[0]), yl[1] - 0.06 * (yl[1] - yl[0]), 'beat', va='center', fontsize=9,
                    color='0.4')
            clock = ax.inset_axes([0.70, 0.72, 0.28, 0.28])
            clock.set_xlim(-1.3, 1.3); clock.set_ylim(-1.3, 1.3); clock.set_aspect('equal'); clock.axis('off')
            u = np.linspace(0, 2 * np.pi, 100)
            clock.plot(np.sin(u), np.cos(u), color='0.6', lw=1)
            clock.plot([0], [1], 'v', color='0.3', ms=5)
            dots = clock.plot([], [], 'o', color=color, ms=3.5, alpha=0.7)[0]
            arrow = clock.plot([], [], color=color, lw=2.5)[0]
            acc = ex['accents'](joints)
            beat_art = (lamp, dots, arrow, acc, ex['phase'](acc), color)
        panels.append((rel, h, up, sgn, lines, beat_art))
    t = np.arange(T) / M.FPS
    strip = fig.add_subplot(gs[1, :])
    if ex.get('beats') is not None:
        for b in ex['beats']:
            strip.axvline(b / M.FPS, color='0.82', lw=0.9)
        for r, (joints, label, color) in enumerate(panels_def):
            c, acc = ex['curve'](joints), ex['accents'](joints)
            off = -3.2 * r
            strip.plot(t[:len(c)], c + off, color=color, lw=1.1, label=label.split('   ')[0])
            strip.plot(acc / M.FPS, c[acc] + off, 'o', color=color, ms=4)
    else:
        strip.plot(t, ex['music'], color='k', lw=1.5, label=ex['music_label'])
        for joints, label, color in panels_def:
            strip.plot(t, ex['curve'](joints), color=color, lw=1.1, label=label.split('   ')[0])
    strip.set_xlim(0, T / M.FPS); strip.set_yticks([])
    strip.set_xlabel(ex['xlabel'])
    strip.legend(loc='upper right', fontsize=7, ncol=npan + 1)
    cursor = strip.axvline(0, color='red', lw=1.5)
    sup = fig.suptitle('')
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # keep the suptitle clear of the panel titles

    out = os.path.join(out_dir, f'{key}_{n}_{suffix}.mp4')
    wav = os.path.join('data/test/wavs_sliced', n + '.wav')
    proc = subprocess.Popen([ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
                             '-s', f'{W}x640', '-r', str(M.FPS), '-i', '-', '-i', wav, '-c:v', 'libx264',
                             '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', out], stdin=subprocess.PIPE)
    for fr in range(T):
        for rel, h, up, sgn, lines, beat_art in panels:
            flash = False
            if beat_art is not None:
                lamp, dots, arrow, acc, ph, color = beat_art
                B = ex['beats']
                lamp.set_color('#16213a' if ((fr - B >= 0) & (fr - B < 4)).any() else '0.88')
                flash = bool(((fr - acc >= 0) & (fr - acc < 4)).any())
                seen = ph[:np.searchsorted(acc[(acc >= B[0]) & (acc < B[-1])], fr, side='right')]
                if len(seen):
                    z = np.exp(2j * np.pi * seen)
                    dots.set_data(np.sin(2 * np.pi * seen), np.cos(2 * np.pi * seen))
                    r = z.mean()
                    arrow.set_data([0, r.imag], [0, r.real])
            for c, ln in zip(range(1, 24), lines):
                p = PARENTS[c]
                ln.set_data([rel[fr, p, h], rel[fr, c, h]], [sgn * rel[fr, p, up], sgn * rel[fr, c, up]])
                if beat_art is not None:
                    ln.set_linewidth(3.6 if flash else 2)
        now = fr / M.FPS
        if ex['window']:
            lo = min(max(0.0, now - ex['window'] / 2), t[-1] - ex['window'])
            strip.set_xlim(lo, lo + ex['window'])
        cursor.set_xdata([now])
        sup.set_text(f"{ex['title']}   ·   song {n}   {now:4.1f} s")
        fig.canvas.draw()
        proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    print(f'{out}\n  real {met} {s_gt:+.2f} | one-shot seed {k} {s_one[k]:+.2f} (10 seeds median '
          f'{np.median(s_one):+.2f}, {min(s_one):+.2f}..{max(s_one):+.2f}) | fixed seed {k} {s_fix[k]:+.2f} '
          f'(10 seeds median {np.median(s_fix):+.2f}, {min(s_fix):+.2f}..{max(s_fix):+.2f})', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--examples', nargs='*', default=list(EXAMPLES))
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'data', 'videos'))
    ap.add_argument('--ffmpeg', default='ffmpeg')
    ap.add_argument('--fixed_dir', default=FIXED, help='run shown in the third panel')
    ap.add_argument('--fixed_label', default='MACE + ±2 s mask')
    ap.add_argument('--oneshot_label', default='MACE one-shot')
    ap.add_argument('--suffix', default='real_oneshot_fixed')
    ap.add_argument('--extra', nargs=2, action='append', default=[], metavar=('DIR', 'LABEL'),
                    help='extra run panel between one-shot and fixed (repeatable)')
    ap.add_argument('--slice', default=None,
                    help='render this test slice instead of the example\'s default (one example only), e.g. '
                         '--examples perc --slice 063_slice0')
    a = ap.parse_args()
    if a.slice:
        assert len(a.examples) == 1, '--slice needs exactly one --examples entry'
        EXAMPLES[a.examples[0]] = (a.slice, EXAMPLES[a.examples[0]][1])
    with open(os.path.join(ONESHOT, 'names.json')) as f:
        names = json.load(f)['names']
    with open(os.path.join(a.fixed_dir, 'names.json')) as f:
        assert json.load(f)['names'] == names
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(device)
    for key in a.examples:
        render(key, names, smpl, device, a.ffmpeg, a.out_dir, a.fixed_dir, a.fixed_label, a.oneshot_label, a.suffix, a.extra)


if __name__ == '__main__':
    main()
