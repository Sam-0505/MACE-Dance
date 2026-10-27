"""Validate the musicality metrics on a few ImperialDance choreographies, per expertise level.

    python fetch_imperial_audio.py ...           # padded music windows
    python eval_imperial.py

motion_dir holds dance_level_<e|i|b>_<choreo>.npy, each (repeats, 3, frames @100 fps, 21 joints).

The motion lengths do not match the README music ranges, so the start of the motion inside
the padded music window is estimated: the offset maximizing the correlation between the
music onset envelope and motion acceleration, averaged over the even-indexed repeats of all
levels. Metrics are reported on the odd-indexed repeats only, so the offset search does not
inflate them.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter as G
from scipy.stats import mannwhitneyu, wilcoxon

import audio_features as af
import metrics as M
from imperial_subset import SUBSET, LEVELS, MOTION_FPS

KEYS = ('BAS', 'VBCS', 'ABHS', 'AAS', 'AAS_max', 'EAS', 'EAS_max')


def load_motion(motion_dir, level, choreo):
    paths = glob.glob(os.path.join(motion_dir, f'dance_level_{level}_{choreo}.npy'))
    if not paths:
        return None
    x = np.load(paths[0]).transpose(0, 2, 3, 1)                 # (N, T, 21, 3)
    n, t = x.shape[:2]
    t_src = np.arange(t) / MOTION_FPS
    t_dst = np.arange(int(t / MOTION_FPS * M.FPS)) / M.FPS
    flat = x.reshape(n, t, -1)
    out = np.empty((n, len(t_dst), flat.shape[2]))
    for i in range(n):
        for c in range(flat.shape[2]):
            out[i, :, c] = np.interp(t_dst, t_src, flat[i, :, c])
    return out.reshape(n, len(t_dst), x.shape[2], 3)


def _zrows(x):
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)


def offset_curve(envelope, joints_list):
    """Mean Pearson r between the onset envelope window starting at each offset and motion acceleration."""
    L = joints_list[0].shape[0]
    acc = _zrows(np.stack([G(M.motion_acceleration(j), M.ACCENT_SMOOTH) for j in joints_list])).mean(axis=0)
    env = G(envelope, M.ACCENT_SMOOTH)
    windows = _zrows(np.lib.stride_tricks.sliding_window_view(env, L))   # (n_offsets, L)
    return windows @ acc / L


def random_segments(signals, L, n, rng):
    """n random length-L windows (feat, rms) drawn from the given (feat, rms) tracks."""
    pool = [(f, r) for f, r in signals if len(f) >= L]
    out = []
    for _ in range(n):
        f, r = pool[rng.integers(len(pool))]
        o = rng.integers(0, len(f) - L + 1)
        out.append((f[o:o + L], r[o:o + L]))
    return out


def mean_scores(segments, joints):
    rows = [M.clip_scores(f, r, joints) for f, r in segments]
    return {k: float(np.nanmean([x[k] for x in rows])) for k in KEYS}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--motion_dir', default=os.path.join(here, 'data', 'imperial', 'motion'))
    ap.add_argument('--audio_dir', default=os.path.join(here, 'data', 'imperial', 'audio'))
    ap.add_argument('--choreos', nargs='*', default=None, help='choreographies to evaluate (default: all available)')
    ap.add_argument('--n_chance', type=int, default=8, help='random wrong-song segments per repeat')
    ap.add_argument('--trim', type=float, default=0.0,
                    help='seconds dropped from the start of every recording (capture start-up movement)')
    ap.add_argument('--length', type=float, default=None, help='keep at most this many seconds after the trim')
    ap.add_argument('--out', default=os.path.join(here, 'results', 'imperial.json'))
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    music, meta = {}, {}
    for c in SUBSET:
        wav = os.path.join(args.audio_dir, c + '.wav')
        if not os.path.exists(wav):
            continue
        y = af.load_audio(wav)
        feat, rms = af.extract_baseline(y), af.rms_envelope(y)
        T = min(len(feat), len(rms))
        music[c] = (feat[:T], rms[:T])
        with open(os.path.join(args.audio_dir, c + '.json')) as f:
            meta[c] = json.load(f)

    choreos = [c for c in (args.choreos or SUBSET) if c in music]
    motions = {}
    for c in list(choreos):
        lv = {l: load_motion(args.motion_dir, l, c) for l in LEVELS}
        if any(v is None for v in lv.values()):
            print('skip', c, '(missing motion)')
            choreos.remove(c)
            continue
        trim = int(round(args.trim * M.FPS))
        end = None if args.length is None else trim + int(round(args.length * M.FPS))
        motions.update({(c, l): v[:, trim:end] for l, v in lv.items()})
    assert len(music) >= 2, 'need music from at least two songs for the chance baseline'

    # Motion start inside the music window: a start verified by eye ('motion_start') uses all
    # repeats; otherwise it is searched on even repeats and scored on odd ones.
    offsets, offset_info, held_out = {}, {}, {}
    for c in choreos:
        if 'motion_start' in SUBSET[c]:
            offsets[c] = int(round((SUBSET[c]['motion_start'] + args.trim - meta[c]['window_start']) * M.FPS))
            offset_info[c] = {'start': SUBSET[c]['motion_start'], 'trim': args.trim, 'source': 'given'}
            held_out.update({(c, l): motions[(c, l)] for l in LEVELS})
        else:
            search = [j for l in LEVELS for j in motions[(c, l)][0::2]]
            curve = offset_curve(music[c][0][:, af.ENVELOPE], search)
            offsets[c] = int(np.argmax(curve))
            offset_info[c] = {'start': meta[c]['window_start'] + offsets[c] / M.FPS, 'source': 'searched',
                              'peak_z': float((curve.max() - curve.mean()) / curve.std())}
            held_out.update({(c, l): motions[(c, l)][1::2] for l in LEVELS})
        print(f"{c}: motion start {offset_info[c]['start']:.2f}s into the source track ({offset_info[c]['source']}), "
              f"README range {meta[c]['readme_start']}-{meta[c]['readme_end']}s")

    results = {'offsets': offset_info, 'per_choreo': {}, 'per_level': {}}
    pooled = {l: {'matched': [], 'chance': [], 'shifted': []} for l in LEVELS}
    for c in choreos:
        f_c, r_c = music[c]
        others = [music[o] for o in music if o != c]
        results['per_choreo'][c] = {}
        for l in LEVELS:
            reps = held_out[(c, l)]
            L = reps.shape[1]
            o = offsets[c]
            own = (f_c[o:o + L], r_c[o:o + L])
            # same song, start moved 2-8 s later (or earlier if the window ends): tests the alignment itself
            shifts = [s for s in range(-8 * M.FPS, 8 * M.FPS + 1, M.FPS) if abs(s) >= 2 * M.FPS and 0 <= o + s <= len(f_c) - L]
            matched, chance, shifted = [], [], []
            for j in reps:
                matched.append(M.clip_scores(*own, j))
                chance.append(mean_scores(random_segments(others, L, args.n_chance, rng), j))
                shifted.append(mean_scores([(f_c[o + s:o + s + L], r_c[o + s:o + s + L]) for s in shifts], j))
            for name, rows in (('matched', matched), ('chance', chance), ('shifted', shifted)):
                pooled[l][name] += rows
            results['per_choreo'][c][LEVELS[l]] = {
                name: {k: float(np.nanmean([r[k] for r in rows])) for k in KEYS}
                for name, rows in (('matched', matched), ('chance', chance), ('shifted', shifted))}
            results['per_choreo'][c][LEVELS[l]]['n'] = len(reps)

    show = ('BAS', 'VBCS', 'ABHS', 'AAS', 'EAS')
    print('\nscores: aligned | same song shifted 2-8 s | other songs')
    for l, name in LEVELS.items():
        rows = pooled[l]
        results['per_level'][name] = {}
        print(f'{name} (n={len(rows["matched"])})')
        for k in show:
            a = np.array([r[k] for r in rows['matched']])
            sh = np.array([r[k] for r in rows['shifted']])
            ch = np.array([r[k] for r in rows['chance']])
            ok = np.isfinite(a) & np.isfinite(sh) & np.isfinite(ch)
            p_sh = float(wilcoxon(a[ok], sh[ok], alternative='greater').pvalue)
            p_ch = float(wilcoxon(a[ok], ch[ok], alternative='greater').pvalue)
            results['per_level'][name][k] = {'aligned': float(a[ok].mean()), 'shifted': float(sh[ok].mean()),
                                             'chance': float(ch[ok].mean()), 'ci95': M.bootstrap_ci(a[ok]),
                                             'p_aligned_gt_shifted': p_sh, 'p_aligned_gt_chance': p_ch}
            print(f'  {k:<5} {a[ok].mean():7.3f} | {sh[ok].mean():7.3f} (p={p_sh:.2g}) | {ch[ok].mean():7.3f} (p={p_ch:.2g})')

    print('\nExpert > other level, aligned scores (one-sided Mann-Whitney)')
    results['expert_vs'] = {}
    for l in ('i', 'b'):
        results['expert_vs'][LEVELS[l]] = {
            k: float(mannwhitneyu([r[k] for r in pooled['e']['matched']], [r[k] for r in pooled[l]['matched']],
                                  alternative='greater').pvalue) for k in show}
        print(f'Expert > {LEVELS[l]:<12} ' + ' | '.join(f'{k} p={p:.2g}' for k, p in results['expert_vs'][LEVELS[l]].items()))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print('saved', args.out)


if __name__ == '__main__':
    main()
