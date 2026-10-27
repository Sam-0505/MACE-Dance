"""Robustness sweep for the loudness <-> body-energy component (real dance vs MACE, FineDance test set).

Every variant changes ONE setting relative to the primary pipeline; all variants are reported.
Variants are fixed here in advance; none is picked by its result.

Primary pipeline (= multicomponent.py 'loud', except the matching):
  music   linear RMS (15360 Hz, hop 512, 30 fps)
  motion  mean squared joint speed, global joints
  smooth  Gaussian sigma 15 frames (0.5 s), both signals
  score   max Pearson r over lags [-9, +9] frames (300 ms)
  match   TEMPO FIRST: 3 wrong songs ranked by relative tempo distance, with half/double tempo
          allowed (min over k in {1/2, 1, 2} of |k*bpm_o - bpm_s| / bpm_s). Songs within TEMPO_TOL
          count as equally close and same-genre songs go first among them.
          BPM = 60 / mean inter-beat interval of the full song (librosa beat tracker; intervals outside
          0.7-1.3x the median dropped) instead of 1800 / median interval in frames, which is quantised.

Variants:
  match_genre_first   old multicomponent.py matching (genre first, then |BPM diff|, quantised BPM).
                      Must reproduce multicomponent.json 'loud' lag-max exactly (checked).
  music_db            RMS in dB (librosa.amplitude_to_db, top_db 80)
  music_aweighted     A-weighted power summed over frequency, in dB (floor at max - 80 dB)
  motion_mean_speed   mean joint speed (QoM-like) instead of squared speed
  motion_root_rel     squared speed of pelvis-relative joints (no floor travel)
  sigma_8, sigma_30   smoothing 0.27 s / 1 s
  lag_0, lag_15, lag_30   zero-lag r / max over +-0.5 s / +-1 s

Units, pairing and tests as in multicomponent.py: song-level means (19 songs), MACE averaged over 10 seeds,
the same wrong slices for real and MACE, two-sided tests on G and on real_own - mace_own,
one-sided H1/H2. No multiple-comparison correction across variants: this is a robustness report,
not a family of hypotheses.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/robustness_loud.py
"""
import json
import os
import pickle
import sys

import librosa
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from model.metric import SMPLSkeleton  # noqa: E402
import audio_features as af  # noqa: E402
import metrics as M  # noqa: E402
from eval_mace import fk  # noqa: E402
from multicomponent import (T, N_MATCH, SEED, GT_DIR, MACE_DIR, smooth, component_stats,  # noqa: E402
                            boot_ci, wil, signflip_p)
from finedance_common import release_joints, SLICE_LEN, SLICE_STRIDE  # noqa: E402

# real-dance joints: 'test' = MACE's shipped test files; 'release' = public FineDance release (finedance_common.py),
# which fixes single-frame root glitches in 5 test songs. Select with --gt.
GT_SOURCE = 'release' if '--gt=release' in sys.argv or sys.argv[-2:] == ['--gt', 'release'] else 'test'
OUT = os.path.join(HERE, 'results', 'robustness_loud.json' if GT_SOURCE == 'test' else 'robustness_loud_release.json')
TEMPO_TOL = 0.03
PRIMARY = dict(match='tempo_first', music='rms', motion='energy', sigma=15, lag=9)
VARIANTS = {
    'primary': {},
    'match_genre_first': {'match': 'genre_first'},
    'music_db': {'music': 'rms_db'},
    'music_aweighted': {'music': 'aweighted_db'},
    'motion_mean_speed': {'motion': 'speed'},
    'motion_root_rel': {'motion': 'energy_root_rel'},
    'sigma_8': {'sigma': 8},
    'sigma_30': {'sigma': 30},
    'lag_0': {'lag': 0},
    'lag_15': {'lag': 15},
    'lag_30': {'lag': 30},
}


# ----------------------------------------------------------------------------- signals
def loudness_signals(y):
    """y: mono audio at af.SR. Returns the three loudness curves on the 30 fps grid (full length)."""
    rms = af.rms_envelope(y)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=af.HOP_LENGTH)) ** 2
    w = librosa.A_weighting(librosa.fft_frequencies(sr=af.SR, n_fft=2048))
    aw = 10 * np.log10((S * 10 ** (w[:, None] / 10)).sum(0) + 1e-12)
    return {'rms': rms, 'rms_db': librosa.amplitude_to_db(rms, ref=1.0, top_db=80),
            'aweighted_db': np.maximum(aw, aw.max() - 80)}


def music_signals(name):
    out = loudness_signals(af.load_audio(os.path.join(GT_DIR, 'wavs_sliced', name + '.wav')))
    assert min(len(v) for v in out.values()) >= T
    return {k: np.asarray(v[:T], dtype=np.float64) for k, v in out.items()}


def motion_signals(j):
    j = j[:T].astype(np.float64)
    sp = np.linalg.norm(j[1:] - j[:-1], axis=2).mean(1)
    return {'energy': M.motion_energy(j), 'speed': np.pad(sp, (0, 1), mode='edge'),
            'energy_root_rel': M.motion_energy(j - j[:, :1])}


def song_bpm(song):
    return bpm_from_wav(os.path.join(GT_DIR, 'wavs', song + '.wav'))


def bpm_from_wav(path):
    y, sr = librosa.load(path, sr=22050)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    ibi = np.diff(beats)
    ibi = ibi[(ibi > 0.7 * np.median(ibi)) & (ibi < 1.3 * np.median(ibi))]
    return float(60 / ibi.mean())


def tempo_dist(b_other, b_self):
    return min(abs(k * b_other - b_self) for k in (0.5, 1, 2)) / b_self


def matches(order, genre, bpm, how):
    out = {}
    for s in order:
        others = [o for o in order if o != s]
        if how == 'genre_first':
            key = lambda o: (genre[o] != genre[s], abs(bpm[o] - bpm[s]))  # noqa: E731
        else:
            def key(o):
                d = tempo_dist(bpm[o], bpm[s])
                return (d > TEMPO_TOL, d <= TEMPO_TOL and genre[o] != genre[s], d)
        out[s] = sorted(others, key=key)[:N_MATCH]
    return out


# ----------------------------------------------------------------------------- main
def main():
    with open(os.path.join(MACE_DIR, 'names.json')) as f:
        meta = json.load(f)
    names = meta['names']
    mace = np.load(os.path.join(MACE_DIR, 'joints.npy')).astype(np.float32)   # (N, S, T, 24, 3)
    N, NS = mace.shape[:2]
    songs = np.array([n.split('_')[0] for n in names])
    order = sorted(set(songs))
    genre = {s: json.load(open(os.path.join(GT_DIR, 'label', s + '.json')))['style2'] for s in order}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if GT_SOURCE == 'release':
        smpl = SMPLSkeleton(device)
        full = {s: release_joints(s, smpl, device) for s in order}
        starts = [SLICE_STRIDE * int(n.split('_slice')[1]) for n in names]
        gt = np.stack([full[s][o:o + SLICE_LEN] for s, o in zip(songs, starts)])
    else:
        poses, trans = [], []
        for n in names:
            with open(os.path.join(GT_DIR, 'motions_sliced', n + '.pkl'), 'rb') as f:
                m = pickle.load(f)
            poses.append(m['q']); trans.append(m['pos'])
        gt = fk(SMPLSkeleton(device), poses, trans, device, from_pred=False)

    mus = [music_signals(n) for n in names]
    gt_m = [motion_signals(j) for j in gt]
    mace_m = [[motion_signals(mace[i, s]) for s in range(NS)] for i in range(N)]
    print('signals done')

    # tempo: old quantised BPM (reproduces multicomponent.py) and full-song BPM
    beats = [np.load(os.path.join(GT_DIR, 'baseline_feats', n + '.npy'))[:T, 34] for n in names]
    bpm_old = {s: float(60 * 30 / np.median(np.concatenate([np.diff(np.flatnonzero(beats[i]))
                                                              for i in np.flatnonzero(songs == s)])))
               for s in order}
    bpm_new = {s: song_bpm(s) for s in order}
    match = {'genre_first': matches(order, genre, bpm_old, 'genre_first'),
             'tempo_first': matches(order, genre, bpm_new, 'tempo_first')}
    pairing = {}
    for how, mt in match.items():
        rng = np.random.default_rng(SEED)
        pairing[how] = [[int(rng.choice(np.flatnonzero(songs == o))) for o in mt[songs[i]]] for i in range(N)]

    def song_level(v):
        return np.array([v[songs == s].mean() for s in order])

    results = {}
    for vname, change in VARIANTS.items():
        cfg = {**PRIMARY, **change}
        sig, lag = cfg['sigma'], cfg['lag']
        ms = [smooth(m[cfg['music']], sig) for m in mus]
        gs = [smooth(m[cfg['motion']], sig) for m in gt_m]
        xs = [[smooth(m[cfg['motion']], sig) for m in row] for row in mace_m]
        pair = pairing[cfg['match']]
        sc = lambda a, b: M.lagged_corr(a, b, lag)[1]  # noqa: E731
        ro = np.array([sc(ms[i], gs[i]) for i in range(N)])
        rx = np.array([np.mean([sc(ms[j], gs[i]) for j in pair[i]]) for i in range(N)])
        mo = np.array([np.mean([sc(ms[i], x) for x in xs[i]]) for i in range(N)])
        mx = np.array([np.mean([sc(ms[j], x) for x in xs[i] for j in pair[i]]) for i in range(N)])
        real_lags = np.array([M.lagged_corr(ms[i], gs[i], lag)[2] for i in range(N)])

        v = {k: song_level(a) for k, a in (('ro', ro), ('rx', rx), ('mo', mo), ('mx', mx))}
        rng = np.random.default_rng(SEED)
        st = component_stats(v['ro'], v['rx'], v['mo'], v['mx'], rng)
        oo = v['ro'] - v['mo']
        st.update({'own_diff': float(oo.mean()), 'own_diff_ci': boot_ci(oo, rng),
                   'own_diff_wilcoxon_p': wil(oo, 'two-sided'), 'own_diff_perm_p': signflip_p(oo, rng),
                   'n_own_real_gt_mace': int((oo > 0).sum()),
                   'real_lag_median': float(np.median(real_lags)),
                   'real_frac_at_lag_edge': float(np.mean(np.abs(real_lags) == lag)) if lag else 0.0,
                   'config': cfg})
        if vname == 'primary':
            st['per_song'] = {s: {k: float(a[n]) for k, a in v.items()} for n, s in enumerate(order)}
        results[vname] = st
        print(f"{vname:20s} dR {st['d_real']:+.3f} dM {st['d_mace']:+.3f} G {st['G']:+.3f} "
              f"[{st['G_ci'][0]:+.3f},{st['G_ci'][1]:+.3f}] p {st['G_perm_p']:.3f} | "
              f"own R-M {st['own_diff']:+.3f} p {st['own_diff_perm_p']:.3f} | edge {st['real_frac_at_lag_edge']:.2f}")

    ref = json.load(open(os.path.join(HERE, 'results', 'multicomponent.json')))['stats']['lagmax']['loud']
    got = results['match_genre_first']
    repro = {k: (got[k], ref[k]) for k in ('real_own', 'real_other', 'mace_own', 'mace_other', 'G')}
    if GT_SOURCE == 'test':
        assert all(abs(a - b) < 1e-9 for a, b in repro.values()), repro
        print('reproduces multicomponent.json loud (genre-first matching): OK')

    match_info = {how: {s: [{'song': o, 'genre': genre[o], 'same_genre': genre[o] == genre[s],
                             'tempo_dist': tempo_dist(bpm_new[o], bpm_new[s])} for o in mt[s]] for s in order}
                  for how, mt in match.items()}
    out = {'variants': results, 'bpm_full_song': bpm_new, 'bpm_quantised': bpm_old, 'genre': genre,
           'matching': match_info, 'tempo_tol': TEMPO_TOL}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
