"""Switching test (2026-11-04): do real dancers change WHICH musical component their movement follows within a song?

Real dance only (FineDance release, 203 songs; the 7 with music/motion lengths differing by > 1 s are skipped).
Music components, 30 fps, all Gaussian-smoothed with sigma 15 frames (the slow scale of the coupling metric):
  loud   RMS                         perc  RMS of the HPSS percussive part
  harm   RMS of the HPSS harmonic part (melody/chord proxy; no vocal separation available)
  treble spectral flux 6.4-12.8 kHz  bass  spectral flux 50-200 Hz
Motion: whole-body energy (mean squared joint displacement), same smoothing.

Each song is cut into non-overlapping 10 s windows. In every window the components and the energy are z-scored and
a ridge regression energy ~ components gives that window's music->motion profile beta_w. Transfer quality
q(v -> w) = r(X_w beta_v, y_w): how well the profile of window v predicts the movement in window w.
  * No switching (one fixed relationship per song): q does not depend on the distance |v - w|.
  * Switching with persistence (e.g. percussion for 30 s, then melody): nearby profiles transfer better than distant
    ones, so q falls with distance.
Collinearity among the components and slow changes in the music can make q fall with distance even with a fixed
relationship, so the null is a surrogate: per song, energy* = X beta_song + phase-randomised residual (fixed
relationship, same music, same residual spectrum), run through the same pipeline (N_SUR surrogates per song).
Statistic: decay = mean q at distance 1-2 windows minus mean q at distance >= 5 windows, per song, averaged over
songs. p = share of surrogate dataset averages >= the real one. Other-song transfer is reported as a floor.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/switching_test.py
    .venv/bin/python ../Evaluation-Musicality/switching_test.py --dataset motorica   # 62 Motorica takes (2026-11-04)

Motorica: motorica_common.pool() takes, real motion = 21 BVH body joints (energy enters only via correlations);
cache data/switching/signals_motorica.npz, output results/switching_test_motorica.json.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import librosa
import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import audio_features as af  # noqa: E402
from multicomponent import FLUX_SR, FLUX_HOP, FLUX_NFFT, HIGH_BAND, LOW_BAND, band_flux  # noqa: E402
from finedance_common import RELEASE  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

COMPS = ('loud', 'perc', 'harm', 'treble', 'bass')
FPS, SIGMA, WIN, RIDGE, N_SUR = 30, 15, 300, 1.0, 50
NEAR, FAR = (1, 2), 5
CACHE = os.path.join(HERE, 'data', 'switching', 'signals.npz')
OUT = os.path.join(HERE, 'results', 'switching_test.json')
DATASET = 'finedance'


def configure(dataset):
    global CACHE, OUT, DATASET
    DATASET = dataset
    if dataset == 'motorica':
        CACHE = os.path.join(HERE, 'data', 'switching', 'signals_motorica.npz')
        OUT = os.path.join(HERE, 'results', 'switching_test_motorica.json')


def wav_of(song):
    if DATASET == 'motorica':
        from motorica_common import wav_path
        return wav_path(song)
    return os.path.join(RELEASE, 'music_wav', song + '.wav')


def music_signals(song, dataset='finedance'):
    configure(dataset)
    path = wav_of(song)
    y = af.load_audio(path)
    harm, perc = librosa.effects.hpss(y)
    y48, _ = librosa.load(path, sr=FLUX_SR)
    S = np.abs(librosa.stft(y48, n_fft=FLUX_NFFT, hop_length=FLUX_HOP))
    f = librosa.fft_frequencies(sr=FLUX_SR, n_fft=FLUX_NFFT)
    return song, {'loud': af.rms_envelope(y), 'perc': librosa.feature.rms(y=perc, hop_length=af.HOP_LENGTH)[0],
                  'harm': librosa.feature.rms(y=harm, hop_length=af.HOP_LENGTH)[0],
                  'treble': band_flux(S, f, HIGH_BAND), 'bass': band_flux(S, f, LOW_BAND)}


def songs():
    if DATASET == 'motorica':
        from motorica_common import pool
        return pool()
    out = []
    for p in sorted(glob.glob(os.path.join(RELEASE, 'motion', '*.npy'))):
        s = os.path.basename(p)[:-4]
        w = os.path.join(RELEASE, 'music_wav', s + '.wav')
        if os.path.exists(w) and abs(sf.info(w).duration - np.load(p, mmap_mode='r').shape[0] / FPS) <= 1:
            out.append(s)
    return out


def build():
    if os.path.exists(CACHE):
        return dict(np.load(CACHE, allow_pickle=True))['data'].item()
    from finedance_common import release_joints
    from model.metric import SMPLSkeleton
    from model import musicality as MU
    names = songs()
    from functools import partial
    with Pool(8) as p:
        music = dict(p.map(partial(music_signals, dataset=DATASET), names))
    smpl = SMPLSkeleton('cuda')
    data = {}
    for s in names:
        if DATASET == 'motorica':
            from motorica_common import body_joints
            j = torch.as_tensor(body_joints(s), dtype=torch.float32)
        else:
            j = torch.as_tensor(np.asarray(release_joints(s, smpl, 'cuda')), dtype=torch.float32)
        e = MU.motion_energy(j[None])[0].numpy()
        T = min(len(e), *(len(v) for v in music[s].values()))
        data[s] = {'energy': e[:T], **{c: np.asarray(music[s][c][:T], dtype=np.float32) for c in COMPS}}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, data=np.array(data, dtype=object))
    return data


def zs(a):
    a = a - a.mean(0)
    return a / (np.linalg.norm(a, axis=0) + 1e-12)


def windowed(X, y):
    """X (T, K), y (T,) smoothed -> per-window z-scored (Xw (n, WIN, K), yw (n, WIN)) and ridge profiles (n, K)."""
    n = len(y) // WIN
    Xw = np.stack([zs(X[k * WIN:(k + 1) * WIN]) for k in range(n)])
    yw = np.stack([zs(y[k * WIN:(k + 1) * WIN]) for k in range(n)])
    K = X.shape[1]
    B = np.stack([np.linalg.solve(x.T @ x + RIDGE * np.eye(K), x.T @ v) for x, v in zip(Xw, yw)])
    return Xw, yw, B


def transfer(Xw, yw, B):
    """Q[v, w] = r(Xw[w] @ B[v], yw[w])."""
    P = np.einsum('wtk,vk->vwt', Xw, B)
    P = P - P.mean(-1, keepdims=True)
    P = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-12)
    return np.einsum('vwt,wt->vw', P, yw)


def decay_curve(Q):
    n = len(Q)
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None])
    curve = {k: float(Q[d == k].mean()) for k in range(1, n)}
    near = Q[(d >= NEAR[0]) & (d <= NEAR[1])].mean()
    far = Q[d >= FAR].mean() if (d >= FAR).any() else np.nan
    return curve, near - far


def phase_randomise(x, rng):
    F = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(F))
    ph[0] = 0
    if len(x) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=len(x))


def main():
    data = build()
    rng = np.random.default_rng(0)
    keep = [s for s in data if len(data[s]['energy']) // WIN >= FAR + 2]
    print(len(data), 'songs,', len(keep), 'with >=', (FAR + 2) * WIN // FPS, 's', flush=True)
    real_decay, sur_decay, curves, other, self_q, winners = [], [], [], [], [], []
    prof = {}
    for s in keep:
        X = np.stack([gaussian_filter1d(data[s][c].astype(np.float64), SIGMA) for c in COMPS], 1)
        y = gaussian_filter1d(data[s]['energy'].astype(np.float64), SIGMA)
        Xw, yw, B = windowed(X, y)
        prof[s] = (Xw, yw, B)
        Q = transfer(Xw, yw, B)
        c, dec = decay_curve(Q)
        curves.append(c)
        real_decay.append(dec)
        self_q.append(float(np.diag(Q).mean()))
        r1 = np.einsum('wtk,wt->wk', Xw, yw)  # per-window single-component r
        winners.append(r1.argmax(1))
        # surrogate: one fixed relationship for the whole song
        Xz, yz = zs(X), zs(y)
        b = np.linalg.solve(Xz.T @ Xz + RIDGE * np.eye(len(COMPS)), Xz.T @ yz)
        fit = Xz @ b
        res = yz - fit
        sd = []
        for _ in range(N_SUR):
            ys = fit + phase_randomise(res, rng)
            sd.append(decay_curve(transfer(*windowed(X, ys)))[1])
        sur_decay.append(sd)
    real_decay, sur_decay = np.array(real_decay), np.array(sur_decay)  # (S,), (S, N_SUR)
    # other-song floor: profile of a random window of another song applied to every window of this song
    for s in keep:
        Xw, yw, _ = prof[s]
        o = [t for t in keep if t != s]
        Bo = np.stack([prof[t][2][rng.integers(len(prof[t][2]))] for t in rng.choice(o, 20, replace=False)])
        other.append(float(transfer(Xw, yw, Bo).mean()))
    agg_real = float(np.nanmean(real_decay))
    agg_sur = np.nanmean(sur_decay, 0)
    p = float((agg_sur >= agg_real).mean())
    maxd = max(len(c) for c in curves)
    curve = {k: float(np.mean([c[k] for c in curves if k in c])) for k in range(1, min(maxd, 13) + 1)}
    n_songs = {k: int(sum(k in c for c in curves)) for k in curve}
    # winner changes between consecutive windows, real vs the surrogate expectation is not computed (descriptive)
    switch_rate = float(np.mean([np.mean(w[1:] != w[:-1]) for w in winners]))
    share = {c: float(np.mean(np.concatenate(winners) == i)) for i, c in enumerate(COMPS)}
    res = {'n_songs': len(keep), 'window_s': WIN / FPS, 'components': COMPS, 'ridge': RIDGE,
           'decay_real': agg_real, 'decay_surrogate_mean': float(agg_sur.mean()),
           'decay_surrogate_95': float(np.percentile(agg_sur, 95)), 'p': p,
           'songs_decay_above_own_surrogate_median': float(np.mean(real_decay > np.median(sur_decay, 1))),
           'transfer_by_distance': curve, 'n_songs_by_distance': n_songs,
           'q_self_fit': float(np.mean(self_q)), 'q_other_song': float(np.mean(other)),
           'q_same_song_far': float(np.nanmean([np.mean([c[k] for k in c if k >= FAR]) for c in curves])),
           'winner_share': share, 'winner_change_rate': switch_rate}
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('n_songs_by_distance',)}, indent=1))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('finedance', 'motorica'), default='finedance')
    configure(ap.parse_args().dataset)
    main()
