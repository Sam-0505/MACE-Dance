"""BPL beat-reference surrogates with drift control: does BPL respond to the ORGANISATION of beat timing, or only to drift?

Background (shift_surrogate.py): the IBI-shuffled and phase-randomized controls lowered BPL. Neither has a seam (only the
circular-shift control rolls the motion; these two replace the beat list and the motion is untouched), but both can let
the beat positions DRIFT away from the true beats: real librosa intervals alternate around the tempo (e.g. 15,16,15,16 frames)
so their rounding errors cancel, and shuffling them turns that into a random walk. This script separates the two effects.

Notation: own beats b_k, line fit b_k ~ a + I k (I = own mean interval, BPL's `interval`), deviations d_k = b_k - (a + I k).
Every surrogate keeps the motion, the band-pass (centred on 1/I, width 15 %), the Hilbert phase and the 4 s trim unchanged
(lock_models.lock_r definition); only the beat reference changes. Nothing is rescaled unless stated.
  metro        a + I k                               regular grid, own tempo and mean phase (d = 0)
  ibi_global   b_0 + cumsum(permuted intervals)       interval multiset, first and last beat, mean interval EXACT; order
                                                      randomised -> beat positions random-walk (drift unbounded)
  ibi_global_r same, then rescaled to I (line fit)   = the committed shift_surrogate.py 'ibi' control, for reference
  ibi_block4   intervals permuted within blocks of 4 every 4th beat stays where it was -> local re-ordering with drift
                                                      bounded to one block; interval multiset, duration, mean exact
  dev_shuffle  a + I k + permuted d                   deviation distribution kept, their order randomised; no
                                                      accumulated drift (positions, not intervals, are permuted)
  dev_iaaft    a + I k + IAAFT(d)                     = phase-randomized beat reference: deviation distribution AND power
                                                      spectrum (autocorrelation) kept, Fourier phases randomised; no drift
  phase_audio  the committed audio-level control: phase-randomized onset envelope -> librosa beat tracker -> rescaled to I
  phase_audio_clean  the same, only tracks without a skipped / doubled beat (every interval within 0.7-1.3 x I)
Diagnostics per condition: RMS and max |b' - b| in beats (drift), share of non-increasing beat lists (none expected), and for
the tracker the share of tracks with skipped / doubled beats. Statistics as shift_surrogate.aggregate (unit = song / piece,
Delta = Original - mean control, bootstrap CI, one-sided sign-flip p, Monte-Carlo null of the dataset mean).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/bpl_beat_surrogates.py --dataset finedance     # or motorica
"""
import argparse
import json
import os
import sys
import time

import librosa
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

import audio_features as af  # noqa: E402
from lock_models import interval, beat_phase, analytic_at, FPS  # noqa: E402
from multicomponent import SEED, T  # noqa: E402
from shift_surrogate import load, aggregate, lock_from_analytic, iaaft, phase_randomize  # noqa: E402

K, K_TRACK, TRIM = 200, 100, 4 * FPS
CONDS = ['metro', 'ibi_global', 'ibi_global_r', 'ibi_block4', 'dev_shuffle', 'dev_iaaft', 'phase_audio', 'phase_audio_clean']


def bpl(A, beats, iv, rescale=False):
    beats = np.asarray(beats, float)
    if len(beats) < 3 or np.any(np.diff(beats) <= 0):   # a non-increasing surrogate is invalid (counted in the diagnostics)
        return np.nan
    return float(np.abs(lock_from_analytic(A, beat_phase(beats, iv, 1.0, rescale, T), TRIM)[0]))


def block_shuffle(ibis, rng, size=4):
    out = ibis.copy()
    for s in range(0, len(ibis), size):
        out[s:s + size] = rng.permutation(out[s:s + size])
    return out


def score_slice(D, i, rng):
    b = np.asarray(D['beats'][i], float)
    iv = interval(b)
    k = np.arange(len(b))
    a = np.polyfit(k, b, 1)[1]
    line = a + iv * k
    dev = b - line
    ibis = np.diff(b)
    A = analytic_at(D['q'][i], 1.0 / iv)
    # NB with rescale=False beat_phase uses the list's own line-fit interval for extrapolation outside the beats only;
    # the band-pass stays at 1 / iv (the own beats) in every condition.
    surr = {c: [] for c in CONDS}
    surr['metro'] = [line]
    for _ in range(K):
        surr['ibi_global'].append(b[0] + np.concatenate([[0], np.cumsum(rng.permutation(ibis))]))
        surr['ibi_block4'].append(b[0] + np.concatenate([[0], np.cumsum(block_shuffle(ibis, rng))]))
        surr['dev_shuffle'].append(line + rng.permutation(dev))
        surr['dev_iaaft'].append(line + iaaft(dev, rng))
    surr['ibi_global_r'] = surr['ibi_global']
    # audio-level tracker control (as in shift_surrogate.py)
    kk, o = D['key'][i], D['off'][i]
    env = librosa.onset.onset_strength(y=D['audio'][kk][o * af.HOP_LENGTH:(o + T + 1) * af.HOP_LENGTH], sr=af.SR)[:T]
    n_bad = 0
    for _ in range(K_TRACK):
        _, bi = librosa.beat.beat_track(onset_envelope=phase_randomize(env, rng), sr=af.SR, hop_length=af.HOP_LENGTH,
                                        start_bpm=1800 / iv, tightness=100)
        bi = np.asarray(bi, float)
        surr['phase_audio'].append(bi)
        r = np.diff(bi) / iv if len(bi) >= 2 else np.array([0.0])
        if len(bi) >= 3 and np.all((r > 0.7) & (r < 1.3)):
            surr['phase_audio_clean'].append(bi)
        else:
            n_bad += 1
    out = {'orig': bpl(A, b, iv), 'null': {}, 'diag': {}}
    for c in CONDS:
        resc = c in ('ibi_global_r', 'phase_audio', 'phase_audio_clean')
        out['null'][c] = np.array([bpl(A, s, iv, resc) for s in surr[c]], float)
        if c.startswith(('ibi', 'dev', 'metro')):
            drift = np.array([np.sqrt(np.mean((np.sort(s) - b) ** 2)) / iv for s in surr[c]])
            dmax = np.array([np.max(np.abs(np.sort(s) - b)) / iv for s in surr[c]])
            out['diag'][c] = {'rms_drift_beats': float(drift.mean()), 'max_drift_beats': float(dmax.mean()),
                              'non_increasing': float(np.mean([np.any(np.diff(s) <= 0) for s in surr[c]]))}
    out['diag']['phase_audio'] = {'share_skip_or_double': n_bad / K_TRACK}
    out['diag']['own'] = {'rms_dev_frames': float(np.sqrt(np.mean(dev ** 2))),
                          'lag1_autocorr_ibi': float(np.corrcoef(ibis[:-1], ibis[1:])[0, 1]) if len(ibis) > 2 else np.nan}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('finedance', 'motorica'), required=True)
    a = ap.parse_args()
    t0 = time.time()
    D = load(a.dataset)
    S = len(D['beats'])
    rng = np.random.default_rng(SEED)
    per = []
    for i in range(S):
        per.append(score_slice(D, i, rng))
        if i % 20 == 0:
            print(f'slice {i}/{S}  {time.time() - t0:.0f}s', flush=True)
    P = [{'bpl': {'orig': p['orig'], 'orig_full': p['orig'], 'null': p['null']}} for p in per]
    res = {}
    for c in CONDS:
        r = aggregate(D, P, 'bpl', c, np.random.default_rng(SEED))
        r.pop('_mc', None)
        if c in per[0]['diag']:
            r['diag'] = {k: float(np.nanmean([p['diag'][c][k] for p in per])) for k in per[0]['diag'][c]}
        res[c] = r
        extra = f" p_mc {r['p_mc']:.4f}" if 'p_mc' in r else ''
        print(f"{a.dataset:9s} {c:18s} orig {r['orig']:.3f} ctrl {r['null']:.3f} d {r['delta']:+.3f} "
              f"[{r['delta_ci'][0]:+.3f},{r['delta_ci'][1]:+.3f}] p {r['p_perm']:.4f}{extra} {r.get('diag', '')}", flush=True)
    own = {k: float(np.nanmean([p['diag']['own'][k] for p in per])) for k in per[0]['diag']['own']}
    print('own beats:', own)
    out = {'settings': {'K': K, 'K_track': K_TRACK, 'trim_s': 4, 'block': 4, 'n_perm': 10000, 'n_mc': 2000, 'rng_seed': SEED,
                        'n_slices': S, 'n_units': len(D['order'])}, 'own_beats': own, 'results': res}
    fn = os.path.join(HERE, 'results', f'bpl_beat_surrogates_{a.dataset}.json')
    with open(fn, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', fn, '%.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
