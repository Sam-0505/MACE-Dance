"""Tempo-detuning control for BPL (Lock) on REAL dance: how precisely must the dance match the beat rate to score high?

Per slice, the real dance is scored with the BPL definition (lock_models.lock_r: QoM, FFT band-pass of width 15 % around the
beat frequency, Hilbert phase, |mean exp(i(phi - theta))| over the clip without 4 s at both ends) against
  original   the clip's own librosa beats (the validity-table value)
  grid(e)    a perfectly regular beat grid with interval = own mean interval x (1 + e), e in DETUNE (%), e = 0 is the
             metronome control of shift_surrogate.py. The detuned grid is treated as the music: the band-pass is centred on
             ITS beat rate, as BPL would do for a song e % faster/slower. BPL is offset-free, so the grid phase does not matter.
Drift over the scored window: |e| x (beats in 26 s), e.g. 1 % at 120 BPM ~ 0.5 beat. Note: beyond +-7.5 % the true beat rate
leaves the 15 % band, so large detunings also test the filter, not only the phase drift.
Statistics, unit = song (FineDance 19) / piece (Motorica 52): mean BPL per detuning, Delta = original - grid(e) per unit, bootstrap
CI, one-sided sign-flip p (10,000), and Delta0(e) = grid(0) - grid(e) (how much a wrong tempo costs relative to the right one).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/tempo_detune.py --dataset finedance     # or motorica
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

import lock_models as L  # noqa: E402
from lock_models import interval, lock_r, FPS  # noqa: E402
from multicomponent import SEED, T, boot_ci, signflip_p  # noqa: E402
from robustness_loud import motion_signals  # noqa: E402

DETUNE = [-10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10]   # percent
TRIM = 4 * FPS


def grid(iv_e, n=T):
    return np.arange(-iv_e, n + iv_e, iv_e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('finedance', 'motorica'), required=True)
    a = ap.parse_args()
    units, order, beats, sets, pair, bpm = (L.load_finedance if a.dataset == 'finedance' else L.load_motorica)(1, [])
    units = np.asarray(units)
    S = len(beats)
    orig = np.full(S, np.nan)
    det = np.full((S, len(DETUNE)), np.nan)
    drift = np.full((S, len(DETUNE)), np.nan)
    for i in range(S):
        b = beats[i]
        if len(b) < 3:
            continue
        q = motion_signals(sets['Real dance'][i][0])['speed']
        iv = interval(b)
        orig[i] = lock_r(q, iv, b, 1.0, TRIM)
        for k, e in enumerate(DETUNE):
            iv_e = iv * (1 + e / 100)
            det[i, k] = lock_r(q, iv_e, grid(iv_e), 1.0, TRIM)
            drift[i, k] = abs(e / 100) * (T - 2 * TRIM) / iv     # beats of accumulated drift over the scored window
    order = list(order)
    W = np.zeros((S, len(order)))
    for i in range(S):
        if not np.isnan(orig[i]):
            W[i, order.index(units[i])] = 1
    W = W[:, W.sum(0) > 0]
    W /= W.sum(0)
    uo = np.nan_to_num(orig) @ W                                    # slices without beats have zero weight
    ud = np.nan_to_num(det).T @ W                                  # (detune, units)
    k0 = DETUNE.index(0)
    rng = np.random.default_rng(SEED)
    rows = []
    for k, e in enumerate(DETUNE):
        d, d0 = uo - ud[k], ud[k0] - ud[k]
        rows.append({'detune_pct': e, 'bpl': float(ud[k].mean()), 'bpl_ci': boot_ci(ud[k], rng),
                     'delta_vs_original': float(d.mean()), 'delta_ci': boot_ci(d, rng), 'p_original_gt': signflip_p(d, rng, 'greater'),
                     'n_pos': int((d > 0).sum()), 'delta_vs_grid0': float(d0.mean()),
                     'p_grid0_gt': signflip_p(d0, rng, 'greater') if e != 0 else None,
                     'drift_beats_median': float(np.nanmedian(drift[:, k]))})
        print(f"{a.dataset:9s} detune {e:+5.1f}%  BPL {rows[-1]['bpl']:.3f}  original-grid {d.mean():+.3f} "
              f"p {rows[-1]['p_original_gt']:.4f}  grid0-grid {d0.mean():+.3f}  drift {rows[-1]['drift_beats_median']:.2f} beats",
              flush=True)
    out = {'settings': {'detune_pct': DETUNE, 'trim_s': 4, 'bandwidth': L.BW, 'n_perm': 10000, 'rng_seed': SEED,
                        'n_slices': S, 'n_units': int(W.shape[1])},
           'original': {'bpl': float(uo.mean()), 'ci': boot_ci(uo, np.random.default_rng(SEED))}, 'rows': rows}
    fn = os.path.join(HERE, 'results', f'tempo_detune_{a.dataset}.json')
    with open(fn, 'w') as f:
        json.dump(out, f, indent=1)
    print('original', round(out['original']['bpl'], 3), 'saved', fn)


if __name__ == '__main__':
    main()
