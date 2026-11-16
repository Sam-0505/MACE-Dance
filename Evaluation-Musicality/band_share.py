"""Beat-band share diagnostic for BPL (Lock): is a model's BPL carried by a substantial beat-rate movement component, or by a
nearly invisible one? BPL keeps only the phase of the band-passed QoM, so a weak but phase-stable beat-band component can score high
(Experiment 2: a bounce with 0.04 % of its speed variance in the band reaches BPL 0.77 / 0.69).

Per clip (QoM = mean joint speed, own beats, mean beat interval iv, band = |f - 1/iv| <= 0.075/iv, the BPL band; all over the BPL
window, trim 4 s at both ends):
  BPL      |mean exp(i(angle A - theta))|                    the metric as used (A = analytic band-passed QoM, theta = beat phase)
  r_band   Var(Re A) / Var(q)                                 share of the speed variance inside the beat band (the diagnostic)
  -- supplementary variants, NOT replacements (no separate validation) --
  BPL_aw   |mean |A| exp(i(angle A - theta))| / mean |A|      amplitude-weighted BPL: weights frames by band amplitude; still
                                                              scale-free, so it does NOT penalise a weak band
  r_lock   |mean A exp(-i theta)|^2 / (2 Var(q))              share of the speed variance in the beat-LOCKED sinusoid (= 1 for a pure
                                                              phase-locked sinusoid, -> 0 for a weak or unlocked band); about BPL_aw^2 * r_band
Models are the main-table rows: FineDance seed 1 (first sample per slice; 84 slices, 19 songs), Motorica 3 samples per slice
(289 slices, 52 pieces). Clip value = mean over its samples. Unit = song / piece. Reported: clip-level percentiles, unit-level
mean with bootstrap CI, paired unit-level differences (two-sided sign-flip p, 10,000), and the share of clips whose r_band is below
the 5th / 25th percentile of real dance.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/band_share.py --dataset finedance
    .venv/bin/python ../Evaluation-Musicality/band_share.py --dataset motorica
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

import lock_models as L  # noqa: E402
from lock_models import interval, beat_phase, analytic_at, FPS  # noqa: E402
from multicomponent import boot_ci, signflip_p  # noqa: E402
from robustness_loud import motion_signals  # noqa: E402

TRIM = 4 * FPS
SEED = 0
MODELS = {'finedance': (1, ['Lodge', 'MEGADance', 'MACE released', 'FlowerDance', 'Ours critic v5 + gain']),
          'motorica': (3, ['Lodge', 'MEGADance', 'MACE released', 'Ours critic v5 + gain'])}
STATS = ('BPL', 'r_band', 'BPL_aw', 'r_lock')


def clip_stats(q, beats):
    """returns dict of the four per-clip statistics (NaN if < 3 beats)."""
    if len(beats) < 3:
        return {k: np.nan for k in STATS}
    n = len(q)
    iv = interval(beats)
    A = analytic_at(q, 1.0 / iv)[TRIM:n - TRIM]
    th = beat_phase(beats, iv, 1.0, True, n)[TRIM:n - TRIM]
    qw = q[TRIM:n - TRIM]
    vq = max(qw.var(), 1e-30)
    z = A * np.exp(-1j * th)
    return {'BPL': float(abs(np.exp(1j * np.angle(z)).mean())),
            'r_band': float(np.real(A).var() / vq),
            'BPL_aw': float(abs(z.mean()) / max(np.abs(A).mean(), 1e-30)),
            'r_lock': float(abs(z.mean()) ** 2 / (2 * vq))}


def unit_level(v, units, order):
    return np.array([np.nanmean(v[units == u]) for u in order])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('finedance', 'motorica'), required=True)
    a = ap.parse_args()
    t0 = time.time()
    ns, labels = MODELS[a.dataset]
    units, order, beats, sets, _, _ = (L.load_finedance if a.dataset == 'finedance' else L.load_motorica)(ns, labels)
    units = np.asarray(units).astype(str)
    N = len(beats)
    print(f'{a.dataset}: {N} slices, {len(order)} units, loaded {time.time() - t0:.0f}s', flush=True)
    clip = {}
    for label, mot in sets.items():
        v = np.full((N, len(STATS)), np.nan)
        for i in range(N):
            per = [clip_stats(motion_signals(m)['speed'], beats[i]) for m in mot[i]]
            v[i] = [np.mean([p[k] for p in per]) for k in STATS]
        clip[label] = v
    rng = np.random.default_rng(SEED)
    real_band = clip['Real dance'][:, 1]
    p5, p25 = np.nanpercentile(real_band, [5, 25])
    res = {}
    for label, v in clip.items():
        r = {}
        for k, s in enumerate(STATS):
            x = v[:, k]
            u = unit_level(x, units, order)
            r[s] = {'unit_mean': float(np.nanmean(u)), 'unit_ci': boot_ci(u[~np.isnan(u)], rng),
                    'clip_pct': dict(zip(['p5', 'p25', 'p50', 'p75', 'p95'],
                                         np.nanpercentile(x, [5, 25, 50, 75, 95]).tolist()))}
        r['frac_below_real_p5'] = float(np.nanmean(v[:, 1] < p5))
        r['frac_below_real_p25'] = float(np.nanmean(v[:, 1] < p25))
        ok = ~np.isnan(v[:, 0]) & ~np.isnan(v[:, 1])
        from scipy.stats import spearmanr
        r['spearman_BPL_rband_clips'] = float(spearmanr(v[ok, 0], v[ok, 1])[0])
        res[label] = r
    comps = {}
    ours, base = 'Ours critic v5 + gain', 'MACE released'
    for A_, B_ in [(ours, 'Real dance'), (ours, base), (base, 'Real dance')] + [(m, 'Real dance') for m in labels if m not in (ours, base)]:
        c = {}
        for k, s in enumerate(STATS):
            d = unit_level(clip[A_][:, k], units, order) - unit_level(clip[B_][:, k], units, order)
            d = d[~np.isnan(d)]
            c[s] = {'delta': float(d.mean()), 'ci': boot_ci(d, rng), 'p': signflip_p(d, rng), 'n_pos': int((d > 0).sum()),
                    'n_units': int(len(d))}
        comps[f'{A_} - {B_}'] = c
    for label, r in res.items():
        print(f"{label:24s} " + '  '.join(f"{s} {r[s]['unit_mean']:.4f} (med {r[s]['clip_pct']['p50']:.4f})" for s in STATS)
              + f"  <real p5 {r['frac_below_real_p5']:.2f}  <real p25 {r['frac_below_real_p25']:.2f}  rho {r['spearman_BPL_rband_clips']:+.2f}")
    for k, c in comps.items():
        print(f"{k:40s} " + '  '.join(f"{s} {c[s]['delta']:+.4f} p {c[s]['p']:.4f}" for s in STATS))
    out = {'settings': {'dataset': a.dataset, 'n_slices': N, 'n_units': len(order), 'samples_per_slice': ns, 'trim_s': TRIM / FPS,
                        'bandwidth': L.BW, 'real_rband_p5': float(p5), 'real_rband_p25': float(p25), 'seed': SEED},
           'results': res, 'comparisons': comps,
           'clips': {k: {s: v[:, j].tolist() for j, s in enumerate(STATS)} for k, v in clip.items()},
           'units': units.tolist()}
    with open(os.path.join(HERE, 'results', f'band_share_{a.dataset}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
