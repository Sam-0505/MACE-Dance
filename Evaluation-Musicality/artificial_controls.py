"""Experiment 2 - metric specificity and limitations: can simple, non-expressive motion score high BPL (Lock) or IMC (Loud)?

Real slices of FineDance test (19 songs) and Motorica (52 pieces); every score uses the validity-table definitions exactly
(shift_surrogate.py): BPL = |mean exp(i(phase_QoM - theta_beats))| with QoM band-passed at the own beat rate, trim 4 s, beats
rescaled to the own mean tempo; IMC = Pearson r of sigma-15 smoothed QoM and RMS, zero lag. Both metrics only read QoM (mean
joint speed), so each synthetic motion is a whole-body translation d(t) (T, 3) of a fixed pose: QoM = |d(t+1) - d(t)| for any
pose. The trajectories are saved so they can be put on a real pose and rendered later.

Controls (per real slice; the slice's own beats b, mean beat interval iv, beat phase theta(t), audio RMS):
  A  beat oscillators (fixed amplitude; scored on own music)
     A1 sway   x = sin(theta(t) / 2): one side-to-side swing per beat, speed peaks once per beat, follows the tracked beats
     A2 metro  the same sway on a perfectly regular grid at the own mean tempo and mean beat phase (no tempo fluctuation)
     A3 bounce y = (1 - cos theta(t)) / 2: up-down once per beat; speed peaks TWICE per beat (energy at 2x the beat rate; any
               beat-rate energy comes only from the beat-to-beat tempo fluctuation of the tracked beats)
     A4 bounce on the regular grid of A2 (no beat-rate energy at all, up to numerical noise)
  B  RMS-following motion: the body moves around a circle with angular step proportional to the RMS -> speed proportional to RMS
     B1 raw frame-level RMS     B2 sigma-15 smoothed RMS
     scored on own music and on the 3 tempo-matched mismatched slices of the validity table (lock_models pairing, seed 0)
  C  temporal shuffling of the REAL QoM (equivalent to shuffling velocity segments; the head before the first cut beat and the tail
     after the last cut beat stay in place, the k-beat middle segments are permuted, 200 permutations per slice and setting)
     C1 beat-aligned  cuts at every k-th beat (k = 1, 2, 4, 8): segments keep their phase relative to the beat
     C2 random cuts   the same number of cuts at uniformly random frames inside the same middle region: phase scrambled
  D  same-tempo mismatched music: A1 and A2 scored on own beats vs the 3 tempo-matched mismatched beat lists (rescaled to the own
     tempo, i.e. exactly the same tempo), beside the real-dance own-vs-mismatch gap.
Diagnostic 'band' (r_band, band_share.py): Var(band-passed QoM) / Var(QoM) over the BPL window, band |f - f_beat| <= 0.075 f_beat,
to show how much of the motion BPL actually looks at (BPL reads only the phase in that band, not its amplitude).
Reference rows recomputed here: Real dance on own music and on the mismatched music (reproduces the validity table).

Statistics, unit = song / piece (slice values averaged within a unit): unit mean with bootstrap CI; paired differences vs a
reference (Real dance own, or own-vs-mismatch for B and D) with bootstrap CI and two-sided sign-flip p (10,000).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/artificial_controls.py --dataset finedance
    .venv/bin/python ../Evaluation-Musicality/artificial_controls.py --dataset motorica
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

import shift_surrogate as SS  # noqa: E402
from lock_models import interval, beat_phase, analytic_at, FPS  # noqa: E402
from multicomponent import T, smooth, boot_ci, signflip_p  # noqa: E402

TRIM = 4 * FPS
K_SHUF = 200
KS = (1, 2, 4, 8)
SEED = 0


# ------------------------------------------------------------------------------------------------ motion builders
def qom(d):
    """QoM of a whole-body translation d (n, 3): per-frame displacement norm, edge-padded like robustness_loud.motion_signals."""
    sp = np.linalg.norm(np.diff(d, axis=0), axis=1)
    return np.pad(sp, (0, 1), mode='edge')


def traj_sway(theta, amp=0.15):
    d = np.zeros((len(theta), 3))
    d[:, 0] = amp * np.sin(theta / 2)
    return d


def traj_bounce(theta, amp=0.10):
    d = np.zeros((len(theta), 3))
    d[:, 1] = amp * (1 - np.cos(theta)) / 2
    return d


def traj_rms(rms, radius=0.3, max_step=0.15):
    """circle in the horizontal plane, angular step proportional to the RMS (max step 0.15 rad, so the chord length
    2 r sin(step / 2) is proportional to the step within 0.1 %)."""
    r = np.maximum(np.asarray(rms, float), 0)
    step = max_step * r / max(r.max(), 1e-12)
    phi = np.concatenate([[0], np.cumsum(step[:-1])])
    return np.stack([radius * np.cos(phi), np.zeros_like(phi), radius * np.sin(phi)], 1)


def metro_theta(b, iv, n):
    a0 = np.polyfit(np.arange(len(b)), np.asarray(b, float), 1)[1]
    return 2 * np.pi * (np.arange(n) - a0) / iv


# ------------------------------------------------------------------------------------------------ scorers
def bpl(q, iv, beats):
    if len(beats) < 3:
        return np.nan
    A = analytic_at(q, 1.0 / iv)
    return float(abs(SS.lock_from_analytic(A, beat_phase(beats, iv, 1.0, True, len(q)), TRIM)[0]))


def band_share(q, iv):
    """r_band = Var(Re A) / Var(q) over the BPL window (trim 4 s), A = the BPL band-passed analytic QoM (band_share.py)."""
    n = len(q)
    A = analytic_at(q, 1.0 / iv)[TRIM:n - TRIM]
    return float(np.real(A).var() / max(q[TRIM:n - TRIM].var(), 1e-30))


def zs(x):
    x = np.asarray(x, float)
    return (x - x.mean(-1, keepdims=True)) / np.maximum(x.std(-1, keepdims=True), 1e-12)


def imc(q, rms):
    return float((zs(smooth(q, SS.SIGMA)) * zs(smooth(rms, SS.SIGMA))).mean())


# ------------------------------------------------------------------------------------------------ shuffles
def shuffle_segments(q, cuts, rng):
    """cuts: sorted interior boundaries c0 < ... < cm; q[:c0] and q[cm:] stay, the m middle segments are permuted."""
    segs = [q[cuts[j]:cuts[j + 1]] for j in range(len(cuts) - 1)]
    perm = rng.permutation(len(segs))
    return np.concatenate([q[:cuts[0]]] + [segs[p] for p in perm] + [q[cuts[-1]:]])


def shuffle_scores(q, iv, b, rms, k, rng):
    """returns (BPL, IMC) means over K_SHUF permutations for C1 (beat-aligned) and C2 (random cuts), or NaNs if < 3 segments."""
    bi = np.round(np.asarray(b, float)).astype(int)
    bi = bi[(bi > 0) & (bi < len(q))]
    cuts = bi[::k]
    if len(cuts) < 4:
        return [np.nan] * 4
    lo, hi, m = cuts[0], cuts[-1], len(cuts)
    out = {'c1': [], 'c2': []}
    for _ in range(K_SHUF):
        qa = shuffle_segments(q, cuts, rng)
        inner = np.sort(rng.choice(np.arange(lo + 1, hi), m - 2, replace=False))
        qb = shuffle_segments(q, np.concatenate([[lo], inner, [hi]]), rng)
        out['c1'].append((bpl(qa, iv, b), imc(qa, rms)))
        out['c2'].append((bpl(qb, iv, b), imc(qb, rms)))
    c1, c2 = np.mean(out['c1'], 0), np.mean(out['c2'], 0)
    return [c1[0], c1[1], c2[0], c2[1]]


# ------------------------------------------------------------------------------------------------ per slice
def score_slice(D, i, rng):
    q, b = D['q'][i], D['beats'][i]
    n = len(q)
    iv = interval(b)
    th = beat_phase(b, iv, 1.0, True, n)
    rms = D['rms_own'][i]
    pair = list(D['pair'][i])
    wrong_b = [D['beats'][j] for j in pair]
    wrong_r = [D['rms_own'][j] for j in pair]
    s, traj = {}, {}

    def both(name, qq, own_only=False):
        s[f'{name}|BPL|own'] = bpl(qq, iv, b)
        s[f'{name}|IMC|own'] = imc(qq, rms)
        s[f'{name}|band|own'] = band_share(qq, iv)
        if not own_only:
            s[f'{name}|BPL|mis'] = float(np.nanmean([bpl(qq, iv, w) for w in wrong_b]))
            s[f'{name}|IMC|mis'] = float(np.mean([imc(qq, r) for r in wrong_r]))

    both('Real', q)
    thm = metro_theta(b, iv, n)
    traj['A1'] = traj_sway(th)
    traj['A2'] = traj_sway(thm)
    traj['A3'] = traj_bounce(th)
    traj['A4'] = traj_bounce(thm)
    traj['B1'] = traj_rms(rms)
    traj['B2'] = traj_rms(smooth(rms, SS.SIGMA))
    for name, d in traj.items():
        both(name, qom(d))
    for k in KS:
        c1b, c1i, c2b, c2i = shuffle_scores(q, iv, b, rms, k, rng)
        s[f'C1_k{k}|BPL|own'], s[f'C1_k{k}|IMC|own'] = c1b, c1i
        s[f'C2_k{k}|BPL|own'], s[f'C2_k{k}|IMC|own'] = c2b, c2i
    return s, traj


# ------------------------------------------------------------------------------------------------ statistics
def unit_means(vals, units, order):
    vals = np.asarray(vals, float)
    return np.array([np.nanmean(vals[units == u]) if np.any(~np.isnan(vals[units == u])) else np.nan for u in order])


def summarize(per, units, order, rng):
    keys = list(per[0].keys())
    U = {k: unit_means([p[k] for p in per], units, order) for k in keys}
    res = {}
    for k, v in U.items():
        ok = ~np.isnan(v)
        res[k] = {'mean': float(v[ok].mean()), 'ci': boot_ci(v[ok], rng), 'n_units': int(ok.sum())}

    def diff(a, b_):
        d = U[a] - U[b_]
        d = d[~np.isnan(d)]
        return {'delta': float(d.mean()), 'ci': boot_ci(d, rng), 'p': signflip_p(d, rng), 'n_pos': int((d > 0).sum()),
                'n_units': int(len(d))}
    comps = {}
    for k in keys:
        name, met, cond = k.split('|')
        if met == 'band':
            continue
        if name != 'Real' and cond == 'own':
            comps[f'{k} - Real|{met}|own'] = diff(k, f'Real|{met}|own')
        if cond == 'own' and f'{name}|{met}|mis' in U:
            comps[f'{k} - {name}|{met}|mis'] = diff(k, f'{name}|{met}|mis')
    for met in ('BPL', 'IMC'):        # does the own-vs-mismatch gap of a control differ from the real-dance gap?
        gr = U[f'Real|{met}|own'] - U[f'Real|{met}|mis']
        for name in ('A1', 'A2', 'A3', 'A4', 'B1', 'B2'):
            g = U[f'{name}|{met}|own'] - U[f'{name}|{met}|mis']
            d = (g - gr)[~np.isnan(g - gr)]
            comps[f'gap {name}|{met} - gap Real|{met}'] = {'delta': float(d.mean()), 'ci': boot_ci(d, rng),
                                                            'p': signflip_p(d, rng), 'n_units': int(len(d))}
    return res, comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('finedance', 'motorica'), required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    D = SS.load(a.dataset)
    S = a.limit or len(D['beats'])
    units = np.asarray(D['units'][:S]).astype(str)
    order = [u for u in D['order'] if u in set(units)]
    print(f'{a.dataset}: {S} slices, {len(order)} units; loaded in {time.time() - t0:.0f}s', flush=True)
    rng = np.random.default_rng(SEED)
    per, trajs = [], {}
    for i in range(S):
        s, tr = score_slice(D, i, rng)
        per.append(s)
        for k, d in tr.items():
            trajs.setdefault(k, []).append(d.astype(np.float32))
        if i % 50 == 0:
            print(f'  slice {i}/{S}  {time.time() - t0:.0f}s', flush=True)
    res, comps = summarize(per, units, order, rng)
    for k, r in res.items():
        print(f"{k:22s} {r['mean']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
    for k, r in comps.items():
        print(f"{k:42s} {r['delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] p {r['p']:.4f}")
    out = {'settings': {'dataset': a.dataset, 'n_slices': S, 'n_units': len(order), 'trim_s': TRIM / FPS, 'imc_sigma': SS.SIGMA,
                        'k_shuffles': K_SHUF, 'ks_beats': list(KS), 'seed': SEED, 'unit': 'song' if a.dataset == 'finedance' else 'piece'},
           'results': res, 'comparisons': comps}
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', f'artificial_controls_{a.dataset}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(HERE, 'results', f'artificial_controls_traj_{a.dataset}.npz'),
                        units=units, **{k: np.stack(v) for k, v in trajs.items()})
    print('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
