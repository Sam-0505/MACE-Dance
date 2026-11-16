"""Does BPL + IMC carry information beyond BAS? Original-vs-mismatched music ranking on REAL dance (FineDance test, Motorica).

Task: every real-dance slice is scored against 4 candidate music slices: its own music and the 3 tempo-matched mismatched
slices of the validity table (lock_models pairing, seed 0). The candidates are ranked by a score; the original should be first.
This is "original music vs tempo-controlled mismatches", NOT unrestricted retrieval: mismatches are tempo-matched songs and,
for BPL and BAS, their beat lists are rescaled to the own mean beat interval (the validity-table protocol, unchanged).

Per-candidate scores (definitions identical to shift_surrogate.py, i.e. the validity table):
  BPL  |mean exp(i(phase_QoM - theta_beats))|, QoM band-passed at the own beat rate, trim 4 s, beats rescaled to own tempo
  IMC  Pearson r of smoothed (sigma 15) QoM and the candidate slice's RMS envelope, zero lag
  BAS  authors' Beat Alignment Score (velocity minima vs beats, sigma^2 = 9), beats rescaled to own tempo
Single metrics rank by the raw score (higher = better, fixed a priori, no fitting). Combinations: conditional-logit ranking model
(softmax over the 4 candidates of a query, no intercept, L2 penalty), features z-scored with training-fold candidates only;
penalty chosen by inner song-grouped 5-fold CV (held-out log-likelihood) on the training songs only. Outer CV: leave-one-song-out
(FineDance song / Motorica music piece), so no song is in both training and test.
Metrics: top-1 accuracy and MRR (ties broken uniformly at random, in expectation); 95 % CIs by song-cluster bootstrap (10,000);
paired differences with cluster-bootstrap CI and a song-level sign-flip p (two-sided, 10,000).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/mismatch_ranking.py --dataset finedance     # scores (cached) + ranking
    .venv/bin/python ../Evaluation-Musicality/mismatch_ranking.py --dataset motorica
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

METRICS = ('BAS', 'BPL', 'IMC')
MODELS = [('BAS',), ('BPL',), ('IMC',), ('BPL', 'IMC'), ('BAS', 'BPL', 'IMC')]
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
N_BOOT, N_PERM, SEED = 10000, 10000, 0


# ------------------------------------------------------------------------------------------------ scores
def compute_scores(dataset):
    import shift_surrogate as SS
    from lock_models import interval, beat_phase, analytic_at, FPS
    from multicomponent import T, smooth
    from tmd_beat_metrics import kin_beats, bas
    D = SS.load(dataset)
    S = len(D['beats'])
    trim = 4 * FPS
    sc = np.full((S, 4, 3), np.nan)
    for i in range(S):
        q, b = D['q'][i], D['beats'][i]
        iv = interval(b)
        A = analytic_at(q, 1.0 / iv)
        qs = smooth(q, SS.SIGMA)
        zq = (qs - qs.mean()) / max(qs.std(), 1e-12)
        v = D['vel'][i]
        K = kin_beats(v)
        for c, j in enumerate([i] + list(D['pair'][i])):
            bj = D['beats'][j]
            if len(bj) >= 3:
                sc[i, c, 1] = abs(SS.lock_from_analytic(A, beat_phase(bj, iv, 1.0, True, T), trim)[0])
                bb = np.asarray(bj, float)
                bb = bb[0] + (bb - bb[0]) * (iv / interval(bb))     # rescaled to own tempo (identity for the own beats)
                bb = bb[(bb >= 0) & (bb < len(v))]
                if len(bb) >= 3:
                    sc[i, c, 0] = bas(K, bb)
            r = smooth(D['rms_own'][j], SS.SIGMA)
            sc[i, c, 2] = float(((r - r.mean()) / max(r.std(), 1e-12) * zq).mean())
    return sc, np.asarray(D['units']).astype(str), [list(map(int, p)) for p in D['pair']]


# ------------------------------------------------------------------------------------------------ ranking
def rank_stats(s):
    """s (N, 4) scores, candidate 0 = original. NaN = worst. Returns expected top-1 and reciprocal rank under random tie-breaks."""
    s = np.where(np.isnan(s), -np.inf, s)
    o = s[:, :1]
    gt = (s[:, 1:] > o).sum(1)
    eq = (s[:, 1:] == o).sum(1)
    top1 = np.where(gt == 0, 1.0 / (eq + 1), 0.0)
    rr = np.array([np.mean(1.0 / (g + 1 + np.arange(e + 1))) for g, e in zip(gt, eq)])
    return top1, rr


def nll(w, X, lam):
    """conditional logit: X (N, 4, d), candidate 0 is the positive. Mean NLL + lam/2 ||w||^2, and its gradient."""
    u = X @ w
    u = u - u.max(1, keepdims=True)
    p = np.exp(u)
    p /= p.sum(1, keepdims=True)
    f = -np.log(p[:, 0] + 1e-300).mean() + 0.5 * lam * w @ w
    g = -(X[:, 0] - (p[:, :, None] * X).sum(1)).mean(0) + lam * w
    return f, g


def fit(X, lam):
    return minimize(nll, np.zeros(X.shape[2]), args=(X, lam), jac=True, method='L-BFGS-B').x


def standardize(Xtr, Xte):
    flat = Xtr.reshape(-1, Xtr.shape[2])
    mu, sd = np.nanmean(flat, 0), np.nanstd(flat, 0)
    sd = np.where(sd > 0, sd, 1)
    f = lambda X: np.nan_to_num((X - mu) / sd)   # noqa: E731  (NaN -> training mean)
    return f(Xtr), f(Xte)


def choose_lambda(X, g, rng):
    songs = np.unique(g)
    folds = np.array_split(rng.permutation(songs), min(5, len(songs)))
    loss = []
    for lam in LAMBDAS:
        tot = 0.0
        for fs in folds:
            te = np.isin(g, fs)
            Xtr, Xte = standardize(X[~te], X[te])
            tot += nll(fit(Xtr, lam), Xte, 0.0)[0] * te.sum()
        loss.append(tot / len(g))
    return LAMBDAS[int(np.argmin(loss))]


def loso_scores(X, g, rng):
    """leave-one-song-out held-out candidate scores (N, 4); also the fitted weights and penalties per fold."""
    out = np.full(X.shape[:2], np.nan)
    W, lams = [], []
    for s in np.unique(g):
        te = g == s
        lam = choose_lambda(X[~te], g[~te], rng)
        Xtr, Xte = standardize(X[~te], X[te])
        w = fit(Xtr, lam)
        out[te] = Xte @ w
        W.append(w)
        lams.append(lam)
    return out, np.array(W), lams


def cluster_boot(vals, g, rng, n=N_BOOT):
    """vals (N,) or (N, k); clip-weighted mean; songs resampled with replacement."""
    songs = np.unique(g)
    idx = [np.flatnonzero(g == s) for s in songs]
    sums = np.array([vals[i].sum(0) for i in idx])
    cnt = np.array([len(i) for i in idx])
    pick = rng.integers(len(songs), size=(n, len(songs)))
    bs = sums[pick].sum(1) / cnt[pick].sum(1)[..., None] if sums.ndim > 1 else sums[pick].sum(1) / cnt[pick].sum(1)
    return np.percentile(bs, [2.5, 97.5], axis=0)


def cluster_signflip(d, g, rng, n=N_PERM):
    songs = np.unique(g)
    sums = np.array([d[g == s].sum() for s in songs])
    obs = abs(sums.sum())
    sg = rng.choice([-1, 1], size=(n, len(songs)))
    return float((np.sum(np.abs(sg @ sums) >= obs - 1e-12) + 1) / (n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('finedance', 'motorica'), required=True)
    ap.add_argument('--recompute', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    cache = os.path.join(HERE, 'results', f'mismatch_ranking_scores_{a.dataset}.npz')
    if a.recompute or not os.path.exists(cache):
        sc, g, pair = compute_scores(a.dataset)
        np.savez_compressed(cache, scores=sc, units=g, pair=np.array(pair), metrics=np.array(METRICS))
    z = np.load(cache)
    sc, g = z['scores'], z['units']
    N = len(sc)
    print(f'{a.dataset}: {N} slices, {len(np.unique(g))} songs; scores in {time.time() - t0:.0f}s', flush=True)
    nan = {m: int(np.isnan(sc[:, :, k]).any(1).sum()) for k, m in enumerate(METRICS)}
    print('slices with a NaN candidate score:', nan)
    # sanity vs the validity table: song-weighted own and mismatch-mean scores
    sanity = {}
    for k, m in enumerate(METRICS):
        own = np.mean([np.nanmean(sc[g == s, 0, k]) for s in np.unique(g)])
        mis = np.mean([np.nanmean(sc[g == s, 1:, k]) for s in np.unique(g)])
        sanity[m] = {'own_song_mean': float(own), 'mismatch_song_mean': float(mis)}
        print(f'  {m}: own {own:.4f}  mismatch {mis:.4f}')

    rng = np.random.default_rng(SEED)
    res, held, per = {}, {}, {}
    for model in MODELS:
        name = '+'.join(model)
        cols = [METRICS.index(m) for m in model]
        if len(model) == 1:
            s, W, lams = sc[:, :, cols[0]], None, None
        else:
            s, W, lams = loso_scores(sc[:, :, cols], g, rng)
        held[name] = s
        t1, rr = rank_stats(s)
        per[name] = (t1, rr)
        ci = cluster_boot(np.stack([t1, rr], 1), g, rng)
        res[name] = {'top1': float(t1.mean()), 'top1_ci': ci[:, 0].tolist(), 'mrr': float(rr.mean()), 'mrr_ci': ci[:, 1].tolist()}
        if W is not None:
            res[name]['weights_mean'] = dict(zip(model, W.mean(0).tolist()))
            res[name]['weights_min'] = dict(zip(model, W.min(0).tolist()))
            res[name]['weights_max'] = dict(zip(model, W.max(0).tolist()))
            res[name]['lambda_counts'] = {str(l): lams.count(l) for l in sorted(set(lams))}
        print(f"{name:12s} top1 {t1.mean():.3f} [{ci[0, 0]:.3f}, {ci[1, 0]:.3f}]  MRR {rr.mean():.3f} [{ci[0, 1]:.3f}, {ci[1, 1]:.3f}]"
              + (f"  w {np.round(W.mean(0), 2).tolist()}" if W is not None else ''), flush=True)

    singles = ['BAS', 'BPL', 'IMC']
    best = max(singles, key=lambda m: res[m]['top1'])
    comps = [('BAS+BPL+IMC', 'BAS'), ('BPL+IMC', 'BAS'), ('BAS+BPL+IMC', 'BPL+IMC'), ('BPL+IMC', 'IMC'), ('BPL+IMC', 'BPL'),
             ('BAS+BPL+IMC', 'IMC'), ('BPL', 'BAS'), ('IMC', 'BAS')]
    if best not in ('BAS', 'IMC', 'BPL'):
        raise RuntimeError
    diffs = {}
    for A, B in comps:
        d = {}
        for k, stat in enumerate(('top1', 'mrr')):
            x = per[A][k] - per[B][k]
            ci = cluster_boot(x, g, rng)
            d[stat] = {'delta': float(x.mean()), 'ci': ci.tolist(), 'p': cluster_signflip(x, g, rng)}
        diffs[f'{A} - {B}'] = d
        print(f"{A} - {B}: top1 {d['top1']['delta']:+.3f} [{d['top1']['ci'][0]:+.3f}, {d['top1']['ci'][1]:+.3f}] p {d['top1']['p']:.4f}"
              f" | MRR {d['mrr']['delta']:+.3f} [{d['mrr']['ci'][0]:+.3f}, {d['mrr']['ci'][1]:+.3f}] p {d['mrr']['p']:.4f}")
    out = {'settings': {'n_slices': N, 'n_songs': int(len(np.unique(g))), 'candidates': 4, 'chance_top1': 0.25,
                        'chance_mrr': float(np.mean(1 / np.arange(1, 5))), 'outer_cv': 'leave-one-song-out',
                        'inner_cv': 'song-grouped 5-fold, held-out NLL', 'lambdas': LAMBDAS, 'n_boot': N_BOOT, 'n_perm': N_PERM,
                        'best_single_top1': best, 'nan_slices': nan},
           'sanity': sanity, 'results': res, 'differences': diffs}
    with open(os.path.join(HERE, 'results', f'mismatch_ranking_{a.dataset}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
