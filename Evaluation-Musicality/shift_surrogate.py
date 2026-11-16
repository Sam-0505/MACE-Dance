"""Temporal-shift and surrogate controls for BPL (Lock) and IMC (Loud) on REAL dance: FineDance test and Motorica.

Question: does the own-music score exceed what the SAME motion gets from music with a destroyed time correspondence?
Per slice the Original is the validity-table value (BPL = lock_models.lock_r rescaled null, IMC = loud_models Pearson, sigma 15,
zero lag, trim 4 s for BPL). Controls, each with a null distribution per slice:
  circ     motion circularly shifted against the music by every integer delta in [4 s, T-4 s] (781 values). IMC: the RMS
           envelope is rolled against QoM (distribution and autocorrelation preserved). BPL: the QoM is rolled against the
           beats (the FFT band-pass is circular, so this equals rolling the beat phase); mean beat frequency unchanged.
  rand     the SAME song's music from a random other window of the full song (>= 5 s away, 200 draws): RMS window / beat
           track window, beats rescaled to the own mean tempo. Reference = the own window read from the same full-song track.
  phase    IMC: IAAFT surrogate of the RMS (amplitude distribution and power spectrum kept, phases randomised; 200).
           BPL: phase-randomised onset-strength envelope of the own audio -> librosa beat tracker (start_bpm = own tempo,
           tightness 100) -> beats rescaled to the own mean tempo (100 surrogates).
  ibi      (BPL, extra) the own inter-beat intervals in random order (same mean tempo and interval distribution; 200).
  metro    (BPL, extra) a perfectly regular beat grid at the own mean tempo (one value: BPL is offset-free).
  mismatch the existing control: mean over the 3 tempo-matched wrong slices (reproduces the validity table).
Statistics, unit = song (FineDance 19) / piece (Motorica 52): Delta = Original - mean(null) per unit; bootstrap CI; one-sided
sign-flip permutation p (10,000). Dataset-level Monte-Carlo p: the observed mean over units against 2,000 draws of the same
statistic with one random null draw per slice (p = (1 + #null >= obs) / 2001). Also the share of slices whose Original is above
the 95th percentile of their own null (5 % expected by chance) and the mean percentile rank of the Original (0.5 = chance).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/shift_surrogate.py --dataset finedance
    .venv/bin/python ../Evaluation-Musicality/shift_surrogate.py --dataset motorica
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
import lock_models as L  # noqa: E402
from lock_models import interval, beat_phase, analytic_at, FPS  # noqa: E402
from multicomponent import SEED, T, GT_DIR, MACE_DIR, smooth, boot_ci, signflip_p  # noqa: E402
from robustness_loud import motion_signals, music_signals  # noqa: E402
from tmd_beat_metrics import velocity, accents, kin_beats, bas  # noqa: E402
from tmd_beat_phase import phases, plv  # noqa: E402

SIGMA = 15
MIN_SHIFT = 4 * FPS          # circular shifts and random-window distance: 4 s / 5 s
MIN_DIST = 5 * FPS
K_RAND, K_PHASE_IMC, K_PHASE_BPL, K_IBI, N_MC = 200, 200, 100, 200, 2000
COND_BPL = ['circ', 'rand', 'phase', 'ibi', 'metro', 'mismatch']
COND_BPLF = COND_BPL + ['metro_rand']
COND_IMC = ['circ', 'rand', 'phase', 'mismatch']


# ------------------------------------------------------------------------------------------------ helpers
def lock_from_analytic(A, theta, trim):
    """A (K, n) or (n,) analytic signals, theta (K, n) or (n,) beat phases -> COMPLEX mean exp(i(angle A - theta)) without
    trim. BPL (offset-free) = its modulus; BPL-fixed (offset-sensitive) = Re(z exp(-i c)) with one dataset-wide offset c."""
    A = np.atleast_2d(A)
    n = A.shape[1]
    d = np.angle(A) - np.atleast_2d(theta)
    return np.exp(1j * d)[:, trim:n - trim].mean(1)


def window_beats(fb, o, n=T):
    b = fb[(fb >= o) & (fb < o + n)] - o
    return b


def full_beat_track(y, start_bpm):
    env = librosa.onset.onset_strength(y=y, sr=af.SR)
    _, idx = librosa.beat.beat_track(onset_envelope=env, sr=af.SR, hop_length=af.HOP_LENGTH, start_bpm=start_bpm, tightness=100)
    return np.asarray(idx)


def phase_randomize(x, rng):
    X = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0
    if len(x) % 2 == 0:
        ph[-1] = 0
    return np.maximum(np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x)) + x.mean(), 0)


def iaaft(x, rng, iters=30):
    """Schreiber-Schmitz surrogate: same amplitude distribution, (approximately) the same power spectrum, random phases."""
    srt, amp = np.sort(x), np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    for _ in range(iters):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n=len(x))
        y = srt[np.argsort(np.argsort(y))]
    return y


# ------------------------------------------------------------------------------------------------ data
def load(dataset):
    """Returns dict with per-slice lists: unit, q (raw QoM), beats, iv, rms_own (raw), rms_wrong (3 raw), beats_wrong,
    pair, y_win (audio window), full (per-slice dict: rms full, beat track full, offset)."""
    if dataset == 'finedance':
        units, order, beats, sets, pair, bpm = L.load_finedance(1, [])
        names = json.load(open(os.path.join(MACE_DIR, 'names.json')))['names']
        songs = [n.split('_')[0] for n in names]
        off = [(FPS * 17) * int(n.split('_slice')[1]) for n in names]
        audio = {s: af.load_audio(os.path.join(GT_DIR, 'wavs', s + '.wav')) for s in order}
        key = songs
        rms_own = [music_signals(n)['rms'] for n in names]            # the table's sliced-wav RMS
    else:
        from motorica_common import SLICE_LEN, wav_path
        units, order, beats, sets, pair, bpm = L.load_motorica(1, [])
        meta = json.load(open(os.path.join(os.path.dirname(L.__file__), 'data', 'motorica', 'mace', 'slices.json')))
        sl = [tuple(s) for s in meta['slices'] if s[1] % SLICE_LEN == 0]
        assert len(sl) == len(beats)
        key = [t for t, _ in sl]
        off = [o for _, o in sl]
        audio = {t: af.load_audio(wav_path(t)) for t in sorted(set(key))}
        rms_own = [None] * len(sl)
    rms_full = {k: af.rms_envelope(y) for k, y in audio.items()}
    if dataset == 'motorica':
        rms_own = [rms_full[k][o:o + T] for k, o in zip(key, off)]
    # full-song beat track, start tempo = median own slice interval of that song/take
    fb = {}
    for k in audio:
        iv = np.median([interval(beats[i]) for i in range(len(beats)) if key[i] == k and len(beats[i]) >= 3])
        fb[k] = full_beat_track(audio[k], 1800 / iv)
    q = [motion_signals(sets['Real dance'][i][0])['speed'] for i in range(len(beats))]
    vel = [velocity(np.asarray(sets['Real dance'][i][0], dtype=np.float64)[:T], '3d') for i in range(len(beats))]
    # sanity: full-audio RMS window == the table's sliced RMS
    chk = [np.corrcoef(rms_full[key[i]][off[i]:off[i] + T], rms_own[i])[0, 1] for i in range(0, len(beats), 7)]
    print('full-audio window vs table RMS: min r %.4f' % min(chk), flush=True)
    return dict(units=np.array(units), order=order, beats=beats, pair=pair, key=key, off=off, audio=audio,
                rms_full=rms_full, fb=fb, q=q, vel=vel, rms_own=[np.asarray(r[:T], float) for r in rms_own])


# ------------------------------------------------------------------------------------------------ per-slice scores
def score_slice(D, i, rng):
    n = T
    trim = 4 * FPS
    q, b = D['q'][i], D['beats'][i]
    iv = interval(b)
    fc = 1.0 / iv
    A = analytic_at(q, fc)
    th_own = beat_phase(b, iv, 1.0, True, n)
    orig = complex(lock_from_analytic(A, th_own, trim)[0])
    out = {'bpl': {'orig': orig}, 'imc': {}}

    # ---- BPL controls
    shifts = np.arange(MIN_SHIFT, n - MIN_SHIFT + 1)
    idx = (np.arange(n)[None, :] - shifts[:, None]) % n
    bpl = {'circ': lock_from_analytic(A[idx], th_own, trim)}
    k, o = D['key'][i], D['off'][i]
    fb, L_full = D['fb'][k], len(D['rms_full'][k])
    wb = window_beats(fb, o)
    out['bpl']['orig_full'] = (complex(lock_from_analytic(A, beat_phase(wb, iv, 1.0, True, n), trim)[0])
                               if len(wb) >= 3 else np.nan)
    cand = [x for x in range(0, L_full - n + 1) if abs(x - o) >= MIN_DIST]
    r_rand, B = [], {'rand': [], 'phase': [], 'ibi': [], 'metro_rand': []}
    for x in rng.choice(cand, K_RAND, replace=len(cand) < K_RAND):
        w = window_beats(fb, int(x))
        B['rand'].append(w)
        r_rand.append(lock_from_analytic(A, beat_phase(w, iv, 1.0, True, n), trim)[0] if len(w) >= 3 else np.nan + 0j)
    bpl['rand'] = np.array(r_rand, dtype=complex)
    env = librosa.onset.onset_strength(y=D['audio'][k][o * af.HOP_LENGTH:(o + n + 1) * af.HOP_LENGTH], sr=af.SR)[:n]
    r_ph = []
    for _ in range(K_PHASE_BPL):
        e2 = phase_randomize(env, rng)
        _, bi = librosa.beat.beat_track(onset_envelope=e2, sr=af.SR, hop_length=af.HOP_LENGTH, start_bpm=1800 / iv, tightness=100)
        B['phase'].append(np.asarray(bi))
        r_ph.append(lock_from_analytic(A, beat_phase(np.asarray(bi), iv, 1.0, True, n), trim)[0] if len(bi) >= 3 else np.nan + 0j)
    bpl['phase'] = np.array(r_ph, dtype=complex)
    ibis = np.diff(b.astype(float))
    r_ibi = []
    for _ in range(K_IBI):
        b2 = b[0] + np.concatenate([[0], np.cumsum(rng.permutation(ibis))])
        B['ibi'].append(b2)
        r_ibi.append(lock_from_analytic(A, beat_phase(b2, iv, 1.0, True, n), trim)[0])
    bpl['ibi'] = np.array(r_ibi, dtype=complex)
    a0 = np.polyfit(np.arange(len(b)), b.astype(float), 1)[1]   # own mean phase: intercept of the beat-vs-index line fit
    kk = np.arange(int(np.floor(-a0 / iv)) - 1, int(np.ceil((n - a0) / iv)) + 2)
    bpl['metro'] = lock_from_analytic(A, beat_phase(a0 + iv * kk, iv, 1.0, True, n), trim)
    phs = rng.uniform(0, iv, 200)
    B['metro_rand'] = [a0 + ph + iv * kk for ph in phs]
    bpl['metro_rand'] = np.concatenate([lock_from_analytic(A, beat_phase(bb, iv, 1.0, True, n), trim) for bb in B['metro_rand']])
    bpl['mismatch'] = np.array([lock_from_analytic(A, beat_phase(D['beats'][j], iv, 1.0, True, n), trim)[0]
                                for j in D['pair'][i]])
    bpl = {c: np.asarray(v, dtype=complex) for c, v in bpl.items()}
    out['bpl']['null'] = bpl

    # ---- IMC controls
    qs = smooth(q, SIGMA)
    rms = D['rms_own'][i]
    rs = smooth(rms, SIGMA)
    z = lambda X: (X - X.mean(-1, keepdims=True)) / np.maximum(X.std(-1, keepdims=True), 1e-12)  # noqa: E731
    corr = lambda X: (z(X) * z(qs)).mean(-1)  # noqa: E731
    out['imc']['orig'] = float(corr(rs))
    imc = {'circ': corr(rs[idx])}
    rf = D['rms_full'][k]
    imc['rand'] = np.array([corr(smooth(rf[int(x):int(x) + n], SIGMA)) for x in rng.choice(cand, K_RAND, replace=len(cand) < K_RAND)])
    imc['phase'] = np.array([corr(smooth(iaaft(rms, rng), SIGMA)) for _ in range(K_PHASE_IMC)])
    imc['mismatch'] = np.array([corr(smooth(D['rms_own'][j], SIGMA)) for j in D['pair'][i]])
    out['imc']['null'] = imc

    # ---- event metrics (beat_suite.py definitions, no trim): Event PLV = |mean exp(2 pi i phase)| of the velocity maxima
    # inside their beat interval (offset-free); BAS = authors' Beat Alignment Score, velocity minima vs beats (offset-sensitive).
    # Same control beat lists as BPL, every list rescaled to the own mean beat interval; circular shift = event times rolled.
    v = D['vel'][i]
    nv = len(v)
    M, K = accents(v), kin_beats(v)

    def prep(bb, resc=True):
        bb = np.asarray(bb, float)
        if len(bb) < 3:
            return None
        if resc:
            bb = bb[0] + (bb - bb[0]) * (iv / interval(bb))
        bb = bb[(bb >= 0) & (bb < nv)]
        return bb if len(bb) >= 3 else None

    def ev(bb, m=M, kb=K):
        if bb is None:
            return (np.nan, np.nan)
        return (plv(phases(m, bb)), bas(kb, bb))
    bo = prep(b, False)
    e = {'orig': ev(bo), 'orig_full': ev(prep(wb))}
    en = {'circ': [ev(bo, np.sort((M + d) % nv), np.sort((K + d) % nv)) for d in shifts],
          'rand': [ev(prep(w)) for w in B['rand']], 'phase': [ev(prep(w)) for w in B['phase']],
          'ibi': [ev(prep(w, False)) for w in B['ibi']], 'metro': [ev(prep(a0 + iv * kk, False))],
          'metro_rand': [ev(prep(w, False)) for w in B['metro_rand']],
          'mismatch': [ev(prep(D['beats'][j])) for j in D['pair'][i]]}
    for m_i, name in enumerate(('eplv', 'bas')):
        out[name] = {'orig': float(e['orig'][m_i]), 'orig_full': float(e['orig_full'][m_i]),
                     'null': {c: np.array([x[m_i] for x in v_], float) for c, v_ in en.items()}}
    return out


# ------------------------------------------------------------------------------------------------ statistics
def aggregate(D, per, metric, cond, rng):
    units, order = D['units'], D['order']
    S = len(per)
    ref = 'orig_full' if (metric in ('bpl', 'eplv', 'bas') and cond == 'rand') else 'orig'
    orig = np.array([p[metric][ref] for p in per])
    nulls = [np.asarray(p[metric]['null'][cond], float) for p in per]
    nulls = [x[~np.isnan(x)] for x in nulls]
    ok = np.array([len(x) > 0 and not np.isnan(o) for x, o in zip(nulls, orig)])
    nm = np.array([x.mean() if len(x) else np.nan for x in nulls])
    pct = np.array([(np.sum(x >= o) + 1) / (len(x) + 1) if len(x) and not np.isnan(o) else np.nan for x, o in zip(nulls, orig)])
    q95 = np.array([np.percentile(x, 95) if len(x) else np.nan for x in nulls])
    W = np.zeros((S, len(order)))
    for i in range(S):
        if ok[i]:
            W[i, order.index(units[i])] = 1
    keep = W.sum(0) > 0
    W = W[:, keep] / W[:, keep].sum(0)
    uo = np.nan_to_num(orig) @ W
    un = np.nan_to_num(nm) @ W
    d = uo - un
    res = {'orig': float(uo.mean()), 'null': float(un.mean()), 'delta': float(d.mean()), 'delta_ci': boot_ci(d, rng),
           'p_perm': signflip_p(d, rng, 'greater'), 'n_units': int(len(d)), 'n_pos': int((d > 0).sum()),
           'pct_slices_above_q95': float(np.mean(orig[ok] > q95[ok])), 'mean_percentile': float(1 - np.nanmean(pct[ok])),
           'p_slice_lt_0.05': float(np.mean(pct[ok] < 0.05)), 'n_slices': int(ok.sum())}
    if cond not in ('metro', 'mismatch'):
        K = max(len(x) for x in nulls)
        pad = np.full((S, K), np.nan)
        for i, x in enumerate(nulls):
            pad[i, :len(x)] = x
        ks = np.array([max(len(x), 1) for x in nulls])
        idx = np.floor(rng.random((N_MC, S)) * ks).astype(int)
        draws = np.nan_to_num(pad[np.arange(S)[None, :], idx])
        mc = (draws @ W).mean(1)
        res.update({'mc_null_mean': float(mc.mean()), 'mc_null_q95': float(np.percentile(mc, 95)),
                    'p_mc': float((np.sum(mc >= uo.mean()) + 1) / (N_MC + 1))})
        res['_mc'] = mc
    return res


def realize(per, D, mode):
    """Complex BPL values -> real scores. mode 'abs': |z| (offset-free BPL). mode 'fixed': Re(z exp(-i c_u)), c_u = circular
    mean phase of the own-beat z of real dance over all units EXCEPT the slice's own unit (leave-one-unit-out; controls never
    enter the estimate). Returns (new per list with metric 'bpl' real, offsets dict, resultant length across units)."""
    units = np.asarray(D['units'][:len(per)])
    if mode == 'within':
        return realize_within(per, D)
    order = sorted(set(units))
    uz = {u: np.mean([per[i]['bpl']['orig'] for i in range(len(per)) if units[i] == u]) for u in order}
    c = {u: float(np.angle(sum(uz[v] / abs(uz[v]) for v in order if v != u))) for u in order}
    f = (lambda z, u: np.abs(z)) if mode == 'abs' else (lambda z, u: np.real(np.asarray(z) * np.exp(-1j * c[u])))
    new = []
    for i, p in enumerate(per):
        u = units[i]
        b = {'orig': float(f(p['bpl']['orig'], u)), 'orig_full': float(f(p['bpl']['orig_full'], u)),
             'null': {k: np.asarray(f(v, u), float) for k, v in p['bpl']['null'].items()}}
        new.append({'bpl': b, 'imc': p['imc']})
    R = float(abs(np.mean([uz[u] / abs(uz[u]) for u in order])))
    return new, c, R


def realize_within(per, D):
    """BPL-within: Re(z exp(-i c_i)), c_i = circular mean phase of the own-beat z of the OTHER non-overlapping slices of the same
    take/song (so the offset only has to be stable within one dance, not across genres). Slices without such a partner are NaN."""
    S = len(per)
    z0 = [per[i]['bpl']['orig'] for i in range(S)]
    new, n_used = [], 0
    for i in range(S):
        part = [z0[j] / abs(z0[j]) for j in range(S) if j != i and D['key'][j] == D['key'][i]
                and abs(D['off'][j] - D['off'][i]) >= T]
        p = per[i]['bpl']
        if not part:
            nan = np.nan
            new.append({'bpl': {'orig': nan, 'orig_full': nan, 'null': {k: np.full(len(v), nan) for k, v in p['null'].items()}},
                        'imc': per[i]['imc']})
            continue
        n_used += 1
        c = float(np.angle(sum(part)))
        f = lambda z: np.real(np.asarray(z) * np.exp(-1j * c))  # noqa: E731
        new.append({'bpl': {'orig': float(f(p['orig'])), 'orig_full': float(f(p['orig_full'])),
                            'null': {k: np.asarray(f(v), float) for k, v in p['null'].items()}}, 'imc': per[i]['imc']})
    print(f'BPL-within: {n_used}/{S} slices have a non-overlapping partner slice of the same take', flush=True)
    return new, {}, float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('finedance', 'motorica'), required=True)
    ap.add_argument('--limit', type=int, default=0, help='debug: first N slices only')
    a = ap.parse_args()
    t0 = time.time()
    D = load(a.dataset)
    S = len(D['beats']) if not a.limit else a.limit
    print(a.dataset, S, 'slices,', len(D['order']), 'units; loaded in %.0fs' % (time.time() - t0), flush=True)
    rng = np.random.default_rng(SEED)
    per = []
    for i in range(S):
        per.append(score_slice(D, i, rng))
        if i % 10 == 0:
            print(f'slice {i}/{S}  {time.time() - t0:.0f}s', flush=True)
    if a.limit:
        D['units'] = D['units'][:S]
        D['order'] = [u for u in D['order'] if u in set(D['units'])]
    per_abs, _, _ = realize(per, D, 'abs')
    per_fix, offs, R = realize(per, D, 'fixed')
    per_win, _, _ = realize(per, D, 'within')
    print(f'fixed offset c per unit (leave-one-unit-out): mean {np.angle(np.mean(np.exp(1j * np.array(list(offs.values()))))):+.2f} rad; '
          f'resultant length of the unit phases R = {R:.2f}', flush=True)
    res, mcs = {}, {}
    for metric, src, conds in (('bpl', per_abs, COND_BPL), ('bplf', per_fix, COND_BPLF), ('bplw', per_win, COND_BPLF), ('eplv', per, COND_BPL),
                                ('bas', per, COND_BPLF), ('imc', per, COND_IMC)):
        res[metric] = {}
        for c in conds:
            r = aggregate(D, src, 'bpl' if metric in ('bplf', 'bplw') else metric, c, np.random.default_rng(SEED))
            if '_mc' in r:
                mcs[f'{metric}_{c}'] = r.pop('_mc')
            res[metric][c] = r
            extra = f" p_mc {r['p_mc']:.4f}" if 'p_mc' in r else ''
            print(f"{a.dataset:9s} {metric:4s} {c:10s} orig {r['orig']:.3f} null {r['null']:.3f} d {r['delta']:+.3f} "
                  f"[{r['delta_ci'][0]:+.3f},{r['delta_ci'][1]:+.3f}] p {r['p_perm']:.4f}{extra} "
                  f">q95 {r['pct_slices_above_q95']:.2f} pct {r['mean_percentile']:.2f}", flush=True)
    tag = '_debug' if a.limit else ''
    out = {'settings': {'sigma': SIGMA, 'trim_s': 4, 'min_shift_s': 4, 'min_window_distance_s': 5, 'K': {
        'circ': 'all integer shifts 4 s..T-4 s (781)', 'rand': K_RAND, 'phase_imc': K_PHASE_IMC, 'phase_bpl': K_PHASE_BPL,
        'ibi': K_IBI, 'metro_rand': 200}, 'n_mc': N_MC, 'n_perm': 10000, 'rng_seed': SEED, 'n_slices': S,
        'n_units': len(D['order']),
        'bplf': 'Re(z exp(-i c)), z = mean exp(i(phi_QoM - theta)); c = circular mean over units of the real-dance own-beat z, '
                'leave-one-unit-out; controls never enter c', 'bplf_offsets_rad': {str(k): v for k, v in offs.items()},
        'bplf_resultant_length': R},
        'results': res,
        'per_slice': [{'key': D['key'][i], 'offset': D['off'][i], 'unit': str(D['units'][i]),
                       'bpl_orig': per_abs[i]['bpl']['orig'], 'bpl_orig_full': per_abs[i]['bpl']['orig_full'],
                       'bplf_orig': per_fix[i]['bpl']['orig'], 'imc_orig': per[i]['imc']['orig'],
                       'eplv_orig': per[i]['eplv']['orig'], 'bas_orig': per[i]['bas']['orig'],
                       'bpl_null_mean': {c: float(np.nanmean(v)) for c, v in per_abs[i]['bpl']['null'].items()},
                       'bplf_null_mean': {c: float(np.nanmean(v)) for c, v in per_fix[i]['bpl']['null'].items()},
                       'imc_null_mean': {c: float(np.nanmean(v)) for c, v in per[i]['imc']['null'].items()}} for i in range(S)]}
    with open(os.path.join(HERE, 'results', f'shift_surrogate_{a.dataset}{tag}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(HERE, 'results', f'shift_surrogate_{a.dataset}_mcnull{tag}.npz'), **mcs)
    # pooled per-slice null values for the figure (histogram of the null of ONE slice-level statistic is not needed)
    print('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
