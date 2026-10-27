"""Multi-component music-motion dynamics evaluation: real dance (GT) vs MACE Motion Expert.

Design: docs/musicality_new.docx, with the overrides agreed with the user:
  * Data = MACE test set (FineDance): 84 slices x 1020 frames (34 s, 30 fps) from 19 songs.
    Slices overlap 50 %, so the UNIT OF ANALYSIS IS THE SONG (slices and MACE seeds averaged per song).
  * GT and MACE share the SMPL 24-joint skeleton -> no skeleton mapping, no scale normalization
    (Pearson correlations are scale-invariant).
  * Smoothing fixed in advance (not tuned): slow components (loud, pulse) Gaussian sigma = 15 frames
    (= metrics.ENERGY_SMOOTH); fast components (perc, onset, lowflux, highflux) sigma = 2 frames
    (= metrics.ACCENT_SMOOTH). Raw signals are truncated to 1020 frames BEFORE smoothing.
  * Score S = lag-max Pearson r over tau in [-9, +9] frames (300 ms; positive = motion after music).
    Zero-lag r is also stored.
  * Wrong-song baseline: for each song, the 3 other songs ranked by (different genre, |BPM diff|);
    each dance slice is paired with one seeded-random slice of each matched song (the same music
    slices are used for GT and for every MACE seed, so Delta_real - Delta_MACE is paired).

Components (music <-> motion):
  loud      RMS (15360 Hz, hop 512)                     <-> whole-body energy (mean squared joint speed)   slow
  perc      RMS of HPSS percussive component             <-> upper-body accel |2nd diff| (9,12-23)          fast
  onset     baseline_feats dim 0 onset envelope (2ndary) <-> same upper-body accel                          fast
  lowflux   HWR spectral flux 50-200 Hz                  <-> lower-body speed (0,1,2,4,5,7,8,10,11)          fast
  highflux  HWR spectral flux 6.4-12.8 kHz               <-> head/hand speed (15,20,21,22,23)                fast
  pulse     APPROXIMATE rolling pulse clarity: centred 120-frame (4 s) window, max Pearson autocorrelation
            over lags 7-60 frames, on the onset envelope   <-> same on whole-body mean joint speed           slow
            (MIRtoolbox mirpulseclarity is NOT reproduced exactly.)
  Flux: |STFT| at 48 kHz, n_fft 4096, hop 1600 (30 fps), sum over band bins of max(0, |X_t| - |X_t-1|).

Stats per component (songs as units): Delta = S_own - S_other; G = mean(Delta_real - Delta_MACE).
  G test: two-sided Wilcoxon + two-sided paired sign-flip permutation (10,000) on Delta_real - Delta_MACE.
  H1 / H2: one-sided (Delta > 0) Wilcoxon + sign-flip permutation on Delta_real / Delta_MACE.
  Bootstrap (songs, 10,000) 95 % CIs. BH-FDR over the 5 primary components within each test family.
  H3: BAS (metrics.bas on beat one-hots) own vs matched wrong songs, GT and MACE.

Step 1 (stop rule): reproduce eas_multiseed.json (zero-lag EAS, random other-song baseline).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/multicomponent.py
"""
import json
import os
import pickle
import sys

import librosa
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from model.metric import SMPLSkeleton  # noqa: E402
import audio_features as af  # noqa: E402
import metrics as M  # noqa: E402
from eval_mace import fk  # noqa: E402

# ----------------------------------------------------------------------------- settings
T = 1020
SLOW, FAST = M.ENERGY_SMOOTH, M.ACCENT_SMOOTH          # 15, 2 frames
MAX_LAG = 9                                            # 300 ms at 30 fps
FLUX_SR, FLUX_HOP, FLUX_NFFT = 48000, 1600, 4096
LOW_BAND, HIGH_BAND = (50, 200), (6400, 12800)
PULSE_WIN, PULSE_LAGS = 120, (7, 60)
N_MATCH = 3
N_PERM = N_BOOT = 10000
SEED = 0
UPPER = [9] + list(range(12, 24))
LOWER = [0, 1, 2, 4, 5, 7, 8, 10, 11]
HEADHAND = [15, 20, 21, 22, 23]

COMPONENTS = ['loud', 'perc', 'lowflux', 'highflux', 'pulse']     # primary (FDR family)
ALL_COMPONENTS = COMPONENTS + ['onset']                             # onset = secondary perc variant
SIGMA = {'loud': SLOW, 'pulse': SLOW, 'perc': FAST, 'onset': FAST, 'lowflux': FAST, 'highflux': FAST}
MOTION_OF = {'loud': 'energy', 'perc': 'upper_acc', 'onset': 'upper_acc', 'lowflux': 'lower_speed',
             'highflux': 'headhand_speed', 'pulse': 'pulse'}

GT_DIR = 'data/test'
MACE_DIR = os.path.join(HERE, 'data', 'mace_test')
OUT = os.path.join(HERE, 'results', 'multicomponent.json')
FIG_DIR = os.path.join(HERE, 'results', 'figures')


# ----------------------------------------------------------------------------- features
def band_flux(S, freqs, band):
    m = (freqs >= band[0]) & (freqs < band[1])
    d = np.diff(S[m], axis=1)
    return np.concatenate([[0.0], np.maximum(d, 0).sum(axis=0)])


def rolling_pulse(x, win=PULSE_WIN, lags=PULSE_LAGS, device='cpu'):
    """x: (B, T). Per frame, max over lags of the Pearson autocorrelation inside a centred window
    (windows truncated at the clip edges). Returns (B, T)."""
    x = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    B, n = x.shape
    h = win // 2
    xp = torch.nn.functional.pad(x, (h, win - h - 1), value=float('nan'))
    W = xp.unfold(1, win, 1)                                   # (B, T, win)
    best = torch.full((B, n), -1.0, device=device)
    for k in range(lags[0], lags[1] + 1):
        a, b = W[..., :-k], W[..., k:]
        v = ~(torch.isnan(a) | torch.isnan(b))
        cnt = v.sum(-1).clamp(min=1)
        a0, b0 = torch.where(v, a, 0.), torch.where(v, b, 0.)
        ma, mb = a0.sum(-1) / cnt, b0.sum(-1) / cnt
        da, db = torch.where(v, a - ma[..., None], 0.), torch.where(v, b - mb[..., None], 0.)
        r = (da * db).sum(-1) / torch.sqrt((da ** 2).sum(-1) * (db ** 2).sum(-1) + 1e-12)
        r = torch.where(cnt >= 2 * lags[0], r, torch.full_like(r, -1.0))
        best = torch.maximum(best, r)
    return best.cpu().numpy()


def music_features(name):
    y = af.load_audio(os.path.join(GT_DIR, 'wavs_sliced', name + '.wav'))
    rms = af.rms_envelope(y)
    perc = librosa.feature.rms(y=librosa.effects.percussive(y), hop_length=af.HOP_LENGTH)[0]
    y48, _ = librosa.load(os.path.join(GT_DIR, 'wavs_sliced', name + '.wav'), sr=FLUX_SR)
    S = np.abs(librosa.stft(y48, n_fft=FLUX_NFFT, hop_length=FLUX_HOP))
    freqs = librosa.fft_frequencies(sr=FLUX_SR, n_fft=FLUX_NFFT)
    bf = np.load(os.path.join(GT_DIR, 'baseline_feats', name + '.npy'))
    f = {'loud': rms, 'perc': perc, 'onset': bf[:, 0], 'lowflux': band_flux(S, freqs, LOW_BAND),
         'highflux': band_flux(S, freqs, HIGH_BAND), 'beats': bf[:, 34]}
    lens = {k: len(v) for k, v in f.items()}
    assert min(lens.values()) >= T, lens
    return {k: np.asarray(v[:T], dtype=np.float64) for k, v in f.items()}


def speed(j, idx=None):
    j = j if idx is None else j[:, idx]
    return np.pad(np.linalg.norm(j[1:] - j[:-1], axis=2).mean(1), (0, 1), mode='edge')


def motion_features(j):
    j = j[:T].astype(np.float64)
    acc = np.linalg.norm(j[2:, UPPER] - 2 * j[1:-1, UPPER] + j[:-2, UPPER], axis=2).mean(1)
    return {'energy': M.motion_energy(j), 'upper_acc': np.pad(acc, (1, 1), mode='edge'),
            'lower_speed': speed(j, LOWER), 'headhand_speed': speed(j, HEADHAND), 'body_speed': speed(j)}


def smooth(x, sigma):
    return gaussian_filter1d(np.asarray(x, dtype=np.float64), sigma)


def score(music_s, motion_s):
    r0, rmax, lag = M.lagged_corr(music_s, motion_s, MAX_LAG)
    return r0, rmax, lag


# ----------------------------------------------------------------------------- stats
def signflip_p(d, rng, alternative='two-sided'):
    d = np.asarray(d)
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = (signs * d).mean(1)
    obs = d.mean()
    if alternative == 'greater':
        return float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1))


def boot_ci(v, rng):
    v = np.asarray(v)
    m = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out.tolist()


def wil(d, alternative):
    d = np.asarray(d)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d, alternative=alternative).pvalue)


def component_stats(ro, rx, mo, mx, rng):
    dr, dm = ro - rx, mo - mx
    dd = dr - dm
    return {
        'real_own': float(ro.mean()), 'real_other': float(rx.mean()),
        'mace_own': float(mo.mean()), 'mace_other': float(mx.mean()),
        'd_real': float(dr.mean()), 'd_real_ci': boot_ci(dr, rng),
        'd_mace': float(dm.mean()), 'd_mace_ci': boot_ci(dm, rng),
        'G': float(dd.mean()), 'G_ci': boot_ci(dd, rng),
        'G_wilcoxon_p': wil(dd, 'two-sided'), 'G_perm_p': signflip_p(dd, rng),
        'H1_wilcoxon_p': wil(dr, 'greater'), 'H1_perm_p': signflip_p(dr, rng, 'greater'),
        'H2_wilcoxon_p': wil(dm, 'greater'), 'H2_perm_p': signflip_p(dm, rng, 'greater'),
        'n_real_pos': int((dr > 0).sum()), 'n_mace_pos': int((dm > 0).sum()),
        'n_G_pos': int((dd > 0).sum()), 'n': len(dr),
    }


# ----------------------------------------------------------------------------- main
def main():
    with open(os.path.join(MACE_DIR, 'names.json')) as f:
        meta = json.load(f)
    names = meta['names']
    assert names == sorted(names)
    mace = np.load(os.path.join(MACE_DIR, 'joints.npy')).astype(np.float32)   # (N, S, T, 24, 3)
    N, NS = mace.shape[:2]
    songs = np.array([n.split('_')[0] for n in names])
    order = sorted(set(songs))
    genre = {s: json.load(open(os.path.join(GT_DIR, 'label', s + '.json')))['style2'] for s in order}
    for n, g in zip(names, meta['genres']):
        assert genre[n.split('_')[0]] == g, (n, g)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- GT joints
    poses, trans = [], []
    for n in names:
        with open(os.path.join(GT_DIR, 'motions_sliced', n + '.pkl'), 'rb') as f:
            m = pickle.load(f)
        poses.append(m['q']); trans.append(m['pos'])
    gt = fk(SMPLSkeleton(device), poses, trans, device, from_pred=False)
    print('GT joints', gt.shape, 'MACE joints', mace.shape)

    # ---- music features
    mus = [music_features(n) for n in names]
    pulse_m = rolling_pulse(np.stack([m['onset'] for m in mus]), device=device)
    for m, p in zip(mus, pulse_m):
        m['pulse'] = p.astype(np.float64)
    mus_s = [{k: smooth(m[k], SIGMA[k]) for k in ALL_COMPONENTS} for m in mus]
    print('music features done')

    # ---- step 1: sanity check (exact eas_multiseed procedure, zero-lag EAS sigma 15)
    def rand_other(seed):
        rng = np.random.default_rng(seed)
        return [int(rng.choice(np.flatnonzero(songs != songs[i]))) for i in range(N)]

    eas0 = lambda rms, j: M.eas(rms, j)[0]
    rms_raw = [m['loud'] for m in mus]
    gt_own = np.array([eas0(rms_raw[i], gt[i]) for i in range(N)])
    gt_oth = np.mean([[eas0(rms_raw[j], gt[i]) for i, j in enumerate(rand_other(k))] for k in range(10)], 0)
    mo, mx = [], []
    for s in range(NS):
        mo.append([eas0(rms_raw[i], mace[i, s]) for i in range(N)])
        mx.append([eas0(rms_raw[j], mace[i, s]) for i, j in enumerate(rand_other(s))])
    ps = lambda v: np.array([np.mean(np.asarray(v)[songs == s]) for s in order])
    sc = {'gt_own': ps(gt_own), 'gt_other': ps(gt_oth), 'mace_own': ps(np.mean(mo, 0)),
          'mace_other': ps(np.mean(mx, 0))}
    sanity = {k: float(v.mean()) for k, v in sc.items()}
    sanity['gt_n_own_gt_other'] = int((sc['gt_own'] > sc['gt_other']).sum())
    sanity['mace_n_own_gt_other'] = int((sc['mace_own'] > sc['mace_other']).sum())
    print('SANITY', json.dumps(sanity))
    ref = {'gt_own': 0.1837, 'gt_other': 0.0497, 'mace_own': 0.0611, 'mace_other': 0.0495}
    bad = {k: (sanity[k], v) for k, v in ref.items() if abs(sanity[k] - v) > 0.01}
    if bad:
        print('STOP: sanity check failed', bad)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump({'sanity_check': sanity, 'sanity_failed': {k: list(v) for k, v in bad.items()}},
                  open(OUT, 'w'), indent=1)
        return

    # ---- motion features (smoothed per component)
    gt_f = [motion_features(j) for j in gt]
    mace_f = [[motion_features(mace[i, s]) for s in range(NS)] for i in range(N)]
    gt_pulse = rolling_pulse(np.stack([f['body_speed'] for f in gt_f]), device=device)
    mace_pulse = rolling_pulse(np.stack([mace_f[i][s]['body_speed'] for i in range(N) for s in range(NS)]),
                               device=device).reshape(N, NS, T)
    for i in range(N):
        gt_f[i]['pulse'] = gt_pulse[i].astype(np.float64)
        for s in range(NS):
            mace_f[i][s]['pulse'] = mace_pulse[i, s].astype(np.float64)

    def smooth_motion(f):
        return {c: smooth(f[MOTION_OF[c]], SIGMA[c]) for c in ALL_COMPONENTS}
    gt_s = [smooth_motion(f) for f in gt_f]
    mace_s = [[smooth_motion(f) for f in fs] for fs in mace_f]
    print('motion features done')

    # ---- tempo + matching
    bpm = {}
    for s in order:
        ibis = np.concatenate([np.diff(np.flatnonzero(mus[i]['beats'])) for i in np.flatnonzero(songs == s)])
        bpm[s] = float(60 * 30 / np.median(ibis))
    match = {}
    for s in order:
        cand = sorted((o for o in order if o != s), key=lambda o: (genre[o] != genre[s], abs(bpm[o] - bpm[s])))
        match[s] = cand[:N_MATCH]
    rng = np.random.default_rng(SEED)
    pairing = []                                                  # per slice: list of wrong slice indices
    for i in range(N):
        pairing.append([int(rng.choice(np.flatnonzero(songs == o))) for o in match[songs[i]]])
    match_info = {s: {'genre': genre[s], 'bpm': bpm[s], 'n_slices': int((songs == s).sum()),
                      'matches': [{'song': o, 'genre': genre[o], 'bpm': bpm[o],
                                   'abs_bpm_diff': abs(bpm[o] - bpm[s]), 'same_genre': genre[o] == genre[s]}
                                  for o in match[s]]} for s in order}
    diffs = [m['abs_bpm_diff'] for v in match_info.values() for m in v['matches']]
    same = [m['same_genre'] for v in match_info.values() for m in v['matches']]
    match_summary = {'n_pairs': len(diffs), 'n_same_genre': int(sum(same)),
                     'abs_bpm_diff_mean': float(np.mean(diffs)), 'abs_bpm_diff_median': float(np.median(diffs)),
                     'abs_bpm_diff_max': float(np.max(diffs)),
                     'songs_with_3_same_genre': int(sum(all(m['same_genre'] for m in v['matches'])
                                                        for v in match_info.values()))}
    print('MATCH', json.dumps(match_summary))

    # ---- slice-level scores: arrays [N, (own, other)] for r0, rmax, lag
    keys = ALL_COMPONENTS + ['BAS']
    res = {who: {k: {'own': np.zeros((N, 3)), 'other': np.zeros((N, 3))} for k in keys} for who in ('real', 'mace')}
    for i in range(N):
        for c in ALL_COMPONENTS:
            res['real'][c]['own'][i] = score(mus_s[i][c], gt_s[i][c])
            res['real'][c]['other'][i] = np.mean([score(mus_s[j][c], gt_s[i][c]) for j in pairing[i]], 0)
            res['mace'][c]['own'][i] = np.mean([score(mus_s[i][c], mace_s[i][s][c]) for s in range(NS)], 0)
            res['mace'][c]['other'][i] = np.mean([score(mus_s[j][c], mace_s[i][s][c])
                                                  for s in range(NS) for j in pairing[i]], 0)
        res['real']['BAS']['own'][i] = M.bas(mus[i]['beats'], gt[i][:T])
        res['real']['BAS']['other'][i] = np.mean([M.bas(mus[j]['beats'], gt[i][:T]) for j in pairing[i]])
        res['mace']['BAS']['own'][i] = np.mean([M.bas(mus[i]['beats'], mace[i, s, :T]) for s in range(NS)])
        res['mace']['BAS']['other'][i] = np.mean([M.bas(mus[j]['beats'], mace[i, s, :T])
                                                  for s in range(NS) for j in pairing[i]])
    print('scores done')

    def song_level(arr):                                          # (N,) -> (19,)
        return np.array([arr[songs == s].mean() for s in order])

    # ---- statistics
    rng = np.random.default_rng(SEED)
    stats, per_song = {}, {}
    for score_kind, col in (('lagmax', 1), ('zerolag', 0)):
        st = {}
        for c in keys:
            if c == 'BAS' and score_kind == 'zerolag':
                continue
            cc = 0 if c == 'BAS' else col
            v = {f'{w}_{o}': song_level(res[w][c][o][:, cc]) for w in ('real', 'mace') for o in ('own', 'other')}
            st[c] = component_stats(v['real_own'], v['real_other'], v['mace_own'], v['mace_other'], rng)
            if score_kind == 'lagmax':
                per_song[c] = {s: {k: float(x[n]) for k, x in v.items()} for n, s in enumerate(order)}
                if c != 'BAS':
                    for w in ('real', 'mace'):
                        lags = res[w][c]['own'][:, 2]
                        st[c][f'{w}_own_lag_mean'] = float(lags.mean())
                        st[c][f'{w}_own_lag_median'] = float(np.median(lags))
        for fam in ('G_wilcoxon_p', 'G_perm_p', 'H1_wilcoxon_p', 'H1_perm_p', 'H2_wilcoxon_p', 'H2_perm_p'):
            q = bh([st[c][fam] for c in COMPONENTS])
            for c, qq in zip(COMPONENTS, q):
                st[c][fam[:-2] + '_q'] = qq
        stats[score_kind] = st

    # ---- feature correlation matrices (smoothed, each with its own sigma)
    def corr_mats(feats_list):
        X = np.stack([np.concatenate([f[c] for f in feats_list]) for c in ALL_COMPONENTS])
        pooled = np.corrcoef(X)
        within = np.mean([np.corrcoef(np.stack([f[c] for c in ALL_COMPONENTS])) for f in feats_list], 0)
        return {'labels': ALL_COMPONENTS, 'pooled': pooled.round(4).tolist(), 'mean_within_slice': within.round(4).tolist()}
    corr = {'music': corr_mats(mus_s), 'motion_real': corr_mats(gt_s),
            'motion_mace': corr_mats([mace_s[i][s] for i in range(N) for s in range(NS)])}
    corr['motion_labels'] = {c: MOTION_OF[c] for c in ALL_COMPONENTS}

    out = {
        'settings': {'T': T, 'sigma_slow': SLOW, 'sigma_fast': FAST, 'sigma': SIGMA, 'max_lag_frames': MAX_LAG,
                     'flux': {'sr': FLUX_SR, 'hop': FLUX_HOP, 'n_fft': FLUX_NFFT, 'low': LOW_BAND, 'high': HIGH_BAND},
                     'pulse': {'win': PULSE_WIN, 'lags': PULSE_LAGS, 'note': 'approximation, not MIRtoolbox'},
                     'joints': {'upper': UPPER, 'lower': LOWER, 'headhand': HEADHAND},
                     'n_match': N_MATCH, 'n_perm': N_PERM, 'n_boot': N_BOOT, 'seed': SEED,
                     'n_slices': N, 'n_seeds': NS, 'n_songs': len(order),
                     'primary_components': COMPONENTS, 'secondary_components': ['onset'],
                     'tests': 'G: two-sided; H1/H2: one-sided Delta>0; BH-FDR over primary components per family'},
        'sanity_check': {'values': sanity, 'reference': ref,
                         'per_song': {s: {k: float(v[n]) for k, v in sc.items()} for n, s in enumerate(order)}},
        'matching': {'summary': match_summary, 'per_song': match_info,
                     'pairing': {n: [names[j] for j in p] for n, p in zip(names, pairing)}},
        'stats': stats, 'per_song': per_song, 'feature_correlations': corr,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', OUT)
    make_figures(out, order)
    print_table(out)


# ----------------------------------------------------------------------------- report helpers
def print_table(out):
    for kind in ('lagmax', 'zerolag'):
        print(f'\n== {kind} ==')
        for c, s in out['stats'][kind].items():
            print(f"{c:9s} R {s['real_own']:.3f}/{s['real_other']:.3f} M {s['mace_own']:.3f}/{s['mace_other']:.3f} "
                  f"dR {s['d_real']:+.3f} [{s['d_real_ci'][0]:+.3f},{s['d_real_ci'][1]:+.3f}] "
                  f"dM {s['d_mace']:+.3f} [{s['d_mace_ci'][0]:+.3f},{s['d_mace_ci'][1]:+.3f}] "
                  f"G {s['G']:+.3f} [{s['G_ci'][0]:+.3f},{s['G_ci'][1]:+.3f}] "
                  f"pG w{s['G_wilcoxon_p']:.3f}/q{s.get('G_wilcoxon_q', np.nan):.3f} "
                  f"perm{s['G_perm_p']:.3f}/q{s.get('G_perm_q', np.nan):.3f} "
                  f"H1 w{s['H1_wilcoxon_p']:.3f}/q{s.get('H1_wilcoxon_q', np.nan):.3f} "
                  f"perm{s['H1_perm_p']:.3f}/q{s.get('H1_perm_q', np.nan):.3f} "
                  f"H2 w{s['H2_wilcoxon_p']:.3f}/q{s.get('H2_wilcoxon_q', np.nan):.3f} "
                  f"perm{s['H2_perm_p']:.3f}/q{s.get('H2_perm_q', np.nan):.3f} "
                  f"n+ R{s['n_real_pos']} M{s['n_mace_pos']} G{s['n_G_pos']}/{s['n']}"
                  + (f" lag R{s['real_own_lag_mean']:+.1f} M{s['mace_own_lag_mean']:+.1f}" if 'real_own_lag_mean' in s else ''))


def make_figures(out, order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    REAL, MACE, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#8a8984'
    comps = COMPONENTS + ['onset', 'BAS']
    st, ps = out['stats']['lagmax'], out['per_song']
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED})

    # (a) slope plots
    fig, axes = plt.subplots(2, len(comps), figsize=(2.1 * len(comps), 5.4), sharey='col')
    for col, c in enumerate(comps):
        for row, (w, color, lab) in enumerate((('real', REAL, 'Real'), ('mace', MACE, 'MACE'))):
            ax = axes[row, col]
            o = np.array([ps[c][s][f'{w}_own'] for s in order])
            x = np.array([ps[c][s][f'{w}_other'] for s in order])
            for a, b in zip(o, x):
                ax.plot([0, 1], [a, b], color=color, alpha=0.35, lw=1)
            ax.plot([0, 1], [o.mean(), x.mean()], color=color, lw=2.5, marker='o', ms=7)
            ax.set_xticks([0, 1], ['own', 'other']); ax.set_xlim(-0.3, 1.3)
            ax.axhline(0, color=MUTED, lw=0.6, ls=':')
            if row == 0:
                ax.set_title(c, color=INK)
            if col == 0:
                ax.set_ylabel(f'{lab}: song mean S')
    fig.suptitle('Own vs BPM/genre-matched wrong song, per song (lag-max S; BAS for beat timing)', color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'mc_a_slope.png'), dpi=150); plt.close(fig)

    # (b) deltas with CIs
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs = np.arange(len(comps))
    for off, w, color, lab in ((-0.12, 'real', REAL, 'Real Δ'), (0.12, 'mace', MACE, 'MACE Δ')):
        m = np.array([st[c][f'd_{w}'] for c in comps])
        ci = np.array([st[c][f'd_{w}_ci'] for c in comps])
        ax.errorbar(xs + off, m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt='o', color=color, ms=8, lw=2,
                    capsize=0, label=lab)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(xs, comps); ax.set_ylabel('S_own − S_other (song mean)')
    ax.legend(frameon=False, loc='upper right')
    ax.set_title('Song-specific advantage with bootstrap 95% CI (19 songs)', color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'mc_b_delta_ci.png'), dpi=150); plt.close(fig)

    # (c) per-song scatter
    fig, axes = plt.subplots(1, len(comps), figsize=(2.4 * len(comps), 2.8))
    for ax, c in zip(axes, comps):
        dr = np.array([ps[c][s]['real_own'] - ps[c][s]['real_other'] for s in order])
        dm = np.array([ps[c][s]['mace_own'] - ps[c][s]['mace_other'] for s in order])
        lim = max(np.abs(dr).max(), np.abs(dm).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls='--')
        ax.axhline(0, color=MUTED, lw=0.5); ax.axvline(0, color=MUTED, lw=0.5)
        ax.scatter(dr, dm, s=22, color=REAL, edgecolor='white', lw=0.8, zorder=3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
        ax.set_title(c, color=INK); ax.set_xlabel('Δ_real')
    axes[0].set_ylabel('Δ_MACE')
    fig.suptitle('Per-song Δ_real vs Δ_MACE (below diagonal: MACE loses song-specific coupling)', color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'mc_c_scatter.png'), dpi=150); plt.close(fig)
    print('figures saved to', FIG_DIR)


if __name__ == '__main__':
    main()
