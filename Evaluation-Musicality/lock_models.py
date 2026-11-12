"""Beat phase locking (continuous, after Burger et al. 2014 Exp. 2) for real dance and models; FineDance test or Motorica.

Metric (one clip, one dance):
  signal  QoM(t) = mean over the 24 joints of the 3D frame-to-frame displacement norm (unsquared, global joints; the same
          signal as loud_models.py, NO extra smoothing - the band-pass is the smoothing)
  phi(t)  phase of the QoM at the clip's beat frequency f_c: zero-phase FFT band-pass, total width 15 % of f_c, Hilbert
          transform (Burger: same filter and transform on the root marker position)
  theta   beat phase = 2 pi x (beat index), linear interpolation between beats (Burger), beats = the librosa beats of the clip
          (the conditioning's beat column 34 for FineDance, the beats_slices.npz cache for Motorica)
  r       |mean_t exp(i (phi - theta))| over the clip without TRIM seconds at both ends. One constant offset per clip.
Specificity (ours, not Burger): Delta = r(own beats) - mean r(3 wrong slices), the wrong slices exactly as loud_models.py
  (tempo-first matched songs / pieces, one seeded random slice each), but their beat intervals are RESCALED TO THE OWN
  TEMPO (mean interval from a line fit of beat frame vs index) so the wrong beats differ from the own ones in phase only.
  Unit = song (FineDance, 19) or music piece (Motorica, 52). Tests: one-sided sign-flip permutation of Delta > 0 (10,000),
  Wilcoxon next to it; q = BH over the models for the H2 family; the real-dance H1 is a single test.
  --level L filters at L x the beat frequency (theta scaled by L as well); --trim S seconds cut at both ends.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/lock_models.py --dataset finedance --seeds 10
    .venv/bin/python ../Evaluation-Musicality/lock_models.py --dataset motorica --seeds 3  [--models all]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from multicomponent import SEED, boot_ci, wil, signflip_p, bh  # noqa: E402
from robustness_loud import motion_signals, matches, TEMPO_TOL  # noqa: E402

T, FPS, BW = 1020, 30, 0.15
FD_MODELS = {'MACE released': 'mace_test', 'MACE mask2s': 'mace_test_local2s',
             'Ours final': 'mace_test_mask2s_guide10_s0.003_foot10_frz',
             'Ours critic v5': 'mace_test_mask2s_criticv5_s0.003_foot10_frz',
             'Ours critic v5 + gain': 'mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10',
             'Ours mask + guidance': 'mace_test_mask2s_guide10_s0.003',
             'Abl - mask': 'mace_test_abl_nomask', 'Abl - critic': 'mace_test_abl_nocritic', 'Abl - foot': 'mace_test_abl_nofoot', 'Abl - frozen root': 'mace_test_abl_nofrz', 'Ours + foot term': 'mace_test_mask2s_guide10_s0.003_foot10_10seeds', 'EDGE': 'mace_test_edge',
             'Lodge': 'mace_test_lodge', 'Bailando': 'mace_test_bailando', 'MEGADance': 'mace_test_mega',
             'FlowerDance': 'mace_test_flower'}
MOT_MODELS = {'MACE released': 'mace', 'MACE mask2s': 'mace_local2s',
              'Ours final': 'mace_local2s_guide_s0.003_foot10_frz',
              'Ours critic v5': 'mace_local2s_criticv5_s0.003_foot10_frz',
              'Ours critic v5 + gain': 'mace_local2s_criticv5_s0.003_foot10_frz_gain',
              'Ours mask + guidance': 'mace_local2s_guide_s0.003',
              'Ours + foot term': 'mace_local2s_guide_s0.003_foot10', 'EDGE': 'edge', 'Lodge': 'lodge',
              'Bailando': 'bailando', 'MEGADance': 'mega', 'Ours AIST++': 'ours_aist'}


def interval(beats):
    """frames per beat: slope of a line fit of beat frame vs beat index (sub-frame tempo)."""
    return float(np.polyfit(np.arange(len(beats)), np.asarray(beats, float), 1)[0])


def beat_phase(beats, own_interval, level, rescale=True):
    """unwrapped beat phase (radians) of frames 0..T-1; beats rescaled so their mean interval = own_interval; the
    phase is extrapolated at one beat per own_interval before the first and after the last beat."""
    b = np.asarray(beats, float)
    k = np.arange(len(b))
    if rescale:
        b = b[0] + (b - b[0]) * (own_interval / interval(b))
    else:
        own_interval = interval(b)   # raw wrong beats at their own tempo
    t = np.arange(T)
    th = np.interp(t, b, 2 * np.pi * k)
    th = np.where(t < b[0], 2 * np.pi * (t - b[0]) / own_interval, th)
    th = np.where(t > b[-1], 2 * np.pi * (len(b) - 1) + 2 * np.pi * (t - b[-1]) / own_interval, th)
    return level * th


def analytic_at(x, fc):
    """zero-phase FFT band-pass of width BW * fc around fc (cycles/frame), as analytic signal (Hilbert transform)."""
    X = np.fft.fft(x - x.mean())
    f = np.fft.fftfreq(len(x))
    return np.fft.ifft(2 * X * ((f > 0) & (np.abs(f - fc) <= BW * fc / 2)))


def lock_r(q, own_interval, beats, level, trim, rescale=True):
    """r of one QoM signal q against one beat list (already rescaled to the own tempo inside beat_phase)."""
    if len(beats) < 3:
        return np.nan
    a = analytic_at(q, level / own_interval)
    d = np.angle(a) - beat_phase(beats, own_interval, level, rescale)
    return float(np.abs(np.exp(1j * d)[trim:T - trim].mean()))


def load_finedance(n_seeds, model_labels):
    import torch
    from model.metric import SMPLSkeleton
    from eval_mace import fk
    from multicomponent import GT_DIR, MACE_DIR
    from robustness_loud import song_bpm
    names = json.load(open(os.path.join(MACE_DIR, 'names.json')))['names']
    N = len(names)
    units = np.array([n.split('_')[0] for n in names])
    order = sorted(set(units))
    genre = {s: json.load(open(os.path.join(GT_DIR, 'label', s + '.json')))['style2'] for s in order}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    poses, trans = [], []
    for n in names:
        with open(os.path.join(GT_DIR, 'motions_sliced', n + '.pkl'), 'rb') as f:
            m = pickle.load(f)
        poses.append(m['q'])
        trans.append(m['pos'])
    real = fk(SMPLSkeleton(device), poses, trans, device, from_pred=False)
    beats = [np.flatnonzero(np.load(f'data/test/baseline_feats/{n}.npy')[:, 34]) for n in names]
    sets = {'Real dance': [[np.asarray(real[i], dtype=np.float64)] for i in range(N)]}
    for label in model_labels:
        p = os.path.join(HERE, 'data', FD_MODELS[label], 'joints.npy')
        J = np.load(p, mmap_mode='r')
        assert J.shape[0] == N and J.shape[2] == T and J.shape[1] >= n_seeds, (label, J.shape)
        sets[label] = [[np.asarray(J[i, s], dtype=np.float64) for s in range(n_seeds)] for i in range(N)]
    bpm = {s: song_bpm(s) for s in order}
    mt = matches(order, genre, bpm, 'tempo_first')
    rng = np.random.default_rng(SEED)
    pair = [[int(rng.choice(np.flatnonzero(units == o))) for o in mt[units[i]]] for i in range(N)]
    return units, order, beats, sets, pair, bpm


def load_motorica(n_seeds, model_labels):
    from finedance_release_loud import music_groups
    from motorica_common import ROOT, SLICE_LEN, body_joints, style, wav_path
    from robustness_loud import bpm_from_wav
    mot = os.path.dirname(ROOT)
    meta = json.load(open(os.path.join(mot, 'mace', 'slices.json')))
    keep = [k for k, (t, o) in enumerate(meta['slices']) if o % SLICE_LEN == 0]
    slices = [tuple(meta['slices'][k]) for k in keep]
    z = np.load(os.path.join(mot, 'beats_slices.npz'), allow_pickle=True)
    assert [tuple(s) for s in z['slices'].tolist()] == slices
    beats = [np.asarray(b) for b in z['beats']]
    takes = sorted({t for t, _ in slices})
    group = music_groups(takes, path=wav_path)
    pieces = sorted(set(group.values()))
    genre = {p: style(p) for p in pieces}
    units = np.array([group[t] for t, _ in slices])
    cache, real = {}, []
    for t, o in slices:
        if t not in cache:
            cache = {t: body_joints(t)}
        real.append([np.asarray(cache[t][o:o + SLICE_LEN], dtype=np.float64)])
    sets = {'Real dance': real}
    for label in model_labels:
        p = os.path.join(mot, MOT_MODELS[label], 'joints.npy')
        J = np.load(p, mmap_mode='r')
        if J.shape[0] == len(keep):
            assert [tuple(x) for x in json.load(open(os.path.join(mot, MOT_MODELS[label], 'slices.json')))['slices']] == slices
            idx = list(range(len(keep)))
        else:
            idx = keep
        assert J.shape[1] >= n_seeds, (label, J.shape)
        sets[label] = [[np.asarray(J[k, s], dtype=np.float64) for s in range(n_seeds)] for k in idx]
    bpm = {p: bpm_from_wav(wav_path(p)) for p in pieces}
    mt = matches(pieces, genre, bpm, 'tempo_first')
    piece_slices = {p: [i for i, q in enumerate(units) if q == p] for p in pieces}
    rng = np.random.default_rng(SEED)
    pair = [[piece_slices[w][int(rng.integers(len(piece_slices[w])))] for w in mt[group[t]]] for t, _ in slices]
    return units, pieces, beats, sets, pair, bpm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('finedance', 'motorica'), required=True)
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--models', default='released', help="'released' (default), 'all' or comma-separated labels")
    ap.add_argument('--level', type=float, default=1.0)
    ap.add_argument('--trim', type=float, default=4.0, help='seconds cut at both ends')
    ap.add_argument('--null', choices=('rescaled', 'raw'), default='rescaled',
                    help='wrong beats rescaled to the own tempo (default) or raw at their own tempo (tempo-first matching only)')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    trim = int(round(a.trim * FPS))
    table = FD_MODELS if a.dataset == 'finedance' else MOT_MODELS
    labels = (['MACE released'] if a.models == 'released' else list(table) if a.models == 'all'
              else a.models.split(','))
    units, order, beats, sets, pair, bpm = (load_finedance if a.dataset == 'finedance' else load_motorica)(a.seeds, labels)
    N = len(beats)
    iv = [interval(b) if len(b) >= 3 else np.nan for b in beats]
    qom = lambda j: motion_signals(j)['speed']  # noqa: E731
    print(a.dataset, N, 'slices', len(order), 'units; sets', {k: len(v[0]) for k, v in sets.items()}, flush=True)

    def unit_level(v):
        return np.array([np.nanmean(v[units == u]) for u in order])

    per = {}
    for label, mot in sets.items():
        ns = len(mot[0])
        o_s = np.full((N, ns), np.nan)
        w_s = np.full((N, ns), np.nan)
        for i in range(N):
            for s in range(ns):
                q = qom(mot[i][s])
                o_s[i, s] = lock_r(q, iv[i], beats[i], a.level, trim)
                w_s[i, s] = np.nanmean([lock_r(q, iv[i], beats[j], a.level, trim, a.null == 'rescaled') for j in pair[i]])
        own, wrong = unit_level(np.nanmean(o_s, 1)), unit_level(np.nanmean(w_s, 1))
        per[label] = {'own': own, 'wrong': wrong, 'delta': own - wrong,
                      'seed_delta': [float((unit_level(o_s[:, s]) - unit_level(w_s[:, s])).mean()) for s in range(ns)]}
        print(label, 'done', flush=True)

    rng = np.random.default_rng(SEED)
    res = {}
    for label, v in per.items():
        d = v['delta']
        res[label] = {'r_own': float(v['own'].mean()), 'r_wrong': float(v['wrong'].mean()), 'delta': float(d.mean()),
                      'delta_ci': boot_ci(d, rng), 'p_perm': signflip_p(d, rng, 'greater'),
                      'p_wilcoxon': wil(d, 'greater'), 'n_pos': int((d > 0).sum()), 'n': len(d),
                      'seed_delta': v['seed_delta']}
        if label != 'Real dance':
            g = per['Real dance']['delta'] - d
            res[label].update({'G': float(g.mean()), 'G_ci': boot_ci(g, rng), 'G_p_perm': signflip_p(g, rng),
                               'G_p_wilcoxon': wil(g, 'two-sided')})
    ml = [k for k in res if k != 'Real dance']
    for key, q in (('p_perm', 'q_perm'), ('p_wilcoxon', 'q_wilcoxon'), ('G_p_perm', 'G_q_perm')):
        for k, qv in zip(ml, bh([res[k][key] for k in ml])):
            res[k][q] = qv
    res['Real dance']['q_perm'] = res['Real dance']['p_perm']
    res['Real dance']['q_wilcoxon'] = res['Real dance']['p_wilcoxon']
    allk = list(res)
    for k, qv in zip(allk, bh([res[k]['p_perm'] for k in allk])):   # BH over real + models together
        res[k]['q_joint'] = qv

    print(f"\n{a.seeds} seed(s); level {a.level}x, trim {a.trim}s, null {a.null}")
    print(f"{'model':16s} {'r_own':>6s} {'r_wrong':>7s} {'Delta':>6s} {'CI95':>15s} {'pos':>6s} {'p':>7s} {'q':>7s} {'q_joint':>7s} "
          f"{'p_wilc':>7s} {'seed Delta min..max':>20s}")
    for k, v in res.items():
        sd = v['seed_delta']
        print(f"{k:16s} {v['r_own']:6.3f} {v['r_wrong']:7.3f} {v['delta']:6.3f} "
              f"[{v['delta_ci'][0]:+.3f},{v['delta_ci'][1]:+.3f}] {v['n_pos']:2d}/{v['n']} {v['p_perm']:7.4f} "
              f"{v['q_perm']:7.4f} {v['q_joint']:7.4f} {v['p_wilcoxon']:7.4f} {min(sd):+.3f}..{max(sd):+.3f}")
    for k, v in res.items():
        if 'G' in v:
            print(f"G real - {k}: {v['G']:+.3f} [{v['G_ci'][0]:+.3f},{v['G_ci'][1]:+.3f}] p {v['G_p_perm']:.4f}")
    out = {'settings': {'dataset': a.dataset, 'level': a.level, 'trim_s': a.trim, 'bandwidth': BW, 'n_seeds': a.seeds,
                        'signal': 'QoM (unsquared mean joint speed, global), no extra smoothing',
                        'beats': 'librosa (conditioning column 34 / beats_slices.npz)', 'null': a.null,
                        'match': 'tempo_first', 'tempo_tol': TEMPO_TOL, 'n_units': len(order), 'models': labels,
                        'n_perm': 10000, 'rng_seed': SEED},
           'results': res, 'per_unit': {k: {kk: dict(zip(order, map(float, vv))) for kk, vv in v.items() if kk != 'seed_delta'}
                                        for k, v in per.items()}}
    fn = os.path.join(HERE, 'results', f'lock_models_{a.dataset}_{a.seeds}seed_{a.null}{a.tag}.json')
    with open(fn, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', fn)


if __name__ == '__main__':
    main()
