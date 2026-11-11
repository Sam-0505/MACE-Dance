"""AIST++ in-domain comparison table (2026-11-11): Real dance, EDGE, Bailando, Ours on the 20 AIST++ test clips.

Columns: FID_k, FID_g, FID_s, DIV_k, DIV_g, DIV_s, BAS, Lock, Loud.
Inputs: data/aist/eval/clips.json and <model>/<name>.npy of shape (S, T, 24, 3) (Y-up, 30 fps, metres), made by
  aist_eval.py --save (ours), dance_baselines/EDGE/gen_aist_test.py, dance_baselines/Bailando/gen_aist_test.py.
Real dance = the test dances' first T frames.

  FID_k, FID_g, DIV_k, DIV_g, BAS   Bailando's protocol (aist_eval.py): features of the generated dances (30 fps upsampled to 60 fps
        by linear interpolation) against ALL 1408 real AIST++ motions; one set of 20 dances per seed, mean over seeds. Real row = the
        20 real test dances scored the same way (its FID is not 0 because it is 20 samples against the full dataset).
  FID_s, DIV_s   style metrics (MEGADance's recipe: classifier output, normalised by the real features, Frechet distance, mean pairwise
        distance) with our own 10-genre AIST++ classifier on joint positions (aist_genre_cls.py), 5 s windows (stride 2.5 s) of the
        20 dances; real reference = the real test windows (so the real row's FID_s is 0 by construction and is shown as a dash).
  Loud   r = Pearson(Gauss15(audio RMS), Gauss15(QoM)), QoM = mean joint speed; Delta = r(own music) - mean r(3 wrong pieces, tempo-first
        matched among the 10 test pieces, the first T frames of their music). Unit = music piece (10), mean over its clips and seeds.
  Lock   beat phase locking r (lock_models.py: band-pass of the QoM at the beat frequency, Hilbert phase, circular concentration against
        the beat phase), Delta = r(own beats) - mean r(3 wrong pieces' beats rescaled to the own tempo), unit = piece, trim 1 s at both
        ends (clips are 5-10 s; FineDance used 4 s of 34 s).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/aist_table.py [--models ours=data/aist/eval/ours edge=... bailando=...]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import aist_eval as AE  # noqa: E402
import audio_features as af  # noqa: E402
from aist_common import nominal_bpm  # noqa: E402

EV = os.path.join(HERE, 'data', 'aist', 'eval')
FPS, BW = 30, 0.15
LABELS = {'edge': 'EDGE', 'bailando': 'Bailando', 'ours': 'Ours'}


def up60(j):
    """(T, 24, 3) 30 fps -> 60 fps by linear interpolation."""
    t = np.arange(len(j))
    t2 = np.arange(2 * len(j)) / 2.0
    t2 = t2[t2 <= t[-1]]
    return np.stack([np.stack([np.interp(t2, t, j[:, a, c]) for c in range(3)], -1) for a in range(24)], 1)


def qom(j):
    sp = np.linalg.norm(j[1:] - j[:-1], axis=2).mean(1)
    return np.pad(sp, (0, 1), mode='edge')


def interval(beats):
    return float(np.polyfit(np.arange(len(beats)), np.asarray(beats, float), 1)[0])


def beat_phase(beats, own_interval, T):
    b = np.asarray(beats, float)
    k = np.arange(len(b))
    b = b[0] + (b - b[0]) * (own_interval / interval(b))
    t = np.arange(T)
    th = np.interp(t, b, 2 * np.pi * k)
    th = np.where(t < b[0], 2 * np.pi * (t - b[0]) / own_interval, th)
    th = np.where(t > b[-1], 2 * np.pi * (len(b) - 1) + 2 * np.pi * (t - b[-1]) / own_interval, th)
    return th


def lock_r(q, iv, beats, trim):
    if len(beats) < 3:
        return np.nan
    T = len(q)
    X = np.fft.fft(q - q.mean())
    f = np.fft.fftfreq(T)
    fc = 1.0 / iv
    a = np.fft.ifft(2 * X * ((f > 0) & (np.abs(f - fc) <= BW * fc / 2)))
    d = np.angle(a) - beat_phase(beats, iv, T)
    return float(np.abs(np.exp(1j * d)[trim:T - trim].mean()))


def tempo_dist(b_other, b_self):
    return min(abs(k * b_other - b_self) for k in (0.5, 1, 2)) / b_self


def signflip(d, rng, n=20000):
    d = np.asarray(d)
    d = d[~np.isnan(d)]
    s = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return float((np.sum(np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-12) + 1) / (n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=[f'{k}={EV}/{k}' for k in LABELS])
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'aist_table'))
    ap.add_argument('--trim', type=float, default=1.0)
    a = ap.parse_args()
    clips = json.load(open(os.path.join(EV, 'clips.json')))['clips']
    N = len(clips)
    names = [c['name'] for c in clips]
    pieces = sorted({c['piece'] for c in clips})
    piece = np.array([c['piece'] for c in clips])
    trim = int(round(a.trim * FPS))

    # music: beats (training features' beat column), RMS, per clip over the first T frames; per piece for the wrong music
    feats, rms = [], []
    for c in clips:
        y = af.load_audio(c['wav'])
        T0 = min(len(pickle.load(open(f'{AE.RAW}/motions/{c["name"]}.pkl', 'rb'))['smpl_poses']) // 2, 600)
        f = af.extract_baseline(y[:(T0 + 1) * af.HOP_LENGTH], start_bpm=nominal_bpm(c['piece']))[:T0]
        feats.append(f)
        rms.append(af.rms_envelope(y, 600))
    beats = [np.flatnonzero(f[:c['T'], 34] > 0) for f, c in zip(feats, clips)]
    # wrong music = the full piece recording (the per-sequence wavs of the EDGE release are cut to the dance and can be shorter than T)
    from aist_common import DATA
    pm_rms, pm_beats = {}, {}
    for p in pieces:
        y = af.load_audio(os.path.join(DATA, 'music', p + '.wav'))
        pm_rms[p] = af.rms_envelope(y, 600)
        pm_beats[p] = np.flatnonzero(af.extract_baseline(y[:601 * af.HOP_LENGTH], start_bpm=nominal_bpm(p))[:600, 34] > 0)
    bpm = {p: nominal_bpm(p) for p in pieces}
    genre = {p: next(c['genre'] for c in clips if c['piece'] == p) for p in pieces}
    wrong = {}
    for p in pieces:
        oth = [o for o in pieces if o != p]
        key = lambda o: (tempo_dist(bpm[o], bpm[p]) > 0.03, tempo_dist(bpm[o], bpm[p]) <= 0.03 and genre[o] != genre[p],  # noqa: E731
                         tempo_dist(bpm[o], bpm[p]))
        wrong[p] = sorted(oth, key=key)[:3]
    print('wrong pieces', wrong, flush=True)

    gtk, gtm = AE.gt_features()
    cls = None
    from aist_genre_cls import embed, load_model
    cls = load_model()

    def real_joints():
        from vis import SMPLSkeleton
        sk = SMPLSkeleton(device='cpu')
        out = []
        for c in clips:
            d = pickle.load(open(f'{AE.RAW}/motions/{c["name"]}.pkl', 'rb'))
            with torch.no_grad():
                kp = sk.forward(torch.from_numpy(d['smpl_poses'][:2 * c['T']]).float().reshape(1, -1, 24, 3),
                                torch.from_numpy(d['smpl_trans'][:2 * c['T']] / d['smpl_scaling']).float()[None]).squeeze(0).numpy()
            out.append(kp[::2][None, :c['T']])
        return out

    sets = {'Real dance': real_joints()}
    for spec in a.models:
        k, d = spec.split('=')
        sets[LABELS.get(k, k)] = [np.load(f'{d}/{n}.npy') for n in names]

    def windows(j):
        return [j[s:s + 150] for s in range(0, len(j) - 150 + 1, 75)]

    real_w = np.stack([w for j in sets['Real dance'] for w in windows(j[0])])
    real_s = embed(cls, torch.from_numpy(real_w)).numpy()

    rng = np.random.default_rng(0)
    res, per = {}, {}
    for label, mot in sets.items():
        S = mot[0].shape[0]
        seeds = []
        for s in range(S):
            m = [x[s] for x in mot]
            r = AE.score([up60(j) for j in m], [2 * b for b in beats], gtk, gtm)
            ws = np.stack([w for j in m for w in windows(j)])
            fs = embed(cls, torch.from_numpy(ws)).numpy()
            gs, fs_n = AE.normalize(real_s, fs)
            r['FID_s'] = float(AE.calc_fid(fs_n, gs))
            r['DIV_s'] = float(AE.calculate_avg_distance(fs_n))
            # loud / lock deltas per clip
            ld, lk = np.zeros(N), np.zeros(N)
            for i, c in enumerate(clips):
                T, q = c['T'], qom(m[i])
                qs = gaussian_filter1d(q, 15)
                own = np.corrcoef(gaussian_filter1d(rms[i][:T].astype(float), 15), qs)[0, 1]  # clip wavs are >= T frames (checked)
                wr = [np.corrcoef(gaussian_filter1d(pm_rms[w][:T].astype(float), 15), qs)[0, 1] for w in wrong[c['piece']]]
                ld[i] = own - np.mean(wr)
                iv = interval(beats[i]) if len(beats[i]) >= 3 else np.nan
                o = lock_r(q, iv, beats[i], trim)
                wl = [lock_r(q, iv, pm_beats[w][pm_beats[w] < T], trim) for w in wrong[c['piece']]]
                lk[i] = o - np.nanmean(wl)
            r['_loud_clip'], r['_lock_clip'] = ld, lk
            seeds.append(r)
        unit = lambda v: np.array([np.nanmean(v[piece == p]) for p in pieces])  # noqa: E731
        loud_u = unit(np.mean([r['_loud_clip'] for r in seeds], 0))
        lock_u = unit(np.mean([r['_lock_clip'] for r in seeds], 0))
        row = {k: float(np.mean([r[k] for r in seeds])) for k in ('FID_k', 'FID_g', 'FID_s', 'Div_k', 'Div_g', 'DIV_s', 'BAS')}
        row['std'] = {k: float(np.std([r[k] for r in seeds])) for k in ('FID_k', 'FID_g', 'FID_s')}
        row['Loud'], row['Loud_p'], row['Loud_pos'] = float(loud_u.mean()), signflip(loud_u, rng), f'{int((loud_u > 0).sum())}/{len(loud_u)}'
        row['Lock'], row['Lock_p'], row['Lock_pos'] = float(np.nanmean(lock_u)), signflip(lock_u, rng), f'{int((lock_u > 0).sum())}/{len(lock_u)}'
        res[label] = row
        per[label] = {'loud_piece': dict(zip(pieces, map(float, loud_u))), 'lock_piece': dict(zip(pieces, map(float, lock_u)))}
        print(label, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != 'std'}, flush=True)

    json.dump({'results': res, 'per_piece': per, 'n_clips': N, 'pieces': pieces, 'wrong': wrong, 'trim_s': a.trim},
              open(a.out + '.json', 'w'), indent=1)
    cols = [('FID_k', 'FID_k ↓', False), ('FID_g', 'FID_g ↓', False), ('FID_s', 'FID_s ↓', False), ('Div_k', 'DIV_k ↑', True),
            ('Div_g', 'DIV_g ↑', True), ('DIV_s', 'DIV_s ↑', True), ('BAS', 'BAS ↑', True), ('Lock', 'Lock ↑', True), ('Loud', 'Loud ↑', True)]
    mod = [k for k in res if k != 'Real dance']
    lines = ['| Model | ' + ' | '.join(c[1] for c in cols) + ' |', '|---|' + '---:|' * len(cols)]
    for k in res:
        cells = []
        for key, _, hi in cols:
            v = res[k][key]
            if k == 'Real dance' and key == 'FID_s':
                cells.append('–')
                continue
            s = f'{v:.3f}' if key in ('BAS', 'Lock', 'Loud') else f'{v:.2f}'
            if k in mod:
                vals = sorted(res[m][key] for m in mod)
                best = vals[-1] if hi else vals[0]
                second = vals[-2] if hi else vals[1] if len(vals) > 1 else vals[0]
                s = f'**{s}**' if v == best else f'<u>{s}</u>' if v == second and len(mod) > 2 else s
            cells.append(s)
        lines.append(f'| {k} | ' + ' | '.join(cells) + ' |')
    open(a.out + '.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
