"""Is the music-motion coupling measurable inside one 8 s training window, on the fast and the slow scale?

Fixed before running (2026-11-02). The Motion Expert trains on 240-frame (8 s) slices, the evaluation uses 34 s.
This checks the plan's four pairs (treble, perc, loud -> body energy; perc -> upper-body acceleration) at
s = 2 and s = 15 on the real training motion, built the way the loss will see it: cached training pose ->
unnormalize -> FK (world coordinates). Music: data/dataset_backups/music_signals_train.npz.
Pool: training slices of songs in the FineDance release pool (finedance_release_loud.json; duets merged into
pieces). Wrong music: one random training slice from each of the piece's 3 tempo-first matches (a match with no
training slices is dropped and the remaining ones are reused to fill 3 draws). Both signals are
smoothed inside the window (Gaussian, reflect); 'trim' also drops 2s frames at each edge. Zero-lag Pearson.
Reported per pair x scale: Delta_real over pieces (one-sided sign-flip p), and the per-window SNR =
mean / SD over slices of (r_own - mean r_wrong), i.e. how much of one window's correlation is signal.
Reading rule, treble -> energy: a scale is usable for the loss if Delta_real > 0 with p < .05. If both are, start
with slow (the plan's choice) unless its per-window SNR is below half the fast scale's; then start with fast.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/train_window_check.py
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from dataset.quaternion import ax_from_6v  # noqa: E402
from vis import SMPLSkeleton  # noqa: E402
import metrics as M  # noqa: E402
from multicomponent import SEED, UPPER, smooth, boot_ci, signflip_p  # noqa: E402
from finedance_common import release_joints  # noqa: E402

OUT = os.path.join(HERE, 'results', 'train_window_check.json')
DATASET = 'data/dataset_backups/train_tensor_dataset.pkl'
SIGNALS = 'data/dataset_backups/music_signals_train.npz'
SCALES = {'fast': 2, 'slow': 15}
PAIRS = {'treble|energy': ('treble', 'energy'), 'perc|energy': ('perc', 'energy'),
         'perc|uacc': ('perc', 'uacc'), 'loud|energy': ('loud', 'energy')}


def training_joints(ds, idx, smpl, device, chunk=512):
    """(len(idx), 240, 24, 3) joints from the cached normalized pose, as the musicality loss will build them."""
    out = []
    for s in range(0, len(idx), chunk):
        x = ds.normalizer.unnormalize(torch.as_tensor(ds.data['pose'][idx[s:s + chunk]]).clone()).to(device)
        _, x = torch.split(x, (4, x.shape[2] - 4), dim=2)
        b, t, _ = x.shape
        q = ax_from_6v(x[:, :, 3:].reshape(b, t, 24, 6))
        out.append(smpl.forward(q, x[:, :, :3]).cpu().numpy().astype(np.float64))
    return np.concatenate(out)


def motion_signals(j):
    e = np.stack([M.motion_energy(x) for x in j])
    a = np.linalg.norm(j[:, 2:, UPPER] - 2 * j[:, 1:-1, UPPER] + j[:, :-2, UPPER], axis=3).mean(2)
    return {'energy': e, 'uacc': np.pad(a, ((0, 0), (1, 1)), mode='edge')}


def rowcorr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-30)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(device)
    rel = json.load(open(os.path.join(HERE, 'results', 'finedance_release_loud.json')))
    piece_of = {s: p for p, ss in rel['pieces'].items() for s in ss}
    ds = pickle.load(open(DATASET, 'rb'))
    sig = np.load(SIGNALS)
    names = list(sig['names'])
    song = np.array([n.split('_slice')[0] for n in names])
    pool = np.flatnonzero(np.isin(song, list(piece_of)))
    print(f'{len(pool)} of {len(names)} training slices in the release pool, '
          f'{len(set(song[pool]))} songs, {len({piece_of[s] for s in song[pool]})} pieces', flush=True)

    joints = training_joints(ds, pool, smpl, device)
    motion = motion_signals(joints)

    # FK path check: body energy from the cached training pose vs from the release motion, on 3 slices
    check = []
    for i in (0, len(pool) // 2, len(pool) - 1):
        n = names[pool[i]]
        s, k = n.split('_slice')
        rj = release_joints(s, smpl, device)[int(k) * 30:int(k) * 30 + 240]
        er = M.motion_energy(rj)
        check.append((n, float(np.corrcoef(er, motion['energy'][i])[0, 1]), float(motion['energy'][i].mean() / er.mean())))
    print('FK check (slice, r, energy ratio cached/release):', check, flush=True)

    rng = np.random.default_rng(SEED)
    slices_of = {}
    for i, s in enumerate(song[pool]):
        slices_of.setdefault(piece_of[s], []).append(i)
    match = {p: [w for w in rel['matching']['tempo_first'][p] if w in slices_of] for p in slices_of}
    print('pieces with fewer than 3 matches in the training set:',
          {p: len(m) for p, m in match.items() if len(m) < 3})
    wrong = np.array([[rng.choice(slices_of[w]) for w in (match[piece_of[s]] * 3)[:3]] for s in song[pool]])
    units = np.array([piece_of[s] for s in song[pool]])
    pieces = sorted(set(units))
    music = {c: sig[c][pool].astype(np.float64) for c in ('loud', 'perc', 'treble')}
    silent = music['loud'].std(1) < 1e-8
    print(f'silent slices excluded: {silent.sum()}')

    results = {'n_slices': int((~silent).sum()), 'n_pieces': len(pieces), 'fk_check': check, 'pairs': {}}
    for pname, (mk, dk) in PAIRS.items():
        for sname, sg in SCALES.items():
            ms, ds_ = smooth(music[mk], sg), smooth(motion[dk], sg)
            for variant in ('full', 'trim'):
                sl = slice(None) if variant == 'full' else slice(2 * sg, -2 * sg)
                own = rowcorr(ms[:, sl], ds_[:, sl])
                wr = np.stack([rowcorr(ms[wrong[:, w], sl], ds_[:, sl]) for w in range(wrong.shape[1])], 1).mean(1)
                keep = ~silent & ~silent[wrong].any(1)
                d = own - wr
                dp = np.array([d[keep & (units == p)].mean() for p in pieces])
                key = f'{pname}|{sname}|{variant}'
                results['pairs'][key] = st = {
                    'r_own': float(own[keep].mean()), 'r_wrong': float(wr[keep].mean()),
                    'delta': float(dp.mean()), 'ci': boot_ci(dp, rng),
                    'p': signflip_p(dp, rng, 'greater'), 'pieces_pos': int((dp > 0).sum()),
                    'sd_wrong_window': float(wr[keep].std()), 'snr_window': float(d[keep].mean() / d[keep].std())}
                print(f"{key:28s} own {st['r_own']:.3f} wrong {st['r_wrong']:.3f} delta {st['delta']:+.3f} "
                      f"[{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}] p {st['p']:.4f} ({st['pieces_pos']}/{len(pieces)}) "
                      f"SD(r_wrong) {st['sd_wrong_window']:.3f} SNR {st['snr_window']:.3f}")

    t = {s: results['pairs'][f'treble|energy|{s}|full'] for s in SCALES}
    usable = {s: t[s]['delta'] > 0 and t[s]['p'] < .05 for s in SCALES}
    if usable['slow'] and (not usable['fast'] or t['slow']['snr_window'] >= 0.5 * t['fast']['snr_window']):
        start = 'slow'
    elif usable['fast']:
        start = 'fast'
    else:
        start = 'neither'
    results['rule'] = {'usable': usable, 'start_with': start}
    print('rule (treble -> energy, full window):', results['rule'])
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
