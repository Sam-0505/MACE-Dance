"""Part 2 of aist_transfer_probe.py: which MFCC coefficient drives the speed on Motorica? (2026-11-11)
Per-coefficient means (AIST++ train windows vs Motorica), then sampling with the AIST++ checkpoint on 48 Motorica windows where
(a) each MFCC coefficient in turn is replaced by the AIST++ mean, (b) MFCC 0 (overall level) is shifted by a constant so its mean matches
AIST++, (c) a dose-response: MFCC 0 of AIST++ windows shifted by -2, -1, +1, +2 AIST++ standard deviations.
    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/aist_transfer_probe2.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from aist_transfer_probe import speeds  # noqa: E402
from mace_sampler import MaceSampler  # noqa: E402
import audio_features as af  # noqa: E402
from motorica_common import STYLE_TO_MACE, pool, slice_starts, style, wav_path  # noqa: E402
from aist_common import GENRE_TO_STYLE  # noqa: E402

D = os.path.join(HERE, '..', 'Expert-Motion', 'data', 'aist', 'train', 'baseline_feats')
files = sorted(glob.glob(f'{D}/*.npy'))
rng = np.random.default_rng(0)
sel = rng.choice(len(files), 1500, replace=False)
A = np.stack([np.load(files[i]) for i in sel])
agen = [GENRE_TO_STYLE[os.path.basename(files[i])[:3]] for i in sel]
takes = pool()[:48]
M, st = [], []
for t in takes:
    y = af.load_audio(wav_path(t))
    o = slice_starts(t)[0]
    M.append(af.extract_baseline(y[o * af.HOP_LENGTH:(o + 151) * af.HOP_LENGTH])[:150])
    st.append(STYLE_TO_MACE[style(t)])
M = np.stack(M)
out = {'coef_means': {}, 'speed': {}}
print('MFCC coef: AIST mean (sd of window means) | Motorica mean (sd) | gap in AIST sds')
for k in range(20):
    am, mm = A[:, :, 1 + k].mean(1), M[:, :, 1 + k].mean(1)
    gap = (mm.mean() - am.mean()) / (am.std() + 1e-9)
    out['coef_means'][k] = {'aist': float(am.mean()), 'aist_sd': float(am.std()), 'mot': float(mm.mean()), 'mot_sd': float(mm.std()), 'gap_sd': float(gap)}
    if k < 6 or abs(gap) > 1.5:
        print(f'{k:2d}: {am.mean():8.2f} ({am.std():6.2f}) | {mm.mean():8.2f} ({mm.std():6.2f}) | {gap:+.2f}')
sm = MaceSampler('runs/aist/mace/weights/train-100.pt')


def run(name, feats, styles):
    s, r = speeds(sm.sample(feats, styles, seed=0))
    out['speed'][name] = {'speed': s, 'root_rel': r}
    print(f'{name:44s} speed {s:5.2f} root-rel {r:5.2f}', flush=True)


run('Motorica as is', M, st)
for k in range(20):
    X = M.copy()
    X[:, :, 1 + k] = A[:, :, 1 + k].mean()
    run(f'Motorica, MFCC{k} -> AIST mean', X, st)
X = M.copy()
X[:, :, 1] += A[:, :, 1].mean() - M[:, :, 1].mean()
run('Motorica, MFCC0 mean-shifted to AIST (const)', X, st)
ai = np.arange(48)
base = A[ai].copy()
run('AIST++ windows (own genre labels)', base, [agen[i] for i in ai])
sd = A[:, :, 1].mean(1).std()
for d in (-2, -1, 1, 2):
    X = base.copy()
    X[:, :, 1] += d * sd
    run(f'AIST++ windows, MFCC0 {d:+d} sd', X, [agen[i] for i in ai])
json.dump(out, open(os.path.join(HERE, 'results', 'aist_transfer_probe2.json'), 'w'), indent=1)
