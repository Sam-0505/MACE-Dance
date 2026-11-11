"""Part 3: style label effect. AIST++ checkpoint, 48 Motorica windows and 48 AIST++ windows, every window given each of the six AIST++
style labels in turn (Breaking, Popping, Locking, Urban, Hiphop, Jazz), speeds in m/s. Also Motorica with its own labels and all MFCC
coefficients replaced by the AIST++ mean (the part-1 'mfcc' ablation), to see the two effects together.
    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/aist_transfer_probe3.py
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

files = sorted(glob.glob(os.path.join(HERE, '..', 'Expert-Motion', 'data', 'aist', 'train', 'baseline_feats', '*.npy')))
rng = np.random.default_rng(0)
sel = rng.choice(len(files), 1500, replace=False)
A = np.stack([np.load(files[i]) for i in sel])
agen = [GENRE_TO_STYLE[os.path.basename(files[i])[:3]] for i in sel]
M, st = [], []
for t in pool()[:48]:
    y = af.load_audio(wav_path(t))
    o = slice_starts(t)[0]
    M.append(af.extract_baseline(y[o * af.HOP_LENGTH:(o + 151) * af.HOP_LENGTH])[:150])
    st.append(STYLE_TO_MACE[style(t)])
M = np.stack(M)
sm = MaceSampler('runs/aist/mace/weights/train-100.pt')
out = {}
print('Motorica label counts', {s: st.count(s) for s in sorted(set(st))})
for lab in ['Breaking', 'Popping', 'Locking', 'Urban', 'Hiphop', 'Jazz']:
    for nm, F in (('Motorica', M), ('AIST++', A[:48])):
        s, r = speeds(sm.sample(F, [lab] * 48, seed=0))
        out[f'{nm} / {lab}'] = {'speed': s, 'root_rel': r}
        print(f'{nm:9s} all labelled {lab:9s} speed {s:5.2f} root-rel {r:5.2f}', flush=True)
for nm, F, S in (('Motorica own labels', M, st), ('AIST++ own labels', A[:48], agen[:48])):
    s, r = speeds(sm.sample(F, S, seed=0))
    out[nm] = {'speed': s, 'root_rel': r}
    print(f'{nm:30s} speed {s:5.2f} root-rel {r:5.2f}')
X = M.copy()
X[:, :, 1:21] = A[:, :, 1:21].mean((0, 1))   # per-coefficient AIST++ mean
s, r = speeds(sm.sample(X, st, seed=0))
out['Motorica own labels, all MFCC -> AIST mean'] = {'speed': s, 'root_rel': r}
print(f'Motorica own labels, MFCC -> AIST mean  speed {s:5.2f} root-rel {r:5.2f}')
json.dump(out, open(os.path.join(HERE, 'results', 'aist_transfer_probe3.json'), 'w'), indent=1)
