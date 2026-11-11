"""Part 4: does the conditioning strength explain the Motorica blow-up? MACE infers with pred = beta*cond + (1-beta)*uncond (model.infer_pred,
default beta 0.75; it is trained guidance-free by mixing the two with a random beta). Sweep beta at inference on 48 Motorica and 48 AIST++
windows with the AIST++ checkpoint (own style labels). Speeds in m/s.
    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/aist_transfer_probe4.py
"""
import functools
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
sel = rng.choice(len(files), 48, replace=False)
A = np.stack([np.load(files[i]) for i in sel])
ag = [GENRE_TO_STYLE[os.path.basename(files[i])[:3]] for i in sel]
M, st = [], []
for t in pool()[:48]:
    y = af.load_audio(wav_path(t))
    o = slice_starts(t)[0]
    M.append(af.extract_baseline(y[o * af.HOP_LENGTH:(o + 151) * af.HOP_LENGTH])[:150])
    st.append(STYLE_TO_MACE[style(t)])
M = np.stack(M)
sm = MaceSampler('runs/aist/mace/weights/train-100.pt')
net = sm.model.diffusion.model
orig = net.infer_pred
out = {}
print('beta   Motorica speed / root-rel    AIST++ speed / root-rel')
for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
    net.infer_pred = functools.partial(orig, beta=beta)
    sm_, rm_ = speeds(sm.sample(M, st, seed=0))
    sa_, ra_ = speeds(sm.sample(A, ag, seed=0))
    out[str(beta)] = {'motorica': [sm_, rm_], 'aist': [sa_, ra_]}
    print(f'{beta:4.2f}   {sm_:5.2f} / {rm_:5.2f}              {sa_:5.2f} / {ra_:5.2f}', flush=True)
json.dump(out, open(os.path.join(HERE, 'results', 'aist_transfer_probe4.json'), 'w'), indent=1)
