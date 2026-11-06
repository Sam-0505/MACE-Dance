"""Glitch-repaired variants of the baseline dances on Motorica, to check that model artefacts do not drive the beat
scores (2026-11-06). lodge_seamfix: frames k-3..k+3 around Lodge's local-window joins (k = 256, 512, 768; one-frame
jerk at ~2x speed) linearly interpolated. edge_despike: 3-frame median filter over time (EDGE's rare arm flips).

    cd Expert-Motion && .venv/bin/python ../Evaluation-Musicality/baselines/robust_variants.py
"""
import os
import shutil

import numpy as np
from scipy.ndimage import median_filter

MOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'motorica')

L = np.load(f'{MOT}/lodge/joints.npy').astype(np.float32)
for k in (256, 512, 768):
    a, b = k - 4, k + 4
    w = np.linspace(0, 1, b - a + 1)[1:-1].reshape(1, 1, -1, 1, 1)
    L[:, :, a + 1:b] = L[:, :, a:a + 1] * (1 - w) + L[:, :, b:b + 1] * w
E = median_filter(np.load(f'{MOT}/edge/joints.npy').astype(np.float32), size=(1, 1, 3, 1, 1), mode='nearest')
for name, X in (('lodge_seamfix', L), ('edge_despike', E)):
    os.makedirs(f'{MOT}/{name}', exist_ok=True)
    np.save(f'{MOT}/{name}/joints.npy', X.astype(np.float16))
    shutil.copy(f'{MOT}/slices_wav/slices.json', f'{MOT}/{name}/slices.json')
