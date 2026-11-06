"""Fix the up axis of the stored Lodge joint arrays (2026-11-06).

lodge_gen_{motorica,finedance}.py applied MACE's align_axis heading rotation (+90 deg about z) directly to Lodge's
Y-up FK output, so the stored joints had up = -x. The generation scripts are fixed (Y-up -> Z-up first); this applies
the same correction to the already generated arrays instead of regenerating them:
    stored s = Rz90(F) = (-y, x, z)  ->  correct Rz90(YupToZup(F)) = (z, x, y) = (s2, s1, -s0)
Speed-based metrics (Lock, Loud, Perc, BAS) are unchanged by this rotation; PFC, FID_k, Div_k and the TMD 2D view are not.
The original array is kept as joints_upx_bug.npy. Idempotent: an array whose pelvis->head already points along +z
is skipped.

    Expert-Motion/.venv/bin/python Evaluation-Musicality/baselines/fix_lodge_axes.py
"""
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DIRS = ['motorica/lodge', 'motorica/lodge_official', 'motorica/lodge_seamfix', 'mace_test_lodge',
        'mace_test_lodge_seamfix']


def up_axis(j):
    d = (j[:4, 0, :, 15].astype(np.float32) - j[:4, 0, :, 0].astype(np.float32)).reshape(-1, 3).mean(0)
    return d


for d in DIRS:
    p = os.path.join(DATA, d, 'joints.npy')
    if not os.path.exists(p):
        print('missing', p)
        continue
    j = np.load(p)
    u = up_axis(j)
    if u[2] > 0.4:
        print(f'{d}: already z-up (pelvis->head {np.round(u, 3)}), skipped')
        continue
    assert u[0] < -0.4, (d, u)
    os.replace(p, os.path.join(DATA, d, 'joints_upx_bug.npy'))
    f = np.stack([j[..., 2], j[..., 1], -j[..., 0]], -1).astype(j.dtype)
    np.save(p, f)
    h = f[:2, 0, :, 0, 2].astype(np.float32) - f[:2, 0, :, [7, 8, 10, 11], 2].astype(np.float32).min(0)
    print(f'{d}: {j.shape} fixed, pelvis->head {np.round(up_axis(f), 3)}, pelvis above lowest foot median '
          f'{np.median(h):.2f} m')
