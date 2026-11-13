"""Lodge (seed 0) axis-angle + root translation from its FineDance Y-up frame to the renderer's frame (2026-11-13).

Input data/rot_export/lodge_seed0.npz (lodge_rot_export.py; FK of these reproduces the stored mace_test_lodge joints). The stored
joints were mapped by lodge_gen_finedance.smpl_fk: Y-up -> Z-up ((x, y, z) -> (x, -z, y)) then align_axis (+90 deg about z).
The same fixed rotation Q = Rz(90) Rx(90) is left-multiplied onto the root rotation and applied to the root translation (as
mega_to_smpl_zup.py does for MEGADance). Output data/rot_export/lodge_zup_seed0.npz: poses (84, 1020, 72), trans (84, 1020, 3).

    cd Expert-Motion; .venv/bin/python ../Evaluation-Musicality/baselines/lodge_to_smpl_zup.py
"""
import os

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'rot_export')
rx = torch.tensor([[1., 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float64)
rz = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)
Q = rz @ rx
z = np.load(os.path.join(D, 'lodge_seed0.npz'))
p = torch.from_numpy(z['poses']).double().reshape(84, 1020, 24, 3)
R = axis_angle_to_matrix(p[:, :, 0])
p[:, :, 0] = matrix_to_axis_angle(Q @ R)
t = torch.from_numpy(z['trans']).double() @ Q.T
np.savez(os.path.join(D, 'lodge_zup_seed0.npz'), poses=p.reshape(84, 1020, 72).float().numpy(), trans=t.float().numpy())
print('wrote', os.path.join(D, 'lodge_zup_seed0.npz'))
