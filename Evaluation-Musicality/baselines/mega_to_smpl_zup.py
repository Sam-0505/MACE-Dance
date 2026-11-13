"""MEGADance FineDance test slices -> SMPL axis-angle poses + root translation in the renderer's frame (2026-11-13).

Input: data/rot_export/mega_native.npz (mega_rot_export.py): trans (84, 1020, 3) and p6 (84, 1020, 144) = 24 joints x 6D,
MEGADance's own representation in the FineDance Y-up frame. The stored joints (data/mace_test_mega/joints.npy) were made
by gen_mace_eval.to_mace_axes: Y-up -> Z-up ((x, y, z) -> (x, -z, y)) then align_axis (+90 deg about z). The same fixed
rotation Q = Rz(90) Rx(90) is applied to the root rotation (left-multiplied) and to the root translation; the other
joints are local rotations and stay unchanged. Output matches export_rot_all.py: poses (84, 1020, 72), trans (84, 1020, 3).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/baselines/mega_to_smpl_zup.py
"""
import os

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, rotation_6d_to_matrix

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'rot_export')


def q_matrix():
    rx = torch.tensor([[1., 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float64)  # (x, y, z) -> (x, -z, y)
    rz = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)  # +90 deg about z: (x, y) -> (-y, x)
    return rz @ rx


def main():
    z = np.load(os.path.join(DATA, 'mega_native.npz'))
    p6 = torch.from_numpy(z['p6']).double().reshape(84, 1020, 24, 6)
    tr = torch.from_numpy(z['trans']).double()
    R = rotation_6d_to_matrix(p6)
    Q = q_matrix()
    R[:, :, 0] = Q @ R[:, :, 0]
    aa = matrix_to_axis_angle(R)
    assert float((axis_angle_to_matrix(aa) - R).abs().max()) < 1e-6
    t = tr @ Q.T
    out = os.path.join(DATA, 'mega_seed0.npz')
    np.savez(out, poses=aa.reshape(84, 1020, 72).float().numpy(), trans=t.float().numpy(),
             max_err_m=float(z['max_err_m']))
    print('wrote', out, aa.shape, 'trans mean', t.mean((0, 1)).numpy())


if __name__ == '__main__':
    main()
