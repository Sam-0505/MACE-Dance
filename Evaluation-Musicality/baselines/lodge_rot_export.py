"""Lodge: axis-angle poses + root translation for the 84 FineDance test slices, seed 0 (2026-11-10).

Reads the stored raw Lodge outputs (finedance_work/raw/{i}_s{s}.npy: 1024 x [4 contact, 3 root, 22 x 6D]), applies the same
root continuity fix as lodge_gen_finedance.stage_joints (the one behind the stored mace_test_lodge joints), keeps the first
1020 frames, converts the 22 body 6D rotations to axis-angle (+ zero rotations for hand joints 22/23, as the FK does) in
Lodge's own FineDance Y-up frame, and checks that FK of the raw arrays reproduces the stored joints and that the
axis-angle round trip reproduces the rotation matrices.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/baselines/lodge_rot_export.py --seed 0

Writes Evaluation-Musicality/data/rot_export/lodge_seed<k>.npz: poses (84, 1020, 72), trans (84, 1020, 3) (Y-up).
"""
import argparse
import os
import sys

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix, rotation_6d_to_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lodge_gen_finedance import N_CLIPS, N_FRAMES, root_continuous, smpl_fk  # noqa: E402

RAW = '/home/sam0505/Documents/Code/dance_baselines/LODGE/finedance_work/raw'
DATA = os.path.join(os.path.dirname(HERE), 'data')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    stored = np.load(os.path.join(DATA, 'mace_test_lodge', 'joints.npy'), mmap_mode='r')
    poses, trans, jerr, rerr = [], [], 0.0, 0.0
    for i in range(N_CLIPS):
        d = root_continuous(np.load(os.path.join(RAW, f'{i:03d}_s{a.seed}.npy')))[:N_FRAMES]
        raw = torch.from_numpy(d).double()[None]
        r6 = raw[:, :, 7:].reshape(1, N_FRAMES, 22, 6)
        root = raw[:, :, 4:7]
        jerr = max(jerr, float(np.abs(smpl_fk(r6, root).numpy() - np.asarray(stored[i, a.seed], np.float32)).max()))
        m = rotation_6d_to_matrix(r6)
        aa = matrix_to_axis_angle(m)
        rerr = max(rerr, float((axis_angle_to_matrix(aa) - m).abs().max()))
        q = torch.cat([aa, torch.zeros(1, N_FRAMES, 2, 3, dtype=aa.dtype)], dim=2).reshape(N_FRAMES, 72)
        poses.append(q.float().numpy()); trans.append(root[0].float().numpy())
    print(f'FK vs stored joints max |err| {jerr:.4f} m (float16 storage); axis-angle round trip max |err| {rerr:.2e}')
    assert jerr < 0.01 and rerr < 1e-5
    out = os.path.join(DATA, 'rot_export', f'lodge_seed{a.seed}.npz')
    np.savez(out, poses=np.stack(poses), trans=np.stack(trans))
    print('wrote', out, np.stack(poses).shape, 'trans mean/std', np.stack(trans).mean(), np.stack(trans).std())


if __name__ == '__main__':
    main()
