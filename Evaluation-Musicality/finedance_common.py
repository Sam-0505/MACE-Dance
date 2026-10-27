"""Real-dance joints from the public FineDance release (data/finedance/finedance/, authors' Google Drive, 7.7 h).

Format (FineDance data/code/pre_motion.py): motion/<song>.npy is (T, 315) at 30 fps = root position (3) +
52 joints x 6D rotation (pytorch3d convention, decoded with rotation_6d_to_matrix).
The 24-joint SMPL pose used by MACE's test files = release joints 0-21 plus 25 and 40 (first joint of each hand);
checked on all 19 test songs (2026-10-27): 13 songs give identical FK joints (< 1.2 cm). In the other 5 songs
(012, 036, 037, 109, 143) the shipped test files have single-frame root-orientation glitches (joint speeds up to
28.6 m/s); the release is smooth there.
"""
import os

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.join(HERE, 'data', 'finedance', 'finedance')
SMPL24_FROM_SMPLH52 = list(range(22)) + [25, 40]
SLICE_LEN, SLICE_STRIDE = 1020, 510              # MACE test slices: 34 s windows every 17 s


def release_pose(song):
    """(q (T, 72) axis-angle, pos (T, 3)) in the layout of Expert-Motion/data/test/motions/<song>.pkl."""
    m = np.load(os.path.join(RELEASE, 'motion', song + '.npy'), allow_pickle=False).astype(np.float64)
    mat = rotation_6d_to_matrix(torch.from_numpy(m[:, 3:]).reshape(len(m), 52, 6))[:, SMPL24_FROM_SMPLH52]
    return matrix_to_axis_angle(mat).reshape(len(m), 72).numpy(), m[:, :3]


def release_joints(song, smpl, device):
    """(T, 24, 3) joints built exactly like eval_mace.fk builds the test-file joints."""
    from eval_mace import fk
    q, pos = release_pose(song)
    return fk(smpl, [q], [pos], device, from_pred=False)[0]
