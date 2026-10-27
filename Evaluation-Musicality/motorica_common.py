"""Motorica Dance dataset helpers: take list, BVH forward kinematics, music pieces.

Facts (checked 2026-10-27; LDA paper arXiv 2211.09707, Transflower arXiv 2106.13871, dataset README, the files):
  * data/motorica/motorica_dance_dataset/{bvh,wav}: 121 BVH takes, 74 with a WAV of the same stem (48 kHz mono).
  * BVH at 120 fps, 51 joints (30 of them fingers), cm, Y up; root channels X/Y/Z position + Z/X/Y rotation,
    other joints Z/X/Y rotation. Root positions are absolute (root OFFSET ~ first-frame position; not added).
  * Sync is undocumented. Motion and audio lengths agree to < 0.1 s in 59/74 pairs, so frame 0 = song time 0.
  * Improvised vs choreographed is not stated in either paper.
Motion here: the 21 body joints (fingers dropped), metres, 30 fps (every 4th frame).
"""
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, 'data', 'motorica', 'motorica_dance_dataset')
BODY = ['Hips', 'Spine', 'Spine1', 'Neck', 'Head',
        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
        'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']
STRIDE = 4                                       # 120 -> 30 fps
# MACE (FineDance) style label per Motorica style; Krump has no FineDance genre -> 'Urban' (as for AIST++),
# Charleston and Tap -> 'Jazz'
STYLE_TO_MACE = {'gLH': 'Hiphop', 'gPO': 'Popping', 'gKR': 'Urban', 'gLO': 'Locking', 'gJZ': 'Jazz', 'gCH': 'Jazz',
                 'gTP': 'Jazz'}


def bvh_path(take):
    return os.path.join(ROOT, 'bvh', take + '.bvh')


def wav_path(take):
    return os.path.join(ROOT, 'wav', take + '.wav')


def takes_with_audio():
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(ROOT, 'wav', '*.wav'))
                  if os.path.exists(bvh_path(os.path.basename(f)[:-4])))


def style(take):
    return take.split('_')[1]


def read_bvh(path):
    """Returns names, parents, offsets (J, 3), channel lists, frame time, data (F, C)."""
    with open(path) as f:
        text = f.read()
    head, motion = text.split('MOTION', 1)
    names, parents, offsets, channels = [], [], [], []
    stack, tok, i, end_site = [], head.split(), 0, False
    while i < len(tok):
        t = tok[i]
        if t in ('ROOT', 'JOINT'):
            names.append(tok[i + 1]); parents.append(stack[-1] if stack else -1)
            offsets.append(None); channels.append([]); i += 2
        elif t == 'End':
            end_site = True; i += 2
        elif t == '{':
            stack.append(-1 if end_site else len(names) - 1); i += 1
        elif t == '}':
            stack.pop(); end_site = False; i += 1
        elif t == 'OFFSET':
            if stack[-1] != -1:
                offsets[stack[-1]] = np.array(tok[i + 1:i + 4], dtype=np.float64)
            i += 4
        elif t == 'CHANNELS':
            n = int(tok[i + 1]); channels[-1] = tok[i + 2:i + 2 + n]; i += 2 + n
        else:
            i += 1
    lines = motion.strip().split('\n')
    n_frames = int(lines[0].split(':')[1])
    frame_time = float(lines[1].split(':')[1])
    data = np.array(' '.join(lines[2:]).split(), dtype=np.float64).reshape(n_frames, -1)
    return names, parents, np.stack(offsets), channels, frame_time, data


def body_joints(take):
    """(T, 21, 3) world positions of BODY joints in metres at 30 fps."""
    names, parents, offsets, channels, frame_time, data = read_bvh(bvh_path(take))
    assert abs(frame_time - 1 / 120) < 1e-5, frame_time
    data = data[::STRIDE]
    T, col = len(data), 0
    rot, pos = [None] * len(names), [None] * len(names)
    for j, ch in enumerate(channels):
        idx = {c: col + k for k, c in enumerate(ch)}
        col += len(ch)
        rch = [c for c in ch if c.endswith('rotation')]
        local = Rotation.from_euler(''.join(c[0] for c in rch), data[:, [idx[c] for c in rch]], degrees=True)
        if parents[j] == -1:
            rot[j] = local
            pos[j] = data[:, [idx['Xposition'], idx['Yposition'], idx['Zposition']]]
        else:
            p = parents[j]
            rot[j] = rot[p] * local
            pos[j] = pos[p] + rot[p].apply(np.broadcast_to(offsets[j], (T, 3)))
    return np.stack([pos[names.index(b)] for b in BODY], axis=1) / 100.0


def n_frames_30(take):
    with open(bvh_path(take)) as f:
        for line in f:
            if line.startswith('Frames:'):
                return (int(line.split(':')[1]) + STRIDE - 1) // STRIDE


MAX_LEN_DIFF = 1.0                               # same rule as the FineDance replication
SLICE_LEN, SLICE_STRIDE = 1020, 510              # 34 s every 17 s, as on the FineDance test set


def pool():
    """Takes with audio whose motion and audio lengths agree to within MAX_LEN_DIFF s and that hold at least one
    34 s slice (same rules as the FineDance replication)."""
    import soundfile as sf
    return [t for t in takes_with_audio()
            if abs(n_frames_30(t) / 30 - sf.info(wav_path(t)).duration) <= MAX_LEN_DIFF and slice_starts(t)]


def slice_starts(take):
    import soundfile as sf
    n = min(n_frames_30(take), int(sf.info(wav_path(take)).duration * 30))
    return list(range(0, n - SLICE_LEN + 1, SLICE_STRIDE))
