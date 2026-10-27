"""Shared AIST++ helpers: sequence lists, segment lengths, tempo, joints.

Facts (checked 2026-10-27, see readme): motion frame 0 = music time 0; basic dances are 16 beats, advanced 64;
motion is 60 fps and is taken at 30 fps (every second frame); root = smpl_trans / smpl_scaling.
"""
import glob
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'aist')
MAX_T = 1020                     # 34 s at 30 fps: FineDance slice length, inside MACE's 40 s horizon
GENRES = ['gBR', 'gPO', 'gLO', 'gWA', 'gMH', 'gLH', 'gHO', 'gKR', 'gJS', 'gJB']
GENRE_NAMES = {'gBR': 'Break', 'gPO': 'Pop', 'gLO': 'Lock', 'gWA': 'Waack', 'gMH': 'Middle Hip-hop',
               'gLH': 'LA Hip-hop', 'gHO': 'House', 'gKR': 'Krump', 'gJS': 'Street Jazz', 'gJB': 'Ballet Jazz'}
# MACE (FineDance) style label per AIST genre; Waack, House and Krump have no FineDance genre -> 'Urban'
GENRE_TO_STYLE = {'gBR': 'Breaking', 'gPO': 'Popping', 'gLO': 'Locking', 'gWA': 'Urban', 'gMH': 'Hiphop',
                  'gLH': 'Hiphop', 'gHO': 'Urban', 'gKR': 'Urban', 'gJS': 'Jazz', 'gJB': 'Jazz'}


def nominal_bpm(piece):
    """ISMIR'19 §3.2: pieces 0-5 are 80-130 BPM in steps of 10; house is 110-135 in steps of 5."""
    k = int(piece[3])
    return 110 + 5 * k if piece[1:3] == 'HO' else 80 + 10 * k


def sequences():
    """{name: (piece, kind)} for public sequences not in the ignore list; kind is 'basic' or 'adv'."""
    ignore = set(open(os.path.join(DATA, 'ignore_list.txt')).read().split())
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, 'motions_npz', '*.npz'))):
        n = os.path.basename(f)[:-4]
        if n in ignore:
            continue
        p = n.split('_')
        out[n] = (p[4], 'basic' if p[1] == 'sBM' else 'adv')
    return out


def n_frames(name):
    return (len(np.load(os.path.join(DATA, 'motions_npz', name + '.npz'))['smpl_trans']) + 1) // 2


def segments():
    """{(piece, kind): T} with T = shortest real dance on that piece (30 fps), capped at MAX_T."""
    seg = {}
    for n, key in sequences().items():
        seg[key] = min(seg.get(key, MAX_T), n_frames(n))
    return seg


def joints(name, skel, device):
    """(T, 24, 3) SMPL-scale joints at 30 fps."""
    d = np.load(os.path.join(DATA, 'motions_npz', name + '.npz'))
    poses = torch.from_numpy(d['smpl_poses'][::2]).to(device).reshape(1, -1, 24, 3)
    root = torch.from_numpy(d['smpl_trans'][::2] / d['smpl_scaling']).to(device)[None]
    return skel.forward(poses, root).squeeze(0).cpu().numpy()
