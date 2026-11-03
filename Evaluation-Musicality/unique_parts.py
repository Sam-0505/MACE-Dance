"""Loudness, percussion, treble: does each have a unique part, on the fast and the slow scale?

Fixed before running (2026-10-27). Datasets, pools, matching and slice pairing as in perc_distinct.py.
Music per slice: loud = RMS, perc = RMS of the HPSS percussive part, treble = 6.4-12.8 kHz flux, each smoothed at
s = 2 (fast) and s = 15 (slow). Unique part of a component = its signal with the other two regressed out within the
slice at the same scale (OLS with intercept) = its unique share in a joint linear model of all three.
Motion at the same scale: energy = whole-body motion energy (world coordinates, the loudness analyses' measure),
body = travel-free mean speed of the bass_treble.py leg and head/hand joints.
Reported: plain and unique Delta for 3 components x 2 scales x 2 motions. BH-FDR over the 12 unique tests within
each dataset (H1, one-sided sign-flip p).
Reading rule: a component has a unique part on a scale if its unique Delta_real has q < .05 for at least one motion
on BOTH large datasets (release, motorica).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/unique_parts.py [test|release|motorica ...]
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from model.metric import SMPLSkeleton  # noqa: E402
import metrics as M  # noqa: E402
from multicomponent import smooth  # noqa: E402
from finedance_common import SLICE_LEN  # noqa: E402
import perc_distinct as PD  # noqa: E402
import bass_treble as BT  # noqa: E402
from accent_overlap import full_music, resid  # noqa: E402

OUT = os.path.join(HERE, 'results', 'unique_parts.json')
SCALES = {'fast': PD.FAST, 'slow': PD.SLOW}
COMPONENTS = ['loud', 'perc', 'treble']
MOTIONS = ['energy', 'body']
PAIRS = {f'{c}{u}_{s}|{d}_{s}': (f'{c}{u}_{s}', f'{d}_{s}')
         for s in SCALES for c in COMPONENTS for u in ('', '_u') for d in MOTIONS}


def music_views(m, o):
    v = {}
    for s, sig in SCALES.items():
        x = {c: smooth(np.asarray(m[c][o:o + SLICE_LEN], dtype=np.float64), sig) for c in COMPONENTS}
        assert all(len(a) == SLICE_LEN for a in x.values())
        for c in COMPONENTS:
            v[f'{c}_{s}'] = x[c]
            v[f'{c}_u_{s}'] = resid(x[c], *[x[k] for k in COMPONENTS if k != c])
    return v


def motion_views(j, src):
    J = BT.JOINTS['motorica' if src == 'motorica' else 'smpl']
    ax, _ = BT.UP[src]
    j = np.asarray(j[:SLICE_LEN], dtype=np.float64)
    horiz = [a for a in range(3) if a != ax]
    tf = j.copy()
    tf[:, :, horiz] -= j[:, J['pelvis']:J['pelvis'] + 1, horiz]
    idx = J['legs'] + J['hands']
    sp = np.pad(np.linalg.norm(tf[1:, idx] - tf[:-1, idx], axis=2).mean(1), (0, 1), mode='edge')
    e = M.motion_energy(j)
    return {f'{k}_{s}': smooth(x, sig) for s, sig in SCALES.items() for k, x in (('energy', e), ('body', sp))}


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def main():
    which = [a for a in sys.argv[1:] if a in ('test', 'release', 'motorica')] or ['test', 'release', 'motorica']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(device)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in which:
        print(name, flush=True)
        data = PD.load_dataset(name, smpl, device, full_music, music_views, motion_views)
        stats, _ = PD.score_pairs(*data, PAIRS)
        uniq = [k for k in PAIRS if '_u_' in k]
        for who, key in (('real', 'H1_perm_p'), ('mace', 'H2_perm_p')):
            if key in stats[uniq[0]]:
                for k, q in zip(uniq, bh([stats[k][key] for k in uniq])):
                    stats[k][f'q_{who}'] = float(q)
        print('  unique parts (BH q, real):')
        for k in uniq:
            st = stats[k]
            extra = f" | dM {st['d_mace']:+.3f} q {st['q_mace']:.3f}" if 'd_mace' in st else ''
            print(f"    {k:28s} dR {st['d_real']:+.3f} q {st['q_real']:.4f}{extra}")
        results[name] = stats
        with open(OUT, 'w') as f:
            json.dump(results, f, indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
