"""Lodge's Foot Skating Ratio (FSR) for every model of the comparison tables, FineDance test and Motorica (2026-11-06).

Port of the official code, li-ronghui/LODGE metric/foot_skating.py (calc_foot_skating_ratio), to our Z-up SMPL joint
arrays (30 fps, metres), keeping its rules exactly:
  * ground: the toe heights of the FIRST frame; if the two toes differ by < 2 cm their mean, else the lower one
    (set_on_ground_139);
  * a foot is in contact when its ankle is <= 8 cm and its toe <= 5 cm above that ground;
  * per-frame velocity of the ankle and toe (frame t+1 - t); its "tangent" magnitude is |mean(v_x, v_horizontal2)|
    (the official code averages the two horizontal components, it does not take the norm);
  * a contact frame slides when the ankle or the toe has tangent magnitude > 0.01 m / frame (0.3 m/s);
  * ratio = sliding frames / contact frames, per foot; per clip; FSR = mean over clips of (left + right) / 2.
Lodge reports it in % (FineDance GT 6.22 %, Lodge 2.76 %, EDGE 20.04 %, Bailando 18.76 %, its Table 1) and MACE's
Table 2 copies those numbers as .062-style fractions. Here: %.

Joints: SMPL ankles 7 / 8, toes 10 / 11 (models; real FineDance via motion_quality_extra.real_joints, Z-up). Real
Motorica: BVH LeftFoot / RightFoot (ankle) and LeftToeBase / RightToeBase, Y-up -> heights along y; its joint placement
differs from SMPL, so its contact thresholds are not strictly comparable (reported, not ranked).
Clips: FineDance 84 slices x all samples (deterministic models: one); Motorica 289 slices x 3 samples (one for
deterministic models).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/fsr_table.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
OUT = os.path.join(HERE, 'results', 'fsr_table.json')
FD = {'real': None, 'edge': 'mace_test_edge', 'lodge': 'mace_test_lodge', 'bailando': 'mace_test_bailando',
      'mega': 'mace_test_mega', 'released': 'mace_test', 'ours': 'mace_test_mask2s_guide10_s0.003'}
MOT = {'real': None, 'edge': 'edge', 'lodge': 'lodge', 'bailando': 'bailando', 'mega': 'mega', 'released': 'mace',
       'ours': 'mace_local2s_guide_s0.003'}
SMPL_FEET = (7, 8, 10, 11)       # l ankle, r ankle, l toe, r toe
BODY_FEET = (15, 19, 16, 20)     # BVH LeftFoot, RightFoot, LeftToeBase, RightToeBase


def fsr(j, feet, up):
    """j (T, J, 3); returns (left_ratio, right_ratio) as in calc_foot_skating_ratio."""
    j = np.asarray(j, np.float64)
    hor = [a for a in range(3) if a != up]
    la, ra, lt, rt = feet
    l0, r0 = j[0, lt, up], j[0, rt, up]
    ground = (l0 + r0) / 2 if abs(l0 - r0) < 0.02 else min(l0, r0)
    h = j[..., up] - ground
    vel = np.zeros_like(j)
    vel[:-1] = j[1:] - j[:-1]
    out = []
    for ank, toe in ((la, lt), (ra, rt)):
        fc = (h[:, ank] <= 0.08) & (h[:, toe] <= 0.05)
        tang = np.abs(vel[:, [ank, toe]][:, :, hor].mean(-1))          # (T, 2): |mean of the two horizontal comps|
        slide = (tang > 0.01).any(-1) & fc
        out.append(slide.sum() / fc.sum() if fc.sum() else np.nan)
    return out


def summarize(rows):
    r = np.array(rows, float)  # (clips, 2)
    return {'FSR_pct': float(np.nanmean(np.nanmean(r, 0)) * 100), 'left_pct': float(np.nanmean(r[:, 0]) * 100),
            'right_pct': float(np.nanmean(r[:, 1]) * 100), 'n_clips': len(r),
            'clips_without_contact': int(np.isnan(r).any(1).sum())}


def samples(J, i, s_max):
    x = np.asarray(J[i, :s_max], np.float32)
    return x[:1] if len(x) > 1 and np.array_equal(x[0], x[1]) else x


def main():
    import argparse
    from motion_quality_extra import real_joints
    from motorica_common import body_joints
    ap = argparse.ArgumentParser()
    ap.add_argument('--fd', nargs='*', help='label=dir (under data/) FineDance models to score instead of FD')
    ap.add_argument('--mot', nargs='*', help='label=dir (under data/motorica) Motorica models instead of MOT')
    ap.add_argument('--seeds', type=int, default=None, help='score only the first n samples per slice')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    global FD, MOT, OUT
    if args.fd is not None or args.mot is not None:
        FD = dict(x.split('=', 1) for x in (args.fd or []))
        MOT = dict(x.split('=', 1) for x in (args.mot or []))
    OUT = OUT.replace('.json', f'{args.tag}.json')
    out = {'finedance': {}, 'motorica': {}}
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    for lab, d in FD.items():
        if d is None:
            rows = [fsr(j, SMPL_FEET, 2) for j in real_joints(names)]
        else:
            J = np.load(os.path.join(HERE, 'data', d, 'joints.npy'), mmap_mode='r')
            rows = [fsr(j, SMPL_FEET, 2) for i in range(len(names)) for j in samples(J, i, args.seeds or J.shape[1])]
        out['finedance'][lab] = summarize(rows)
        print('finedance', lab, out['finedance'][lab], flush=True)

    if not MOT:
        json.dump(out, open(OUT, 'w'), indent=1)
        return
    mot = os.path.join(HERE, 'data', 'motorica')
    meta = json.load(open(os.path.join(mot, 'mace', 'slices.json')))
    keep = [k for k, (t, o) in enumerate(meta['slices']) if o % 1020 == 0]
    slices = [tuple(meta['slices'][k]) for k in keep]
    for lab, d in MOT.items():
        if d is None:
            rows, cache = [], {}
            for t, o in slices:
                if t not in cache:
                    cache = {t: body_joints(t)}
                rows.append(fsr(cache[t][o:o + 1020], BODY_FEET, 1))
        else:
            J = np.load(os.path.join(mot, d, 'joints.npy'), mmap_mode='r')
            rws = range(len(slices)) if J.shape[0] == len(slices) else keep
            rows = [fsr(j, SMPL_FEET, 2) for r in rws for j in samples(J, r, args.seeds or 3)]
        out['motorica'][lab] = summarize(rows)
        print('motorica', lab, out['motorica'][lab], flush=True)
    json.dump(out, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
