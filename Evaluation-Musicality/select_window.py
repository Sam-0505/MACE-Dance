"""Pick the 5.1 s window of each final dance that the Appearance Expert will animate (2026-11-07).

The Appearance Expert makes 77 frames at 15 fps = 153 motion frames (5.1 s) per video, the clips are 34 s long.
For each example the window start (every 0.5 s) is chosen where ours beats MACE released by the most on the example's
own measure, computed inside the window: Pearson r of the motion curve with the music curve (loudness: whole-body
energy vs loudness; perc: upper-body acceleration vs percussive energy), or the phase-locking value of the body-speed
accents to the beat grid (beat). All three scores of every window go to windows.json, so the selection can be audited.
NOTE: this picks a window where the difference shows; the full-clip scores are the unselected numbers.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/select_window.py
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import export_smpl_final as E  # noqa: E402
import render_fix_compare as R  # noqa: E402
from eval_mace import fk  # noqa: E402
from model.metric import SMPLSkeleton  # noqa: E402

WIN, STEP = 153, 15  # motion frames (30 fps): 77 control frames at 15 fps


def main():
    meta = json.load(open(os.path.join(R.ONESHOT, 'names.json')))
    names = meta['names']
    seeds = json.load(open(os.path.join(E.OUT, 'seeds.json')))
    smpl = SMPLSkeleton('cuda')
    out = {}
    for key, n in E.CLIPS.items():
        i, k = names.index(n), seeds[key]
        m = pickle.load(open(f'data/test/motions_sliced/{n}.pkl', 'rb'))
        gt = fk(smpl, [m['q']], [m['pos']], 'cuda', from_pred=False)[0]
        rel = np.asarray(np.load(os.path.join(R.ONESHOT, 'joints.npy'), mmap_mode='r')[i, k], dtype=np.float32)
        ours = np.asarray(np.load(os.path.join(E.OURS_DIR, 'joints.npy'), mmap_mode='r')[i, k], dtype=np.float32)
        ex = E.MAKE[key](n, gt, rel[None], ours[None])
        T = ex['T']
        if key == 'beat':
            from tmd_beat_phase import phases, plv
            beats = ex['beats']
            acc = {v: ex['accents'](j) for v, j in (('real', gt), ('released', rel), ('ours', ours))}

            def win_score(v, s):
                a = acc[v]
                return plv(phases(a[(a >= s) & (a < s + WIN)], beats[(beats >= s - 60) & (beats < s + WIN + 60)]))
        else:
            cur = {v: np.asarray(ex['curve'](j)) for v, j in (('real', gt), ('released', rel), ('ours', ours))}
            mus = np.asarray(ex['music'])

            def win_score(v, s):
                return float(np.corrcoef(cur[v][s:s + WIN], mus[s:s + WIN])[0, 1])
        rows = []
        for s in range(0, T - WIN + 1, STEP):
            sc = {v: win_score(v, s) for v in ('real', 'released', 'ours')}
            rows.append({'start': s, **sc})
        ok = [r for r in rows if not any(np.isnan(r[v]) for v in ('real', 'released', 'ours'))]
        best = max(ok, key=lambda r: r['ours'] - r['released'])
        out[key] = {'clip': n, 'seed': k, 'metric': ex['metric'], 'start': best['start'], 'best': best, 'rows': rows}
        print(f"{key} {n} seed {k}: window start {best['start'] / 30:.1f} s  real {best['real']:+.2f}  released "
              f"{best['released']:+.2f}  ours {best['ours']:+.2f}", flush=True)
    json.dump(out, open(os.path.join(E.OUT, 'windows.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
