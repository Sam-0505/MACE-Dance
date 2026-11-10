"""Best long windows (10 / 15 / 20 / 30 s) for the loudness and beat examples (2026-11-07).
Same per-window scores as select_window.py (loudness: Pearson r of whole-body energy with loudness; beat: PLV of the body-speed
accents to the beat grid), now over longer windows; start every 1 s. Prints the best start by ours - released and the scores.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/select_long_window.py
"""
import json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.getcwd())
import export_smpl_final as E  # noqa: E402
import render_fix_compare as R  # noqa: E402
from eval_mace import fk  # noqa: E402
from model.metric import SMPLSkeleton  # noqa: E402

names = json.load(open(os.path.join(R.ONESHOT, 'names.json')))['names']
seeds = json.load(open(os.path.join(E.OUT, 'seeds.json')))
smpl = SMPLSkeleton('cuda')
res = {}
for key in ('loudness', 'beat'):
    n = E.CLIPS[key]; i, k = names.index(n), seeds[key]
    m = pickle.load(open(f'data/test/motions_sliced/{n}.pkl', 'rb'))
    gt = fk(smpl, [m['q']], [m['pos']], 'cuda', from_pred=False)[0]
    rel = np.asarray(np.load(os.path.join(R.ONESHOT, 'joints.npy'), mmap_mode='r')[i, k], dtype=np.float32)
    ours = np.asarray(np.load(os.path.join(E.OURS_DIR, 'joints.npy'), mmap_mode='r')[i, k], dtype=np.float32)
    ex = E.MAKE[key](n, gt, rel[None], ours[None]); T = ex['T']
    V = {'real': gt, 'released': rel, 'ours': ours}
    if key == 'beat':
        from tmd_beat_phase import phases, plv
        beats = ex['beats']; acc = {v: ex['accents'](j) for v, j in V.items()}
        def sc(v, s, W):
            a = acc[v]; return plv(phases(a[(a >= s) & (a < s + W)], beats[(beats >= s - 60) & (beats < s + W + 60)]))
    else:
        cur = {v: np.asarray(ex['curve'](j)) for v, j in V.items()}; mus = np.asarray(ex['music'])
        def sc(v, s, W): return float(np.corrcoef(cur[v][s:s + W], mus[s:s + W])[0, 1])
    print(f'== {key} {n} seed {k} ({ex["metric"]}), clip {T / 30:.0f} s; whole clip: ' +
          '  '.join(f'{v} {ex["score"](j[:T]):+.2f}' for v, j in V.items()))
    for L in (10, 15, 20, 30):
        W = L * 30; rows = []
        for s in range(0, T - W + 1, 30):
            r = {v: sc(v, s, W) for v in V}
            if not any(np.isnan(x) for x in r.values()): rows.append((s, r))
        rows.sort(key=lambda x: -(x[1]['ours'] - x[1]['released']))
        for s, r in rows[:2]:
            print(f'  {L:2d} s  start {s / 30:5.1f} s   real {r["real"]:+.2f}  released {r["released"]:+.2f}  ours {r["ours"]:+.2f}')
