"""Real dancer's steady on-beat stretches (2026-11-09), FineDance and Motorica.
A window of N=8 beats (stride 1) qualifies when the real dancer's accents (peaks of mean joint speed) number >= 8 and keep the same
place in the beat (R >= 0.75) within 0.1 beat of the beat (|mu| <= 0.1). A stretch = consecutive qualifying windows (length in
beats = N + run - 1). For each stretch the same beats are scored for MACE released and ours (all seeds): median R and the share of
seeds that also qualify, plus the real dancer's limb speed in the stretch (is there a visible hit?). Writes results/real_stretch_<dataset>.json.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/find_real_stretch.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import rank_beat_runs as RB  # noqa: E402

N, FPS = 8, 30


def qual(w):
    return [(not np.isnan(x[2])) and x[1] >= 8 and x[2] >= 0.75 and abs(x[3]) <= 0.1 for x in w]


def main():
    for name in ('finedance', 'motorica'):
        out = []
        for n, bpm, A, gt, one, ours in RB.clips(name):
            wr = RB.windows(gt, A, N, 1)
            q = qual(wr)
            if not any(q):
                continue
            wo = [RB.windows(j, A, N, 1) for j in one]
            wu = [RB.windows(j, A, N, 1) for j in ours]
            k = 0
            while k < len(q):
                if not q[k]:
                    k += 1
                    continue
                e = k
                while e + 1 < len(q) and q[e + 1]:
                    e += 1
                idx = range(k, e + 1)
                def gen(ws):
                    R = np.array([[w[i][2] for i in idx] for w in ws], float)
                    ok = np.array([[qual([w[i]])[0] for i in idx] for w in ws])
                    return float(np.nanmedian(R)), float(ok.mean())
                b0, b1 = wr[k][0], wr[e][0] + N
                seg = gt[A[b0]:A[min(b1, len(A) - 1)]]
                sp = float(np.linalg.norm(np.diff(seg, axis=0), axis=-1).mean() * FPS)
                out.append({'n': n, 'bpm': bpm, 'beat0': int(b0), 'beats': int(b1 - b0), 't0': float(A[b0] / FPS), 't1': float(A[min(b1, len(A) - 1)] / FPS),
                            'real_R': float(np.mean([wr[i][2] for i in idx])), 'real_mu': float(np.mean([wr[i][3] for i in idx])),
                            'one_R': gen(wo)[0], 'one_ok': gen(wo)[1], 'ours_R': gen(wu)[0], 'ours_ok': gen(wu)[1], 'limb_speed': sp})
                k = e + 1
        out.sort(key=lambda x: (-x['beats'], -x['real_R']))
        json.dump(out, open(os.path.join(HERE, 'results', f'real_stretch_{name}.json'), 'w'))
        print(f'\n=== {name}: {len(out)} stretches (real steady on the beat, >= {N} beats)')
        print(f'  {"clip":56s} {"bpm":>4s} {"t0-t1 s":>11s} {"beats":>5s} | real R mu | released R ok | ours R ok | body m/s')
        for x in out[:12]:
            print(f'  {x["n"]:56s} {x["bpm"]:4.0f} {x["t0"]:5.1f}-{x["t1"]:5.1f} {x["beats"]:5d} | {x["real_R"]:.2f} {x["real_mu"]:+.2f} | '
                  f'{x["one_R"]:.2f} {100 * x["one_ok"]:3.0f}% | {x["ours_R"]:.2f} {100 * x["ours_ok"]:3.0f}% | {x["limb_speed"]:.2f}')


if __name__ == '__main__':
    main()
