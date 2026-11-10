"""Per-clip beat phase locking (the metric behind the p / q values, lock_models.py) for ranking clips (2026-11-09).

For every FineDance test clip i and seed s: r_own = |mean_t exp(i(phi(t) - theta_own(t)))| of the band-passed QoM against the clip's own
beat phase (constant offset free), r_wrong = the same against the beats of the 3 tempo-matched wrong-song clips rescaled to the own
tempo, delta = r_own - r_wrong, offset = mean phase difference in beats. Same code, trim, bandwidth and pairing as lock_models.py.
Writes results/lock_clip_rank.json and prints the clips where ours (critic v5 + gain) is high and MACE released is not.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/rank_lock_clips.py [--seeds 10]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import lock_models as LM  # noqa: E402
from robustness_loud import motion_signals  # noqa: E402

OURS = 'Ours critic v5 + gain'
REL = 'MACE released'


def own_and_offset(q, iv, beats, trim):
    a = LM.analytic_at(q, 1.0 / iv)
    d = np.angle(a) - LM.beat_phase(beats, iv, 1.0)
    z = np.exp(1j * d)[trim:LM.T - trim].mean()
    return float(abs(z)), float(np.angle(z) / (2 * np.pi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    a = ap.parse_args()
    trim = int(round(4.0 * LM.FPS))
    units, order, beats, sets, pair, bpm = LM.load_finedance(a.seeds, [REL, OURS])
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    iv = [LM.interval(b) if len(b) >= 3 else np.nan for b in beats]
    res = {}
    for i, n in enumerate(names):
        if not np.isfinite(iv[i]):
            continue
        row = {'unit': str(units[i]), 'bpm': float(60 * LM.FPS / iv[i])}
        for label, mot in sets.items():
            r_o, off, r_w = [], [], []
            for s in range(len(mot[i])):
                q = motion_signals(mot[i][s])['speed']
                ro, of = own_and_offset(q, iv[i], beats[i], trim)
                rw = np.nanmean([LM.lock_r(q, iv[i], beats[j], 1.0, trim, True) for j in pair[i]])
                r_o.append(ro); off.append(of); r_w.append(float(rw))
            row[label] = {'r_own': r_o, 'r_wrong': r_w, 'offset': off}
        res[n] = row
    json.dump(res, open(os.path.join(HERE, 'results', 'lock_clip_rank.json'), 'w'))
    print('clips', len(res))
    rows = []
    for n, r in res.items():
        d = {k: np.array(r[k]['r_own']) - np.array(r[k]['r_wrong']) for k in r if isinstance(r[k], dict)}
        o, m, g = d[OURS], d[REL], d['Real dance'][0]
        rows.append((np.median(o) - np.median(m), (o > np.median(m)).mean(), np.median(o), np.median(m), g, n))
    print(f'mean delta over clips: real {np.mean([x[4] for x in rows]):+.3f} released {np.mean([x[3] for x in rows]):+.3f} ours {np.mean([x[2] for x in rows]):+.3f}')
    print('clips ranked by median over seeds of delta(ours) - delta(released); share of ours seeds above released median; real delta')
    for x in sorted(rows, reverse=True)[:12]:
        print(f'  {x[5]:11s} gap {x[0]:+.3f}  seeds {100 * x[1]:3.0f}%  ours {x[2]:+.3f} released {x[3]:+.3f} real {x[4]:+.3f}')
    print('clips with ours delta > released median in >= 8/10 seeds:', sum(x[1] >= 0.8 for x in rows), 'of', len(rows))


if __name__ == '__main__':
    main()
