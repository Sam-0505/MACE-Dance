"""Rank test clips with the relative-speed (local contrast) measure of the beat phase grid (2026-11-09).

Per clip and seed: s(t) = total speed (mean over the 24 joints of the root-relative speed, sigma 1 frame); ratio(t) = s(t) / its mean over
one beat either side. profile(phi) = mean of ratio at (beat + phi x beat interval) over the beats of the clip without 4 s at both ends,
phi = j/16. Scores:
  on-beat   = profile(0)                                  (what the 'at the beat' row of the grid shows; 1 = the dancer's own average)
  own-phase = depth = (max - min) / 2 of the profile      (a constant offset is free, like lock_models.py)
  delta     = depth(own beats) - mean depth(3 wrong-song grids rescaled to the own tempo, as lock_models.py)
Ranking per clip: median over seeds of (ours - released) and the share of ours seeds above the released median.
Writes results/rel_speed_clip_rank.json.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/rank_relspeed_clips.py [--seeds 10]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import lock_models as LM  # noqa: E402

OURS, REL, REAL = 'Ours critic v5 + gain', 'MACE released', 'Real dance'
PH = np.arange(16) / 16


def total_speed(j):
    j = np.asarray(j, np.float32)
    r = j - j[:, :1]
    return gaussian_filter1d(np.linalg.norm(np.gradient(r, axis=0) * LM.FPS, axis=-1), 1.0, axis=0).mean(1)


def profile(ratio, beats, iv, trim):
    out = []
    for p in PH:
        g = np.round(np.asarray(beats, float) + p * iv).astype(int)
        g = g[(g >= trim) & (g < len(ratio) - trim)]
        out.append(ratio[g].mean() if len(g) else np.nan)
    return np.array(out)


def rescale(b, iv):
    b = np.asarray(b, float)
    return b[0] + (b - b[0]) * (iv / LM.interval(b))


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
        w = max(3, int(round(2 * iv[i])))
        row = {'bpm': float(60 * LM.FPS / iv[i])}
        for label, mot in sets.items():
            on, depth, delta, peak = [], [], [], []
            for s in range(len(mot[i])):
                sp = total_speed(mot[i][s])
                ratio = sp / uniform_filter1d(sp, w, mode='nearest')
                pr = profile(ratio, beats[i], iv[i], trim)
                dp = (np.nanmax(pr) - np.nanmin(pr)) / 2
                wr = [(lambda q: (np.nanmax(q) - np.nanmin(q)) / 2)(profile(ratio, rescale(beats[j], iv[i]), iv[i], trim)) for j in pair[i]]
                on.append(float(pr[0])); depth.append(float(dp)); delta.append(float(dp - np.nanmean(wr))); peak.append(float(PH[np.nanargmax(pr)]))
            row[label] = {'on_beat': on, 'depth': depth, 'delta': delta, 'peak_phase': peak}
        res[n] = row
    json.dump(res, open(os.path.join(HERE, 'results', 'rel_speed_clip_rank.json'), 'w'))
    print('clips', len(res))

    def table(key, title):
        rows = []
        for n, r in res.items():
            o, m, g = np.array(r[OURS][key]), np.array(r[REL][key]), np.mean(r[REAL][key])
            rows.append((np.median(o) - np.median(m), (o > np.median(m)).mean(), np.median(o), np.median(m), g, n))
        print(f'\n{title}\n  median over clips: ours {np.median([x[2] for x in rows]):+.3f}  released {np.median([x[3] for x in rows]):+.3f}  real {np.median([x[4] for x in rows]):+.3f}; '
              f'ours > released in {100 * np.mean([x[0] > 0 for x in rows]):.0f}% of clips; clips with >= 8/10 ours seeds above released: {sum(x[1] >= 0.8 for x in rows)}')
        for x in sorted(rows, reverse=True)[:10]:
            print(f'  {x[5]:11s} gap {x[0]:+.3f}  seeds {100 * x[1]:3.0f}%  ours {x[2]:+.3f} released {x[3]:+.3f} real {x[4]:+.3f}')
    table('on_beat', 'on-beat relative speed (1 = own average; the grid row "at the beat")')
    table('delta', 'own-phase relative-speed depth minus the wrong-song level')


if __name__ == '__main__':
    main()
