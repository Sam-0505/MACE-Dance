"""Showcase search (2026-11-21): FineDance test clips x seeds x 10 s windows where ours (critic v5 + gain) locks to the beat
clearly better than BOTH MACE-Dance released and FlowerDance, for side-by-side videos (render_pink_grid.py).

Lock r = the lock_models.py metric (QoM of the 24 joints, zero-phase band-pass at the beat frequency, 15 % width, Hilbert phase
minus beat phase, offset free), with the band-pass run on the whole clip and |mean exp(i d)| taken inside each window
(10 s, 1 s steps, inside the 4 s trim). Per window and model also the beat-wave amplitude (mean |analytic| / mean QoM) so a window
is not picked just because a baseline is quiet there (earlier amplitude selection effect, see beat-beads notes).

Two comparisons are written per window:
  same_seed   baselines at the same seed index as ours
  vs_typical  baselines at their MEDIAN r over the 10 seeds in that window (ours best case vs baseline typical; fairer to show)
A window qualifies when ours' beat-wave amplitude >= 0.8 x each baseline's. Score = ours r - max(baseline r).
Clip level (whole clip minus trim) r is reported too, plus the MLLM VA score (seed 0 only, results/mllm_judge.csv + v2 mapping).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/rank_showcase_windows.py

Writes Evaluation-Musicality/results/showcase_windows.json (+ _top.md).
"""
import collections
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import lock_models as LM  # noqa: E402
from robustness_loud import motion_signals  # noqa: E402

MODELS = {'ours': 'Ours critic v5 + gain', 'mace': 'MACE released', 'flower': 'FlowerDance'}
FPS, WIN, STEP, TRIM, SEEDS = 30, 300, 30, 120, 10


def phase_series(q, beats):
    """per-frame exp(i (phi - theta)) and the beat-wave amplitude (|analytic| / mean QoM) of one QoM signal."""
    iv = LM.interval(beats)
    a = LM.analytic_at(q, 1 / iv)
    d = np.angle(a) - LM.beat_phase(beats, iv, 1.0, True, len(q))
    return np.exp(1j * d), np.abs(a) / (q.mean() + 1e-9)


def main():
    units, order, beats, sets, pair, bpm = LM.load_finedance(SEEDS, list(MODELS.values()))
    names = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))['names']
    starts = np.arange(TRIM, LM.T - TRIM - WIN + 1, STEP)

    # MLLM VA (one sample per clip = seed 0)
    mp = {r['video_id']: r for r in csv.DictReader(open(os.path.join(HERE, 'data/mllm_finedance_v2/mapping.csv')))}
    meth = {'released': 'mace', 'ours': 'ours', 'flower': 'flower'}
    va = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(HERE, 'results/mllm_judge.csv'))):
        if r['video_id'] in mp and r['status'] == 'ok' and mp[r['video_id']]['method'] in meth:
            va[mp[r['video_id']]['slice']][meth[mp[r['video_id']]['method']]] = int(r['va_alignment'])

    clips, wins = [], []
    for i, n in enumerate(names):
        b = beats[i]
        real_z, real_amp = phase_series(motion_signals(sets['Real dance'][i][0])['speed'], b)
        Z, A = {}, {}
        for k, lab in MODELS.items():
            zs, am = zip(*[phase_series(motion_signals(j)['speed'], b) for j in sets[lab][i]])
            Z[k], A[k] = np.stack(zs), np.stack(am)  # (S, T)
        whole = {k: np.abs(Z[k][:, TRIM:-TRIM].mean(1)) for k in Z}  # (S,)
        clips.append({'clip': n, 'bpm': round(60 * FPS / LM.interval(b), 1), 'real_r': round(float(np.abs(real_z[TRIM:-TRIM].mean())), 3),
                      **{f'{k}_r_median': round(float(np.median(whole[k])), 3) for k in Z},
                      'ours_beats_both_seeds': int(sum(whole['ours'][s] > max(whole['mace'][s], whole['flower'][s]) for s in range(SEEDS))),
                      'mllm_va_seed0': va.get(n, {})})
        Wr = {k: np.stack([np.abs(Z[k][:, s0:s0 + WIN].mean(1)) for s0 in starts], 1) for k in Z}  # (S, W)
        Wa = {k: np.stack([A[k][:, s0:s0 + WIN].mean(1) for s0 in starts], 1) for k in Z}
        rr = [float(np.abs(real_z[s0:s0 + WIN].mean())) for s0 in starts]
        for s in range(SEEDS):
            for w, s0 in enumerate(starts):
                typ = {k: float(np.median(Wr[k][:, w])) for k in ('mace', 'flower')}
                typ_a = {k: float(np.median(Wa[k][:, w])) for k in ('mace', 'flower')}
                same_ok = Wa['ours'][s, w] >= 0.8 * max(Wa['mace'][s, w], Wa['flower'][s, w])
                typ_ok = Wa['ours'][s, w] >= 0.8 * max(typ_a.values())
                wins.append({'clip': n, 'seed': s, 'start_s': round(s0 / FPS, 2), 'ours_r': round(float(Wr['ours'][s, w]), 3),
                             'mace_r': round(float(Wr['mace'][s, w]), 3), 'flower_r': round(float(Wr['flower'][s, w]), 3),
                             'mace_r_typ': round(typ['mace'], 3), 'flower_r_typ': round(typ['flower'], 3), 'real_r': round(rr[w], 3),
                             'amp_ours_over_max_base': round(float(Wa['ours'][s, w] / max(Wa['mace'][s, w], Wa['flower'][s, w])), 2),
                             'same_seed_score': round(float(Wr['ours'][s, w] - max(Wr['mace'][s, w], Wr['flower'][s, w])), 3) if same_ok else None,
                             'vs_typical_score': round(float(Wr['ours'][s, w] - max(typ.values())), 3) if typ_ok else None})
        print(n, clips[-1]['ours_r_median'], clips[-1]['mace_r_median'], clips[-1]['flower_r_median'], flush=True)

    def best_per_clip(key):
        best = {}
        for w in wins:
            if w[key] is not None and (w['clip'] not in best or w[key] > best[w['clip']][key]):
                best[w['clip']] = w
        return sorted(best.values(), key=lambda w: -w[key])

    top_typ, top_same = best_per_clip('vs_typical_score'), best_per_clip('same_seed_score')
    allw = [w for w in wins if w['same_seed_score'] is not None]
    S = {'settings': {'window_s': WIN / FPS, 'step_s': STEP / FPS, 'trim_s': TRIM / FPS, 'seeds': SEEDS, 'amp_rule': 'ours >= 0.8 x baseline'},
         'share_windows_ours_r_ge_0.8': {k: round(float(np.mean([w[f'{k}_r'] >= 0.8 for w in wins])), 3) for k in MODELS},
         'share_windows_ours_beats_both_same_seed': round(float(np.mean([w['same_seed_score'] > 0 for w in allw])), 3),
         'clips': clips, 'top_vs_typical': top_typ[:25], 'top_same_seed': top_same[:25]}
    out = os.path.join(HERE, 'results', 'showcase_windows.json')
    json.dump(S, open(out, 'w'), indent=1)
    md = ['# Showcase windows: ours vs MACE-Dance and FlowerDance (10 s, offset-free lock r)', '',
          'Best window per clip, ours best seed vs each baseline\'s median over 10 seeds in the same window; ours beat-wave amplitude >= 0.8x baselines.', '',
          '| clip | seed | start s | ours r | MACE r (typ) | Flower r (typ) | real r | score | amp ratio | MLLM VA s0 ours/MACE/Flower |', '|---|---|---|---|---|---|---|---|---|---|']
    for w in top_typ[:15]:
        v = va.get(w['clip'], {})
        md.append(f"| {w['clip']} | {w['seed']} | {w['start_s']} | {w['ours_r']} | {w['mace_r_typ']} | {w['flower_r_typ']} | {w['real_r']} | "
                  f"{w['vs_typical_score']} | {w['amp_ours_over_max_base']} | {v.get('ours', '-')}/{v.get('mace', '-')}/{v.get('flower', '-')} |")
    open(out.replace('.json', '_top.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(json.dumps({k: S[k] for k in ('share_windows_ours_r_ge_0.8', 'share_windows_ours_beats_both_same_seed')}))


if __name__ == '__main__':
    main()
