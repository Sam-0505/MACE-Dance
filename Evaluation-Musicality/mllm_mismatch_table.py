"""MLLM original-vs-mismatched evaluation with paired significance (released MACE-Dance vs MMC-Dance).

Per clip and model: matched score (own music, v3 videos) and mismatched score (mean over the wrong songs scored for that clip; same wrong audio for
both models). Delta = matched - mismatched. Tests, all paired over clips (statistics as in validity_table.py / multicomponent.py: 95% bootstrap CI,
sign-flip permutation, Wilcoxon):
  A  per model: Delta > 0 (does the judge notice the wrong music?), one-sided
  B  ours - released on the matched score, two-sided
  C  ours - released on Delta (which model depends more on its own music?), two-sided
Reads results/mllm_judge.csv, data/mllm_finedance_v3/mapping.csv, data/mllm_finedance_v3_mismatch/mapping.csv.
    cd Expert-Motion && .venv/bin/python ../Evaluation-Musicality/mllm_mismatch_table.py
Writes results/mllm_mismatch_{table.md,summary.json}."""
import collections
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from multicomponent import SEED, boot_ci, wil, signflip_p  # noqa: E402

MODELS = ['released', 'ours']
NAME = {'released': 'Released MACE-Dance', 'ours': 'MMC-Dance (ours)'}
orig = {r['video_id']: r for r in csv.DictReader(open(os.path.join(HERE, 'data/mllm_finedance_v3/mapping.csv')))}
mism = {r['video_id']: r for r in csv.DictReader(open(os.path.join(HERE, 'data/mllm_finedance_v3_mismatch/mapping.csv')))}
score = {r['video_id']: int(r['va_alignment']) for r in csv.DictReader(open(os.path.join(HERE, 'results/mllm_judge.csv')))
         if r['status'] == 'ok' and float(r['requested_judge_fps'] or 0) == 24.0}
M0, MM = collections.defaultdict(dict), collections.defaultdict(lambda: collections.defaultdict(list))
for v, r in orig.items():
    if v in score and r['method'] in MODELS:
        M0[r['method']][r['slice']] = score[v]
for v, r in mism.items():
    if v in score:
        MM[r['base_method']][r['slice']].append(score[v])
clips = sorted(c for c in MM['released'] if c in MM['ours'] and all(c in M0[m] for m in MODELS))
K = max(len(MM[m][c]) for m in MODELS for c in clips)
mat = {m: np.array([M0[m][c] for c in clips], float) for m in MODELS}
mis = {m: np.array([np.mean(MM[m][c]) for c in clips], float) for m in MODELS}
dlt = {m: mat[m] - mis[m] for m in MODELS}
rng = np.random.default_rng(SEED)


def stats(d, alt):
    d = np.asarray(d, float)
    return {'mean': round(float(d.mean()), 3), 'ci95': [round(x, 3) for x in boot_ci(d, rng)],
            'p_perm': round(signflip_p(d, rng, 'greater' if alt == 'greater' else 'two-sided'), 4),
            'p_wilcoxon': round(wil(d, alt), 4), 'n_pos_tie_neg': [int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())]}


S = {'n_clips': len(clips), 'mismatched_per_clip': K, 'clips': clips,
     'A': {m: {'matched': round(float(mat[m].mean()), 2), 'mismatched': round(float(mis[m].mean()), 2), **stats(dlt[m], 'greater')} for m in MODELS},
     'B_ours_minus_released_matched': {'released': round(float(mat['released'].mean()), 2), 'ours': round(float(mat['ours'].mean()), 2),
                                       **stats(mat['ours'] - mat['released'], 'two-sided')},
     'C_ours_minus_released_delta': {'released_delta': round(float(dlt['released'].mean()), 3), 'ours_delta': round(float(dlt['ours'].mean()), 3),
                                     **stats(dlt['ours'] - dlt['released'], 'two-sided')}}
json.dump(S, open(os.path.join(HERE, 'results/mllm_mismatch_summary.json'), 'w'), indent=1)
f = lambda s: f"{s['mean']:+.2f} [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]"  # noqa: E731
md = f"""# MLLM VA alignment: matched vs mismatched music, FineDance ({len(clips)} clips, {K} wrong song(s) per clip)

Same judge, prompt and 24 FPS as the main table. The mismatched video is the matched v3 video with only the audio replaced by a tempo-matched wrong
song (same wrong audio for both models). Delta = matched - mismatched; CI = 95% bootstrap over clips; p = sign-flip permutation (Wilcoxon in the JSON).

| Model | Matched | Mismatched | Delta [95% CI] | clips Delta>0 / =0 / <0 | p (one-sided, Delta>0) |
|---|---:|---:|---|---|---:|
""" + ''.join(f"| {NAME[m]} | {S['A'][m]['matched']:.2f} | {S['A'][m]['mismatched']:.2f} | {f(S['A'][m])} | "
              f"{'/'.join(map(str, S['A'][m]['n_pos_tie_neg']))} | {S['A'][m]['p_perm']:.4f} |\n" for m in MODELS) + f"""
MMC-Dance minus MACE-Dance (paired over the same {len(clips)} clips, two-sided):

| Quantity | Released | Ours | Ours - released [95% CI] | p |
|---|---:|---:|---|---:|
| Matched score | {S['B_ours_minus_released_matched']['released']:.2f} | {S['B_ours_minus_released_matched']['ours']:.2f} | {f(S['B_ours_minus_released_matched'])} | {S['B_ours_minus_released_matched']['p_perm']:.4f} |
| Delta (matched - mismatched) | {S['C_ours_minus_released_delta']['released_delta']:+.2f} | {S['C_ours_minus_released_delta']['ours_delta']:+.2f} | {f(S['C_ours_minus_released_delta'])} | {S['C_ours_minus_released_delta']['p_perm']:.4f} |
"""
open(os.path.join(HERE, 'results/mllm_mismatch_table.md'), 'w').write(md)
print(md)
