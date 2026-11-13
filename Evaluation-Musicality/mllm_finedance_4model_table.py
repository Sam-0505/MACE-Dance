"""FineDance TMD-Bench VA table: released MACE, ours, MEGADance, Lodge. One mean raw 1-5 score per model over the clips where all
four have a valid score (plus the 84-clip released/ours mean). Reads results/mllm_judge.csv and data/mllm_finedance_v2/mapping.csv.
Writes results/mllm_finedance_4model_{summary.json,table.md}."""
import collections
import csv
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
M = ['released', 'ours', 'mega', 'lodge']
NAME = {'released': 'Released MACE-Dance', 'ours': 'MMC-Dance (ours)', 'mega': 'MEGADance', 'lodge': 'Lodge'}
mp = {r['video_id']: r for r in csv.DictReader(open(os.path.join(HERE, 'data/mllm_finedance_v2/mapping.csv')))}
rows = [r for r in csv.DictReader(open(os.path.join(HERE, 'results/mllm_judge.csv'))) if r['video_id'] in mp]
by = collections.defaultdict(dict)
for r in rows:
    if r['status'] == 'ok':
        by[mp[r['video_id']]['slice']][mp[r['video_id']]['method']] = int(r['va_alignment'])
rendered = collections.defaultdict(set)
for r in mp.values():
    rendered[r['method']].add(r['slice'])
full = sorted(c for c in by if all(m in by[c] for m in M))
S = {'clips_with_all_four_scores': len(full), 'per_clip_mean': {m: round(st.mean(by[c][m] for c in full), 2) for m in M},
     'clips_rendered': {m: len(rendered[m]) for m in M}, 'clips_scored': {m: sum(m in d for d in by.values()) for m in M},
     'released_ours_all_clips': {m: round(st.mean(d[m] for d in by.values() if 'released' in d and 'ours' in d), 2) for m in ('released', 'ours')},
     'requests': {m: dict(collections.Counter(r['status'].split(':')[0] for r in rows if mp[r['video_id']]['method'] == m)) for m in M},
     'effective_fps': dict(collections.Counter(r['confirmed_effective_judge_fps'] for r in rows if r['status'] == 'ok'))}
json.dump(S, open(os.path.join(HERE, 'results/mllm_finedance_4model_summary.json'), 'w'), indent=1)
md = f"""# TMD-Bench Visual Audio Alignment (MLLM judge), FineDance

Judge: Gemini 2.5 Pro (`protected.gemini-2.5-pro`, TAMU AI API), TMD-Bench VA prompt (Appendix E, Figure 6) unmodified, one fresh request per
video, 24 FPS, 34.0 s clips, 1280x720 mesh renders with the song's audio, same avatar/background/lighting/audio for every model; the camera of
each clip is fitted to the released+ours pair. One sample (seed 0) per clip. Raw 1-5 integers, no rescaling.

| Model | Mean VA Alignment (1-5), {len(full)} clips |
|---|---:|
""" + ''.join(f"| {NAME[m]} | {S['per_clip_mean'][m]:.2f} |\n" for m in M) + f"""
Clips used: {len(full)} (only clips where all four models have a valid score). On all 84 clips, released MACE {S['released_ours_all_clips']['released']:.2f}, ours {S['released_ours_all_clips']['ours']:.2f}.
Rendered / scored clips per model: {', '.join(f"{NAME[m]} {S['clips_rendered'][m]}/{S['clips_scored'][m]}" for m in M)}.
Effective judge FPS from tokens: {S['effective_fps']}.
MEGADance and Lodge were rendered only where they stay in the shared camera's frame (< 1% of frames out of view); MEGADance travels most, so it is
missing on the clips where it leaves the frame. Lodge's world places the dancer 1.26 m higher than MACE's, so it is shifted rigidly (constant
floor shift, start on the released clip's start position); its motion is unchanged. Single-sample 1-5 scores are noisy.
"""
open(os.path.join(HERE, 'results/mllm_finedance_4model_table.md'), 'w').write(md)
print(md)
