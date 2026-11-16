"""Mismatched-music versions of the v3 FineDance videos for the MLLM judge (original vs mismatched control).

The video stream of each v3 video is copied unchanged; only the audio track is replaced by a 34 s slice of a tempo-matched WRONG song, chosen
exactly like the validity tables (robustness_loud.matches 'tempo_first': full-song BPM with half/double allowed, 3 % tolerance, same genre first
among ties; one seeded random slice of each wrong song; 3 wrong songs per clip). Both models of a clip get the same wrong audio (paired).
Writes data/mllm_finedance_v3_mismatch/{<hash>.mp4, mapping.csv} and data/mllm_mismatch_plan.json (all 3 wrong slices of every clip).

    cd Expert-Motion && .venv/bin/python ../Evaluation-Musicality/make_mismatched_mllm.py --k 1 --clips subset
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from multicomponent import SEED, GT_DIR  # noqa: E402
from robustness_loud import song_bpm, matches  # noqa: E402

ROOT = os.path.dirname(HERE)
FF = os.path.join(ROOT, 'Expert-Appearance', '.venv', 'lib', 'python3.10', 'site-packages', 'imageio_ffmpeg', 'binaries', 'ffmpeg-linux-x86_64-v7.0.2')
WAV = os.path.join(ROOT, 'Expert-Motion', 'data', 'test', 'wavs_sliced')
V3, OUT = os.path.join(HERE, 'data', 'mllm_finedance_v3'), os.path.join(HERE, 'data', 'mllm_finedance_v3_mismatch')
SALT = 'mmc-mllm-v3-mismatch'

ap = argparse.ArgumentParser()
ap.add_argument('--k', type=int, nargs='+', default=[1], help='which of the 3 wrong songs (1..3) to build')
ap.add_argument('--clips', choices=['subset', 'all'], default='subset', help='subset = one seeded clip per song (19)')
ap.add_argument('--methods', nargs='+', default=['released', 'ours'])
a = ap.parse_args()

names = [str(x) for x in np.load(os.path.join(HERE, 'data/rot_export/oneshot_seed0.npz'), allow_pickle=True)['names']]
songs = np.array([n.split('_')[0] for n in names])
order = sorted(set(songs))
genre = {s: json.load(open(os.path.join(GT_DIR, 'label', s + '.json')))['style2'] for s in order}
bpm = {s: song_bpm(s) for s in order}
mt = matches(order, genre, bpm, 'tempo_first')
rng = np.random.default_rng(SEED)
pair = [[int(rng.choice(np.flatnonzero(songs == o))) for o in mt[songs[i]]] for i in range(len(names))]
plan = {names[i]: [names[j] for j in pair[i]] for i in range(len(names))}
json.dump({'wrong_slices': plan, 'wrong_songs': mt, 'bpm': bpm}, open(os.path.join(HERE, 'data', 'mllm_mismatch_plan.json'), 'w'), indent=1)

pick = np.random.default_rng(SEED)
clips = [names[int(pick.choice(np.flatnonzero(songs == s)))] for s in order] if a.clips == 'subset' else names
v3 = {(r['slice'], r['method']): r for r in csv.DictReader(open(os.path.join(V3, 'mapping.csv')))}
os.makedirs(OUT, exist_ok=True)
mp_path, rows = os.path.join(OUT, 'mapping.csv'), []
have = {r['video_id'] for r in csv.DictReader(open(mp_path))} if os.path.exists(mp_path) else set()
for c in clips:
    for k in a.k:
        wrong = plan[c][k - 1]
        for m in a.methods:
            vid = hashlib.sha1(f'{SALT}|{c}|{m}|{k}'.encode()).hexdigest()[:10]
            if vid in have:
                continue
            src = os.path.join(V3, v3[(c, m)]['video_id'] + '.mp4')
            out = os.path.join(OUT, vid + '.mp4')
            subprocess.run([FF, '-y', '-loglevel', 'error', '-i', src, '-i', os.path.join(WAV, wrong + '.wav'), '-map', '0:v', '-map', '1:a',
                            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', '-t', '34.0', out], check=True)
            rows.append({'video_id': vid, 'slice': c, 'method': f'{m}_mismatch{k}', 'base_method': m, 'k': k, 'wrong_slice': wrong,
                         'orig_video_id': v3[(c, m)]['video_id'], 'path': out, 'fps': 30})
if rows:
    new = not os.path.exists(mp_path)
    with open(mp_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
print(f'{len(clips)} clips, built {len(rows)} mismatched videos in {OUT}')
for c in clips[:19]:
    print(c, '->', [plan[c][k - 1] for k in a.k], 'BPM', round(bpm[c.split('_')[0]]), '->', [round(bpm[plan[c][k - 1].split('_')[0]]) for k in a.k])
