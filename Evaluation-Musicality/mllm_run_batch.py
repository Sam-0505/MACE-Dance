"""Judge one batch of 10 not-yet-scored FineDance clips (both models = 20 videos) and print the running report.

Clips = the 84 test slices in export order, minus the 19 slice0 clips of the declared subset, taken 10 at a time
(batch 1 = the first 10 of the 65, ... batch 7 = the last 5). Waits until both videos of a clip are rendered
(mapping.csv rows are appended per finished pair), runs tamu_judge.py (24 FPS, 2 workers, shuffled), retries any failed
request ONCE unchanged, and reports the batch plus cumulative means.

    python mllm_run_batch.py 1
"""
import collections
import csv
import json
import os
import statistics as st
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, 'data', 'mllm_finedance_v2')
CSV = os.path.join(HERE, 'results', 'mllm_judge.csv')
BATCH = 10


def clip_list():
    names = [str(x) for x in np.load(os.path.join(HERE, 'data', 'rot_export', 'oneshot_seed0.npz'), allow_pickle=True)['names']]
    return [n for n in names if not n.endswith('_slice0')]


def mapping():
    return list(csv.DictReader(open(os.path.join(V2, 'mapping.csv'))))


def judge(videos):
    env = dict(os.environ, TAMUS_MODEL_ID=os.environ.get('TAMUS_MODEL_ID', 'protected.gemini-2.5-pro'))
    p = subprocess.run([sys.executable, os.path.join(HERE, 'tamu_judge.py'), '--videos', *videos, '--fps', '24', '--workers', '2',
                        '--dataset', 'finedance'], cwd=HERE, env=env, check=False, capture_output=True, text=True)
    print(p.stdout, p.stderr[-500:], flush=True)
    return 'BUDGET EXCEEDED' in p.stdout  # True = the $20 daily allowance is spent


def rows():
    return list(csv.DictReader(open(CSV)))


def main():
    k = int(sys.argv[1])
    clips = clip_list()[(k - 1) * BATCH: k * BATCH]
    while True:  # wait for the renderer
        m = {(r['slice'], r['method']): r for r in mapping()}
        if all((c, mm) in m for c in clips for mm in ('released', 'ours')):
            break
        print('waiting for renders...', flush=True)
        time.sleep(60)
    vids = [os.path.relpath(m[(c, mm)]['path'], HERE) if not os.path.isabs(m[(c, mm)]['path']) else m[(c, mm)]['path']
            for c in clips for mm in ('released', 'ours')]
    vids = [os.path.join('data', 'mllm_finedance_v2', os.path.basename(v)) for v in vids]
    if len(sys.argv) > 2:  # optional: sleep until this UTC time (e.g. after the daily budget reset), ISO 'YYYY-MM-DDTHH:MM:SS'
        import datetime
        target = datetime.datetime.fromisoformat(sys.argv[2]).replace(tzinfo=datetime.timezone.utc)
        while (left := (target - datetime.datetime.now(datetime.timezone.utc)).total_seconds()) > 0:
            print(f'sleeping until {target.isoformat()} ({left / 60:.0f} min left)', flush=True)
            time.sleep(min(left, 600))
    t0 = time.time()
    budget = judge(vids)  # first pass
    ids = {os.path.splitext(os.path.basename(v))[0] for v in vids}
    ok = {r['video_id'] for r in rows() if r['status'] == 'ok'}
    redo = [v for v in vids if os.path.splitext(os.path.basename(v))[0] not in ok]
    if redo and not budget:
        print(f'retrying {len(redo)} failed request(s) once, unchanged', flush=True)
        budget = judge(redo)
    if budget:
        print('*** DAILY BUDGET EXHAUSTED: batch incomplete; it resumes (done videos are skipped) after the reset ***')
    wall = time.time() - t0

    allrows = rows()
    okrows = [r for r in allrows if r['status'] == 'ok']
    by = collections.defaultdict(dict)
    for r in okrows:
        by[r['clip_id']][r['method']] = int(r['va_alignment'])
    print(f'\n=== BATCH {k}: clips {clips[0]} .. {clips[-1]} ===')
    print('clip         released  ours')
    for c in clips:
        d = by.get(c, {})
        print(f'{c:<12} {d.get("released", "MISSING"):>8} {d.get("ours", "MISSING"):>5}')
    bat = [by[c] for c in clips if len(by.get(c, {})) == 2]
    full = [d for d in by.values() if len(d) == 2]
    for label, ds in (('this batch', bat), ('CUMULATIVE', full)):
        if not ds:
            print(f'{label:<11} n= 0 clips with both scores')
            continue
        print(f'{label:<11} n={len(ds):>2} clips  released mean {st.mean(d["released"] for d in ds):.2f}  '
              f'ours mean {st.mean(d["ours"] for d in ds):.2f}')
    bad = [r for r in allrows if r['video_id'] in ids and r['status'] != 'ok']
    still = [v for v in ids if v not in {r['video_id'] for r in okrows}]
    print(f'failed attempts logged this batch: {len(bad)}; videos still without a valid score: {len(still)}')
    br = [r for r in okrows if r['video_id'] in ids]
    i, o = sum(float(r['input_tokens']) for r in br), sum(float(r['output_tokens']) for r in br)
    print(f'usage this batch: {int(i):,} input + {int(o):,} output tokens; flat ${i * 1.25 / 1e6 + o * 10 / 1e6:.2f}, '
          f'tiered ${sum(float(r["estimated_public_api_cost_usd"]) for r in br):.2f}; wall {wall / 60:.1f} min; '
          f'video-token check {collections.Counter(r["confirmed_effective_judge_fps"] for r in br)}')
    tot = sum(float(r['input_tokens']) for r in okrows if r['status'] == 'ok')
    print(f'all valid rows so far: {len(okrows)} videos, {int(tot):,} input tokens')


if __name__ == '__main__':
    main()
