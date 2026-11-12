"""TMD-Bench Visual Audio Alignment judge via the TAMU AI API (Gemini 2.5 Pro), one fresh request per video.

Sends the UNMODIFIED prompt in ../tmd_va_original_prompt.txt (verified identical to Section 5 of MMC_Dance_Evaluation_Plan.md)
plus the video (neutral hashed filename; method identity is never sent). Parses va_alignment (integer 1-5), keeps the raw
response, and appends one row per video to results/mllm_judge.csv (schema = the plan's suggested fields).

    python tamu_judge.py --videos data/mllm_pilot/dd4a0fc303.mp4 --fps 24 --dataset finedance --tag pilot

Credentials/endpoint come from <repo>/.env via tamu_gemini_probe.load_env (never printed). If the requested fps is not
accepted the request FAILS and is logged as such; it is never silently changed.
"""
import argparse
import base64
import csv
import json
import os
import random
import re
import threading
import time
import concurrent.futures

import tamu_gemini_probe as P

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = open(os.path.join(os.path.dirname(HERE), 'tmd_va_original_prompt.txt')).read().strip()
RAW_DIR = os.path.join(HERE, 'results', 'tamu_judge_raw')
CSV = os.path.join(HERE, 'results', 'mllm_judge.csv')
FIELDS = ['dataset', 'song_id', 'clip_id', 'method', 'video_id', 'video_path', 'duration_s', 'source_fps',
          'requested_judge_fps', 'confirmed_effective_judge_fps', 'gemini_model_id', 'va_alignment',
          'raw_response_location', 'elapsed_seconds', 'input_tokens', 'video_tokens', 'audio_tokens', 'output_tokens',
          'thinking_tokens', 'actual_tamu_cost_usd', 'estimated_public_api_cost_usd', 'status']
TOK_PER_FRAME = 258  # Gemini video tokens per sampled frame (matches the 1 FPS measurement: 34 s = 8,772)


def parse_score(text):
    """First JSON object in the reply; va_alignment must be an integer 1-5."""
    for m in re.finditer(r'\{.*?\}', text, re.S):
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        v = d.get('va_alignment')
        if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5:
            return v, d.get('reasoning')
        if isinstance(v, str) and v.strip() in '12345' and v.strip():
            return int(v), d.get('reasoning')
    return None, None


def public_cost(inp, out):
    """gemini-2.5-pro public standard rates as previously documented in the plan; MUST be reconfirmed before use."""
    hi = inp > 200_000
    return inp / 1e6 * (2.50 if hi else 1.25) + out / 1e6 * (15.0 if hi else 10.0)


def judge(path, fps, key, base, model, meta):
    b64 = base64.b64encode(open(path, 'rb').read()).decode()
    fobj = {'filename': os.path.basename(path), 'file_data': f'data:video/mp4;base64,{b64}'}
    if fps:
        fobj['video_metadata'] = {'fps': fps}
    body = {'model': model, 'max_tokens': 16000,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT}, {'type': 'file', 'file': fobj}]}]}
    code, js, hdr, dt = P.call('POST', f'{base}/chat/completions', key, body)
    vid = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(RAW_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DIR, f'{vid}_fps{fps:g}_{int(time.time())}.json')
    json.dump({'status': code, 'response': js, 'headers': hdr, 'seconds': dt, 'requested_fps': fps, 'model': model,
               'prompt_file': 'tmd_va_original_prompt.txt'}, open(raw_path, 'w'), indent=1)
    row = {**meta, 'video_id': vid, 'video_path': path, 'requested_judge_fps': fps, 'gemini_model_id': model,
           'raw_response_location': raw_path, 'elapsed_seconds': round(dt, 1), 'actual_tamu_cost_usd': 'unknown'}
    if code != 200 or not isinstance(js, dict) or 'choices' not in js:
        row['status'] = f'http_{code}: {str(js)[:150]}'
        return row
    u = js.get('usage') or {}
    det = u.get('prompt_tokens_details') or {}
    cdet = u.get('completion_tokens_details') or {}
    row.update(input_tokens=u.get('prompt_tokens'), video_tokens=det.get('video_tokens'), audio_tokens=det.get('audio_tokens'),
               output_tokens=u.get('completion_tokens'), thinking_tokens=cdet.get('reasoning_tokens'))
    if det.get('video_tokens'):
        dur = float(meta['duration_s'])
        vt = det['video_tokens']
        if det.get('cached_tokens'):  # a repeated prefix is billed as cached and the modality split shrinks: use the total
            vt = u['prompt_tokens'] - 25 * dur - 414  # total input minus audio (25 tok/s) and the ~414-token prompt text
        row['confirmed_effective_judge_fps'] = round(vt / TOK_PER_FRAME / dur, 2)  # inferred from tokens
    if u.get('prompt_tokens') is not None and u.get('completion_tokens') is not None:
        row['estimated_public_api_cost_usd'] = round(public_cost(u['prompt_tokens'], u['completion_tokens']), 4)
    text = js['choices'][0]['message']['content']
    score, reasoning = parse_score(text)
    row['va_alignment'] = score
    row['status'] = ('ok' if score is not None else 'invalid_response') if det.get('video_tokens') else 'video_not_seen'
    print(f'{vid}: status {row["status"]}, score {score}, {dt:.0f} s, usage {u}')
    print('  reasoning:', (reasoning or text)[:600])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--fps', type=float, required=True, help='requested judge sampling FPS (file.video_metadata.fps)')
    ap.add_argument('--dataset', default='finedance')
    ap.add_argument('--tag', default='')
    ap.add_argument('--workers', type=int, default=1, help='parallel requests (1M TPM cap ~ 4 requests/min)')
    ap.add_argument('--shuffle_seed', type=int, default=0)
    a = ap.parse_args()
    key, base, model = P.load_env()
    model = model or 'protected.gemini-2.5-pro'
    mapping = {}
    mp = os.path.join(os.path.dirname(a.videos[0]), 'mapping.csv')
    if os.path.exists(mp):
        mapping = {r['video_id']: r for r in csv.DictReader(open(mp))}
    done = set()
    if os.path.exists(CSV):  # never re-judge (and re-bill) a video that already has an ok row at this fps
        done = {r['video_id'] for r in csv.DictReader(open(CSV)) if r['status'] == 'ok' and float(r['requested_judge_fps'] or 0) == a.fps}
    todo = [v for v in a.videos if os.path.splitext(os.path.basename(v))[0] not in done]
    random.Random(a.shuffle_seed).shuffle(todo)  # method must not be confounded with time of day / position
    print(f'{len(todo)} to judge ({len(a.videos) - len(todo)} already done), {a.workers} worker(s)', flush=True)
    lock = threading.Lock()

    budget_hit = threading.Event()  # TAMU answers 429 'Budget Exceeded' once the $20 daily allowance is spent

    def one(v):
        if budget_hit.is_set():
            return  # do not hammer the API or log a failure for every remaining video
        vid = os.path.splitext(os.path.basename(v))[0]
        m = mapping.get(vid, {})
        meta = {'dataset': a.dataset, 'song_id': m.get('slice', '').split('_')[0], 'clip_id': m.get('slice', ''),
                'method': m.get('method', ''), 'duration_s': 34.0, 'source_fps': m.get('fps', 30)}
        row = judge(v, a.fps, key, base, model, meta)
        if row['status'].startswith('http_429'):
            budget_hit.set()
            print('BUDGET EXCEEDED (HTTP 429): stopping, remaining videos are not sent', flush=True)
            return  # not a scoring failure: nothing is logged in the results CSV
        with lock:
            new = not os.path.exists(CSV)
            with open(CSV, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
                if new:
                    w.writeheader()
                w.writerow(row)

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, todo))


if __name__ == '__main__':
    main()
