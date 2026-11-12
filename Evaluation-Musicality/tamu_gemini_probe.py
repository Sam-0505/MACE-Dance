"""Feasibility probe for the TAMU AI API (OpenAI-style) as the Gemini 2.5 Pro judge (MMC_Dance_Evaluation_Plan.md, Step 3).

Reads TAMUS_API_KEY / TAMUS_API_BASE / TAMUS_MODEL_ID from <repo>/.env (or the environment). The key is never printed or
logged. Raw responses (no request headers) go to results/tamu_probe/.

    python tamu_gemini_probe.py models                       # list model ids (flags Gemini ones)
    python tamu_gemini_probe.py text                         # tiny text-only call: auth + model id + usage
    python tamu_gemini_probe.py video --video PATH [--fps 24] [--shapes video_url file image_url]

`video` sends ONE video with a neutral prompt (not the Section 5 rubric) and tries payload shapes in order until one is
accepted, recording which shape works, the usage tokens, the response time, and whether the answer shows the audio was used.
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'results', 'tamu_probe')
PROBE_PROMPT = ('Describe the music you hear (instruments, tempo, any lyrics) and the first time you hear a loud drum '
                'or clap, with timestamps. Then describe in one sentence what the person in the video is doing.')


def load_env():
    p = os.path.join(ROOT, '.env')
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key, base, model = (os.environ.get(k, '') for k in ('TAMUS_API_KEY', 'TAMUS_API_BASE', 'TAMUS_MODEL_ID'))
    if not key or not base:
        sys.exit('TAMUS_API_KEY and TAMUS_API_BASE must be set in .env')
    return key, base.rstrip('/'), model


def call(method, url, key, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json', 'User-Agent': 'curl/8.5.0', 'Accept': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt, code, hdr = r.read().decode(), r.status, r.headers
    except urllib.error.HTTPError as e:
        txt, code, hdr = e.read().decode(errors='replace'), e.code, e.headers
    except Exception as e:  # network / timeout
        return None, f'{type(e).__name__}: {e}', {}, time.time() - t0
    keep = {k: v for k, v in hdr.items() if any(s in k.lower() for s in ('ratelimit', 'rate-limit', 'quota', 'usage', 'retry', 'x-request'))}
    try:
        js = json.loads(txt)
    except ValueError:
        js = sse_to_completion(txt) if txt.lstrip().startswith('data:') else {'_raw': txt[:2000]}
    return code, js, keep, time.time() - t0


def sse_to_completion(txt):
    """The gateway streams (SSE) even when not asked: fold the chunks into one chat.completion-like dict."""
    content, usage, finish, raw = [], None, None, []
    for line in txt.splitlines():
        line = line.strip()
        if not line.startswith('data:') or line.endswith('[DONE]'):
            continue
        try:
            c = json.loads(line[5:])
        except ValueError:
            continue
        raw.append(c)
        for ch in c.get('choices', []):
            content.append((ch.get('delta') or {}).get('content') or '')
            finish = ch.get('finish_reason') or finish
        usage = c.get('usage') or usage
    return {'choices': [{'message': {'content': ''.join(content)}, 'finish_reason': finish}], 'usage': usage,
            '_chunks': len(raw), '_sse': True}


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, 'w'), indent=1)
    return p


def cmd_models(a, key, base, model):
    code, js, hdr, dt = call('GET', f'{base}/models', key)
    print('GET /models ->', code, f'{dt:.1f}s')
    if code != 200:
        print(json.dumps(js)[:600]); return
    ids = [m.get('id', m) for m in (js.get('data', js) if isinstance(js, dict) else js)]
    print(len(ids), 'models')
    for i in ids:
        print('  ', i, '  <-- Gemini' if 'gemini' in str(i).lower() else '')
    print('saved', save('models.json', js))


def cmd_text(a, key, base, model):
    if not model:
        sys.exit('set TAMUS_MODEL_ID in .env (see `models`)')
    body = {'model': model, 'messages': [{'role': 'user', 'content': 'Reply with the single word: ok'}], 'max_tokens': 2000}
    code, js, hdr, dt = call('POST', f'{base}/chat/completions', key, body)
    print('POST /chat/completions ->', code, f'{dt:.1f}s', hdr)
    print(json.dumps(js)[:800])
    save('text.json', {'status': code, 'response': js, 'headers': hdr, 'seconds': dt})


def payloads(shape, b64, fname, fps):
    uri = f'data:video/mp4;base64,{b64}'
    part = {'video_url': {'type': 'video_url', 'video_url': {'url': uri}},
            'file': {'type': 'file', 'file': {'filename': fname, 'file_data': uri}},
            'image_url': {'type': 'image_url', 'image_url': {'url': uri}}}[shape]
    # Findings 2026-11-12: 'video_url' returns 200 but the video is silently dropped (42 prompt tokens); 'file' and
    # 'image_url' process video+audio. fps is honored ONLY inside the file object (file.video_metadata.fps = 2 doubled
    # the video tokens); part-level fps, top-level fps and extra_body were ignored.
    if fps and shape == 'file':
        part['file']['video_metadata'] = {'fps': fps}
    return part, {}


def cmd_video(a, key, base, model):
    if not model:
        sys.exit('set TAMUS_MODEL_ID in .env (see `models`)')
    raw = open(a.video, 'rb').read()
    b64 = base64.b64encode(raw).decode()
    print(f'video {os.path.basename(a.video)} {len(raw) / 1e6:.2f} MB, fps request: {a.fps or "none"}')
    for shape in a.shapes:
        part, extra = payloads(shape, b64, os.path.basename(a.video), a.fps)
        body = {'model': model, 'max_tokens': 1500,
                'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': PROBE_PROMPT}, part]}], **extra}
        code, js, hdr, dt = call('POST', f'{base}/chat/completions', key, body)
        print(f'[{shape}] -> {code} in {dt:.1f}s {hdr}')
        tag = f'video_{shape}_fps{a.fps or 0}.json'
        save(tag, {'status': code, 'response': js, 'headers': hdr, 'seconds': dt, 'shape': shape, 'fps': a.fps})
        if code == 200:
            msg = js['choices'][0]['message']['content']
            usage = js.get('usage') or {}
            print('usage:', usage)
            vt = (usage.get('prompt_tokens_details') or {}).get('video_tokens', 0)
            if not vt:
                print('WARNING: 200 but no video tokens, so the video was NOT seen; the answer is not grounded')
                print('answer (ignore):', msg[:300])
                continue
            print(f'video tokens {vt} = {vt / 258 / (len(raw) and 34):.2f} frames/s at 258 tokens/frame (34 s clip)')
            print('answer:', msg[:1500])
            return
        print('   ', json.dumps(js)[:400])
    print('no payload shape accepted')


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('models'); sp.add_parser('text')
    v = sp.add_parser('video')
    v.add_argument('--video', required=True)
    v.add_argument('--fps', type=float, default=0)
    v.add_argument('--shapes', nargs='+', default=['file', 'image_url'])
    a = ap.parse_args()
    key, base, model = load_env()
    {'models': cmd_models, 'text': cmd_text, 'video': cmd_video}[a.cmd](a, key, base, model)


if __name__ == '__main__':
    main()
