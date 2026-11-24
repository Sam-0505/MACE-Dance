"""Static review site: pick a clip and which models to compare, videos play in sync (2026-11-23).

Working files: site/videos/<slice>_seed<s>/{flower,mace,ours,real}.mp4 + audio.m4a, one silent video per model (a 450 x 1056
column of the --imc row: title, dancer with beat clock + BPL, IMC graph) and the song. Published (2026-11-24): --stitch puts
FlowerDance | MACE-Dance | Ours side by side with the song as one video per clip, site/videos/<slice>_seed<s>.mp4 (1350 x 1056);
the page shows the clips one below the other, each with its own player (play, seek, speed, full screen). Plain HTML/JS, no external requests (anonymous review link).
Copy index.html, clips.js and the stitched videos into the anonymous repo; file:// also works (clips.js, not fetch).

Final videos: render_pink_grid.py --imc --split --slice X --seed S writes site/videos/X_seedS/ directly.
Placeholders: --from_rows X_seedS ... cuts the existing data/videos/showcase_windows/row_imc_X_seedS.mp4 into the four columns.
Then (always) this script rewrites site/clips.js from the clip folders, in --order, with the whole-clip BPL/IMC of
results/fullclip_bpl_imc_seeds.json.

    python make_site.py --ffmpeg $FF --from_rows 092_slice0_seed8 012_slice0_seed8 ...
    python make_site.py --ffmpeg $FF --stitch
    python make_site.py            # only rewrite clips.js
"""
import argparse
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, 'site')
VIDS = os.path.join(SITE, 'videos')
ROWS = os.path.join(HERE, 'data', 'videos', 'showcase_windows')
MODELS = ['flower', 'mace', 'ours', 'real']  # = left-to-right order of the --row video (render_pink_grid.ROW)
PW, HROW = 450, 1056
# small, streamable, no metadata (no tool/author tags in the anonymous files)
VENC = ['-c:v', 'libx264', '-preset', 'slow', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-map_metadata', '-1',
        '-an']
AENC = ['-vn', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', '-map_metadata', '-1']


def encode_video(ffmpeg, src, dst, vf=None):
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', src] + (['-vf', vf] if vf else []) + VENC + [dst], check=True)


def encode_audio(ffmpeg, src, dst, start=0.0, dur=None):
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-ss', f'{start:.3f}', '-i', src] + (['-t', f'{dur:.3f}'] if dur else [])
                   + AENC + [dst], check=True)


def from_row(ffmpeg, key):
    src = os.path.join(ROWS, f'row_imc_{key}.mp4')
    d = os.path.join(VIDS, key)
    os.makedirs(d, exist_ok=True)
    for c, m in enumerate(MODELS):
        encode_video(ffmpeg, src, os.path.join(d, m + '.mp4'), vf=f'crop={PW}:{HROW}:{c * PW}:0')
    encode_audio(ffmpeg, src, os.path.join(d, 'audio.m4a'))
    print(key, 'placeholder from', os.path.basename(src), flush=True)


SHOWN = ['flower', 'mace', 'ours']  # the page shows one stitched video per clip: FlowerDance | MACE-Dance | Ours (2026-11-24)


def stitch(ffmpeg, key):
    """videos/<key>.mp4 = the three generated-model columns side by side (1350 x 1056) with the song as its audio track."""
    d = os.path.join(VIDS, key)
    ins = sum([['-i', os.path.join(d, m + '.mp4')] for m in SHOWN], []) + ['-i', os.path.join(d, 'audio.m4a')]
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error'] + ins +
                   ['-filter_complex', f'hstack=inputs={len(SHOWN)}[v]', '-map', '[v]', '-map', f'{len(SHOWN)}:a',
                    '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-b:a', '128k', '-shortest', '-movflags', '+faststart', '-map_metadata', '-1',
                    os.path.join(VIDS, key + '.mp4')], check=True)
    print(key, 'stitched', flush=True)


def write_clips(order):
    rows = {(r['clip'], r['seed']): r for r in json.load(open(os.path.join(HERE, 'results', 'fullclip_bpl_imc_seeds.json')))}
    keys = sorted(k[:-4] for k in os.listdir(VIDS) if k.endswith('.mp4')) if os.path.isdir(VIDS) else []  # stitched videos
    keys = [k for k in order if k in keys] + [k for k in keys if k not in order]
    clips = []
    for i, key in enumerate(keys):
        clip, seed = re.fullmatch(r'(.+)_seed(\d+)', key).groups()
        r = rows[(clip, int(seed))]
        met = {m: {'bpl': round(r['bpl'][j], 2), 'imc': round(r['imc'][j], 2)} for j, m in enumerate(['ours', 'mace', 'flower'])}
        # no song name or BPM: the review page does not show them (anonymous copy)
        clips.append({'id': key, 'name': f'Clip {i + 1}', 'video': f'videos/{key}.mp4', 'models': SHOWN, 'metrics': met})
    with open(os.path.join(SITE, 'clips.js'), 'w') as f:  # neutral header: the file is copied as is into the anonymous repo
        f.write('// clip list\nwindow.CLIPS = ' + json.dumps(clips, indent=1) + ';\n')
    print('clips.js:', [c['id'] for c in clips])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ffmpeg', default='ffmpeg')
    ap.add_argument('--from_rows', nargs='*', default=[], help='<slice>_seed<s> keys to cut from the existing row_imc videos')
    ap.add_argument('--stitch', action='store_true', help='(re)make videos/<key>.mp4 for every clip folder')
    ap.add_argument('--order', nargs='*', default=None, help='clip order on the site (default: --from_rows order, then by name)')
    a = ap.parse_args()
    for key in a.from_rows:
        from_row(a.ffmpeg, key)
    if a.stitch:
        for key in sorted(k for k in os.listdir(VIDS) if os.path.isfile(os.path.join(VIDS, k, 'audio.m4a'))):
            stitch(a.ffmpeg, key)
    order = a.order or a.from_rows
    if not order and os.path.isfile(os.path.join(SITE, 'clips.js')):  # keep the current site order
        order = re.findall(r'"id": "([^"]+)"', open(os.path.join(SITE, 'clips.js')).read())
    write_clips(order)


if __name__ == '__main__':
    main()
