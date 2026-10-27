"""Download the music for the ImperialDance subset.

Needs yt-dlp and ffmpeg. Writes <out_dir>/<choreo>.wav holding the README time range
padded by --pad seconds on each side (the motion lengths do not always match the
README ranges, so eval_imperial.py locates the motion inside this window), plus
<choreo>.json recording where the window starts in the source track.
    python fetch_imperial_audio.py --ffmpeg /path/to/ffmpeg
"""
import argparse
import json
import os
import subprocess
import tempfile

from imperial_subset import SUBSET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'imperial', 'audio'))
    ap.add_argument('--only', nargs='*', default=list(SUBSET))
    ap.add_argument('--pad', type=float, default=10.0)
    ap.add_argument('--ffmpeg', default='ffmpeg')
    ap.add_argument('--yt_dlp', default='yt-dlp')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        for key in args.only:
            src = SUBSET[key]
            out = os.path.join(args.out_dir, key + '.wav')
            if os.path.exists(out):
                print('exists', out)
                continue
            subprocess.run([args.yt_dlp, '-x', '--audio-format', 'wav', '--no-playlist',
                            '--ffmpeg-location', args.ffmpeg, '-o', os.path.join(tmp, key + '.%(ext)s'), src['url']],
                           check=True)
            win_start = max(0.0, src['start'] - args.pad)
            subprocess.run([args.ffmpeg, '-y', '-loglevel', 'error', '-i', os.path.join(tmp, key + '.wav'),
                            '-ss', str(win_start), '-to', str(src['end'] + args.pad), '-ac', '1', out], check=True)
            with open(os.path.join(args.out_dir, key + '.json'), 'w') as f:
                json.dump({'window_start': win_start, 'readme_start': src['start'], 'readme_end': src['end']}, f)
            print('wrote', out)


if __name__ == '__main__':
    main()
