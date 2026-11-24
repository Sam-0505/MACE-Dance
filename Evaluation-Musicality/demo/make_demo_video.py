"""Instructions video for the review site (2026-11-24): the Real Dance panel of site Clip 1 (FineDance test song 092,
site/videos/092_slice0_seed8/real.mp4, 450 x 1056, made by render_pink_grid.py --imc --split), scaled 2x, with one hint drawn
on top: from 10 s on, a shaded band over 10-13.5 s on both intensity graphs, where motion and loudness climb together from
their lowest point (local correlation 0.97-1.00; see real_demo_script.md).
Narration: the ElevenLabs clips demo/out/voice/{A,B,C,D}.mp3 (lines of real_demo_script.md) placed at VOICE starts; the song
is turned down to DUCK while a voice clip plays (RAMP s fades). Captions: out/instructions.vtt, one cue per clip.

    python demo/make_demo_video.py --ffmpeg $FF --still 14      # one frame as png (layout check)
    python demo/make_demo_video.py --ffmpeg $FF                 # out/instructions.mp4 + out/instructions.vtt
"""
import argparse
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, '..', 'site', 'videos', '092_slice0_seed8')
OUT = os.path.join(HERE, 'out')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
W0, H0, S, FPS, NF = 450, 1056, 2, 30, 1020           # panel size, scale, frame rate, frames in the clip
# panel layout of render_pink_grid.py (--imc): graph x range and the two strips (y0, y1), in panel pixels
GX0, GX1 = 16, W0 - 16
STRIPS = [(46 + 800 + 22, 46 + 800 + 92), (46 + 800 + 112, 46 + 800 + 182)]
BAND = (10.0, 13.5)                                   # seconds: both curves climb together
LABEL = ['both', 'rise']                              # two short lines, so it fits inside the band
SHOW = 10.0                                           # the band appears when the red line gets there
FADE = 0.5
SR = 44100
# (clip, start s, caption). Starts shifted from the script's 0/8/15/26 s to fit the recorded lengths (8.44, 4.68, 9.09,
# 8.36 s): B's "as here" still falls inside the band, D ends at 33.8 s.
VOICE = [('A', 0.0, 'Below the dancer are motion intensity, QoM, and audio intensity, RMS; the red line marks the current moment.'),
         ('B', 8.6, 'IMC is high when both curves rise and fall together, as here.'),
         ('C', 15.0, 'Top right is the beat clock. Each dot is a movement accent, placed by its phase after the beat: '
                     'top is on the beat, clockwise is later.'),
         ('D', 25.4, 'The longer the hand, the steadier the timing; BPL scores this. Each video shows its BPL and IMC values.')]
DUCK, RAMP = 0.25, 0.3
CUE = 'line:-12'  # first line 12 caption lines up from the bottom; the cue grows downward, so up to 4 lines end above the graphs (bottom 20 %); wraps, unlike a % line


def xp(t):
    return S * (GX0 + t * FPS / (NF - 1) * (GX1 - GX0))


def overlay(alpha):
    """RGBA layer with the band at the given opacity (0..1)."""
    lay = Image.new('RGBA', (W0 * S, H0 * S), (0, 0, 0, 0))
    if alpha <= 0:
        return lay
    d = ImageDraw.Draw(lay)
    xa, xb = xp(BAND[0]), xp(BAND[1])
    for y0, y1 in STRIPS:
        d.rectangle([xa, S * y0, xb, S * y1], fill=(255, 170, 0, int(70 * alpha)), outline=(225, 130, 0, int(255 * alpha)), width=3)
    f = ImageFont.truetype(FONT, 22)
    xm = (xa + xb) / 2
    for j, word in enumerate(LABEL):                   # inside the band, top of the motion box (the curve is low there)
        d.text((xm - d.textlength(word, font=f) / 2, S * STRIPS[0][0] + 8 + 26 * j), word, fill=(205, 110, 0, int(255 * alpha)), font=f)
    return lay


def frames(ffmpeg, src):
    p = subprocess.Popen([ffmpeg, '-loglevel', 'error', '-i', src, '-vf', f'scale={W0 * S}:{H0 * S}:flags=lanczos',
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    n = W0 * S * H0 * S * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield Image.frombuffer('RGB', (W0 * S, H0 * S), b)
    p.wait()


def decode(ffmpeg, path):
    raw = subprocess.run([ffmpeg, '-loglevel', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def ts(t):
    return '%02d:%02d:%06.3f' % (t // 3600, t % 3600 // 60, t % 60)


def narration(ffmpeg, wav, vtt):
    """Song with the voice clips at their starts, song lowered to DUCK under each clip; WebVTT captions, one cue per clip."""
    music = decode(ffmpeg, os.path.join(SITE, 'audio.m4a'))
    n = len(music)
    gain, voice, cues = np.ones(n), np.zeros(n), []
    tt = np.arange(n) / SR
    for name, t0, text in VOICE:
        v = decode(ffmpeg, os.path.join(OUT, 'voice', name + '.mp3'))
        i0 = int(t0 * SR)
        assert i0 + len(v) <= n, f'{name} ends at {t0 + len(v) / SR:.2f} s, after the clip'
        voice[i0:i0 + len(v)] += v
        t1 = t0 + len(v) / SR
        # 1 outside, DUCK inside [t0, t1], linear RAMP s fades on both sides
        g = np.clip(np.maximum((t0 - tt) / RAMP, (tt - t1) / RAMP), 0, 1)   # 0 inside the clip, 1 beyond the fades
        gain = np.minimum(gain, DUCK + (1 - DUCK) * g)
        cues.append((t0, t1, text))
        print(f'{name}: {t0:.2f}-{t1:.2f} s', flush=True)
    mix = music * gain + voice
    mix /= max(1.0, np.abs(mix).max() / 0.98)
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-f', 'f64le', '-ar', str(SR), '-ac', '1', '-i', '-', wav],
                   input=mix.tobytes(), check=True)
    with open(vtt, 'w') as f:
        f.write('WEBVTT\n\n' + ''.join(f'{i + 1}\n{ts(a)} --> {ts(b)} {CUE}\n{text}\n\n' for i, (a, b, text) in enumerate(cues)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ffmpeg', default='ffmpeg')
    ap.add_argument('--still', type=float, default=None, help='only this second, as png')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(SITE, 'real.mp4')
    layers = {}

    def draw(img, t):
        al = min(max((t - SHOW) / FADE, 0), 1)
        if al <= 0:
            return img
        key = round(al, 2)
        if key not in layers:
            layers[key] = overlay(key)
        return Image.alpha_composite(img.convert('RGBA'), layers[key]).convert('RGB')

    if a.still is not None:
        out = os.path.join(OUT, f'still_{a.still:g}s.png')
        img = subprocess.run([a.ffmpeg, '-loglevel', 'error', '-ss', f'{a.still}', '-i', src, '-frames:v', '1',
                              '-vf', f'scale={W0 * S}:{H0 * S}:flags=lanczos', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                             capture_output=True, check=True).stdout
        draw(Image.frombuffer('RGB', (W0 * S, H0 * S), img), a.still).save(out)
        print(out)
        return
    wav, vtt, out = (os.path.join(OUT, f) for f in ('narration.wav', 'instructions.vtt', 'instructions.mp4'))
    narration(a.ffmpeg, wav, vtt)
    enc = subprocess.Popen([a.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W0 * S}x{H0 * S}',
                            '-r', str(FPS), '-i', '-', '-i', wav, '-map', '0:v', '-map', '1:a',
                            '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k',
                            '-shortest', '-movflags', '+faststart', '-map_metadata', '-1', out], stdin=subprocess.PIPE)
    for i, img in enumerate(frames(a.ffmpeg, src)):
        enc.stdin.write(np.asarray(draw(img, i / FPS)).tobytes())
    enc.stdin.close()
    enc.wait()
    import shutil   # the review site plays these (site/videos/ is not in git; copy them into the anonymous repo)
    for f in (out, vtt):
        shutil.copy(f, os.path.join(HERE, '..', 'site', 'videos', os.path.basename(f)))
    print(out, vtt, '-> site/videos/')


if __name__ == '__main__':
    main()
