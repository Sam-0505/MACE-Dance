"""Instructions video for the review site (2026-11-24), same frame as the comparison clips (1350 x 1056).

Left: the Real Dance panel of site Clip 1 (FineDance test song 092, site/videos/092_slice0_seed8/real.mp4, 450 x 1056, made by
render_pink_grid.py --imc --split), with a shaded 'both rise' band over 11-15 s on both intensity graphs from 11 s on.
Right: the narration as large text (the phrase being spoken in dark, the rest grey) and arrows from it to the marker it
explains (graphs, red line, IMC label, band, beat clock, dots, top, clockwise, hand, BPL label).
Sound: the ElevenLabs clips demo/out/voice/{A,B,C,D}.mp3 (lines of real_demo_script.md) at their VOICE starts, the song
turned down to DUCK while a clip plays (RAMP s fades).

    python demo/make_demo_video.py --ffmpeg $FF --still 14 28   # frames as png (layout check)
    python demo/make_demo_video.py --ffmpeg $FF                 # out/instructions.mp4, copied to site/videos/
"""
import argparse
import math
import os
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, '..', 'site', 'videos', '092_slice0_seed8')
OUT = os.path.join(HERE, 'out')
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_R = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
PW, H, W, FPS, NF = 450, 1056, 1350, 30, 1020        # panel width, frame height/width, frame rate, frames in the clip
# panel layout of render_pink_grid.py (--imc), panel pixels
TITLE, PH = 46, 800
GX0, GX1 = 16, PW - 16
STRIPS = [(TITLE + PH + 22, TITLE + PH + 92), (TITLE + PH + 112, TITLE + PH + 182)]
CX, CY, R = PW - 58, TITLE + 60, 40                   # beat clock centre and radius
BAND = (11.0, 15.0)                                   # seconds, shaded on both graphs from BAND[0] on
LABEL = ['both', 'rise']
FADE = 0.4
ORANGE = (205, 110, 0)
INK, GREY = (28, 28, 30), (170, 170, 176)
SR = 44100
DUCK, RAMP = 0.25, 0.3

# (clip, start s, text block top y, phrases (start s within the clip, text, arrow targets)).
# Starts fit the recorded lengths (8.44, 4.68, 9.09, 8.36 s); phrase starts sit in the pauses found in each clip.
VOICE = [
    ('A', 0.0, 770, [(0.0, 'Below the dancer are motion intensity, QoM,', ['motion']),
                     (3.45, 'and audio intensity, RMS;', ['audio']),
                     (6.2, 'the red line marks the current moment.', ['redline'])]),
    ('B', 8.6, 770, [(0.0, 'IMC is high when both curves rise and fall together,', ['imc']),
                     (3.85, 'as here.', ['band'])]),
    ('C', 15.0, 70, [(0.0, 'Top right is the beat clock.', ['clock']),
                     (1.8, 'Each dot is a movement accent, placed by its phase after the beat:', ['dots']),
                     (5.9, 'top is on the beat,', ['top']),
                     (7.5, 'clockwise is later.', ['clockwise'])]),
    ('D', 25.4, 330, [(0.0, 'The longer the hand, the steadier the timing;', ['hand']),
                      (2.7, 'BPL scores this.', ['bpl']),
                      (4.65, 'Each video shows its BPL and IMC values.', ['bpl', 'imc'])]),
]
CLIP_LEN = {}


def xp(t):
    return GX0 + t * FPS / (NF - 1) * (GX1 - GX0)


def target(name, t):
    """Where an arrow points (frame pixels), for a marker of the left panel at time t."""
    return {'motion': (GX1 + 3, (STRIPS[0][0] + STRIPS[0][1]) / 2),
            'audio': (GX1 + 3, (STRIPS[1][0] + STRIPS[1][1]) / 2),
            'redline': (xp(t) + 3, STRIPS[0][0] - 4),
            'imc': (GX1 + 3, STRIPS[0][0] - 10),
            'band': (xp(BAND[1]) + 2, (STRIPS[0][1] + STRIPS[1][0]) / 2),
            'clock': (CX + R + 3, CY),
            'dots': (CX + R * 0.75 + 3, CY + R * 0.66 + 3),
            'top': (CX + 9, CY - R - 8),
            'clockwise': (CX + (R + 14) * math.sin(math.radians(60)) + 3, CY - (R + 14) * math.cos(math.radians(60))),
            'hand': (CX + 5, CY + 2),
            'bpl': (CX + 44, CY + R + 30)}[name]


def arrow(d, p0, p1, col, w=4, head=16):
    d.line([p0, p1], fill=col, width=w)
    a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    pts = [p1, (p1[0] - head * math.cos(a - 0.4), p1[1] - head * math.sin(a - 0.4)),
           (p1[0] - head * math.cos(a + 0.4), p1[1] - head * math.sin(a + 0.4))]
    d.polygon(pts, fill=col)


def clockwise_arc(d, col):
    """Curved arrow just outside the clock, from the top going clockwise."""
    rr = R + 14
    pts = [(CX + rr * math.sin(math.radians(a)), CY - rr * math.cos(math.radians(a))) for a in range(-10, 75, 5)]
    d.line(pts, fill=col, width=4)
    arrow(d, pts[-2], pts[-1], col, w=4, head=14)


def wrap(d, text, font, width):
    lines, cur = [], ''
    for word in text.split():
        t = (cur + ' ' + word).strip()
        if d.textlength(t, font=font) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


class Right:
    """Draws the right part (text + arrows) for a time t."""
    X0, X1, FS, LH = PW + 70, W - 50, 36, 50

    def __init__(self):
        self.f = ImageFont.truetype(FONT_B, self.FS)
        self.f_head = ImageFont.truetype(FONT_B, 24)

    def layout(self, d, phrases, top):
        """Lines of the block, each phrase starting on a new line: [(phrase index, line text, y)]."""
        out, y = [], top
        for k, (_, text, _) in enumerate(phrases):
            for line in wrap(d, text, self.f, self.X1 - self.X0):
                out.append((k, line, y))
                y += self.LH
        return out

    def draw(self, img, t):
        d = ImageDraw.Draw(img)
        d.rectangle([PW, 0, W, H], fill=(255, 255, 255))
        d.line([(PW, 0), (PW, H)], fill=(225, 225, 230), width=2)
        d.text((self.X0, 24), 'How to read the videos', fill=GREY, font=self.f_head)
        cur = None                                     # the clip being spoken, else the last one (all grey)
        for v in VOICE:
            if t >= v[1]:
                cur = v
        if cur is None:
            return img
        name, t0, top, phrases = cur
        k_now = None
        if t < t0 + CLIP_LEN[name] + 0.25:
            for k, (ps, _, _) in enumerate(phrases):
                if t - t0 >= ps:
                    k_now = k
        lines = self.layout(d, phrases, top)
        for k, line, y in lines:
            d.text((self.X0, y), line, fill=INK if k == k_now else GREY, font=self.f)
        if k_now is not None:
            y_first = min(y for k, _, y in lines if k == k_now)
            p0 = (self.X0 - 16, y_first + self.FS * 0.6)
            for tg in phrases[k_now][2]:
                if tg == 'clockwise':
                    clockwise_arc(d, ORANGE)
                arrow(d, p0, target(tg, t), ORANGE)
        return img


def band_layer(alpha):
    lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    if alpha <= 0:
        return lay
    d = ImageDraw.Draw(lay)
    xa, xb = xp(BAND[0]), xp(BAND[1])
    for y0, y1 in STRIPS:
        d.rectangle([xa, y0, xb, y1], fill=(255, 170, 0, int(70 * alpha)), outline=(225, 130, 0, int(255 * alpha)), width=2)
    f = ImageFont.truetype(FONT_B, 12)
    for j, word in enumerate(LABEL):                   # inside the band, top of the motion box (the curve is low there)
        d.text(((xa + xb) / 2 - d.textlength(word, font=f) / 2, STRIPS[0][0] + 4 + 13 * j), word,
               fill=ORANGE + (int(255 * alpha),), font=f)
    return lay


def decode(ffmpeg, path):
    raw = subprocess.run([ffmpeg, '-loglevel', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(np.float64)


def narration(ffmpeg, wav):
    """Song with the voice clips at their starts, song lowered to DUCK under each clip."""
    music = decode(ffmpeg, os.path.join(SITE, 'audio.m4a'))
    n = len(music)
    gain, voice = np.ones(n), np.zeros(n)
    tt = np.arange(n) / SR
    for name, t0, _, _ in VOICE:
        v = decode(ffmpeg, os.path.join(OUT, 'voice', name + '.mp3'))
        i0 = int(t0 * SR)
        assert i0 + len(v) <= n, f'{name} ends at {t0 + len(v) / SR:.2f} s, after the clip'
        voice[i0:i0 + len(v)] += v
        t1 = t0 + len(v) / SR
        g = np.clip(np.maximum((t0 - tt) / RAMP, (tt - t1) / RAMP), 0, 1)   # 0 inside the clip, 1 beyond the fades
        gain = np.minimum(gain, DUCK + (1 - DUCK) * g)
        print(f'{name}: {t0:.2f}-{t1:.2f} s', flush=True)
    mix = music * gain + voice
    mix /= max(1.0, np.abs(mix).max() / 0.98)
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-f', 'f64le', '-ar', str(SR), '-ac', '1', '-i', '-', wav],
                   input=mix.tobytes(), check=True)


def panel_frames(ffmpeg, src, start=None, count=None):
    cmd = [ffmpeg, '-loglevel', 'error'] + (['-ss', f'{start}'] if start is not None else []) + ['-i', src]
    cmd += (['-frames:v', str(count)] if count else []) + ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = PW * H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield Image.frombuffer('RGB', (PW, H), b)
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ffmpeg', default='ffmpeg')
    ap.add_argument('--still', type=float, nargs='*', default=None, help='only these seconds, as png')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(SITE, 'real.mp4')
    for name, _, _, _ in VOICE:
        CLIP_LEN[name] = len(decode(a.ffmpeg, os.path.join(OUT, 'voice', name + '.mp3'))) / SR
    right, layers = Right(), {}

    def frame(panel, t):
        img = Image.new('RGB', (W, H), (255, 255, 255))
        img.paste(panel, (0, 0))
        al = min(max((t - BAND[0]) / FADE, 0), 1)
        if al > 0:
            key = round(al, 2)
            if key not in layers:
                layers[key] = band_layer(key)
            img = Image.alpha_composite(img.convert('RGBA'), layers[key]).convert('RGB')
        return right.draw(img, t)

    if a.still is not None:
        for s in a.still:
            panel = next(panel_frames(a.ffmpeg, src, start=s, count=1))
            out = os.path.join(OUT, f'still_{s:g}s.png')
            frame(panel, s).save(out)
            print(out)
        return
    wav, out = os.path.join(OUT, 'narration.wav'), os.path.join(OUT, 'instructions.mp4')
    narration(a.ffmpeg, wav)
    enc = subprocess.Popen([a.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                            '-r', str(FPS), '-i', '-', '-i', wav, '-map', '0:v', '-map', '1:a',
                            '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k',
                            '-shortest', '-movflags', '+faststart', '-map_metadata', '-1', out], stdin=subprocess.PIPE)
    for i, panel in enumerate(panel_frames(a.ffmpeg, src)):
        enc.stdin.write(np.asarray(frame(panel, i / FPS)).tobytes())
    enc.stdin.close()
    enc.wait()
    shutil.copy(out, os.path.join(HERE, '..', 'site', 'videos', 'instructions.mp4'))   # site/videos/ is not in git
    print(out, '-> site/videos/instructions.mp4')


if __name__ == '__main__':
    main()
