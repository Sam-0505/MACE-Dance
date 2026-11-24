# Instructions video: narration script (site Clip 1 = FineDance test song 092, Real Dance)

Video: the site's Real Dance panel for Clip 1 (song 092) (`site/videos/092_slice0_seed8/real.mp4` + `audio.m4a`), played
once (34 s), scaled 2x to 900 x 2112, made by `demo/make_demo_video.py`. The song plays throughout; it is turned down while
the voice speaks. Placed at the top of the review site, with switchable captions. The BPL and IMC labels of the panel stay.

Hint drawn on the video: from 10 s on, a shaded band over 10-13.5 s on both intensity graphs, labelled "both rise".

What the data shows for this dancer (checked 2026-11-24, as drawn = each curve scaled to its own min..max over the clip):
- Intensity: both curves bottom out at about 10 s (motion 0.01-0.02, audio 0.00-0.04) and climb together to 13.5 s (motion
  0.36, audio 0.81); local correlation over 9-13 s, 10-14 s and 11-15 s is 0.97, 0.99 and 1.00. Whole clip IMC 0.68.
- Beat clock: accents land mostly on or just after the beat (phase 0.00-0.13) or just before it (0.87-0.94); one short
  stretch at 7.3-9 s lands at about 0.6. The running hand length is 0.63-0.74 from 15 s on, pointing near the top.

Definitions used (from the code): QoM = mean over the 24 SMPL joints of the frame-to-frame displacement; accents = peaks of
QoM band-passed around the beat frequency (band width 15 % of it); beat clock dot = accent phase inside its beat, i.e.
(accent - previous beat) / beat length in 0..1, drawn clockwise from the top (top = on the beat, right = a quarter beat
later, bottom = halfway between beats), arrow = their mean resultant; BPL = steadiness of the phase of the beat-band QoM
relative to the beat phase over the whole clip (0 to 1); RMS = frame-wise root-mean-square of the waveform (30 fps); both
graph curves Gaussian-smoothed with sigma 15 frames (0.5 s); IMC = Pearson correlation of the two smoothed curves, zero lag.

## Script: 34 s, IMC first, then the beat clock, then where the values are (about 75 words)

| # | Start | Max | Narration | Text to paste into a TTS tool |
|---|------:|----:|-----------|-------------------------------|
| A | 0:00 | 8 s | Below the dancer are motion intensity, QoM, and audio intensity, RMS; the red line marks the current moment. | Below the dancer are motion intensity, Q-o-M, and audio intensity, R-M-S; the red line marks the current moment. |
| B | 0:08 | 7 s | IMC is high when both curves rise and fall together, as here. | I-M-C is high when both curves rise and fall together, as here. |
| C | 0:15 | 11 s | Top right is the beat clock. Each dot is a movement accent, placed by its phase after the beat: top is on the beat, clockwise is later. | Top right is the beat clock. Each dot is a movement accent, placed by its phase after the beat: top is on the beat, clockwise is later. |
| D | 0:26 | 8 s | The longer the hand, the steadier the timing; BPL scores this. Each video shows its BPL and IMC values. | The longer the hand, the steadier the timing; B-P-L scores this. Each video shows its B-P-L and I-M-C values. |

Line B's "as here" falls at about 0:11-0:14, while both curves climb inside the shaded band (it appears at 0:10).

Generate each line as its own clip (A.mp3 ... D.mp3) with the same voice and settings; each is placed at its start time.
Each clip must be shorter than its "Max"; if one runs long, raise the speed a little or shorten it.
