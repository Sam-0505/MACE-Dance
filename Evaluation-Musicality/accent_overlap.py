"""Are the treble and percussion effects beyond loudness one signal or two?

Fixed before running (2026-10-27). Datasets, pools, matching and slice pairing as in perc_distinct.py. Per slice,
s = 2, regressions within the slice (OLS with intercept):
  music  perc_x_loud          percussive RMS with loudness regressed out          (perc_distinct.py)
         treble_x_loud        6.4-12.8 kHz flux with loudness regressed out       (bass_treble.py)
         treble_x_loud_perc   treble with loudness AND percussive RMS regressed out
         perc_x_loud_treble   percussive RMS with loudness AND treble regressed out
  motion uacc   upper-body acceleration (perc_distinct.py)
         body   travel-free mean speed of the bass_treble.py leg and head/hand joints
Reading rule (Delta_real on the two large datasets, release and motorica):
  neither *_x_loud_<other> keeps Delta_real > 0 (p < .05)  -> one shared accent signal
  exactly one keeps it                                     -> that one contains the other
  both keep it                                             -> two separate parts
Also reported: median within-slice r between perc_x_loud and treble_x_loud (own music).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/accent_overlap.py [test|release|motorica ...]
"""
import json
import os
import sys

import librosa
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

from model.metric import SMPLSkeleton  # noqa: E402
import audio_features as af  # noqa: E402
from multicomponent import FLUX_SR, FLUX_HOP, FLUX_NFFT, HIGH_BAND, UPPER, band_flux, smooth  # noqa: E402
from finedance_common import SLICE_LEN  # noqa: E402
import perc_distinct as PD  # noqa: E402
import bass_treble as BT  # noqa: E402

OUT = os.path.join(HERE, 'results', 'accent_overlap.json')
MUSIC = ['perc_x_loud', 'treble_x_loud', 'treble_x_loud_perc', 'perc_x_loud_treble']
PAIRS = {f'{m}|{d}': (m, d) for m in MUSIC for d in ('uacc', 'body')}


def resid(y, *xs):
    X = np.stack(list(xs) + [np.ones_like(y)], 1)
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def full_music(path):
    y = af.load_audio(path)
    y48, _ = librosa.load(path, sr=FLUX_SR)
    S = np.abs(librosa.stft(y48, n_fft=FLUX_NFFT, hop_length=FLUX_HOP))
    return {'perc': librosa.feature.rms(y=librosa.effects.percussive(y), hop_length=af.HOP_LENGTH)[0],
            'loud': af.rms_envelope(y),
            'treble': band_flux(S, librosa.fft_frequencies(sr=FLUX_SR, n_fft=FLUX_NFFT), HIGH_BAND)}


def music_views(m, o):
    p, l, t = (smooth(np.asarray(m[k][o:o + SLICE_LEN], dtype=np.float64), PD.FAST) for k in ('perc', 'loud', 'treble'))
    assert len(p) == len(l) == len(t) == SLICE_LEN
    return {'perc_x_loud': resid(p, l), 'treble_x_loud': resid(t, l),
            'treble_x_loud_perc': resid(t, l, p), 'perc_x_loud_treble': resid(p, l, t)}


def motion_views(j, src):
    J = BT.JOINTS['motorica' if src == 'motorica' else 'smpl']
    ax, _ = BT.UP[src]
    j = np.asarray(j[:SLICE_LEN], dtype=np.float64)
    horiz = [a for a in range(3) if a != ax]
    tf = j.copy()
    tf[:, :, horiz] -= j[:, J['pelvis']:J['pelvis'] + 1, horiz]
    return {'uacc': PD.motion_views(j, PD.UPPER_MOTORICA if src == 'motorica' else UPPER)['uacc'],
            'body': BT.speed(tf, J['legs'] + J['hands'])}


def main():
    which = [a for a in sys.argv[1:] if a in ('test', 'release', 'motorica')] or ['test', 'release', 'motorica']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl = SMPLSkeleton(device)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in which:
        print(name, flush=True)
        units, own, wrong, real, mace = PD.load_dataset(name, smpl, device, full_music, music_views, motion_views)
        r = [np.corrcoef(v['perc_x_loud'], v['treble_x_loud'])[0, 1] for v in own]
        print(f'  within-slice r(perc_x_loud, treble_x_loud): median {np.median(r):.2f} '
              f'[IQR {np.percentile(r, 25):.2f}, {np.percentile(r, 75):.2f}]', flush=True)
        stats, _ = PD.score_pairs(units, own, wrong, real, mace, PAIRS)
        results[name] = {'pairs': stats, 'r_perc_treble_beyond_loud': {
            'median': float(np.median(r)), 'q25': float(np.percentile(r, 25)), 'q75': float(np.percentile(r, 75))}}
        with open(OUT, 'w') as f:
            json.dump(results, f, indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
