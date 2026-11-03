"""Music signals for every Motion Expert training slice (targets for the musicality fine-tune).

Training slice <song>_slice<k> = seconds k to k + 8 of the FineDance release wav. Signals are computed on the
whole song with the evaluation's extractors (Evaluation-Musicality/accent_overlap.full_music), then cut per slice,
unsmoothed, 30 fps: loud = RMS, perc = RMS of the HPSS percussive part, treble = 6.4-12.8 kHz positive flux.

Alignment check over all slices: per-slice Pearson r between the cached MFCC 0 and MFCC 0 recomputed from the
release wav at the slice's offset. Result (2026-11-02): median r = .999, min .836; 113 slices have a constant
cached MFCC 0 (silence, mostly song 169, whose wav is 92% digital silence) and give r = nan. A ±2 s offset search
on 3 slices per song found offset 0 in 544/544.

    cd Expert-Motion
    .venv/bin/python precompute_music_signals.py

Writes data/dataset_backups/music_signals_train.npz: names (N,), loud / perc / treble (N, 240) float32, in the
order of train_tensor_dataset.pkl.
"""
import os
import pickle
import sys
from multiprocessing import Pool

import librosa
import numpy as np

EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Evaluation-Musicality')
sys.path.insert(0, os.path.abspath(EVAL))
sys.path.insert(0, os.getcwd())

import audio_features as af  # noqa: E402
from accent_overlap import full_music  # noqa: E402
from finedance_common import RELEASE  # noqa: E402

DATASET = 'data/dataset_backups/train_tensor_dataset.pkl'
OUT = 'data/dataset_backups/music_signals_train.npz'
COMPONENTS = ('loud', 'perc', 'treble')
FPS, SLICE_FRAMES = 30, 240


def song_signals(song):
    path = os.path.join(RELEASE, 'music_wav', song + '.wav')
    m = {c: np.asarray(a, dtype=np.float32) for c, a in full_music(path).items() if c in COMPONENTS}
    m['mfcc0'] = librosa.feature.mfcc(y=af.load_audio(path), sr=af.SR, n_mfcc=20)[0]
    return song, m


def main():
    ds = pickle.load(open(DATASET, 'rb'))
    names = [os.path.splitext(os.path.basename(w))[0] for w in ds.data['wavs']]
    songs = sorted({n.split('_slice')[0] for n in names})
    print(f'{len(names)} slices from {len(songs)} songs', flush=True)
    with Pool(12) as pool:
        music = dict(pool.imap_unordered(song_signals, songs))

    cached = np.asarray(ds.data['features'])[:, :SLICE_FRAMES, 1]
    out = {c: np.zeros((len(names), SLICE_FRAMES), dtype=np.float32) for c in COMPONENTS}
    r = np.full(len(names), np.nan)
    for i, n in enumerate(names):
        song, k = n.split('_slice')
        o = int(k) * FPS
        for c in COMPONENTS + ('mfcc0',):
            seg = music[song][c][o:o + SLICE_FRAMES]
            assert len(seg) == SLICE_FRAMES, (n, c, len(seg))
            if c == 'mfcc0':
                if cached[i].std() > 0:
                    r[i] = np.corrcoef(seg, cached[i])[0, 1]
            else:
                out[c][i] = seg

    bad = sorted({n.split('_slice')[0] for n, x in zip(names, r) if x < 0.9})
    print(f'alignment: cached vs recomputed MFCC 0, median r = {np.nanmedian(r):.4f}, min {np.nanmin(r):.3f}, '
          f'{np.sum(r < 0.9)} slices below .9 (songs {bad}), {np.isnan(r).sum()} constant slices')

    np.savez(OUT, names=np.array(names), **out)
    print('saved', OUT)


if __name__ == '__main__':
    main()
