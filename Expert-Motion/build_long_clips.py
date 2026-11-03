"""Long training clips for the long-clip musicality fine-tune (2026-11-03).

One-shot MACE matches real dancers' music coupling up to ~17 s and falls behind from ~20 s
(Evaluation-Musicality/length_curve.py), while it trains only on 8 s slices. This builds 34 s training clips
(1020 frames, the test length) so the loss L_EDGE + lambda [1 - r(slow treble, slow body energy)] sees that regime.

Motion: not stored again. The 8 s training slices <song>_slice<k> start every 1 s (30 frames) and are identical where
they overlap (checked: max difference 0), so a clip that starts at slice k is assembled on the fly from slices
k .. k+26 (dataset/long_clips.py). Only runs of consecutive slice indices are used.

Music features: EDGE's 35-dim baseline features (onset envelope, 20 MFCC, 12 chroma CENS, onset peaks, beats),
recomputed on each 34 s wav the way the test slices were made: librosa.load(sr=None) mono, slice, sf.write,
then librosa at 15360 Hz with the resampy kaiser_best resampler (the old librosa default). Check (2026-11-03): this
reproduces the shipped test features exactly (env / MFCC / chroma error 0.0, beats agree on >= 99.8% of frames)
and the shipped 8 s training features to 0.03 MFCC units. With librosa 0.11's default soxr resampler, the MFCCs
were off by about 5 units.

Music signals for the loss: loud / perc / treble from the whole song (precompute_music_signals.song_signals), cut at
the clip's offset, as for the 8 s slices.

    cd Expert-Motion
    .venv/bin/python build_long_clips.py            # 1020 frames, a clip every 4 s

Writes data/dataset_backups/train_long1020.npz: names, slices (N, 27) indices into train_tensor_dataset.pkl,
genres, features (N, 1020, 35), loud / perc / treble (N, 1020).
"""
import argparse
import os
import pickle
import sys
import tempfile
import warnings
from collections import defaultdict
from multiprocessing import Pool

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.getcwd())
from precompute_music_signals import COMPONENTS, RELEASE, song_signals  # noqa: E402

warnings.filterwarnings('ignore')
FPS, HOP, SLICE, STEP = 30, 512, 240, 30
SR = FPS * HOP


def baseline_features(path):
    """EDGE data/audio_extraction/baseline_features.extract, on a wav file. (T + 1, 35)."""
    y, _ = librosa.load(path, sr=SR, res_type='kaiser_best')
    env = librosa.onset.onset_strength(y=y, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP, n_chroma=12).T
    peaks = librosa.onset.onset_detect(onset_envelope=env.flatten(), sr=SR, hop_length=HOP)
    peak = np.zeros_like(env, dtype=np.float32)
    peak[peaks] = 1
    start_bpm = librosa.beat.tempo(y=librosa.load(path, res_type='kaiser_best')[0])[0]
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP, start_bpm=start_bpm, tightness=100)
    beat = np.zeros_like(env, dtype=np.float32)
    beat[beats] = 1
    return np.concatenate([env[:, None], mfcc, chroma, peak[:, None], beat[:, None]], -1).astype(np.float32)


def song_clips(job):
    """All clips of one song: (song, [(start_slice, features (T, 35))], music signals of the whole song)."""
    song, starts, frames = job
    audio, sr = librosa.load(os.path.join(RELEASE, 'music_wav', song + '.wav'), sr=None)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in starts:
            seg = audio[k * sr:k * sr + frames // FPS * sr]
            path = os.path.join(tmp, f'{song}_long{k}.wav')
            sf.write(path, seg, sr)
            f = baseline_features(path)
            assert len(f) >= frames, (song, k, len(f))
            out.append((k, f[:frames]))
    return song, out, song_signals(song)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=1020)
    ap.add_argument('--clip_step', type=int, default=4, help='seconds between clip starts')
    ap.add_argument('--workers', type=int, default=20)
    args = ap.parse_args()
    n_slices = (args.frames - SLICE) // STEP + 1
    assert (args.frames - SLICE) % STEP == 0

    ds = pickle.load(open('data/dataset_backups/train_tensor_dataset.pkl', 'rb')).data
    names = [os.path.splitext(os.path.basename(w))[0] for w in ds['wavs']]
    index = {n: i for i, n in enumerate(names)}
    by_song = defaultdict(set)
    for n in names:
        s, k = n.split('_slice')
        by_song[s].add(int(k))
    jobs = []
    for s, ks in sorted(by_song.items()):
        starts = [k for k in sorted(ks) if all(k + j in ks for j in range(n_slices))]
        # one clip every clip_step seconds along each run of consecutive slices
        keep, last = [], None
        for k in starts:
            if last is None or k - last >= args.clip_step or k - 1 not in starts:
                keep.append(k)
                last = k
        if keep:
            jobs.append((s, keep, args.frames))
    print(f'{sum(len(j[1]) for j in jobs)} clips of {args.frames} frames from {len(jobs)} songs', flush=True)

    rows = []
    with Pool(args.workers) as pool:
        for i, (song, clips, music) in enumerate(pool.imap_unordered(song_clips, jobs)):
            for k, f in clips:
                o = k * FPS
                sig = {c: music[c][o:o + args.frames] for c in COMPONENTS}
                assert all(len(v) == args.frames for v in sig.values()), (song, k)
                sl = [index[f'{song}_slice{k + j}'] for j in range(n_slices)]
                rows.append((f'{song}_long{k}', sl, int(ds['genres'][sl[0]]), f, sig))
            print(f'{i + 1}/{len(jobs)} songs', flush=True)
    rows.sort(key=lambda r: r[0])
    out = os.path.join('data/dataset_backups', f'train_long{args.frames}.npz')
    np.savez(out, names=np.array([r[0] for r in rows]), slices=np.array([r[1] for r in rows]),
             genres=np.array([r[2] for r in rows]), features=np.stack([r[3] for r in rows]),
             **{c: np.stack([r[4][c] for r in rows]).astype(np.float32) for c in COMPONENTS})
    print('saved', out, len(rows), 'clips')


if __name__ == '__main__':
    main()
