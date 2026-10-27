"""Generate Motion Expert dances on the AIST++ music segments used by aist_loudness.py.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/gen_mace_aist.py --n 10

One segment per (piece, set): the first T frames of the piece, where T is the length of the real AIST++ dances
on that piece (basic: 16 beats; advanced: 64 beats, capped at 1020 frames = 34 s, the FineDance slice length and
inside MACE's 40 s training horizon). Baseline features are extracted from the segment audio with the piece's
nominal BPM as the beat-tracker start tempo.

AIST genres without a FineDance counterpart are mapped to the closest MACE style label (GENRE_TO_STYLE).
Own- and wrong-music scores use the same MACE dance, so the label does not enter Delta.

Writes Evaluation-Musicality/data/aist/mace/<piece>_<set>.npy: (n, T, 24, 3) float16 joints at 30 fps.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mace_sampler import MaceSampler  # noqa: E402
import audio_features as af  # noqa: E402
from aist_common import DATA, GENRE_TO_STYLE, nominal_bpm, segments  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    out_dir = os.path.join(DATA, 'mace')
    os.makedirs(out_dir, exist_ok=True)

    sampler = MaceSampler()
    for (piece, kind), T in sorted(segments().items()):
        y = af.load_audio(os.path.join(DATA, 'music', piece + '.wav'))
        feat = af.extract_baseline(y[:(T + 1) * af.HOP_LENGTH], start_bpm=nominal_bpm(piece))[:T]
        assert len(feat) == T, (piece, kind, len(feat), T)
        feats = np.repeat(feat[None], args.n, axis=0)
        joints = sampler.sample(feats, [GENRE_TO_STYLE['g' + piece[1:3]]] * args.n, seed=args.seed)
        np.save(os.path.join(out_dir, f'{piece}_{kind}.npy'), joints.astype(np.float16))
        print(piece, kind, joints.shape)


if __name__ == '__main__':
    main()
