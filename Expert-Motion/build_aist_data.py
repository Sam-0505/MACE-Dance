"""Build the AIST++ training/test data in the FineDance folder layout that AISTPPDataset reads.

Source: data/aist_raw/edge_aistpp (EDGE's copy of the full AIST++ release: 1408 motions + wavs).
Split: EDGE's crossmodal_train / crossmodal_test minus ignore_list (data/aist/splits).
Output (data/aist/{train,test}):
  motions_sliced/<seq>_slice<k>.pkl   {pos: (150,3) metres, q: (150,72)}  5 s at 30 fps (60 fps taken every 2nd frame)
  baseline_feats/<seq>_slice<k>.npy   (150,35) EDGE's librosa features, computed on the whole song, then sliced
  wavs_sliced/<seq>_slice<k>.wav      symlink to the full wav (the dataset only needs the names to match)
  label/<genre>.json                  {"style2": MACE style}, read through the first 3 characters of the file name
Train slices: 5 s window, 0.5 s stride (EDGE). Test slices: 5 s window, 2.5 s stride (EDGE evaluation protocol).
Motion frame 0 = music time 0 (checked in Evaluation-Musicality/readme.md).
"""
import glob
import json
import os
import pickle
import sys
from multiprocessing import Pool

import librosa
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Evaluation-Musicality'))
from aist_common import GENRE_TO_STYLE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'data', 'aist_raw', 'edge_aistpp')
OUT = os.path.join(HERE, 'data', 'aist')
FPS, HOP = 30, 512
SR = FPS * HOP
WIN = 5 * FPS
STRIDE = {'train': FPS // 2, 'test': int(2.5 * FPS)}


def read_list(f):
    return [x.strip() for x in open(f) if x.strip()]


def tempo(name):
    p = name.split('_')[4]
    return int(p[3]) * 5 + 110 if p[:3] == 'mHO' else int(p[3]) * 10 + 80


def features(wav, name):
    y, _ = librosa.load(wav, sr=SR)
    env = librosa.onset.onset_strength(y=y, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP, n_chroma=12).T
    peak = np.zeros_like(env, dtype=np.float32)
    peak[librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP)] = 1.0
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP, start_bpm=tempo(name), tightness=100)
    beat = np.zeros_like(env, dtype=np.float32)
    beat[beats] = 1.0
    return np.concatenate([env[:, None], mfcc, chroma, peak[:, None], beat[:, None]], axis=-1).astype(np.float32)


def do(args):
    split, name = args
    d = pickle.load(open(f'{RAW}/motions/{name}.pkl', 'rb'))
    pos = d['smpl_trans'][::2] / d['smpl_scaling'][0]
    q = d['smpl_poses'][::2]
    f = features(f'{RAW}/wavs/{name}.wav', name)
    n = min(len(pos), len(f))
    k = 0
    for s in range(0, n - WIN + 1, STRIDE[split]):
        stem = f'{name}_slice{k}'
        pickle.dump({'pos': pos[s:s + WIN].astype(np.float32), 'q': q[s:s + WIN].astype(np.float32)},
                    open(f'{OUT}/{split}/motions_sliced/{stem}.pkl', 'wb'))
        np.save(f'{OUT}/{split}/baseline_feats/{stem}.npy', f[s:s + WIN])
        link = f'{OUT}/{split}/wavs_sliced/{stem}.wav'
        if not os.path.lexists(link):
            os.symlink(f'{RAW}/wavs/{name}.wav', link)
        k += 1
    return name, k


if __name__ == '__main__':
    ignore = set(read_list(f'{OUT}/splits/ignore_list.txt'))
    jobs = []
    for split, lst in (('train', 'crossmodal_train'), ('test', 'crossmodal_test')):
        for sub in ('motions_sliced', 'baseline_feats', 'wavs_sliced', 'label'):
            os.makedirs(f'{OUT}/{split}/{sub}', exist_ok=True)
        for g, style in GENRE_TO_STYLE.items():
            json.dump({'style2': style}, open(f'{OUT}/{split}/label/{g}.json', 'w'))
        jobs += [(split, n) for n in read_list(f'{OUT}/splits/{lst}.txt') if n not in ignore]
    print({s: sum(j[0] == s for j in jobs) for s in ('train', 'test')}, flush=True)
    with Pool(20) as p:
        res = p.map(do, jobs, chunksize=4)
    for s in ('train', 'test'):
        print(s, 'slices', len(glob.glob(f'{OUT}/{s}/motions_sliced/*.pkl')))
