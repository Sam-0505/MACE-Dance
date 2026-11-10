"""Drum-hit test (2026-11-09): give the models a plain drum track instead of a song and see where the limbs speed up.

Track: 112 BPM (the tempo of 161_slice2), 34 s, kick on every beat, snare on beats 2 and 4, first beat at 0.40 s. Features:
EDGE's 35-dim baseline features (build_long_clips.baseline_features, copied) of the track, so the models see what they see
for a song. MACE released (one shot, as data/mace_test) and ours (mask 2 s + critic v5 + foot + frozen root + gain 0.49,
as in export_smpl_gain.py), seeds 0..SEEDS-1, genre and critic group of --clip (default 161_slice2). Per seed: speed of the
four limbs (root-relative, z-scored) at the beat grid column 34 of the track features vs shift of the grid
(z at the beat, best shift, depth/2 = (max - min)/2), as render_beads.py; and steady accent stretches (R, mu).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/gen_drum_test.py [--seeds 10] [--analyse_only]

Writes data/drum_test/drums112.wav, drums112_feats.npy, <model>.npy (seeds, T, 24, 3), results/drum_test.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
OUT = os.path.join(HERE, 'data', 'drum_test')
FPS, HOP = 30, 512
SR = FPS * HOP
LIMBS = [7, 8, 20, 21]
SHIFTS = np.linspace(0, 0.95, 20)


def drum_track(bpm=112.0, dur=34.0, first=0.40, sr=44100):
    y = np.zeros(int(dur * sr) + sr)
    rng = np.random.RandomState(0)
    t = np.arange(int(0.45 * sr)) / sr
    kick = np.sin(2 * np.pi * np.cumsum(50 + 110 * np.exp(-t * 35)) / sr) * np.exp(-t * 8)
    ts = np.arange(int(0.25 * sr)) / sr
    snare = (rng.randn(len(ts)) * 0.6 + np.sin(2 * np.pi * 190 * ts) * 0.5) * np.exp(-ts * 22)
    for k, b in enumerate(np.arange(first, dur, 60 / bpm)):
        i = int(b * sr)
        y[i:i + len(kick)] += kick
        if k % 4 in (1, 3):
            y[i:i + len(snare)] += snare * 0.8
    y = y[:int(dur * sr)]
    return (0.9 * y / np.abs(y).max()).astype(np.float32), sr


def baseline_features(path):
    """EDGE data/audio_extraction/baseline_features.extract, on a wav file (build_long_clips.baseline_features). (T + 1, 35)."""
    import librosa
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


def limb_speed(j):
    r = np.asarray(j, np.float32)[:, LIMBS] - np.asarray(j, np.float32)[:, :1]
    v = gaussian_filter1d(np.linalg.norm(np.gradient(r, axis=0) * FPS, axis=-1), 1.0, axis=0).mean(1)
    return (v - v.mean()) / (v.std() + 1e-8)


def beat_profile(j, A, iv):
    z = limb_speed(j)
    T = len(z)
    prof = np.array([z[np.clip(np.round(A + s * iv).astype(int), 0, T - 1)].mean() for s in SHIFTS])
    return {'z0': float(prof[0]), 'best': float(SHIFTS[prof.argmax()]), 'depth': float((prof.max() - prof.min()) / 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', default='161_slice2')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--models', nargs='+', default=['released', 'ours'])
    ap.add_argument('--analyse_only', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    wav = os.path.join(OUT, 'drums112.wav')
    fpath = os.path.join(OUT, 'drums112_feats.npy')
    if not os.path.exists(fpath):
        y, sr = drum_track()
        sf.write(wav, y, sr)
        f = baseline_features(wav)
        np.save(fpath, f)
    F = np.load(fpath)[:1020]
    A = np.flatnonzero(F[:, 34])
    true_beats = np.arange(0.40, 34, 60 / 112) * FPS
    d = np.abs(A[:, None] - true_beats[None]).min(1)
    iv = float(np.median(np.diff(A)))
    print(f'features {F.shape}: {len(A)} beats in col 34, interval {iv:.2f} frames ({60 * FPS / iv:.1f} BPM); '
          f'distance to the true drum hits: median {np.median(d):.1f} frames, max {d.max():.1f}', flush=True)

    if not a.analyse_only:
        from critic import load_critic
        from length_curve import local_cross_attention
        from mace_sampler import MaceSampler
        genre = json.load(open(f'data/test/label/{a.clip.split("_")[0]}.json'))['style2']
        # critic guidance compares with the music of other songs of the batch (3 closest in beat count), so the drum clip (index 0)
        # goes in a batch with other test clips; only its sample is kept. Released uses the same batch (paired noise).
        others = ['036_slice2', '130_slice2', '012_slice0', '143_slice0']
        F2 = np.stack([F] + [np.load(f'data/test/baseline_feats/{n}.npy')[:1020] for n in others])
        gens = [genre] + [json.load(open(f'data/test/label/{n.split("_")[0]}.json'))['style2'] for n in others]
        groups = [a.clip.split('_')[0]] + [n.split('_')[0] for n in others]
        for m in a.models:
            s = MaceSampler()
            if m == 'ours':
                local_cross_attention(s.model.diffusion.model, 60)
                run = lambda seed: s.sample_guided(F2, gens, seed=seed, scale=0.003, t_max=600, root_relative=True,  # noqa: E731
                                                   guide_shift=0, foot_iters=10, foot_mu=0.1, freeze_root_rot=True,
                                                   gain_ref=0.49, critic=load_critic(), critic_groups=groups)
            else:
                run = lambda seed: s.sample(F2, gens, seed=seed)  # noqa: E731
            out = np.stack([np.asarray(run(sd)[0], np.float16) for sd in range(a.seeds)], 0)
            np.save(os.path.join(OUT, f'{m}.npy'), out)
            print(m, out.shape, flush=True)
            del s
            import torch
            torch.cuda.empty_cache()

    import rank_beat_runs as RB
    res = {}
    print(f'\nspeed of the four limbs at the drum beat (z-score), best shift of the grid (beats), depth/2; steady 8-beat windows')
    print(f'{"model":9s} {"seed":>4s} {"z at beat":>10s} {"best shift":>11s} {"depth/2":>8s}   windows R>=0.75 / with |mu|<=0.1')
    for m in a.models:
        J = np.load(os.path.join(OUT, f'{m}.npy')).astype(np.float32)
        rows = []
        for sd, j in enumerate(J):
            p = beat_profile(j, A, iv)
            w = RB.windows(j, A, 8, 1)
            ok = [x for x in w if x[2] == x[2] and x[2] >= 0.75]
            on = [x for x in ok if abs(x[3]) <= 0.1]
            p.update(seed=sd, n_steady=len(ok), n_on=len(on), windows=w)
            rows.append(p)
            print(f'{m:9s} {sd:4d} {p["z0"]:+10.2f} {p["best"]:11.2f} {p["depth"]:8.2f}   {len(ok)} / {len(on)}')
        print(f'{m:9s} mean z at beat {np.mean([r["z0"] for r in rows]):+.2f}, depth/2 {np.mean([r["depth"] for r in rows]):.2f}, '
              f'seeds within 0.15 beat of the beat: {sum(min(r["best"], 1 - r["best"]) <= 0.15 for r in rows)}/{len(rows)}')
        res[m] = rows
    json.dump(res, open(os.path.join(HERE, 'results', 'drum_test.json'), 'w'), default=float)


if __name__ == '__main__':
    main()
