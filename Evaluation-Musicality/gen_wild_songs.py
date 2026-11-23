"""Dances for user-supplied songs (2026-11-21): MACE-Dance released (oneshot) and ours (critic v5 + gain), 10 seeds, for user-study videos.

Each song is decoded with ffmpeg, a 34 s segment starting PRE seconds before the requested start is cut (the lock metric trims 4 s at both
ends, so the shown window starts at frame PRE*30 = the requested start), and EDGE's baseline features are computed on that segment
(build_long_clips.baseline_features, the same extractor as the FineDance long clips). The custom slices are generated IN ONE BATCH with
the 84 FineDance test slices, because ours' critic guidance contrasts each slice with the other songs in the batch (critic_groups);
the settings are those of the evaluated runs (MACE released: MaceSampler.sample; ours: local attention 2 s, lock_guide 0.003, t_max 600,
root-relative, foot 10 iters mu 0.1, frozen root rotation, critic v5, gain_ref 0.49, grad_smooth 0 = export_smpl_gain CANDIDATES[0]).
MACE / ours / FlowerDance all need a FineDance genre label; --genre sets it for every song.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/gen_wild_songs.py --ffmpeg $FF --genre Choreography \
        --song dilruba "/path/Dilruba Dilnasheen.mp3" 0:48 --song mainepayal "/path/Maine Payal Hai Chhankai.mp3" 0:36

Writes Evaluation-Musicality/data/wild/<name>/{segment.wav, feats.npy, meta.json} and data/wild/rot/{oneshot,gain}.npz:
poses (n, S, 1020, 72) axis-angle, trans (n, S, 1020, 3), joints (n, S, 1020, 24, 3) float16, names (n,). Z-up like rot_export.
FlowerDance: baselines/flower_gen_wild.py (its own process: FlowerDance's own `dataset` package clashes with Expert-Motion's).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
WILD = os.path.join(HERE, 'data', 'wild')
T, FPS = 1020, 30


def seconds(s):
    p = [float(x) for x in str(s).split(':')]
    return sum(v * 60 ** k for k, v in enumerate(reversed(p)))


def cut_song(name, path, start, pre, ffmpeg):
    """segment wav (start - pre .. + 34 s) and its (1020, 35) baseline features."""
    import soundfile as sf
    from build_long_clips import baseline_features
    d = os.path.join(WILD, name)
    os.makedirs(d, exist_ok=True)
    s0 = max(start - pre, 0.0)
    seg = os.path.join(d, 'segment.wav')
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-ss', f'{s0:.3f}', '-t', f'{T / FPS + 0.5:.3f}', '-i', path, '-ac', '1', '-ar', '44100', seg], check=True)
    y, sr = sf.read(seg)
    assert len(y) >= T / FPS * sr, f'{name}: song too short after {s0:.1f} s ({len(y) / sr:.1f} s left)'
    with tempfile.TemporaryDirectory() as tmp:  # features on exactly 34 s, as the FineDance clips
        p = os.path.join(tmp, 'x.wav')
        sf.write(p, y[:int(round(T / FPS * sr))], sr)
        f = baseline_features(p)[:T]
    assert len(f) == T, (name, f.shape)
    np.save(os.path.join(d, 'feats.npy'), f)
    beats = np.flatnonzero(f[:, 34])
    bpm = 60 * FPS / np.polyfit(np.arange(len(beats)), beats, 1)[0]
    meta = {'name': name, 'source': path, 'requested_start_s': start, 'segment_start_s': s0, 'window_start_in_segment_s': start - s0,
            'bpm': round(float(bpm), 1), 'n_beats': int(len(beats))}
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
    print(meta, flush=True)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--song', nargs=3, action='append', metavar=('NAME', 'PATH', 'START'), required=True)
    ap.add_argument('--genre', default='Choreography')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--pre', type=float, default=4.0)
    ap.add_argument('--ffmpeg', default='ffmpeg')
    ap.add_argument('--models', nargs='+', default=['oneshot', 'gain'], choices=['oneshot', 'gain'])
    a = ap.parse_args()
    from critic import load_critic
    from export_smpl_gain import make_sampler
    from dataset.quaternion import ax_from_6v
    from model.metric import style2s
    assert a.genre in style2s, style2s

    wild = [cut_song(n, p, seconds(s), a.pre, a.ffmpeg) for n, p, s in a.song]
    wnames = [n for n, _, _ in a.song]
    meta = json.load(open(os.path.join(HERE, 'data', 'mace_test', 'names.json')))
    tnames, tgen = meta['names'], meta['genres']
    tfeats = [np.load(os.path.join('data', 'test', 'baseline_feats', n + '.npy')) for n in tnames]
    feats = np.stack(tfeats + wild).astype(np.float32)
    genres = tgen + [a.genre] * len(wild)
    groups = [n.split('_')[0] for n in tnames] + [f'wild_{n}' for n in wnames]
    idx = list(range(len(tnames), len(tnames) + len(wild)))
    os.makedirs(os.path.join(WILD, 'rot'), exist_ok=True)

    for m in a.models:
        sampler, keep = make_sampler(m == 'gain')
        critic = load_critic() if m == 'gain' else None
        P, X, J = [], [], []
        for s in range(a.seeds):
            if m == 'oneshot':
                j = sampler.sample(feats, genres, seed=s)
            else:
                j = sampler.sample_guided(feats, genres, seed=s, scale=0.003, t_max=600, root_relative=True, guide_shift=0, foot_iters=10,
                                          foot_mu=0.1, freeze_root_rot=True, critic=critic, critic_groups=groups, grad_smooth=0.0, gain_ref=0.49)
            x = sampler.model.normalizer.unnormalize(keep['x'][idx])
            _, x = torch.split(x, (4, x.shape[2] - 4), dim=2)
            P.append(ax_from_6v(x[:, :, 3:].reshape(len(idx), T, 24, 6)).reshape(len(idx), T, 72).numpy())
            X.append(x[:, :, :3].numpy())
            J.append(np.asarray(j)[idx].astype(np.float16))
            print(f'{m} seed {s} done, finite {np.isfinite(J[-1]).all()}', flush=True)
        np.savez(os.path.join(WILD, 'rot', f'{m}.npz'), poses=np.stack(P, 1), trans=np.stack(X, 1), joints=np.stack(J, 1),
                 names=np.array(wnames), genre=a.genre)
        print('saved', m, np.stack(J, 1).shape, flush=True)
        del sampler
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
