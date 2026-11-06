"""Generate Bailando (official AIST++ actor-critic checkpoint) dances for the MACE evaluation clips (2026-11-06).

    cd /home/sam0505/Documents/Code/dance_baselines/Bailando
    .venv/bin/python gen_mace_eval.py --dataset finedance     # 84 FineDance test slices -> data/mace_test_bailando
    .venv/bin/python gen_mace_eval.py --dataset motorica      # 289 Motorica slices      -> data/motorica/bailando
    .venv/bin/python gen_mace_eval.py --dataset motorica --clips 0 1   # quick test

Pipeline (Bailando's own eval path, actor_critic.py AC.eval with configs/actor_critic.yaml):
  1. Music features exactly as _prepro_aistpp_music.py (default 7.5 fps setting): essentia MonoLoader at
     3840 Hz, librosa 0.8.1 mfcc(20) + delta(20) + chroma_cqt(12, 5 octaves) + onset env + onset beat + tempogram
     = 438 dims, hop 512 -> 7.5 fps. load_test_data_aist padding (repeat last row once), then music_seq[:, 1:].
  2. VQ-VAE sep_vqvae_root epoch 500 + GPT actor_critic epoch 10 (the paper's final model). There is no ground-truth
     dance for our music, so the first (up, down) codes are the ones Bailando's own demo mode uses: 423 / 12.
     GPT.sample picks the top-1 code at every step (no sampling), so the output is deterministic: one dance per clip.
     It is copied to every seed slot so the array has the same shape as the other models.
  3. Decode, integrate the root velocity (global_vel) as in eval(), absolute joints = root + offsets
     (visualizeAndWrite). 60 fps -> 30 fps (every other frame), first 1020 frames.
  4. Axes: AIST++ joints are Y-up metres; converted to the Z-up convention of MACE/EDGE (rotate +90 deg about x:
     (x, y, z) -> (x, -z, y)), then the same +90 deg heading rotation about z that align_axis applies to Lodge.
Output joints.npy (N, S, 1020, 24, 3) float16, SMPL joints in metres, 30 fps.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import essentia.standard  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from easydict import EasyDict  # noqa: E402

import models  # noqa: E402
from extractor import FeatureExtractor  # noqa: E402

DATA = "/home/sam0505/Documents/Code/MACE-Dance/Evaluation-Musicality/data"
SETS = {
    "finedance": dict(wav=os.path.join(DATA, "finedance_test_wav"), out=os.path.join(DATA, "mace_test_bailando"),
                      seeds=10, names=os.path.join(DATA, "mace_test", "names.json")),
    "motorica": dict(wav=os.path.join(DATA, "motorica", "slices_wav"), out=os.path.join(DATA, "motorica", "bailando"),
                     seeds=3, names=None),
}
SR = 3840  # 15360 * 2 / 8, the 7.5 fps feature setting
N_FRAMES = 1020
START_CODES = (423, 12)  # Bailando demo mode
extractor = FeatureExtractor()


def music_features(wav):
    audio = np.array(essentia.standard.MonoLoader(filename=wav, sampleRate=SR)()).T
    melspe_db = extractor.get_melspectrogram(audio, SR)
    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)
    audio_harmonic, audio_percussive = extractor.get_hpss(audio)
    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, SR, octave=5)
    onset_env = extractor.get_onset_strength(audio_percussive, SR)
    tempogram = extractor.get_tempogram(onset_env, SR)
    onset_beat = extractor.get_onset_beat(onset_env, SR)[0]
    feat = np.concatenate([mfcc, mfcc_delta, chroma_cqt, onset_env.reshape(1, -1), onset_beat, tempogram], axis=0).T
    assert feat.shape[1] == 438, feat.shape
    return np.concatenate([feat, feat[-1:]], axis=0)  # load_test_data_aist with external_wav_rate 1 adds one row


def load_models(cfg):
    vqvae = nn.DataParallel(models.SepVQVAER(cfg.structure)).cuda().eval()
    gpt = nn.DataParallel(models.CrossCondGPT2AC(cfg.structure_generate)).cuda().eval()
    for net, path in ((vqvae, cfg.vqvae_weight), (gpt, "experiments/actor_critic/ckpt/epoch_10.pt")):
        sd = torch.load(path, map_location="cuda")["model"]
        res = net.load_state_dict(sd, strict=False)
        print(path, "missing", len(res.missing_keys), "unexpected", len(res.unexpected_keys))
        assert not res.missing_keys, res.missing_keys[:5]
    return vqvae, gpt


@torch.no_grad()
def generate(vqvae, gpt, feat, cfg):
    music = torch.from_numpy(feat).float().cuda()[None, :, :cfg.structure_generate.n_music]
    music = music[:, 1:]
    x = ([torch.full((1, 1), START_CODES[0], device="cuda").long()],
         [torch.full((1, 1), START_CODES[1], device="cuda").long()])
    x = tuple(x[i][0][:, :1] for i in range(2))
    zs = gpt.module.sample(x, cond=music)
    pose = vqvae.module.decode(zs)
    vel = pose[:, :, :3].clone()
    pose[:, 0, :3] = 0
    for t in range(1, pose.size(1)):
        pose[:, t, :3] = pose[:, t - 1, :3] + vel[:, t - 1, :]
    d = pose[0].cpu().numpy()  # (T60, 72): root + offsets
    root = d[:, :3].copy()
    d = d + np.tile(root, (1, 24))
    d[:, :3] = root
    j = d.reshape(-1, 24, 3)[::2][:N_FRAMES]  # 60 -> 30 fps
    assert len(j) == N_FRAMES, j.shape
    j = np.stack([j[..., 0], -j[..., 2], j[..., 1]], -1)  # Y-up -> Z-up
    j = np.stack([-j[..., 1], j[..., 0], j[..., 2]], -1)  # align_axis: +90 deg about z
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SETS), required=True)
    ap.add_argument("--clips", type=int, nargs="*", default=None)
    args = ap.parse_args()
    s = SETS[args.dataset]
    meta = json.load(open(os.path.join(s["wav"], "slices.json")))
    n = len(meta["slices"])
    clips = args.clips if args.clips is not None else list(range(n))
    work = os.path.join(ROOT, f"{args.dataset}_out")
    os.makedirs(work, exist_ok=True)

    with open("configs/actor_critic.yaml") as f:
        cfg = EasyDict(yaml.safe_load(f))
    vqvae, gpt = load_models(cfg)
    for i in clips:
        p = os.path.join(work, f"{i:03d}.npy")
        if os.path.exists(p):
            continue
        t0 = time.time()
        feat = music_features(os.path.join(s["wav"], f"{i:03d}.wav"))
        j = generate(vqvae, gpt, feat, cfg)
        np.save(p, j.astype(np.float32))
        print(f"clip {i} feat {feat.shape} -> {j.shape}  z [{j[..., 2].min():.2f}, {j[..., 2].max():.2f}]  "
              f"{time.time() - t0:.1f}s", flush=True)

    if args.clips is None:
        out = np.zeros((n, s["seeds"], N_FRAMES, 24, 3), np.float16)
        for i in range(n):
            out[i] = np.load(os.path.join(work, f"{i:03d}.npy"))[None]
        os.makedirs(s["out"], exist_ok=True)
        np.save(os.path.join(s["out"], "joints.npy"), out)
        shutil.copy(os.path.join(s["wav"], "slices.json"), os.path.join(s["out"], "slices.json"))
        if s["names"]:
            shutil.copy(s["names"], os.path.join(s["out"], "names.json"))
        print("saved", s["out"], out.shape)


if __name__ == "__main__":
    main()
