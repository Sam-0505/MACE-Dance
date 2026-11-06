"""Generate EDGE (official AIST++ checkpoint) dances for the 289 Motorica 34 s music clips.

Run from the EDGE repo root with its own venv:
    cd /home/sam0505/Documents/Code/dance_baselines/EDGE
    .venv/bin/python gen_motorica.py                 # all clips, seeds 0 1 2 (resumable)
    .venv/bin/python gen_motorica.py --clips 0 1     # quick test on a few clips
    .venv/bin/python gen_motorica.py --stack_only    # just stack per-clip files into the final array

Pipeline (mirrors EDGE test.py, without its random chunk selection):
  1. wav (34 s, 48 kHz) -> append 1 s of silence (35 s) so EDGE's slicer yields 13 windows
     (5 s windows, 2.5 s stride) -> 150 + 12*75 = 1050 frames >= 1020.
  2. EDGE's slice_audio + EDGE's Jukebox extractor (jukemirlib layer 66, 30 fps) per 5 s slice,
     exactly like test.py; features cached per clip in motorica_out/feats/.
  3. EDGE.render_sample(..., mode="long") = long_ddim_sample (50 DDIM steps, CFG weight 2,
     half-window inpainting) + EDGE's pos crossfade / quaternion slerp stitching + EDGE's SMPL FK.
     Video rendering is disabled by replacing model.diffusion.skeleton_render with a no-op.
  4. Crop to the first 1020 frames; save per clip motorica_out/joints/{i:03d}.npy (3,1020,24,3) f32.
  5. Stack -> MACE-Dance/Evaluation-Musicality/data/motorica/edge/joints.npy (289,3,1020,24,3) f16.

Seeding: sample for (seed s, clip i) uses torch.manual_seed(s * 1000 + i) (distinct noise per clip,
same convention as gen_mace_motorica.py's s*1000+b).
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import time
from tempfile import TemporaryDirectory

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import jukemirlib  # noqa: E402

# keep the Jukebox weights inside the EDGE folder instead of ~/.cache/jukemirlib
jukemirlib.constants.CACHE_DIR = os.path.join(ROOT, "jukebox_cache")

import soundfile as sf  # noqa: E402
import librosa as lr  # noqa: E402
import torch  # noqa: E402

import model.diffusion as edge_diffusion  # noqa: E402
from data.slice import slice_audio  # noqa: E402
from data.audio_extraction.jukebox_features import extract as juke_extract  # noqa: E402
from test import stringintkey  # noqa: E402

MOT = "/home/sam0505/Documents/Code/MACE-Dance/Evaluation-Musicality/data/motorica"
WAV_DIR = os.path.join(MOT, "slices_wav")
OUT_DIR = os.path.join(MOT, "edge")
WORK = os.path.join(ROOT, "motorica_out")
N_FRAMES = 1020
FPS = 30
SEEDS = (0, 1, 2)
PAD_SEC = 1.0  # 34 s + 1 s silence = 35 s -> 13 EDGE windows


def clip_features(i, tmp):
    """Jukebox features for clip i, shape (13, 150, 4800), plus the slice wav paths (EDGE order)."""
    feat_path = os.path.join(WORK, "feats", f"{i:03d}.npy")
    wav = os.path.join(WAV_DIR, f"{i:03d}.wav")
    audio, sr = lr.load(wav, sr=None)
    assert sr == 48000 and abs(len(audio) / sr - 34.0) < 1e-3, (wav, sr, len(audio))
    padded = np.concatenate([audio, np.zeros(int(PAD_SEC * sr), dtype=audio.dtype)])
    pwav = os.path.join(tmp, f"{i:03d}.wav")
    sf.write(pwav, padded, sr)
    n = slice_audio(pwav, 2.5, 5.0, tmp)
    files = sorted(glob.glob(f"{tmp}/{i:03d}_slice*.wav"), key=stringintkey)
    assert n == 13 and len(files) == 13, (n, files)
    if os.path.exists(feat_path):
        feats = np.load(feat_path)
    else:
        reps = []
        for f in files:
            r, _ = juke_extract(f)  # jukemirlib layer 66 @ 30 fps, as in EDGE test.py
            reps.append(r)
        feats = np.array(reps, dtype=np.float32)
        np.save(feat_path + ".tmp.npy", feats)
        os.replace(feat_path + ".tmp.npy", feat_path)
    assert feats.shape == (13, 150, 4800), feats.shape
    return feats, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, nargs="*", default=None)
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "ckpt", "checkpoint.pt"))
    ap.add_argument("--stack_only", action="store_true")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(WAV_DIR, "slices.json")))
    n_clips = len(meta["slices"])
    clips = args.clips if args.clips is not None else list(range(n_clips))
    os.makedirs(os.path.join(WORK, "feats"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "joints"), exist_ok=True)

    if not args.stack_only:
        from EDGE import EDGE

        # no video rendering / stitched-wav side effects; everything else is EDGE's own code path
        edge_diffusion.skeleton_render = lambda *a, **k: None
        model = EDGE("jukebox", args.checkpoint)
        model.eval()

        for i in clips:
            out_path = os.path.join(WORK, "joints", f"{i:03d}.npy")
            if os.path.exists(out_path):
                continue
            t0 = time.time()
            with TemporaryDirectory() as tmp:
                feats, files = clip_features(i, tmp)
                t1 = time.time()
                cond = torch.from_numpy(feats)
                res = []
                for s in SEEDS:
                    torch.manual_seed(s * 1000 + i)
                    fk_dir = os.path.join(tmp, "fk")
                    model.render_sample((None, cond, files), f"s{s}", os.path.join(tmp, "render"),
                                        render_count=-1, fk_out=fk_dir, render=False)
                    pk = os.path.join(fk_dir, f"s{s}_{i:03d}.pkl")
                    full_pose = pickle.load(open(pk, "rb"))["full_pose"]
                    os.remove(pk)
                    assert full_pose.shape == (1050, 24, 3), full_pose.shape
                    res.append(full_pose[:N_FRAMES])
                res = np.stack(res).astype(np.float32)
                assert np.isfinite(res).all()
                np.save(out_path + ".tmp.npy", res)
                os.replace(out_path + ".tmp.npy", out_path)
            print(f"clip {i:03d}: feats {t1 - t0:.1f}s gen {time.time() - t1:.1f}s "
                  f"speed {np.linalg.norm(np.diff(res, axis=1), axis=-1).mean() * FPS:.3f} m/s "
                  f"peak_gpu {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)

    # stack when all clips are present
    paths = [os.path.join(WORK, "joints", f"{i:03d}.npy") for i in range(n_clips)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"{len(missing)} clips missing; not stacking")
        return
    allj = np.stack([np.load(p) for p in paths]).astype(np.float16)
    assert allj.shape == (n_clips, len(SEEDS), N_FRAMES, 24, 3) and np.isfinite(allj).all()
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "joints.npy"), allj)
    shutil.copyfile(os.path.join(WAV_DIR, "slices.json"), os.path.join(OUT_DIR, "slices.json"))
    print("wrote", os.path.join(OUT_DIR, "joints.npy"), allj.shape, allj.dtype)


if __name__ == "__main__":
    main()
