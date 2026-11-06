"""Generate MEGADance (official FineDance checkpoints: cls 30, fsq 120, gpt 90) dances for the MACE evaluation clips
(2026-11-06).

    cd /home/sam0505/Documents/Code/dance_baselines/MEGADance
    .venv/bin/python gen_mace_eval.py --dataset finedance     # 84 FineDance test slices -> data/mace_test_mega
    .venv/bin/python gen_mace_eval.py --dataset motorica      # 289 Motorica slices      -> data/motorica/mega
    .venv/bin/python gen_mace_eval.py --dataset motorica --clips 0 1   # quick test

Pipeline = the repo's demo path (demo_gpt.py: createDemoDataset + Trainer.demo), with two changes needed for our clips:
  1. Music features: the repo's extract_librosa (essentia MonoLoader at 15360 Hz, hop 512 -> 30 fps; mfcc 20,
     chroma 12, onset env, onset beat, onset peak = 35 dims). A 34 s clip gives 1021 rows; the demo crops to a multiple
     of 8 (1016 < 1020 frames), so instead the last row is repeated up to 1024 rows (as done for Lodge's features).
  2. Genre: one label per clip instead of the demo's loop over all 16 genres: FineDance = the slice's FineDance style2
     (slices.json mace_style); Motorica = MACE's style label of the take (Hiphop/Jazz/Locking/Popping/Urban, the same
     labels Lodge got).
  Start pose: the demo's Pretrained/Init1.pkl (8 frames). GPT.inference picks the arg-max code at every step
  (infer_result uses torch.max), so the output is deterministic: one dance per clip, copied to every seed slot.
Output: keypoints of Trainer's postprocess (SMPL joints 0-23 from smplx, FineDance Y-up, metres), first 1020 frames,
converted to MACE's Z-up ((x, y, z) -> (x, -z, y)) and align_axis' +90 deg heading about z, like the Bailando and
(fixed) Lodge arrays. joints.npy (N, S, 1020, 24, 3) float16.
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from models.GPT.dataloader import DemoDataset, style2s  # noqa: E402
from models.GPT.trainer import Trainer  # noqa: E402
from utils.librosa_extraction.get_librosa import extract_acoustic_feature  # noqa: E402
from utils.load_model import load_model  # noqa: E402
from utils.utils import seed_everything  # noqa: E402
import essentia.standard  # noqa: E402

DATA = "/home/sam0505/Documents/Code/MACE-Dance/Evaluation-Musicality/data"
SETS = {
    "finedance": dict(wav=os.path.join(DATA, "finedance_test_wav"), out=os.path.join(DATA, "mace_test_mega"),
                      seeds=10, names=os.path.join(DATA, "mace_test", "names.json")),
    "motorica": dict(wav=os.path.join(DATA, "motorica", "slices_wav"), out=os.path.join(DATA, "motorica", "mega"),
                     seeds=3, names=None),
}
SR = 15360
N_FRAMES, N_FEAT = 1020, 1024
BATCH = 16


def features(wav):
    audio = np.array(essentia.standard.MonoLoader(filename=wav, sampleRate=SR)()).T
    f = extract_acoustic_feature(audio, SR)
    assert f.shape[1] == 35 and N_FRAMES <= len(f) <= N_FEAT, f.shape
    return np.concatenate([f, np.repeat(f[-1:], N_FEAT - len(f), 0)], 0)


def to_mace_axes(j):
    j = np.stack([j[..., 0], -j[..., 2], j[..., 1]], -1)  # Y-up -> Z-up
    return np.stack([-j[..., 1], j[..., 0], j[..., 2]], -1)  # align_axis: +90 deg about z


@torch.no_grad()
def run_batch(model, feats, labels, device):
    init = pickle.load(open("Pretrained/Init1.pkl", "rb"))
    n = len(feats)
    ds = DemoDataset(np.repeat(init["smpl_poses"][None], n, 0), np.repeat(init["smpl_trans"][None], n, 0),
                     np.stack(feats), np.array(labels), [f"{k}.wav" for k in range(n)], ROOT)
    data = next(iter(torch.utils.data.DataLoader(ds, batch_size=n, shuffle=False)))
    # Trainer.demo without its file writing
    model.gpt.eval()
    music, _, poses_gt, root_vel_gt, root_init_gt, _, label = model.fsq.preprocess(data, device)
    up, down = model.fsq.pose_encode(poses_gt, root_vel_gt)
    up_p, down_p = model.gpt.inference(music[:, ::8, :], up, down, label)
    root_vel, pose = model.fsq.pose_decode(up_p, down_p)
    _, _, _, kp, _, _ = model.fsq.postprocess(pose, root_vel, root_init_gt)
    kp = kp.cpu().numpy()
    assert kp.shape[1] >= N_FRAMES and kp.shape[2] == 24, kp.shape
    return to_mace_axes(kp[:, :N_FRAMES])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SETS), required=True)
    ap.add_argument("--clips", type=int, nargs="*", default=None)
    args = ap.parse_args()
    s = SETS[args.dataset]
    meta = json.load(open(os.path.join(s["wav"], "slices.json")))
    n = len(meta["slices"])
    styles = meta["mace_style"]
    assert len(styles) == n
    clips = args.clips if args.clips is not None else list(range(n))
    work = os.path.join(ROOT, f"{args.dataset}_out")
    os.makedirs(work, exist_ok=True)

    seed_everything(seed=42)
    device = torch.device("cuda")
    model = Trainer(device=device).to(device)
    model = load_model(model, "gpt", 90)
    model.eval()

    todo = [i for i in clips if not os.path.exists(os.path.join(work, f"{i:03d}.npy"))]
    for b in range(0, len(todo), BATCH):
        idx = todo[b:b + BATCH]
        t0 = time.time()
        feats = [features(os.path.join(s["wav"], f"{i:03d}.wav")) for i in idx]
        t1 = time.time()
        J = run_batch(model, feats, [style2s.index(styles[i]) for i in idx], device)
        for i, j in zip(idx, J):
            np.save(os.path.join(work, f"{i:03d}.npy"), j.astype(np.float32))
        print(f"clips {idx[0]}-{idx[-1]}: features {t1 - t0:.0f}s, generation {time.time() - t1:.0f}s, "
              f"z [{J[..., 2].min():.2f}, {J[..., 2].max():.2f}]", flush=True)

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
