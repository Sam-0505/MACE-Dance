"""Generate Lodge (official FineDance checkpoint) dances for the 289 Motorica music slices.
# FineDance test copy of gen_motorica.py (2026-11-06): 84 test slices (data/finedance_test_wav), 10 seeds, output data/mace_test_<model>.

Run from the LODGE repo root with its venv:
    cd /home/sam0505/Documents/Code/dance_baselines/LODGE
    .venv/bin/python -W ignore gen_motorica.py --stage feats      # Lodge's own 35-d librosa features (CPU)
    .venv/bin/python -W ignore gen_motorica.py --stage gen        # global + local diffusion, 3 seeds per clip
    .venv/bin/python -W ignore gen_motorica.py --stage joints     # FK -> (289, S, 1020, 24, 3) float16

Pipeline per (clip i, sample s) = infer_lodge.py test() for one song, unchanged in substance:
  * music features: dld.data.utils.audio.extract (librosa 0.7.2, 30 fps, 35-d) on the wav.
  * Lodge cuts the features to a multiple of 256 frames (length2). A 34 s clip gives 1021 feature frames ->
    only 3 local windows (768 frames). To cover the whole clip we append 3 frames (copy of the last feature
    frame with onset-peak/beat one-hots zeroed) so T = 1024 = exactly one global window (length1) of 4 local
    windows; the output is then cropped to 1020 frames.
  * global stage: Global_Module.render_sample_ori(..., setmode="normal")  (DDIM 50 steps) -> 104 key frames.
  * local stage: Local_Module.render_sample(..., setmode="inpaint_soft_ddim", soft_hint='dod', genre) with
    cfg.soft = 1.0 (README command), then concatenation of the 4 local windows as in concat_res.py.
  * seed: torch/numpy/random seeded with 1000*s + i before each (clip, sample).
Raw 1024x139 outputs (4 contact + 3 root pos + 22x6D) are kept in motorica_work/raw/{i:03d}_s{s}.npy.

Joints: 22 Lodge body rotations (SMPL-X body = SMPL joints 0-21) + zero rotations for SMPL joints 22/23 (hand
joints; their rotations do not move any of the 24 joint positions), root position = Lodge root position,
FK with the EDGE/MACE SMPLSkeleton offsets and then align_axis, i.e. exactly what
Evaluation-Musicality/finedance_common.release_joints does for FineDance (eval_mace.fk, from_pred=False).
"""
import argparse
import json
import os
import random
import shutil
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOTORICA = '/home/sam0505/Documents/Code/MACE-Dance/Evaluation-Musicality/data'
WAV_DIR = os.path.join(MOTORICA, 'finedance_test_wav')
OUT_DIR = os.path.join(MOTORICA, 'mace_test_lodge')
OUT_DIR_OFFICIAL = os.path.join(MOTORICA, 'mace_test_lodge_official')
WORK = os.path.join(HERE, 'finedance_work')
N_CLIPS, N_FRAMES, SEEDS = 84, 1020, tuple(range(10))
LENGTH1, LENGTH2 = 1024, 256
GLOBAL_CFG = 'exp/Global_Module/FineDance_Global/global_train.yaml'
LOCAL_CFG = 'exp/Local_Module/FineDance_FineTuneV2_Local/local_train.yaml'
GLOBAL_CKPT = 'exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt'
LOCAL_CKPT = 'exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt'
GENRE_MAP = {g: g for g in ('Breaking', 'Popping', 'Locking', 'Hiphop', 'Urban', 'Jazz', 'HanTang', 'ShenYun', 'Choreography', 'Miao', 'Dai', 'Wei', 'Korean')}  # FineDance style2


def styles():
    with open(os.path.join(WAV_DIR, 'slices.json')) as f:
        return json.load(f)['mace_style']


# ----------------------------------------------------------------------------------------------- features
def _feat_one(i):
    from dld.data.utils.audio import extract
    path = os.path.join(WORK, 'feats', f'{i:03d}.npy')
    if os.path.exists(path):
        return i
    fea, _ = extract(fpath=os.path.join(WAV_DIR, f'{i:03d}.wav'), skip_completed=False)
    np.save(path, fea.astype(np.float32))
    return i


def stage_feats(workers):
    import multiprocessing as mp
    os.makedirs(os.path.join(WORK, 'feats'), exist_ok=True)
    with mp.Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_feat_one, range(N_CLIPS))):
            if k % 50 == 0:
                print('feats', k, flush=True)


def pad_to_window(fea):
    """Append frames (last frame, peak/beat one-hots = 0) up to a multiple of LENGTH2 covering N_FRAMES."""
    need = int(np.ceil(max(len(fea), N_FRAMES) / LENGTH2) * LENGTH2)
    if len(fea) < need:
        pad = np.repeat(fea[-1:], need - len(fea), axis=0)
        pad[:, -2:] = 0.0
        fea = np.concatenate([fea, pad], axis=0)
    return fea[:need]


# ----------------------------------------------------------------------------------------------- generation
def build_models(device):
    sys.modules.setdefault('pyrender', types.ModuleType('pyrender'))   # render.py imports it; no OSMesa here
    import torch
    from omegaconf import OmegaConf
    from dld.models.get_model import get_module

    cfg = OmegaConf.merge(OmegaConf.load(LOCAL_CFG), OmegaConf.load('configs/data/assets.yaml'))
    cfg.soft = 1.0
    cfg.DEVICE = [0]
    cfg.length1, cfg.length2 = LENGTH1, LENGTH2
    cfg_coarse = OmegaConf.load(GLOBAL_CFG)
    cfg_coarse.DEVICE = [0]
    # global_train.yaml points at the author's /data2/.../lodge302/data/Normalizer.pth (same file the local
    # config uses); the repo ships it as data/Normalizer.pth, which assets.yaml also uses for the local model.
    cfg_coarse.DATASET.FINEDANCE_139CUT.normalizer = 'data/Normalizer.pth'

    models = []
    for c, ck in ((cfg_coarse, GLOBAL_CKPT), (cfg, LOCAL_CKPT)):
        m = get_module(c, None)            # the datamodule is not used for sampling
        m.load_state_dict(torch.load(ck, map_location='cpu')['state_dict'], strict=True)
        models.append(m.to(device).eval())
    return cfg, models[0], models[1]


def generate_one(cfg, model_coarse, model_fine, fea, genre_id, tmp, device):
    """infer_lodge.test() for one song with one global window; returns (T, 139)."""
    import torch
    from infer_lodge import load_modata

    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    file_name = 'clip'
    local_num = fea.shape[0] // LENGTH2
    ratio = LENGTH1 // LENGTH2
    global_num = int(np.ceil(local_num / ratio))
    assert global_num == 1 and local_num == ratio, 'only the single-global-window case is used here'

    music_fea_cat = torch.from_numpy(fea).to(device).unsqueeze(0)
    names = [file_name + 'g000g']
    with torch.no_grad():
        model_coarse.render_sample_ori((None, music_fea_cat, names), 'global', tmp, render_count=-1,
                                       fk_out=tmp, render=False, setmode='normal', device=device)
    modata_13 = load_modata(os.path.join(tmp, 'global_0_' + file_name + 'g000g.npy'), device)
    modata = modata_13[4:-4]
    scale = 8 * (ratio - 1)
    molist = [modata[k * scale:(k + 1) * scale] for k in range(modata.shape[0] // scale)]
    local_names = [file_name + 'g000g_l' + str(k).zfill(3) for k in range(len(molist))]
    music_loc = music_fea_cat.reshape(-1, LENGTH2, 35)[:local_num]
    genre = torch.from_numpy(np.array(genre_id)).unsqueeze(0)
    with torch.no_grad():
        model_fine.render_sample((None, music_loc, local_names, molist), 'dod', tmp, render_count=-1, fk_out=tmp,
                                 render=False, setmode='inpaint_soft_ddim', cons=molist, soft_hint='dod',
                                 device=device, genre=genre)
    parts = [np.load(os.path.join(tmp, f'dod_{k}_{n}.npy')) for k, n in enumerate(local_names)]
    return np.concatenate(parts, axis=0), modata_13


def stage_gen(clips):
    import contextlib
    import io
    import torch
    from dld.data.FineDance_dataset import Genres_fd

    device = 'cuda:0'
    os.makedirs(os.path.join(WORK, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(WORK, 'global'), exist_ok=True)
    cfg, mc, mf = build_models(device)
    st = styles()
    tmp = os.path.join(WORK, 'tmp')
    for i in clips:
        for s in SEEDS:
            out = os.path.join(WORK, 'raw', f'{i:03d}_s{s}.npy')
            if os.path.exists(out):
                continue
            fea = pad_to_window(np.load(os.path.join(WORK, 'feats', f'{i:03d}.npy')))
            seed = 1000 * s + i
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            with contextlib.redirect_stdout(io.StringIO()):      # Lodge prints a lot
                dance, glob = generate_one(cfg, mc, mf, fea, Genres_fd[GENRE_MAP[st[i]]], tmp, device)
            assert dance.shape == (LENGTH1, 139) and np.isfinite(dance).all(), dance.shape
            np.save(os.path.join(WORK, 'global', f'{i:03d}_s{s}.npy'), glob.astype(np.float32))
            np.save(out + '.tmp.npy', dance.astype(np.float32))
            os.replace(out + '.tmp.npy', out)
        print(f'clip {i} done ({st[i]})', flush=True)


# ----------------------------------------------------------------------------------------------- joints
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SMPL_OFFSETS = [   # Expert-Motion/vis.py smpl_offsets (EDGE)
    [0.0, 0.0, 0.0], [0.05858135, -0.08228004, -0.01766408], [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522], [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304], [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428], [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505], [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329], [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817], [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987], [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207], [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873], [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676], [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708]]


def smpl_fk(rot6d22, root):
    """rot6d22 (B, T, 22, 6), root (B, T, 3) -> (B, T, 24, 3); SMPLSkeleton.forward + align_axis."""
    import torch
    from pytorch3d.transforms import (RotateAxisAngle, matrix_to_quaternion, quaternion_apply,
                                      quaternion_multiply, rotation_6d_to_matrix)
    q = matrix_to_quaternion(rotation_6d_to_matrix(rot6d22))
    ident = torch.zeros(q.shape[:2] + (2, 4), dtype=q.dtype); ident[..., 0] = 1
    q = torch.cat([q, ident], dim=2)
    off = torch.tensor(SMPL_OFFSETS, dtype=q.dtype)
    pos, rotw = [], []
    for j, p in enumerate(SMPL_PARENTS):
        if p == -1:
            pos.append(root); rotw.append(q[:, :, 0])
        else:
            pos.append(quaternion_apply(rotw[p], off[j].expand_as(root)) + pos[p])
            rotw.append(quaternion_multiply(rotw[p], q[:, :, j]))
    x = torch.stack(pos, dim=2).float()
    rot = RotateAxisAngle(90, axis='z', degrees=True)
    return rot.transform_points(x.reshape(-1, 3)).reshape(x.shape)


def root_continuous(dance):
    """Optional: shift the root x/z of local windows 2..4 so each continues from the previous one."""
    d = dance.copy()
    for k in range(1, len(d) // LENGTH2):
        a = k * LENGTH2
        vel = d[a - 1, [4, 6]] - d[a - 2, [4, 6]]
        d[a:, [4, 6]] += (d[a - 1, [4, 6]] + vel) - d[a, [4, 6]]
    return d


def stage_joints(fix_root):
    import torch
    out = np.zeros((N_CLIPS, len(SEEDS), N_FRAMES, 24, 3), dtype=np.float16)
    for i in range(N_CLIPS):
        raw = []
        for s in SEEDS:
            d = np.load(os.path.join(WORK, 'raw', f'{i:03d}_s{s}.npy'))
            raw.append(root_continuous(d) if fix_root else d)
        raw = torch.from_numpy(np.stack(raw)[:, :N_FRAMES]).double()
        j = smpl_fk(raw[:, :, 7:].reshape(len(SEEDS), N_FRAMES, 22, 6), raw[:, :, 4:7])
        out[i] = j.numpy().astype(np.float16)
    assert np.isfinite(out).all()
    dest = OUT_DIR if fix_root else OUT_DIR_OFFICIAL
    os.makedirs(dest, exist_ok=True)
    np.save(os.path.join(dest, 'joints.tmp.npy'), out)
    os.replace(os.path.join(dest, 'joints.tmp.npy'), os.path.join(dest, 'joints.npy'))
    shutil.copy(os.path.join(WAV_DIR, 'slices.json'), os.path.join(dest, 'slices.json'))
    print('saved', os.path.join(dest, 'joints.npy'), out.shape)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['feats', 'gen', 'joints'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--clips', type=str, default='0-83', help='e.g. 0-288 or 0,5,7')
    ap.add_argument('--no_fix_root', action='store_true', help='joints from the plain official concatenation')
    a = ap.parse_args()
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    if '-' in a.clips:
        lo, hi = map(int, a.clips.split('-')); clips = list(range(lo, hi + 1))
    else:
        clips = [int(c) for c in a.clips.split(',')]
    if a.stage == 'feats':
        stage_feats(a.workers)
    elif a.stage == 'gen':
        stage_gen(clips)
    else:
        stage_joints(not a.no_fix_root)
