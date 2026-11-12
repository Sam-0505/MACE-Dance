"""MEGADance: regenerate the 84 FineDance test slices and keep the decoded rotations (2026-11-10).

Same pipeline as gen_mace_eval.py (imported, not copied: features, Trainer, checkpoints, genre = slice style, start
pose Init1), but the postprocess outputs are kept: root translation and the decoded 6D poses (MEGADance's own
representation, Y-up FineDance frame), which is exactly what its genre classifier consumes in its own evaluation.
MEGADance is deterministic (arg-max codes), so one dance per slice. The regenerated keypoints are compared with the
stored finedance_out/*.npy (max error must be tiny).

    cd /home/sam0505/Documents/Code/dance_baselines/MEGADance
    .venv/bin/python /home/sam0505/Documents/Code/MACE-Dance/Evaluation-Musicality/baselines/mega_rot_export.py

Writes Evaluation-Musicality/data/rot_export/mega_native.npz: trans (84, 1020, 3), p6 (84, 1020, 144).
"""
import os
import sys
import time

import numpy as np
import torch

MEGA = '/home/sam0505/Documents/Code/dance_baselines/MEGADance'
sys.path.insert(0, MEGA)
import gen_mace_eval as g  # noqa: E402  (chdir's to the MEGADance root)

OUT = '/home/sam0505/Documents/Code/MACE-Dance/Evaluation-Musicality/data/rot_export/mega_native.npz'


@torch.no_grad()
def run_batch(model, feats, labels, device):
    import pickle
    init = pickle.load(open('Pretrained/Init1.pkl', 'rb'))
    n = len(feats)
    ds = g.DemoDataset(np.repeat(init['smpl_poses'][None], n, 0), np.repeat(init['smpl_trans'][None], n, 0),
                       np.stack(feats), np.array(labels), [f'{k}.wav' for k in range(n)], g.ROOT)
    data = next(iter(torch.utils.data.DataLoader(ds, batch_size=n, shuffle=False)))
    model.gpt.eval()
    music, _, poses_gt, root_vel_gt, root_init_gt, _, label = model.fsq.preprocess(data, device)
    up, down = model.fsq.pose_encode(poses_gt, root_vel_gt)
    up_p, down_p = model.gpt.inference(music[:, ::8, :], up, down, label)
    root_vel, pose = model.fsq.pose_decode(up_p, down_p)
    trans, _, p6, kp, _, _ = model.fsq.postprocess(pose, root_vel, root_init_gt)
    return (trans.cpu().numpy()[:, :g.N_FRAMES], p6.cpu().numpy()[:, :g.N_FRAMES],
            g.to_mace_axes(kp.cpu().numpy()[:, :g.N_FRAMES]))


def main():
    import json
    s = g.SETS['finedance']
    meta = json.load(open(os.path.join(s['wav'], 'slices.json')))
    n, styles = len(meta['slices']), meta['mace_style']
    g.seed_everything(seed=42)
    device = torch.device('cuda')
    model = g.Trainer(device=device).to(device)
    model = g.load_model(model, 'gpt', 90)
    model.eval()
    T, P, errs = [], [], []
    for b in range(0, n, g.BATCH):
        idx = list(range(b, min(b + g.BATCH, n)))
        t0 = time.time()
        feats = [g.features(os.path.join(s['wav'], f'{i:03d}.wav')) for i in idx]
        tr, p6, kp = run_batch(model, feats, [g.style2s.index(styles[i]) for i in idx], device)
        for k, i in enumerate(idx):
            errs.append(float(np.abs(kp[k] - np.load(os.path.join(MEGA, 'finedance_out', f'{i:03d}.npy'))).max()))
        T.append(tr); P.append(p6)
        print(f'clips {idx[0]}-{idx[-1]} {time.time() - t0:.0f}s  max |kp - stored| so far {max(errs):.5f}', flush=True)
    assert max(errs) < 0.01, ('regenerated MEGA keypoints differ from the stored ones', max(errs))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, trans=np.concatenate(T), p6=np.concatenate(P), max_err_m=max(errs))
    print('wrote', OUT, np.concatenate(T).shape, np.concatenate(P).shape)


if __name__ == '__main__':
    main()
