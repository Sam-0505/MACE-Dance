"""Is the model's shortfall a long-window effect? Score the existing 34 s test samples (10 seeds) with diag.py's
zero-lag Delta, on the whole 1020-frame slice and on its four 240-frame pieces, next to real dance.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/sweep_diagnostics_scale.py
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, '../Evaluation-Musicality')
from EDGE import EDGE  # noqa: E402
from model import musicality as MU  # noqa: E402
from accent_overlap import full_music  # noqa: E402
from sweep_diagnostics import score  # noqa: E402

DEV = 'cuda'
RUNS = {'released': 'mace_test', 'lambda_0': 'mace_test_sweep_l0', 'lambda_3e-4': 'mace_test_sweep_l3e-4',
        'lambda_1e-3': 'mace_test_sweep_l1e-3'}
D = '../Evaluation-Musicality/data'


def main():
    model = EDGE('baseline', checkpoint_path='runs/train/exp/weights/train-3750.pt')
    d = model.diffusion
    dt = pickle.load(open('data/dataset_backups/test_tensor_dataset.pkl', 'rb')).data
    names = [os.path.splitext(os.path.basename(f))[0] for f in dt['filenames']]
    assert names == json.load(open(f'{D}/mace_test/names.json'))['names']
    music = torch.stack([torch.from_numpy(np.asarray(full_music(f'data/test/wavs_sliced/{n}.wav')['treble'][:1020],
                                                     dtype=np.float32)) for n in names]).to(DEV)
    song = np.array([n.split('_slice')[0] for n in names])
    with torch.no_grad():
        real = MU.pose_to_joints(dt['pose'].to(DEV), model.normalizer, d.smpl)

    def both(j):
        """j: (84, 1020, 24, 3) -> scores on the whole slice and on its four 8 s pieces."""
        e = MU.motion_energy(j)
        out = {'34s': score(MU.gaussian_smooth(e, 15), MU.gaussian_smooth(music, 15), song)}
        pe = torch.cat([e[:, o:o + 240] for o in range(0, 960, 240)])
        pm = torch.cat([music[:, o:o + 240] for o in range(0, 960, 240)])
        out['8s'] = score(MU.gaussian_smooth(pe, 15), MU.gaussian_smooth(pm, 15), np.tile(song, 4))
        # the same two pieces per slice that diag.py generates directly (frames 0 and 240)
        out['8s_first2'] = score(MU.gaussian_smooth(pe[:168], 15), MU.gaussian_smooth(pm[:168], 15), np.tile(song, 2))
        return out

    res = {'real': both(real)}
    print('real', {k: round(v['delta'], 3) for k, v in res['real'].items()}, flush=True)
    for label, sub in RUNS.items():
        g = np.load(f'{D}/{sub}/joints.npy')
        per_seed = [both(torch.from_numpy(g[:, s].astype(np.float32)).to(DEV)) for s in range(g.shape[1])]
        res[label] = {k: {'delta_mean': float(np.mean([p[k]['delta'] for p in per_seed])),
                          'delta_seed_sd': float(np.std([p[k]['delta'] for p in per_seed])),
                          'r_own_mean': float(np.mean([p[k]['r_own'] for p in per_seed]))} for k in ('34s', '8s', '8s_first2')}
        print(label, {k: round(v['delta_mean'], 3) for k, v in res[label].items()}, flush=True)
    json.dump(res, open('../Evaluation-Musicality/results/sweep_diagnostics_scale.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
