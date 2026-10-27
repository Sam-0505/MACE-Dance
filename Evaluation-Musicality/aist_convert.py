"""Convert the downloaded AIST++ motion pickles to .npz once, so nothing else unpickles downloaded data.

The unpickler only accepts the numpy classes needed to rebuild arrays; anything else raises.
Output: data/aist/motions_npz/<seq>.npz with smpl_poses (N, 72), smpl_trans (N, 3), smpl_scaling (1,).

    Expert-Motion/.venv/bin/python -I Evaluation-Musicality/aist_convert.py
"""
import glob
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'data', 'aist', 'motions')
DST = os.path.join(HERE, 'data', 'aist', 'motions_npz')
ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
           ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'),
           ('numpy._core.multiarray', 'scalar')}
KEYS = ('smpl_poses', 'smpl_trans', 'smpl_scaling')


class NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f'blocked {module}.{name}')
        return super().find_class(module, name)


def main():
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, '*.pkl')))
    for f in files:
        with open(f, 'rb') as fh:
            d = NumpyOnly(fh).load()
        arrs = {k: np.asarray(d[k], dtype=np.float32) for k in KEYS}
        assert arrs['smpl_poses'].shape[1] == 72 and len(arrs['smpl_poses']) == len(arrs['smpl_trans'])
        np.savez(os.path.join(DST, os.path.basename(f)[:-4] + '.npz'), **arrs)
    print(f'converted {len(files)} files to {DST}')


if __name__ == '__main__':
    main()
