"""Sanity checks for the Lodge Motorica joints (run after gen_motorica.py --stage joints, both variants)."""
import numpy as np

M = '/home/sam0505/Documents/Code/MACE-Dance/Evaluation-Musicality/data/motorica'
RAW = '/home/sam0505/Documents/Code/dance_baselines/LODGE/motorica_work/raw'


def stats(name, path):
    j = np.load(path, mmap_mode='r')
    print(f'== {name}: shape {j.shape} dtype {j.dtype}')
    sp, mx, nan = [], 0.0, 0
    for i in range(j.shape[0]):
        x = np.asarray(j[i], dtype=np.float32)
        nan += int(np.isnan(x).sum())
        d = np.linalg.norm(np.diff(x, axis=-3), axis=-1)       # (S, T-1, 24)
        sp.append(d.mean(axis=-1) * 30)                         # (S, T-1) mean over joints, m/s
        mx = max(mx, float(d.max()))
    sp = np.stack(sp)
    print(f'   NaNs {nan}; mean joint speed {sp.mean():.3f} m/s (per-clip median {np.median(sp.mean(axis=(1, 2))):.3f}); '
          f'largest frame-to-frame joint jump {mx:.3f} m')
    return sp


def root_jumps():
    jumps, steps = [], []
    for i in range(289):
        for s in range(3):
            r = np.load(f'{RAW}/{i:03d}_s{s}.npy')[:, [4, 6]]
            st = np.linalg.norm(np.diff(r, axis=0), axis=1)
            jumps += [st[k - 1] for k in (256, 512, 768)]
            steps.append(np.delete(st, [255, 511, 767]))
    jumps, steps = np.array(jumps), np.concatenate(steps)
    print(f'== root x/z jump at frames 256/512/768 (official concat): median {np.median(jumps):.4f} m, '
          f'p95 {np.percentile(jumps, 95):.4f} m, max {jumps.max():.4f} m')
    print(f'   typical frame-to-frame root x/z displacement elsewhere: median {np.median(steps):.4f} m, '
          f'p95 {np.percentile(steps, 95):.4f} m')


if __name__ == '__main__':
    stats('lodge (root-continuity fix)', f'{M}/lodge/joints.npy')
    stats('lodge_official (plain concat)', f'{M}/lodge_official/joints.npy')
    stats('mace_guide_s0.003', f'{M}/mace_guide_s0.003/joints.npy')
    root_jumps()
