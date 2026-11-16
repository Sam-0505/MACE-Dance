"""FID_s / DIV_s (+ genre accuracy) for the ablation rows, seed 1 (2026-11-15). Same protocol as genre_metric.py.

Rows: real dance (reference), MACE released, + mask only, Full - gain, Full - mask / critic / foot term / frozen root, Full.
Released and Full come from genre_inputs_seed0.npz (export_rot_all.py + prep_genre_inputs.py); the other six from
ablation_seed0_yup.npz (export_rot_ablation.py). MEGADance's classifier, its Euler-style pose transform (the convention on which
real dance is classified best, see genre_metric.py), 1024-frame clips (1020 padded by repeating the last frame), features
normalised by the real dance's, FID_s = Frechet distance to real, DIV_s = mean pairwise distance. The bootstrap (300 resamples
of the 84 slices, paired across rows) gives a 95% interval of each row minus Full.

    cd /home/sam0505/Documents/Code/dance_baselines/MEGADance
    .venv/bin/python /home/sam0505/Documents/Code/MACE-Dance/Evaluation-Musicality/genre_metric_ablation.py
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import genre_metric as gm  # noqa: E402  (chdir's to the MEGADance root, loads its classifier code)

ROWS = {  # key: (label, file, array prefix)
    'real': ('Real dance', 'genre_inputs_seed0.npz', 'real'),
    'released': ('MACE released', 'genre_inputs_seed0.npz', 'oneshot'),
    'mask': ('+ mask only', 'ablation_seed0_yup.npz', 'mask'),
    'nogain': ('Full - gain', 'ablation_seed0_yup.npz', 'nogain'),
    'nomask': ('Full - mask', 'ablation_seed0_yup.npz', 'nomask'),
    'nocritic': ('Full - critic', 'ablation_seed0_yup.npz', 'nocritic'),
    'nofoot': ('Full - foot term', 'ablation_seed0_yup.npz', 'nofoot'),
    'nofrz': ('Full - frozen root', 'ablation_seed0_yup.npz', 'nofrz'),
    'full': ('Full (headline)', 'genre_inputs_seed0.npz', 'gain'),
}


def main():
    files = {f: np.load(os.path.join(HERE, 'data', 'rot_export', f)) for f in {v[1] for v in ROWS.values()}}
    base = files['genre_inputs_seed0.npz']
    for f in files.values():
        assert list(f['names']) == list(base['names'])
    labels = torch.tensor([gm.STYLE2S.index(g) for g in base['genres']])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cls = gm.CLS(device).to(device)
    cls = gm.load_model(cls, 'cls', 30)
    cls.eval()
    feats = {k: gm.logits(cls, gm.pad(files[f][f'{p}_q']), gm.pad(files[f][f'{p}_pos']), 'euler', device)
             for k, (_, f, p) in ROWS.items()}
    res = {}
    for k, (label, *_ ) in ROWS.items():
        row = {'label': label, 'acc': gm.compute_acc(torch.as_tensor(feats[k]), labels)}
        g, p = gm.norm(feats['real'], feats[k])
        row['DIV_s'] = float(gm.calculate_avg_distance(p))
        if k != 'real':
            row['FID_s'] = float(gm.calc_fid(p, g))
        res[k] = row
    rng = np.random.default_rng(0)
    boot = {k: {'FID_s': [], 'DIV_s': []} for k in ROWS if k != 'real'}
    for _ in range(300):
        idx = rng.integers(0, len(labels), len(labels))
        for k in boot:
            g, p = gm.norm(feats['real'][idx], feats[k][idx])
            boot[k]['FID_s'].append(float(gm.calc_fid(p, g)))
            boot[k]['DIV_s'].append(float(gm.calculate_avg_distance(p)))
    for k in boot:
        if k == 'full':
            continue
        for m in ('FID_s', 'DIV_s'):
            d = np.array(boot[k][m]) - np.array(boot['full'][m])
            res[k][f'{m}_minus_full_ci95'] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    json.dump(res, open(os.path.join(HERE, 'results', 'genre_metric_ablation_seed0.json'), 'w'), indent=1)
    for k, r in res.items():
        print(f"{r['label']:22s} acc {r['acc']:.3f}  FID_s {r.get('FID_s', float('nan')):7.3f}  DIV_s {r['DIV_s']:.3f}  "
              f"FID_s-full CI {r.get('FID_s_minus_full_ci95')}  DIV_s-full CI {r.get('DIV_s_minus_full_ci95')}")


if __name__ == '__main__':
    main()
