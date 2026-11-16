"""Leave-one-out ablation table of the headline model, SEED 1 (the first sample per slice), FineDance test (2026-11-10).

84 slices / 19 songs. Rows: MACE released, + mask only, Full - gain, Full - mask / critic / foot term / frozen root, Full.
All rows use seed 0's noise (gen_mace_test.py: same seed -> same noise), so differences from Full are paired in the noise.
Metrics: FID_k, FID_g, DIV_k, DIV_g, BAS (authors' code), FSR (Lodge formula), Lock and Loud (own-music r, 19 songs). No PFC.

One seed gives no interval for the first six metrics, so a difference from Full is judged against the Full model's own
seed-to-seed SD over its 10 stored seeds (authors_metrics_gain10.json): marked only if |diff| > 2.83 SD (= 2 SD of the
difference of two independent seeds). FSR / Lock / Loud have no stored per-seed SD; Lock and Loud use a bootstrap over the
19 songs (paired by song), FSR has no noise estimate. '*' = beyond noise / CI excludes 0.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/ablation_table.py            # seed 1 -> results/ablation_table.{json,md}
    .venv/bin/python ../Evaluation-Musicality/ablation_table.py --tenseed  # 10-seed means -> results/ablation_table_10seed.{json,md}

--tenseed: every row has 10 seeds and seed k uses the same noise in every row, so each metric gives 10 paired differences
(row - Full, one per seed). FID_k/FID_g/DIV_k/DIV_g/BAS/FID_s/DIV_s: '*' if the 95% t-interval of their mean (9 df) excludes 0
(FID_s/DIV_s also need |difference| >= 0.05, as for seed 1). Lock/Loud: 95% bootstrap over the 19 songs of the 10-seed
per-song means, paired by song. FSR: pooled over the 10 seeds, no noise estimate, never starred.
Inputs: authors_metrics_ablation10.json, fsr_table_abl10.json, lock_models_finedance_10seed_rescaled_abl10.json,
loud_models_10seed_abl10.json, genre_metric_ablation_seed0..9.json.
Inputs: authors_metrics_ablation.json (--seeds 1), fsr_table_abl1.json, lock_models_finedance_1seed_rescaled_abl1.json,
loud_models_finedance_seed1_abl.json (commands in Expert-Motion/run_ablation.sh and the session log).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
TEN = '--tenseed' in sys.argv
T975_9 = 2.2621571627982                     # t quantile, 9 df
# MACE released: the cells MACE's paper reports (Table 2, = finedance_main_table.PAPER['MACE']) are copied from it, as in the
# main table. They have no per-seed runs, so their differences from Full get no interval and no star (marked ‡).
MACE_PAPER = dict(FID_k=17.83, FID_g=25.09, DIV_k=10.30, DIV_g=8.09, BAS=0.229)

FULL = 'Full (headline)'
ROWS = [  # label, data dir, fsr key, Lock/Loud model key
    ('MACE released', 'mace_test', 'released', 'MACE released'),
    ('+ mask only', 'mace_test_local2s', 'mask', 'MACE mask2s'),
    ('Full - gain', 'mace_test_mask2s_criticv5_s0.003_foot10_frz', 'nogain', 'Ours critic v5'),
    ('Full - mask', 'mace_test_abl_nomask', 'nomask', 'Abl - mask'),
    ('Full - critic', 'mace_test_abl_nocritic', 'nocritic', 'Abl - critic'),
    ('Full - foot term', 'mace_test_abl_nofoot', 'nofoot', 'Abl - foot'),
    ('Full - frozen root', 'mace_test_abl_nofrz', 'nofrz', 'Abl - frozen root'),
    (FULL, 'mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10', 'full', 'Ours critic v5 + gain'),
]
AUTH = [('FID_k', 'FID_k', False), ('FID_g', 'FID_g', False), ('DIV_k', 'Div_k', True), ('DIV_g', 'Div_g', True),
        ('BAS', 'BAS', True)]
COLS = ['FID_k', 'FID_g', 'FID_s', 'FSR', 'DIV_k', 'DIV_g', 'DIV_s', 'BAS', 'Lock', 'Loud']
UP = {'FID_s': False, 'DIV_s': True, 'FID_k': False, 'FID_g': False, 'FSR': False, 'DIV_k': True, 'DIV_g': True, 'BAS': True, 'Lock': True, 'Loud': True}


def load(n):
    return json.load(open(os.path.join(R, n)))


def main():
    if TEN:
        am = load('authors_metrics_ablation10.json')
        fsr = load('fsr_table_abl10.json')['finedance']
        lock = load('lock_models_finedance_10seed_rescaled_abl10.json')['per_unit']
        loud = load('loud_models_10seed_abl10.json')['per_song']
        gseeds = [load(f'genre_metric_ablation_seed{k}.json') for k in range(10)]
    else:
        am = load('authors_metrics_ablation.json')
        fsr = load('fsr_table_abl1.json')['finedance']
        lock = load('lock_models_finedance_1seed_rescaled_abl1.json')['per_unit']
        loud = load('loud_models_finedance_seed1_abl.json')['per_song']
        gseeds = [load('genre_metric_ablation_seed0.json')]
    genre = gseeds[0]
    sd_full = {c: load('authors_metrics_gain10.json')[ROWS[-1][1]][m]['sd'] for c, m, _ in AUTH}
    vals, songs, runs = {}, {}, {}
    for label, d, fk, key in ROWS:
        runs[label] = {c: np.array(am[d][m]['runs'][:len(gseeds)]) for c, m, _ in AUTH}
        runs[label] |= {c: np.array([g[fk][c] for g in gseeds]) for c in ('FID_s', 'DIV_s')}
        assert all(len(v) == len(gseeds) for v in runs[label].values()), (label, {c: len(v) for c, v in runs[label].items()})
        vals[label] = {c: float(runs[label][c].mean()) for c in runs[label]} | {'FSR': fsr[fk]['FSR_pct']}
        order = sorted(lock[key]['own'])
        songs[label] = {'Lock': np.array([lock[key]['own'][s] for s in order]),
                        'Loud': np.array([loud[key]['own'][s] for s in order])}
        vals[label] |= {c: float(songs[label][c].mean()) for c in ('Lock', 'Loud')}
    vals['MACE released'] |= MACE_PAPER
    boot = np.random.default_rng(0).integers(0, 19, (5000, 19))
    out = {}
    for label, *_ in ROWS:
        row = {}
        for c in COLS:
            diff = vals[label][c] - vals[FULL][c]
            cell = {'value': vals[label][c], 'diff': diff, 'flag': None, 'ci': None}
            if label == 'MACE released' and c in MACE_PAPER:
                cell['src'] = 'MACE paper'
            elif c in ('Lock', 'Loud'):
                bs = (songs[label][c] - songs[FULL][c])[boot].mean(1)
                cell['ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                cell['flag'] = bool(label != FULL and (cell['ci'][0] > 0 or cell['ci'][1] < 0))
            elif TEN and c != 'FSR':        # paired over the 10 seeds (same noise per seed in every row)
                dd = runs[label][c] - runs[FULL][c]
                h = T975_9 * dd.std(ddof=1) / np.sqrt(len(dd)) if label != FULL else 0.0
                cell['ci'] = [float(dd.mean() - h), float(dd.mean() + h)] if label != FULL else None
                cell['flag'] = bool(label != FULL and (cell['ci'][0] > 0 or cell['ci'][1] < 0)
                                    and (c not in ('FID_s', 'DIV_s') or abs(diff) >= 0.05))
            elif c in ('FID_s', 'DIV_s'):   # bootstrap over slices (genre_metric_ablation.py); also needs a practical size
                g = genre[[k for l, d_, k, _ in ROWS if l == label][0]].get(f'{c}_minus_full_ci95')
                cell['ci'] = g
                cell['flag'] = bool(g and (g[0] > 0 or g[1] < 0) and abs(diff) >= 0.05)
            elif c != 'FSR':
                cell['flag'] = bool(label != FULL and abs(diff) > 2.83 * sd_full[c])
            row[c] = cell
        out[label] = row
    tag = '_10seed' if TEN else ''
    json.dump({'rows': out, 'full_seed_sd_10seeds': sd_full, 'n_seeds': len(gseeds)},
              open(os.path.join(R, f'ablation_table{tag}.json'), 'w'), indent=1)

    def fmt(c, x):
        return f'{x:.2f}' if c in ('FID_k', 'FID_g', 'DIV_k', 'DIV_g') else (f'{x:.1f}' if c == 'FSR' else f'{x:.3f}')

    def dfmt(c, x):
        return f'{x:+.2f}' if c in ('FID_k', 'FID_g', 'DIV_k', 'DIV_g') else (f'{x:+.1f}' if c == 'FSR' else f'{x:+.3f}')

    arrow = {c: ('↑' if UP[c] else '↓') for c in COLS}
    head = '| Row | ' + ' | '.join(f'{c} {arrow[c]}' for c in COLS) + ' |\n|---|' + '---:|' * len(COLS)
    L = ['**Values** (' + ('10-seed mean' if TEN else 'seed 1') + ')', '', head]
    for label, *_ in ROWS:
        L.append(f'| {"**" + label + "**" if label == FULL else label} | ' +
                 ' | '.join(fmt(c, out[label][c]['value']) + ('‡' if out[label][c].get('src') else '') for c in COLS) + ' |')
    if not TEN:
        L.append('| *Full seed SD (10 seeds)* | ' + ' | '.join(
            f'{sd_full[c]:.3f}'.rstrip('0') if c in sd_full else '–' for c in COLS) + ' |')
    L += ['', '**Difference from Full** (row − Full; `*` = beyond noise; ▲ better than Full, ▼ worse)', '', head]
    for label, *_ in ROWS:
        if label == FULL:
            continue
        cells = []
        for c in COLS:
            r = out[label][c]
            mark = ''
            if r['flag']:
                mark = '*' + (' ▲' if (r['diff'] > 0) == UP[c] else ' ▼')
            ci = f' [{dfmt(c, r["ci"][0])}, {dfmt(c, r["ci"][1])}]' if r['ci'] else ''
            cells.append(f'{dfmt(c, r["diff"])}{"‡" if r.get("src") else ""}{ci}{mark}')
        L.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    if TEN:
        L += ['', '‡ = MACE released value copied from the MACE paper (Table 2), as in the main table; it has no per-seed runs, so its difference from Full has no interval and is never starred. The MACE row\'s FID_s, FSR, DIV_s, Lock and Loud are ours.', '', '10 seeds per row; seed k uses the same noise in every row, so differences from Full are paired by seed. '
              'FID_k/FID_g/FID_s/DIV_k/DIV_g/DIV_s/BAS: 95% t-interval (9 df) of the mean of the 10 per-seed differences; `*` if it '
              'excludes 0 (FID_s/DIV_s also need |difference| >= 0.05). FID_s / DIV_s: MEGADance genre-classifier style metrics '
              '(genre_metric_ablation.py, one value per seed, averaged). Lock/Loud: 95% bootstrap interval over the 19 songs of '
              'the 10-seed per-song means, paired by song. FSR pooled over the 10 seeds; it has no noise estimate, so it is never '
              'starred. FSR in %. "Full - critic" keeps the foot term with a vanishing guidance scale (gain and frozen root act '
              'only on the guidance push, so they have no effect there).']
        md = '\n'.join(L)
        open(os.path.join(R, 'ablation_table_10seed.md'), 'w').write(md + '\n')
        print(md)
        return
    L += ['', 'FID_s / DIV_s: MEGADance genre-classifier style metrics (genre_metric_ablation.py); `*` needs the 95% slice-bootstrap '
          'interval to exclude 0 AND |difference| >= 0.05. They respond only to the mask: every variant that keeps the mask matches Full '
          'to within 0.002 (classifier outputs differ by ~0.002 against a spread of 1.4 across clips), so the guidance components '
          'neither help nor hurt genre style by this measure.', '',
          '‡ = MACE released value copied from the MACE paper (Table 2), as in the main table; it has no per-seed runs, so its difference from Full has no interval and is never starred. The MACE row\'s FID_s, FSR, DIV_s, Lock and Loud are ours.', '',
          'One seed (the first sample per slice; every row shares its noise). FID_k/FID_g/DIV_k/DIV_g/BAS: `*` if |difference| > 2.83 x the Full '
          'model\'s seed SD over its 10 stored seeds (shown in the italic row). Lock/Loud: 95% bootstrap interval over the 19 songs, '
          'paired by song. FSR has no noise estimate, so it is never starred. FSR in %. "Full - critic" keeps the foot term with a '
          'vanishing guidance scale (gain and frozen root act only on the guidance push, so they have no effect there).']
    md = '\n'.join(L)
    open(os.path.join(R, 'ablation_table.md'), 'w').write(md + '\n')
    print(md)


if __name__ == '__main__':
    main()
