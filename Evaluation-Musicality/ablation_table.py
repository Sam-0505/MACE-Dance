"""Leave-one-out ablation table of the headline model, SEED 1 (the first sample per slice), FineDance test (2026-11-10).

84 slices / 19 songs. Rows: MACE released, + mask only, Full - gain, Full - mask / critic / foot term / frozen root, Full.
All rows use seed 0's noise (gen_mace_test.py: same seed -> same noise), so differences from Full are paired in the noise.
Metrics: FID_k, FID_g, DIV_k, DIV_g, BAS (authors' code), FSR (Lodge formula), Lock and Loud (own-music r, 19 songs). No PFC.

One seed gives no interval for the first six metrics, so a difference from Full is judged against the Full model's own
seed-to-seed SD over its 10 stored seeds (authors_metrics_gain10.json): marked only if |diff| > 2.83 SD (= 2 SD of the
difference of two independent seeds). FSR / Lock / Loud have no stored per-seed SD; Lock and Loud use a bootstrap over the
19 songs (paired by song), FSR has no noise estimate. '*' = beyond noise / CI excludes 0.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/ablation_table.py        # writes results/ablation_table.{json,md}
Inputs: authors_metrics_ablation.json (--seeds 1), fsr_table_abl1.json, lock_models_finedance_1seed_rescaled_abl1.json,
loud_models_finedance_seed1_abl.json (commands in Expert-Motion/run_ablation.sh and the session log).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')

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
COLS = ['FID_k', 'FID_g', 'FSR', 'DIV_k', 'DIV_g', 'BAS', 'Lock', 'Loud']
UP = {'FID_k': False, 'FID_g': False, 'FSR': False, 'DIV_k': True, 'DIV_g': True, 'BAS': True, 'Lock': True, 'Loud': True}


def load(n):
    return json.load(open(os.path.join(R, n)))


def main():
    am = load('authors_metrics_ablation.json')
    fsr = load('fsr_table_abl1.json')['finedance']
    lock = load('lock_models_finedance_1seed_rescaled_abl1.json')['per_unit']
    loud = load('loud_models_finedance_seed1_abl.json')['per_song']
    sd_full = {c: load('authors_metrics_gain10.json')[ROWS[-1][1]][m]['sd'] for c, m, _ in AUTH}
    vals, songs = {}, {}
    for label, d, fk, key in ROWS:
        vals[label] = {c: am[d][m]['runs'][0] for c, m, _ in AUTH} | {'FSR': fsr[fk]['FSR_pct']}
        order = sorted(lock[key]['own'])
        songs[label] = {'Lock': np.array([lock[key]['own'][s] for s in order]),
                        'Loud': np.array([loud[key]['own'][s] for s in order])}
        vals[label] |= {c: float(songs[label][c].mean()) for c in ('Lock', 'Loud')}
    boot = np.random.default_rng(0).integers(0, 19, (5000, 19))
    out = {}
    for label, *_ in ROWS:
        row = {}
        for c in COLS:
            diff = vals[label][c] - vals[FULL][c]
            cell = {'value': vals[label][c], 'diff': diff, 'flag': None, 'ci': None}
            if c in ('Lock', 'Loud'):
                bs = (songs[label][c] - songs[FULL][c])[boot].mean(1)
                cell['ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                cell['flag'] = bool(label != FULL and (cell['ci'][0] > 0 or cell['ci'][1] < 0))
            elif c != 'FSR':
                cell['flag'] = bool(label != FULL and abs(diff) > 2.83 * sd_full[c])
            row[c] = cell
        out[label] = row
    json.dump({'rows': out, 'full_seed_sd_10seeds': sd_full}, open(os.path.join(R, 'ablation_table.json'), 'w'), indent=1)

    def fmt(c, x):
        return f'{x:.2f}' if c in ('FID_k', 'FID_g', 'DIV_k', 'DIV_g') else (f'{x:.1f}' if c == 'FSR' else f'{x:.3f}')

    def dfmt(c, x):
        return f'{x:+.2f}' if c in ('FID_k', 'FID_g', 'DIV_k', 'DIV_g') else (f'{x:+.1f}' if c == 'FSR' else f'{x:+.3f}')

    arrow = {c: ('↑' if UP[c] else '↓') for c in COLS}
    head = '| Row | ' + ' | '.join(f'{c} {arrow[c]}' for c in COLS) + ' |\n|---|' + '---:|' * len(COLS)
    L = ['**Values** (seed 1)', '', head]
    for label, *_ in ROWS:
        L.append(f'| {"**" + label + "**" if label == FULL else label} | ' +
                 ' | '.join(fmt(c, out[label][c]['value']) for c in COLS) + ' |')
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
            cells.append(f'{dfmt(c, r["diff"])}{ci}{mark}')
        L.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    L += ['', 'One seed (the first sample per slice; every row shares its noise). FID/DIV/BAS: `*` if |difference| > 2.83 x the Full '
          'model\'s seed SD over its 10 stored seeds (shown in the italic row). Lock/Loud: 95% bootstrap interval over the 19 songs, '
          'paired by song. FSR has no noise estimate, so it is never starred. FSR in %. "Full - critic" keeps the foot term with a '
          'vanishing guidance scale (gain and frozen root act only on the guidance push, so they have no effect there).']
    md = '\n'.join(L)
    open(os.path.join(R, 'ablation_table.md'), 'w').write(md + '\n')
    print(md)


if __name__ == '__main__':
    main()
