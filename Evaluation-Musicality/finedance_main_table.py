"""FineDance main comparison table: MACE Table 2 columns + Lock + Loud, in-domain rows only (2026-11-10).

Rows: Real dance (reference), Lodge, MEGA, MACE (released = the paper's "Motion Expert (Full)"), Ours (critic v5 + gain).
Columns: FID_k, FID_g, DIV_k, DIV_g, BAS (MACE Table 2) + FSR, Lock, Loud (ours) + FID_s, DIV_s (MEGADance's style metrics,
ours; seed 1 only).

Rule: 1) every cell the MACE paper reports for a row is copied from Table 2 (PAPER below);
      2) every other cell is OUR value for seed 1 (the first sample per slice; default) or the 10-seed mean (--tenseed),
         FineDance test, 84 slices, 19 songs. Lock / Loud / FID_s / DIV_s exist only in our data, so they are ours for every
         row. Real dance has no seed. MEGADance is deterministic (one dance per slice); Lodge has 10 seeds, seed 1 = the first.
Marks: bold = best, underline = second best, over the four model rows.

    python3 finedance_main_table.py            # seed 1 -> results/finedance_main_table.{json,md}
    python3 finedance_main_table.py --tenseed  # 10-seed means (no FID_s/DIV_s) -> results/finedance_main_table_10seed.{json,md}
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')

SEED1 = '--tenseed' not in sys.argv
BASE_COLS = ['FID_k', 'FID_g', 'FSR', 'DIV_k', 'DIV_g', 'BAS', 'Lock', 'Loud']        # the 8 columns of the first version
COLS = ['FID_k', 'FID_g', 'FID_s', 'FSR', 'DIV_k', 'DIV_g', 'DIV_s', 'BAS', 'Lock', 'Loud']
UP = {'FID_s': False, 'DIV_s': True, 'FID_k': False, 'FID_g': False, 'FSR': False, 'DIV_k': True, 'DIV_g': True, 'BAS': True, 'Lock': True, 'Loud': True}

# MACE paper Table 2 / FlowerDance Table 1. FSR is NOT taken from the papers: their FSR columns mix two measures (see the
# footnote); FSR is our own foot-skating ratio for every row.
PAPER = {
    'Real dance': dict(DIV_k=9.94, DIV_g=7.54, BAS=0.201),
    'Lodge': dict(FID_k=50.00, FID_g=35.52, DIV_k=5.67, DIV_g=4.96, BAS=0.226),
    'MEGA': dict(FID_k=50.00, FID_g=13.02, DIV_k=6.23, DIV_g=6.27, BAS=0.226),
    'MACE': dict(FID_k=17.83, FID_g=25.09, DIV_k=10.30, DIV_g=8.09, BAS=0.229),
    # FlowerDance (ECCV'26) Table 1, FineDance; same baseline rows as MACE's table. (DiscoForcing was dropped: no checkpoint.)
    'FlowerDance': dict(FID_k=29.73, FID_g=19.59, DIV_k=8.42, DIV_g=7.18, BAS=0.232),
}
ORDER = ['Real dance', 'Lodge', 'MEGA', 'MACE', 'FlowerDance', 'Ours (critic v5 + gain)']
# result keys per row in each of our 10-seed files
KEYS = {
    'Real dance': dict(am='real', lock='Real dance', loud='Real dance', fsr='real'),
    'Lodge': dict(am='mace_test_lodge', lock='Lodge', loud='Lodge', fsr='lodge'),
    'MEGA': dict(am='mace_test_mega', lock='MEGADance', loud='MEGADance', fsr='mega'),
    'MACE': dict(am='mace_test', lock='MACE released', loud='MACE released', fsr='released'),
    'Ours (critic v5 + gain)': dict(am='mace_test_mask2s_criticv5_s0.003_foot10_frz_gain10',
                                    lock='Ours critic v5 + gain', loud='Ours critic v5 + gain', fsr='gain'),
}


def load(name):
    return json.load(open(os.path.join(R, name)))


GENRE_KEY = {'Real dance': 'real', 'Lodge': 'lodge', 'MEGA': 'mega', 'MACE': 'oneshot', 'Ours (critic v5 + gain)': 'gain'}


def build():
    am = {**load('authors_metrics_baselines.json'), **load('authors_metrics_gain10.json')}
    if SEED1:
        lock = {**load('lock_models_finedance_1seed_rescaled_seed1.json')['results'],
                **load('lock_models_finedance_1seed_rescaled_seed1_flower.json')['results']}
        loud = load('loud_models_finedance_seed1_main.json')['results']
        fsr = {**load('fsr_table.json')['finedance'], **load('fsr_table_seed1.json')['finedance'],
               **load('fsr_table_seed1_flower.json')['finedance']}
        genre = load('genre_metric_seed0.json')['euler']
        src = 'seed 1'
    else:
        lock = load('lock_models_finedance_10seed_rescaled_main.json')['results']
        loud = load('loud_models_10seed.json')['results']
        fsr = {**load('fsr_table.json')['finedance'], **load('fsr_table_gain10.json')['finedance']}
        genre, src = None, '10-seed'

    def amv(key, m):   # authors' metric: seed 1 = first of the stored runs (the real row has a single value)
        return am[key][m]['runs'][0] if SEED1 else am[key][m]['mean']
    rows = {}
    for row, k in KEYS.items():
        ours = {
            'FID_k': amv(k['am'], 'FID_k'), 'FID_g': amv(k['am'], 'FID_g'),
            'FSR': fsr[k['fsr']]['FSR_pct'], 'DIV_k': amv(k['am'], 'Div_k'),
            'DIV_g': amv(k['am'], 'Div_g'), 'BAS': amv(k['am'], 'BAS'),
            'Lock': lock[k['lock']]['r_own'], 'Loud': loud[k['loud']]['r_own'],
            'FID_s': genre[GENRE_KEY[row]].get('FID_s') if genre else None,
            'DIV_s': genre[GENRE_KEY[row]].get('DIV_s') if genre else None,
        }
        paper = {c: (v * 100 if c == 'FSR' else v) for c, v in PAPER.get(row, {}).items()}
        rows[row] = {c: ({'v': paper[c], 'src': 'paper'} if c in paper else {'v': ours[c], 'src': src}) for c in COLS}
        rows[row]['_our_FSR_pct'] = ours['FSR']
    # FlowerDance: paper values for the six Table-1 columns; Lock, Loud, FID_s, DIV_s are ours (seed 1) from its released
    # FineDance checkpoint (dance_baselines/FlowerDance/ckpt/train-3700.pt).
    extra = {'FlowerDance': {'FID_s': genre['flower']['FID_s'], 'DIV_s': genre['flower']['DIV_s'],
                             'Lock': lock['FlowerDance']['r_own'], 'Loud': loud['FlowerDance']['r_own'],
                             'FSR': fsr['flower']['FSR_pct']}} if SEED1 else {}
    for row in PAPER:                                   # paper-only rows: dashes where missing
        if row not in rows:
            rows[row] = {}
            for c in COLS:
                if c in PAPER[row]:
                    rows[row][c] = {'v': PAPER[row][c] * (100 if c == 'FSR' else 1), 'src': 'paper'}
                elif c in extra.get(row, {}):
                    rows[row][c] = {'v': extra[row][c], 'src': src}
                else:
                    rows[row][c] = {'v': None, 'src': 'none'}
            rows[row]['_our_FSR_pct'] = fsr['flower']['FSR_pct'] if row == 'FlowerDance' and SEED1 else None
    return {r: rows[r] for r in ORDER}


def fmt(c, v):
    if c in ('FID_k', 'FID_g', 'FID_s', 'DIV_k', 'DIV_g', 'DIV_s'):
        return f'{v:.2f}'
    if c == 'FSR':
        return f'{v:.1f}'
    return f'{v:.3f}'


def table_lines(rows, dagger=True, cols=None):
    """Markdown table lines; cells with v=None print as a dash, bold/underline over the non-real rows."""
    cols = cols or COLS
    models = [r for r in rows if r != 'Real dance']
    best, second = {}, {}
    for c in cols:
        vals = sorted({round(rows[r][c]['v'], 6) for r in models if rows[r][c]['v'] is not None}, reverse=UP[c])
        best[c] = {r for r in models if rows[r][c]['v'] is not None and round(rows[r][c]['v'], 6) == vals[0]}
        second[c] = {r for r in models if rows[r][c]['v'] is not None and len(vals) > 1
                     and round(rows[r][c]['v'], 6) == vals[1]}
    arrows = {c: ('↑' if UP[c] else '↓') for c in cols}
    out = ['| Model | ' + ' | '.join(f'{c} {arrows[c]}' for c in cols) + ' |', '|---|' + '---:|' * len(cols)]
    for r, vals in rows.items():
        cells = []
        for c in cols:
            if (r == 'Real dance' and c in ('FID_k', 'FID_g', 'FID_s')) or vals[c]['v'] is None:
                cells.append('–')
                continue
            s = fmt(c, vals[c]['v']) + ('' if (vals[c]['src'] == 'paper' or not dagger) else '†')
            if r in best[c]:
                s = f'**{s}**'
            elif r in second[c]:
                s = f'<u>{s}</u>'
            cells.append(s)
        out.append(f'| {r} | ' + ' | '.join(cells) + ' |')
    return out


def markdown(rows):
    cols = COLS if SEED1 else BASE_COLS
    out = table_lines(rows, cols=cols)
    out.append('')
    if SEED1:
        out.append('Bold = best, underline = second best among the model rows (Real dance excluded). † = our value for seed 1 '
                   '(first sample per slice; MEGA is deterministic, one dance per slice); unmarked = copied from the MACE paper '
                   '(Lodge, MEGA, MACE, Real dance) or the FlowerDance paper (FlowerDance). FSR, Lock, Loud, FID_s and DIV_s are '
                   'ours for every row (see below). FID_s / DIV_s: MEGADance\'s genre-classifier style metrics on our 84 slices '
                   '(results/genre_metric_seed0.json); Lodge\'s FID_s differs a lot from MEGADance\'s own report (5.22), so treat '
                   'it with caution.\n\n')
    else:
        out.append('Bold = best, underline = second best among the model rows (Real dance excluded). † = our 10-seed mean '
                   '(FSR: pooled over the 10 seeds); unmarked = MACE Table 2. FSR, Lock, Loud are ours for every row.\n\n')
    if SEED1:
        out.append('FlowerDance: FID_k, FID_g, DIV_k, DIV_g, BAS copied from its paper; FSR, FID_s, DIV_s, Lock, Loud are ours from '
                   'its released FineDance checkpoint (train-3700.pt, 21 Euler steps, shipped test features, seed 1). Re-scoring that '
                   'checkpoint on our 84 slices gives FID_k 40.3, FID_g 17.9, DIV_k 7.87, DIV_g 7.29, BAS 0.230 (paper: 29.73 / '
                   '19.59 / 8.42 / 7.18 / 0.232).\n\n')
    out.append('FSR (%) is our foot-skating ratio: a port of Lodge\'s official foot-skating code (fsr_table.py), computed on the '
               'generated joints of every row (84 slices; real test dance for Real dance), so all rows share one definition. The '
               'papers\' FSR columns were not used because they mix two measures: the MACE paper\'s Real dance (0.216) and MACE '
               '(0.210) cells equal our PFC values, while its Lodge, EDGE and Bailando cells are Lodge\'s own foot-skating numbers (MEGA\'s cell is from the '
               'MEGADance paper; FlowerDance\'s Table 1 repeats the same column). As a cross-check, our real dance gives 6.4 % and Lodge 3.2 % '
               '(10 seeds), next to the 6.22 % and 2.76 % that the Lodge paper reports.')
    return '\n'.join(out)


if __name__ == '__main__':
    rows = build()
    tag = '' if SEED1 else '_10seed'
    json.dump(rows, open(os.path.join(R, f'finedance_main_table{tag}.json'), 'w'), indent=1)
    md = markdown(rows)
    open(os.path.join(R, f'finedance_main_table{tag}.md'), 'w').write(md + '\n')
    print(md)
    print('\nFSR measured by us (Lodge formula, %), same rows and seeds as above:',
          {r: round(v['_our_FSR_pct'], 1) for r, v in rows.items() if v['_our_FSR_pct'] is not None})
