"""FineDance main comparison table: MACE Table 2 columns + Lock + Loud, in-domain rows only (2026-11-10).

Rows: Real dance (reference), Lodge, MEGA, MACE (released = the paper's "Motion Expert (Full)"), Ours (critic v5 + gain).
Columns: FID_k, FID_g, FSR, DIV_k, DIV_g, BAS (MACE Table 2) + Lock, Loud (ours) + FID_s, DIV_s (MEGADance's style metrics,
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

# MACE paper Table 2 (FSR as printed, a fraction; shown here x100 like Lodge's %).
PAPER = {
    'Real dance': dict(FSR=0.216, DIV_k=9.94, DIV_g=7.54, BAS=0.201),
    'Lodge': dict(FID_k=50.00, FID_g=35.52, FSR=0.028, DIV_k=5.67, DIV_g=4.96, BAS=0.226),
    'MEGA': dict(FID_k=50.00, FID_g=13.02, FSR=0.243, DIV_k=6.23, DIV_g=6.27, BAS=0.226),
    'MACE': dict(FID_k=17.83, FID_g=25.09, FSR=0.210, DIV_k=10.30, DIV_g=8.09, BAS=0.229),
}
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
        lock = load('lock_models_finedance_1seed_rescaled_seed1.json')['results']
        loud = load('loud_models_finedance_seed1_main.json')['results']
        fsr = {**load('fsr_table.json')['finedance'], **load('fsr_table_seed1.json')['finedance']}
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
    return rows


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
        out.append('Bold = best, underline = second best among the four models (Real dance excluded). FSR in %. '
                   '† = our value for seed 1 (first sample per slice; MEGA is deterministic, one dance per slice); unmarked = '
                   'MACE Table 2. Lock, Loud, FID_s and DIV_s are ours for every row (the MACE paper has no value for '
                   'them). FID_s / DIV_s: MEGADance\'s genre-classifier style metrics on our 84 slices (see '
                   'results/genre_metric_seed0.json); Lodge\'s FID_s differs a lot from MEGADance\'s own report (5.22), '
                   'so treat it with caution.\n\n')
    else:
        out.append('Bold = best, underline = second best among the four models (Real dance excluded). FSR in %. '
                   '† = our 10-seed mean; unmarked = MACE Table 2. Lock/Loud are ours for every row.\n\n')
    out.append('FSR note: MACE Table 2 copies the foot-skating numbers for the baselines, but its Real dance (0.216) '
               'and MACE (0.210) cells equal our PFC values, so the column mixes two measures. The Ours cell is our '
               'foot-skating ratio (Lodge formula). Measured the same way for all rows (pooled over the 10 seeds): '
               'Real 6.4, Lodge 3.2, MEGA 10.1, MACE 16.5, Ours 13.4 (%).')
    return '\n'.join(out)


if __name__ == '__main__':
    rows = build()
    tag = '' if SEED1 else '_10seed'
    json.dump(rows, open(os.path.join(R, f'finedance_main_table{tag}.json'), 'w'), indent=1)
    md = markdown(rows)
    open(os.path.join(R, f'finedance_main_table{tag}.md'), 'w').write(md + '\n')
    print(md)
    print('\nFSR measured by us (Lodge formula, %), same rows and seeds as above:',
          {r: round(v['_our_FSR_pct'], 1) for r, v in rows.items()})
