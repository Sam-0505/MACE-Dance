"""Motorica main comparison table, same layout as finedance_main_table.py (2026-11-10).

Rows: Real dance, Lodge, MEGA, MACE (released), Ours (critic v5 + gain). Every cell is ours (3 samples per slice):
the MACE paper has no Motorica numbers. Motorica is out of domain for all four models (trained on FineDance).

  FID_k, FID_g, DIV_k, DIV_g : authors' protocol (authors_metrics code) against their FineDance test reference, from
                               motorica_compare_*.json 'fid' (FID_k_fd, FID_g_fd, Div_k, Div_g). Not defined for the
                               real Motorica row (not computed), so those cells are dashes.
  FSR  : Lodge's foot-skating ratio (fsr_table.py), % (BVH joint placement differs from SMPL, thresholds not exact).
  BAS  : beat_suite_motorica_baselines2 / _tempo_gain.
  Lock, Loud : lock_models_motorica_3seed_rescaled_slides, loud_models_motorica_3seed_gain.

    python3 motorica_main_table.py     # Motorica reference, kinetic only -> results/motorica_main_table.{json,md}\n    python3 motorica_main_table.py --fd # FineDance-reference version -> results/motorica_main_table_fdref.{json,md}
"""
import json
import os
import sys

from finedance_main_table import BASE_COLS as COLS, table_lines

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')


def load(n):
    return json.load(open(os.path.join(R, n)))


REF = 'fd' if '--fd' in sys.argv else 'mot'   # default: Motorica reference (kinetic only); --fd: FineDance reference


def build():
    cmp_b = load('motorica_compare_baselines2.json')['fid']
    cmp_g = load('motorica_compare_gain.json')['fid']
    cmp_b_real = cmp_b['real']['Div_k_mot']['mean']
    beat = {**load('beat_suite_motorica_baselines2.json')['table'], **{'gain': load('beat_suite_motorica_tempo_gain.json')['table']['gain']}}
    fsr = {**load('fsr_table.json')['motorica'], **load('fsr_table_gain_mot.json')['motorica']}
    lock = load('lock_models_motorica_3seed_rescaled_slides.json')['results']
    loud = load('loud_models_motorica_3seed_gain.json')['results']
    spec = {  # row: (fid dict, beat key, fsr key, lock key, loud key)
        'Real dance': (None, 'real', 'real', 'Real dance', 'Real dance'),
        'Lodge': (cmp_b['lodge'], 'lodge', 'lodge', 'Lodge', 'Lodge'),
        'MEGA': (cmp_b['mega'], 'mega', 'mega', 'MEGADance', 'MEGADance'),
        'MACE': (cmp_b['released'], 'released', 'released', 'MACE released', 'MACE released'),
        'Ours (critic v5 + gain)': (cmp_g['gain'], 'gain', 'gain', 'Ours critic v5 + gain',
                                    'Ours (critic v5 + mask + gain)'),
    }
    rows = {}
    for row, (f, b, s, lk, ld) in spec.items():
        if REF == 'fd':
            v = {'FID_k': f and f['FID_k_fd']['mean'], 'FID_g': f and f['FID_g_fd']['mean'],
                 'DIV_k': f and f['Div_k']['mean'], 'DIV_g': f and f['Div_g']['mean']}
        else:   # kinetic only, against real Motorica; no geometric features on the 21-joint body
            v = {'FID_k': f and f['FID_k_mot']['mean'], 'FID_g': None,
                 'DIV_k': f['Div_k_mot']['mean'] if f else cmp_b_real, 'DIV_g': None}
        v |= {
             'FSR': fsr[s]['FSR_pct'], 'BAS': beat[b]['BAS'],
             'Lock': lock[lk]['r_own'], 'Loud': loud[ld]['r_own']}
        rows[row] = {c: {'v': v[c], 'src': '3-seed'} for c in COLS}
    return rows, cmp_b, cmp_g


if __name__ == '__main__':
    rows, cb, cg = build()
    tag = '_fdref' if REF == 'fd' else ''
    json.dump(rows, open(os.path.join(R, f'motorica_main_table{tag}.json'), 'w'), indent=1)
    note = ('Bold = best, underline = second best among the four models (Real dance excluded). FSR in %. '
            'All cells are ours (3 samples per slice, 289 slices); MACE\'s paper has no Motorica numbers. ')
    if REF == 'fd':
        note += ('FID/DIV use the authors\' protocol against their FineDance test reference (how FineDance-like the '
                 'Motorica dances are), not a Motorica reference.')
    else:
        note += ('FID_k / DIV_k: Bailando kinetic features on the shared 21-joint body, against real Motorica '
                 '(Real dance DIV_k is the real slices themselves; no FID for the real row). FID_g / DIV_g are not '
                 'reported: the geometric features need SMPL-24 joints that real Motorica does not have.')
    md = '\n'.join(table_lines(rows, dagger=False, cols=COLS)) + '\n\n' + note
    open(os.path.join(R, f'motorica_main_table{tag}.md'), 'w').write(md + '\n')
    print(md)
