"""Motorica table for the AIST++-comparison models: Real dance, EDGE, Bailando (the AIST++-trained MACE is not shown, see below).
Layout and sources as motorica_main_table.py (Lock / Loud = r_own, the main table's convention; Delta and p are in the json):
FID_k, DIV_k from motorica_compare_aist.json (Motorica reference, kinetic only: FID_g / DIV_g need SMPL-24 joints, which real Motorica
lacks), FSR from fsr_table.json (real) and fsr_table_aist_mot.json, BAS from beat_suite_motorica_aist.json,
Lock from lock_models_motorica_1seed_rescaled_aist.json, Loud from loud_models_motorica_1seed_only.json.
    python3 aist_motorica_table.py     -> results/aist_motorica_table.{json,md}
"""
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
L = lambda n: json.load(open(os.path.join(R, n)))  # noqa: E731
cmp_, beat, lock, loud = L('motorica_compare_aist.json')['fid'], L('beat_suite_motorica_aist.json')['table'], \
    L('lock_models_motorica_1seed_rescaled_aist.json')['results'], L('loud_models_motorica_1seed_only.json')['results']
fsr = {**L('fsr_table.json')['motorica'], **L('fsr_table_aist_mot.json')['motorica']}
spec = {'Real dance': (None, 'real', 'real', 'Real dance'), 'EDGE': ('edge', 'edge', 'edge', 'EDGE'),
        'Bailando': ('bailando', 'bailando', 'bailando', 'Bailando')}
# The AIST++-trained model's Motorica row is NOT part of the comparison (user decision 2026-11-11: it fails out of domain, see
# aist_transfer_probe*.py). Its raw scores stay in motorica_compare_aist.json etc.; add ('ours_aist', ...) here to show it again.
rows = {}
for name, (c, b, f, k) in spec.items():
    rows[name] = {'FID_k': cmp_[c]['FID_k_mot']['mean'] if c else None,
                  'DIV_k': (cmp_[c]['Div_k_mot']['mean'] if c else cmp_['real']['Div_k_mot']['mean']),
                  'FSR': fsr[f]['FSR_pct'], 'BAS': beat[b]['BAS'],
                  'Lock': lock[k]['r_own'], 'Lock_delta': lock[k]['delta'], 'Lock_p': lock[k]['p_perm'],
                  'Loud': loud[k]['r_own'], 'Loud_delta': loud[k]['delta'], 'Loud_p': loud[k]['p_perm']}
json.dump(rows, open(os.path.join(R, 'aist_motorica_table.json'), 'w'), indent=1)
cols = [('FID_k', 'FID_k ↓', '{:.2f}', False), ('FSR', 'FSR ↓', '{:.1f}', False), ('DIV_k', 'DIV_k ↑', '{:.2f}', True),
        ('BAS', 'BAS ↑', '{:.3f}', True), ('Lock', 'Lock ↑', '{:.3f}', True), ('Loud', 'Loud ↑', '{:.3f}', True)]
mod = [k for k in rows if k != 'Real dance']
out = ['| Model | ' + ' | '.join(c[1] for c in cols) + ' |', '|---|' + '---:|' * len(cols)]
for k, r in rows.items():
    cells = []
    for key, _, fmt, hi in cols:
        v = r[key]
        if v is None:
            cells.append('–')
            continue
        s = fmt.format(v)
        if k in mod:
            vals = sorted(rows[m][key] for m in mod)
            s = f'**{s}**' if v == (vals[-1] if hi else vals[0]) else s
        cells.append(s)
    out.append(f'| {k} | ' + ' | '.join(cells) + ' |')
open(os.path.join(R, 'aist_motorica_table.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
for k, r in rows.items():
    print(k, 'Lock delta %+.3f p %.4f | Loud delta %+.3f p %.4f' % (r['Lock_delta'], r['Lock_p'], r['Loud_delta'], r['Loud_p']))
