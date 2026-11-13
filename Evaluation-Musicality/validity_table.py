"""Metric validity table on real dance: Delta = r(own music) - r(wrong music), per dataset and metric.

Reads the real-dance rows of loud_models_10seed.json (FineDance), loud_models_motorica_3seed.json,
lock_models_{finedance_10seed,motorica_3seed}_{rescaled,raw}.json and validity_real_{aist,popdance_*}.json.
q = Benjamini-Hochberg over the PRIMARY tests of the table (Loud and Lock with the rescaled null, every dataset row);
the raw-null Lock rows are supplementary (the raw null credits following the tempo) and carry no q.

    python3 Evaluation-Musicality/validity_table.py   # writes results/validity_table.md and .json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda f: json.load(open(os.path.join(HERE, 'results', f)))  # noqa: E731


def bh(p):
    n = len(p)
    order = sorted(range(n), key=lambda i: p[i])
    q, prev = [0.0] * n, 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def row(r):
    return {k: r[k] for k in ('r_own', 'r_wrong', 'delta', 'delta_ci', 'n_pos', 'n', 'p_perm', 'p_wilcoxon')}


aist, pop = R('validity_real_aist.json')['results'], R('validity_real_popdance_min16_trim4.json')['results']
pop_rr = R('validity_real_popdance_min16_trim4_rootrel.json')['results']
fd_l, mo_l = R('loud_models_10seed.json')['results']['Real dance'], R('loud_models_motorica_3seed.json')['results']['Real dance']
fd_k = {k: R(f'lock_models_finedance_10seed_{k}.json')['results']['Real dance'] for k in ('rescaled', 'raw')}
mo_k = {k: R(f'lock_models_motorica_3seed_{k}.json')['results']['Real dance'] for k in ('rescaled', 'raw')}

DATASETS = [  # label, unit, trim, Loud, Lock rescaled, Lock raw
    ('FineDance', '19 songs', '4 s', fd_l, fd_k['rescaled'], fd_k['raw']),
    ('Motorica', '52 pieces', '4 s', mo_l, mo_k['rescaled'], mo_k['raw']),
    ('AIST++ advanced', '10 pieces', '4 s', aist['adv']['Loud'], aist['adv']['Lock rescaled'], aist['adv']['Lock raw']),
    ('AIST++ basic', '60 pieces', '1 s', aist['basic']['Loud'], aist['basic']['Lock rescaled'], aist['basic']['Lock raw']),
    ('PopDanceSet (clips >= 16 s)', f"{pop['Loud']['n']} videos", '4 s', pop['Loud'], pop['Lock rescaled'], pop['Lock raw']),
]
rows = []
for name, unit, trim, loud, lock, raw in DATASETS:
    for metric, r, primary in (('Loud', loud, True), ('Lock', lock, True), ('Lock, raw null', raw, False)):
        rows.append({'dataset': name, 'unit': unit, 'trim': trim, 'metric': metric, 'primary': primary, **row(r)})
prim = [r for r in rows if r['primary']]
for r, q in zip(prim, bh([r['p_perm'] for r in prim])):
    r['q'] = q
sens = {'PopDance, root-relative QoM': {m: row(pop_rr[m]) for m in ('Loud', 'Lock rescaled', 'Lock raw')}}


def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


lines = ['| Dataset (unit) | Metric | r own | r wrong | Δ [95% CI] | Δ>0 | p | q |', '|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    q = fmt_p(r['q']) if 'q' in r else '–'
    lines.append(f"| {r['dataset']} ({r['unit']}{', trim ' + r['trim'] if r['metric'] != 'Loud' else ''}) | {r['metric']} | "
                 f"{r['r_own']:.3f} | {r['r_wrong']:.3f} | {r['delta']:+.3f} [{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}] | "
                 f"{r['n_pos']}/{r['n']} | {fmt_p(r['p_perm'])} | {q} |")
md = '\n'.join(lines)
with open(os.path.join(HERE, 'results', 'validity_table.md'), 'w') as f:
    f.write(md + '\n')
with open(os.path.join(HERE, 'results', 'validity_table.json'), 'w') as f:
    json.dump({'rows': rows, 'sensitivity': sens, 'q': 'BH over the primary tests (Loud, Lock rescaled) of all dataset rows'}, f, indent=1)
print(md)
print('\nPopDance root-relative:', {m: (round(v['delta'], 3), round(v['p_perm'], 4)) for m, v in sens['PopDance, root-relative QoM'].items()})
