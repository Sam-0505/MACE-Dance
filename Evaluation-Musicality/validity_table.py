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


# older beat metrics (beat_suite.py), tempo-matched null; sign-flip two-sided in the files -> one-sided below
BS = {'FineDance': ('beat_suite_finedance_tempo_null.json', 'real', 'tempo'),
      'Motorica': ('beat_suite_motorica_3seed.json', 'real', 'tempo'),
      'AIST++ (basic + advanced, same BPM)': ('beat_suite_aist.json', 'real', 'same_bpm'),
      'PopDanceSet (clips >= 16 s)': ('beat_suite_popdance.json', 'real', 'tempo')}
aist, pop = R('validity_real_aist.json')['results'], R('validity_real_popdance_min16_trim4.json')['results']
pop_rr = R('validity_real_popdance_min16_trim4_rootrel.json')['results']
fd_l, mo_l = R('loud_models_10seed.json')['results']['Real dance'], R('loud_models_motorica_3seed.json')['results']['Real dance']
fd_k = {k: R(f'lock_models_finedance_10seed_{k}.json')['results']['Real dance'] for k in ('rescaled', 'raw')}
mo_k = {k: R(f'lock_models_motorica_3seed_{k}.json')['results']['Real dance'] for k in ('rescaled', 'raw')}

DATASETS = [  # label, unit, trim, Loud, Lock rescaled, Lock raw
    ('FineDance', '19 songs', '4 s', fd_l, fd_k['rescaled'], fd_k['raw']),
    ('Motorica', '52 pieces', '4 s', mo_l, mo_k['rescaled'], mo_k['raw']),
    ('AIST++ (advanced)', '10 pieces', '4 s', aist['adv']['Loud'], aist['adv']['Lock rescaled'], aist['adv']['Lock raw']),
    ('PopDanceSet (clips >= 16 s)', f"{pop['Loud']['n']} videos", '4 s', pop['Loud'], pop['Lock rescaled'], pop['Lock raw']),
]
rows = []
for name, unit, trim, loud, lock, raw in DATASETS:
    for metric, r, primary in (('Loud', loud, True), ('Lock', lock, True), ('Lock, raw null', raw, False)):
        rows.append({'dataset': name, 'unit': unit, 'trim': trim, 'metric': metric, 'primary': primary, **row(r)})
prim = [r for r in rows if r['primary']]
for r, q in zip(prim, bh([r['p_perm'] for r in prim])):
    r['q'] = q
ref = []
for ds, (f, lab, nl) in BS.items():
    j = R(f)
    t = j['table'][lab]
    for m, mname in (('BAS', 'BAS'), ('PLV', 'Event PLV')):
        d, p2 = t[f'{m}_delta_{nl}'], t[f'{m}_delta_{nl}_p']
        ref.append({'dataset': ds, 'metric': mname, 'r_own': t[m], 'delta': d, 'pos': t[f'{m}_delta_{nl}_pos'],
                    'p_perm': p2 / 2 if d > 0 else 1 - p2 / 2, 'n_groups': j['n_groups']})
sens = {'PopDance, root-relative QoM': {m: row(pop_rr[m]) for m in ('Loud', 'Lock rescaled', 'Lock raw')}}


def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


lines = ['| Dataset (unit) | Metric | r own | r wrong | Δ [95% CI] | Δ>0 | p | q |', '|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    q = fmt_p(r['q']) if 'q' in r else '–'
    lines.append(f"| {r['dataset']} ({r['unit']}{', trim ' + r['trim'] if r['metric'] != 'Loud' else ''}) | {r['metric']} | "
                 f"{r['r_own']:.3f} | {r['r_wrong']:.3f} | {r['delta']:+.3f} [{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}] | "
                 f"{r['n_pos']}/{r['n']} | {fmt_p(r['p_perm'])} | {q} |")
lines += ['', 'Reference: older beat metrics, tempo-matched null (one-sided p from the two-sided sign-flip; no CI, no q)', '',
          '| Dataset | Metric | score | Δ | Δ>0 | p |', '|---|---|---:|---:|---:|---:|']
for r in ref:
    lines.append(f"| {r['dataset']} ({r['n_groups']} groups) | {r['metric']} | {r['r_own']:.3f} | {r['delta']:+.3f} | "
                 f"{r['pos']} | {fmt_p(r['p_perm'])} |")
md = '\n'.join(lines)
with open(os.path.join(HERE, 'results', 'validity_table.md'), 'w') as f:
    f.write(md + '\n')
with open(os.path.join(HERE, 'results', 'validity_table.json'), 'w') as f:
    json.dump({'rows': rows, 'reference_beat_metrics': ref, 'sensitivity': sens, 'q': 'BH over the primary tests (Loud, Lock rescaled) of all dataset rows'}, f, indent=1)
print(md)
print('\nPopDance root-relative:', {m: (round(v['delta'], 3), round(v['p_perm'], 4)) for m, v in sens['PopDance, root-relative QoM'].items()})


# ---------------------------------------------------------------- paper layout: sections BPL (Lock) and IMC (Loud)
mace = {'FineDance': {'Loud': R('loud_models_10seed.json')['results']['MACE released'],
                      'Lock': R('lock_models_finedance_10seed_rescaled.json')['results']['MACE released']}}
# Motorica MACE rows removed: MACE is trained on FineDance, Motorica is out of its domain
SECTIONS = [('Beat-Phase Locking (BPL)', 'Lock', lambda d: d[4]), ('Intensity--Motion Coupling (IMC)', 'Loud', lambda d: d[3])]
paper = []   # (section, dataset, source, r)
for sec, metric, pick in SECTIONS:
    for d in DATASETS:
        name = d[0].replace(' (clips >= 16 s)', '')
        paper.append((sec, name, 'Real Dance', pick(d)))
        if name in mace:
            paper.append((sec, name, 'MACE-Dance', mace[name][metric]))


def p4(p):
    return f'{max(p, 1e-4):.4f}'   # 10,000 permutations: the smallest reportable p is 0.0001


def cells(r):
    return [f"{r['r_own']:.3f}", f"{r['r_wrong']:.3f}", f"{r['delta']:+.3f}",
            f"[{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}]", p4(r['p_perm']), 'Yes' if r['p_perm'] < 0.05 else 'No']


head = ['Dataset', 'Motion source', 'Original', 'Mismatched', 'Δ', '95% CI', 'p', 'Advantage detected?']
md_l = ['| ' + ' | '.join(head) + ' |', '|---|---|---:|---:|---:|---:|---:|:---:|']
tex = [r'\begin{tabular}{llrrrrrc}', r'\toprule',
       r'Dataset & Motion source & Original & Mismatched & $\Delta$ & 95\% CI & $p$ & \shortstack{Advantage\\detected?} \\']
prev_sec = prev_ds = None
for sec, ds, src, r in paper:
    if sec != prev_sec:
        md_l.append(f'| **{sec.replace("--", "–")}** | | | | | | | |')
        tex += [r'\midrule', rf'\multicolumn{{8}}{{l}}{{\textbf{{{sec}}}}} \\', r'\midrule']
        prev_sec = sec
        prev_ds = None
    c = cells(r)
    md_l.append(f"| {ds if ds != prev_ds else ''} | {src} | " + ' | '.join(c) + ' |')
    tc = [c[0], c[1], c[2], c[3].replace('[', '[').replace('+', '+'), c[4].replace('<', '$<$'), c[5]]
    tex.append(f"{ds if ds != prev_ds else ''} & {src} & " + ' & '.join(tc) + r' \\')
    prev_ds = ds
tex += [r'\bottomrule', r'\end{tabular}']
note = ('Original = own music, Mismatched = mean over 3 tempo-matched wrong songs, Δ = Original − Mismatched, p = one-sided '
        'sign-flip permutation test of Δ > 0 (10,000), Advantage detected = p < 0.05. Units: songs (FineDance 19), pieces '
        '(Motorica 52, AIST++ advanced 10), videos (PopDanceSet 132, clips ≥ 16 s). BPL trim: 4 s each end. '
        'AIST++ basic dances (7–12 s) are not included: too short for the 4 s trim. MACE-Dance rows: FineDance only (10 seeds), the dataset MACE was trained on; '
        'Motorica, AIST++ and PopDanceSet are out of its domain.')
open(os.path.join(HERE, 'results', 'validity_table_paper.md'), 'w').write('\n'.join(md_l) + '\n\n' + note + '\n')
open(os.path.join(HERE, 'results', 'validity_table_paper.tex'), 'w').write('\n'.join(tex) + '\n')
print('\n'.join(md_l))
