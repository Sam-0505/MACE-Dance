"""Markdown table from results/mismatch_ranking_{finedance,motorica}.json -> results/mismatch_ranking_table.md

    python3 Evaluation-Musicality/mismatch_ranking_table.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
DS = [('FineDance', 'finedance'), ('Motorica', 'motorica')]
MODELS = ['BAS', 'BPL', 'IMC', 'BPL+IMC', 'BAS+BPL+IMC']
COMPS = ['BAS+BPL+IMC - BAS', 'BPL+IMC - BAS', 'BAS+BPL+IMC - BPL+IMC', 'BPL+IMC - IMC', 'BPL+IMC - BPL', 'BPL - BAS', 'IMC - BAS']
d = {k: json.load(open(os.path.join(R, f'mismatch_ranking_{k}.json'))) for _, k in DS}
ci = lambda x, v: f"{x:.3f} [{v[0]:.3f}, {v[1]:.3f}]"            # noqa: E731
dci = lambda x: f"{x['delta']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"  # noqa: E731

hdr = '| Score | ' + ' | '.join(f'{dn} top-1 | {dn} MRR' for dn, _ in DS) + ' |'
L = ['**Identifying the original music among 4 candidates (original + 3 tempo-matched mismatches), real dance**', '', hdr,
     '|---|' + '---:|' * (2 * len(DS))]
L.append('| Chance | ' + ' | '.join('0.250 | 0.521' for _ in DS) + ' |')
for m in MODELS:
    L.append(f'| {m} | ' + ' | '.join(f"{ci(d[k]['results'][m]['top1'], d[k]['results'][m]['top1_ci'])} | "
                                       f"{ci(d[k]['results'][m]['mrr'], d[k]['results'][m]['mrr_ci'])}" for _, k in DS) + ' |')
L += ['', '**Paired differences (held-out)**', '',
      '| Comparison | ' + ' | '.join(f'{dn} Δ top-1 | p | {dn} Δ MRR | p' for dn, _ in DS) + ' |',
      '|---|' + '---:|' * (4 * len(DS))]
for c in COMPS:
    L.append(f'| {c.replace(" - ", " − ")} | ' + ' | '.join(
        f"{dci(d[k]['differences'][c]['top1'])} | {d[k]['differences'][c]['top1']['p']:.3f} | "
        f"{dci(d[k]['differences'][c]['mrr'])} | {d[k]['differences'][c]['mrr']['p']:.3f}" for _, k in DS) + ' |')
L += ['', '**Learned weights (z-scored features; mean [min, max] over leave-one-song-out folds)**', '',
      '| Model | Dataset | ' + ' | '.join(['BAS', 'BPL', 'IMC']) + ' |', '|---|---|---:|---:|---:|']
for m in ('BPL+IMC', 'BAS+BPL+IMC'):
    for dn, k in DS:
        r = d[k]['results'][m]
        cells = [f"{r['weights_mean'][x]:+.2f} [{r['weights_min'][x]:+.2f}, {r['weights_max'][x]:+.2f}]"
                 if x in r['weights_mean'] else '–' for x in ('BAS', 'BPL', 'IMC')]
        L.append(f'| {m} | {dn} | ' + ' | '.join(cells) + ' |')
n = ' · '.join(f"{dn}: {d[k]['settings']['n_slices']} slices, {d[k]['settings']['n_songs']} songs" for dn, k in DS)
note = (f'Real-dance slices ({n}) ranked against their own music and the 3 tempo-matched mismatched slices of the validity '
        'table. For BPL and BAS the mismatched beat lists are rescaled to the own mean beat interval (validity-table protocol), '
        'so the task is distinguishing the original music from tempo-controlled mismatches, not unrestricted music retrieval. '
        'Single scores rank by the raw value (higher = better, no fitting). Combinations: conditional-logit ranking model '
        '(softmax over the 4 candidates, L2 penalty chosen by inner song-grouped 5-fold CV), features z-scored on the training '
        'fold only; outer CV leave-one-song-out (FineDance song / Motorica music piece), so no song is in both training and test. '
        'Ties broken uniformly at random (in expectation). 95 % CIs: song-cluster bootstrap (10,000). p: two-sided song-level '
        'sign-flip permutation (10,000).')
txt = '\n'.join(L) + '\n\n' + note + '\n'
open(os.path.join(R, 'mismatch_ranking_table.md'), 'w').write(txt)
print(txt)
