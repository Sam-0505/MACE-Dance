"""Table and null-distribution figure from results/shift_surrogate_{finedance,motorica}.json (+ _mcnull.npz).

    python3 Evaluation-Musicality/shift_surrogate_table.py   # writes results/shift_surrogate_table.md and shift_surrogate_nulls.png
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
DS = [('FineDance', 'finedance'), ('Motorica', 'motorica')]
LABEL = {'circ': 'Circular shift', 'rand': 'Random shift (same song)', 'phase': 'Phase-randomized',
         'ibi': 'IBI-shuffled beats (extra)', 'metro': 'Metronome grid, own phase (extra)',
         'metro_rand': 'Metronome grid, random phase (extra)', 'mismatch': 'Mismatched song'}
B6 = ['circ', 'rand', 'phase', 'ibi', 'metro', 'mismatch']
MET = [('bpl', 'Beat-Phase Locking (BPL, offset-free)', B6),
       ('bplf', 'BPL-fixed: offset-sensitive, one dataset-wide offset (leave-one-unit-out)', B6 + ['metro_rand']),
       ('bplw', 'BPL-within: offset-sensitive, offset from other slices of the same take', B6 + ['metro_rand']),
       ('eplv', 'Event PLV (beat_suite.py; offset-free, accents inside their beat interval)', B6),
       ('bas', 'BAS (authors\' Beat Alignment Score; offset-sensitive)', B6 + ['metro_rand']),
       ('imc', 'Intensity–Motion Coupling (IMC)', ['circ', 'rand', 'phase', 'mismatch'])]


def fp(p):
    return f'{max(p, 1e-4):.4f}'


data = {k: json.load(open(os.path.join(R, f'shift_surrogate_{k}.json'))) for _, k in DS}
lines = ['| Dataset | Condition | Original | Control | Δ | 95% CI | p (units) | p (MC null) | Slices > null 95th pct | Mean percentile of Original | Detected? |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|']
for m, title, conds in MET:
    lines.append(f'| **{title}** | | | | | | | | | | |')
    for name, k in DS:
        prev = ''
        for c in conds:
            r = data[k]['results'][m][c]
            pmc = fp(r['p_mc']) if 'p_mc' in r else '–'
            share = f"{100 * r['pct_slices_above_q95']:.0f}%" if c not in ('metro', 'mismatch') else '–'
            pct = f"{r['mean_percentile']:.2f}" if c not in ('metro', 'mismatch') else '–'
            det = 'Yes' if r['p_perm'] < 0.05 and r['delta'] > 0 else 'No'
            lines.append(f"| {name if name != prev else ''} | {LABEL[c]} | {r['orig']:.3f} | {r['null']:.3f} | {r['delta']:+.3f} | "
                         f"[{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}] | {fp(r['p_perm'])} | {pmc} | {share} | {pct} | {det} |")
            prev = name
s = data['finedance']['settings']
note = ('Original = own music (BPL: for the random-shift row the own window of the same full-song beat track, so both sides share one '
        'pipeline). Control = mean of the per-slice null (circular: all 781 integer shifts of 4 s..T−4 s; random shift: 200 same-song '
        'windows ≥ 5 s away; phase-randomized: 200 IAAFT RMS surrogates for IMC, 100 beat-tracked phase-randomized onset envelopes for BPL; '
        'IBI shuffle: 200; metronome: one regular grid). BPL controls keep the own mean beat frequency (beats rescaled to the own tempo). '
        'Δ = Original − Control per unit (song 19 / piece 52); p (units) = one-sided sign-flip permutation (10,000); '
        'p (MC null) = observed mean over units against 2,000 draws of the same statistic with one random null draw per slice; '
        '"Slices > null 95th pct" = share of slices whose Original exceeds the 95th percentile of their own null (5% expected by chance); '
        'BPL-fixed / BPL-within score = Re(mean exp(i(φ_QoM − θ_beats − c))); c = circular mean phase of the real-dance own-beat locking value over all other units (fixed) or over the other non-overlapping slices of the same take (within; slices without a partner dropped: FineDance 75/84, Motorica 287/289); controls never enter c. '
        'mean percentile = mean of 1 − empirical one-sided p of the Original within its null (0.5 = chance). Detected = p (units) < 0.05 and Δ > 0.')
open(os.path.join(R, 'shift_surrogate_table.md'), 'w').write('\n'.join(lines) + '\n\n' + note + '\n')
print('\n'.join(lines))

# ---------------------------------------------------------------- figure: Monte-Carlo null of the dataset mean vs observed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 6, figsize=(28, 6.5))
colors = {'circ': '#1f77b4', 'rand': '#2ca02c', 'phase': '#9467bd', 'ibi': '#ff7f0e'}
for row, (name, k) in enumerate(DS):
    mc = np.load(os.path.join(R, f'shift_surrogate_{k}_mcnull.npz'))
    for col, (m, title, conds) in enumerate(MET):
        a = ax[row, col]
        obs = data[k]['results'][m]['circ']['orig']
        for c in ('circ', 'rand', 'phase', 'ibi'):
            if f'{m}_{c}' in mc:
                a.hist(mc[f'{m}_{c}'], bins=40, histtype='stepfilled', alpha=0.45, color=colors[c], label=LABEL[c])
        a.axvline(obs, color='k', lw=2, label='Original')
        if m != 'imc':
            a.axvline(data[k]['results'][m]['metro']['null'], color='r', ls='--', lw=1.5, label='Metronome grid')
        short = {'imc': 'IMC', 'eplv': 'Event PLV', 'bas': 'BAS'}.get(m, m.upper())
        a.set_title(f'{name}: {short}', fontsize=10)
        a.set_xlabel('dataset mean over units')
        a.set_yticks([])
        if row == 0 and col == 0:
            a.legend(fontsize=7, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(R, 'shift_surrogate_nulls.png'), dpi=150)
print('saved figure')
