"""Table and curve from results/tempo_detune_{finedance,motorica}.json.

    cd Expert-Motion && .venv/bin/python ../Evaluation-Musicality/tempo_detune_table.py
writes results/tempo_detune_table.md and results/tempo_detune_curve.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
DS = [('FineDance', 'finedance', '#1f77b4'), ('Motorica', 'motorica', '#d62728')]
data = {k: json.load(open(os.path.join(R, f'tempo_detune_{k}.json'))) for _, k, _ in DS}

lines = []
for name, k, _ in DS:
    o = data[k]['original']
    lines += [f'**{name}** (Original, own beats: BPL {o["bpl"]:.3f} [{o["ci"][0]:.3f}, {o["ci"][1]:.3f}])', '',
              '| Detuning | BPL [95% CI] | Original − grid [95% CI] | p | Right tempo − grid [95% CI] | p | Drift (beats) |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for r in data[k]['rows']:
        g0 = ('–', '–') if r['detune_pct'] == 0 else (
            f"{r['delta_vs_grid0']:+.3f} [{r['delta0_ci'][0]:+.3f}, {r['delta0_ci'][1]:+.3f}]", f"{max(r['p_grid0_gt'], 1e-4):.4f}")
        lines.append(f"| {r['detune_pct']:+.1f}% | {r['bpl']:.3f} [{r['bpl_ci'][0]:.3f}, {r['bpl_ci'][1]:.3f}] | "
                     f"{r['delta_vs_original']:+.3f} [{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}] | "
                     f"{max(r['p_original_gt'], 1e-4):.4f} | {g0[0]} | {g0[1]} | {r['drift_beats_median']:.2f} |")
    lines.append('')
note = ('Real dance scored with BPL against a perfectly regular beat grid whose interval is the own mean beat interval × (1 + e); '
        'e = 0 is the metronome control. The band-pass follows the grid tempo, as BPL would for a song e % faster or slower. '
        'Right tempo − grid = paired difference between the 0 % grid and the detuned grid (one-sided test of the 0 % grid scoring higher). Drift = median accumulated phase error over the scored 26 s window, in beats. p = one-sided sign-flip permutation of '
        'Original − grid > 0 over units (FineDance 19 songs, Motorica 52 pieces; 10,000). Beyond ±7.5 % the true beat rate '
        'leaves the 15 % band.')
open(os.path.join(R, 'tempo_detune_table.md'), 'w').write('\n'.join(lines) + '\n\n' + note + '\n')
print('\n'.join(lines))

fig, ax = plt.subplots(figsize=(7, 4))
for name, k, col in DS:
    rows = data[k]['rows']
    x = [r['detune_pct'] for r in rows]
    ax.plot(x, [r['bpl'] for r in rows], 'o-', color=col, label=f'{name}: regular grid at detuned tempo')
    ax.fill_between(x, [r['bpl_ci'][0] for r in rows], [r['bpl_ci'][1] for r in rows], color=col, alpha=0.15)
    ax.axhline(data[k]['original']['bpl'], color=col, ls='--', lw=1.2, label=f'{name}: own beats')
ax.axvspan(-7.5, 7.5, color='0.92', zorder=0)
ax.set_xscale('symlog', linthresh=1)
ax.set_xticks([-10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10])
ax.set_xticklabels(['−10', '−5', '−3', '−2', '−1', '−0.5', '0', '0.5', '1', '2', '3', '5', '10'])
ax.set_xlabel('tempo detuning of the beat grid (%)')
ax.set_ylabel('BPL (real dance)')
ax.set_title('BPL needs the tempo to within about 1 %', fontsize=11)
ax.legend(fontsize=7, loc='lower center')
fig.tight_layout()
fig.savefig(os.path.join(R, 'tempo_detune_curve.png'), dpi=150)
print('saved figure')
