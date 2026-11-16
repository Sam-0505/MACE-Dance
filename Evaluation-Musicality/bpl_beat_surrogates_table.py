"""Markdown table from results/bpl_beat_surrogates_{finedance,motorica}.json -> results/bpl_beat_surrogates_table.md

    python3 Evaluation-Musicality/bpl_beat_surrogates_table.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
DS = [('FineDance', 'finedance'), ('Motorica', 'motorica')]
ROWS = [('metro', 'Metronome grid (own tempo and mean phase)', 'tempo, mean phase', 'all local timing removed'),
        ('ibi_global', 'IBI shuffle, global (no rescale)', 'interval set, first/last beat, mean interval', 'interval order -> drift'),
        ('ibi_global_r', 'IBI shuffle, global, rescaled (= committed control)', 'interval set, mean interval', 'interval order -> drift'),
        ('ibi_block4', 'IBI shuffle within 4-beat blocks', 'interval set, every 4th beat position', 'interval order inside a bar'),
        ('dev_shuffle', 'Deviation shuffle', 'line fit, deviation distribution', 'deviation order'),
        ('dev_iaaft', 'Phase-randomized deviations (IAAFT)', 'line fit, deviation distribution and spectrum', 'deviation phases'),
        ('phase_audio', 'Phase-randomized onset envelope -> tracker (= committed)', 'envelope spectrum, mean interval', 'onset timing'),
        ('phase_audio_clean', '… only tracks without skipped/doubled beats', 'as above', 'as above')]
d = {k: json.load(open(os.path.join(R, f'bpl_beat_surrogates_{k}.json'))) for _, k in DS}
L = ['| Control | Preserved | Randomized | Dataset | Original | Control | Δ [95% CI] | p (units) | p (MC) | RMS / max drift (beats) |',
     '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
for key, name, keep, rand in ROWS:
    for j, (dn, k) in enumerate(DS):
        r = d[k]['results'][key]
        dg = r.get('diag', {})
        drift = f"{dg['rms_drift_beats']:.2f} / {dg['max_drift_beats']:.2f}" if 'rms_drift_beats' in dg else '–'
        pmc = f"{max(r['p_mc'], 1e-4):.4f}" if 'p_mc' in r else '–'
        L.append(f"| {name if j == 0 else ''} | {keep if j == 0 else ''} | {rand if j == 0 else ''} | {dn} | {r['orig']:.3f} | "
                 f"{r['null']:.3f} | {r['delta']:+.3f} [{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}] | "
                 f"{max(r['p_perm'], 1e-4):.4f} | {pmc} | {drift} |")
own = ' · '.join(f"{dn}: own-beat deviation RMS {d[k]['own_beats']['rms_dev_frames']:.2f} frames, lag-1 IBI autocorr "
                 f"{d[k]['own_beats']['lag1_autocorr_ibi']:+.2f}" for dn, k in DS)
skip = ' · '.join(f"{dn} {100 * d[k]['results']['phase_audio'].get('diag', {}).get('share_skip_or_double', float('nan')):.1f}%"
                  for dn, k in DS)
note = ('Motion, band-pass (15 % around the own beat rate), Hilbert phase and 4 s trim are the BPL definition in every row; '
        'only the beat reference changes. Δ = Original − mean control per unit (FineDance 19 songs, Motorica 52 pieces); '
        'p (units) = one-sided sign-flip permutation (10,000); p (MC) = observed mean over units against 2,000 draws of one '
        'surrogate per slice. Drift = RMS and max |surrogate beat − own beat| over the clip, in beats. No control rolls the '
        f'motion, so none has a seam. {own}. Tracker tracks with a skipped/doubled beat: {skip}.')
open(os.path.join(R, 'bpl_beat_surrogates_table.md'), 'w').write('\n'.join(L) + '\n\n' + note + '\n')
print('\n'.join(L) + '\n\n' + note)
