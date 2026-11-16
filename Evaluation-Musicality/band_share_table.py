"""Markdown tables for the beat-band share diagnostic (band_share.py).

    python Evaluation-Musicality/band_share_table.py   -> results/band_share_table.md
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
DS = [('finedance', 'FineDance', '19 songs, seed 1'), ('motorica', 'Motorica', '52 pieces, 3 samples per slice')]
NAMES = {'Real dance': 'Real dance', 'Lodge': 'Lodge', 'MEGADance': 'MEGA', 'MACE released': 'MACE-Dance',
         'FlowerDance': 'FlowerDance', 'Ours critic v5 + gain': 'Ours (critic v5 + gain)'}


def pct(x, d=2):
    return f'{100 * x:.{d}f}'


def pfmt(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def main():
    J = {d: json.load(open(os.path.join(R, f'band_share_{d}.json'))) for d, _, _ in DS}
    AC = {d: json.load(open(os.path.join(R, f'artificial_controls_{d}.json')))['results'] for d, _, _ in DS}
    L = ['# Beat-band share diagnostic for BPL', '',
         'BPL keeps only the phase of the band-passed movement speed, so a weak but phase-stable beat-band component can score high '
         '(Experiment 2: a bounce with 0.02-0.03 % of its speed variance in the band reaches BPL 0.77 / 0.69). '
         'r_band = Var(q_b) / Var(q): share of the QoM (mean joint speed) variance inside the BPL band (beat rate ± 7.5 %), over '
         'the BPL window (trim 4 s). A diagnostic reported next to BPL, not a replacement.', '',
         '## Table 1. BPL and r_band per model', '',
         'Unit mean = mean over songs / pieces of the slice means [95 % bootstrap CI]; median [IQR] over clips. '
         '"< real p5 / p25" = share of clips whose r_band is below the 5th / 25th percentile of real dance.', '',
         '| Dataset | Model | BPL | r_band % (unit mean [CI]) | r_band % median [IQR] | < real p5 | < real p25 | Spearman BPL vs r_band (clips) |',
         '|---|---|---:|---:|---:|---:|---:|---:|']
    for d, dn, info in DS:
        for k, r in J[d]['results'].items():
            b, c = r['r_band'], r['r_band']['clip_pct']
            L.append(f"| {dn} ({info}) | {NAMES[k]} | {r['BPL']['unit_mean']:.3f} | {pct(b['unit_mean'])} [{pct(b['unit_ci'][0])}, "
                     f"{pct(b['unit_ci'][1])}] | {pct(c['p50'])} [{pct(c['p25'])}, {pct(c['p75'])}] | "
                     f"{pct(r['frac_below_real_p5'], 0)} % | {pct(r['frac_below_real_p25'], 0)} % | {r['spearman_BPL_rband_clips']:+.2f} |")
        a3, a1 = AC[d]['A3|band|own']['mean'], AC[d]['A1|band|own']['mean']
        L.append(f"| {dn} ({info}) | *synthetic A3 bounce (BPL {AC[d]['A3|BPL|own']['mean']:.3f})* | – | {pct(a3)} | – | – | – | – |")
        L.append(f"| {dn} ({info}) | *synthetic A1 sway (BPL {AC[d]['A1|BPL|own']['mean']:.3f})* | – | {pct(a1)} | – | – | – | – |")
    L += ['', '## Table 2. Paired unit-level differences', '',
          'Δ = A − B over songs / pieces [95 % bootstrap CI], two-sided sign-flip p (10,000).', '',
          '| Dataset | A − B | Δ BPL | p | Δ r_band (pp) | p |', '|---|---|---:|---:|---:|---:|']
    for d, dn, _ in DS:
        for k, c in J[d]['comparisons'].items():
            a_, b_ = k.split(' - ')
            x, y = c['BPL'], c['r_band']
            L.append(f"| {dn} | {NAMES[a_]} − {NAMES[b_]} | {x['delta']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] | {pfmt(x['p'])} | "
                     f"{100 * y['delta']:+.2f} [{100 * y['ci'][0]:+.2f}, {100 * y['ci'][1]:+.2f}] | {pfmt(y['p'])} |")
    L += ['', '## Supplementary: amplitude-weighted variants (not validated; not replacements for BPL)', '',
          'BPL_aw = |mean |A| e^{iΔφ}| / mean |A| weights frames by the band amplitude but stays scale-free, so it does not penalise '
          'a weak band. r_lock = |mean A e^{-iθ}|² / (2 Var q) = share of the speed variance carried by the beat-LOCKED sinusoid '
          '(≈ BPL_aw² × r_band): high only when the beat component is both strong and phase-stable.', '',
          '| Dataset | Model | BPL_aw | r_lock % (unit mean [CI]) | r_lock % median | Δ r_lock vs MACE-Dance (pp), p | Δ r_lock vs Real (pp), p |',
          '|---|---|---:|---:|---:|---:|---:|']
    for d, dn, _ in DS:
        comps = J[d]['comparisons']
        for k, r in J[d]['results'].items():
            lk = r['r_lock']

            def cm(key):
                if key not in comps:
                    return '–'
                x = comps[key]['r_lock']
                return f"{100 * x['delta']:+.2f} [{100 * x['ci'][0]:+.2f}, {100 * x['ci'][1]:+.2f}], {pfmt(x['p'])}"
            L.append(f"| {dn} | {NAMES[k]} | {r['BPL_aw']['unit_mean']:.3f} | {pct(lk['unit_mean'])} [{pct(lk['unit_ci'][0])}, "
                     f"{pct(lk['unit_ci'][1])}] | {pct(lk['clip_pct']['p50'])} | {cm(f'{k} - MACE released')} | {cm(f'{k} - Real dance')} |")
    L.append('')
    out = os.path.join(R, 'band_share_table.md')
    open(out, 'w').write('\n'.join(L))
    print(open(out).read())


if __name__ == '__main__':
    main()
