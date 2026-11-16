"""Markdown tables for Experiment 2 (artificial_controls.py): metric specificity and limitations of BPL (Lock) and IMC (Loud).

    python Evaluation-Musicality/artificial_controls_table.py   -> results/artificial_controls_table.md
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
DS = [('finedance', 'FineDance (19 songs)'), ('motorica', 'Motorica (52 pieces)')]
MODELS = [('MACE', 'MACE-Dance', 'MACE released'), ('Ours (critic v5 + gain)', 'Ours', 'Ours critic v5 + gain')]

ROWS = [
    ('Real', 'Real dance', 'reference'),
    ('A1', 'A1 sway, follows tracked beats', 'one speed peak per beat'),
    ('A2', 'A2 sway, metronome at own tempo', 'one speed peak per beat, no tempo fluctuation'),
    ('A3', 'A3 bounce, follows tracked beats', 'two speed peaks per beat'),
    ('A4', 'A4 bounce, metronome at own tempo', 'two speed peaks per beat, no tempo fluctuation'),
    ('B1', 'B1 speed = raw RMS', 'speed follows frame-level loudness'),
    ('B2', 'B2 speed = smoothed RMS (sigma 15)', 'speed follows the slow loudness envelope'),
]


def load():
    return {d: json.load(open(os.path.join(R, f'artificial_controls_{d}.json'))) for d, _ in DS}


def pfmt(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def dfmt(c):
    return f"{c['delta']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"


def sig(c):
    return '*' if c['p'] < 0.05 else ''


def main():
    J = load()
    main_t = {d: json.load(open(os.path.join(R, f'{d}_main_table.json'))) for d, _ in DS}
    band = {d: json.load(open(os.path.join(R, f'band_share_{d}.json')))['results'] for d, _ in DS}
    L = ['# Experiment 2: can simple non-expressive motion score high BPL or IMC?', '',
         'Metric specificity and limitations, not an attempt to break the metrics. Synthetic motions are whole-body translations of '
         'a fixed pose built per real slice (same slices, beats, RMS, trim and mismatch pairs as the validity table); both metrics '
         'read only QoM (mean joint speed). Unit = song (FineDance) / piece (Motorica); values are unit means. '
         '* = differs from Real dance on own music (two-sided sign-flip p < 0.05 over units). '
         'r_band = Var(band-passed QoM) / Var(QoM) over the BPL window (band = beat rate ± 7.5 %; band_share.py). Chance floor of BPL ≈ 0.30 (random-cut shuffles, C2).', '',
         '## Table 1. Scores on the own music', '',
         '| Motion | Construction | ' + ' | '.join(f'{n} BPL | {n.split()[0]} IMC | {n.split()[0]} r_band' for _, n in DS) + ' |',
         '|---|---|' + '---:|---:|---:|' * len(DS)]
    for key, name, what in ROWS:
        cells = []
        for d, _ in DS:
            r, c = J[d]['results'], J[d]['comparisons']
            for met in ('BPL', 'IMC'):
                v = r[f'{key}|{met}|own']['mean']
                s = '' if key == 'Real' else sig(c[f'{key}|{met}|own - Real|{met}|own'])
                cells.append(f'{v:.3f}{s}')
            cells.append(f"{100 * r[f'{key}|band|own']['mean']:.2f} %")
        L.append(f'| {name} | {what} | ' + ' | '.join(cells) + ' |')
    for mk, mname, bk in MODELS:
        cells = []
        for d, _ in DS:
            m = main_t[d][mk]
            cells += [f"{m['Lock']['v']:.3f}", f"{m['Loud']['v']:.3f}", f"{100 * band[d][bk]['r_band']['unit_mean']:.2f} %"]
        L.append(f'| {mname} (main table) | generated | ' + ' | '.join(cells) + ' |')
    L += ['', 'Model rows are copied from the main tables (FineDance seed 1; Motorica 3 samples per slice) for scale; their r_band is from band_share.py (same samples).', '']

    # ---- Table 2: own vs same-tempo mismatched music (D for BPL, B for IMC)
    L += ['## Table 2. Own vs same-tempo mismatched music (3 tempo-matched songs, beats rescaled to the own tempo)', '',
          'Does the score depend on WHICH song plays, or only on its tempo? Δ = own − mismatched, 95 % bootstrap CI over units, '
          'two-sided sign-flip p. "vs Real gap" = this row\'s Δ minus the Real-dance Δ.', '',
          '| Metric | Motion | Dataset | own | mismatched | Δ [95 % CI] | p | vs Real gap |', '|---|---|---|---:|---:|---:|---:|---:|']
    for met, keys in (('BPL', ['Real', 'A1', 'A2', 'A3', 'A4', 'B1']), ('IMC', ['Real', 'B1', 'B2'])):
        for key in keys:
            name = dict((k, n) for k, n, _ in ROWS)[key]
            for d, dn in DS:
                r, c = J[d]['results'], J[d]['comparisons']
                cc = c[f'{key}|{met}|own - {key}|{met}|mis']
                gap = '–' if key == 'Real' else dfmt(c[f'gap {key}|{met} - gap Real|{met}'])
                L.append(f"| {met} | {name} | {dn.split()[0]} | {r[f'{key}|{met}|own']['mean']:.3f} | "
                         f"{r[f'{key}|{met}|mis']['mean']:.3f} | {dfmt(cc)} | {pfmt(cc['p'])} | {gap} |")
    L.append('')

    # ---- Table 3: shuffling real motion
    ks = J['finedance']['settings']['ks_beats']
    L += ['## Table 3. Temporal shuffling of real motion (200 permutations per slice)', '',
          'C1 cuts at every k-th beat (segments keep their phase relative to the beat); C2 makes the same number of cuts at random '
          'frames (phase scrambled). Head and tail outside the cut region stay in place. Δ = shuffled − original real dance '
          '[95 % CI], p two-sided sign-flip.', '',
          '| Shuffle | k beats | Dataset | BPL | Δ BPL | p | IMC | Δ IMC | p |', '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for d, dn in DS:
        r = J[d]['results']
        L.append(f"| none (Real dance) | – | {dn.split()[0]} | {r['Real|BPL|own']['mean']:.3f} | – | – | "
                 f"{r['Real|IMC|own']['mean']:.3f} | – | – |")
    for cname, lab in (('C1', 'C1 beat-aligned'), ('C2', 'C2 random cuts')):
        for k in ks:
            for d, dn in DS:
                r, c = J[d]['results'], J[d]['comparisons']
                cb, ci = c[f'{cname}_k{k}|BPL|own - Real|BPL|own'], c[f'{cname}_k{k}|IMC|own - Real|IMC|own']
                L.append(f"| {lab} | {k} | {dn.split()[0]} | {r[f'{cname}_k{k}|BPL|own']['mean']:.3f} | {dfmt(cb)} | {pfmt(cb['p'])} | "
                         f"{r[f'{cname}_k{k}|IMC|own']['mean']:.3f} | {dfmt(ci)} | {pfmt(ci['p'])} |")
    L.append('')
    out = os.path.join(R, 'artificial_controls_table.md')
    open(out, 'w').write('\n'.join(L))
    print(open(out).read())


if __name__ == '__main__':
    main()
