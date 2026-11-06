"""Before / after table for our sampling fixes on Lodge (2026-11-06; Expert-Motion/run_lodge_guided.sh).

Collects the results files written by run_lodge_guided.sh (no recomputation) into results/lodge_guided_summary.json and
prints a markdown table: unguided Lodge, guided Lodge (guidance 0.003 + frozen root rotation + foot term), released MACE,
our final MACE. Also summarises the per-step guidance logs of the guided Lodge run.

    python Evaluation-Musicality/lodge_guided_table.py
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
LOGS = '/home/sam0505/Documents/Code/dance_baselines/LODGE/finedance_work_guided/guide_frz_foot/log'
ROWS = [  # label, beat_suite / fsr / tmd / jitter label, data dir, finetune_eval label
    ('Lodge (official)', 'lodge', 'mace_test_lodge', 'fd_lodge'),
    ('Lodge + guidance (frz) + foot', 'lodge_guide_frz_foot', 'mace_test_lodge_guide_frz_foot',
     'fd_lodge_guide_frz_foot'),
    ('MACE released', 'released', 'mace_test', 'released'),
    ('MACE final (mask + guidance (frz) + foot)', 'final', 'mace_test_mask2s_guide10_s0.003_foot10_frz',
     'mask2s_guide10_s0.003_foot10_frz'),
]
PAIRS = {'Loud': 'loud slow -> energy', 'Perc': 'perc fast -> upper accel', 'Perc_slow': 'perc slow -> energy'}


def load(name):
    return json.load(open(os.path.join(R, name)))


def main():
    bs, ft, au = load('beat_suite_finedance_lodge_guided.json'), load('finetune_sweep.json'), load('authors_metrics.json')
    fsr, tmd, jit = load('fsr_table_lodge_guided.json'), load('tmd_bench_table_lodge_guided.json'), \
        load('rotation_jitter_lodge_guided.json')
    out = {}
    for name, lab, d, fl in ROWS:
        b = bs['table'][lab]
        row = {'Lock_tempo': b['PLV_delta_tempo'], 'Lock_tempo_p': b['PLV_delta_tempo_p'],
               'Lock_tempo_pos': b['PLV_delta_tempo_pos'], 'Lock_other': b['PLV_delta_other'],
               'Lock_other_p': b['PLV_delta_other_p'], 'BAS': b['BAS']}
        for k, p in PAIRS.items():
            v = ft[fl]['pairs'][p]
            row[k], row[k + '_pos'], row[k + '_p'] = v['d_mace'], f"{v['n_mace_pos']}/{v['n']}", v['H2_wilcoxon_p']
        t = tmd['finedance'][lab]['2d']
        row.update(VBCS=t['VBCS'], ABHS=t['ABHS'])
        for k in ('FID_k', 'FID_g', 'PFC', 'Div_k', 'Div_g'):
            row[k] = au[d][k]['mean']
        row['FSR_pct'] = fsr['finedance'][lab]['FSR_pct']
        row['pelvis_jitter'], row['torso_jitter'] = jit[lab]['pelvis_jitter'], jit[lab]['torso_jitter']
        out[name] = row
    for lab in ('real',):
        b = bs['table'][lab]
        out['real FineDance'] = {'Lock_tempo': b['PLV_delta_tempo'], 'Lock_tempo_pos': b['PLV_delta_tempo_pos'],
                                 'Lock_other': b['PLV_delta_other'], 'BAS': b['BAS'],
                                 'VBCS': tmd['finedance']['real']['2d']['VBCS'],
                                 'ABHS': tmd['finedance']['real']['2d']['ABHS'],
                                 'FSR_pct': load('fsr_table.json')['finedance']['real']['FSR_pct'],
                                 'pelvis_jitter': jit['real']['pelvis_jitter'], 'torso_jitter': jit['real']['torso_jitter']}
    logs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(LOGS, '*.json')))]
    if logs:
        s = np.array([[r['score'] for r in L] for L in logs])
        out['guidance_log'] = {
            'n_samples': len(logs), 'guided_steps': s.shape[1],
            'score_first_step': float(s[:, 0].mean()), 'score_mid_step': float(s[:, s.shape[1] // 2].mean()),
            'score_last_step': float(s[:, -1].mean()),
            'root_rot_share_of_grad': float(np.mean([r['root_rot_share'] for L in logs for r in L])),
            'foot_loss_before': float(np.mean([r['foot_before'] for L in logs for r in L])),
            'foot_loss_after': float(np.mean([r['foot_after'] for L in logs for r in L])),
            'contact_share': float(np.mean([r['contact_share'] for L in logs for r in L]))}
    json.dump(out, open(os.path.join(R, 'lodge_guided_summary.json'), 'w'), indent=1)

    cols = ['Lock_tempo', 'Lock_other', 'Loud', 'Perc', 'Perc_slow', 'VBCS', 'ABHS', 'BAS', 'FID_k', 'FID_g', 'PFC',
            'FSR_pct', 'Div_k', 'pelvis_jitter', 'torso_jitter']
    print('| model | ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * (len(cols) + 1))
    for name, row in out.items():
        if name == 'guidance_log':
            continue
        cells = []
        for c in cols:
            v = row.get(c)
            if v is None:
                cells.append('')
            elif c == 'Lock_tempo' and 'Lock_tempo_p' in row:
                cells.append(f"{v:+.3f} (p {row['Lock_tempo_p']:.3g}, {row['Lock_tempo_pos']})")
            elif c in ('Loud', 'Perc', 'Perc_slow'):
                cells.append(f"{v:+.3f} ({row[c + '_pos']})")
            elif c.startswith('Lock'):
                cells.append(f'{v:+.3f}')
            else:
                cells.append(f'{v:.3f}' if abs(v) < 10 else f'{v:.2f}')
        print(f'| {name} | ' + ' | '.join(cells) + ' |')
    if 'guidance_log' in out:
        print(json.dumps(out['guidance_log'], indent=1))


if __name__ == '__main__':
    main()
