"""Lock metric = r_own (no chance correction) per model from a lock_models_*.json; paired difference to real dance.
    .venv/bin/python ../Evaluation-Musicality/lock_metric_table.py results/lock_models_finedance_10seed_rescaled_all.json
Per unit (song/piece): own r averaged over its slices and seeds. CI: bootstrap over units. Difference to real dance:
two-sided sign-flip permutation (10,000, seed 0) of the paired per-unit difference; q = BH over the models."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from multicomponent import SEED, boot_ci, signflip_p, bh
d = json.load(open(sys.argv[1])); P = d['per_unit']
units = list(P['Real dance']['own']); own = {k: np.array([v['own'][u] for u in units]) for k, v in P.items()}
rng = np.random.default_rng(SEED); out = {}
for k, v in own.items():
    out[k] = {'r': float(v.mean()), 'ci': boot_ci(v, rng)}
    if k != 'Real dance':
        g = own['Real dance'] - v
        out[k].update({'diff_real': float(g.mean()), 'p': signflip_p(g, rng), 'n_real_higher': int((g > 0).sum())})
ml = [k for k in out if k != 'Real dance']
for k, q in zip(ml, bh([out[k]['p'] for k in ml])): out[k]['q'] = q
print(f"{'model':16s} {'Lock r':>7s} {'CI95':>15s} {'real - model':>12s} {'p':>8s} {'q':>8s} {'real higher':>11s}")
for k, v in out.items():
    if k == 'Real dance': print(f"{k:16s} {v['r']:7.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]"); continue
    print(f"{k:16s} {v['r']:7.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] {v['diff_real']:+12.3f} {v['p']:8.4f} {v['q']:8.4f} {v['n_real_higher']:>8d}/{len(units)}")
json.dump(out, open(sys.argv[1].replace('.json', '_metric.json'), 'w'), indent=1)
