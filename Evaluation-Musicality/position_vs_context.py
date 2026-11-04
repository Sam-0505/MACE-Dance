"""Position vs attention interference (2026-11-04): why does one-shot coupling fall later in a full-song generation?

H1 position: frames far into the generated sequence are out of distribution (MACE has no positional encoding;
   position can only enter through the BiMamba scans).
H2 interference: cross-attention to distant, unrelated music frames dilutes the local music-motion link.

The same 34 s excerpt E of the same FineDance test song is generated in five ways and scored only on E's frames
(zero-lag own-song Delta, slow treble flux vs slow body energy, sigma 15, other songs' same excerpt as the wrong
music; 10 samples per song). "Other" is the rest of the song's music up to E (or, for the first excerpt, the next
34 s), so the late and early versions contain exactly the same music.
  A      E alone                                  early, only E's music            (reference)
  B_blk  E + Other, E's frames attend to E only   early, only E's music            (BiMamba sees Other)
  B      E + Other, full attention                early, all music
  C      Other + E, E's frames attend to E only   late,  only E's music
  D      Other + E, full attention                late,  all music  (= the full-song generation up to E)
  A_loc  E alone, +-2 s local cross-attention    early, music within 2 s          (follow-up)
  D_loc  Other + E, +-2 s local cross-attention   late,  music within 2 s          (follow-up)
H1 effect = C - B_blk (blocked attention) and D - B (full attention); H2 effect = B - B_blk (early), D - C (late).
Follow-up: blocking music beyond the excerpt (C) leaves the music within the excerpt; A_loc - A asks whether
attention to music 2-34 s away inside the excerpt is what weakens the coupling.
Block masks: each query frame attends only to music frames of its own block (E or Other).
Excerpts: k = 0 (0-34 s; A, B_blk, B only), 1 (34-68 s), 2 (68-102 s; real dance shows ~0 coupling there).

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/position_vs_context.py --label lora_l0 \
        --checkpoint runs/train/lora_l0/weights/train-30.pt
    .venv/bin/python ../Evaluation-Musicality/position_vs_context.py --label released
    .venv/bin/python ../Evaluation-Musicality/position_vs_context.py --report
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from model import musicality as MU  # noqa: E402

E_LEN, SEEDS, CHUNK, SIGMA = 1020, 10, 5, 15
CONDS = ('A', 'B_blk', 'B', 'C', 'D', 'A_loc', 'D_loc')
LOCAL = 60  # +-2 s
INPUTS = os.path.join(HERE, 'data', 'full_song', 'inputs.npz')  # full_song_curve.prepare()
CACHE = os.path.join(HERE, 'data', 'position_vs_context')
OUT = os.path.join(HERE, 'results', 'position_vs_context.json')


def block_attention(net, holder):
    """Cross-attention with an optional mask: holder['mask'] (Tq, Tk) bool, True = blocked; None = full."""
    for layer in net.dancelayerstack:
        t = layer.transformer

        def forward(x, mem, t=t):
            kv = t.norm_music(mem)
            x = t.dropout_cross(t.cross_attn(t.norm_dance(x), kv, kv, attn_mask=holder['mask'], need_weights=False)[0])
            return t.dropout2(t.linear2(t.dropout1(t.activation(t.linear1(t.norm(x))))))

        t.forward = forward


def block_mask(lengths, device):
    b = torch.cat([torch.full((n,), i, device=device) for i, n in enumerate(lengths)])
    return b[:, None] != b[None]


def layout(T, k):
    """Frame ranges (start, end) of E and Other in the song, or None if the song is too short."""
    e = (k * E_LEN, (k + 1) * E_LEN)
    if e[1] > T:
        return None
    if k == 0:
        o = (E_LEN, 2 * E_LEN)
        return (e, o) if o[1] <= T else None
    return e, (0, e[0])


def conditions(k):
    return CONDS[:3] + ('A_loc',) if k == 0 else CONDS


def build(f, e, o, cond):
    """-> (features, block lengths or None, slice of E in the generated sequence)."""
    E, O = f[e[0]:e[1]], f[o[0]:o[1]]
    if cond.endswith('_loc'):
        x, _, sl = build(f, e, o, cond[0])
        return x, 'local', sl
    if cond == 'A':
        return E, None, slice(0, E_LEN)
    if cond in ('B', 'B_blk'):
        return np.concatenate([E, O]), ([E_LEN, len(O)] if cond == 'B_blk' else None), slice(0, E_LEN)
    return np.concatenate([O, E]), ([len(O), E_LEN] if cond == 'C' else None), slice(len(O), len(O) + E_LEN)


def generate(label, checkpoint, inp):
    from mace_sampler import MaceSampler
    sampler = MaceSampler(checkpoint) if checkpoint else MaceSampler()
    holder = {'mask': None}
    block_attention(sampler.model.diffusion.model, holder)
    d = os.path.join(CACHE, label)
    os.makedirs(d, exist_ok=True)
    for k in (0, 1, 2):
        for s, f, g in zip(inp['songs'], inp['feats'], inp['genres']):
            lay = layout(len(f), k)
            if lay is None:
                continue
            for cond in conditions(k):
                path = os.path.join(d, f'{s}_k{k}_{cond}.npy')
                if os.path.exists(path):
                    continue
                x, blocks, sl = build(f, *lay, cond)
                if blocks == 'local':
                    q = torch.arange(len(x), device=sampler.device)
                    holder['mask'] = (q[:, None] - q[None]).abs() > LOCAL
                else:
                    holder['mask'] = block_mask(blocks, sampler.device) if blocks else None
                e = []
                for c in range(SEEDS // CHUNK):
                    j = sampler.sample(np.repeat(x[None], CHUNK, 0), [g] * CHUNK, seed=c)
                    e.append(MU.motion_energy(torch.from_numpy(j)).numpy()[:, sl])
                np.save(path, np.concatenate(e).astype(np.float32))
            print(label, 'k', k, s, flush=True)


def z(a):
    a = MU.gaussian_smooth(torch.as_tensor(np.stack(a), dtype=torch.float64), SIGMA)
    a = a - a.mean(1, keepdim=True)
    return a / a.norm(dim=1, keepdim=True)


def deltas(energy, music):
    """energy, music: lists (one per song) of (E_LEN,). Own r minus mean r with the other songs' music."""
    R = (z(music) @ z(energy).T).numpy()
    n = len(R)
    return np.diag(R) - (R.sum(0) - np.diag(R)) / (n - 1)


def signflip(x, rng):
    s = rng.choice([-1, 1], (20000, len(x)))
    return float((np.abs((s * x).mean(1)) >= abs(x.mean())).mean())


def evaluate(label, inp):
    rng = np.random.default_rng(0)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res[label] = {}
    for k in (0, 1, 2):
        idx = [i for i, f in enumerate(inp['feats']) if layout(len(f), k)]
        songs = [inp['songs'][i] for i in idx]
        music = [inp['treble'][i][k * E_LEN:(k + 1) * E_LEN] for i in idx]
        real = deltas([inp['real_energy'][i][k * E_LEN:(k + 1) * E_LEN] for i in idx], music)
        per = {}
        for cond in conditions(k):
            E = [np.load(os.path.join(CACHE, label, f'{s}_k{k}_{cond}.npy')) for s in songs]
            per[cond] = np.mean([deltas([e[j] for e in E], music) for j in range(SEEDS)], 0)
        r = {'n_songs': len(songs), 'songs': songs, 'real': float(real.mean()), 'real_per_song': real.tolist(),
             'conds': {c: {'mean': float(v.mean()), 'per_song': v.tolist()} for c, v in per.items()}, 'effects': {}}
        eff = {'H2 early (B - B_blk)': ('B', 'B_blk'), 'A vs B_blk (BiMamba context)': ('B_blk', 'A'),
               'local, alone (A_loc - A)': ('A_loc', 'A')}
        if k > 0:
            eff.update({'H1 blocked (C - B_blk)': ('C', 'B_blk'), 'H1 full (D - B)': ('D', 'B'),
                        'H2 late (D - C)': ('D', 'C'), 'alone vs full song (A - D)': ('A', 'D'),
                        'local, late (D_loc - D)': ('D_loc', 'D'), 'local late vs alone (D_loc - A_loc)': ('D_loc', 'A_loc')})
        for name, (a, b) in eff.items():
            x = per[a] - per[b]
            bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
            r['effects'][name] = {'mean': float(x.mean()), 'ci': [float(np.percentile(bs, 2.5)),
                                                                  float(np.percentile(bs, 97.5))], 'p': signflip(x, rng)}
        res[label][f'k{k}'] = r
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    report()


def report():
    res = json.load(open(OUT))
    for label, ks in res.items():
        print(f'\n== {label}')
        for k, r in ks.items():
            print(f"  excerpt {k} ({r['n_songs']} songs)  real {r['real']:+.3f}  " +
                  '  '.join(f"{c} {v['mean']:+.3f}" for c, v in r['conds'].items()))
            for name, e in r['effects'].items():
                print(f"      {name:30s} {e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}] p {e['p']:.3f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--label')
    ap.add_argument('--checkpoint')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report:
        report()
    else:
        inp = dict(np.load(INPUTS, allow_pickle=True))
        generate(a.label, a.checkpoint, inp)
        evaluate(a.label, inp)
