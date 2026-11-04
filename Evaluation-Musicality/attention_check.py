"""Attention check (2026-11-03): does the dance decoder's cross-attention to the music spread thinner when one-shot
MACE generates 34 s instead of its 8 s training length?

Hypothesis (dilution): the cross-attention has no explicit position information, so with 1020 music frames instead
of 240 each dance frame's attention spreads over more of the song, and the dance follows an average of the music
instead of the music at that moment. Measured during DDIM sampling of the FineDance test slices (one seed), per
dance layer, averaged over heads, query frames, slices and the 50 steps:
  eff_s   : effective number of attended music frames, exp(entropy), in seconds
  norm_H  : entropy / log(T)  (1 = uniform over the whole clip)
  local1s : attention mass within +-1 s of the query frame (uniform: about 61 / T)
  dist_s  : attention-weighted mean |query - key| distance in seconds
L = 240 (training length) vs 1020 (test length). Also, for 1020, the same statistics for the queries in the first
240 frames only, so the comparison with L = 240 is on the same music and position range.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/attention_check.py                          # released model
    .venv/bin/python ../Evaluation-Musicality/attention_check.py --checkpoint runs/train/lora_l0/weights/train-30.pt \
        --label lora_l0
"""
import argparse
import json
import math
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from mace_sampler import MaceSampler  # noqa: E402
from model import model as MM  # noqa: E402
from model.metric import style2s  # noqa: E402

FPS, BATCH = 30, 12
STATS = ('eff_s', 'norm_H', 'local1s', 'dist_s')


class Recorder:
    """Replaces CrossModalTransformer.forward; accumulates attention statistics per layer."""

    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.reset(None)

    def reset(self, first):
        self.first = first  # None, or number of leading query frames to restrict to
        self.calls = 0
        self.sum = {}  # (layer, step_bucket, stat) -> [sum, count]

    def add(self, layer, step, w):
        """w: (B, H, Tq, Tk) attention weights."""
        T = w.shape[-1]
        if self.first:
            w = w[:, :, :self.first]
        tq = w.shape[2]
        q = torch.arange(tq, device=w.device).view(-1, 1).float()
        k = torch.arange(T, device=w.device).view(1, -1).float()
        dist = (q - k).abs()  # (tq, T)
        H = -(w * (w.clamp_min(1e-12)).log()).sum(-1)  # (B, H, tq)
        vals = {'eff_s': H.exp() / FPS, 'norm_H': H / math.log(T), 'local1s': (w * (dist <= FPS)).sum(-1),
                'dist_s': (w * dist).sum(-1) / FPS}
        bucket = 'all'
        for b in (bucket, 'last10' if step >= 40 else 'first40'):
            for s, v in vals.items():
                acc = self.sum.setdefault((layer, b, s), [0.0, 0])
                acc[0] += float(v.sum())
                acc[1] += v.numel()

    def result(self):
        out = {}
        for (layer, b, s), (tot, n) in self.sum.items():
            out.setdefault(b, {}).setdefault(s, [None] * self.n_layers)[layer] = tot / n
        return out


def patch(model, rec):
    layers = list(model.dancelayerstack)
    for i, layer in enumerate(layers):
        t = layer.transformer

        def forward(x, mem, t=t, i=i):
            q, kv = t.norm_dance(x), t.norm_music(mem)
            a, w = t.cross_attn(q, kv, kv, need_weights=True, average_attn_weights=False)
            rec.add(i, rec.calls // len(layers), w)
            rec.calls += 1
            x = t.dropout_cross(a)
            return t.dropout2(t.linear2(t.dropout1(t.activation(t.linear1(t.norm(x))))))

        t.forward = forward


def uniform(T, first=None):
    """The same statistics for uniform attention over T frames."""
    tq = first or T
    q, k = np.arange(tq)[:, None], np.arange(T)[None]
    d = np.abs(q - k)
    return {'eff_s': T / FPS, 'norm_H': 1.0, 'local1s': float((d <= FPS).mean(1).mean()), 'dist_s': float(d.mean() / FPS)}


def main(checkpoint, label):
    dt = pickle.load(open('data/dataset_backups/test_tensor_dataset.pkl', 'rb')).data
    names = [os.path.splitext(os.path.basename(f))[0] for f in dt['filenames']]
    feats = np.stack([np.load(f'data/test/baseline_feats/{n}.npy') for n in names])
    genres = [style2s[int(g)] for g in dt['genres']]
    sampler = MaceSampler(checkpoint) if checkpoint else MaceSampler()
    net = sampler.model.diffusion.model
    assert isinstance(net, MM.DanceDecoder)
    rec = Recorder(len(net.dancelayerstack))
    patch(net, rec)

    res = {'label': label, 'checkpoint': checkpoint or 'released', 'n_slices': len(names), 'runs': {}}
    for L, first in ((240, None), (1020, None), (1020, 240)):
        rec.reset(first)
        for i in range(0, len(names), BATCH):
            rec.calls = 0
            sampler.sample(feats[i:i + BATCH, :L], genres[i:i + BATCH], seed=0)
        key = f'{L}' + (f'_first{first}' if first else '')
        res['runs'][key] = {'model': rec.result(), 'uniform': uniform(L, first)}
        r = res['runs'][key]['model']['all']
        print(f'{key:12s} ' + '  '.join(f"{s} " + ' '.join(f'{v:.3f}' for v in r[s]) for s in STATS), flush=True)
        print(f'{"":12s} uniform: ' + '  '.join(f'{s} {v:.3f}' for s, v in res['runs'][key]['uniform'].items()),
              flush=True)
    out = os.path.join(HERE, 'results', f'attention_check{"" if label == "released" else "_" + label}.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    print('saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint')
    ap.add_argument('--label', default='released')
    a = ap.parse_args()
    main(a.checkpoint, a.label)
