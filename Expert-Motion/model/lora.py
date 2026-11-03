"""LoRA adapters for the Motion Expert (2026-11-03): fine-tune low-rank updates W + (alpha / r) B A and keep every
released weight frozen, so the model keeps what it learned on the 8 s windows.

Added through torch parametrization rather than by wrapping the layers: Mamba's fused kernel (mamba_inner_fn) reads
in_proj.weight / out_proj.weight directly instead of calling the Linear, and nn.MultiheadAttention reads
in_proj_weight. A parametrized weight is recomputed on every access, so both see the adapted weight.

Targets, in all 8 dance and 2 music layers: both Mamba directions' in_proj and out_proj; in the dance layers also the
cross-attention in/out projections and the feed-forward linear1/linear2. Embeddings, FiLM, genre MLP, norms, the
input/cond projections and the final layer stay frozen. B starts at zero, so training starts at the released model.

Checkpoints store plain merged weights (plain_state_dict) under the usual keys, so sampling/evaluation code loads a
LoRA fine-tune like any other checkpoint; the adapters alone are saved too, as lora_state_dict.
"""
import math

import torch
from torch import nn
from torch.nn.utils import parametrize


class LoRA(nn.Module):
    def __init__(self, out_features, in_features, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, W):
        return W + self.scale * (self.B @ self.A).to(W.dtype)


def targets(model):
    """(module, weight name) pairs that get an adapter."""
    out = []
    for layer in list(model.dancelayerstack) + list(model.musiclayerstack):
        for m in (layer.bimamba.mamba_forward, layer.bimamba.mamba_backward):
            out += [(m.in_proj, 'weight'), (m.out_proj, 'weight')]
        if hasattr(layer, 'transformer'):
            t = layer.transformer
            out += [(t.cross_attn, 'in_proj_weight'), (t.cross_attn.out_proj, 'weight'),
                    (t.linear1, 'weight'), (t.linear2, 'weight')]
    return out


def add_lora(model, rank, alpha=None, seed=0):
    """Freeze model, add adapters; returns the number of trainable parameters. Same seed -> same initial A."""
    alpha = alpha or rank
    for p in model.parameters():
        p.requires_grad_(False)
    g = torch.random.fork_rng(devices=[])
    with g:
        torch.manual_seed(seed)
        for m, name in targets(model):
            W = getattr(m, name)
            lora = LoRA(W.shape[0], W.shape[1], rank, alpha).to(W.device)
            parametrize.register_parametrization(m, name, lora)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plain_state_dict(model):
    """state_dict with each adapted weight merged (W + scale B A) under its original key; unchanged without LoRA."""
    sd = {k: v for k, v in model.state_dict().items() if '.parametrizations.' not in k}
    for name, m in model.named_modules():
        if parametrize.is_parametrized(m):
            for p in m.parametrizations:
                sd[f'{name}.{p}' if name else p] = getattr(m, p).detach().clone()
    return sd


def lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if '.parametrizations.' in k and not k.endswith('.original')}
