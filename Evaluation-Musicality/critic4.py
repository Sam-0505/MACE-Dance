"""Critic v5 (2026-11-08): low-capacity, phase-marginalised beat-synchronous critic (the only critic kept).

Why: critics v1-v3 used constant-shift wrong-time negatives, but our PLV ignores a constant offset and real dancers lock at a
song-specific offset, so they failed their gates. v4 built the symmetry in (global beat phase is a nuisance, phase
CONSISTENCY is the signal) but its 73-channel conv encoder was gamed by guidance through acceleration and per-joint patterns.
v5 keeps v4's grid and score with a ~80-parameter motion side. (v1-v4 code removed 2026-11-08.)

  motion   mean root-relative speed of 4 body parts (legs, arms, torso, head), low-passed (sigma 2), each RMS-normalised per
           clip. First-order only, no root, no per-joint channels.
  events   non-negative linear mix (softplus weights) of the 4 part speeds -> E = 4 event channels per frame. Motion only:
           the critic never sees music features except the beat grid.
  grid     the clip's beats (librosa one-hot, feature col 34). Each beat interval is resampled into P = 16 phase bins
           (linear interpolation, SUB = 4 sub-samples per bin): tempo disappears. Differentiable in the motion.
  score    phase histogram h[p, e] = mean over beats of the binned events, normalised per channel to mean 1 (so only the
           distribution over phase counts, not the amount of motion); learned zero-mean template Q[p, e];
           C(d) = sum_{p,e} h[p, e] Q[(p + d) mod P, e] / E;  S = logsumexp_d C(d) - log P.
           A constant phase shift of the grid only rotates h, so S ignores it; drift spreads h and lowers S; a uniform h
           gives S = 0. The music enters only through the beat grid (no MFCC / chroma: they leaked style in v3).
  training REAL FineDance train dance only (seed-0 90/10 song split), 8 s clips. Positives: own grid.
           Negatives, same motion: (1) drifted grid (AR(1) random walk on beat times, stationary std 0.1-0.35 beat, rho
           0-0.9, mean removed); (2) tempo-warped grid (scaled 1 +- 0.05..0.15 about its centre); (3) the grid of another
           song's clip with a beat count within 1 (the metric's tempo-matched null). No constant-shift negatives.
           Loss: pairwise logistic, softplus(-(S_pos - S_neg)), K = 2 + 2 + 2 negatives per clip.
           Checkpoint: best mean held-out pairwise accuracy over the three negative types.
No PLV values, labels or code are used. Released / generated dances are used only in critic_gate4.py.

    cd Expert-Motion
    .venv/bin/python ../Evaluation-Musicality/critic4.py --lite --lr 1e-2 --seed 0 --out ../Evaluation-Musicality/data/critic/critic_v5.pt
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import critic as C  # noqa: E402
from critic import BEAT_COL  # noqa: E402

P = 16  # phase bins per beat
SUB = 4  # sub-samples per bin
WIN = 240
BIG = 10 ** 6


def grid(onehot):
    """(B, T) beat one-hot -> positions (B, M) float (frames, sorted), mask (B, M)."""
    b = onehot > 0.5
    idx = torch.arange(b.shape[1], device=b.device).expand_as(b)
    srt = torch.where(b, idx, torch.full_like(idx, BIG)).sort(1).values
    m = max(int(b.sum(1).max()), 2)
    srt = srt[:, :m]
    mask = srt < BIG
    return srt.float().masked_fill(~mask, 0), mask


def periods(pos, mask):
    ok = mask[:, 1:] & mask[:, :-1]
    d = (pos[:, 1:] - pos[:, :-1]) * ok
    per = d.sum(1) / ok.sum(1).clamp_min(1)
    return torch.where(ok.sum(1) > 0, per, torch.full_like(per, 15.0))


def drift(pos, mask, amp, rho, gen=None):
    """AR(1) random walk on the beat times: stationary std amp (B,) beats, correlation rho (B,), mean removed, intervals kept
    >= 0.3 period."""
    b, m = pos.shape
    per = periods(pos, mask)
    sd = amp * per
    eps = torch.randn(b, m, device=pos.device, generator=gen)
    eta = [eps[:, 0] * sd]
    for k in range(1, m):
        eta.append(rho * eta[-1] + (1 - rho ** 2).sqrt() * sd * eps[:, k])
    eta = torch.stack(eta, 1)
    w = mask.float()
    eta = eta - (eta * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1)
    new = pos + eta
    d = (new[:, 1:] - new[:, :-1]).maximum(0.3 * per[:, None])
    new = torch.cat([new[:, :1], new[:, :1] + d.cumsum(1)], 1)
    return new.masked_fill(~mask, 0), mask


def warp(pos, mask, alpha):
    w = mask.float()
    c = (pos * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1)
    return (c + alpha[:, None] * (pos - c)).masked_fill(~mask, 0), mask


def shift_frac(pos, mask, f):
    """Constant phase shift: every beat moves by f of its own interval (the last one by the previous interval)."""
    d = pos[:, 1:] - pos[:, :-1]
    ok = mask[:, 1:]
    d = torch.where(ok, d, torch.zeros_like(d))
    last = mask.sum(1) - 1
    dd = torch.cat([d, torch.zeros_like(d[:, :1])], 1)
    prev = d.gather(1, (last - 1).clamp_min(0)[:, None])
    dd.scatter_(1, last.clamp_min(0)[:, None], prev)
    return (pos + f * dd).masked_fill(~mask, 0), mask


def interp(x, pos):
    """x (n, T, ...) sampled at fractional frame positions pos (n, M), linear, edge-clamped."""
    n, t = x.shape[:2]
    pos = pos.clamp(0, t - 1)
    i0 = pos.floor().long()
    i1 = (i0 + 1).clamp(max=t - 1)
    w = (pos - i0).reshape(n, -1, *([1] * (x.dim() - 2)))
    ar = torch.arange(n, device=x.device)[:, None]
    return x[ar, i0] * (1 - w) + x[ar, i1] * w


def sync(A, pos, mask):
    """Events A (B, T, E) on the beat grid -> H (B, K, P, E) (K = M - 1 intervals), valid (B, K)."""
    b, t, e = A.shape
    lo, hi = pos[:, :-1], pos[:, 1:]
    valid = mask[:, :-1] & mask[:, 1:] & (lo >= 0) & (hi <= t - 1) & (hi - lo >= 4) & (hi - lo <= 60)
    u = (torch.arange(P * SUB, device=A.device, dtype=A.dtype) + 0.5) / (P * SUB)
    q = lo[..., None] + (hi - lo)[..., None] * u
    k = lo.shape[1]
    H = interp(A, q.reshape(b, -1)).reshape(b, k, P, SUB, e).mean(3)
    return H, valid


PARTS = {'legs': [1, 2, 4, 5, 7, 8, 10, 11], 'arms': [16, 17, 18, 19, 20, 21, 22, 23], 'torso': [3, 6, 9, 13, 14], 'head': [12, 15]}


def motion_feats_lite(joints):
    """(B, T, 24, 3) -> (B, T, 4): mean root-relative speed of each body part (low-passed sigma 2, central differences),
    each part RMS-normalised per clip. First-order only (no acceleration), no root, no per-joint channels: guidance can raise
    the score only by making a body part move faster at a consistent phase of the beat (2026-11-08, after the full critic was
    gamed through acceleration and per-joint patterns)."""
    from model.musicality import lowpass_joints
    jl = lowpass_joints(joints, C.SIGMA)
    rel = jl - jl[:, :, :1]
    pad = torch.cat([rel[:, :1], rel, rel[:, -1:]], 1)
    sp = (((pad[:, 2:] - pad[:, :-2]) / 2).pow(2).sum(-1) + 1e-8).sqrt()
    x = torch.stack([sp[:, :, idx].mean(-1) for idx in PARTS.values()], -1)
    return x / x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)


class LiteEvents(nn.Module):
    """Events = non-negative learned mix of the 4 part speeds (no temporal layers): (B, T, 4) -> (B, T, n_ev)."""
    def __init__(self, n_ev=4):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_ev, len(PARTS)) * 0.1)

    def forward(self, x):
        return x @ F.softplus(self.w).T


class Critic4(nn.Module):
    beat_sync = True

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.events, self.feats = LiteEvents(cfg['n_ev']), motion_feats_lite
        self.template = nn.Parameter(torch.randn(P, cfg['n_ev']) * 0.5)

    def ev(self, joints):
        return self.events(self.feats(joints))

    def histogram(self, A, pos, mask):
        H, valid = sync(A, pos, mask)
        w = valid.float()[..., None, None]
        h = (H * w).sum(1) / w.sum(1).clamp_min(1)
        return h / h.mean(1, keepdim=True).clamp_min(1e-4), valid.sum(1)

    def score(self, A, pos, mask):
        """(B,) phase-marginalised score of events A against the grid (pos, mask); 0 for fewer than 3 valid intervals."""
        h, nv = self.histogram(A, pos, mask)
        Q = self.template - self.template.mean(0, keepdim=True)
        Qs = torch.stack([Q.roll(-d, 0) for d in range(P)])
        c = torch.einsum('bpe,dpe->bd', h, Qs) / Q.shape[1]
        s = c.logsumexp(1) - math.log(P)
        return torch.where(nv >= 3, s, torch.zeros_like(s))

    def windowed(self, joints, feats, null_feats, win=WIN, stride=120):
        """Guidance score: own grid - mean over the null clips' grids, averaged over windows. joints (B, T, 24, 3)
        (differentiable), feats (B, T, 35), null_feats list of (B, T, 35). Returns (B,)."""
        t = joints.shape[1]
        tot, n = 0, 0
        for s in range(0, max(t - win, 0) + 1, stride):
            A = self.ev(joints[:, s:s + win])
            own = self.score(A, *grid(feats[:, s:s + win, BEAT_COL]))
            nul = sum(self.score(A, *grid(nf[:, s:s + win, BEAT_COL])) for nf in null_feats) / len(null_feats)
            tot = tot + own - nul
            n += 1
        return tot / n


class Ensemble(nn.Module):
    """Mean guidance score of several critics (load_critic with comma-separated checkpoints). A pattern that fools one
    network rarely fools the others; a critic left out of the ensemble checks whether guidance still games them."""
    beat_sync = True

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def windowed(self, joints, feats, null_feats, win=WIN, stride=120):
        return sum(m.windowed(joints, feats, null_feats, win, stride) for m in self.members) / len(self.members)


def negatives(pos, mask, song, nb, kd, kw, ko, rng, gen=None):
    """List of (type, pos, mask) negative grids for a batch."""
    dev = pos.device
    n = pos.shape[0]
    out = []
    for _ in range(kd):
        amp = torch.as_tensor(rng.uniform(0.1, 0.35, n), device=dev, dtype=torch.float32)
        rho = torch.as_tensor(rng.uniform(0.0, 0.9, n), device=dev, dtype=torch.float32)
        out.append(('drift',) + drift(pos, mask, amp, rho, gen))
    for _ in range(kw):
        al = 1 + rng.choice([-1, 1], n) * rng.uniform(0.05, 0.15, n)
        out.append(('warp',) + warp(pos, mask, torch.as_tensor(al, device=dev, dtype=torch.float32)))
    valid = C.neg_mask(song, nb).cpu().numpy()
    for _ in range(ko):
        pick = np.array([rng.choice(np.flatnonzero(valid[i])) if valid[i].any() else -1 for i in range(n)])
        j = torch.as_tensor(np.where(pick >= 0, pick, np.arange(n)), device=dev)
        p2, m2 = pos[j], mask[j]
        has = torch.as_tensor(pick >= 0, device=dev)
        dp, dm = drift(pos, mask, torch.full((n,), 0.25, device=dev), torch.full((n,), 0.5, device=dev), gen)  # fallback
        out.append(('other', torch.where(has[:, None], p2, dp), torch.where(has[:, None], m2, dm)))
    return out


def train(args):
    from vis import SMPLSkeleton
    from model.musicality import pose_to_joints
    dev = 'cuda'
    pose, feats, song, normalizer = C.load_train(args.data)
    smpl = SMPLSkeleton(dev)
    n_songs = song.max() + 1
    rng = np.random.default_rng(0)  # same split as critics v1-v3
    val_songs = set(rng.choice(n_songs, max(1, n_songs // 10), replace=False).tolist())
    is_val = np.array([s in val_songs for s in song])
    nbeat = (feats[:, :, BEAT_COL] > 0.5).sum(1).numpy()
    keep = nbeat >= 4
    tr, va = np.flatnonzero(~is_val & keep), np.flatnonzero(is_val & keep)
    cfg = {'n_ev': args.n_ev, 'lite': True}
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    print(f'cfg {cfg}: {len(tr)} train clips, {len(va)} val clips of {len(val_songs)} held-out songs', flush=True)
    model = Critic4(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(tr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    types = ('drift', 'warp', 'other')

    def batch(idx, r, gen=None):
        with torch.no_grad():
            j = pose_to_joints(pose[idx].to(dev), normalizer, smpl)
        f = feats[idx].to(dev)
        pos, mask = grid(f[:, :, BEAT_COL])
        A = model.ev(j)
        sp = model.score(A, pos, mask)
        negs = negatives(pos, mask, song[idx], mask.sum(1), args.kd, args.kw, args.ko, r, gen)
        sn = torch.stack([model.score(A, p2, m2) for _, p2, m2 in negs], 1)
        loss = F.softplus(-(sp[:, None] - sn)).mean()
        return loss, sp, sn, [t for t, _, _ in negs]

    best = (-1.0, 0)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(tr)
        tot = 0.0
        nb_ = len(tr) // args.batch
        for k in range(nb_):
            loss = batch(perm[k * args.batch:(k + 1) * args.batch], rng)[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        model.eval()
        vr = np.random.default_rng(123)
        gen = torch.Generator(device=dev).manual_seed(123)
        acc = {t: [] for t in types}
        vl = []
        with torch.no_grad():
            for k in range(0, len(va), args.batch):
                loss, sp, sn, tl = batch(va[k:k + args.batch], vr, gen)
                vl.append(float(loss))
                for q, t in enumerate(tl):
                    acc[t].append((sp > sn[:, q]).float().cpu().numpy())
        acc = {t: float(np.concatenate(v).mean()) for t, v in acc.items()}
        ma = float(np.mean(list(acc.values())))
        if ma > best[0]:
            best = (ma, ep)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({'model': model.state_dict(), 'kind': 'd', 'cfg': cfg, 'epoch': ep, 'val_acc': acc,
                        'val_songs': sorted(val_songs)}, args.out)
        print(f'epoch {ep} train loss {tot / nb_:.4f} | val loss {np.mean(vl):.4f} pairwise acc drift {acc["drift"]:.3f} '
              f'warp {acc["warp"]:.3f} other-song {acc["other"]:.3f} (mean {ma:.3f})', flush=True)
    print(f'best held-out mean pairwise accuracy {best[0]:.3f} at epoch {best[1]}; saved {args.out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_ev', type=int, default=4)
    ap.add_argument('--kd', type=int, default=2)
    ap.add_argument('--kw', type=int, default=2)
    ap.add_argument('--ko', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-2)  # v5 was trained with 1e-2
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lite', action='store_true', help='accepted for old run scripts; v5 is always lite')
    ap.add_argument('--data', default='finedance', choices=['finedance', 'aist'])
    ap.add_argument('--out', required=True)
    train(ap.parse_args())


if __name__ == '__main__':
    main()
