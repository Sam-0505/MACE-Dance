"""Differentiable offset-free beat locking: the score behind the formula-guidance variant (Evaluation-Musicality/mace_sampler.py).

Evaluation (Evaluation-Musicality/beat_suite.py): accents = local maxima of the Gaussian-smoothed (sigma 5 frames)
mean joint speed; phase of an accent = its position inside its beat interval (0 = on the beat, 0.5 = half way);
PLV = |mean exp(2 pi i phase)|. The PLV credits any constant offset, so it does not ask the dancer to hit exactly on the beat.

Differentiable stand-in (local maxima are not differentiable):
  v_t   smoothed mean joint speed, as in the evaluation
  w_t   relu(v_t - G_wide(v)_t): how far the speed rises above its local mean; largest at accents, 0 in lulls
  phi_t phase of frame t inside its beat interval, from the beat one-hot of the conditioning (column 34; no gradient)
  R     |sum_t w_t exp(2 pi i phi_t)| / sum_t w_t      (frames before the first or after the last beat get no weight)
Chance correction: D = R(own beats) - R(beats of other clips in the batch). An 8 s clip has ~13 beats, so R is ~0.12 even
for motion unrelated to the music, and a motion with a few large speed peaks has a high R against ANY beat grid.

The training-time uses of this score (hinge and contrastive locking losses) were removed 2026-11-10: they raised the
score of the predicted x0 but not the locking of sampled dances (see NEGATIVE_RESULTS.md).
"""
import math

import torch

from model.musicality import gaussian_smooth

BEAT_COL = 34  # baseline features: librosa beat_track one-hot (0 = onset envelope, 33 = onset peaks)
SPEED_SIGMA = 5  # frames, as the evaluation (tmd_beat_metrics.SMOOTH)
W_MIN = 1e-6  # total accent weight (m/frame summed over frames) below which R is undefined and set to 0


def speed(joints, sigma=SPEED_SIGMA):
    """(B, T, J, 3) -> (B, T-1) smoothed mean joint speed. eps keeps the gradient of the norm finite at rest."""
    d = joints[:, 1:] - joints[:, :-1]
    return gaussian_smooth(((d ** 2).sum(-1) + 1e-12).sqrt().mean(-1), sigma)


def beat_phase(beats):
    """(B, T) beat one-hot -> phase (B, T) in [0, 1) and valid mask (B, T): frame lies between two beats."""
    b, t = beats.shape
    on = beats > 0.5
    idx = torch.arange(t, device=beats.device).expand(b, t)
    prev = torch.where(on, idx, torch.full_like(idx, -1)).cummax(1).values
    big = torch.full_like(idx, 10 ** 9)
    nxt = torch.where(on, idx, big).flip(1).cummin(1).values.flip(1)
    nxt = torch.cat([nxt[:, 1:], big[:, :1]], 1)  # strictly after t: a frame on a beat has phase 0
    ok = (prev >= 0) & (nxt < big)
    phase = (idx - prev).float() / (nxt - prev).clamp_min(1).float()
    return torch.where(ok, phase, torch.zeros_like(phase)), ok


def locking(v, beats, wide_sigma):
    """v (B, T-1) speed, beats (B, T) one-hot -> R (B,) and the number of beats (B,).

    Fix 2026-11-05: a clip with < 2 beats (silence) gives no frame a phase, so the total weight was 0 and R was
    1e-6 / 1e-12 = 1e6, which wrecked the first pilot. R is now 0 there, and callers drop clips with few beats."""
    w = torch.relu(v - gaussian_smooth(v, wide_sigma))
    phase, ok = beat_phase(beats[:, :v.shape[1]])
    w = w * ok
    ang = 2 * math.pi * phase
    c, s = (w * ang.cos()).sum(1), (w * ang.sin()).sum(1)
    tot = w.sum(1)
    r = (c ** 2 + s ** 2 + 1e-20).sqrt() / tot.clamp_min(W_MIN)
    return torch.where(tot > W_MIN, r, torch.zeros_like(r)), (beats > 0.5).sum(1)


def windowed_locking(joints, beats, wide_sigma, win=240, n_null=3, null_beats=None):
    """Sampling-time guidance score (2026-11-05): chance-corrected locking of (B, T, J, 3) joints, averaged over
    half-overlapping windows of win frames, each with its own offset (real dancers' offset drifts slowly).
    The null is the beats of n_null other clips of the batch (rolled by B * k / (n_null + 1)), so with a batch
    sorted by song they come from other songs. Windows with < 4 own or null beats are left out. Returns (B,).
    null_beats: optional list of (B, T) beat one-hots used as the null instead of the rolled batch (for batches that
    are not sorted by song, e.g. Motorica, where a take spans up to 16 slices)."""
    v = speed(joints)
    b, t1 = v.shape
    starts = list(range(0, max(t1 - win, 0) + 1, win // 2))
    shifts = [max(1, b * k // (n_null + 1)) for k in range(1, n_null + 1)]
    tot = torch.zeros(b, device=v.device)
    cnt = torch.zeros(b, device=v.device)
    for s in starts:
        vw, bw = v[:, s:s + win], beats[:, s:s + win + 1]
        r, n = locking(vw, bw, wide_sigma)
        ok = n >= 4
        null = 0
        nulls = [bw.roll(sh, 0) for sh in shifts] if null_beats is None else [nb[:, s:s + win + 1] for nb in null_beats]
        for nb in nulls:
            ro, no = locking(vw, nb, wide_sigma)
            null = null + ro / len(nulls)
            ok = ok & (no >= 4)
        tot = tot + torch.where(ok, r - null, torch.zeros_like(r))
        cnt = cnt + ok.float()
    return tot / cnt.clamp_min(1)
