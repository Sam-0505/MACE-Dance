"""Beat-locking loss (2026-11-05, draft for review): offset-free phase locking of motion accents to the music beats.

Evaluation (Evaluation-Musicality/beat_suite.py): accents = local maxima of the Gaussian-smoothed (sigma 5 frames)
mean joint speed; phase of an accent = its position inside its beat interval (0 = on the beat, 0.5 = half way);
PLV = |mean exp(2 pi i phase)|. Real dance is above chance on FineDance, Motorica and AIST++; every MACE variant is at
chance. The PLV credits any constant offset, so it does not ask the dancer to hit exactly on the beat.

Differentiable stand-in (local maxima are not differentiable):
  v_t   smoothed mean joint speed, as in the evaluation
  w_t   relu(v_t - G_wide(v)_t): how far the speed rises above its local mean; largest at accents, 0 in lulls
  phi_t phase of frame t inside its beat interval, from the beat one-hot of the conditioning (column 34; no gradient)
  R     |sum_t w_t exp(2 pi i phi_t)| / sum_t w_t      (frames before the first or after the last beat get no weight)
Chance correction (null = True): D = R(own beats) - R(beats of another clip in the batch, batch rolled by one). An 8 s
clip has ~13 beats, so R is ~0.12 even for motion unrelated to the music; and a motion with a few large speed peaks has
a high R against ANY beat grid. D removes both: only locking to this clip's beats raises it.
Loss (hinge vs real, as change B of the slow-coupling loss): max(0, D_real - D_pred), with D_real computed the same way
on the ground-truth motion of the same clip, so the model is pulled up to real dance's locking, not past it.
Option (diffusion.lock_batch): max(0, mean D_real - mean D_pred) over the batch. D_real of one clip is noisy, so the
per-clip hinge stays positive even when the model matches real dance on average and keeps pushing past it.
Scale: at lambda = 1 the gradient norm of this term is ~200x that of the four EDGE losses (released checkpoint, batch 32),
so lambda ~ 1e-3 to 1e-2 puts it on the same footing.
Check on FineDance test, 8 s windows (Evaluation-Musicality/beatlock_validate.py): soft R own - other is +0.048 for
real dance (p 0.02) and +0.008 for MACE (n.s.); real > MACE in 17/19 songs (p < 0.001); r = 0.50 with the hard PLV.
"""
import math

import torch

from model.musicality import gaussian_smooth, pose_to_joints

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


def locking_terms(pred, target, cond, normalizer, smpl, wide_sigma, null=True):
    """Per-sample locking D (or R if null is False) of the predicted x0 and of the real motion (no gradient), plus
    the number of beats of the clip (with null: the smaller of the own and the other clip's count); all (B,)."""
    beats = cond[:, :, BEAT_COL]
    other = beats.roll(1, 0)

    def score(v):
        r, n = locking(v, beats, wide_sigma)
        if not null:
            return r, n
        r_o, n_o = locking(v, other, wide_sigma)
        return r - r_o, torch.minimum(n, n_o)

    d_pred, n = score(speed(pose_to_joints(pred, normalizer, smpl)))
    with torch.no_grad():
        d_real, _ = score(speed(pose_to_joints(target, normalizer, smpl)))
    return d_pred, d_real, n
