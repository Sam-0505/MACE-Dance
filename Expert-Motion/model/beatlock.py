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

from model.musicality import gaussian_smooth, lowpass_joints, pose_to_joints

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


def locking_terms(pred, target, cond, normalizer, smpl, wide_sigma, null=True, joint_sigma=0):
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

    def joints(x):
        return lowpass_joints(pose_to_joints(x, normalizer, smpl), joint_sigma)

    d_pred, n = score(speed(joints(pred)))
    with torch.no_grad():
        d_real, _ = score(speed(joints(target)))
    return d_pred, d_real, n


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


def contrastive_locking(pred, cond, songs, normalizer, smpl, wide_sigma, joint_sigma=0, n_neg=7, tau=0.05, min_beats=4):
    """Contrastive beat locking (2026-11-07): InfoNCE of own beats against the beats of other songs.

    For each clip i of the batch, R_i0 = locking of the predicted x0 against the clip's own beats and R_ik, k = 1..n_neg,
    its locking against the beats of other clips of the batch. The negatives are clips of OTHER songs (songs: one label
    per clip) with the closest beat count (same length of clip, so closest tempo): a tempo score cannot tell them apart,
    only the phase of the beat grid can. loss_i = -log softmax(R_i / tau)[0]. R is the offset-free locking (|resultant|
    of the accents' beat phases), so a dancer's own offset is not penalised, and there is no hinge against real dance:
    the term only asks that the dance fits its own song better than other songs.
    Returns per-clip loss (B,), the clip's own R and mean negative R (B,), and keep (B,): clip has >= min_beats beats
    and at least one valid negative. Clips with fewer valid negatives than n_neg mask the missing ones (-inf logit)."""
    beats = cond[:, :, BEAT_COL]
    joints = lowpass_joints(pose_to_joints(pred, normalizer, smpl), joint_sigma)
    v = speed(joints)
    r_own, n_own = locking(v, beats, wide_sigma)
    nb = (beats > 0.5).sum(1).float()
    b = beats.shape[0]
    same = torch.as_tensor(songs, device=beats.device).view(-1, 1) == torch.as_tensor(songs, device=beats.device).view(1, -1)
    bad = same | (nb < min_beats).view(1, -1)
    dist = (nb.view(-1, 1) - nb.view(1, -1)).abs() + 1e6 * bad.float()
    k = min(n_neg, b - 1)
    d_k, idx = dist.topk(k, dim=1, largest=False)
    valid = d_k < 1e5
    r_neg = torch.stack([locking(v, beats[idx[:, j]], wide_sigma)[0] for j in range(k)], 1)
    logits = torch.cat([r_own.unsqueeze(1), r_neg], 1) / tau
    logits = torch.cat([logits[:, :1], logits[:, 1:].masked_fill(~valid, float('-inf'))], 1)
    loss = -torch.log_softmax(logits, 1)[:, 0]
    keep = (n_own >= min_beats) & valid.any(1) & loss.isfinite()
    r_neg_mean = (r_neg * valid).sum(1) / valid.sum(1).clamp_min(1)
    return loss, r_own, r_neg_mean, keep
