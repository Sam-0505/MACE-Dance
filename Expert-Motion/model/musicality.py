"""Differentiable signal helpers shared by the sampler, the critic and the evaluation scripts.

Definitions match Evaluation-Musicality: body energy = mean squared joint speed in world coordinates, edge-padded
(metrics.motion_energy); smoothing = scipy gaussian_filter1d (truncate 4, mode 'reflect').
(The training-time musicality loss that used to live here was removed 2026-11-10; see NEGATIVE_RESULTS.md.)
"""
import torch

from dataset.quaternion import ax_from_6v


def gaussian_smooth(x, sigma):
    """(B, T) -> (B, T), equal to scipy.ndimage.gaussian_filter1d(x, sigma, axis=-1)."""
    r = int(4.0 * sigma + 0.5)
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    # scipy 'reflect' repeats the edge sample (d c b a | a b c d), unlike torch's 'reflect'
    xp = torch.cat([x[:, :r].flip(1), x, x[:, -r:].flip(1)], dim=1)
    return torch.nn.functional.conv1d(xp[:, None], k.view(1, 1, -1))[:, 0]


def motion_energy(joints):
    """(B, T, J, 3) -> (B, T): mean over joints of the squared frame-to-frame displacement, last frame repeated."""
    v2 = ((joints[:, 1:] - joints[:, :-1]) ** 2).sum(-1).mean(-1)
    return torch.cat([v2, v2[:, -1:]], dim=1)


def pose_to_joints(x, normalizer, smpl):
    """Normalized model output (B, T, 151) -> world joints (B, T, 24, 3), differentiably.

    Same as diffusion.render_sample, but out of place and without the [-1, 1] clip of Normalizer.unnormalize,
    which would zero the gradient outside that range.
    """
    sc = normalizer.scaler
    x = (x - sc.min_.to(x.device)) / sc.scale_.to(x.device)
    x = x[:, :, 4:]
    b, t, _ = x.shape
    q = ax_from_6v(x[:, :, 3:].reshape(b, t, 24, 6))
    return smpl.forward(q, x[:, :, :3])


def lowpass_joints(joints, sigma):
    """(B, T, J, 3) -> same, Gaussian-smoothed over time (sigma in frames; 0 = unchanged).

    Change A (2026-11-03): energy from low-passed joints, so fast jitter cannot raise body energy. Squared speed
    weights a frequency f by f^2, which made switching jitter on and off with the music the cheapest way to raise r.
    """
    if not sigma:
        return joints
    b, t, j, c = joints.shape
    x = joints.permute(0, 2, 3, 1).reshape(b * j * c, t)
    return gaussian_smooth(x, sigma).reshape(b, j, c, t).permute(0, 3, 1, 2)
