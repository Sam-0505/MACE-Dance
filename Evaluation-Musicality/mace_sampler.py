"""Thin wrapper around the Motion Expert for generating SMPL joints from 35-dim baseline features.
Import from inside Expert-Motion (its modules and checkpoint paths are relative to it)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from EDGE import EDGE
from model.metric import style2s
from dataset.quaternion import ax_from_6v


class MaceSampler:
    def __init__(self, checkpoint='./runs/train/exp/weights/train-3750.pt'):
        self.model = EDGE('baseline', checkpoint_path=checkpoint)
        self.model.eval()
        self.device = self.model.accelerator.device

    @torch.no_grad()
    def sample(self, feats, genres, seed):
        """feats: (B, T, 35) raw baseline features; genres: list of style names. Returns (B, T, 24, 3) joints.
        The same seed gives the same noise for the same batch shape, so runs differing only in
        conditioning are paired."""
        cond = torch.as_tensor(np.asarray(feats), dtype=torch.float32, device=self.device)
        genre = torch.tensor([style2s.index(g) for g in genres], dtype=torch.long, device=self.device)
        torch.manual_seed(seed)
        d = self.model.diffusion
        x = d.ddim_sample((cond.shape[0], cond.shape[1], self.model.repr_dim), cond, genre)
        return self.to_joints(x)

    @torch.no_grad()
    def sample_stitched(self, feats, genres, seed, window=240, stride=130):
        """Long-form sampling as in EDGE's long_ddim_sample: the clip is cut into overlapping training-length
        windows, all denoised together, and after every DDIM step each window's first window - stride frames are
        set to the previous window's last ones. Windows keep absolute root positions (as in training), so the
        overlaps are identical and the clip is the first window plus the non-overlapping tail of each later one.
        (T - window) must be a multiple of stride; 1020 frames -> 7 windows of 240 overlapping by 110 (3.7 s)."""
        cond = torch.as_tensor(np.asarray(feats), dtype=torch.float32, device=self.device)
        b, t = cond.shape[:2]
        assert (t - window) % stride == 0, (t, window, stride)
        n, ov = (t - window) // stride + 1, window - stride
        cond = torch.stack([cond[:, k * stride:k * stride + window] for k in range(n)], 1).flatten(0, 1)
        genre = torch.tensor([style2s.index(g) for g in genres], dtype=torch.long, device=self.device)
        genre = genre.repeat_interleave(n)
        torch.manual_seed(seed)
        d = self.model.diffusion
        times = list(reversed(torch.linspace(-1, d.n_timestep - 1, steps=51).int().tolist()))
        x = torch.randn((b * n, window, self.model.repr_dim), device=self.device)
        for time, time_next in zip(times[:-1], times[1:]):
            tc = torch.full((b * n,), time, device=self.device, dtype=torch.long)
            pred_noise, x_start = d.model_predictions(x, cond, genre, tc, clip_x_start=d.clip_denoised)
            if time_next < 0:
                x = x_start
                continue
            a, an = d.alphas_cumprod[time], d.alphas_cumprod[time_next]
            sigma = ((1 - a / an) * (1 - an) / (1 - a)).sqrt()
            x = x_start * an.sqrt() + (1 - an - sigma ** 2).sqrt() * pred_noise + sigma * torch.randn_like(x)
            x = x.view(b, n, window, -1)
            x[:, 1:, :ov] = x[:, :-1, stride:]
            x = x.flatten(0, 1)
        # the last step is not constrained, so the windows' final predictions of the overlap differ slightly;
        # EDGE's render_sample crossfades joint positions linearly over the overlap, as done here
        j = torch.from_numpy(self.to_joints(x)).view(b, n, window, 24, 3)
        fade = torch.linspace(0, 1, ov + 2)[1:-1].view(1, ov, 1, 1)
        out = j[:, 0]
        for k in range(1, n):
            blend = out[:, -ov:] * (1 - fade) + j[:, k, :ov] * fade
            out = torch.cat([out[:, :-ov], blend, j[:, k, ov:]], 1)
        return out.numpy()

    def sample_guided(self, feats, genres, seed, scale, t_max=600, joint_sigma=2, wide=10, root_relative=True,
                      guide_shift=0, groups=None, n_null=3):
        """DDIM sampling as diffusion.ddim_sample (same noise for the same seed; scale 0 gives identical samples),
        with beat-locking guidance (2026-11-05): at steps with t <= t_max the model's x0 prediction is moved along
        the gradient of the chance-corrected, windowed locking score (model.beatlock.windowed_locking, on joints
        low-passed with joint_sigma frames), normalised per sample to RMS 1, times scale; the noise estimate is then
        recomputed from the moved x0. No backprop through the network.
        root_relative: score the joints minus the root, so guidance cannot create speed peaks by jerking the root
        translation (with world joints, scale 0.003 locked at real-dance level by raising root speed 0.33 -> 1.22 m/s
        while the body motion stayed the same).
        The batch should hold several songs, sorted by song (the null beats are other clips of the batch).
        guide_shift (placebo): guide towards the beats of the clip guide_shift places earlier in the batch (another
        song) instead of the clip's own beats; the model's conditioning is unchanged.
        groups: song / take of each clip; if given, the null beats are those of n_null random clips of other groups
        (fixed rng) instead of the rolled batch."""
        from model.beatlock import BEAT_COL, windowed_locking
        from model.musicality import lowpass_joints, pose_to_joints
        cond = torch.as_tensor(np.asarray(feats), dtype=torch.float32, device=self.device)
        genre = torch.tensor([style2s.index(g) for g in genres], dtype=torch.long, device=self.device)
        d = self.model.diffusion
        beats = cond[:, :, BEAT_COL]
        if guide_shift:
            beats = beats.roll(guide_shift, 0)
        null_beats = None
        if groups is not None:
            g = np.asarray(groups)
            rng = np.random.default_rng(0)
            idx = np.stack([rng.choice(np.flatnonzero(g != g[i]), n_null, replace=False) for i in range(len(g))])
            null_beats = [beats[torch.as_tensor(idx[:, k], device=self.device)] for k in range(n_null)]
        torch.manual_seed(seed)
        times = list(reversed(torch.linspace(-1, d.n_timestep - 1, steps=51).int().tolist()))
        b = cond.shape[0]
        x = torch.randn((b, cond.shape[1], self.model.repr_dim), device=self.device)
        self.guide_log = []
        for time, time_next in zip(times[:-1], times[1:]):
            tc = torch.full((b,), time, device=self.device, dtype=torch.long)
            with torch.no_grad():
                pred_noise, x_start = d.model_predictions(x, cond, genre, tc, clip_x_start=d.clip_denoised)
            if scale > 0 and time <= t_max:
                with torch.enable_grad():
                    x0 = x_start.detach().requires_grad_(True)
                    j = pose_to_joints(x0, self.model.normalizer, d.smpl)
                    if root_relative:
                        j = j - j[:, :, :1]
                    j = lowpass_joints(j, joint_sigma)
                    score = windowed_locking(j, beats, wide, null_beats=null_beats)
                    g, = torch.autograd.grad(score.sum(), x0)
                g = g / g.pow(2).mean((1, 2), keepdim=True).sqrt().clamp_min(1e-12)
                self.guide_log.append((time, float(score.mean())))
                x_start = x_start + scale * g
                pred_noise = d.predict_noise_from_start(x, tc, x_start)
            if time_next < 0:
                x = x_start
                continue
            a, an = d.alphas_cumprod[time], d.alphas_cumprod[time_next]
            sigma = ((1 - a / an) * (1 - an) / (1 - a)).sqrt()
            x = x_start * an.sqrt() + (1 - an - sigma ** 2).sqrt() * pred_noise + sigma * torch.randn_like(x)
        return self.to_joints(x)

    def to_joints(self, x):
        """Normalized samples (B, T, 151) -> (B, T, 24, 3) joints as numpy."""
        d = self.model.diffusion
        x = self.model.normalizer.unnormalize(x.detach().cpu())
        _, x = torch.split(x, (4, x.shape[2] - 4), dim=2)
        q = ax_from_6v(x[:, :, 3:].reshape(x.shape[0], x.shape[1], 24, 6)).to(self.device)
        return d.smpl.forward(q, x[:, :, :3].to(self.device)).cpu().numpy()
