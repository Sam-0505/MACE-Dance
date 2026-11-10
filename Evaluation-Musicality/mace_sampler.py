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
        # original-EDGE denoiser (--arch edge): EDGE's long_ddim_sample ramps the guidance weight 0 -> 2w, capped at w
        w_final = getattr(d.model, 'guidance_weight', None)
        ramp = None if w_final is None else np.clip(np.linspace(0, 2 * w_final, len(times) - 1), None, w_final)
        for step, (time, time_next) in enumerate(zip(times[:-1], times[1:])):
            tc = torch.full((b * n,), time, device=self.device, dtype=torch.long)
            if ramp is not None:
                d.model.guidance_weight = float(ramp[step])
            pred_noise, x_start = d.model_predictions(x, cond, genre, tc, clip_x_start=d.clip_denoised)
            if ramp is not None:
                d.model.guidance_weight = w_final
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
                      guide_shift=0, groups=None, n_null=3, foot_iters=0, foot_mu=0.1, contact_thresh=0.95,
                      freeze_root_rot=False, critic=None, critic_groups=None, grad_smooth=0.0, gain_ref=0.0):
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
        (fixed rng) instead of the rolled batch.
        foot_iters (option, 2026-11-06; 0 = off, identical to before): foot-contact term. After the locking step,
        foot_iters descent steps on L_foot = mean over frames of the squared world displacement of the 4 foot joints
        (SMPL 7, 8, 10, 11) in frames where the model's own x0 contact prediction (unnormalised, first 4 channels)
        exceeds contact_thresh, i.e. MACE's training foot loss (diffusion.p_losses) applied to x0. Each step is
        foot_mu times the Polyak step L / |grad L|^2 (per sample). Why not an RMS-normalised step like the locking
        one: 91 % of grad L lies on the root-translation channels and alternates frame to frame, so a step of the
        locking step's size moved the root by cm per frame and raised L 30-fold (FSR 26 -> 83 % on 8 clips); repeated
        small loss-sized steps lower it (final released samples: FSR 18.6 -> 7.6 % with 30 x 0.1).
        Guidance makes accents by moving the body around the root, and the planted feet were dragged along (Lodge FSR
        16.5 -> 26.9 % on FineDance); this term moves the root so they stay put.
        freeze_root_rot (option, 2026-11-06; False = as before): zero the locking gradient on the root's global
        rotation (6D channels 7:13) before normalising. Without it 69 % of the step's energy went to those 6 of 151
        channels: guidance made accents by twitching the whole skeleton about the pelvis (visible as a rotation
        jitter), the cheapest way to move every joint at once. Frozen, accents must come from the joints.
        grad_smooth (option, 2026-11-08; 0 = off, identical to before): Gaussian low-pass (sigma in frames) of the guidance
        gradient along time, before the RMS normalisation, so the step changes the pose only slowly (accent scale) and cannot
        add frame-to-frame twitching (critic v5 guidance gave jitter about the hip and other joints).
        gain_ref (option, 2026-11-08; 0 = off, identical to before): motion-proportional guidance gain. The step is
        multiplied per frame by w(t) = min(1, v(t) / gain_ref), v = the x0 prediction's own root-relative mean joint speed
        (m/s, Gaussian-smoothed sigma 8 frames, no gradient). The critic is RMS-normalised per window and the step is
        RMS-normalised per sample, so the push had the same size (about 0.5 cm/s) however still the model wanted the
        body to be (quiet music + mask) and showed up as foot/ankle twitching on a still body. Real FineDance median
        v = 0.49 m/s (use gain_ref 0.49): frames moving less than that get a proportionally smaller push."""
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
        c_feats = c_null = None
        if critic is not None:
            # critic guidance (2026-11-07, critic.py): the locking score is replaced by a frozen music-motion critic
            # trained on real dance (no beat-locking formula in the guidance). Own music = the clip's (guide_shift: another
            # clip's, the placebo); null = music of the 3 other-song clips with the closest beat count (tempo-matched).
            from critic import windowed_critic
            c_feats = cond[:, :, :]
            if guide_shift:
                c_feats = c_feats.roll(guide_shift, 0)
            nbt = (c_feats[:, :, BEAT_COL] > 0.5).sum(1).float()
            g = np.asarray(critic_groups) if critic_groups is not None else np.arange(cond.shape[0])
            if guide_shift:
                g = np.roll(g, guide_shift)
            other = torch.as_tensor(g.reshape(-1, 1) != g.reshape(1, -1), device=self.device)
            dist = (nbt.view(-1, 1) - nbt.view(1, -1)).abs() + 1e6 * (~other).float()
            idx = dist.topk(n_null, dim=1, largest=False).indices
            c_null = [c_feats[idx[:, k]] for k in range(n_null)]
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
                    jw = pose_to_joints(x0, self.model.normalizer, d.smpl)
                    if critic is not None:
                        score = windowed_critic(critic, jw, c_feats, c_null)
                    else:
                        j = jw - jw[:, :, :1] if root_relative else jw
                        j = lowpass_joints(j, joint_sigma)
                        score = windowed_locking(j, beats, wide, null_beats=null_beats)
                    g, = torch.autograd.grad(score.sum(), x0)
                if freeze_root_rot:
                    g[:, :, 7:13] = 0
                if grad_smooth > 0:
                    g = self.smooth_time(g, grad_smooth)
                g = g / g.pow(2).mean((1, 2), keepdim=True).sqrt().clamp_min(1e-12)
                if gain_ref > 0:
                    x_start = x_start + scale * g * self.motion_gain(jw.detach(), gain_ref)
                else:
                    x_start = x_start + scale * g
                if foot_iters:
                    x_start, f0, f1, cshare = self.foot_correct(x_start, foot_iters, foot_mu, contact_thresh)
                    self.guide_log.append((time, float(score.mean()), f0, f1, cshare))
                else:
                    self.guide_log.append((time, float(score.mean())))
                pred_noise = d.predict_noise_from_start(x, tc, x_start)
            if time_next < 0:
                x = x_start
                continue
            a, an = d.alphas_cumprod[time], d.alphas_cumprod[time_next]
            sigma = ((1 - a / an) * (1 - an) / (1 - a)).sqrt()
            x = x_start * an.sqrt() + (1 - an - sigma ** 2).sqrt() * pred_noise + sigma * torch.randn_like(x)
        return self.to_joints(x)

    @staticmethod
    def motion_gain(jw, ref, sigma=8):
        """(B, T, 1) gain min(1, v / ref), v = smoothed root-relative mean joint speed (m/s) of joints jw (B, T, 24, 3)."""
        rel = jw - jw[:, :, :1]
        sp = (rel[:, 1:] - rel[:, :-1]).norm(dim=-1).mean(-1) * 30
        sp = torch.cat([sp, sp[:, -1:]], 1).unsqueeze(-1)
        v = MaceSampler.smooth_time(sp, sigma)
        return (v / ref).clamp(max=1.0)

    @staticmethod
    def smooth_time(g, sigma):
        """Gaussian low-pass of g (B, T, C) along T (reflect padding)."""
        r = int(3 * sigma + 0.5)
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=g.device, dtype=g.dtype) / sigma) ** 2)
        k = (k / k.sum()).view(1, 1, -1).expand(g.shape[2], 1, -1)
        x = torch.nn.functional.pad(g.transpose(1, 2), (r, r), mode='reflect')
        return torch.nn.functional.conv1d(x, k, groups=g.shape[2]).transpose(1, 2)

    def foot_loss(self, x, contact_thresh=0.95):
        """MACE's training foot loss on a normalised sample x (B, T, 151): mean over frames of the squared displacement
        of the 4 foot joints in frames the model's own contact channels mark as planted. Returns (B,), contact share."""
        from model.musicality import pose_to_joints
        sc = self.model.normalizer.scaler
        contact = (x[:, :, :4].detach() - sc.min_[:4].to(x.device)) / sc.scale_[:4].to(x.device) > contact_thresh
        feet = pose_to_joints(x, self.model.normalizer, self.model.diffusion.smpl)[:, :, [7, 8, 10, 11]]
        step = (feet[:, 1:] - feet[:, :-1]).pow(2).sum(-1) * contact[:, :-1]
        return step.sum((1, 2)) / step.shape[1], float(contact.float().mean())

    def foot_correct(self, x, iters, mu, contact_thresh=0.95):
        """iters steps of mu x the Polyak step on foot_loss (see sample_guided). Returns x, L before, L after, share."""
        x = x.detach()
        l0 = None
        for _ in range(iters):
            with torch.enable_grad():
                xr = x.requires_grad_(True)
                lf, share = self.foot_loss(xr, contact_thresh)
                g, = torch.autograd.grad(lf.sum(), xr)
            l0 = float(lf.mean()) if l0 is None else l0
            eta = (lf.detach() / g.pow(2).sum((1, 2)).clamp_min(1e-20)).view(-1, 1, 1)
            x = (x - mu * eta * g).detach()
        with torch.no_grad():
            l1, _ = self.foot_loss(x, contact_thresh)
        return x, l0, float(l1.mean()), share

    def to_joints(self, x):
        """Normalized samples (B, T, 151) -> (B, T, 24, 3) joints as numpy."""
        d = self.model.diffusion
        x = self.model.normalizer.unnormalize(x.detach().cpu())
        _, x = torch.split(x, (4, x.shape[2] - 4), dim=2)
        q = ax_from_6v(x[:, :, 3:].reshape(x.shape[0], x.shape[1], 24, 6)).to(self.device)
        return d.smpl.forward(q, x[:, :, :3].to(self.device)).cpu().numpy()
