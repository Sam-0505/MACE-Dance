"""Lodge (official FineDance checkpoints) with our training-free MACE sampling fixes as options (2026-11-06).

Question: do the fixes we add to MACE at sampling time (released weights unchanged) also work on another diffusion
dance model? Same pipeline as lodge_gen_finedance.py (84 FineDance test slices, seeds 1000 * s + i, root x/z
continuity fix, Y-up -> Z-up FK); with every option off the local stage is a line-by-line copy of Lodge's
GaussianDiffusion.inpaint_soft_ddim and reproduces the stored mace_test_lodge raw outputs (--check_off).
Run from anywhere with Lodge's venv (the official repo files are not modified; the local diffusion's
inpaint_soft_ddim is replaced on the instance):

    P=/home/sam0505/Documents/Code/dance_baselines/LODGE/.venv/bin/python
    $P -W ignore Evaluation-Musicality/baselines/lodge_gen_finedance_guided.py --stage gen --check_off --clips 0,40 --seeds 0
    $P -W ignore Evaluation-Musicality/baselines/lodge_gen_finedance_guided.py --stage gen --tag guide_frz_foot \
        --lock_guide 0.003 --lock_foot 10 --lock_freeze_root --seeds 0,1,2
    $P -W ignore Evaluation-Musicality/baselines/lodge_gen_finedance_guided.py --stage joints --tag guide_frz_foot --seeds 0,1,2

How each MACE fix maps onto Lodge
  Lodge = global stage (DanceDecoder, DDIM 50, 1024 music frames -> 104 key frames = 13 primitives of 8 frames) +
  local stage (Refine_DanceDecoder, inpaint_soft_ddim, 4 windows of 256 frames = 8.5 s denoised as one batch, the
  global key frames pinned at fixed window positions: frames 0-3, 28-35, 60-67, 156-163, 188-195, 252-255, contact and
  rotation channels; cfg.soft = 1 -> pinned at every step). Representation 139 = 4 foot contacts + 3 root position +
  22 x 6D (SMPL-X body = SMPL 0-21), MinMax-normalised to [-1, 1] like MACE's 151 (same EDGE normaliser class).
  1. +-2 s cross-attention mask: NOT applied. MACE's mask repairs attention over a 34 s clip sampled in one pass by a
     model trained on short windows. Lodge's local model attends to the 256 music frames of its own 8.5 s window,
     exactly its training length (full_seq_len 256), so there is nothing to restore (Lodge already couples to
     loudness: Loud +.119 on FineDance).
  2. Beat-locking guidance: on the LOCAL stage only. The local stage produces the final 30 fps motion; the global
     stage outputs sparse key primitives whose placement in the final motion is set by fixed window positions, not
     by music time, so a beat-locking score is not defined on them, and the local stage pins only 40 of 256 frames to
     them. At local steps t <= t_max: x0 of the 4 windows -> joints (eval FK: EDGE SMPL offsets, hand joints 22/23
     with identity rotation, as lodge_gen_finedance.smpl_fk; no axis change, the score is rotation invariant) ->
     concatenated to the 1024-frame clip, cut to the evaluated 1020 frames -> root-relative -> low-pass (sigma 2) ->
     model.beatlock.windowed_locking (wide 10, 240-frame windows) with the clip's own beats (baseline_feats col 34,
     the beats MACE gets and the metric uses) and as null the beats of clips i-21, i-42, i-63 (= MACE's batch roll
     over the 84 test clips sorted by song). Gradient through the score only; x0 += scale * g / RMS(g), RMS over the
     clip's 4 x 256 x 139 entries (MACE: per sample over T x 151); noise estimate recomputed from the moved x0.
     --lock_freeze_root: g = 0 on the root 6D rotation channels 7:13 (same layout as MACE's 151).
     Lodge-specific: g = 0 on the entries the local stage pins to the global key frames (the inpainting mask). In
     every step but the last those entries of x_t are overwritten anyway; at the last step x = x0 is not re-pinned,
     so without this guidance would alter the global stage's key poses and the window seams.
  3. Foot-contact term (--lock_foot n): same as MACE (n steps of mu x Polyak on L = squared frame-to-frame
     displacement of SMPL 7, 8, 10, 11 in frames whose own predicted contact > 0.95). Lodge predicts the same 4
     contacts (EDGE layout, 7, 8, 10, 11) and was trained with the same foot loss (smpl_loss_relative, threshold
     0.95 on the unnormalised contacts), so this is the same term, not a stand-in. L is per local window (each
     window's root translation starts at 0, so there is no displacement across seams) and the Polyak step is per
     window (L is a sum over windows); pinned entries are frozen as above. FK = eval FK (Lodge trained with its
     SMPL-X skeleton; the offsets differ by mm-cm, the root translation, where 90 % of grad L lies, not at all).
"""
import argparse
import contextlib
import io
import json
import os
import random
import shutil
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LODGE = '/home/sam0505/Documents/Code/dance_baselines/LODGE'
EXPERT = '/home/sam0505/Documents/Code/MACE-Dance/Expert-Motion'
DATA = os.path.join(os.path.dirname(HERE), 'data')
sys.path.insert(0, HERE)
import lodge_gen_finedance as base  # noqa: E402  (shared: models, padding, FK, root fix)

BASE_WORK = os.path.join(LODGE, 'finedance_work')       # features + unguided raw outputs (mace_test_lodge)
WORK_ROOT = os.path.join(LODGE, 'finedance_work_guided')
N_CLIPS, N_FRAMES, LENGTH2 = base.N_CLIPS, base.N_FRAMES, base.LENGTH2
NULL_SHIFTS = (21, 42, 63)   # MACE: batch of 84 rolled by B * k / 4, k = 1..3 (windowed_locking)


def test_names():
    names = sorted(f[:-4] for f in os.listdir(os.path.join(EXPERT, 'data', 'test', 'baseline_feats')))
    with open(os.path.join(DATA, 'finedance_test_wav', 'slices.json')) as f:
        assert json.load(f)['names'] == names
    return names


def test_beats():
    """(84, 1020) beat one-hots of the FineDance test slices (baseline_feats column 34)."""
    from model.beatlock import BEAT_COL
    return np.stack([np.load(os.path.join(EXPERT, 'data', 'test', 'baseline_feats', n + '.npy'))[:, BEAT_COL]
                     for n in test_names()]).astype(np.float32)


# ----------------------------------------------------------------------------------------------- guidance
class Guide:
    """Options + per-clip context for the guided local stage. scale 0 and foot_iters 0 = Lodge unchanged."""

    def __init__(self, normalizer, device, scale=0.0, t_max=600, joint_sigma=2, wide=10, freeze_root_rot=False,
                 foot_iters=0, foot_mu=0.1, contact_thresh=0.95):
        import torch
        self.scale, self.t_max, self.joint_sigma, self.wide = scale, t_max, joint_sigma, wide
        self.freeze_root_rot, self.foot_iters, self.foot_mu, self.contact_thresh = (freeze_root_rot, foot_iters,
                                                                                    foot_mu, contact_thresh)
        sc = normalizer.scaler
        self.min_, self.scale_ = sc.min_.to(device).float(), sc.scale_.to(device).float()
        self.offsets = torch.tensor(base.SMPL_OFFSETS, dtype=torch.float32, device=device)
        self.device = device
        self.beats = self.null_beats = None
        self.log = []

    def set_clip(self, beats, null_beats):
        import torch
        self.beats = torch.as_tensor(beats, device=self.device)[None]
        self.null_beats = [torch.as_tensor(b, device=self.device)[None] for b in null_beats]
        self.log = []

    def joints(self, x):
        """Normalised local-stage x (W, 256, 139) -> (W, 256, 24, 3) joints (eval FK, Lodge's Y-up frame),
        differentiably and without the [-1, 1] clip of Normalizer.unnormalize (as musicality.pose_to_joints)."""
        import torch
        from pytorch3d.transforms import rotation_6d_to_matrix
        raw = (x - self.min_) / self.scale_
        w, t, _ = raw.shape
        rot = rotation_6d_to_matrix(raw[:, :, 7:].reshape(w, t, 22, 6))
        root = raw[:, :, 4:7]
        pos, rw = [], []
        for j, p in enumerate(base.SMPL_PARENTS):
            if p == -1:
                pos.append(root); rw.append(rot[:, :, 0])
            else:
                pos.append(torch.einsum('wtab,b->wta', rw[p], self.offsets[j]) + pos[p])
                rw.append(rw[p] @ rot[:, :, j] if j < 22 else rw[p])
        return torch.stack(pos, dim=2)

    def lock_score(self, x):
        """(W, 256, 139) -> (1,) chance-corrected windowed locking of the concatenated, evaluated 1020 frames."""
        from model.beatlock import windowed_locking
        from model.musicality import lowpass_joints
        j = self.joints(x).reshape(1, -1, 24, 3)[:, :N_FRAMES]
        j = lowpass_joints(j - j[:, :, :1], self.joint_sigma)
        return windowed_locking(j, self.beats, self.wide, null_beats=self.null_beats)

    def foot_loss(self, x):
        """Per window (W,): mean over frames of the squared displacement of the 4 foot joints in frames whose own
        contact prediction (unnormalised channels 0-3) exceeds contact_thresh. Returns loss, contact share."""
        contact = (x[:, :, :4].detach() - self.min_[:4]) / self.scale_[:4] > self.contact_thresh
        feet = self.joints(x)[:, :, [7, 8, 10, 11]]
        step = (feet[:, 1:] - feet[:, :-1]).pow(2).sum(-1) * contact[:, :-1]
        return step.sum((1, 2)) / step.shape[1], float(contact.float().mean())

    def foot_correct(self, x, free):
        import torch
        x = x.detach()
        l0 = None
        for _ in range(self.foot_iters):
            with torch.enable_grad():
                xr = x.requires_grad_(True)
                lf, share = self.foot_loss(xr)
                g, = torch.autograd.grad(lf.sum(), xr)
            g = g * free
            l0 = float(lf.mean()) if l0 is None else l0
            eta = (lf.detach() / g.pow(2).sum((1, 2)).clamp_min(1e-20)).view(-1, 1, 1)
            x = (x - self.foot_mu * eta * g).detach()
        with torch.no_grad():
            l1, _ = self.foot_loss(x)
        return x, l0, float(l1.mean()), share

    def step(self, x_start, free, time_):
        """One guidance step on the local stage's x0 prediction (W, 256, 139); free = 1 - inpainting mask."""
        import torch
        with torch.enable_grad():
            x0 = x_start.detach().float().requires_grad_(True)
            score = self.lock_score(x0)
            g, = torch.autograd.grad(score.sum(), x0)
        g = g * free
        rot_share = float(g[:, :, 7:13].pow(2).sum() / g.pow(2).sum().clamp_min(1e-30))   # diagnostic (MACE: 69 %)
        if self.freeze_root_rot:
            g[:, :, 7:13] = 0
        g = g / g.pow(2).mean().sqrt().clamp_min(1e-12)        # one sample = the clip's 4 windows
        x_start = x_start + self.scale * g
        rec = {'t': int(time_), 'score': float(score), 'root_rot_share': rot_share}
        if self.foot_iters:
            x_start, f0, f1, share = self.foot_correct(x_start, free)
            rec.update(foot_before=f0, foot_after=f1, contact_share=share)
        self.log.append(rec)
        return x_start


def guided_inpaint_soft_ddim(self, shape, cond, genre, noise=None, constraint=None, return_diffusion=False,
                             start_point=None):
    """GaussianDiffusion.inpaint_soft_ddim (dld/models/architectures/diffusion.py) line by line, plus the optional
    guidance step on x0 (self.guide). Same random draws in the same order, so scale 0 gives Lodge's sample."""
    import torch
    batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1
    assert batch > 1
    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
    time_pairs = list(zip(times[:-1], times[1:], weights))

    x = torch.randn(shape, device=device)
    cond = cond.to(device)
    assert x.shape[1] % 2 == 0
    mask = constraint["mask"].to(device)
    value = constraint["value"].to(device)
    guide = self.guide
    free = 1.0 - mask

    for time_, time_next, weight in time_pairs:
        time_cond = torch.full((batch,), time_, device=device, dtype=torch.long)
        pred_noise, x_start, *_ = self.model_predictions(x, cond, genre, time_cond, weight=weight,
                                                         clip_x_start=self.clip_denoised)
        if guide is not None and guide.scale > 0 and time_ <= guide.t_max:
            x_start = guide.step(x_start, free, time_)
            pred_noise = self.predict_noise_from_start(x, time_cond, x_start)

        if time_next < 0:
            x = x_start
            continue

        alpha = self.alphas_cumprod[time_]
        alpha_next = self.alphas_cumprod[time_next]
        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()
        noise = torch.randn_like(x)
        x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        if time_ > 50 * (1 - self.cfg.soft):
            x = value * mask + (1.0 - mask) * x
        x[:, :4, :] = value[:, :4, :] * mask[:, :4, :] + (1.0 - mask[:, :4, :]) * x[:, :4, :]
        x[:, -4:, :] = value[:, -4:, :] * mask[:, -4:, :] + (1.0 - mask[:, -4:, :]) * x[:, -4:, :]
    return x


# ----------------------------------------------------------------------------------------------- stages
def parse_list(s, hi):
    if s == 'all':
        return list(range(hi))
    out = []
    for part in s.split(','):
        if '-' in part:
            lo, h = map(int, part.split('-')); out += list(range(lo, h + 1))
        else:
            out.append(int(part))
    return out


def stage_gen(a, clips, seeds):
    import torch
    from dld.data.FineDance_dataset import Genres_fd

    device = 'cuda:0'
    work = os.path.join(WORK_ROOT, a.tag)
    for sub in ('raw', 'global', 'log'):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    with open(os.path.join(work, 'args.json'), 'w') as f:
        json.dump(vars(a), f, indent=1)
    cfg, mc, mf = base.build_models(device)
    guide = Guide(mf.normalizer, device, scale=a.lock_guide, t_max=a.lock_guide_tmax, freeze_root_rot=a.lock_freeze_root,
                  foot_iters=a.lock_foot, foot_mu=a.lock_foot_mu)
    mf.diffusion.guide = guide
    mf.diffusion.inpaint_soft_ddim = types.MethodType(guided_inpaint_soft_ddim, mf.diffusion)
    st = base.styles()
    beats = test_beats()
    tmp = os.path.join(work, 'tmp')
    times, max_diff = [], 0.0
    for s in seeds:
        t0 = time.time()
        for i in clips:
            out = os.path.join(work, 'raw', f'{i:03d}_s{s}.npy')
            if os.path.exists(out) and not a.check_off:
                continue
            guide.set_clip(beats[i], [beats[(i - k) % N_CLIPS] for k in NULL_SHIFTS])
            fea = base.pad_to_window(np.load(os.path.join(BASE_WORK, 'feats', f'{i:03d}.npy')))
            seed = 1000 * s + i
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            with contextlib.redirect_stdout(io.StringIO()):
                dance, glob = base.generate_one(cfg, mc, mf, fea, Genres_fd[base.GENRE_MAP[st[i]]], tmp, device)
            assert dance.shape == (base.LENGTH1, 139) and np.isfinite(dance).all(), dance.shape
            if a.check_off:
                ref = np.load(os.path.join(BASE_WORK, 'raw', f'{i:03d}_s{s}.npy'))
                d = float(np.abs(dance - ref).max())
                max_diff = max(max_diff, d)
                print(f'check_off clip {i} seed {s}: max |diff| vs stored Lodge raw = {d:.3g} '
                      f'(mean |x| {np.abs(ref).mean():.3g})', flush=True)
            np.save(os.path.join(work, 'global', f'{i:03d}_s{s}.npy'), glob.astype(np.float32))
            np.save(out + '.tmp.npy', dance.astype(np.float32))
            os.replace(out + '.tmp.npy', out)
            if guide.log:
                with open(os.path.join(work, 'log', f'{i:03d}_s{s}.json'), 'w') as f:
                    json.dump(guide.log, f)
        dt = time.time() - t0
        times.append(dt)
        print(f'seed {s}: {len(clips)} clips in {dt:.0f} s', flush=True)
    with open(os.path.join(work, 'timing.json'), 'a') as f:
        f.write(json.dumps({'clips': len(clips), 'seeds': list(seeds), 'sec_per_seed': times}) + '\n')
    if a.check_off:
        print('check_off max |diff| over all:', max_diff)


def stage_joints(a, seeds):
    import torch
    work = os.path.join(WORK_ROOT, a.tag)
    out = np.zeros((N_CLIPS, len(seeds), N_FRAMES, 24, 3), dtype=np.float16)
    for i in range(N_CLIPS):
        raw = [base.root_continuous(np.load(os.path.join(work, 'raw', f'{i:03d}_s{s}.npy'))) for s in seeds]
        raw = torch.from_numpy(np.stack(raw)[:, :N_FRAMES]).double()
        out[i] = base.smpl_fk(raw[:, :, 7:].reshape(len(seeds), N_FRAMES, 22, 6), raw[:, :, 4:7]).numpy()
    assert np.isfinite(out).all()
    dest = os.path.join(DATA, a.out_name or f'mace_test_lodge_{a.tag}')
    assert not os.path.exists(os.path.join(dest, 'joints.npy')) or a.overwrite, f'{dest} exists (--overwrite)'
    os.makedirs(dest, exist_ok=True)
    np.save(os.path.join(dest, 'joints.tmp.npy'), out)
    os.replace(os.path.join(dest, 'joints.tmp.npy'), os.path.join(dest, 'joints.npy'))
    shutil.copy(os.path.join(DATA, 'finedance_test_wav', 'slices.json'), os.path.join(dest, 'slices.json'))
    shutil.copy(os.path.join(DATA, 'mace_test', 'names.json'), os.path.join(dest, 'names.json'))
    print('saved', os.path.join(dest, 'joints.npy'), out.shape)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['gen', 'joints'])
    ap.add_argument('--tag', default='off', help='work dir finedance_work_guided/<tag>, output data/mace_test_lodge_<tag>')
    ap.add_argument('--out_name', help='output dir name under data/ instead of mace_test_lodge_<tag>')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--clips', default='all', help='e.g. all, 0-83, 0,5,7')
    ap.add_argument('--seeds', default='0', help='e.g. 0,1,2 or 0-9')
    ap.add_argument('--check_off', action='store_true', help='compare every sample with the stored unguided raw output')
    ap.add_argument('--lock_guide', type=float, default=0, help='beat-locking guidance scale (0 = off)')
    ap.add_argument('--lock_guide_tmax', type=int, default=600)
    ap.add_argument('--lock_freeze_root', action='store_true', help='no guidance on the root rotation (7:13)')
    ap.add_argument('--lock_foot', type=int, default=0, help='foot-contact steps after each guidance step (0 = off)')
    ap.add_argument('--lock_foot_mu', type=float, default=0.1)
    a = ap.parse_args()
    os.chdir(LODGE)
    sys.path.insert(0, LODGE)
    sys.path.append(EXPERT)          # model.beatlock / model.musicality (namespace package, no name clash with dld)
    clips, seeds = parse_list(a.clips, N_CLIPS), parse_list(a.seeds, 10)
    if a.stage == 'gen':
        stage_gen(a, clips, seeds)
    else:
        stage_joints(a, seeds)
