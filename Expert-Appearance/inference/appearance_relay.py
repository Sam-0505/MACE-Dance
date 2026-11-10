"""Relay rendering of one chunk of a long dance video (2026-11-07), after the MACE paper's "pose-anchored relay generation"
(supplement, Long-Sequence Generation): every chunk is constrained by (i) its slice of the global 2D pose sequence,
(ii) the last frame of the previous chunk and (iii) the constant reference image. NOT part of the authors' released code:
the released appearance_infer.py renders each row independently.

How (ii) is fed in: the vendored Wan2.2-Animate pipeline already accepts animate_inpaint_video + animate_mask_video, which
build the conditioning channels `y` of the control frames (diffsynth/pipelines/wan_video_new.py, WanVideoPostUnit_AnimateInpaint).
Here the inpaint video holds the previous chunk's last frame at position 0 and mid-grey elsewhere, and the mask marks
position 0 as known (Wan's own temporal guidance with refert_num = 1; the paper says "the last frame").

Chunks overlap by one frame: chunk k's pose slice is control frames [76 k, 76 k + 77) of the full pose video, so its
frame 0 shows the same motion frame as the last frame of chunk k-1 and is dropped when the chunks are joined
(77 + 76 (n-1) frames: 3 chunks = 229 frames = 15.3 s at 15 fps).

    python inference/appearance_relay.py --name beat_ours --chunk 1 --pose_video <full pose.mp4> \
        --prev_last_frame outputs/long/beat_ours_c0_last.png --out_dir outputs/long
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

from appearance_infer import create_pipeline  # noqa: E402
from diffsynth import VideoData, save_video  # noqa: E402

CONTROL = 77  # control frames per chunk (num_frames 81 - 4)
STRIDE = CONTROL - 1  # chunks overlap by one frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--chunk', type=int, required=True)
    ap.add_argument('--pose_video', required=True, help='pose video of the WHOLE window (77 + 76 (n-1) frames)')
    ap.add_argument('--face_video', default=str(PROJECT_ROOT / 'data/0001/face_video.mp4'))
    ap.add_argument('--reference', default=str(PROJECT_ROOT / 'data/0001/video.mp4'))
    ap.add_argument('--prompt', default='视频中的人在跳舞')
    ap.add_argument('--prev_last_frame', default=None, help='PNG: last frame of the previous chunk (chunk >= 1)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--height', type=int, default=1280)
    ap.add_argument('--width', type=int, default=720)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{a.name}_c{a.chunk}.mp4'
    if path.exists() and not a.overwrite:
        print(f'Skipping existing {path}')
        return
    pose_all = VideoData(a.pose_video, height=a.height, width=a.width).raw_data()
    s = STRIDE * a.chunk
    pose = pose_all[s:s + CONTROL]
    if len(pose) < CONTROL:
        raise ValueError(f'pose video has {len(pose_all)} frames, chunk {a.chunk} needs {s + CONTROL}')
    face = VideoData(a.face_video, height=512, width=512).raw_data()[:CONTROL]
    ref = VideoData(a.reference, height=a.height, width=a.width)[0]

    extra = {}
    if a.chunk > 0:
        prev = Image.open(a.prev_last_frame).convert('RGB').resize((a.width, a.height))
        grey = Image.new('RGB', (a.width, a.height), (128, 128, 128))
        black, white = Image.new('RGB', (a.width, a.height), 0), Image.new('RGB', (a.width, a.height), (255, 255, 255))
        # mask video: black = known (kept), white = generated; frame 0 is the known frame
        extra = {'animate_inpaint_video': [prev] + [grey] * (CONTROL - 1),
                 'animate_mask_video': [black] + [white] * (CONTROL - 1)}

    pipe = create_pipeline(PROJECT_ROOT / 'models/Wan-AI/Wan2.2-Animate-14B', PROJECT_ROOT / 'models/train/Adapter.safetensors',
                           PROJECT_ROOT / 'models/train/LoRA.safetensors')
    # Host-RAM fix (2026-11-08): before the VAE decode the pipeline offloads the 14B DiT, which allocates a second CPU copy of
    # its GPU-resident layers (anonymous RSS 41 -> 53 GB; with ~8.5 GB used by the rest of the 64 GB host the kernel OOM-killed
    # the process at the decode, in the relay pilot and in appearance_infer.py alike). The DiT and the adapter are not needed
    # after denoising and the process ends after the chunk, so drop them instead of offloading.
    import gc
    import torch
    orig_load, orig_fn, calls = pipe.load_models_to_device, pipe.model_fn, [0]

    def model_fn(*args, **kw):  # cfg_scale 1: one call per denoising step
        calls[0] += 1
        return orig_fn(*args, **kw)
    pipe.model_fn = model_fn

    def load_models_to_device(names=[]):  # noqa: B006
        if calls[0] >= a.steps:  # denoising is over: decode, then the process ends
            if pipe.dit is not None:
                pipe.dit = None
                pipe.animate_adapter = None
                gc.collect()
                torch.cuda.empty_cache()
            if not names:
                return
        return orig_load(names)
    pipe.load_models_to_device = load_models_to_device
    video = pipe(prompt=a.prompt, seed=a.seed, tiled=True, input_image=ref, animate_pose_video=pose, animate_face_video=face,
                 num_frames=CONTROL + 4, height=a.height, width=a.width, num_inference_steps=a.steps, cfg_scale=1.0, **extra)
    save_video(video, str(path), fps=15, quality=5)
    video[-1].save(out / f'{a.name}_c{a.chunk}_last.png')
    print(f'Saved {path} ({len(video)} frames)', flush=True)


if __name__ == '__main__':
    main()
