"""3D-to-2D Motion Projector: Motion Expert SMPL output -> Wan-Animate pose video.

Follows the MACE-Dance paper's description: SMPL sequence -> 3D mesh -> pyrender
under a fixed frontal camera -> ViTPose 2D keypoints -> Wan-Animate skeleton.
This is a reimplementation from that description, not the authors' code. Keypoint
extraction and skeleton drawing reuse Wan2.2's own preprocess code (vendored in
wan_preprocess/), so the output matches the format Wan-Animate expects.

Example:
    python smpl_to_pose.py \
        --motion ../Expert-Motion/eval/train_exp2/0/012_slice0.pkl \
        --smpl_model /path/to/SMPL_NEUTRAL_chumpyfree.pkl \
        --out pose_video.mp4
"""

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "wan_preprocess"))  # vendored modules use flat imports

import smpl_render  # noqa: E402  (sets PYOPENGL_PLATFORM before pyrender import)

OUTPUT_FPS = 15  # the Appearance Expert saves 15 fps videos
DEFAULT_CONTROL_FRAMES = 77  # Wan Animate consumes num_frames - 4 (81 -> 77)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", required=True, help="Motion Expert .pkl (smpl_poses, smpl_trans)")
    p.add_argument("--smpl_model", required=True, help="SMPL neutral model .pkl (not distributed here)")
    p.add_argument("--out", required=True, help="Output pose video (.mp4)")
    p.add_argument("--vitpose", default=str(HERE / "models/process_checkpoint/pose2d/vitpose_h_wholebody.onnx"))
    p.add_argument("--detector", default=str(HERE / "models/process_checkpoint/det/yolov10m.onnx"))
    p.add_argument("--num_frames", type=int, default=DEFAULT_CONTROL_FRAMES, help="output frames; 0 = whole clip")
    p.add_argument("--start", type=int, default=0, help="first motion frame (at 30 fps)")
    p.add_argument("--fps", type=int, default=OUTPUT_FPS)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=1280)
    p.add_argument("--save_render", default=None, help="optionally also save the mesh render as mp4")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    if smpl_render.MOTION_FPS % args.fps:
        raise ValueError(f"--fps must divide the {smpl_render.MOTION_FPS} fps motion rate")
    stride = smpl_render.MOTION_FPS // args.fps

    poses, trans = smpl_render.load_motion(args.motion)
    idx = np.arange(args.start, len(poses), stride)
    if args.num_frames:
        idx = idx[: args.num_frames]
    if len(idx) == 0:
        raise ValueError(f"No frames: motion has {len(poses)} frames, start={args.start}")
    print(f"Motion {len(poses)} frames @ {smpl_render.MOTION_FPS} fps -> {len(idx)} frames @ {args.fps} fps")

    verts, faces = smpl_render.smpl_vertices(poses[idx], trans[idx], args.smpl_model)
    forward = smpl_render.facing_direction(poses[idx[0]])
    frames = smpl_render.render_frames(verts, faces, forward, args.width, args.height)
    if args.save_render:
        Path(args.save_render).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.save_render, frames, fps=args.fps, quality=8)

    from human_visualization import draw_aapose_by_meta_new
    from pose2d import Pose2d
    from pose2d_utils import AAPoseMeta

    pose2d = Pose2d(checkpoint=args.vitpose, detector_checkpoint=args.detector, device=args.device)
    # Pose2d takes a list of frames and mirrors Wan's own preprocessing; the render is
    # passed exactly as Wan's pipeline passes decoded RGB frames.
    metas = pose2d(frames)

    canvas_frames = []
    for meta in tqdm(metas, desc="Drawing skeleton"):
        aa_meta = AAPoseMeta.from_humanapi_meta(meta)
        canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        canvas_frames.append(draw_aapose_by_meta_new(canvas, aa_meta))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, canvas_frames, fps=args.fps, quality=8)
    print(f"Saved {len(canvas_frames)} pose frames: {args.out}")


if __name__ == "__main__":
    main()
