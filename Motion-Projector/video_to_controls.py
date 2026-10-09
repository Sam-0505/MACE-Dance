"""Real dance video -> the Appearance Expert's three inputs (reference, pose, face).

Follows the preprocessing recipe of Wan2.2's `process_pipepline.py` (non-retarget path):
frames resized by area to 1280x720 (divisor 16), ViTPose-H wholebody keypoints, the
Wan-Animate skeleton drawn on black, and a 512x512 face crop (face-landmark box, scale 1.3).
Reference, pose and face all come from the same clip, i.e. a reconstruction setup. It is
what the sample data in Expert-Appearance/data/ looks like, and how training pairs could be
built from MA-Data's real-video (Appearance) subset.

Example:
    python video_to_controls.py --video clip.mp4 --out_dir outputs/controls/clip
"""

import argparse
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "wan_preprocess"))  # vendored modules use flat imports

DEFAULT_CONTROL_FRAMES = 77  # Wan Animate consumes num_frames - 4 (81 -> 77)
SAMPLE_FPS = 16  # frame rate of the sample controls in Expert-Appearance/data/


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vitpose", default=str(HERE / "models/process_checkpoint/pose2d/vitpose_h_wholebody.onnx"))
    p.add_argument("--detector", default=str(HERE / "models/process_checkpoint/det/yolov10m.onnx"))
    p.add_argument("--fps", type=int, default=SAMPLE_FPS)
    p.add_argument("--num_frames", type=int, default=DEFAULT_CONTROL_FRAMES, help="0 = whole clip")
    p.add_argument("--start_sec", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def read_frames(path, fps, num_frames, start_sec, resize_by_area, get_frame_indices):
    cap = cv2.VideoCapture(str(path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")

    first = int(round(start_sec * video_fps))
    frames = frames[first:]
    target_num = int(len(frames) / video_fps * fps)
    idxs = get_frame_indices(len(frames), video_fps, target_num, fps)
    if num_frames:
        idxs = idxs[:num_frames]
    print(f"{path.name}: {len(frames)} frames @ {video_fps:.1f} fps -> {len(idxs)} frames @ {fps} fps")
    return [resize_by_area(frames[i], 1280 * 720, divisor=16) for i in idxs]


def main():
    args = parse_args()
    from human_visualization import draw_aapose_by_meta_new
    from pose2d import Pose2d
    from pose2d_utils import AAPoseMeta
    from utils import get_face_bboxes, get_frame_indices, resize_by_area

    frames = read_frames(Path(args.video), args.fps, args.num_frames, args.start_sec, resize_by_area, get_frame_indices)
    height, width = frames[0].shape[:2]

    pose2d = Pose2d(checkpoint=args.vitpose, detector_checkpoint=args.detector, device=args.device)
    metas = pose2d(frames)

    pose_frames, face_frames = [], []
    for frame, meta in tqdm(list(zip(frames, metas)), desc="Pose + face"):
        face_box = get_face_bboxes(meta["keypoints_face"][:, :2], scale=1.3, image_shape=(height, width))
        x1, x2, y1, y2 = face_box
        face_frames.append(cv2.resize(frame[y1:y2, x1:x2], (512, 512)))
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        pose_frames.append(draw_aapose_by_meta_new(canvas, AAPoseMeta.from_humanapi_meta(meta)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out / "reference.png", frames[0])
    imageio.mimsave(out / "video.mp4", frames, fps=args.fps, quality=8)
    imageio.mimsave(out / "pose_video.mp4", pose_frames, fps=args.fps, quality=8)
    imageio.mimsave(out / "face_video.mp4", face_frames, fps=args.fps, quality=8)
    print(f"Saved reference.png, video.mp4, pose_video.mp4, face_video.mp4 to {out}")


if __name__ == "__main__":
    main()
