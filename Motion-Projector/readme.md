# Motion Projector (3D → 2D)

Converts the Motion Expert's SMPL output into the 2D skeleton **pose-control video** that the
Appearance Expert (Wan-Animate) consumes.

> **Status: reimplementation from the paper's description, not the authors' code.** The paper
> says: SMPL → 3D mesh → pyrender under a fixed frontal camera → ViTPose 2D keypoints. Details the
> paper does not give (camera field of view, distance, mesh material, frame-rate handling) are my
> choices and may differ from the authors'.

```text
Motion Expert .pkl ──► SMPL mesh ──► pyrender (fixed frontal camera) ──► ViTPose-H wholebody
 (smpl_poses/trans)                                                        │
                                  pose_video.mp4  ◄── Wan-Animate skeleton ◄┘
```

Keypoint extraction and skeleton drawing are Wan2.2's own code, vendored unmodified in
`wan_preprocess/` (see `ATTRIBUTION.md`), so the output matches Wan-Animate's expected format.

## Setup

```bash
cd Motion-Projector
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Weights (ViTPose-H wholebody ≈ 2.5 GB, YOLOv10m ≈ 60 MB) are in the Wan2.2-Animate repo under
`process_checkpoint/`. The ViTPose "file" is a folder of ~390 files that `hf download --include`
misses, so use the provided script (needs `huggingface_hub`):

```bash
python download_weights.py   # -> models/process_checkpoint/{pose2d,det}/
```

You also need the **SMPL neutral model** (`SMPL_NEUTRAL*.pkl`). It is licensed separately and
is not distributed here: <https://smpl.is.tue.mpg.de/>. Pass its path with `--smpl_model`.
The chumpy-free variant works directly with `smplx`.

## Usage

```bash
python smpl_to_pose.py \
  --motion ../Expert-Motion/eval/train_exp2/0/012_slice0.pkl \
  --smpl_model /path/to/SMPL_NEUTRAL_chumpyfree.pkl \
  --out outputs/pose_video.mp4 --save_render outputs/render.mp4
```

Defaults: 77 frames at 15 fps from the 30 fps motion (every 2nd frame), 720×1280, matching the
Appearance Expert defaults (81 frames → 77 control frames, 15 fps output). Use `--start` to pick
a different segment and `--num_frames 0` for the whole clip.

Headless rendering uses EGL (`PYOPENGL_PLATFORM=egl` is set automatically).

## Conventions (verified on released Motion Expert outputs)

- `smpl_poses`: (T, 72) axis-angle, `smpl_trans`: (T, 3) in **meters**, **z-up** world, 30 fps.
- `smpl_trans` equals the root joint of `full_pose`.
- The camera is placed in front of the root's frame-0 facing direction and then held fixed.

## Known gaps

- **Face-control video is not produced.** SMPL has no facial information, and the paper does not
  describe how the face video is obtained for a generated dance. Wan-Animate needs one, so for
  tests a face video from `Expert-Appearance/data/` has been used as a stand-in.
- SMPL has no finger articulation, so hand keypoints are coarse compared to real-video pose.
- No retargeting of the skeleton to the reference image's body proportions or position.
