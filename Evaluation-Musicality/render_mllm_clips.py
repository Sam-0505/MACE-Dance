"""Blind 34 s mesh videos of released MACE vs ours (critic v5 + gain) for the TMD-Bench MLLM judge (MMC_Dance_Evaluation_Plan.md).

For each requested slice both methods are rendered with the IDENTICAL avatar, camera, framing, resolution, background,
lighting, encoding and audio (the slice's own 34.0 s wav). No overlays or text. Files get neutral hashed names; the
private video-id -> method mapping goes to mapping.csv next to them.

    cd Motion-Projector
    .venv/bin/python ../Evaluation-Musicality/render_mllm_clips.py --slices 012_slice0 --out_dir ../Evaluation-Musicality/data/mllm_pilot

Motion: Evaluation-Musicality/data/rot_export/{oneshot,gain}_seed0.npz (SMPL axis-angle, Z-up, 30 fps, 1020 frames).
The camera is shared by the pair: look-at point and distance come from the pooled vertices of both clips, facing from
the released clip's frame 0, so neither method gets its own framing.
"""
import argparse
import csv
import hashlib
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'Motion-Projector'))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import smpl_render as SR  # noqa: E402

import pyrender  # noqa: E402
import trimesh  # noqa: E402

METHODS = {'released': 'oneshot_seed0.npz', 'ours': 'gain_seed0.npz'}
SMPL = os.path.join(ROOT, 'Motion-Projector', 'models', 'smpl', 'SMPL_NEUTRAL_chumpyfree.pkl')
WAV_DIR = os.path.join(ROOT, 'Expert-Motion', 'data', 'test', 'wavs_sliced')
FF = os.path.join(ROOT, 'Expert-Appearance', '.venv', 'lib', 'python3.10', 'site-packages', 'imageio_ffmpeg',
                  'binaries', 'ffmpeg-linux-x86_64-v7.0.2')
SALT = 'mmc-mllm-v2'  # v2 = full-trajectory camera, 1280x720; v1 videos (portrait, height-only framing) are superseded


def vid_id(slice_name, method):
    return hashlib.sha1(f'{SALT}|{slice_name}|{method}'.encode()).hexdigest()[:10]


def shared_camera(verts_list, forward, aspect, fov=SR.FOV_Y, margin=1.12):
    """Fixed camera shared by the pair, framed so EVERY vertex of EVERY frame of both clips fits in width and height
    (v1 fitted the height only, so dancers that travel sideways left the portrait frame; v2 fix 2026-11-12)."""
    allv = np.concatenate([v[:, ::4].reshape(-1, 3) for v in verts_list])  # all frames, every 4th vertex
    center = (allv.min(0) + allv.max(0)) / 2
    up = np.array([0.0, 0.0, 1.0])
    fwd = np.array([forward[0], forward[1], 0.0])  # target -> camera
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    vert = np.cross(fwd, right)
    rel = allv - center
    u, xr, yr = rel @ fwd, np.abs(rel @ right), np.abs(rel @ vert)
    t = np.tan(fov / 2)
    dist = max(float(np.max(u + margin * np.maximum(xr / (t * aspect), yr / t))), float(u.max()) + 0.5)
    return SR._look_at(center + fwd * dist, center, up)


def out_of_frame(verts, cam_pose, aspect, fov=SR.FOV_Y):
    """Fraction of frames in which any vertex projects outside the image."""
    c = np.concatenate([verts, np.ones(verts.shape[:2] + (1,))], -1) @ np.linalg.inv(cam_pose).T
    t = np.tan(fov / 2)
    x, y = c[..., 0] / -c[..., 2] / (t * aspect), c[..., 1] / -c[..., 2] / t
    return float(((np.abs(x) > 1) | (np.abs(y) > 1)).any(1).mean())


def render(verts, faces, cam_pose, wav, out, width, height, fps, crf):
    scene = pyrender.Scene(bg_color=[0.78, 0.80, 0.82, 1.0], ambient_light=[0.45, 0.45, 0.45])
    scene.add(pyrender.PerspectiveCamera(yfov=SR.FOV_Y, aspectRatio=width / height), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)
    mat = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, roughnessFactor=0.8,
                                             baseColorFactor=[0.85, 0.62, 0.55, 1.0])
    r = pyrender.OffscreenRenderer(width, height)
    proc = subprocess.Popen([FF, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s',
                             f'{width}x{height}', '-r', str(fps), '-i', '-', '-i', wav, '-map', '0:v', '-map', '1:a',
                             '-c:v', 'libx264', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k',
                             '-t', f'{len(verts) / fps:.3f}', out], stdin=subprocess.PIPE)
    node = None
    try:
        for v in verts:
            if node is not None:
                scene.remove_node(node)
            node = scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v, faces, process=False), material=mat, smooth=True))
            color, _ = r.render(scene)
            proc.stdin.write(np.ascontiguousarray(color).tobytes())
    finally:
        r.delete()
        proc.stdin.close()
        proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slices', nargs='+', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--width', type=int, default=1280)  # v2: landscape, dancers that travel sideways stay in shot
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--crf', type=int, default=23)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    data = {m: np.load(os.path.join(HERE, 'data', 'rot_export', f), allow_pickle=True) for m, f in METHODS.items()}
    names = list(data['released']['names'])
    assert names == list(data['ours']['names'])
    for n in a.slices:
        i = names.index(n)
        wav = os.path.join(WAV_DIR, n + '.wav')
        verts, faces = {}, None
        for m, z in data.items():
            verts[m], faces = SR.smpl_vertices(z['poses'][i].astype(np.float32), z['trans'][i].astype(np.float32), SMPL, a.device)
        forward = SR.facing_direction(data['released']['poses'][i][0].astype(np.float32))
        cam = shared_camera(list(verts.values()), forward, a.width / a.height)
        oof = {m: out_of_frame(v, cam, a.width / a.height) for m, v in verts.items()}
        print(f'{n} frames with any vertex out of frame: ' + ', '.join(f'{m} {x:.1%}' for m, x in oof.items()), flush=True)
        if max(oof.values()) >= 0.01:  # skip this clip (reported at the end) instead of aborting the whole run
            print(f'SKIPPED {n}: dancer leaves the frame in >=1% of frames', flush=True)
            continue
        rows = []
        for m in METHODS:
            vid = vid_id(n, m)
            out = os.path.join(a.out_dir, f'{vid}.mp4')
            t0 = time.time()
            render(verts[m], faces, cam, wav, out, a.width, a.height, SR.MOTION_FPS, a.crf)
            print(f'{n} {m} -> {out} ({time.time() - t0:.0f} s, {os.path.getsize(out) / 1e6:.2f} MB)', flush=True)
            rows.append({'video_id': vid, 'slice': n, 'method': m, 'path': out, 'wav': wav, 'frames': len(verts[m]),
                         'fps': SR.MOTION_FPS})
        mp = os.path.join(a.out_dir, 'mapping.csv')  # appended per clip so a crash cannot lose the video-id -> method map
        new = not os.path.exists(mp)
        with open(mp, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
