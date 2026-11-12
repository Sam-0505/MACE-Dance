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
SALT = 'mmc-mllm-v1'


def vid_id(slice_name, method):
    return hashlib.sha1(f'{SALT}|{slice_name}|{method}'.encode()).hexdigest()[:10]


def shared_camera(verts_list, forward, fov=SR.FOV_Y, margin=1.12):
    """Look-at point + distance so every frame of every clip in the pair fits, then a fixed camera."""
    allv = np.concatenate([v[::10] for v in verts_list])
    lo, hi = allv.min((0, 1)), allv.max((0, 1))
    center = (lo + hi) / 2
    height = (hi[2] - lo[2]) * margin
    dist = height / (2 * np.tan(fov / 2))
    eye = center + np.array([forward[0], forward[1], 0.0]) * dist
    return SR._look_at(eye, center, np.array([0.0, 0.0, 1.0]))


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
    ap.add_argument('--width', type=int, default=720)
    ap.add_argument('--height', type=int, default=1280)
    ap.add_argument('--crf', type=int, default=23)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    data = {m: np.load(os.path.join(HERE, 'data', 'rot_export', f), allow_pickle=True) for m, f in METHODS.items()}
    names = list(data['released']['names'])
    assert names == list(data['ours']['names'])
    rows = []
    for n in a.slices:
        i = names.index(n)
        wav = os.path.join(WAV_DIR, n + '.wav')
        verts, faces = {}, None
        for m, z in data.items():
            verts[m], faces = SR.smpl_vertices(z['poses'][i].astype(np.float32), z['trans'][i].astype(np.float32), SMPL, a.device)
        forward = SR.facing_direction(data['released']['poses'][i][0].astype(np.float32))
        cam = shared_camera(list(verts.values()), forward)
        for m in METHODS:
            vid = vid_id(n, m)
            out = os.path.join(a.out_dir, f'{vid}.mp4')
            t0 = time.time()
            render(verts[m], faces, cam, wav, out, a.width, a.height, SR.MOTION_FPS, a.crf)
            print(f'{n} {m} -> {out} ({time.time() - t0:.0f} s, {os.path.getsize(out) / 1e6:.2f} MB)', flush=True)
            rows.append({'video_id': vid, 'slice': n, 'method': m, 'path': out, 'wav': wav, 'frames': len(verts[m]),
                         'fps': SR.MOTION_FPS})
    mp = os.path.join(a.out_dir, 'mapping.csv')
    new = not os.path.exists(mp)
    with open(mp, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
