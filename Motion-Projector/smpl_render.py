"""Render a Motion Expert SMPL sequence to frames with pyrender (headless EGL).

Part of the 3D-to-2D Motion Projector described in the MACE-Dance paper:
SMPL parameters -> 3D mesh -> pyrender under a fixed frontal camera.
This is a reimplementation from the paper's description, not the authors' code.

Motion Expert convention (verified on released outputs): axis-angle poses
(T, 72), root translation (T, 3) in meters, z-up world, 30 fps.
"""

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import smplx
import torch
import trimesh

MOTION_FPS = 30
FOV_Y = np.deg2rad(40.0)


def load_motion(path):
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["smpl_poses"].astype(np.float32), data["smpl_trans"].astype(np.float32)


def smpl_vertices(poses, trans, smpl_model_path, device="cpu", chunk=128):
    """Pose the neutral SMPL body for every frame. Returns (verts (T,6890,3), faces)."""
    model = smplx.SMPL(smpl_model_path, num_betas=10).to(device)
    verts = []
    with torch.no_grad():
        for i in range(0, len(poses), chunk):
            p = torch.from_numpy(poses[i : i + chunk]).to(device)
            t = torch.from_numpy(trans[i : i + chunk]).to(device)
            out = model(global_orient=p[:, :3], body_pose=p[:, 3:], transl=t)
            verts.append(out.vertices.cpu().numpy())
    return np.concatenate(verts), np.asarray(model.faces, dtype=np.int64)


def _look_at(eye, target, up):
    """Camera-to-world pose for pyrender (camera looks down its -z axis)."""
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


def camera_target_and_distance(first_verts, body_fraction=0.62):
    """Point to aim at and camera distance so the body fills `body_fraction` of the frame height."""
    center = (first_verts.max(0) + first_verts.min(0)) / 2
    height = first_verts[:, 2].max() - first_verts[:, 2].min()
    dist = height / body_fraction / (2 * np.tan(FOV_Y / 2))
    return center, dist


def facing_direction(first_pose):
    """Horizontal facing direction (unit xy vector, z=0) of the root in frame 0.

    SMPL's canonical forward is +z of its y-up frame; the global orientation
    rotates it into the world.
    """
    from scipy.spatial.transform import Rotation as R

    rot = R.from_rotvec(first_pose[:3]).as_matrix()
    forward = rot @ np.array([0.0, 0.0, 1.0])
    forward[2] = 0.0
    return forward / np.linalg.norm(forward)


def render_frames(verts, faces, forward, width=720, height=1280, body_fraction=0.62):
    """Render all frames from one fixed camera in front of the dancer.

    "Front" is the direction the root faces in frame 0; the camera is then held
    fixed for the whole sequence.
    """
    center, dist = camera_target_and_distance(verts[0], body_fraction)
    eye = center + np.array([forward[0], forward[1], 0.0]) * dist
    cam_pose = _look_at(eye, center, np.array([0.0, 0.0, 1.0]))

    scene = pyrender.Scene(bg_color=[0.78, 0.80, 0.82, 1.0], ambient_light=[0.45, 0.45, 0.45])
    camera = pyrender.PerspectiveCamera(yfov=FOV_Y, aspectRatio=width / height)
    scene.add(camera, pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, roughnessFactor=0.8, baseColorFactor=[0.85, 0.62, 0.55, 1.0]
    )

    renderer = pyrender.OffscreenRenderer(width, height)
    frames = []
    mesh_node = None
    try:
        for v in verts:
            if mesh_node is not None:
                scene.remove_node(mesh_node)
            mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
            mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
            color, _ = renderer.render(scene)
            frames.append(color)
    finally:
        renderer.delete()
    return frames
