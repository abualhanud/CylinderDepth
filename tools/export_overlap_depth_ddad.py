#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from dgp.datasets import SynchronizedSceneDataset



POSE_IS_T_WORLD_FROM_CAM = True

USE_OCCLUSION_CHECK = False
Z_TOLERANCE_M = 1.0

OVERLAP_ORDER = ['CAMERA_01', 'CAMERA_05', 'CAMERA_07', 'CAMERA_09', 'CAMERA_08', 'CAMERA_06']

root_dataset = '/mnt/james/data/CylinderDepth/data/ddad/ddad_train_val'
ddad_json = os.path.join(root_dataset, 'ddad.json')
save_root = '/mnt/james/data/CylinderDepth/data/ddad/overlap_depth'

camera_names = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']

index_pkl = '/mnt/james/data/CylinderDepth/data/ddad/index.pkl'

SAVE_MASKS = False
SAVE_PER_NEIGHBOR = False



def neighbors_map(order):
    n = len(order)
    m = {}
    for i, name in enumerate(order):
        m[name] = (order[(i - 1) % n], order[(i + 1) % n])
    return m

NEIGHBORS = neighbors_map(OVERLAP_ORDER)


def quat_wxyz_to_R(q):
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n == 0:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n

    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z

    R = np.array([
        [1 - (yy + zz), xy - wz,     xz + wy],
        [xy + wz,       1 - (xx + zz), yz - wx],
        [xz - wy,       yz + wx,     1 - (xx + yy)]
    ], dtype=np.float32)
    return R


def pose_to_mat(p):
    """
    conversion for Pose / extrinsics
    """

    # ---- CASE 1: matrix() or as_matrix() exists ----
    for attr in ("matrix", "as_matrix", "to_matrix", "to_homogeneous_matrix"):
        if hasattr(p, attr):
            M = getattr(p, attr)
            M = M() if callable(M) else M
            M = np.asarray(M, dtype=np.float32)
            if M.shape == (4, 4):
                return M

    # ---- CASE 2: DGP Pose object (rotation + translation) ----
    if hasattr(p, "rotation") and hasattr(p, "translation"):
        rot = p.rotation
        t = np.asarray(p.translation, dtype=np.float32).reshape(3)

        # handle quaternion rotations
        if hasattr(rot, "w") and hasattr(rot, "x") and hasattr(rot, "y") and hasattr(rot, "z"):
            q = np.array([rot.w, rot.x, rot.y, rot.z], dtype=np.float32)
        elif hasattr(rot, "q"):  # rot.q = [w,x,y,z]
            q = np.asarray(rot.q, dtype=np.float32)
        else:
            raise ValueError("Unknown rotation object:", rot)

        R = quat_wxyz_to_R(q)

        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    # ---- CASE 3: dict ----
    if isinstance(p, dict) and "rotation" in p:
        rot = p["rotation"]
        t = np.asarray(p["translation"], dtype=np.float32).reshape(3)

        if all(k in rot for k in ("w", "x", "y", "z")):
            q = np.array([rot["w"], rot["x"], rot["y"], rot["z"]], dtype=np.float32)
        else:
            raise ValueError("Unknown rotation dict:", rot)

        R = quat_wxyz_to_R(q)
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    raise ValueError("pose_to_mat: could not parse pose:", p)


def intrinsics_to_params(K):
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def rel_T_B_from_A(poseA, poseB):
    TA = pose_to_mat(poseA)
    TB = pose_to_mat(poseB)

    if POSE_IS_T_WORLD_FROM_CAM:
        return (np.linalg.inv(TB) @ TA).astype(np.float32)
    else:
        return (TB @ np.linalg.inv(TA)).astype(np.float32)


def save_mask_png(path, mask_bool):
    out = Image.fromarray(mask_bool.astype(np.uint8) * 255, mode='L')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.save(path, optimize=True)


def overlap_mask_A_seen_by_B(depthA, KA, KB, T_BA):
    H, W = depthA.shape
    fxA, fyA, cxA, cyA = intrinsics_to_params(KA)
    fxB, fyB, cxB, cyB = intrinsics_to_params(KB)

    valid = depthA > 0
    if not np.any(valid):
        return np.zeros_like(depthA, dtype=bool)

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    zA = depthA

    xA = (xs - cxA) * zA / fxA
    yA = (ys - cyA) * zA / fyA

    PA = np.stack([xA, yA, zA, np.ones_like(zA)], axis=0).reshape(4, -1)

    PB = (T_BA @ PA).reshape(4, H, W)

    Xb = PB[0]
    Yb = PB[1]
    Zb = PB[2]

    posZ = Zb > 0

    uB = fxB * (Xb / Zb) + cxB
    vB = fyB * (Yb / Zb) + cyB

    inside = (uB >= 0) & (uB < W) & (vB >= 0) & (vB < H)

    return valid & posZ & inside

def process_frame(data_tuple):
    """
    Input (data, index_info).
    """
    data, index_info = data_tuple

    # Extract frames
    frames = {}
    for entry in data[0]:
        name = entry["datum_name"]
        if name in camera_names:
            frames[name] = entry
    if not frames:
        return  

    any_cam = next(iter(frames.values()))
    t = str(any_cam['timestamp'])
    scene_id = index_info[t]['scene_name']

    for cam in camera_names:
        os.makedirs(os.path.join(save_root, scene_id, 'depth', cam), exist_ok=True)
        if SAVE_MASKS:
            os.makedirs(os.path.join(save_root, scene_id, 'overlap_mask', cam), exist_ok=True)

    K = {}; P = {}; D = {}

    for cam, m in frames.items():
        K[cam] = np.asarray(m['intrinsics'], dtype=np.float32)
        P[cam] = m['pose']
        D[cam] = np.asarray(m['depth'], dtype=np.float32)

    for camA in camera_names:
        if camA not in frames:
            continue

        depthA = D[camA]

        camL, camR = NEIGHBORS[camA]

        mask_union = np.zeros_like(depthA, dtype=bool)

        for camB in (camL, camR):
            if camB not in frames:
                continue

            T_BA = rel_T_B_from_A(P[camA], P[camB])

            mask = overlap_mask_A_seen_by_B(
                depthA=depthA,
                KA=K[camA],
                KB=K[camB],
                T_BA=T_BA
            )

            mask_union |= mask

            if SAVE_PER_NEIGHBOR:
                path_nb = os.path.join(save_root, scene_id, 'overlap_mask', camA, f"{t}__with_{camB}.png")
                save_mask_png(path_nb, mask)

        depth_overlap = np.where(mask_union, depthA, 0).astype(np.float32)

        np.save(os.path.join(save_root, scene_id, 'depth', camA, f"{t}.npy"), depth_overlap)

        if SAVE_MASKS:
            save_mask_png(os.path.join(save_root, scene_id, 'overlap_mask', camA, f"{t}.png"), mask_union)


def main():
    print("Loading dataset...")

    dataset = SynchronizedSceneDataset(
        ddad_json,
        datum_names=('CAMERA_01', 'CAMERA_05', 'CAMERA_06',
                     'CAMERA_07', 'CAMERA_08', 'CAMERA_09', 'lidar'),
        generate_depth_from_datum='lidar',
        split='val'
    )

    with open(index_pkl, 'rb') as f:
        index_info = pickle.load(f)

    # Parallel processing
    print(f"Using {cpu_count()} CPU cores...")
    data_iter = ((data, index_info) for data in dataset)

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_frame, data_iter), total=len(dataset)))


if __name__ == "__main__":
    main()
