#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

from dgp.datasets import SynchronizedSceneDataset


POSE_IS_T_WORLD_FROM_CAM = True

USE_OCCLUSION_CHECK = False
Z_TOLERANCE_M = 1.0 

OVERLAP_ORDER = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']

root_dataset = '/home/abualhanud/data/abualhanud/full_data'
ddad_json = os.path.join(root_dataset, 'ddad.json')
save_root = '/home/abualhanud/data/abualhanud/full/overlap_depth'  
depth_gt_path = "/home/abualhanud/data/abualhanud/full/depth_lidar_gt_surround"

camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']


# Index (timestamp -> scene_name)
index_pkl = "/home/abualhanud/test/REPOS/CVCDepth_2/dataset/nusc/info_val.pkl"

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
    """q = [w, x, y, z] -> 3x3 rotation."""
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n == 0:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    R = np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy        ],
        [xy + wz,         1.0 - (xx + zz), yz - wx        ],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)]
    ], dtype=np.float32)
    return R

def pose_to_mat(pose_obj):
    for attr in ('matrix', 'as_matrix', 'to_homogeneous_matrix', 'to_transformation_matrix'):
        if hasattr(pose_obj, attr):
            M = getattr(pose_obj, attr)
            M = M() if callable(M) else M
            M = np.asarray(M, dtype=np.float32)
            if M.shape == (4, 4):
                return M

    if isinstance(pose_obj, dict):
        if 'quat' in pose_obj and 'tvec' in pose_obj:
            rot = pose_obj['quat']
            t = np.asarray(pose_obj['tvec'], dtype=np.float32).reshape(3)
            if isinstance(rot, dict):
                if {'w','x','y','z'}.issubset(rot.keys()):
                    q = np.array([rot['w'], rot['x'], rot['y'], rot['z']], dtype=np.float32)
                elif {'x','y','z','w'}.issubset(rot.keys()):
                    q = np.array([rot['w'], rot['x'], rot['y'], rot['z']], dtype=np.float32)
                else:
                    raise ValueError("Unknown rotation dict keys")
            else:
                q = np.array([rot[0], rot[1], rot[2], rot[3]], dtype=np.float32)
            R = quat_wxyz_to_R(q)
            M = np.eye(4, dtype=np.float32)
            M[:3,:3] = R
            M[:3,3] = t
            return M

    if hasattr(pose_obj, 'quat') and hasattr(pose_obj, 'tvec'):
        rot = pose_obj.rotation
        t = np.asarray(pose_obj.tvec, dtype=np.float32).reshape(3)
        if all(hasattr(rot, k) for k in ('w','x','y','z')):
            q = np.array([rot.w, rot.x, rot.y, rot.z], dtype=np.float32)
        elif hasattr(rot, 'q'):
            q = np.array(rot.q, dtype=np.float32)
        else:
            q = np.array([rot.qw, rot.qx, rot.qy, rot.qz], dtype=np.float32)
        R = quat_wxyz_to_R(q)
        M = np.eye(4, dtype=np.float32)
        M[:3,:3] = R
        M[:3,3] = t
        return M

    raise ValueError("pose_to_mat: couldn't parse the pose/extrinsics object.")

def intrinsics_to_params(K):
    fx = float(K[0,0]); fy = float(K[1,1]); cx = float(K[0,2]); cy = float(K[1,2])
    return fx, fy, cx, cy

def rel_T_B_from_A(poseA, poseB):
    TA = pose_to_mat(poseA)
    TB = pose_to_mat(poseB)
    if POSE_IS_T_WORLD_FROM_CAM:
        T = np.linalg.inv(TB) @ TA
    else:
        T = TB @ np.linalg.inv(TA)
    return T.astype(np.float32)

def overlap_mask_A_seen_by_B(depthA, KA, KB, T_BA, depthB=None, z_tol=Z_TOLERANCE_M):
    H, W = depthA.shape
    fxA, fyA, cxA, cyA = intrinsics_to_params(KA)
    fxB, fyB, cxB, cyB = intrinsics_to_params(KB)

    valid = depthA > 0
    if not np.any(valid):
        return np.zeros((H, W), dtype=bool)

    vs, us = np.nonzero(valid)          
    zA = depthA[vs, us]                

    xA = (us - cxA) * zA / fxA
    yA = (vs - cyA) * zA / fyA
    ones = np.ones_like(zA, dtype=np.float32)
    PA = np.stack([xA, yA, zA, ones], axis=0)  # (4, N)

    # Transform to B
    PB = T_BA @ PA
    Xb, Yb, Zb = PB[0], PB[1], PB[2]

    posZ = Zb > 0

    # Project to B
    uB = fxB * (Xb / Zb) + cxB
    vB = fyB * (Yb / Zb) + cyB

    inside = (uB >= 0) & (uB <= (W - 1)) & (vB >= 0) & (vB <= (H - 1))
    ok = posZ & inside

    if USE_OCCLUSION_CHECK and (depthB is not None):
        uBi = np.clip(np.round(uB[ok]).astype(np.int32), 0, W - 1)
        vBi = np.clip(np.round(vB[ok]).astype(np.int32), 0, H - 1)
        zB_img = depthB[vBi, uBi]
        z_ok = (zB_img > 0) & (np.abs(zB_img - Zb[ok]) <= z_tol)
        ok_idx = np.where(ok)[0]
        ok_mask = np.zeros_like(ok, dtype=bool)
        ok_mask[ok_idx[z_ok]] = True
        ok = ok_mask

    maskA = np.zeros((H, W), dtype=bool)
    maskA[vs[ok], us[ok]] = True
    return maskA

def save_mask_png(path, mask_bool):
    img = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode='L')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, optimize=True)

def main():

    with open('/home/abualhanud/test/REPOS/CVCDepth_2/dataset/nusc/val_vf.txt', 'r') as f:
        filenames = f.readlines()
    dataset = filenames
    print("Dataset length:", len(dataset))

    with open(index_pkl, 'rb') as f:
        index_info = pickle.load(f)

    os.makedirs(save_root, exist_ok=True)

    for data in tqdm(dataset):
        info = index_info[data.strip()]
        frames = {}
        for name in info:
            if name in camera_names:
                frames[name] = info[name]

        if not frames:
            continue

        for cam in camera_names:
            os.makedirs(os.path.join(save_root,"samples", cam), exist_ok=True)
            if SAVE_MASKS:
                os.makedirs(os.path.join(save_root,"overlap_mask","samples", cam), exist_ok=True)

        K = {}
        P = {}
        D = {}
        F = {}
        for cam, m in frames.items():
            K[cam] = np.asarray(m['intrinsics'], dtype=np.float32)
            P[cam] = m['extrinsics'] # swap to m['extrinsics'] if you prefer those for relative transforms
            D[cam] = np.load(os.path.join(depth_gt_path, m['rgb_filenames'][0].replace('jpg', 'npy')))
            F[cam] = m['rgb_filenames'][0]

        for camA in camera_names:
            if camA not in frames:
                continue

            depthA = D[camA]
            H, W = depthA.shape

            camL, camR = NEIGHBORS[camA]

            mask_by_neighbor = {}
            for camB in (camL, camR):
                if camB not in frames:
                    continue
                T_BA = rel_T_B_from_A(P[camA], P[camB])
                mask_AB = overlap_mask_A_seen_by_B(
                    depthA=depthA,
                    KA=K[camA],
                    KB=K[camB],
                    T_BA=T_BA,
                    depthB=D[camB] if USE_OCCLUSION_CHECK else None,
                    z_tol=Z_TOLERANCE_M
                )
                mask_by_neighbor[camB] = mask_AB

            if not mask_by_neighbor:
                depth_overlap = np.zeros_like(depthA, dtype=np.float32)
                np.save(os.path.join(save_root, scene_id, 'depth', camA, f"{t}.npy"), depth_overlap)
                if SAVE_MASKS:
                    save_mask_png(os.path.join(save_root, scene_id, 'overlap_mask', camA, f"{t}.png"),
                                  np.zeros((H, W), dtype=bool))
                continue

            # Union of neighbor masks
            mask_union = np.zeros((H, W), dtype=bool)
            for msk in mask_by_neighbor.values():
                mask_union |= msk

            depth_overlap = np.where(mask_union, depthA, 0.0).astype(np.float32)

            # Save depth
            depth_path = os.path.join(save_root,F[camA].replace('jpg', 'npy'))
            np.save(depth_path, depth_overlap)

            # Save masks
            if SAVE_MASKS:
                union_path = os.path.join(save_root,"overlap_mask",F[camA].replace('jpg', 'png'))
                save_mask_png(union_path, mask_union)

                if SAVE_PER_NEIGHBOR:
                    for camB, msk in mask_by_neighbor.items():
                        nb_path = os.path.join(save_root, scene_id, 'overlap_mask', camA, f"{t}__with_{camB}.png")
                        save_mask_png(nb_path, msk)

if __name__ == "__main__":
    main()
