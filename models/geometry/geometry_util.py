# Copyright (c) 2023 42dot. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
# from pytorch3d.transforms import axis_angle_to_matrix
from collections import Counter

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    sin_half_angles_over_angles = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    return torch.cat(
        [torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1
    )

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle: torch.Tensor, fast: bool = False) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if not fast:
        return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

    shape = axis_angle.shape
    device, dtype = axis_angle.device, axis_angle.dtype

    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True).unsqueeze(-1)

    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    zeros = torch.zeros(shape[:-1], dtype=dtype, device=device)
    cross_product_matrix = torch.stack(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
    ).view(shape + (3,))
    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

    identity = torch.eye(3, dtype=dtype, device=device)
    angles_sqrd = angles * angles
    angles_sqrd = torch.where(angles_sqrd == 0, 1, angles_sqrd)
    return (
        identity.expand(cross_product_matrix.shape)
        + torch.sinc(angles / torch.pi) * cross_product_matrix
        + ((1 - torch.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
    )



def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1
        
def vec_to_matrix(rot_angle, trans_vec, invert=False):
    """
    This function transforms rotation angle and translation vector into 4x4 matrix.
    """
    # initialize matrices
    b, _, _ = rot_angle.shape
    R_mat = torch.eye(4).repeat([b, 1, 1]).to(device=rot_angle.device)
    T_mat = torch.eye(4).repeat([b, 1, 1]).to(device=rot_angle.device)

    R_mat[:, :3, :3] = axis_angle_to_matrix(rot_angle).squeeze(1)
    t_vec = trans_vec.clone().contiguous().view(-1, 3, 1)

    if invert == True:
        R_mat = R_mat.transpose(1,2)
        t_vec = -1 * t_vec

    T_mat[:, :3,  3:] = t_vec

    if invert == True:
        P_mat = torch.matmul(R_mat, T_mat)
    else :
        P_mat = torch.matmul(T_mat, R_mat)
    return P_mat


class Projection(nn.Module):
    """
    This class computes projection and reprojection function. 
    """
    def __init__(self, batch_size, height, width, device):
        super().__init__()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        
        # initialize img point grid
        img_points = np.meshgrid(range(width), range(height), indexing='xy')
        img_points = torch.from_numpy(np.stack(img_points, 0)).float()
        img_points = torch.stack([img_points[0].view(-1), img_points[1].view(-1)], 0).repeat(batch_size, 1, 1)
        img_points = img_points.to(device)
        
        self.to_homo = torch.ones([batch_size, 1, width*height]).to(device)
        self.homo_points = torch.cat([img_points, self.to_homo], 1)

    def backproject(self, invK, depth):
        """
        This function back-projects 2D image points to 3D.
        """
        depth = depth.view(self.batch_size, 1, -1)

        points3D = torch.matmul(invK[:, :3, :3], self.homo_points)
        points3D = depth*points3D
        return torch.cat([points3D, self.to_homo], 1)
    
    def reproject(self, K, points3D, T):
        """
        This function reprojects transformed 3D points to 2D image coordinate.
        """
        # project points 
        points2D = (K @ T)[:,:3, :] @ points3D
        # normalize projected points for grid sample function
        norm_points2D = points2D[:, :2, :]/(points2D[:, 2:, :] + 1e-7)
        norm_points2D = norm_points2D.view(self.batch_size, 2, self.height, self.width)
        norm_points2D = norm_points2D.permute(0, 2, 3, 1)

        norm_points2D[..., 0 ] /= self.width - 1
        norm_points2D[..., 1 ] /= self.height - 1
        norm_points2D = (norm_points2D-0.5)*2
        return norm_points2D

    def reproject_unnormed(self, K, points3D, T):
        """
        This function reprojects transformed 3D points to 2D image coordinate.
        """

        # project points
        points2D = (K @ T)[:, :3, :] @ points3D

        # normalize projected points for grid sample function
        points2D[:,:2,:]/=(points2D[:, 2:, :] + 1e-7)
        norm_points2D = points2D
        norm_points2D = norm_points2D.view(self.batch_size, 3, self.height, self.width)
        norm_points2D = norm_points2D.permute(0, 2, 3, 1)

        bs = points2D.shape[0]
        aaaas = []
        for b in range(bs):
            local_norm_points2D = norm_points2D[b].reshape(-1, 3)
            zz = local_norm_points2D[:, 2:]
            # local_norm_points2D = local_norm_points2D.detach().clone()
            local_norm_points2D[:, 0] = torch.round(local_norm_points2D[:, 0]) - 1
            local_norm_points2D[:, 1] = torch.round(local_norm_points2D[:, 1]) - 1
            val_inds = (local_norm_points2D[:, 0] >= 0) & (local_norm_points2D[:, 1] >= 0)
            val_inds = val_inds & (local_norm_points2D[:, 0] < self.width) & (local_norm_points2D[:, 1] < self.height)
            local_norm_points2D = local_norm_points2D[val_inds, :]
            zz = zz[val_inds, :]
            aaa = torch.zeros((self.height, self.width), device=points3D.device, dtype=points3D.dtype)
            aaa[local_norm_points2D[:, 1].long(), local_norm_points2D[:, 0].long()] = zz[:, 0]

            aaaas.append(aaa.unsqueeze(0))
        aaaas = torch.stack(aaaas)

        return aaaas

    def reproject_transform(self,K, points3D, T):
        points2D = (K @ T)[:, :3, :] @ points3D

        # normalize projected points for grid sample function
        points2D = points2D
        points2D = points2D.view(self.batch_size, 3, self.height, self.width)
        return points2D

    def forward(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject(bp_invK, depth)

        pix_coords = self.reproject(rp_K, cam_points, T)
        return pix_coords

    def get_unnormed_projects(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject(bp_invK, depth)

        pix_coords = self.reproject_unnormed(rp_K, cam_points, T)
        return pix_coords

    def transform_depth(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject(bp_invK, depth)

        pix_coords = self.reproject_transform(rp_K, cam_points, T)
        return pix_coords

