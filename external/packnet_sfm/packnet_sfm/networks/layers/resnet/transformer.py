import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
import sys
import timm
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import trunc_normal_
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from typing import Optional



from scipy.spatial.distance import cdist

import hashlib
import os
class TensorCache:
    def __init__(self, cache_dir='tensor_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _tensor_hash(self, *tensors):
        hasher = hashlib.md5()
        for tensor in tensors:
            tensor_bytes = tensor.cpu().numpy().tobytes()
            hasher.update(tensor_bytes)
        return hasher.hexdigest()

    def _get_cache_path(self, tensor_hash):
        return os.path.join(self.cache_dir, f'{tensor_hash}.pt')

    def get(self, *tensors):
        tensor_hash = self._tensor_hash(*tensors)
        cache_path = self._get_cache_path(tensor_hash)
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location=tensors[0].device)
        return None

    def set(self, value, *tensors):
        tensor_hash = self._tensor_hash(*tensors)
        cache_path = self._get_cache_path(tensor_hash)
        torch.save(value, cache_path)

tensor_cache = TensorCache()


def unproject(depth, K, T_cam_to_ego, depth_threshold=500):
    """
    Unprojects depth maps to 3D points.

    Args:
        depth: (B, N, 1, H, W)
        K: (B, N, 3, 3)
        T_cam_to_ego: (B, N, 4, 4)
        depth_threshold: float, max depth to keep

    Returns:
        points_ego: (B, N*H*W, 3)
    """
    B, N = K.shape[0], K.shape[1]
    _, _, _, H, W = depth.shape
    depth = depth.reshape(B, N, 1, H, W)
    device = depth.device

    # Create meshgrid (H, W)
    ys, xs = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device))
    # (H, W) → (1, 1, H, W)
    xs = xs[None, None, ...].float()
    ys = ys[None, None, ...].float()

    # Expand meshgrid to (B, N, H, W)
    xs = xs.expand(B, N, -1, -1)
    ys = ys.expand(B, N, -1, -1)

    # Get depth (B, N, H, W)
    depth = depth.squeeze(2)

    # Intrinsics
    fx = K[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = K[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = K[:, :, 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = K[:, :, 1, 2].unsqueeze(-1).unsqueeze(-1)

    # Project to 3D in camera frame
    X = (xs - cx) / fx * depth
    Y = (ys - cy) / fy * depth
    Z = depth

    # (B, N, H, W, 3)
    points_cam = torch.stack((X, Y, Z), dim=-1)
    points_cam = points_cam.view(B, N, -1, 3)  # (B, N, H*W, 3)

    # Filter invalid depths
    valid = (Z > 0) & (Z <= depth_threshold)  # (B, N, H, W)
    valid = valid.view(B, N, -1)              # (B, N, H*W)

    # Homogeneous coordinates (B, N, H*W, 4)
    ones = torch.ones_like(points_cam[..., :1])
    points_h = torch.cat([points_cam, ones], dim=-1)

    # Apply T_cam_to_ego:
    T = T_cam_to_ego  # (B, N, 4, 4)
    points_ego = torch.matmul(points_h, T.transpose(2, 3))  # (B, N, H*W, 4)
    points_ego = points_ego[..., :3]  # (B, N, H*W, 3)

    # Mask out invalid points
    mask = valid.unsqueeze(-1).expand_as(points_ego)
    points_ego[~mask] = 0.0

    # Reshape to (B, N*H*W, 3)
    points_ego = points_ego.view(B, N * H * W, 3)

    return points_ego

def project_to_cylinder_coords(points, R_fixed=None, center=None):
    """
    Converts 3D points to cylindrical coordinates (angle, height).

    Args:
        points: (B, N, 3) tensor where N = 6 * H * W
        R_fixed: float or None fixed cylinder radius (uses median if None)
        center: (3,) tensor or array default: [0, 0, 1.5]

    Returns:
        cyl_coords: (B, 6, H*W, 2) tensor angle (radians), height
    """
    B, N, _ = points.shape
    assert N % 6 == 0, "N must be divisible by 6"
    HW = N // 6

    device = points.device
    if center is None:
        center = torch.tensor([0.0, 0.0, 1.5], device=device)
    center = center.view(1, 1, 3).to(device)  # (1, 1, 3)

    # Compute vector from center to each point
    v = points - center  # (B, N, 3)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]

    # Compute horizontal radius
    r = torch.sqrt(x**2 + y**2)  # (B, N)

    # Compute or use provided fixed radius
    if R_fixed is None:
        R_fixed = torch.median(r[r > 0]).item()

    r_safe = torch.where(r == 0, torch.tensor(1e-6, device=device), r)
    t = R_fixed / r_safe  # (B, N)
    t = t.unsqueeze(-1)   # (B, N, 1)

    # Outer projection
    proj = center - t * v  # (B, N, 3)

    # Convert to cylindrical
    x_proj = proj[..., 0]
    y_proj = proj[..., 1]
    z_proj = proj[..., 2]

    theta = torch.atan2(y_proj - center[..., 1], x_proj - center[..., 0])  # (B, N)
    height = z_proj - center[..., 2]  # (B, N)

    cyl_coords = torch.stack([theta, height], dim=-1)  # (B, N, 2)
    cyl_coords = cyl_coords.view(B, 6, HW, 2)

    return cyl_coords




#### GPU version of gaussian similarity
def gaussian_similarity(
    cylindrical_coords: torch.Tensor,
    cov_matrix: Optional[torch.Tensor] = None,   # shape (2, 2); default = 0.002 * I
    truncation_range: float = 1.2,
    radius: float = 1.0,
) -> torch.Tensor:
    """
    Compute Gaussian similarities on a cylinder

    Args:
        cylindrical_coords: Tensor with last dim == 2 containing [theta (rad), z].
        cov_matrix: Optional (2,2) covariance matrix. If None, uses 0.002 * I.
        truncation_range: Distances > this are masked to 0 similarity.
        radius: Cylinder radius to convert dtheta to arc length.

    Returns:
        similarities: (B, N, N) truncated Gaussian similarities.
    """
    x = cylindrical_coords
    device = x.device
    dtype = x.dtype if x.dtype.is_floating_point else torch.float32
    x = x.to(dtype)

    if x.dim() == 3 and x.size(-1) == 2:
        coords = x
        B, N = coords.shape[0], coords.shape[1]
    else:
        B = x.shape[0]
        coords = x.view(B, -1, 2)
        N = coords.shape[1]

    if cov_matrix is None:
        cov_matrix = torch.eye(2, dtype=dtype, device=device) * 0.002
    else:
        cov_matrix = cov_matrix.to(device=device, dtype=dtype)

    inv_cov = torch.linalg.inv(cov_matrix)  # (2,2)

    # theta, z
    theta = coords[..., 0]  # (B, N)
    z = coords[..., 1]      # (B, N)

    # Pairwise wrapped angle diffs in [-pi, pi)
    dtheta = theta.unsqueeze(2) - theta.unsqueeze(1)  # (B, N, N)
    dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi

    # Arc-length
    d_arc = radius * dtheta                          # (B, N, N)
    dz = z.unsqueeze(2) - z.unsqueeze(1)            # (B, N, N)


    diffs = torch.stack([d_arc, dz], dim=-1)        # (B, N, N, 2)

    # Mahalanobis distance:
    tmp = torch.matmul(diffs, inv_cov)              # (B, N, N, 2)
    dist2 = (tmp * diffs).sum(dim=-1).clamp_min_(0) # (B, N, N)
    distances = torch.sqrt(dist2)

    # Gaussian similarity
    similarities = torch.exp(-0.5 * dist2)          # (B, N, N)

    # Truncate
    if truncation_range is not None:
        mask = distances <= truncation_range
        similarities = similarities * mask

    return similarities


list_coord = []

class CVT(nn.Module):
    def __init__(self,input_channel,downsample_ratio=2, iter_num=4):
        super().__init__()
        self.embed_dim = input_channel
        self.iter_num = iter_num
        self.self_block = []
        for i in range(iter_num):
            self.self_block.append(Self_Block(dim=self.embed_dim, num_heads=8))
        self.decoder = nn.ModuleList(list(self.self_block))
        
        self.postional_embed_pixel = PositionEmbeddingSine_pixel(num_pos_feats=self.embed_dim)
 
        self.downsample_ratio = downsample_ratio
        self.sep_conv = SeparableConv2d(in_channels=self.embed_dim,out_channels=self.embed_dim,stride=downsample_ratio)
        if downsample_ratio==16:
            kernel_size=3
            dilation = 7
            output_padding = 1
        elif downsample_ratio==8:
            kernel_size=3
            dilation = 3
            output_padding = 1
        elif downsample_ratio==4:
            kernel_size=3
            dilation = 1
            output_padding = 1
        elif downsample_ratio==2:
            kernel_size=1
            dilation = 1
            output_padding = 1
        elif downsample_ratio==1:
            kernel_size=1
            dilation = 1
            output_padding = 0
        self.sep_deconv = SeparableDeConv2d(in_channels=self.embed_dim,out_channels=self.embed_dim,stride=downsample_ratio, kernel_size = kernel_size, 
            dilation =dilation, output_padding=output_padding)
     
        self.list_embed = []

    def forward(self, x, intrinsics, extrinsics, depth=None, attention=False, org_img_size=None, mask=None): # B, N, C, H, W
        B, N, C, H, W = x.shape
        self.sep_conv.to(x.device)
        x = self.sep_conv(x.view(-1,C,H,W)).view(B,N,C,H//self.downsample_ratio,W//self.downsample_ratio)
        scaled_intrinsics = intrinsics[:, :3, :3].clone()
        scale_factor_x = x.shape[3] / org_img_size[0]  # Width scale factor
        scale_factor_y = x.shape[4] / org_img_size[1]  # Height scale factor

        scaled_intrinsics[:, 0, :] *= scale_factor_x  # Scale fx, cx
        scaled_intrinsics[:, 1, :] *= scale_factor_y  # Scale fy, cy
        scaled_intrinsics = scaled_intrinsics.reshape(B, 6, 3, 3)
        extrinsics = extrinsics.reshape(B, 6, 4, 4)

        if attention:
            points_3D = unproject(depth, scaled_intrinsics, extrinsics) # B, N*H*W, 3
            cylinderical_coords = project_to_cylinder_coords(points_3D, R_fixed=1) # B, 6, H*W, 2
            gaussian_similarity_matrix = gaussian_similarity(cylinderical_coords)

        for i in range(1):
            # reshape 
            x = x.permute(0,1,3,4,2).reshape(B,-1,C)
            if attention:
                gaussian_similarity_matrix = gaussian_similarity_matrix/gaussian_similarity_matrix.sum(dim=-1, keepdim=True)
            else: 
                gaussian_similarity_matrix = None
            self.self_block[i].to(x.device)
            x = self.self_block[i](x, x_pos=None, spatial_attn=gaussian_similarity_matrix).reshape(B, N, H//self.downsample_ratio, W//self.downsample_ratio, C).permute(0, 1, 4, 2, 3)
        self.sep_deconv.to(x.device)
        x = self.sep_deconv(x.view(-1, C, H//self.downsample_ratio, W//self.downsample_ratio))
        x = x.view(B,N,C,H,W)

        return x



class Self_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Self_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = timm.models.layers.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_pos, spatial_attn):
        x = x + self.drop_path(self.attn(self.norm1(x),x_pos, spatial_attn)) # for now we are not using the residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.weighting_spatial = 1

    def forward(self, x,x_pos, spatial_attn=None):
        B, N, C = x.shape
        v_vector =  self.v_linear(x)
        if spatial_attn is None:
            spatial_attn = torch.eye(N,N).unsqueeze(0).expand(B,-1,-1).to(x.device) 
        x = torch.bmm(spatial_attn, v_vector)
        x = self.proj_drop(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=4,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class SeparableDeConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=4,padding=0,dilation=4,bias=False,output_padding=0):
        super(SeparableDeConv2d,self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels,padding=padding,dilation=dilation,
            output_padding=output_padding)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class PositionEmbeddingSine_pixel(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # pdb.set_trace()
        # x = x.permute(0,2,3,1)
        mask = torch.zeros_like(x[:,0,:,:]).bool()
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        # pdb.set_trace()
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
