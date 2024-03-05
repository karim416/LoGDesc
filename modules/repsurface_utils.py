"""
ref  :  https://github.com/hancyran/RepSurf/

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys , inspect
import yaml
import pickle


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
Gparentdir = os.path.dirname(parentdir)

sys.path.insert(0, parentdir)
sys.path.insert(0, Gparentdir)

from modules.polar_recons_utils import xyz2sphere, cal_const, cal_normal, cal_center, check_nan_umb




def resort_points(points, idx):
    """
    Resort Set of points along G dim

    :param points: [N, G, 3]
    :param idx: [N, G]
    :return: [N, G, 3]
    """
    device = points.device
    N, G, _ = points.shape

    n_indices = torch.arange(N, dtype=torch.long).to(device).view([N, 1]).repeat([1, G])
    new_points = points[n_indices, idx, :]

    return new_points



def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.
    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).
    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances



def _fixed_rotate(xyz):
    # y-axis:45deg -> z-axis:45deg
    rot_mat = torch.FloatTensor([[0.5, -0.5, 0.7071], [0.7071, 0.7071, 0.], [-0.5, 0.5, 0.7071]]).to(xyz.device)
    return xyz @ rot_mat

def group_by_umbrella_v2(xyz, new_xyz, offset, new_offset, k=9):
    """
    Group a set of points into umbrella surfaces

    :param xyz: [N, 3]
    :param new_xyz: [N', 3]
    :param k: number of homogenous neighbors
    :return: [N', K-1, 3, 3]
    """

    dist_map = torch.sqrt(pairwise_distance(new_xyz, xyz))  # (B, N, N)
    start_index = 0
    group_idx = dist_map.topk(k=k, dim=1, largest=False)[1]
    # print('group_idx',group_idx.shape)
    # group_idx, _ = from_c.knnquery(k, xyz, new_xyz, offset, new_offset)  # [M, K]

    group_xyz = xyz[group_idx.view(-1).long(), :].view(new_xyz.shape[0], k, 3)  # [M, K, 3]

    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(_fixed_rotate(group_xyz_norm))[..., 2]  # [M, K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [M, K-1]

    # [M, K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz,group_idx,sorted_group_xyz


class UmbrellaSurfaceConstructor_v2(nn.Module):
    """
    inspired from Umbrella Surface Representation Constructor,  we need just the normals 

    """

    def __init__(self, k, random_inv=True):
        super(UmbrellaSurfaceConstructor_v2, self).__init__()
        self.k = k
        self.random_inv = random_inv

    def forward(self, center, offset):
        # umbrella surface reconstruction

        group_xyz,group_idx,group_xyz2 = group_by_umbrella_v2(center, center, offset, offset, k=self.k)  # [N, K-1, 3 (points), 3 (coord.)]


        # normal
        group_normal, areas = cal_normal(group_xyz, offset, random_inv=self.random_inv, is_group=True) # [N,K-1,3]

        # coordinate
        group_center = cal_center(group_xyz)

        # coordinate
        group_center2 = cal_center(group_xyz2)

        # polar
        group_polar = xyz2sphere(group_center)

        # surface position
        group_pos = cal_const(group_normal, group_center)

        group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)


        weights = F.softmax(areas/(10**-4),1)


        weighted_normals = group_normal*weights


        point_normal = torch.sum(weighted_normals,1)

        return  group_idx, point_normal

