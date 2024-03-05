"""
ref  :  https://github.com/hancyran/RepSurf/

"""

import torch
import numpy as np

import os, sys , inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
Gparentdir = os.path.dirname(parentdir)

sys.path.insert(0, parentdir)
sys.path.insert(0, Gparentdir)

import _init_paths



def xyz2sphere(xyz, normalize=True):
    """
    Convert XYZ to Spherical Coordinate

    reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    :param xyz: [N, 3] / [N, G, 3]
    :return: (rho, theta, phi) [N, 3] / [N, G, 3]
    """
    rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
    rho = torch.clamp(rho, min=0)  # range: [0, inf]
    theta = torch.acos(xyz[..., 2, None] / rho)  # range: [0, pi]
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    # check nan
    idx = rho == 0
    theta[idx] = 0

    if normalize:
        theta = theta / np.pi  # [0, 1]
        phi = phi / (2 * np.pi) + .5  # [0, 1]
    out = torch.cat([rho, theta, phi], dim=-1)
    return out


def calc_surface(edge_vec1,edge_vec2) : 


    # Calculate dot product
    dot_product = torch.sum(edge_vec1 * edge_vec2, dim=1)

    # Calculate magnitudes of the vectors
    magnitude1 = torch.norm(edge_vec1, dim=1)
    magnitude2 = torch.norm(edge_vec2, dim=1)

    # Calculate cosine of the angle
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    cosine_angle = torch.clamp(cosine_angle, min=-1, max=1)

    # Calculate the angle in radians
    angle_radians = torch.acos(cosine_angle)
    return angle_radians

def cal_normal(group_xyz, offset, random_inv=False, is_group=False):
    """
    Calculate Normal Vector (Unit Form + First Term Positive)

    :param group_xyz: [N, K=3, 3] / [N, G, K=3, 3]
    """
    edge_vec1 = group_xyz[..., 1, :] - group_xyz[..., 0, :]  # [N, 3]
    edge_vec2 = group_xyz[..., 2, :] - group_xyz[..., 0, :]  # [N, 3]

    nor = torch.cross(edge_vec1, edge_vec2, dim=-1)
    unit_nor = nor / torch.norm(nor, dim=-1, keepdim=True)  # [B, N, 3] / [B, N, G, 3]
    if not is_group:
        pos_mask = (unit_nor[..., 0] > 0).float() * 2. - 1.  # keep x_n positive
    else:
        pos_mask = (unit_nor[..., 0:1, 0] > 0).float() * 2. - 1.
    unit_nor = unit_nor * pos_mask.unsqueeze(-1)

    # batch-wise random inverse normal vector (prob: 0.5)
    if random_inv:
        batch_prob = np.random.rand(offset.shape[0]) < 0.5
        random_mask = []
        sample_offset = [0] + list(offset.cpu().numpy())
        for idx in range(len(sample_offset) - 1):
            sample_mask = torch.ones((sample_offset[idx+1] - sample_offset[idx], 1), dtype=torch.float32)
            if not batch_prob[idx]:
                sample_mask *= -1
            random_mask.append(sample_mask)
        random_mask = torch.cat(random_mask, dim=0).to(unit_nor.device)
        # random_mask = torch.randint(0, 2, (group_xyz.size(0), 1)).float() * 2. - 1.
        # random_mask = random_mask.to(unit_nor.device)
        if not is_group:
            unit_nor = unit_nor * random_mask
        else:
            unit_nor = unit_nor * random_mask.unsqueeze(-1)

    areas = 0.5 * torch.norm(nor, dim=-1, keepdim=True) 
    return unit_nor, areas


def cal_center(group_xyz):
    """
    Calculate Global Coordinates of the Center of Triangle

    :param group_xyz: [N, K, 3] / [N, G, K, 3]; K >= 3
    :return: [N, 3] / [N, G, 3]
    """
    center = torch.mean(group_xyz, dim=-2)
    return center


def cal_area(group_xyz):
    """
    Calculate Area of Triangle

    :param group_xyz: [N, K, 3] / [N, G, K, 3]; K = 3
    :return: [N, 1] / [N, G, 1]
    """
    pad_shape = group_xyz[..., 0, None].shape
    det_xy = torch.det(torch.cat([group_xyz[..., 0, None], group_xyz[..., 1, None], torch.ones(pad_shape)], dim=-1))
    det_yz = torch.det(torch.cat([group_xyz[..., 1, None], group_xyz[..., 2, None], torch.ones(pad_shape)], dim=-1))
    det_zx = torch.det(torch.cat([group_xyz[..., 2, None], group_xyz[..., 0, None], torch.ones(pad_shape)], dim=-1))
    area = torch.sqrt(det_xy ** 2 + det_yz ** 2 + det_zx ** 2).unsqueeze(-1)
    return area


def cal_const(normal, center, is_normalize=True):
    """
    Calculate Constant Term (Standard Version, with x_normal to be 1)

    math::
        const = x_nor * x_0 + y_nor * y_0 + z_nor * z_0

    :param is_normalize:
    :param normal: [N, 3] / [N, G, 3]
    :param center: [N, 3] / [N, G, 3]
    :return: [N, 1] / [N, G, 1]
    """
    const = torch.sum(normal * center, dim=-1, keepdim=True)
    factor = torch.sqrt(torch.Tensor([3])).to(normal.device)
    const = const / factor if is_normalize else const

    return const


def check_nan(normal, center, pos=None):
    """
    Check & Remove NaN in normal tensor

    :param pos: [N, 1]
    :param center: [N, 3]
    :param normal: [N, 3]
    """
    N, _ = normal.shape
    mask = torch.sum(torch.isnan(normal), dim=-1) > 0
    mask_first = torch.argmax((~mask).int(), dim=-1)

    normal_first = normal[None, mask_first].repeat([N, 1])
    normal[mask] = normal_first[mask]
    center_first = center[None, mask_first].repeat([N, 1])
    center[mask] = center_first[mask]

    if pos is not None:
        pos_first = pos[None, mask_first].repeat([N, 1])
        pos[mask] = pos_first[mask]
        return normal, center, pos
    return normal, center


def check_nan_umb(normal, center, pos=None):
    """
    Check & Remove NaN in normal tensor

    :param pos: [N, G, 1]
    :param center: [N, G, 3]
    :param normal: [N, G, 3]
    """
    N, G, _ = normal.shape
    mask = torch.sum(torch.isnan(normal), dim=-1) > 0
    mask_first = torch.argmax((~mask).int(), dim=-1)

    normal_first = normal[torch.arange(N), None, mask_first].repeat([1, G, 1])
    normal[mask] = normal_first[mask]
    center_first = center[torch.arange(N), None, mask_first].repeat([1, G, 1])
    center[mask] = center_first[mask]

    if pos is not None:
        pos_first = pos[torch.arange(N), None, mask_first].repeat([1, G, 1])
        pos[mask] = pos_first[mask]
        return normal, center, pos
    return normal, center
