from __future__ import division
from __future__ import print_function

import time
import random
import numpy as np
import open3d as o3d
import torch
#import torch_cluster
import math
from scipy.spatial import cKDTree
from scipy import stats
from sklearn.neighbors import KernelDensity

def check_sign(pt, ptnn, np_hat):


    ptnn_pt = ptnn.transpose(1,0,2)  - pt[..., np.newaxis].transpose(1,0,2)   
    ptnn_pt = ptnn_pt.transpose(1,0,2)   
    scal = np.einsum('bc,cbn->nb',np_hat, -ptnn_pt)  # (NN, 10)

    minus = np.where(np.sum(scal,axis=0)<0)[0]
    # print('min',minus.shape)
    zp = np.copy(np_hat)
    zp[minus] = -np_hat[minus]
    zp = zp.T / np.linalg.norm(zp.T,axis=0)
    # print('zp',zp.shape)
    return zp.T,minus


def patch_estimate_lrf(pt, ptnn, patch_kernel):


    ptnn_pt = ptnn.transpose(1,0,2)  - pt[..., np.newaxis].transpose(1,0,2)   
    ptnn_cov = (1.0 / ptnn.shape[-1]) * np.matmul(ptnn_pt, ptnn_pt.transpose(0, 2, 1),dtype=np.float64)


    eigvals, eigvecs = np.linalg.eig(ptnn_cov) # returns (Number of subsamples,3) et (10,3,3)

    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigvals, axis=1)[:, ::-1]
    sorted_eigvals = np.take_along_axis(eigvals, sorted_indices, axis=1)

    # Compute measures
    planarity = (sorted_eigvals[:, 1] - sorted_eigvals[:, 2]) / sorted_eigvals[:, 0]
    anisotropy = (sorted_eigvals[:, 0] - sorted_eigvals[:, 2]) / sorted_eigvals[:, 0]
    omnivariance = np.prod(sorted_eigvals, axis=1)**(1/3)

    smallest_eigval_idx = np.argmin(eigvals,axis=-1) # (10,)

    np_hat = eigvecs[np.arange(eigvecs.shape[0]), :, smallest_eigval_idx] # (Number of subsamples ,3)

    ptnn_pt = ptnn_pt.transpose(1,0,2)   
    scal = np.einsum('bc,cbn->nb',np_hat, -ptnn_pt)  # (NN, 10)

    minus = np.where(np.sum(scal,axis=0)<0)[0]
    zp = np.copy(np_hat.T)
    zp[...,minus] = -np_hat.T[...,minus]
    
    zp /= np.linalg.norm(zp,axis=0)
    zp = zp.T #(Np,3)

    ptnn_pt_zp = np.einsum('cbn,bck->bnk',ptnn_pt, zp.T[..., np.newaxis].transpose(1,0,2))  # (10,NN-1, 1)
    res = np.transpose(ptnn_pt_zp * zp[:, np.newaxis],(2,0,1))
    v = ptnn_pt - res # (3, 10, NN-1)
    alpha = (patch_kernel - np.linalg.norm(-ptnn_pt, axis=0))**2     # (10, NN-1)
    beta = ptnn_pt_zp.squeeze()**2  # (10, NN-1)
    v_alpha_beta = np.einsum('cbn,bkn->cbk',v, (alpha * beta)[:, np.newaxis]).transpose(1,0,2)  # (N,3,1)

    indices = np.where(np.linalg.norm(v_alpha_beta,axis=1) < 1e-4)[0]
    xp = v_alpha_beta / np.expand_dims(np.linalg.norm(v_alpha_beta,axis=1),1)
    xp = xp.squeeze()  # (Np,3,1) -> (Np,3)

    if len(indices)>0 : 
        xp_i = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        xp_i = np.tile(xp_i[:, np.newaxis], (1, ptnn.shape[1])) # (3,10,1)
        v_alpha_beta[indices,: , :]= np.expand_dims(xp_i[:,indices],-1).transpose(1,0,2) # (3,N,1)
        xp_i = xp_i.T # (Np,3,1)
        xp[indices] = xp_i[indices]

    yp = np.cross(zp, xp) # (N,3 )
    indicesy = np.where(np.linalg.norm(yp,axis=1) < 1e-8)[0]
    yp /= np.expand_dims(np.linalg.norm(yp,axis=1),1)

    # init identity
    xp_r = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    xp_r = np.tile(xp_r[:, np.newaxis], (1, ptnn.shape[1])) # (3,10,1)
    xp_r = xp_r.T # (3,10)


    yp_r = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    yp_r = np.tile(yp_r[:, np.newaxis], (1, ptnn.shape[1])) # (3,10,1)
    yp_r = yp_r.T

    zp_r = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    zp_r = np.tile(zp_r[:, np.newaxis], (1, ptnn.shape[1])) # (3,10,1)
    zp_r = zp_r.T # (3,10)


    if len(indicesy)>0 : 

        yp[indicesy] = yp_r[indicesy]
        xp[indicesy] = xp_r[indicesy]
        zp[indicesy] = zp_r[indicesy]

    xp = np.cross(yp, zp)# (Np,3 )    
    indicesx2 = np.where(np.linalg.norm(xp,axis=1) < 1e-8)[0]
    xp /= np.expand_dims(np.linalg.norm(xp,axis=1),1)

    if len(indicesx2)>0 : 
        xp[indicesx2] = xp_r[indicesx2]
        yp[indicesx2] = yp_r[indicesx2]
        zp[indicesx2] = zp_r[indicesx2]

    lRg = np.stack((xp, yp, zp), axis=1).transpose(0,2,1)
    
    test_ind = np.where(np.isnan(np.sum(lRg,axis=(0,1))))[0]

    iden = np.identity(3, dtype=np.float32)
    iden = np.tile(iden[..., np.newaxis], (1,1, ptnn.shape[1]))
    if len(test_ind)>0:
        lRg[...,test_ind] = iden[...,test_ind]
    return np.asarray(lRg, dtype=np.float32) , planarity, omnivariance, anisotropy



def cosine_similarity(a, b):
    dot_product = np.sum(a * b, axis=1)
    norm_a = np.sqrt(np.sum(a ** 2, axis=1))
    norm_b = np.sqrt(np.sum(b ** 2, axis=1))
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def rods_rotat_formula(a, b):
    #  b (n,3)
    B, _ = a.shape
    c = np.cross(a, b)
    theta = np.arccos(cosine_similarity(a,b))

    c = c / np.linalg.norm(c, axis=1)[:, np.newaxis]
    one = np.ones((B, 1, 1))
    zero = np.zeros((B, 1, 1))
    a11 = zero
    a12 = -c[:, 2][:, np.newaxis, np.newaxis]
    a13 = c[:, 1][:, np.newaxis, np.newaxis]
    a21 = c[:, 2][:, np.newaxis, np.newaxis]
    a22 = zero
    a23 = -c[:, 0][:, np.newaxis, np.newaxis]
    a31 = -c[:, 1][:, np.newaxis, np.newaxis]
    a32 = c[:, 0][:, np.newaxis, np.newaxis]
    a33 = zero
    Rx = np.concatenate(
        (np.concatenate((a11, a12, a13), axis=2),
         np.concatenate((a21, a22, a23), axis=2),
         np.concatenate((a31, a32, a33), axis=2)),
        axis=1)
    I = np.eye(3)
    R = I[np.newaxis, :, :] + np.sin(theta)[:, np.newaxis, np.newaxis] * Rx + (1 - np.cos(theta))[:, np.newaxis, np.newaxis] * np.matmul(Rx, Rx)
    return np.transpose(R, axes=(0,-1, -2))

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = np.tile(np.arange(B).reshape(view_shape),repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def furthest_point_sample(pts, max_points):
    """
    Input:
        pts: pointcloud data, [B, N, C]
        K: number of samples
    Return:
        (B, K, 3)
    """
    K = max_points
    pts = np.expand_dims(pts, axis=0)
    B, N, C = pts.shape
    centroids = np.zeros((B, K), dtype=int)
    distance = np.ones((B, N), dtype=int) * 1e10
    # np.random.seed(0)
    farthest = np.random.randint(0, N, (B,))
    # farthest = np.array([0,0])
    batch_indices = np.arange(B)
    for i in range(K):
        centroids[:, i] = farthest
        centroid = pts[batch_indices, farthest, :].reshape(B, 1, 3)
        dist = np.sum((pts - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=-1)

    return centroids[0]

class KNNSearch(object):

    def __init__(self, points):
        # points: (N, 3)
        self.points = np.asarray(points, dtype=np.float32)
        self.kdtree = cKDTree(points)

    def query(self, kpts, num_samples):
        # kpts: (K, 3)
        kpts = np.asarray(kpts, dtype=np.float32)
        nndists, nnindices = self.kdtree.query(kpts, k=num_samples, workers=8)
        assert len(kpts) == len(nnindices)
        return nnindices  # (K, num_samples)

    def query_ball(self, kpt, radius):
        # kpt: (3, )
        kpt = np.asarray(kpt, dtype=np.float32)
        assert kpt.ndim == 1
        nnindices = self.kdtree.query_ball_point(kpt, radius, workers=8)  # list
        return nnindices


def to_o3d_point_cloud(points, normals=None):
    """ 
    Args:
        points (np.array): (N, 3)
        normals (np.array): (N, 3)

    Returns:
        o3d.geometry.PointCloud:
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def to_o3d_feature(feats):
    """
    Args:
        feats (np.array): (N, D)

    Returns:
        o3d.pipelines.registration.Feature:
    """
    f = o3d.pipelines.registration.Feature()
    f.resize(feats.shape[1], feats.shape[0])
    f.data = np.asarray(feats, dtype=np.float64).transpose()
    return f


def estimate_lrf(pt, ptnn, patch_kernel):
    """Ref:
    https://github.com/fabiopoiesi/dip/blob/master/lrf.py

    Args:
        pt (np.array): (3, )
        ptnn (np.array): (3, NN-1), without pt
        patch_kernel (float):

    Returns:
        np.array: (3, 3)
    """
    ptnn_pt = ptnn - pt[:, np.newaxis]  # (3, NN-1)

    ptnn_cov = (1.0 / ptnn.shape[-1]) * np.dot(ptnn_pt, ptnn_pt.T)  # (3, 3)

    eigvals, eigvecs = np.linalg.eig(ptnn_cov)
    smallest_eigval_idx = np.argmin(eigvals)
    np_hat = eigvecs[:, smallest_eigval_idx]  # (3, )

    zp = np_hat if np.sum(np.dot(np_hat, -ptnn_pt)) >= 0 else -np_hat  # (3, )
    zp /= np.linalg.norm(zp)

    ptnn_pt_zp = np.dot(ptnn_pt.T, zp[:, np.newaxis])  # (NN-1, 1)
    v = ptnn_pt - (ptnn_pt_zp * zp).T  # (3, NN-1)

    # (NN-1, )
    alpha = (patch_kernel - np.linalg.norm(-ptnn_pt, axis=0))**2
    # (NN-1, )
    beta = ptnn_pt_zp.squeeze()**2

    v_alpha_beta = np.dot(v, (alpha * beta)[:, np.newaxis])  # (3, 1)
    if np.linalg.norm(v_alpha_beta) < 1e-4:
        xp = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        xp = v_alpha_beta / np.linalg.norm(v_alpha_beta)
        xp = xp.squeeze()  # (3, 1) -> (3, )

    yp = np.cross(zp, xp)  # (3, )
    if np.linalg.norm(yp) < 1e-8:
        return np.identity(3, dtype=np.float32)
    yp /= np.linalg.norm(yp)

    xp = np.cross(yp, zp)  # (3, )
    if np.linalg.norm(xp) < 1e-8:
        return np.identity(3, dtype=np.float32)
    xp /= np.linalg.norm(xp)

    # LRF
    lRg = np.stack((xp, yp, zp), axis=1)

    if np.isnan(np.sum(lRg)):
        return np.identity(3, dtype=np.float32)
    else:
        return np.asarray(lRg, dtype=np.float32)


def rotation_matrix(axis, angle):
    """
    Ref:
    - https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    - https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py

    Args:
        axis (np.array or List): (3, )
        angle (float): in radians

    Returns:
        np.array: (4, 4)
    """
    assert len(axis) == 3

    axis = np.asarray(axis, dtype=np.float32)
    axis /= np.linalg.norm(axis)

    ca = math.cos(angle)
    sa = math.sin(angle)
    C = 1 - ca

    x, y, z = axis[0], axis[1], axis[2]

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    matrix = np.identity(4, dtype=np.float32)
    matrix[0, 0] = x * xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y * yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z * zC + ca
    return matrix


def translation_matrix(x):
    """
    Args:
        x (np.array or List): (3, )

    Returns:
        np.array: (4, 4)
    """
    assert len(x) == 3
    m = np.identity(4, dtype=np.float32)
    m[:3, 3] = np.asarray(x, dtype=np.float32)
    return m


def random_transform(points, normals, max_rotation_degree, max_translate):
    """
    Args:
        points (np.array): (N, 3)
        normals (np.array): (N, 3)
        max_rotation_degree (float): 
        max_translate (float):

    Returns:
        np.array: 
    """
    center = np.mean(points, axis=0, keepdims=False)  # (3, )
    assert max_rotation_degree <= 180.0
    mrd = max_rotation_degree * math.pi / 180.0
    rot = rotation_matrix(np.random.uniform(-1.0, 1.0, 3), random.uniform(-mrd, mrd))
    trans = translation_matrix(np.random.uniform(-max_translate, max_translate, 3))
    pose = trans @ translation_matrix(center) @ rot @ translation_matrix(-center)

    points = np.concatenate((points, np.ones((len(points), 1), dtype=points.dtype)), axis=1)
    points = points @ pose.T
    points = points[:, :3]

    if normals is not None:
        normals = np.concatenate((normals, np.zeros((len(normals), 1), dtype=normals.dtype)),
                                 axis=1)
        normals = normals @ pose.T
        normals = normals[:, :3]

    return points, normals, pose


def random_rotate(points, normals):
    """
    Args:
        points (np.array): (N, 3)
        normals (np.array): (N, 3)

    Returns:
        np.array: 
    """

    def rotmat(axis, theta):
        s = math.sin(theta)
        c = math.cos(theta)
        m = np.identity(4, dtype=np.float32)
        if axis == 0:
            m[1, 1] = c
            m[1, 2] = -s
            m[2, 1] = s
            m[2, 2] = c
        elif axis == 1:
            m[0, 0] = c
            m[0, 2] = s
            m[2, 0] = -s
            m[2, 2] = c
        elif axis == 2:
            m[0, 0] = c
            m[0, 1] = -s
            m[1, 0] = s
            m[1, 1] = c
        else:
            raise RuntimeError('The axis - {} is not supported.'.format(axis))
        return m

    pose = rotmat(random.randint(0, 2), random.uniform(0, math.pi * 2.))

    points = np.concatenate((points, np.ones((len(points), 1), dtype=points.dtype)), axis=1)
    points = points @ pose.T
    points = points[:, :3]

    if normals is not None:
        normals = np.concatenate((normals, np.zeros((len(normals), 1), dtype=normals.dtype)),
                                 axis=1)
        normals = normals @ pose.T
        normals = normals[:, :3]

    return points, normals, pose


def filter_outliers(points, normals=None, nb_points=256, radius=0.3):
    dtype = points.dtype
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd_flt, _ = pcd.remove_radius_outlier(nb_points, radius)
    out_points = np.asarray(pcd_flt.points, dtype=dtype)
    if normals is not None:
        out_normals = np.asarray(pcd_flt.normals, dtype=dtype)
    else:
        out_normals = None
    return out_points, out_normals



