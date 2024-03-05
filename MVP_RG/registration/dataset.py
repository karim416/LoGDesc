import h5py
import numpy as np
import os
import open3d as o3d
import torch
from torch.utils.data import Dataset

import os, sys , inspect, pickle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
Gparentdir = os.path.dirname(parentdir)

sys.path.insert(0, parentdir)
sys.path.insert(0, Gparentdir)


from scipy.spatial.transform import Rotation

from pcd import (
    furthest_point_sample, 
    patch_estimate_lrf)


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



def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd


def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)





def get_lrfs(kpt_indices,
               points,
               num_points_per_sample=1024,
               sample_radius=0.3,
               with_lrf=True,
               ):

    kpts = points[kpt_indices, :]
    num_kpts = len(kpts)

    patches = np.random.rand(num_kpts, num_points_per_sample, 3)
    patch_normals = np.zeros_like(patches) 
    lrfs = np.random.rand(num_kpts, 3, 3) 

    t_points = torch.from_numpy(points).unsqueeze(0)
    t_kpts = torch.from_numpy(kpts).unsqueeze(0)

    dist_map = torch.sqrt(pairwise_distance(t_kpts, t_points))  # (B, N, N)
    start_index = 0
    end_index = start_index + num_points_per_sample
    knn_indices = dist_map.topk(k=end_index, dim=2, largest=False)[1][:, :, start_index:end_index]
    knn_points = knn_indices.detach().cpu().numpy().squeeze(0)

    patches = points[knn_points]
    lrfs, planarity, omnivariance, anisotropy = patch_estimate_lrf(kpts.T,patches.transpose(2,0,1),sample_radius)
    return lrfs , patches  , knn_points , planarity, omnivariance, anisotropy

class MVP_RG(Dataset):
    """docstring for MVP_RG"""
    def __init__(self, prefix, args , sample_radius=0.3, num_kpts=768):
        self.sample_radius = sample_radius
        self.num_ktps = num_kpts
        self.prefix = prefix
        if self.prefix == "train":
            f = h5py.File(currentdir+'/data/MVP_Train_RG.h5', 'r')
        elif self.prefix == "val":
            f = h5py.File(currentdir+'/data/MVP_Test_RG.h5', 'r')
        elif self.prefix == "test":
            f = h5py.File(currentdir+'/data/MVP_ExtraTest_RG.h5', 'r')
        
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans

        self.label = f['cat_labels'][:].astype('int32')
        if self.prefix == "test":
            self.src = np.array(f['rotated_src'][:].astype('float32'))
            self.tgt = np.array(f['rotated_tgt'][:].astype('float32'))
        else:
            self.match_level = np.array(f['match_level'][:].astype('int32'))

            match_id = []
            for i in range(len(f['match_id'].keys())):
                ds_data = f['match_id'][str(i)][:]
                match_id.append(ds_data)
            self.match_id = np.array(match_id, dtype=object)

            if self.prefix == "train":
                self.src = np.array(f['src'][:].astype('float32'))
                self.tgt = np.array(f['tgt'][:].astype('float32'))
                if args.max_angle > 45:
                    self.rot_level = int(1)
                else:
                    self.rot_level = int(0)
            elif self.prefix == "val":
                self.src = np.array(f['rotated_src'][:].astype('float32'))
                self.tgt = np.array(f['rotated_tgt'][:].astype('float32'))
                self.transforms = np.array(f['transforms'][:].astype('float32'))
                self.rot_level = np.array(f['rot_level'][:].astype('int32'))

        f.close()
        
        if args.category:
            self.src = self.src[self.label==args.category]
            self.tgt = self.tgt[self.label==args.category]
            if self.prefix is not "test":
                self.match_id = self.match_id[self.label==args.category]
                self.match_level = self.match_level[self.label==args.category]
                if self.prefix == False:
                    self.transforms = self.transforms[self.label==args.category]
                    self.rot_level = self.rot_level[self.label==args.category]
            self.label = self.label[self.label==args.category]

        # print(self.src.shape, self.tgt.shape, self.match_id.shape, self.match_level.shape, self.label.shape)

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, index):
        

        src = self.src[index]
        tgt = self.tgt[index]

        if self.prefix == "train":
            transform = random_pose(self.max_angle, self.max_trans / 2)
            pose1 = random_pose(np.pi, self.max_trans)
            pose2 = transform @ pose1
            src = src @ pose1[:3, :3].T + pose1[:3, 3]
            tgt = tgt @ pose2[:3, :3].T + pose2[:3, 3]
            rot_level = self.rot_level
            match_level = self.match_level[index]

        elif self.prefix == "val":
            transform = self.transforms[index]
            rot_level = self.rot_level[index]
            match_level = self.match_level[index]



        if self.num_ktps < 2048 : 
            kpt_indices1 = furthest_point_sample(src, max_points=self.num_ktps)
            kpt_indices2 = furthest_point_sample(tgt, max_points=self.num_ktps)
            kpts0 = src[kpt_indices1, :]
            kpts1 = tgt[kpt_indices2, :]



        else : 
            kpt_indices1 = np.arange(len(src))
            kpt_indices2 = np.arange(len(tgt))
            mactches1, mactches0  = None, None
            kpts0 = src[kpt_indices1, :]
            kpts1 = tgt[kpt_indices2, :]


        lrfs1 , patches1 , knn1, planarity0, omnivariance0, anisotropy0 = get_lrfs(
                    kpt_indices1,
                    src,
                    num_points_per_sample=128,
                    sample_radius=self.sample_radius,
                    with_lrf=True
                    )

        
        lrfs2 , patches2 , knn2 , planarity1, omnivariance1, anisotropy1 = get_lrfs(
                    kpt_indices2,
                    tgt,
                    num_points_per_sample=128,
                    sample_radius=self.sample_radius,
                    with_lrf=True
                    )
                    
            

        out_dict =  {
            'pc0': kpts0,
            'pc1': kpts1,
            'keypoints0':kpts0, 
            'keypoints1':kpts1, 
            'knn0': knn1,
            'knn1': knn2,
            'patches0': patches1,
            'patches1': patches2,
            'planarity0' :  np.expand_dims(planarity0,-1),
            'omnivariance0': np.expand_dims(omnivariance0,-1),
            'anisotropy0':np.expand_dims(anisotropy0,-1),
            'planarity1' :  np.expand_dims(planarity1,-1),
            'omnivariance1':np.expand_dims(omnivariance1,-1),
            'anisotropy1':np.expand_dims(anisotropy1,-1),
            'src_raw' : src,
            'tgt_raw' : tgt,
            'lrfs_i': lrfs1,
            'lrfs_j': lrfs2,
        }


        if self.prefix != "test":
            out_dict['T_gt'] = transform
            out_dict['translation_ab'] =  transform[:3,3]
            out_dict['R_ab'] =  transform[:3,:3]

        return out_dict

