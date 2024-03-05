import h5py
import numpy as np
import os
import open3d as o3d
import torch
from torch.utils.data import Dataset

import os, sys , inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
Gparentdir = os.path.dirname(parentdir)

sys.path.insert(0, parentdir)
sys.path.insert(0, Gparentdir)


from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist, minkowski
from sklearn.neighbors import NearestNeighbors

from utils.pointcloud import (
    apply_transform,
    get_nearest_neighbor,
    get_rotation_translation_from_transform,
)

from modelnet40_data.kde import *

import pickle
import os, sys , inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
Gparentdir = os.path.dirname(parentdir)

sys.path.insert(0, parentdir)
sys.path.insert(0, Gparentdir)


def get_correspondences(ref_points, src_points, transform , matching_radius , mutual_check = False):
    r"""Find the ground truth correspondences within the matching radius between two point clouds.

    Return correspondence indices [indices in ref_points, indices in src_points]
    """


    src_points = apply_transform(src_points, transform)

    dists = cdist(ref_points, src_points)

    '''Find ground true keypoint matching'''
    min1 = np.argmin(dists, axis=0)
    min2 = np.argmin(dists, axis=1)

    min1v = np.min(dists, axis=1)
    min1f = min2[min1v < matching_radius]

    '''For calculating repeatibility'''
    rep = len(min1f)

    match1, match2 = -1 * np.ones((len(src_points)), dtype=np.int16), -1 * np.ones((len(ref_points)), dtype=np.int16)
    if mutual_check :
        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)
        match1[min1[matches]] = matches
        match2[matches] = min1[matches]
    else:
        match1[min1v < matching_radius] = min1f
        min2v = np.min(dists, axis=0)
        min2f = min1[min2v < matching_radius]
        match2[min2v < matching_radius] = min2f

    return match1,match2


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

            if self.prefix != "test":
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




        kpt_indices1 = furthest_point_sample(src, max_points=self.num_ktps)
        kpt_indices2 = furthest_point_sample(tgt, max_points=self.num_ktps)





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
                    
            
        src = src[kpt_indices1, :]
        tgt = tgt[kpt_indices2, :]

        # src = np.random.permutation(src)
        # tgt = np.random.permutation(tgt)
        mactches1, mactches0  = get_correspondences(tgt, src, transform , matching_radius = 0.08 , mutual_check = False)
        # valid_gt = mactches0 > -1
        # print(mactches0[valid_gt].shape)
        # print(mactches1[mactches0[valid_gt]].shape)
        # src = torch.from_numpy(src)
        # tgt = torch.from_numpy(tgt)
        transform = torch.from_numpy(transform)
        match_level = match_level
        rot_level = rot_level
                    
        out_dict =  {
            'pc0': src,
            'pc1': tgt,
            # 'normals0': normals1,
            # 'normals1': normals2,
            'gt_matches0':np.squeeze(mactches0),
            'gt_matches1':np.squeeze(mactches1),
            'T_gt' : transform,
            'planarity0' :  np.expand_dims(planarity0,-1),
            'omnivariance0': np.expand_dims(omnivariance0,-1),
            'anisotropy0':np.expand_dims(anisotropy0,-1),
            'planarity1' :  np.expand_dims(planarity1,-1),
            'omnivariance1':np.expand_dims(omnivariance1,-1),
            'anisotropy1':np.expand_dims(anisotropy1,-1),
            # 'patches_normals_i' : patch_normals1,
            # 'patches_normals_j' : patch_normals2,
            'lrfs_i': lrfs1,
            'lrfs_j': lrfs2,
            'rot_level' : rot_level,
            'match_level' : match_level,
        }



        with open(currentdir+'/pkl_data/'+self.prefix+'/'+str(index)+'.pickle', 'wb') as handle:
                pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(index,' saved !')
        return out_dict
        # return src, tgt, transform, match_level, rot_level
