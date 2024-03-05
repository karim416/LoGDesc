#encoding: utf-8
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import inspect
import sys
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
sys.path.insert(0, currentdir)

import _init_paths
import torch.multiprocessing
from models.LoGDesc_reg import LoGDesc_reg
from MVP_RG.registration.dataset import MVP_RG, furthest_point_sample

from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
import open3d as o3d
import random

import numpy as np
from packaging.version import parse, Version
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree

import pickle



def solve_icp(P, Q):

    up = P.mean(axis = 0)
    uq = Q.mean(axis = 0)

    P_centered = P - up
    Q_centered = Q - uq


    U, s, V = np.linalg.svd(np.dot(Q_centered.T, P_centered), full_matrices=True, compute_uv=True)
    R = np.dot(U, V)
    t = uq - np.dot(R, up)

    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    T[3, 3] = 1.0

    return T,t,R



def rmse_loss(pts, T, T_gt):
	pts_pred = pts @ T[:, :3, :3].transpose(1, 2) + T[:, :3, 3].unsqueeze(1)
	pts_gt = pts @ T_gt[:, :3, :3].transpose(1, 2) + T_gt[:, :3, 3].unsqueeze(1)
	return torch.norm(pts_pred - pts_gt, dim=2).mean(dim=1)

def rotation_error(R, R_gt):
	cos_theta = (torch.einsum('bij,bij->b', R, R_gt) - 1) / 2
	cos_theta = torch.clamp(cos_theta, -1, 1)
	return torch.acos(cos_theta) * 180 / np.pi


def translation_error(t, t_gt):
	return torch.norm(t - t_gt, dim=1)

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='Resgistration on MVP dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--local_rank', type=int, default=[0,1],
    help='Gpu rank.')

parser.add_argument(
    '--resume_model', type=str, default='best_model.pth',
    help='pre trained model')

parser.add_argument(
    '--l', type=int, default=6,
    help='Layers number in Normal Encoder')

parser.add_argument(
    '--sinkhorn_iterations', type=int, default=50,
    help='Number of Sinkhorn iterations performed')

parser.add_argument(
    '--descriptor_dim',  type=int, default=132,
    help=' LoGDesc output dim ')

parser.add_argument(
    '--max_keypoints',  type=int, default=768,
    help='subsampled kpts with fps')

parser.add_argument(
    '--use_kpt', type=bool, default=True, # True False
    help='subsample keypoints with topk')

if __name__ == '__main__':
    dict_params = parser.parse_args()
    import yaml, munch



    cfg_path = currentdir+'/MVP_RG/registration/cfgs/idam.yaml'
    args = munch.munchify(yaml.safe_load(open(cfg_path)))

    test_set =  MVP_RG(prefix = 'val', args = args, sample_radius=0.3, num_kpts=dict_params.max_keypoints)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, batch_size=1, num_workers=2, drop_last=True, pin_memory = True)


    if torch.cuda.is_available():
        map_location ='cuda'
    else:
        map_location ='cpu'

    path_checkpoint = currentdir+'/pre-trained/'+dict_params.resume_model
    checkpoint = torch.load(path_checkpoint, map_location=map_location)

    for key in checkpoint.keys():
        print(key)
    config = {
            'net': {
                'sinkhorn_iterations': dict_params.sinkhorn_iterations,
                'descriptor_dim': dict_params.descriptor_dim,
                'L':dict_params.l,
                'GNN_layers': ['self', 'cross'],
                'use_kpt' : dict_params.use_kpt,
                'lr' : 1e-4
            }
    }

    net = LoGDesc_reg(config.get('net', {}))
    optimizer = torch.optim.Adam(net.parameters(), lr=config.get('net', {}).get('lr'))
    net = torch.nn.DataParallel(net)
    net.load_state_dict(checkpoint)


    if torch.cuda.is_available():
        device=torch.device('cuda:{}'.format(dict_params.local_rank[0]))
        if torch.cuda.device_count() > 1:
            print("Testing using ", torch.cuda.device_count(), "GPUs!")
            net = torch.nn.DataParallel(net, device_ids=dict_params.local_rank)
    else:
        device = torch.device("cpu")
        print("Testing on CPU")

    net.to(device)
    with torch.no_grad():

        t_array = [] ;  t_pred_array = []
        r_array = [] ;  r_pred_array = []
        kpts0_array= np.zeros((len(test_loader),dict_params.max_keypoints,3))
        Trans_gt_array = np.zeros((len(test_loader),4,4))
        Trans_array = np.zeros((len(test_loader),4,4))

        for i, pred in enumerate(test_loader):

            net.double().eval()
            for k in pred:
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].to('cpu').detach())
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).to('cpu').detach())

            data = net(pred)

            pred = {**pred, **data}

            for b in range(len(pred['keypoints0'])):
                kpts0, kpts1 = pred['keypoints0'][b].cpu().numpy(), pred['keypoints1'][b].cpu().numpy()
                matches, matches1, conf = pred['matches0'][b].cpu().detach().numpy(), pred['matches1'][b].cpu().detach().numpy(), pred['matching_scores0'][b].cpu().detach().numpy()
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]
                translations_ab = pred['translation_ab'][b].cpu().numpy()
                R_ab = pred['R_ab'][b].cpu().numpy()
                T_gt = pred['T_gt'][b].cpu().numpy()
                kconf,kpt_indices0 = torch.topk(torch.tensor(mconf), k=np.min([128,len(mconf)]),sorted=False)

                if len(mkpts0)>2:
                    if len(mkpts0)<100:
                         _,translations_ab_pred,rotations_ab_pred=solve_icp(mkpts0[kpt_indices0],mkpts1[kpt_indices0])
                    else : 
                        matched0 = mkpts0
                        matched1 = mkpts1
                        nb_patch =  7
                        patch_size = len(mkpts0) // nb_patch
                        optimal_inliers = 0
                        for j in range (nb_patch) : 
                            sub_indices = furthest_point_sample(matched0, max_points=patch_size)
                            sub0 = matched0[sub_indices]
                            sub1 = matched1[sub_indices]
                            matched0 = np.delete(matched0, sub_indices, axis=0)
                            matched1 = np.delete(matched1, sub_indices, axis=0)
                            _,t_patch,r_patch=solve_icp(sub0,sub1)
                            R = Rotation.from_matrix(r_patch)

                            transformed0 = R.apply(kpts0) + t_patch
                            tree = KDTree(transformed0)
                            distances, indices = tree.query(kpts1)
                            reindexed_point_cloud0 = transformed0[indices]
                            dists = np.linalg.norm (reindexed_point_cloud0 - kpts1,axis=-1)
                            nb_inliers = np.sum(dists < 0.08)

                            if j==0 or nb_inliers > optimal_inliers : 
                                optimal_inliers = nb_inliers
                                translations_ab_pred = t_patch
                                rotations_ab_pred = r_patch
                
                t_array.append(translations_ab)
                t_pred_array.append(translations_ab_pred)
                r_array.append(R_ab)
                r_pred_array.append(rotations_ab_pred)

                kpts0_array[i] = np.array(kpts0)             

                r_err1 = rotation_error(torch.from_numpy(rotations_ab_pred).unsqueeze(0).double(),torch.from_numpy(R_ab).unsqueeze(0).double()).item()
                t_err2 = translation_error(torch.from_numpy(translations_ab).unsqueeze(0).double(),torch.from_numpy(translations_ab_pred).unsqueeze(0).double()).item()
                T = np.zeros((4, 4))
                T[0:3, 0:3] = rotations_ab_pred
                T[0:3, 3] = translations_ab_pred
                T[3, 3] = 1.0

                Trans_array[i] = np.array(T)
                Trans_gt_array [i] = np.array(T_gt)
                rmse = rmse_loss(torch.from_numpy(kpts0).unsqueeze(0).double(), torch.from_numpy(T).unsqueeze(0).double(), torch.from_numpy(T_gt).unsqueeze(0).double()).item()
                print('{:4d}, r_err {:.3f}, t_err {:.3f}, trans rmse {:.3f}'.format(i,r_err1,t_err2,rmse))


        rotations_ab = np.stack(r_array, axis=0)
        translations_ab = np.stack(t_array, axis=0)
        rotations_ab_pred = np.stack(r_pred_array, axis=0)
        translations_ab_pred = np.stack(t_pred_array, axis=0)
        
        tot_rmse = rmse_loss(torch.from_numpy(kpts0_array).double(),torch.from_numpy(Trans_array).double(),torch.from_numpy(Trans_gt_array).double())
        r_err = rotation_error(torch.from_numpy(rotations_ab_pred).double(),torch.from_numpy(rotations_ab).double())
        t_err = translation_error(torch.from_numpy(translations_ab).double(),torch.from_numpy(translations_ab_pred).double())

        print('r_err',r_err.mean().item())
        print('t_err',t_err.mean().item())
        print('tot_rmse',tot_rmse.sum().item()/i)




