"""
 ref :  https://github.com/chenghao-shi/MDGAT-matcher/
"""
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import nn
import time
import argparse
import inspect
import sys
import os
import numpy as np
import open3d as o3d

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import _init_paths
from models.transformer.conditional_transformer import RPEConditionalTransformer
from models.normal_embedding import GeometricStructureEmbedding
from models.LOGDESC import *
from modules.repsurface_utils import *


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm
    return Z



def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)





def get_graph_feature(x, k=20, idx=None, extra_dim=False):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if extra_dim is False:
            idx = knn(x.transpose(-1, -2), x.transpose(-1, -2), k=k)
        else:
            idx = knn(x[:, 6:].transpose(-1, -2), x[:, 6:].transpose(-1, -2), k=k)  # idx = knn(src_xyz[:, :3], k=k)
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx += idx_base
    else: 
        k = idx.size(-1)


    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature 

def group_normals(lrfs, knn, normals):
    """
    lrfs :  (N_points,3,3)
    knn :  (N_points,k)
    normals :  (N_points,3)

    returns 
    patch_normals  (N_points,k,3)
    """

    device = lrfs.device  
    
    patch_normals = normals[knn]

    xp = lrfs[:, :, 0].unsqueeze(-1)
    yp = lrfs[:, :, 1].unsqueeze(-1)
    zp = lrfs[:, :, 2].unsqueeze(-1)

    nx = torch.einsum('nbc,nck->nbk', patch_normals, xp).squeeze()
    ny = torch.einsum('nbc,nck->nbk', patch_normals, yp).squeeze()
    nz = torch.einsum('nbc,nck->nbk', patch_normals, zp).squeeze()

    patch_normals = torch.stack((nx, ny, nz), dim=-1)

    return patch_normals.to(device)


class KeyPointNet(nn.Module):
    def __init__(self, num_keypoints):
        super(KeyPointNet, self).__init__()
        self.num_keypoints = num_keypoints

    def forward(self, src, tgt, n0,n1, src_embedding, tgt_embedding ):
        src_embedding = src_embedding.permute(0,2,1)
        tgt_embedding = tgt_embedding.permute(0,2,1)
        src = src.permute(0,2,1)
        tgt = tgt.permute(0,2,1)
        n0 = n0.permute(0,2,1)
        n1 = n1.permute(0,2,1)

        batch_size, num_dims, num_points = src_embedding.size()
        src_norm = torch.norm(src_embedding, dim=1, keepdim=True)
        tgt_norm = torch.norm(tgt_embedding, dim=1, keepdim=True)
        src_topk_idx = torch.topk(src_norm, k=self.num_keypoints, dim=2, sorted=False)[1]
        tgt_topk_idx = torch.topk(tgt_norm, k=self.num_keypoints, dim=2, sorted=False)[1]
        src_keypoints_idx = src_topk_idx.repeat(1, 3, 1)
        tgt_keypoints_idx = tgt_topk_idx.repeat(1, 3, 1)
        src_embedding_idx = src_topk_idx.repeat(1, num_dims, 1)
        tgt_embedding_idx = tgt_topk_idx.repeat(1, num_dims, 1)
        n0_k= torch.gather(n0, dim=2, index=src_keypoints_idx)
        n1_k= torch.gather(n1, dim=2, index=tgt_keypoints_idx)

        src_keypoints = torch.gather(src, dim=2, index=src_keypoints_idx)
        tgt_keypoints = torch.gather(tgt, dim=2, index=tgt_keypoints_idx)
        
        src_embedding = torch.gather(src_embedding, dim=2, index=src_embedding_idx)
        tgt_embedding = torch.gather(tgt_embedding, dim=2, index=tgt_embedding_idx)
        return src_keypoints.permute(0,2,1), tgt_keypoints.permute(0,2,1), n0_k.permute(0,2,1), n1_k.permute(0,2,1), src_embedding.permute(0,2,1), tgt_embedding.permute(0,2,1)



class LoGDesc_reg(nn.Module):
    default_config = {
        'descriptor_dim': 132,
        'GNN_layers': ['self', 'cross'],
        'L' : 6,
        'sinkhorn_iterations': 50,
        'descriptor' : 'LoGDesc',
        'use_kpt' : False,
        'lr' : 1e-4
        
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.transformer = RPEConditionalTransformer(blocks = self.config['GNN_layers']*self.config['L'], d_model=self.config['descriptor_dim'])
        self.position_embed = GeometricStructureEmbedding( hidden_dim=self.config['descriptor_dim'],sigma_a=15)
        self.final_proj = nn.Conv1d( self.config['descriptor_dim'], self.config['descriptor_dim'],kernel_size=1, bias=True)
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        self.lr = config['lr']
        self.triplet_loss_gamma = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.siwane = UmbrellaSurfaceConstructor_v2(k = 30 )
        if self.config['use_kpt'] :      
            self.kpt_extractor = KeyPointNet(768)

        self.desc2 = LOGDESC(emb_dims=self.config['descriptor_dim'],raw_dim=15)


    def forward(self, data ):


        pc0, pc1 = data['pc0'].double(), data['pc1'].double()

        lrfs0, lrfs1 = data['lrfs_i'].double(), data['lrfs_j'].double()

        planarity0, omnivariance0, anisotropy0 = data['planarity0'].double(), data['omnivariance0'].double(), data['anisotropy0'].double()
        planarity1, omnivariance1, anisotropy1 = data['planarity1'].double(), data['omnivariance1'].double(), data['anisotropy1'].double()
        local_var0 = torch.cat((planarity0, omnivariance0, anisotropy0),-1).to(pc0.device).double()
        local_var1 = torch.cat((planarity1, omnivariance1, anisotropy1),-1).to(pc0.device).double()

        n0_tensor = []
        n1_tensor = []

        idx0_tensor = []
        idx1_tensor = []

        normals_proj_tensor0 = []
        normals_proj_tensor1 = []
        with torch.no_grad():
            for pc_src,pc_tgt,lrf0,lrf1 in zip(pc0,pc1,lrfs0,lrfs1):
                off0 = torch.IntTensor([pc_src.shape[0]]).to(pc_src.device)
                off1 = torch.IntTensor([pc_tgt.shape[0]]).to(pc_tgt.device)
                group_idx0, point_normal0 = self.siwane(pc_src.float(),off0)
                group_idx1, point_normal1 = self.siwane(pc_tgt.float(),off1)
                proj_normal0 = group_normals(lrf0,group_idx0.long(),point_normal0.double())
                proj_normal1 = group_normals(lrf1,group_idx1.long(),point_normal1.double())

                n0_tensor.append(point_normal0.double())
                n1_tensor.append(point_normal1.double())
                idx0_tensor.append(group_idx0.long())
                idx1_tensor.append(group_idx1.long())
                normals_proj_tensor0.append(proj_normal0.double())
                normals_proj_tensor1.append(proj_normal1.double())



        n0 = torch.stack(n0_tensor, dim=0).to(pc0.device).double()
        n1 = torch.stack(n1_tensor, dim=0).to(pc1.device).double()
        normals_proj0 = torch.stack(normals_proj_tensor0, dim=0).to(pc0.device).double()
        normals_proj1 = torch.stack(normals_proj_tensor1, dim=0).to(pc1.device).double()

        idx1 = torch.stack(idx1_tensor, dim=0).to(pc0.device) 
        idx0 = torch.stack(idx0_tensor, dim=0).to(pc0.device)


        f0 = torch.cat((pc0,local_var0),-1).permute(0,2,1).to(pc0.device).double() 
        f1 = torch.cat((pc1,local_var1),-1).permute(0,2,1).to(pc1.device).double()


        clusf0 = get_graph_feature(f0, k=16, idx=idx0, extra_dim=False)
        clusf1 = get_graph_feature(f1, k=16, idx=idx1, extra_dim=False)

        clusf0 =  torch.cat((clusf0, normals_proj0.permute(0,3,1,2)), dim=1)
        clusf1 =  torch.cat((clusf1, normals_proj1.permute(0,3,1,2)), dim=1)

        final_desc0, final_desc1 = self.desc2(clusf0.permute(0,1,3,2),clusf1.permute(0,1,3,2),pc0,pc1)

        if self.config['use_kpt'] :      
            pc0, pc1,n0,n1, final_desc0, final_desc1 = self.kpt_extractor(pc0,pc1,n0,n1,final_desc0, final_desc1) 



        n_embed0 = self.position_embed(pc0[:,:,:3],n0)
        n_embed1 = self.position_embed(pc1[:,:,:3],n1) 
        desc0, desc1 = self.transformer(final_desc0, final_desc1,n_embed0,n_embed1) 

        mdesc0, mdesc1 = self.final_proj(desc0.permute(0,2,1)), self.final_proj(desc1.permute(0,2,1))

        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        
        scores = scores / self.config['descriptor_dim']**.5

        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])


        max0, max1 = scores[:, :-1, :].max(2), scores[:, :, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        valid0, valid1 = indices0<(scores.size(2)-1), indices1<(scores.size(1)-1)
        zero = scores.new_tensor(0)
        if valid0.sum() == 0:
            mscores0 = torch.zeros_like(indices0, device=self.device)
            mscores1 = torch.zeros_like(indices1, device=self.device)
        else:
            mscores0 = torch.where(valid0, max0.values.exp(), zero)
            mscores1 = torch.where(valid1, max1.values.exp(), zero)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))


        output =  {
            'matches0': indices0, 
            'matches1': indices1,
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'desc0' :  mdesc0,
            'desc1' :  mdesc1,
            'scores': scores,
            'keypoints0' : pc0,
            'keypoints1' : pc1,

        }
        return output


