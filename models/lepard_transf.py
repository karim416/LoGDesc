import copy
import math
import torch
from torch import nn
from torch.nn import Module, Dropout
import inspect
import sys
import os
import open3d as o3d


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
import random

class GeometryAttentionLayer(nn.Module):

    def __init__(self, d_model ,pe_type='rotary' ):

        super(GeometryAttentionLayer, self).__init__()

        d_model = d_model
        nhead =  4

        self.dim = d_model // nhead
        self.nhead = nhead
        self.pe_type = pe_type
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # self.attention = Attention() #LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_pe, source_pe, x_mask=None, source_mask=None):

        bs = x.size(0)
        q, k, v = x, source, source
        qp, kvp  = x_pe, source_pe
        q_mask, kv_mask = x_mask, source_mask

        if self.pe_type == 'sinusoidal':
            #w(x+p), attention is all you need : https://arxiv.org/abs/1706.03762
            if qp is not None: # disentangeld
                q = q + qp
                k = k + kvp
            qw = self.q_proj(q).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            kw = self.k_proj(k).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
            vw = self.v_proj(v).view(bs, -1, self.nhead, self.dim)

        elif self.pe_type == 'rotary':
            #Rwx roformer : https://arxiv.org/abs/2104.09864

            qw = self.q_proj(q)
            kw = self.k_proj(k)
            vw = self.v_proj(v)

            if qp is not None: # disentangeld
                q_cos, q_sin = qp[...,0] ,qp[...,1]
                k_cos, k_sin = kvp[...,0],kvp[...,1]
                print(qw.size())
                print(q_cos.size())
                print(q_sin.size())
                qw = VolPE.embed_rotary(qw, q_cos, q_sin)
                kw = VolPE.embed_rotary(kw, k_cos, k_sin)
        
            qw = qw.view(bs, -1, self.nhead, self.dim)
            kw = kw.view(bs, -1, self.nhead, self.dim)
            vw = vw.view(bs, -1, self.nhead, self.dim)

        else:
            raise KeyError()

        # attention
        a = torch.einsum("nlhd,nshd->nlsh", qw, kw)
        if kv_mask is not None:
            a.masked_fill_( q_mask[:, :, None, None] * (~kv_mask[:, None, :, None]), float('-inf'))
        a =  a / qw.size(3) **0.5
        a = torch.softmax(a, dim=2)
        o = torch.einsum("nlsh,nshd->nlhd", a, vw).contiguous()  # [N, L, (H, D)]

        message = self.merge(o.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        e = x + message

        return e






class RepositioningTransformer(nn.Module):

    def __init__(self, d_model = 32, layer_types = ['self','self']*2 , pe_type='rotary'):
        super(RepositioningTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = 4
        self.layer_types = layer_types
        self.pe_type =pe_type

        self.entangled= True

        self.positional_encoding = VolPE(feature_dim = self.d_model , pe_type = self.pe_type)

        encoder_layer = GeometryAttentionLayer (d_model ,pe_type)

        self.layers = nn.ModuleList()

        for l_type in self.layer_types:

            if l_type in ['self','cross']:

                self.layers.append( copy.deepcopy(encoder_layer))

            else:
                raise KeyError()

        self._reset_parameters()



    def forward(self, src_feat, tgt_feat, s_pcd, t_pcd, src_mask=None, tgt_mask=None, T = None, timers = None):

        self.timers = timers

        assert self.d_model == src_feat.size(2), "the feature number of src and transformer must be equal"

        if T is not None:
            R, t = T
            src_pcd_wrapped = (torch.matmul(R, s_pcd.transpose(1, 2)) + t).transpose(1, 2)
            tgt_pcd_wrapped = t_pcd
        else:
            src_pcd_wrapped = s_pcd
            tgt_pcd_wrapped = t_pcd

        src_pe = self.positional_encoding( src_pcd_wrapped)
        tgt_pe = self.positional_encoding( tgt_pcd_wrapped)


        if not self.entangled:

            position_layer = 0

            for layer, name in zip(self.layers, self.layer_types) :

                if name == 'self':
                    if self.timers: self.timers.tic('self atten')
                    src_feat = layer(src_feat, src_feat, src_pe, src_pe, src_mask, src_mask,)
                    tgt_feat = layer(tgt_feat, tgt_feat, tgt_pe, tgt_pe, tgt_mask, tgt_mask)
                    if self.timers: self.timers.toc('self atten')

                elif name == 'cross':
                    if self.timers: self.timers.tic('cross atten')
                    src_feat = layer(src_feat, tgt_feat, src_pe, tgt_pe, src_mask, tgt_mask)
                    tgt_feat = layer(tgt_feat, src_feat, tgt_pe, src_pe, tgt_mask, src_mask)
                    if self.timers: self.timers.toc('cross atten')

                else :
                    raise KeyError

            return src_feat, tgt_feat, src_pe, tgt_pe

        else : # pos. fea. entangeled

            position_layer = 0
            src_feat = VolPE.embed_pos(self.pe_type, src_feat, src_pe)
            tgt_feat = VolPE.embed_pos(self.pe_type, tgt_feat, tgt_pe)

            for layer, name in zip(self.layers, self.layer_types):
                if name == 'self':
                    if self.timers: self.timers.tic('self atten')
                    src_feat = layer(src_feat, src_feat, None, None, src_mask, src_mask, )
                    tgt_feat = layer(tgt_feat, tgt_feat, None, None, tgt_mask, tgt_mask)
                    if self.timers: self.timers.toc('self atten')
                elif name == 'cross':
                    if self.timers: self.timers.tic('cross atten')
                    src_feat = layer(src_feat, tgt_feat, None, None, src_mask, tgt_mask)
                    tgt_feat = layer(tgt_feat, src_feat, None, None, tgt_mask, src_mask)
                    if self.timers: self.timers.toc('cross atten')


            return src_feat, tgt_feat, src_pe, tgt_pe


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



class VolPE(nn.Module):

    def __init__(self, feature_dim, pe_type , voxel_size = 0.08  , vol_bnds = [ [ -3.6, -2.4,  1.14], [ 1.093, 0.78, 2.92 ]]) : 
        super().__init__()

        self.feature_dim = feature_dim
        self.vol_bnds = vol_bnds
        self.voxel_size = voxel_size
        self.vol_origin  = self.vol_bnds[0]
        self.pe_type = pe_type

    def voxelize(self, xyz):
        '''
        @param xyz: B,N,3
        @return: B,N,3
        '''
        if type ( self.vol_origin ) == list :
            self.vol_origin = torch.FloatTensor(self.vol_origin ).view(1, 1, -1).to( xyz.device )
        return (xyz - self.vol_origin) / self.voxel_size

    @staticmethod
    def embed_rotary(x, cos, sin):
        '''
        @param x: [B,N,d]
        @param cos: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @param sin: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @return:
        '''
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    @staticmethod
    def embed_pos(pe_type, x, pe):
        """ combine feature and position code
        """
        if  pe_type == 'rotary':
            return VolPE.embed_rotary(x, pe[..., 0], pe[..., 1])
        elif  pe_type == 'sinusoidal':
            return  x + pe
        else:
            raise KeyError()


    def forward(self,  XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape
        vox = self.voxelize( XYZ)
        x_position, y_position, z_position = vox[..., 0:1], vox[...,1:2], vox[...,2:3]
        div_term = torch.exp( torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device) *  (-math.log(10000.0) / (self.feature_dim // 3)))
        div_term = div_term.view( 1,1, -1) # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term) # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        if self.pe_type == 'sinusoidal' :
            position_code = torch.cat( [ sinx, cosx, siny, cosy, sinz, cosz] , dim=-1 )

        elif self.pe_type == "rotary" :
            # sin/cos [θ0,θ1,θ2......θd/6-1] -> sin/cos [θ0,θ0,θ1,θ1,θ2,θ2......θd/6-1,θd/6-1]
            sinx, cosx, siny, cosy, sinz, cosz = map( lambda  feat:torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
                 [ sinx, cosx, siny, cosy, sinz, cosz] )
            sin_pos = torch.cat([sinx,siny,sinz], dim=-1)
            cos_pos = torch.cat([cosx,cosy,cosz], dim=-1)
            position_code = torch.stack( [cos_pos, sin_pos] , dim=-1) # (B,n,d,2)
        else:
            raise KeyError()


        if position_code.requires_grad:
            position_code = position_code.detach()


        return position_code

if __name__ == '__main__':
    from torch.autograd import Variable
    d = 528//4
    n = 12
    print('d',d)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    pts = Variable(torch.rand(2,n,3)).to(device)
    ftr = Variable(torch.rand(2,n,d)).to(device)
    transformer=RepositioningTransformer(d_model=d)
    transformer.to(device)

    out1,_,_,_=transformer(ftr,ftr,pts,pts)
    print('out1',out1.size())

 