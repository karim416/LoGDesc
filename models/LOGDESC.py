
import torch
import torch.nn as nn
import torch.nn.functional as F

import inspect
import sys
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import _init_paths
from models.lepard_transf import *





class LOGDESC(nn.Module):
    def __init__(self,  emb_dims=96 , raw_dim=10):
        super().__init__()

        self.mlp =  nn.Sequential(
            nn.Conv2d(raw_dim, emb_dims , 1),
            nn.GroupNorm(6, emb_dims),
            nn.ReLU(),
            nn.Conv2d(emb_dims,emb_dims, 1),
            nn.GroupNorm(6, emb_dims),
            nn.ReLU(),
            nn.Conv2d(emb_dims, emb_dims *2, 1),
            nn.GroupNorm(6, emb_dims * 2),
            nn.ReLU(),
         )
        self.self_a = RepositioningTransformer(d_model= emb_dims * 2)
        self.finallayer = nn.Conv1d(emb_dims * 2 , emb_dims , 1)


    def forward(self,new_feat,new_feat1,pc0,pc1):

    
        new_feat = self.mlp(new_feat)
        max_feat = torch.max(new_feat, 2)[0].permute(0, 2, 1)  
    
        new_feat1 = self.mlp(new_feat1)
        max_feat1 = torch.max(new_feat1, 2)[0].permute(0, 2, 1)  

        atten_feat,atten_feat1,_,_ = self.self_a(max_feat, max_feat1, pc0, pc1)  
        final_desc0,final_desc1 = self.finallayer(atten_feat.permute(0, 2, 1) ), self.finallayer(atten_feat1.permute(0, 2, 1) ) 

        logdesc0 = F.normalize(final_desc0.permute(0, 2, 1) ,dim=-1)
        logdesc1 = F.normalize(final_desc1.permute(0, 2, 1) ,dim=-1)

        return logdesc0,logdesc1 


