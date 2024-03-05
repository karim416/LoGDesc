import numpy as np
import torch
import torch.nn as nn

import inspect
import sys
import os
import numpy as np
import open3d as o3d
import torch.nn.functional as F


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import _init_paths
from models.transformer.layers import build_dropout_layer


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

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings


class GeometricStructureEmbedding(nn.Module):
    def __init__(self, hidden_dim=128, sigma_a=15):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

    @torch.no_grad()
    def get_embedding_indices(self, points, normals):
        r"""Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.
        Args:
            points: torch.Tensor (B, N, 3), input point cloud
        Returns:
            d_indices: torch.FloatTensor (B, N, N), distance embedding indices
            a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """

        batch_size, num_point, _ = points.shape

        k = 1

        expanded_normals = normals.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
        knn_normals = normals.expand(batch_size, num_point,3).unsqueeze(-2) # (B, N, 1, 3)
        anc_vectors = expanded_normals # (B, N, N, 3)
        ref_vectors = knn_normals.unsqueeze(2).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, 1, 3)
        anc_vectors = anc_vectors.unsqueeze(3).expand(batch_size, num_point, num_point, k, 3)  # (B, N, N, 1, 3)
        sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N, 1)
        cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N, 1)
        angles = torch.atan2(sin_values, cos_values)  # (B, N, N, 1)
        a_indices = angles * self.factor_a

        return a_indices

    def forward(self, points ,normals=None):
        a_indices = self.get_embedding_indices(points,normals)

        a_embeddings = self.embedding(a_indices).mean(dim=3)
        a_embeddings = self.proj_a(a_embeddings)
 
        return a_embeddings

