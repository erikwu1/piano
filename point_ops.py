"""
utils/point_ops.py

Farthest Point Sampling (FPS) and K-Nearest Neighbours (KNN) implemented in
pure PyTorch so they run on GPU without extra C++ extensions.
"""

import torch
import torch.nn.functional as F


def farthest_point_sample(xyz: torch.Tensor, n_points: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) as described in the SCN paper (Hua et al. 2021)
    and PointNet++ (Qi et al. 2017).

    Args:
        xyz:      (B, N, 3)  input point cloud
        n_points: int        number of centroids to select

    Returns:
        idx:  (B, n_points)  indices of selected centroids
    """
    B, N, _ = xyz.shape
    device = xyz.device

    idx        = torch.zeros(B, n_points, dtype=torch.long, device=device)
    distance   = torch.full((B, N), float('inf'), device=device)

    # Start from a random point in each batch
    farthest   = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    batch_idx  = torch.arange(B, dtype=torch.long, device=device)

    for i in range(n_points):
        idx[:, i] = farthest
        centroid   = xyz[batch_idx, farthest, :].unsqueeze(1)     # (B, 1, 3)
        dist       = torch.sum((xyz - centroid) ** 2, dim=-1)     # (B, N)
        # Update each point's distance to its nearest centroid so far
        mask       = dist < distance
        distance   = torch.where(mask, dist, distance)
        farthest   = distance.max(dim=-1)[1]                       # (B,)

    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points by index.

    Args:
        points: (B, N, C)
        idx:    (B, S)  or  (B, S, K)

    Returns:
        gathered: same trailing shape as idx, last dim = C
    """
    B = points.shape[0]
    device = points.device
    batch_idx = torch.arange(B, dtype=torch.long, device=device)

    if idx.dim() == 2:
        batch_idx = batch_idx.view(B, 1).expand_as(idx)
        return points[batch_idx, idx, :]

    # idx: (B, S, K)
    S, K = idx.shape[1], idx.shape[2]
    batch_idx = batch_idx.view(B, 1, 1).expand(B, S, K)
    return points[batch_idx, idx, :]        # (B, S, K, C)


def knn_query(k: int, xyz: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """
    For each point in `query`, find the k nearest neighbours in `xyz`.

    Args:
        k:     int
        xyz:   (B, N, 3)  full point set
        query: (B, S, 3)  centroid / query points

    Returns:
        idx:   (B, S, k)  indices into xyz
    """
    # Squared-distance matrix via broadcasting
    # (B, S, 1, 3) - (B, 1, N, 3) → (B, S, N)
    diff = query.unsqueeze(2) - xyz.unsqueeze(1)
    dist = (diff ** 2).sum(dim=-1)          # (B, S, N)
    idx  = dist.topk(k, dim=-1, largest=False)[1]   # (B, S, k)
    return idx


def kernel_density_estimation(xyz: torch.Tensor, k: int = 16) -> torch.Tensor:
    """
    Estimate point density at each location using the mean distance to its
    k nearest neighbours.  Returns the *reciprocal* (density coefficient S)
    so denser regions get down-weighted, as in the SCN paper.

    Args:
        xyz: (B, N, 3)
        k:   number of neighbours for KDE

    Returns:
        density_coeff: (B, N, 1)  reciprocal of local density
    """
    diff = xyz.unsqueeze(2) - xyz.unsqueeze(1)     # (B, N, N, 3)
    dist = (diff ** 2).sum(dim=-1)                 # (B, N, N)

    # exclude self (distance 0) by sorting and taking k+1 then dropping first
    knn_dist, _ = dist.topk(k + 1, dim=-1, largest=False)
    knn_dist    = knn_dist[:, :, 1:]               # drop self

    mean_dist   = knn_dist.mean(dim=-1, keepdim=True)   # (B, N, 1)
    # Reciprocal: smaller mean_dist → denser region → smaller weight
    density_inv = 1.0 / (mean_dist + 1e-8)
    return density_inv
