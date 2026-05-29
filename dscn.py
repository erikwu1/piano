"""
models/dscn.py

Dilated Silhouette Convolutional Network (DSCN) for piano technique classification.

Architecture follows Hua et al. (2021) SCN paper, adapted for:
  - Two-class piano technique detection (good=0 / poor=1)
  - Two-handed hand silhouettes (as a single merged point cloud)
  - Multi-scale Slow-to-Fast temporal branches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from utils.point_ops import (
    farthest_point_sample,
    index_points,
    knn_query,
    kernel_density_estimation,
)


# ─── Dilated Silhouette Convolution Block ────────────────────────────────────

class DilatedSilhouetteConv(nn.Module):
    """
    One dilated silhouette convolution layer.

    For each centroid, we:
      1. Extract the K-NN local neighbourhood.
      2. Convert to local coordinates.
      3. Part A: MLP on local coords → weight W  (shape branch)
      4. Part B: MLP on density coefficients → S  (density branch)
      5. Feature = sum over neighbours of  S * W * F_neighbour
      6. Max-pool over neighbours to get a single vector per centroid.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 16,
        dilation: int = 1,
    ):
        super().__init__()
        self.k        = k
        self.dilation = dilation

        # Part A – local coordinate branch
        self.mlp_coord = nn.Sequential(
            nn.Conv1d(3, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Part B – density branch
        self.mlp_density = nn.Sequential(
            nn.Conv1d(1, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Feature transform after combining coord weights, density, and input features
        self.mlp_feat = nn.Sequential(
            nn.Conv1d(in_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        xyz:      torch.Tensor,   # (B, N, 3) – all point positions
        features: torch.Tensor,   # (B, N, C_in) – point features
        centroids_idx: torch.Tensor,  # (B, S) – FPS centroid indices
    ) -> tuple:
        """
        Returns:
            centroid_xyz:  (B, S, 3)
            centroid_feat: (B, S, C_out)
        """
        B, N, _ = xyz.shape
        S = centroids_idx.shape[1]
        k_eff = self.k * self.dilation   # query more neighbours, then subsample

        centroid_xyz  = index_points(xyz, centroids_idx)              # (B, S, 3)
        centroid_feat = index_points(features, centroids_idx)         # (B, S, C_in)

        # KNN in the full point cloud
        nn_idx = knn_query(k_eff, xyz, centroid_xyz)                  # (B, S, k_eff)

        # Dilated subsampling: take every `dilation`-th neighbour
        if self.dilation > 1:
            nn_idx = nn_idx[:, :, ::self.dilation][:, :, :self.k]    # (B, S, k)

        nn_xyz   = index_points(xyz, nn_idx)                          # (B, S, k, 3)
        nn_feats = index_points(features, nn_idx)                     # (B, S, k, C_in)

        # Local coordinates
        local_xyz = nn_xyz - centroid_xyz.unsqueeze(2)                # (B, S, k, 3)

        # Density coefficient for each neighbour (pre-computed globally)
        density_coeff = kernel_density_estimation(xyz, k=self.k)      # (B, N, 1)
        nn_density    = index_points(density_coeff, nn_idx)           # (B, S, k, 1)

        # Reshape for Conv1d: merge (B, S) → treat k as the "sequence" axis
        # local_xyz: (B*S, 3, k)
        B_S = B * S
        local_xyz_flat   = local_xyz.view(B_S, self.k, 3).permute(0, 2, 1)     # (B*S, 3, k)
        nn_density_flat  = nn_density.view(B_S, self.k, 1).permute(0, 2, 1)    # (B*S, 1, k)
        nn_feats_flat    = nn_feats.view(B_S, self.k, -1).permute(0, 2, 1)     # (B*S, C_in, k)

        W = self.mlp_coord(local_xyz_flat)       # (B*S, C_out, k)
        S_weight = self.mlp_density(nn_density_flat)   # (B*S, C_out, k)

        # Combine: density-weighted feature aggregation
        combined = torch.cat([nn_feats_flat, W * S_weight], dim=1)    # (B*S, C_in+C_out, k)
        feat_out = self.mlp_feat(combined)                             # (B*S, C_out, k)

        # Max-pool over neighbours
        feat_out = feat_out.max(dim=-1)[0]                             # (B*S, C_out)
        feat_out = feat_out.view(B, S, -1)                             # (B, S, C_out)

        return centroid_xyz, feat_out


# ─── Single-Scale SCN Encoder ────────────────────────────────────────────────

class SCNEncoder(nn.Module):
    """
    Two-layer hierarchical dilated silhouette encoder (Fig. 5 of paper).
    """

    def __init__(
        self,
        n_pts_l1: int = 512,
        n_pts_l2: int = 128,
        k: int = 16,
        in_channels: int = 3,
        channels: List[int] = [64, 128, 256],
        dilations: List[int] = [1, 2],
    ):
        super().__init__()
        self.n_pts_l1 = n_pts_l1
        self.n_pts_l2 = n_pts_l2

        # Layer 1 – two dilation branches, fused
        self.conv1_d1 = DilatedSilhouetteConv(in_channels, channels[0], k=k, dilation=dilations[0])
        self.conv1_d2 = DilatedSilhouetteConv(in_channels, channels[0], k=k, dilation=dilations[1])

        # Layer 2 – two dilation branches, fused
        c1_out = channels[0] * 2  # fused from d1+d2
        self.conv2_d1 = DilatedSilhouetteConv(c1_out, channels[1], k=k, dilation=dilations[0])
        self.conv2_d2 = DilatedSilhouetteConv(c1_out, channels[1], k=k, dilation=dilations[1])

        # Final 1×1 projection
        c2_out = channels[1] * 2
        self.proj = nn.Sequential(
            nn.Conv1d(c2_out, channels[2], 1, bias=False),
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(inplace=True),
        )

        self.out_channels = channels[2]

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3)

        Returns:
            feat: (B, out_channels)  global feature vector
        """
        B, N, _ = xyz.shape

        # Initial features = raw xyz coordinates
        features = xyz.clone()    # (B, N, 3)

        # ── Layer 1 ──
        idx_l1 = farthest_point_sample(xyz, self.n_pts_l1)        # (B, S1)
        xyz1_d1, f1_d1 = self.conv1_d1(xyz, features, idx_l1)
        _,       f1_d2 = self.conv1_d2(xyz, features, idx_l1)
        f1 = torch.cat([f1_d1, f1_d2], dim=-1)                    # (B, S1, 2*C0)
        xyz1 = xyz1_d1

        # ── Layer 2 ──
        idx_l2 = farthest_point_sample(xyz1, self.n_pts_l2)       # (B, S2)
        xyz2_d1, f2_d1 = self.conv2_d1(xyz1, f1, idx_l2)
        _,       f2_d2 = self.conv2_d2(xyz1, f1, idx_l2)
        f2 = torch.cat([f2_d1, f2_d2], dim=-1)                    # (B, S2, 2*C1)
        xyz2 = xyz2_d1

        # 1×1 projection, then global max-pool
        f2_t = f2.permute(0, 2, 1)                                # (B, 2*C1, S2)
        f3   = self.proj(f2_t)                                     # (B, C2, S2)
        feat = f3.max(dim=-1)[0]                                   # (B, C2)

        return feat


# ─── Slow-to-Fast Multi-Scale DSCN ──────────────────────────────────────────

class SlowToFastDSCN(nn.Module):
    """
    Multi-scale SCN: three temporal sub-samplings of the stacked silhouette
    point cloud are encoded independently and fused before classification.

    Temporal scales: slow (all frames), faster (every 2nd), fastest (every 3rd).
    The z-coordinate encodes frame index, so sub-sampling is just filtering
    by the frame's z-bucket.
    """

    def __init__(
        self,
        n_pts_l1: int = 512,
        n_pts_l2: int = 128,
        k: int = 16,
        channels: List[int] = [64, 128, 256],
        dilations: List[int] = [1, 2],
        num_classes: int = 2,
        dropout: float = 0.3,
        n_temporal_scales: int = 3,
    ):
        super().__init__()

        self.n_temporal_scales = n_temporal_scales

        self.encoders = nn.ModuleList([
            SCNEncoder(
                n_pts_l1=n_pts_l1,
                n_pts_l2=n_pts_l2,
                k=k,
                in_channels=3,
                channels=channels,
                dilations=dilations,
            )
            for _ in range(n_temporal_scales)
        ])

        feat_dim = channels[-1] * n_temporal_scales

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _temporal_subsample(
        self, xyz: torch.Tensor, scale: int, alpha: float = 5.0
    ) -> torch.Tensor:
        """
        Sub-sample the point cloud to only include every `scale`-th frame's points.
        The frame index is encoded as z ∈ [0, alpha].
        We bin z-values and keep bins 0, scale, 2*scale, ...
        """
        if scale == 1:
            return xyz

        z_vals = xyz[:, :, 2]   # (B, N)
        # Invert z back to approximate frame index (0..T-1)
        # alpha / T used during construction; we just keep every `scale`-th bin
        # by using modular arithmetic on a discrete approximation
        n_bins  = 64   # assumes at most 64 distinct z-levels
        bin_idx = (z_vals * n_bins / (xyz[:, :, 2].max(dim=-1, keepdim=True)[0] + 1e-8)).long()
        mask    = (bin_idx % scale == 0)   # (B, N)

        # For a batch, we must return equal-length tensors → keep top-K masked pts
        # Use a safe gather: pad with the first point if needed
        keep_n = xyz.shape[1] // scale
        results = []
        for b in range(xyz.shape[0]):
            pts_b    = xyz[b]                    # (N, 3)
            mask_b   = mask[b]                   # (N,)
            selected = pts_b[mask_b]             # (M, 3)
            if selected.shape[0] >= keep_n:
                selected = selected[:keep_n]
            else:
                pad = pts_b[:keep_n - selected.shape[0]]
                selected = torch.cat([selected, pad], dim=0)
            results.append(selected)

        return torch.stack(results, dim=0)       # (B, keep_n, 3)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3)  stacked silhouette point cloud

        Returns:
            logits: (B, num_classes)
        """
        feats = []
        for s, encoder in enumerate(self.encoders):
            scale    = s + 1  # 1, 2, 3
            sub_xyz  = self._temporal_subsample(xyz, scale)
            feat     = encoder(sub_xyz)
            feats.append(feat)

        fused  = torch.cat(feats, dim=-1)     # (B, n_scales * C2)
        logits = self.classifier(fused)
        return logits


# ─── Baseline: ResNet18 Frame-Average ────────────────────────────────────────

class ResNet18Baseline(nn.Module):
    """
    Baseline A (from milestone): pass 2D binary silhouette frames through
    ResNet-18 and average frame predictions.
    Input is expected as a (B, T, 1, H, W) tensor of binary silhouette frames.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace first conv for 1-channel (silhouette) input
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = backbone.fc.in_features
        backbone.fc  = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 1, H, W)"""
        B, T, C, H, W = x.shape
        x_flat  = x.view(B * T, C, H, W)
        logits  = self.backbone(x_flat)           # (B*T, num_classes)
        logits  = logits.view(B, T, -1).mean(1)  # average over frames
        return logits


# ─── Baseline: PointNet++ ────────────────────────────────────────────────────

class PointNetPPBaseline(nn.Module):
    """
    Baseline B: standard PointNet++-style network without dilation.
    Identical architecture to SCNEncoder but all dilations=1.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = SCNEncoder(dilations=[1, 1])
        feat_dim = self.encoder.out_channels
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        feat   = self.encoder(xyz)
        logits = self.classifier(feat)
        return logits
