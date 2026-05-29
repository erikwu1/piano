#!/usr/bin/env python3
"""
train_pointnet_baseline.py

Standalone PointNet++ (single-scale grouping) baseline for good-vs-poor piano
technique, trained on the cached point clouds produced by
preprocess_forelise_pointnet.py.

Reads:
  <cache_dir>/clouds/<sample_id>.pt   # float32 tensor, shape (n_points, 3)
  <cache_dir>/labels.csv              # sample_id, label, piece_id, start_frame, kind

Does:
  - Splits BY PIECE (not by sample) so windows from one piece never straddle the
    train/val boundary -- otherwise the metric is inflated by leakage.
  - Trains a real PointNet++ SSG classifier (FPS + kNN grouping + mini-PointNet
    set-abstraction layers), implemented in pure PyTorch (no torch-cluster).
  - Reports macro-F1 (the milestone metric), accuracy, and a confusion matrix
    on the held-out pieces, and saves the best checkpoint + a results.json you
    can cite.

No external deps beyond torch/numpy (both already on your DL image).

Run:
  python3 train_pointnet_baseline.py --cache_dir ~/data/processed --epochs 40

Honest caveats (carry these into your writeup):
  - The negative class is synthesized by perturbing real captured hands, so the
    model can partly exploit the perturbation signature rather than learning
    technique in the abstract. Report it as such.
  - This PointNet++ runs on raw 3D-derived points, which differs from a
    silhouette-based DSCN input -- it's a reasonable baseline, not a perfectly
    controlled ablation. See the preprocessing script's header for the full note.
  - With only a handful of pieces the val metric is high-variance; extract more
    pieces for a more trustworthy number.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------- #
# PointNet++ helpers (pure PyTorch)
# --------------------------------------------------------------------------- #
def square_distance(src, dst):
    """Pairwise squared distances. src (B,N,C), dst (B,M,C) -> (B,N,M)."""
    B, N, _ = src.shape
    M = dst.shape[1]
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """Gather points by index. points (B,N,C); idx (B,S) or (B,S,K)."""
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    repeat_shape = [1] + list(idx.shape[1:])
    batch_indices = (torch.arange(B, dtype=torch.long, device=points.device)
                     .view(view_shape).repeat(repeat_shape))
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """FPS. xyz (B,N,C) -> centroid indices (B,npoint)."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn(nsample, xyz, new_xyz):
    """k nearest neighbours: indices (B,S,nsample) of xyz around new_xyz."""
    dists = square_distance(new_xyz, xyz)
    return dists.topk(nsample, dim=-1, largest=False)[1]


class SetAbstraction(nn.Module):
    """One PointNet++ SSG set-abstraction layer (kNN grouping)."""

    def __init__(self, npoint, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.group_all = group_all
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        last = in_channel
        for out in mlp:
            self.convs.append(nn.Conv2d(last, out, 1))
            self.bns.append(nn.BatchNorm2d(out))
            last = out

    def forward(self, xyz, points):
        # xyz (B,N,3); points (B,N,D) or None
        B, N, C = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, C, device=xyz.device)
            grouped_xyz = xyz.view(B, 1, N, C)
            if points is not None:
                grouped = torch.cat([points.view(B, 1, N, -1), grouped_xyz], dim=-1)
            else:
                grouped = grouped_xyz
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)                 # (B,S,3)
            idx = knn(self.nsample, xyz, new_xyz)                # (B,S,nsample)
            grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_pts = index_points(points, idx)
                grouped = torch.cat([grouped_pts, grouped_xyz], dim=-1)
            else:
                grouped = grouped_xyz                            # (B,S,nsample,3)

        grouped = grouped.permute(0, 3, 2, 1)                    # (B,C,K,S)
        for conv, bn in zip(self.convs, self.bns):
            grouped = F.relu(bn(conv(grouped)))
        new_points = torch.max(grouped, 2)[0].permute(0, 2, 1)   # (B,S,mlp[-1])
        return new_xyz, new_points


class PointNet2(nn.Module):
    """PointNet++ SSG classification network for (B,N,3) inputs."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.sa1 = SetAbstraction(512, 32, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = SetAbstraction(128, 32, in_channel=128 + 3, mlp=[128, 128, 256])
        self.sa3 = SetAbstraction(None, None, in_channel=256 + 3,
                                  mlp=[256, 512, 1024], group_all=True)
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, xyz):
        l1_xyz, l1_pts = self.sa1(xyz, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _, l3_pts = self.sa3(l2_xyz, l2_pts)
        return self.head(l3_pts.view(xyz.shape[0], -1))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class CloudDataset(Dataset):
    def __init__(self, cache_dir, rows, n_points, augment=False):
        self.dir = Path(cache_dir) / "clouds"
        self.rows = rows
        self.n_points = n_points
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        sid, label = self.rows[i]
        pts = torch.load(self.dir / f"{sid}.pt").float()         # (N,3)
        N = pts.shape[0]
        if N >= self.n_points:
            sel = torch.randperm(N)[: self.n_points]
        else:
            sel = torch.randint(0, N, (self.n_points,))
        pts = pts[sel]
        if self.augment:
            pts = pts + torch.randn_like(pts) * 0.01             # small jitter
        return pts, label


def load_split(cache_dir, val_frac, seed):
    rows_by_piece = defaultdict(list)
    with open(Path(cache_dir) / "labels.csv") as f:
        for r in csv.DictReader(f):
            rows_by_piece[r["piece_id"]].append((r["sample_id"], int(r["label"])))

    pieces = sorted(rows_by_piece)
    rng = random.Random(seed)
    rng.shuffle(pieces)

    if len(pieces) < 2:
        # fallback: not enough pieces to split by piece -> split by sample
        print("[warn] <2 pieces; falling back to a sample-level split (leakage risk).")
        all_rows = [r for p in pieces for r in rows_by_piece[p]]
        rng.shuffle(all_rows)
        k = max(1, int(len(all_rows) * val_frac))
        return all_rows[k:], all_rows[:k]

    n_val = max(1, int(round(len(pieces) * val_frac)))
    val_pieces = set(pieces[:n_val])
    train, val = [], []
    for p in pieces:
        (val if p in val_pieces else train).extend(rows_by_piece[p])
    print(f"[split] pieces total={len(pieces)}  val_pieces={sorted(val_pieces)}")
    return train, val


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred, num_classes=2):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = float((y_true == y_pred).mean())
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return acc, float(np.mean(f1s)), cm.tolist()


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="~/data/processed")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_points", type=int, default=1024,
                    help="points per cloud fed to the net (subsampled from cache)")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--out", default="~/data/processed/pointnet_baseline")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows, val_rows = load_split(cache_dir, args.val_frac, args.seed)
    print(f"[data] train={len(train_rows)}  val={len(val_rows)}  device={device}")

    train_dl = DataLoader(
        CloudDataset(cache_dir, train_rows, args.n_points, augment=args.augment),
        batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(
        CloudDataset(cache_dir, val_rows, args.n_points),
        batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = PointNet2(num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    crit = nn.CrossEntropyLoss()

    best = {"macro_f1": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for pts, y in train_dl:
            pts, y = pts.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(pts), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()

        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for pts, y in val_dl:
                logits = model(pts.to(device))
                yp.extend(logits.argmax(1).cpu().tolist())
                yt.extend(y.tolist())
        acc, macro_f1, cm = metrics(yt, yp)
        print(f"epoch {epoch:3d} | train_loss {tot / max(n,1):.4f} "
              f"| val_acc {acc:.3f} | val_macroF1 {macro_f1:.3f}")

        if macro_f1 > best["macro_f1"]:
            best = {"epoch": epoch, "macro_f1": macro_f1, "accuracy": acc,
                    "confusion_matrix": cm}
            torch.save(model.state_dict(), out_dir / "best_pointnet.pt")

    best["config"] = vars(args)
    best["n_train"], best["n_val"] = len(train_rows), len(val_rows)
    with open(out_dir / "results.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\n==== BEST (cite these) ====")
    print(f"  macro-F1 : {best['macro_f1']:.4f}")
    print(f"  accuracy : {best['accuracy']:.4f}  (at epoch {best['epoch']})")
    print(f"  confusion matrix [rows=true 0/1, cols=pred 0/1]: {best['confusion_matrix']}")
    print(f"  saved -> {out_dir/'best_pointnet.pt'} and {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
