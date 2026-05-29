"""
data/dataset.py

PyTorch Dataset classes for piano technique detection.

Supports three data sources:
  1. SyntheticHandDataset  – rendered Blender masks (data/raw/synthetic/)
  2. PianoVideoDataset     – real video clips with MediaPipe silhouette extraction
  3. CachedPointCloudDataset – pre-processed .pt tensors (fastest, use for training)
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from utils.silhouette import mask_to_boundary, frames_to_point_cloud, HandSilhouetteExtractor
from utils.point_ops import farthest_point_sample


# ─── 1. Synthetic (Blender) Dataset ──────────────────────────────────────────

class SyntheticHandDataset(Dataset):
    """
    Loads rendered mask images from the Blender pipeline.

    Directory structure expected:
        data/raw/synthetic/
            labels.json            [[(clip_path, label), ...]]
            good/clip_0000/frame_0000.png
            poor/clip_0000/frame_0000.png
    """

    def __init__(
        self,
        root: str = "data/raw/synthetic",
        split: str = "train",           # train | val | test
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        n_points: int = 4096,
        n_contour_pts: int = 256,
        alpha: float = 5.0,
        seed: int = 42,
        cache_dir: Optional[str] = "data/processed",
    ):
        super().__init__()
        self.n_points      = n_points
        self.n_contour_pts = n_contour_pts
        self.alpha         = alpha
        self.cache_dir     = cache_dir

        manifest = os.path.join(root, "labels.json")
        with open(manifest) as f:
            all_clips = json.load(f)   # [(clip_path, label), ...]

        random.seed(seed)
        random.shuffle(all_clips)

        n = len(all_clips)
        n_train = int(n * split_ratio[0])
        n_val   = int(n * split_ratio[1])

        if split == "train":
            self.clips = all_clips[:n_train]
        elif split == "val":
            self.clips = all_clips[n_train:n_train + n_val]
        else:
            self.clips = all_clips[n_train + n_val:]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip_path, label = self.clips[idx]

        # Check for cached .pt
        if self.cache_dir:
            cache_key = clip_path.replace("/", "_").replace("\\", "_")
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
            if os.path.exists(cache_path):
                xyz = torch.load(cache_path)
                return xyz, label

        # Load mask images
        frame_files = sorted(Path(clip_path).glob("frame_*.png"))
        masks = []
        for fp in frame_files:
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            _, bm = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            masks.append(bm)

        # Build stacked point cloud
        raw_pc = frames_to_point_cloud(
            frames_rgb=None,
            masks=masks,
            n_contour_pts=self.n_contour_pts,
            alpha=self.alpha,
        )

        if raw_pc is None or len(raw_pc) < 4:
            xyz = torch.zeros(self.n_points, 3)
        else:
            # FPS to fixed number of points
            pc_t    = torch.from_numpy(raw_pc).unsqueeze(0).float()  # (1, M, 3)
            fps_idx = farthest_point_sample(pc_t, self.n_points)[0]   # (n_points,)
            xyz     = pc_t[0][fps_idx]                                 # (n_points, 3)

        # Cache for future epochs
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(xyz, cache_path)

        return xyz, label


# ─── 2. Real Video Dataset ───────────────────────────────────────────────────

class PianoVideoDataset(Dataset):
    """
    Processes real video clips through MediaPipe to extract silhouettes.

    Expected structure:
        data/raw/videos/
            labels.csv   (clip_path, label)
            good/
            poor/
    """

    def __init__(
        self,
        csv_path: str = "data/raw/videos/labels.csv",
        split: str = "train",
        n_points: int = 4096,
        n_contour_pts: int = 256,
        t_frames: int = 32,
        alpha: float = 5.0,
        cache_dir: Optional[str] = "data/processed",
    ):
        import csv
        self.n_points      = n_points
        self.n_contour_pts = n_contour_pts
        self.t_frames      = t_frames
        self.alpha         = alpha
        self.cache_dir     = cache_dir

        with open(csv_path) as f:
            reader = csv.reader(f)
            all_clips = [(row[0], int(row[1])) for row in reader]

        n = len(all_clips)
        n_train = int(n * 0.7)
        n_val   = int(n * 0.15)

        if split == "train":
            self.clips = all_clips[:n_train]
        elif split == "val":
            self.clips = all_clips[n_train:n_train + n_val]
        else:
            self.clips = all_clips[n_train + n_val:]

        self.extractor = HandSilhouetteExtractor()

    def __len__(self):
        return len(self.clips)

    def _sample_frames(self, video_path: str, t: int) -> List[np.ndarray]:
        """Uniformly sample t frames from a video file."""
        cap    = cv2.VideoCapture(video_path)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs   = np.linspace(0, total - 1, t, dtype=int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def __getitem__(self, idx: int):
        video_path, label = self.clips[idx]

        if self.cache_dir:
            cache_key = video_path.replace("/", "_").replace("\\", "_")
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path), label

        frames = self._sample_frames(video_path, self.t_frames)
        raw_pc = frames_to_point_cloud(
            frames_rgb=frames,
            extractor=self.extractor,
            n_contour_pts=self.n_contour_pts,
            alpha=self.alpha,
        )

        if raw_pc is None or len(raw_pc) < 4:
            xyz = torch.zeros(self.n_points, 3)
        else:
            pc_t    = torch.from_numpy(raw_pc).unsqueeze(0).float()
            fps_idx = farthest_point_sample(pc_t, self.n_points)[0]
            xyz     = pc_t[0][fps_idx]

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(xyz, cache_path)

        return xyz, label


# ─── 3. Pre-cached Dataset (fastest for training loops) ─────────────────────

class CachedPointCloudDataset(Dataset):
    """
    Loads pre-processed .pt tensors from disk.
    Each .pt file is a dict {"xyz": Tensor(N,3), "label": int}.

    Run `scripts/preprocess.py` first to build the cache.
    """

    def __init__(
        self,
        cache_dir: str = "data/processed",
        split: str = "train",
        augment: bool = True,
    ):
        self.augment = augment and (split == "train")

        manifest = os.path.join(cache_dir, f"{split}_manifest.json")
        with open(manifest) as f:
            self.entries = json.load(f)   # [{"path": ..., "label": ...}]

    def __len__(self):
        return len(self.entries)

    def _augment(self, xyz: torch.Tensor) -> torch.Tensor:
        """Random jitter + flip for training augmentation."""
        # Jitter
        xyz = xyz + torch.randn_like(xyz) * 0.005
        # Random x-flip (mirror the hand)
        if random.random() > 0.5:
            xyz[:, 0] = 1.0 - xyz[:, 0]
        # Random z-axis (temporal) shift
        shift = random.uniform(-0.05, 0.05)
        xyz[:, 2] = (xyz[:, 2] + shift).clamp(0, 1)
        return xyz

    def __getitem__(self, idx: int):
        entry  = self.entries[idx]
        data   = torch.load(entry["path"])
        xyz    = data["xyz"].float()
        label  = data["label"]
        if self.augment:
            xyz = self._augment(xyz)
        return xyz, label


# ─── DataLoader factory ──────────────────────────────────────────────────────

def get_dataloaders(
    dataset_cls,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs,
):
    """
    Convenience function to build train/val/test DataLoaders.
    """
    loaders = {}
    for split in ("train", "val", "test"):
        ds = dataset_cls(split=split, **dataset_kwargs)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
