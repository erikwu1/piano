"""
scripts/preprocess.py

Converts raw clip directories (Blender masks or video files) into cached
.pt tensors so training loops aren't bottlenecked by silhouette extraction.

Usage:
    python scripts/preprocess.py --dataset synthetic --data_root data/raw/synthetic
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.silhouette import mask_to_boundary, frames_to_point_cloud
from utils.point_ops import farthest_point_sample
from configs.config import NUM_POINTS, NUM_POINTS, ALPHA, T_FRAMES
import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   default="synthetic", choices=["synthetic"])
    p.add_argument("--data_root", default="data/raw/synthetic")
    p.add_argument("--out_dir",   default="data/processed")
    p.add_argument("--n_points",  type=int, default=NUM_POINTS)
    p.add_argument("--n_contour", type=int, default=256)
    p.add_argument("--alpha",     type=float, default=ALPHA)
    p.add_argument("--split",     nargs="+", default=["train", "val", "test"])
    return p.parse_args()


def process_clip(clip_path: str, n_points: int, n_contour: int, alpha: float):
    """Load mask images from clip_path and return a (n_points, 3) tensor."""
    frame_files = sorted(Path(clip_path).glob("frame_*.png"))
    if not frame_files:
        return None

    masks = []
    for fp in frame_files:
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, bm = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        masks.append(bm)

    if not masks:
        return None

    raw_pc = frames_to_point_cloud(
        frames_rgb=None,
        masks=masks,
        n_contour_pts=n_contour,
        alpha=alpha,
    )

    if raw_pc is None or len(raw_pc) < 4:
        return torch.zeros(n_points, 3)

    pc_t    = torch.from_numpy(raw_pc).unsqueeze(0).float()
    fps_idx = farthest_point_sample(pc_t, n_points)[0]
    return pc_t[0][fps_idx]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    manifest_path = os.path.join(args.data_root, "labels.json")
    with open(manifest_path) as f:
        all_clips = json.load(f)   # [(clip_path, label)]

    # Deterministic split
    import random; random.seed(42); random.shuffle(all_clips)
    n = len(all_clips)
    splits = {
        "train": all_clips[:int(n * 0.70)],
        "val":   all_clips[int(n * 0.70):int(n * 0.85)],
        "test":  all_clips[int(n * 0.85):],
    }

    for split_name in args.split:
        clips = splits[split_name]
        print(f"\n[{split_name}] Processing {len(clips)} clips ...")
        split_dir  = os.path.join(args.out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        manifest   = []

        for i, (clip_path, label) in enumerate(tqdm(clips)):
            pt_path = os.path.join(split_dir, f"sample_{i:05d}.pt")
            if os.path.exists(pt_path):
                manifest.append({"path": pt_path, "label": label})
                continue

            xyz = process_clip(clip_path, args.n_points, args.n_contour, args.alpha)
            if xyz is None:
                print(f"  WARNING: skipping {clip_path}")
                continue

            torch.save({"xyz": xyz, "label": label}, pt_path)
            manifest.append({"path": pt_path, "label": label})

        manifest_out = os.path.join(args.out_dir, f"{split_name}_manifest.json")
        with open(manifest_out, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Saved {len(manifest)} samples → {manifest_out}")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
