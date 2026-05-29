"""
scripts/evaluate.py

Load a saved checkpoint and run full evaluation + comparison table.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_dscn.pt \
        --model dscn \
        --dataset cached
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dscn import SlowToFastDSCN, ResNet18Baseline, PointNetPPBaseline
from data.dataset import CachedPointCloudDataset
from torch.utils.data import DataLoader
from configs.config import *


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model",      default="dscn", choices=["dscn", "pointnet", "resnet"])
    p.add_argument("--cache_dir",  default="data/processed")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--split",      default="test")
    return p.parse_args()


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for xyz, labels in loader:
        xyz    = xyz.to(device)
        logits = model(xyz)
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())

    return all_preds, all_labels, all_probs


def print_metrics(preds, labels, model_name="Model"):
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(labels, preds, target_names=["Good", "Poor"]))
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(cm)

    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="macro")
    prec = precision_score(labels, preds, average="macro")
    rec  = recall_score(labels, preds, average="macro")
    print(f"\nAccuracy={acc:.4f}  F1={f1:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec}


def plot_confusion_matrix(preds, labels, model_name, out_path="confusion.png"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Good", "Poor"])
    ax.set_yticklabels(["Good", "Poor"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix → {out_path}")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    ds     = CachedPointCloudDataset(cache_dir=args.cache_dir, split=args.split, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    if args.model == "dscn":
        model = SlowToFastDSCN(num_classes=NUM_CLASSES, dropout=0.0)
    elif args.model == "pointnet":
        model = PointNetPPBaseline(num_classes=NUM_CLASSES)
    elif args.model == "resnet":
        model = ResNet18Baseline(num_classes=NUM_CLASSES)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    preds, labels, probs = run_eval(model, loader, device)
    metrics = print_metrics(preds, labels, model_name=args.model.upper())
    plot_confusion_matrix(preds, labels, args.model.upper(),
                          out_path=f"confusion_{args.model}.png")


if __name__ == "__main__":
    main()
