#!/usr/bin/env python3
"""
train_resnet_baseline.py

The ResNet-18 baseline from the project plan: "2D CNN on binary silhouette
frames." It reuses the SAME cached point clouds produced by
preprocess_forelise_pointnet.py -- no new data, no Blender, no silhouette
adapter needed -- by rasterizing each cloud's (x, y) coordinates into a 2D
binary silhouette image and training ResNet-18 on it.

Because it reads the identical cache, identical by-piece split, and identical
macro-F1 metric as train_pointnet_baseline.py, the two numbers are directly
comparable -- which is exactly what the milestone's "compare against your
baselines" asks for.

Reads:
  <cache_dir>/clouds/<sample_id>.pt   # (n_points, 3) float32, columns (x, y, t)
  <cache_dir>/labels.csv

Writes:
  <out>/best_resnet.pt
  <out>/results.json    # macro-F1, accuracy, confusion matrix, config

Run (in the same shell where your GPU canary passed, i.e. LD_LIBRARY_PATH unset):
  python3 train_resnet_baseline.py --cache_dir ~/data/processed --epochs 40

Honest caveats to carry into the writeup (same spirit as the other scripts):
  - The silhouette is the windowed points flattened into one image, so it
    aggregates the hand over the time window rather than averaging per-frame
    predictions. It captures shape + motion spread, not fine temporal order.
  - The "poor" class is synthesized by perturbing real hands, so a high F1
    partly reflects detecting the perturbation signature. Report it honestly.
  - ResNet-18 is trained from scratch (no ImageNet weights) because binary
    silhouettes are far out of ImageNet's distribution and to avoid a download.
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

try:
    from torchvision.models import resnet18
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "torchvision is required for ResNet-18. Install it into the same "
        "environment as torch:\n"
        "  python3 -m pip install --break-system-packages torchvision "
        "--index-url https://download.pytorch.org/whl/cu124\n"
        f"(import error was: {e})")


# --------------------------------------------------------------------------- #
# Data: cloud -> binary silhouette image
# --------------------------------------------------------------------------- #
def rasterize(cloud, size, thicken):
    """(N,3) cloud with columns (x,y,t) -> (1,size,size) binary image from x,y."""
    xy = cloud[:, :2]
    mn = xy.min(dim=0).values
    mx = xy.max(dim=0).values
    span = (mx - mn).clamp(min=1e-6)
    norm = (xy - mn) / span                                   # -> [0,1]
    ij = (norm * (size - 1)).round().long().clamp(0, size - 1)
    img = torch.zeros(size, size)
    img[ij[:, 0], ij[:, 1]] = 1.0
    if thicken and thicken >= 3:                              # dilate so it's not sparse
        k = thicken if thicken % 2 == 1 else thicken + 1
        img = F.max_pool2d(img[None, None], kernel_size=k, stride=1, padding=k // 2)
        img = (img[0, 0] > 0).float()[:size, :size]
    return img[None]                                          # (1,size,size)


class SilhouetteDataset(Dataset):
    def __init__(self, cache_dir, rows, size, thicken, augment=False):
        self.dir = Path(cache_dir) / "clouds"
        self.rows = rows
        self.size = size
        self.thicken = thicken
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        sid, label = self.rows[i]
        cloud = torch.load(self.dir / f"{sid}.pt").float()
        img = rasterize(cloud, self.size, self.thicken)
        if self.augment and random.random() < 0.5:
            img = torch.flip(img, dims=[2])                  # horizontal flip
        return img, label


def load_split(cache_dir, val_frac, seed):
    """Identical by-piece split to the PointNet++ trainer (no window leakage)."""
    rows_by_piece = defaultdict(list)
    with open(Path(cache_dir) / "labels.csv") as f:
        for r in csv.DictReader(f):
            rows_by_piece[r["piece_id"]].append((r["sample_id"], int(r["label"])))
    pieces = sorted(rows_by_piece)
    rng = random.Random(seed)
    rng.shuffle(pieces)
    if len(pieces) < 2:
        print("[warn] <2 pieces; sample-level split (leakage risk).")
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


def metrics(y_true, y_pred, num_classes=2):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = float((y_true == y_pred).mean())
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return acc, float(np.mean(f1s)), cm.tolist()


def build_model():
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="~/data/processed")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--thicken", type=int, default=3, help="silhouette dilation kernel")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--out", default="~/data/processed/resnet_baseline")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_rows, val_rows = load_split(cache_dir, args.val_frac, args.seed)
    print(f"[data] train={len(train_rows)}  val={len(val_rows)}  device={device}")

    train_dl = DataLoader(
        SilhouetteDataset(cache_dir, train_rows, args.img_size, args.thicken, args.augment),
        batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(
        SilhouetteDataset(cache_dir, val_rows, args.img_size, args.thicken),
        batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    crit = nn.CrossEntropyLoss()

    best = {"macro_f1": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for img, y in train_dl:
            img, y = img.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(img), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()

        model.eval(); yt, yp = [], []
        with torch.no_grad():
            for img, y in val_dl:
                yp.extend(model(img.to(device)).argmax(1).cpu().tolist())
                yt.extend(y.tolist())
        acc, macro_f1, cm = metrics(yt, yp)
        print(f"epoch {epoch:3d} | train_loss {tot/max(n,1):.4f} "
              f"| val_acc {acc:.3f} | val_macroF1 {macro_f1:.3f}")
        if macro_f1 > best["macro_f1"]:
            best = {"epoch": epoch, "macro_f1": macro_f1, "accuracy": acc,
                    "confusion_matrix": cm}
            torch.save(model.state_dict(), out_dir / "best_resnet.pt")

    best["model"] = "resnet18_silhouette"
    best["config"] = vars(args)
    best["n_train"], best["n_val"] = len(train_rows), len(val_rows)
    with open(out_dir / "results.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\n==== BEST (cite these) ====")
    print(f"  model    : ResNet-18 (silhouette)")
    print(f"  macro-F1 : {best['macro_f1']:.4f}")
    print(f"  accuracy : {best['accuracy']:.4f}  (epoch {best['epoch']})")
    print(f"  confusion matrix [rows=true 0/1, cols=pred 0/1]: {best['confusion_matrix']}")
    print(f"  saved -> {out_dir/'best_resnet.pt'} and {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
