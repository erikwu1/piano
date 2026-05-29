"""
scripts/train.py

Main training script for the Piano Technique DSCN.

Usage (inside tmux on GCP):
    python scripts/train.py --dataset synthetic --batch_size 32 --epochs 80

    # Or with cached tensors (fastest):
    python scripts/train.py --dataset cached --cache_dir data/processed
"""

import os
import sys
import time
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, classification_report

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import *
from models.dscn import SlowToFastDSCN, ResNet18Baseline, PointNetPPBaseline
from data.dataset import SyntheticHandDataset, CachedPointCloudDataset, get_dataloaders


# ─── Argument parser ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Piano Technique DSCN")
    p.add_argument("--dataset",     default="synthetic", choices=["synthetic", "cached", "video"])
    p.add_argument("--data_root",   default="data/raw/synthetic")
    p.add_argument("--cache_dir",   default="data/processed")
    p.add_argument("--model",       default="dscn",   choices=["dscn", "pointnet", "resnet"])
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--epochs",      type=int,   default=NUM_EPOCHS)
    p.add_argument("--lr",          type=float, default=LR_INIT)
    p.add_argument("--n_points",    type=int,   default=NUM_POINTS)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--resume",      default=None, help="path to checkpoint to resume from")
    p.add_argument("--tag",         default="",   help="experiment tag for logging")
    return p.parse_args()


# ─── Logging setup ───────────────────────────────────────────────────────────

def setup_logging(log_dir: str, tag: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{tag}_{int(time.time())}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ─── Build model ─────────────────────────────────────────────────────────────

def build_model(args) -> nn.Module:
    if args.model == "dscn":
        return SlowToFastDSCN(
            n_pts_l1=FPS_NPOINTS_L1,
            n_pts_l2=FPS_NPOINTS_L2,
            k=KNN_K,
            channels=HIDDEN_CHANNELS,
            dilations=DILATIONS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
            n_temporal_scales=len(TEMPORAL_SCALES),
        )
    elif args.model == "pointnet":
        return PointNetPPBaseline(num_classes=NUM_CLASSES)
    elif args.model == "resnet":
        return ResNet18Baseline(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {args.model}")


# ─── Build dataloaders ────────────────────────────────────────────────────────

def build_loaders(args):
    if args.dataset == "synthetic":
        return get_dataloaders(
            SyntheticHandDataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            root=args.data_root,
            n_points=args.n_points,
            cache_dir=args.cache_dir,
        )
    elif args.dataset == "cached":
        return get_dataloaders(
            CachedPointCloudDataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
        )
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' not yet wired up here.")


# ─── LR schedule ─────────────────────────────────────────────────────────────

def adjust_lr(optimizer, epoch: int, args):
    if epoch >= EPOCH_END:
        lr = LR_END
    elif epoch >= EPOCH_MID:
        lr = LR_MID
    else:
        lr = args.lr
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ─── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for xyz, labels in loader:
        xyz    = xyz.to(device)
        labels = labels.to(device)

        logits = model(xyz)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1, all_preds, all_labels


# ─── Training loop ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, epoch, logger):
    model.train()
    total_loss = 0.0
    n_correct  = 0

    for batch_idx, (xyz, labels) in enumerate(loader):
        xyz    = xyz.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(xyz)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        n_correct  += (logits.argmax(-1) == labels).sum().item()

        if batch_idx % 20 == 0:
            logger.info(
                f"Epoch {epoch:03d} | batch {batch_idx:04d}/{len(loader)} "
                f"| loss {loss.item():.4f}"
            )

    avg_loss = total_loss / len(loader.dataset)
    acc      = n_correct  / len(loader.dataset)
    return avg_loss, acc


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    logger = setup_logging(LOG_DIR, args.tag or args.model)
    logger.info(f"Args: {vars(args)}")

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Data ──
    loaders = build_loaders(args)
    logger.info(
        f"Dataset sizes – train: {len(loaders['train'].dataset)}, "
        f"val: {len(loaders['val'].dataset)}, "
        f"test: {len(loaders['test'].dataset)}"
    )

    # ── Model ──
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model} | Parameters: {n_params:,}")

    # ── Optimizer & loss ──
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # ── TensorBoard ──
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, "tb", args.model + args.tag))

    # ── Resume ──
    start_epoch = 0
    best_f1     = 0.0
    if args.resume:
        ckpt        = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_f1     = ckpt.get("best_f1", 0.0)
        logger.info(f"Resumed from epoch {start_epoch} (best F1 = {best_f1:.4f})")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_lr(optimizer, epoch, args)
        logger.info(f"─── Epoch {epoch:03d}/{args.epochs} | LR={lr:.6f} ───")

        t0           = time.time()
        train_loss, train_acc = train_epoch(
            model, loaders["train"], optimizer, criterion, device, epoch, logger
        )
        val_loss, val_f1, _, _ = evaluate(model, loaders["val"], device, criterion)
        elapsed      = time.time() - t0

        logger.info(
            f"Epoch {epoch:03d} done in {elapsed:.1f}s | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_F1={val_f1:.4f}"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("F1/val",     val_f1,     epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("LR",         lr,         epoch)

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "best_f1": best_f1},
                os.path.join(CHECKPOINT_DIR, f"best_{args.model}.pt"),
            )
            logger.info(f"  ✓ New best model saved (F1={best_f1:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "best_f1": best_f1},
                os.path.join(CHECKPOINT_DIR, f"{args.model}_epoch{epoch:03d}.pt"),
            )

    # ── Final test evaluation ──
    best_ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, f"best_{args.model}.pt"), map_location=device
    )
    model.load_state_dict(best_ckpt["model"])
    test_loss, test_f1, test_preds, test_labels = evaluate(
        model, loaders["test"], device, criterion
    )
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST RESULTS | loss={test_loss:.4f} | F1={test_f1:.4f}")
    logger.info("\n" + classification_report(test_labels, test_preds,
                                              target_names=["Good", "Poor"]))
    writer.close()


if __name__ == "__main__":
    main()
