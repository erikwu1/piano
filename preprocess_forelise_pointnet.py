#!/usr/bin/env python3
"""
preprocess_forelise_pointnet.py

Turn the FürElise 3D hand-motion dataset (Wang et al., SIGGRAPH Asia 2024) into
spatio-temporal point-cloud tensors for a PointNet++ good-vs-poor-technique
baseline -- WITHOUT any mesh->silhouette 2D projection.

Pipeline per sample:
  1. Load a piece's motion.pkl  -> per-frame MANO mesh vertices (N, 778, 3)
     for left and right hands.
  2. Slide a T-frame window over the piece.
  3. Build a point cloud from the windowed vertices:
        - mode "xyt": (x, y, scaled_frame_index)  -> mirrors the DSCN input
                       (2 spatial axes + 1 temporal axis). Depth is dropped.
        - mode "xyz": (x, y, z) with all frames pooled (keeps depth, time is
                       implicit). Use this if you'd rather retain 3D depth.
  4. FPS-downsample to n_points.
  5. Emit TWO labelled samples from each window:
        - GOOD  (label 1): the real captured hand, lightly orientation-jittered.
        - POOR  (label 0): the same window with poor-technique perturbations
                            (wrist collapse / tilt, hand-size scaling, palm
                            flattening) applied consistently across the window.
     FürElise contains only elite pianists, so the negative class MUST be
     manufactured -- this is that step.

Output:
  <out_dir>/clouds/<sample_id>.pt   # torch tensor, shape (n_points, 3), float32
  <out_dir>/labels.csv              # sample_id, label, piece_id, start_frame, kind

NOTE ON FIDELITY (read before you cite numbers): the perturbations here are
global geometric approximations of poor technique (tilt/scale/flatten), keyed to
the ranges in your milestone's perturbation table. They are NOT per-finger
joint-angle edits -- biomechanically faithful finger curl/splay would require
manipulating MANO pose parameters (e.g. via manopth) and re-running forward
kinematics. The global "wrist collapse" perturbation is, however, well aligned
with your project's stated poor-technique markers (low/dropped wrist, flat hands).
A known limitation: a baseline can partly exploit the perturbation signature
rather than learning "technique" in the abstract -- report this honestly. The
orientation jitter applied to BOTH classes is there to reduce (not eliminate)
that leakage.
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch

# Standard 21-keypoint hand joint layout (wrist + 4 per finger).
# Kept for reference / future per-finger perturbations.
JOINT_WRIST = 0
FINGER_JOINTS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# Poor-technique perturbation ranges (from the milestone perturbation table).
WRIST_TILT_DEG = (20.0, 40.0)   # wrist collapse
PALM_SCALE     = (0.80, 1.25)   # hand-size variation
FLATTEN_FACTOR = (0.45, 0.75)   # palm-normal compression (proxy for flat hands)

# Orientation jitter applied to BOTH classes (camera-jitter row of the table).
JITTER_YAW_DEG   = 15.0
JITTER_PITCH_DEG = 10.0


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def rodrigues(axis, angle_rad):
    """Rotation matrix for a rotation of angle_rad about a unit axis."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    x, y, z = axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def local_frame(pts):
    """PCA frame of a point set. Returns (centroid, e_long, e_lat, normal)."""
    c = pts.mean(axis=0)
    X = pts - c
    cov = (X.T @ X) / max(len(X), 1)
    w, V = np.linalg.eigh(cov)          # ascending eigenvalues
    normal = V[:, 0]                    # smallest variance ~ palm normal
    e_lat  = V[:, 1]
    e_long = V[:, 2]                    # largest variance ~ length of hand
    return c, e_long, e_lat, normal


def apply_poor_technique(verts_win, wrist_xyz, rng):
    """
    Apply consistent poor-technique perturbations to a (T, M, 3) window.
    Perturbation parameters are sampled once per window so the degradation is
    coherent across time.
    """
    flat = verts_win.reshape(-1, 3)
    _, e_long, e_lat, normal = local_frame(flat)

    tilt = np.deg2rad(rng.uniform(*WRIST_TILT_DEG))
    scale = rng.uniform(*PALM_SCALE)
    flatten = rng.uniform(*FLATTEN_FACTOR)

    # Rotate about the medio-lateral in-plane axis through the wrist (collapse).
    R = rodrigues(e_lat, tilt)
    out = (verts_win - wrist_xyz) @ R.T + wrist_xyz

    # Scale about the window centroid (hand-size variation).
    c = out.reshape(-1, 3).mean(axis=0)
    out = c + scale * (out - c)

    # Compress along the palm normal (flatten the hand).
    disp = out - c
    along = (disp @ normal)[..., None] * normal
    out = out - (1.0 - flatten) * along
    return out


def jitter_orientation(verts_win, rng):
    """Small random yaw+pitch applied to BOTH classes (reduces label leakage)."""
    yaw   = np.deg2rad(rng.uniform(-JITTER_YAW_DEG, JITTER_YAW_DEG))
    pitch = np.deg2rad(rng.uniform(-JITTER_PITCH_DEG, JITTER_PITCH_DEG))
    c = verts_win.reshape(-1, 3).mean(axis=0)
    R = rodrigues(np.array([0.0, 1.0, 0.0]), yaw) @ rodrigues(np.array([1.0, 0.0, 0.0]), pitch)
    return (verts_win - c) @ R.T + c


def fps_torch(pts, n, device):
    """Farthest-point sampling. pts: (N,3) tensor -> indices (n,)."""
    N = pts.shape[0]
    if N <= n:
        # Not enough points: sample with replacement to reach n.
        extra = torch.randint(0, N, (n - N,), device=device)
        return torch.cat([torch.arange(N, device=device), extra])
    sel = torch.zeros(n, dtype=torch.long, device=device)
    dist = torch.full((N,), float("inf"), device=device)
    far = torch.randint(0, N, (1,), device=device).item()
    for i in range(n):
        sel[i] = far
        d = ((pts - pts[far]) ** 2).sum(dim=1)
        dist = torch.minimum(dist, d)
        far = int(torch.argmax(dist).item())
    return sel


# --------------------------------------------------------------------------- #
# Point-cloud construction
# --------------------------------------------------------------------------- #
def build_cloud(verts_win, mode, alpha):
    """
    verts_win: (T, M, 3) windowed vertices (already perturbed/jittered).
    Returns an (P, 3) numpy array before FPS.
    """
    T, M, _ = verts_win.shape
    flat = verts_win.reshape(-1, 3)

    # Normalize to a unit sphere so spatial and temporal scales are comparable.
    c = flat.mean(axis=0)
    flat = flat - c
    extent = np.linalg.norm(flat, axis=1).max() + 1e-9
    flat = flat / extent
    verts_n = flat.reshape(T, M, 3)

    if mode == "xyz":
        return verts_n.reshape(-1, 3)

    # mode == "xyt": two spatial axes + scaled frame index, matching the DSCN.
    t = (np.arange(T) / max(T - 1, 1) - 0.5) * alpha     # centered time axis
    t = np.repeat(t, M)[:, None]
    xy = verts_n.reshape(-1, 3)[:, :2]
    return np.concatenate([xy, t], axis=1)


def load_motion(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    hands = []
    for side in ("left", "right"):
        if side not in data:
            continue
        verts = np.asarray(data[side]["mano_params"]["verts"], dtype=np.float32)
        joints = np.asarray(data[side]["joints"], dtype=np.float32)
        hands.append((verts, joints))
    return hands


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="dir containing <piece_id>/motion.pkl (e.g. for_elise/dataset)")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--window", type=int, default=16, help="frames per sample (T)")
    ap.add_argument("--stride", type=int, default=8, help="frame step between windows")
    ap.add_argument("--n_points", type=int, default=4096)
    ap.add_argument("--mode", choices=["xyt", "xyz"], default="xyt")
    ap.add_argument("--alpha", type=float, default=1.0, help="temporal-axis scale (xyt mode)")
    ap.add_argument("--verts_per_frame", type=int, default=256,
                    help="randomly subsample this many verts per hand per frame (speed)")
    ap.add_argument("--max_pieces", type=int, default=0, help="0 = all; >0 for a smoke test")
    ap.add_argument("--max_windows_per_piece", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}  mode={args.mode}  T={args.window}  n_points={args.n_points}")

    out = Path(args.out_dir)
    (out / "clouds").mkdir(parents=True, exist_ok=True)

    pieces = sorted(Path(args.data_root).glob("*/motion.pkl"))
    if args.max_pieces:
        pieces = pieces[: args.max_pieces]
    if not pieces:
        raise SystemExit(f"No motion.pkl found under {args.data_root}")

    rows = []
    n_good = n_poor = 0
    for pkl in pieces:
        piece_id = pkl.parent.name
        try:
            hands = load_motion(pkl)
        except Exception as e:
            print(f"[skip] {piece_id}: {e}")
            continue

        for hand_i, (verts, joints) in enumerate(hands):
            N = min(len(verts), len(joints))
            verts, joints = verts[:N], joints[:N]
            wins = range(0, N - args.window + 1, args.stride)
            if args.max_windows_per_piece:
                wins = list(wins)[: args.max_windows_per_piece]

            for w0 in wins:
                vw = verts[w0:w0 + args.window]          # (T, 778, 3)
                jw = joints[w0:w0 + args.window]         # (T, 21, 3)
                if not np.isfinite(vw).all() or not np.isfinite(jw).all():
                    continue

                # speed: subsample verts per frame
                if args.verts_per_frame and vw.shape[1] > args.verts_per_frame:
                    idx = rng.choice(vw.shape[1], args.verts_per_frame, replace=False)
                    vw = vw[:, idx, :]

                wrist_xyz = jw[:, JOINT_WRIST:JOINT_WRIST + 1, :]   # (T,1,3) pivot

                for kind, label in (("good", 1), ("poor", 0)):
                    work = vw.copy()
                    if kind == "poor":
                        work = apply_poor_technique(work, wrist_xyz, rng)
                    work = jitter_orientation(work, rng)            # both classes

                    cloud = build_cloud(work, args.mode, args.alpha)
                    pts = torch.from_numpy(cloud.astype(np.float32)).to(device)
                    sel = fps_torch(pts, args.n_points, device)
                    sample = pts[sel].cpu().contiguous()            # (n_points, 3)

                    sid = f"{piece_id}_h{hand_i}_f{w0:06d}_{kind}"
                    torch.save(sample, out / "clouds" / f"{sid}.pt")
                    rows.append([sid, label, piece_id, w0, kind])
                    if kind == "good":
                        n_good += 1
                    else:
                        n_poor += 1

        print(f"[done] {piece_id}: total good={n_good} poor={n_poor}")

    with open(out / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label", "piece_id", "start_frame", "kind"])
        writer.writerows(rows)

    print(f"\n[summary] pieces={len(pieces)}  samples={len(rows)}  "
          f"good={n_good}  poor={n_poor}")
    print(f"[summary] tensors in {out/'clouds'}  (each shape ({args.n_points}, 3))")
    print(f"[summary] labels in {out/'labels.csv'}")


if __name__ == "__main__":
    main()
