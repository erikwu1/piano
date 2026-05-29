# Piano Technique Detection via Dilated Silhouette Convolutional Network

**CS231N Project — James Wei, Arthur Xu, Erik Wu**

Detects poor piano technique from video using a Dilated Silhouette Convolutional Network
(DSCN) based on [Hua et al., 2021](https://zichunzhong.github.io/papers/SCN_CAGD2021.pdf).

---

## Project Structure

```
piano_scn/
├── configs/config.py          # All hyperparameters
├── models/dscn.py             # DSCN + baselines (ResNet18, PointNet++)
├── utils/
│   ├── point_ops.py           # FPS, KNN, KDE
│   └── silhouette.py          # Boundary extraction, point cloud builder
├── data/dataset.py            # Dataset classes + DataLoader factory
├── blender/
│   └── generate_hand_dataset.py  # Synthetic data generator (run inside Blender)
├── scripts/
│   ├── preprocess.py          # Convert raw clips → cached .pt tensors
│   ├── train.py               # Main training script
│   └── evaluate.py            # Evaluation + confusion matrix
└── requirements.txt
```

---

## GCP Setup (SSH + tmux workflow)

### 1. Create and connect to your GCP VM

```bash
# From your local machine
gcloud compute ssh --zone us-central1-a <your-vm-name> -- -L 6006:localhost:6006
```

> The `-L 6006:localhost:6006` port-forwards TensorBoard to your local browser.

### 2. Install tmux and start a session

Always use tmux so training survives SSH disconnections.

```bash
# On the GCP VM
sudo apt-get install -y tmux

# Start a named session
tmux new-session -s piano

# Inside tmux, Ctrl+B then D to detach (training keeps running)
# To re-attach later:
tmux attach -t piano

# Useful tmux shortcuts:
#   Ctrl+B  D    detach (safe disconnect)
#   Ctrl+B  [    scroll mode (q to exit)
#   Ctrl+B  %    split pane vertically
#   Ctrl+B  "    split pane horizontally
```

### 3. Clone the repo and install dependencies

```bash
# On the GCP VM, inside tmux
cd ~
git clone <your-repo-url> piano_scn
cd piano_scn

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU is available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4. Option A — Synthetic Blender data (recommended for quick start)

Install Blender on the VM (headless):
```bash
sudo apt-get install -y blender

# Download a free CC0 rigged hand .blend file (CG Cookie or similar)
# Set the path:
export HAND_BLEND_PATH=/path/to/hand_rig.blend

# Generate synthetic dataset (this runs Blender headlessly)
blender --background $HAND_BLEND_PATH \
        --python blender/generate_hand_dataset.py \
        2>&1 | tee logs/blender_gen.log
```

> **Note on the hand rig:** Download a free rigged hand from:
> - https://www.blendswap.com/blend/13631  (Hand Rig, CC-BY)
> - Or export from MakeHuman with the hand plugin.
> Bone names in the rig must match `FINGER_BONES` in `blender/generate_hand_dataset.py`.
> Edit those names to match your rig if needed.

### 4. Option B — Use FürElise / PianoVAM real datasets

#### FürElise (Stanford SIGGRAPH Asia 2024)
> **License:** CC BY-NC 4.0 — non-commercial research use only.
> 10 hours of 3D hand motion from 15 elite pianists, 153 pieces, 59.94 fps.
> Hosted on Hugging Face: https://huggingface.co/datasets/rcwang/for_elise

```bash
# 1. Install git-lfs (required for large file download)
sudo apt-get install -y git-lfs
git lfs install

# 2. Clone the dataset repo (skip the raw zip — it's too large to clone directly)
#    GIT_LFS_SKIP_SMUDGE=1 avoids downloading all LFS blobs upfront
GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:datasets/rcwang/for_elise
cd for_elise

# 3. Download the actual data (~44 GB) via the provided shell script
sh ./download_data.sh
cd ..
```

> **Note:** You need a Hugging Face account and SSH key configured.
> If you prefer HTTPS over SSH, use:
> `GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/rcwang/for_elise`
> and authenticate with your HF token when prompted.

The dataset structure per piece looks like:
```
for_elise/dataset/<piece_id>/
├── motion.pkl        # dict with left/right → joints (Nx21x3), mano_params, verts (Nx778x3)
├── midi.mid          # synchronized MIDI
├── audio.mp3         # synthesized audio
└── vis/              # 3D visualizer assets
```

To use the 3D hand vertices as input to our pipeline instead of raw video silhouettes,
add a data adapter in `data/dataset.py` that reads `motion.pkl` and projects
`verts` (Nx778x3) through a virtual camera to produce per-frame 2D masks.
This is the cleanest path since the ground-truth meshes give perfect silhouettes:

```python
import pickle, numpy as np

with open("for_elise/dataset/0/motion.pkl", "rb") as f:
    data = pickle.load(f)

left_joints  = data["left"]["joints"]   # (N, 21, 3) — 3D joint positions
right_joints = data["right"]["joints"]  # (N, 21, 3)
left_verts   = data["left"]["mano_params"]["verts"]   # (N, 778, 3) — full mesh
```

---

#### PianoVAM (ISMIR 2025)
> **License:** CC BY-NC 4.0 — non-commercial research use only.
> 21 hours, 106 recordings from 10 amateur pianists.
> Includes synchronized top-view video, audio, MIDI, hand landmarks, fingering labels.
> Paper: https://arxiv.org/abs/2509.08800
> Code & dataset: https://github.com/yonghyunk1m/PianoVAM-Code

```bash
# 1. Install git-lfs
sudo apt-get install -y git-lfs
git lfs install

# 2. Clone the code repo (contains download scripts and preprocessing tools)
git clone https://github.com/yonghyunk1m/PianoVAM-Code.git
cd PianoVAM-Code

# 3. Install dependencies for their preprocessing scripts
pip install -r requirements.txt   # if present, otherwise: pip install numpy opencv-python

# 4. Download the dataset from Hugging Face
#    The dataset is gated — you must first accept the license at:
#    https://huggingface.co/datasets/yonghyunk1m/PianoVAM  (exact HF slug from their repo)
#    Then log in:
pip install huggingface_hub
huggingface-cli login   # paste your HF token

# 5. Use their provided download script or pull directly:
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='yonghyunk1m/PianoVAM', repo_type='dataset', local_dir='../data/raw/pianovam')
"
cd ..
```

PianoVAM provides top-view videos (1080p/60fps) of hands on the keyboard, which
feed directly into `utils/silhouette.py` via MediaPipe. Place the clips and run:

```bash
# Build a labels.csv pointing to PianoVAM clips
# (you will need to define good/poor labels — PianoVAM provides fingering labels
#  which can be used as a proxy: correct fingering = good, incorrect = poor)
python scripts/preprocess.py --dataset video --data_root data/raw/pianovam
```

---

#### After downloading either dataset, continue with step 5 (preprocessing).

### 5. Preprocess: convert clips → cached tensors

```bash
# Run inside tmux
python scripts/preprocess.py \
    --dataset synthetic \
    --data_root data/raw/synthetic \
    --out_dir data/processed \
    --n_points 4096

# This caches everything as .pt files so training is fast.
# On a 300+300 clip dataset it takes ~15–30 minutes on CPU.
```

### 6. Train

```bash
# Open a tmux window or pane: Ctrl+B %
# Start TensorBoard in one pane:
tensorboard --logdir logs/tb --port 6006 &

# Train DSCN (main model) in the other pane:
python scripts/train.py \
    --dataset cached \
    --cache_dir data/processed \
    --model dscn \
    --epochs 80 \
    --batch_size 32 \
    --tag v1 \
    2>&1 | tee logs/train_dscn.log

# Train baselines for comparison:
python scripts/train.py --model pointnet --dataset cached --tag baseline 2>&1 | tee logs/train_pnet.log
python scripts/train.py --model resnet   --dataset cached --tag baseline 2>&1 | tee logs/train_res.log
```

**Monitor TensorBoard** from your local browser at http://localhost:6006  
(the SSH tunnel you set up in step 1 handles the forwarding).

**Monitor training logs:**
```bash
tail -f logs/train_dscn.log
```

**Safe shutdown:** always detach tmux before closing your SSH session:
```bash
Ctrl+B  D    # detach — training keeps running on the VM
```

### 7. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_dscn.pt \
    --model dscn \
    --cache_dir data/processed
```

---

## Model Architecture Summary

```
Input: stacked silhouette point cloud  (B, 4096, 3)
         z-axis = frame index × alpha (temporal encoding)

SlowToFastDSCN
  ├── SCNEncoder (scale=1, slow)
  │     ├── FPS → 512 centroids
  │     ├── DilatedSilhouetteConv (dilation=1) ┐
  │     ├── DilatedSilhouetteConv (dilation=2) ┘ fused → 128 channels
  │     ├── FPS → 128 centroids
  │     ├── DilatedSilhouetteConv (dilation=1) ┐
  │     ├── DilatedSilhouetteConv (dilation=2) ┘ fused → 256 channels
  │     └── Global MaxPool → feature (256,)
  ├── SCNEncoder (scale=2, faster)  → feature (256,)
  └── SCNEncoder (scale=3, fastest) → feature (256,)

Concat → (768,) → MLP(512→256→2) → logits

Loss: CrossEntropyLoss (binary classification)
Metric: Macro F1 (harmonic mean of precision & recall per class)
```

---

## Baselines

| Model | Description | Expected advantage of DSCN |
|---|---|---|
| ResNet-18 (frame avg) | 2D CNN on binary silhouette frames, average predictions | DSCN should win if temporal movement matters |
| PointNet++ (no dilation) | Standard KNN without dilation | DSCN should win if long-range temporal context helps |
| **DSCN (ours)** | Multi-scale dilated silhouette convolution | Joint spatio-temporal modelling |

---

## Synthetic Data Perturbations (poor technique)

The Blender generator adds the following augmentations to simulate bad technique:

| Perturbation | Parameter range | Represents |
|---|---|---|
| Flat fingers | curl reduced 20–25° | Not curving knuckles |
| Wrist collapse | tilt 20–40° | Dropped / tilted wrist |
| Thumb abduction | 0.2–0.5 rad | Thumb tucked under or splayed |
| Finger splay | ±15° | Stiff, uncontrolled spread |
| Palm scale | 0.80–1.25× | Hand size variation |
| Camera jitter | ±15° yaw, ±10° pitch | Viewpoint robustness |

---

## Tips for GCP Cost Management

- Use **preemptible / spot VMs** for training — attach a persistent disk for data.
- Cache all point clouds as `.pt` files **before** training so you can restart cheaply.
- Use `tmux` + `tee` so logs survive disconnections and you can audit runs later.
- Use `gcloud compute instances stop <vm>` when not training to avoid idle GPU charges.
