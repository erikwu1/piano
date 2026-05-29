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

### 4. Option B — Use FurElise / PianoVAM real datasets

```bash
# Download FurElise
# (follow instructions at https://for-elise.github.io/)
# Place clips under data/raw/videos/ with labels.csv

# Or PianoVAM (see paper for download link)
```

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
