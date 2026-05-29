"""
configs/config.py
Central configuration for the Piano Technique DSCN project.
"""

# ─── Data ────────────────────────────────────────────────────────────────────
DATA_ROOT       = "./data"
PROCESSED_DIR   = "./data/processed"       # cached .pt tensors
RAW_DIR         = "./data/raw"

# Silhouette / point cloud settings (from SCN paper)
NUM_POINTS      = 4096   # total points in stacked silhouette point cloud
T_FRAMES        = 32     # number of frames sampled per clip

# Temporal scale factors for Slow-to-Fast (subsample every k frames)
TEMPORAL_SCALES = [1, 2, 3]   # slow, faster, fastest

# z-axis rescaling so temporal spacing ≈ spatial pixel bounds
# set alpha so T_FRAMES * alpha ≈ typical frame width in pixels
ALPHA           = 5.0

# ─── FPS / KNN ───────────────────────────────────────────────────────────────
FPS_NPOINTS_L1  = 512   # centroids after first FPS
FPS_NPOINTS_L2  = 128   # centroids after second FPS
KNN_K           = 16    # neighbours for local region

# ─── Model ───────────────────────────────────────────────────────────────────
IN_CHANNELS     = 3          # (x, y, z)
HIDDEN_CHANNELS = [64, 128, 256]
NUM_CLASSES     = 2          # 0 = good technique, 1 = poor technique
DROPOUT         = 0.3
DILATIONS       = [1, 2]     # dilations used per conv layer (paper uses 1 & 2)

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
NUM_EPOCHS      = 80
LR_INIT         = 1e-3
LR_MID          = 7.5e-4     # applied at epoch EPOCH_MID
LR_END          = 5e-4       # applied at epoch EPOCH_END
EPOCH_MID       = 40
EPOCH_END       = 65
WEIGHT_DECAY    = 1e-4
MOMENTUM        = 0.8        # used if SGD; Adam ignores this
SEED            = 42

# ─── Blender Synthetic Data ──────────────────────────────────────────────────
N_GOOD_CLIPS    = 500        # synthetic good-technique clips
N_POOR_CLIPS    = 500        # synthetic poor-technique clips
FRAMES_PER_CLIP = 32         # rendered frames per clip
RENDER_RES      = (256, 256)

# perturbation ranges for poor-technique augmentation
PALM_SCALE_RANGE      = (0.8, 1.3)   # relative palm length
FINGER_CURL_RANGE_DEG = (10, 45)     # extra curl applied to bad samples
WRIST_TILT_RANGE_DEG  = (-30, 30)
THUMB_ABDUCT_RANGE    = (0.0, 0.4)   # radians
KNUCKLE_HEIGHT_RANGE  = (-0.05, 0.05)

# ─── Logging / checkpoints ───────────────────────────────────────────────────
CHECKPOINT_DIR  = "./checkpoints"
LOG_DIR         = "./logs"
SAVE_EVERY      = 5          # save checkpoint every N epochs
