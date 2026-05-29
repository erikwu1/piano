"""
blender/generate_hand_dataset.py

Run inside Blender:
    blender --background --python blender/generate_hand_dataset.py

Requires:
  - A rigged hand armature .blend file (see README for a free option)
  - The `bpy` module (built-in to Blender ≥ 3.x)

What this script does
─────────────────────
1. Loads a parametric hand rig.
2. For GOOD clips: applies realistic piano-playing keyframe poses (curved
   knuckles, relaxed wrist, thumb floating naturally).
3. For POOR clips: applies perturbations that represent bad technique:
   ▸ Flat fingers (not curved at knuckles)
   ▸ Excessive wrist tilt / collapse
   ▸ Thumb tucked under or hyper-extended
   ▸ Random stiff / splayed fingers
   ▸ Variations in palm scale, skin tone, camera angle
4. Renders a mask-only pass (white hand on black background) for each frame.
5. Saves frames to data/raw/synthetic/{good,poor}/clip_{i}/frame_{t}.png
   and a labels CSV: data/raw/synthetic/labels.csv
"""

import os
import sys
import math
import random
import json

try:
    import bpy
    import mathutils
    BLENDER = True
except ImportError:
    BLENDER = False
    print("[generate] Not running inside Blender – skipping bpy ops.")


# ── Configuration ─────────────────────────────────────────────────────────────

OUT_ROOT   = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "synthetic")
N_GOOD     = 300          # number of good-technique clips to render
N_POOR     = 300          # number of poor-technique clips to render
T_FRAMES   = 32           # frames per clip
RENDER_W   = 256
RENDER_H   = 256
FPS        = 24

# Bone names – adjust to match your rig
WRIST_BONE     = "Hand"
PALM_BONE      = "Palm"           # scales hand size
FINGER_BONES   = {
    "index":  ["Index1", "Index2", "Index3"],
    "middle": ["Middle1", "Middle2", "Middle3"],
    "ring":   ["Ring1",   "Ring2",   "Ring3"],
    "pinky":  ["Pinky1",  "Pinky2",  "Pinky3"],
    "thumb":  ["Thumb1",  "Thumb2",  "Thumb3"],
}

# Good-technique target angles (degrees) for each finger joint
# These represent a natural "C" curve position on the piano
GOOD_CURL = {
    "index":  [30, 40, 35],
    "middle": [30, 45, 35],
    "ring":   [25, 40, 30],
    "pinky":  [20, 35, 25],
    "thumb":  [ 0, 20, 15],
}

# Perturbation ranges for poor technique
PERTURBATIONS = {
    # flat fingers: reduce curl significantly
    "flat_fingers_prob":   0.5,
    "flat_curl_delta_deg": (-25, -20),   # subtract from GOOD_CURL

    # collapsed wrist
    "wrist_collapse_prob":    0.4,
    "wrist_tilt_range_deg":   (20, 40),

    # thumb problems
    "thumb_issue_prob":        0.4,
    "thumb_abduct_range_rad":  (0.2, 0.5),

    # stiff / splayed fingers
    "stiff_prob":            0.35,
    "splay_rot_deg":         (-15, 15),

    # palm scale variation
    "palm_scale_range":      (0.80, 1.25),

    # camera perturbation (rotation in degrees)
    "cam_yaw_range":   (-15, 15),
    "cam_pitch_range": (-10, 10),
}


# ── Blender helpers ───────────────────────────────────────────────────────────

def _deg(degrees: float) -> float:
    return math.radians(degrees)


def setup_scene():
    """Delete default objects, set up lighting and camera."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Lighting
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    bpy.context.object.data.energy = 3.0

    # Camera
    bpy.ops.object.camera_add(location=(0, -0.5, 0.3))
    cam = bpy.context.object
    cam.rotation_euler = (math.radians(80), 0, 0)
    bpy.context.scene.camera = cam

    # White background (for mask rendering)
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

    # Render settings
    bpy.context.scene.render.resolution_x = RENDER_W
    bpy.context.scene.render.resolution_y = RENDER_H
    bpy.context.scene.render.fps          = FPS
    bpy.context.scene.render.image_settings.file_format = 'PNG'


def load_hand_rig(blend_path: str) -> bpy.types.Object:
    """
    Load a hand armature from a .blend file.

    For a free CC0 rigged hand, download from:
    https://www.blendswap.com/blend/13631  (Hand Rig by CG Cookie)
    or use the MakeHuman exported hand.
    """
    bpy.ops.wm.append(
        filepath=os.path.join(blend_path, "Object", "HandArmature"),
        directory=os.path.join(blend_path, "Object"),
        filename="HandArmature",
    )
    return bpy.data.objects["HandArmature"]


def set_pose(armature, curl_angles: dict, wrist_euler=(0, 0, 0), palm_scale=1.0):
    """Apply a pose to the hand armature."""
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    # Palm scale
    if PALM_BONE in armature.pose.bones:
        armature.pose.bones[PALM_BONE].scale = (palm_scale, palm_scale, palm_scale)

    # Wrist
    if WRIST_BONE in armature.pose.bones:
        armature.pose.bones[WRIST_BONE].rotation_euler = wrist_euler

    # Fingers
    for finger, bones in FINGER_BONES.items():
        angles = curl_angles.get(finger, [0, 0, 0])
        for bone_name, angle_deg in zip(bones, angles):
            if bone_name in armature.pose.bones:
                bone = armature.pose.bones[bone_name]
                bone.rotation_mode = 'XYZ'
                bone.rotation_euler.x = _deg(angle_deg)

    bpy.ops.object.mode_set(mode='OBJECT')


def render_mask(output_path: str):
    """Render the current frame as a white-on-black silhouette mask."""
    # Override material to pure white emission
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            mat = bpy.data.materials.new(name="WhiteMask")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()
            emit  = nodes.new('ShaderNodeEmission')
            out   = nodes.new('ShaderNodeOutputMaterial')
            emit.inputs['Color'].default_value = (1, 1, 1, 1)
            mat.node_tree.links.new(emit.outputs['Emission'], out.inputs['Surface'])
            obj.data.materials.clear()
            obj.data.materials.append(mat)

    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def _rand(lo, hi):
    return random.uniform(lo, hi)


def make_good_pose():
    """Return curl angles and wrist pose for a good-technique frame with slight variation."""
    curl = {
        finger: [a + _rand(-5, 5) for a in angles]
        for finger, angles in GOOD_CURL.items()
    }
    wrist = (_deg(_rand(-5, 5)), _deg(_rand(-5, 5)), _deg(_rand(-5, 5)))
    scale = _rand(0.95, 1.05)
    return curl, wrist, scale


def make_poor_pose():
    """Return curl angles and wrist pose encoding poor piano technique."""
    curl  = {f: list(a) for f, a in GOOD_CURL.items()}
    wrist = [0.0, 0.0, 0.0]
    scale = _rand(*PERTURBATIONS["palm_scale_range"])

    # Flat fingers
    if random.random() < PERTURBATIONS["flat_fingers_prob"]:
        delta = _rand(*PERTURBATIONS["flat_curl_delta_deg"])
        for finger in curl:
            curl[finger] = [max(0, a + delta) for a in curl[finger]]

    # Wrist collapse
    if random.random() < PERTURBATIONS["wrist_collapse_prob"]:
        tilt = _rand(*PERTURBATIONS["wrist_tilt_range_deg"])
        wrist[0] = _deg(tilt)

    # Thumb issues
    if random.random() < PERTURBATIONS["thumb_issue_prob"]:
        abduct = _rand(*PERTURBATIONS["thumb_abduct_range_rad"])
        if "thumb" in curl:
            curl["thumb"][0] += math.degrees(abduct)

    # Stiff / splayed fingers
    if random.random() < PERTURBATIONS["stiff_prob"]:
        for finger in ["index", "middle", "ring", "pinky"]:
            splay = _rand(*PERTURBATIONS["splay_rot_deg"])
            curl[finger] = [a + splay for a in curl[finger]]

    return curl, tuple(wrist), scale


def generate_clip(armature, clip_dir: str, label: int, n_frames: int):
    """
    Generate one clip (sequence of mask renders) for a given label.
    label 0 = good, 1 = poor.
    """
    os.makedirs(clip_dir, exist_ok=True)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end   = n_frames

    for t in range(n_frames):
        bpy.context.scene.frame_set(t + 1)

        if label == 0:
            curl, wrist, scale = make_good_pose()
        else:
            curl, wrist, scale = make_poor_pose()

        # Add slight animation jitter to simulate live playing
        wrist = tuple(w + _rand(-0.02, 0.02) for w in wrist)

        set_pose(armature, curl, wrist, scale)
        out_path = os.path.join(clip_dir, f"frame_{t:04d}")
        render_mask(out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not BLENDER:
        print("This script must be run inside Blender.")
        print("Usage:  blender --background hand_rig.blend --python blender/generate_hand_dataset.py")
        return

    # Path to your hand rig blend file
    HAND_BLEND = os.environ.get("HAND_BLEND_PATH", "hand_rig.blend")
    if not os.path.exists(HAND_BLEND):
        print(f"ERROR: hand rig not found at {HAND_BLEND}")
        print("Set the HAND_BLEND_PATH env variable or edit the script.")
        sys.exit(1)

    setup_scene()
    armature = load_hand_rig(HAND_BLEND)

    labels = []  # [(clip_path, label), ...]

    for i in range(N_GOOD):
        clip_dir = os.path.join(OUT_ROOT, "good", f"clip_{i:04d}")
        generate_clip(armature, clip_dir, label=0, n_frames=T_FRAMES)
        labels.append((clip_dir, 0))
        print(f"[good] clip {i+1}/{N_GOOD} done")

    for i in range(N_POOR):
        clip_dir = os.path.join(OUT_ROOT, "poor", f"clip_{i:04d}")
        generate_clip(armature, clip_dir, label=1, n_frames=T_FRAMES)
        labels.append((clip_dir, 1))
        print(f"[poor] clip {i+1}/{N_POOR} done")

    # Write label manifest
    manifest_path = os.path.join(OUT_ROOT, "labels.json")
    with open(manifest_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Labels written to {manifest_path}")


if __name__ == "__main__":
    main()
