"""
utils/silhouette.py

Extract hand silhouette boundary curves from video frames.
We use MediaPipe Hands to obtain hand segmentation / landmarks,
then compute the outer contour (boundary curve) of the hand mask.

For synthetic Blender data the masks are already rendered, so we
skip MediaPipe and go straight to contour extraction.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    print("[silhouette] mediapipe not found – only mask-based extraction available.")


# ─── MediaPipe-based extraction ──────────────────────────────────────────────

class HandSilhouetteExtractor:
    """
    Extracts an outer-boundary silhouette curve of hands in an RGB frame.

    Returns a (N_pts, 2) array of (x, y) contour coordinates, or None if
    no hand is detected.
    """

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.5):
        if not _MP_AVAILABLE:
            raise ImportError("mediapipe is required for real-video silhouette extraction.")
        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def _landmarks_to_mask(
        self,
        landmarks,
        h: int,
        w: int,
        expand_px: int = 12,
    ) -> np.ndarray:
        """
        Convert 21 MediaPipe landmarks to a filled binary hand mask using a
        convex hull + dilation, which is more stable than the official
        connection drawing.
        """
        pts = np.array(
            [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark],
            dtype=np.int32,
        )
        hull  = cv2.convexHull(pts)
        mask  = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px, expand_px))
        mask   = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def extract_boundary(
        self,
        frame_rgb: np.ndarray,
        n_contour_pts: int = 256,
    ) -> Optional[np.ndarray]:
        """
        Args:
            frame_rgb:    H×W×3 uint8 RGB image
            n_contour_pts: number of boundary points to resample to

        Returns:
            boundary (n_contour_pts, 2) float32 or None
        """
        h, w = frame_rgb.shape[:2]
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None

        # Merge masks from all detected hands
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for hand_lm in results.multi_hand_landmarks:
            m = self._landmarks_to_mask(hand_lm, h, w)
            combined_mask = np.maximum(combined_mask, m)

        return mask_to_boundary(combined_mask, n_contour_pts)

    def close(self):
        self.hands.close()


# ─── Mask-based boundary extraction (usable for synthetic data) ─────────────

def mask_to_boundary(
    binary_mask: np.ndarray,
    n_pts: int = 256,
) -> Optional[np.ndarray]:
    """
    Given a binary mask (uint8, 0/255), find the outer contour and resample
    it to exactly n_pts equi-arc-length points.

    Returns:
        boundary: (n_pts, 2) float32 array of (x, y) pixel coordinates
        or None if no contour found.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None

    # Take the longest contour (outer boundary)
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze(1).astype(np.float32)   # (M, 2)

    if len(contour) < 4:
        return None

    # Resample to n_pts by arc-length interpolation
    return _resample_curve(contour, n_pts)


def _resample_curve(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample an open polyline to exactly n equi-spaced points."""
    diff      = np.diff(pts, axis=0)
    seg_len   = np.linalg.norm(diff, axis=1)
    cum_len   = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = cum_len[-1]

    if total_len < 1e-6:
        return np.tile(pts[:1], (n, 1))

    target   = np.linspace(0, total_len, n)
    resampled = np.column_stack([
        np.interp(target, cum_len, pts[:, 0]),
        np.interp(target, cum_len, pts[:, 1]),
    ])
    return resampled.astype(np.float32)


# ─── Build spatio-temporal point cloud from a sequence of frames ─────────────

def frames_to_point_cloud(
    frames_rgb: List[np.ndarray],
    extractor: Optional[HandSilhouetteExtractor] = None,
    masks: Optional[List[np.ndarray]] = None,
    n_contour_pts: int = 256,
    alpha: float = 5.0,
) -> Optional[np.ndarray]:
    """
    Build the 3D stacked silhouette point cloud from a sequence of T frames.

    Each frame contributes a 2D boundary curve; the curves are stacked along
    a z-axis that encodes the frame index scaled by `alpha`.

    Args:
        frames_rgb:    list of T  H×W×3 uint8 images (use None entries to skip)
        extractor:     HandSilhouetteExtractor (if real video)
        masks:         list of T binary masks (if synthetic / pre-computed)
        n_contour_pts: boundary points per frame
        alpha:         temporal axis scale factor

    Returns:
        point_cloud: (T * n_contour_pts, 3) float32  or None
    """
    curves = []
    T = len(frames_rgb) if frames_rgb is not None else len(masks)

    for t in range(T):
        if masks is not None:
            boundary = mask_to_boundary(masks[t], n_contour_pts)
        else:
            boundary = extractor.extract_boundary(frames_rgb[t], n_contour_pts)

        if boundary is None:
            # Pad with zeros if a frame fails – FPS will still work
            boundary = np.zeros((n_contour_pts, 2), dtype=np.float32)

        # Normalise to [0, 1] by frame size
        h, w = frames_rgb[t].shape[:2] if frames_rgb is not None else masks[t].shape[:2]
        boundary = boundary / np.array([w, h], dtype=np.float32)

        z = np.full((n_contour_pts, 1), t * alpha / T, dtype=np.float32)
        curves.append(np.concatenate([boundary, z], axis=1))   # (n_pts, 3)

    return np.concatenate(curves, axis=0)   # (T * n_contour_pts, 3)
