"""Turn a short run of COCO-17 keypoints into the classifier input.

Input to the classifier: float32 array (3, 16, 17) with values in 0..255
  channel 0: x, channel 1: y (centred on the hip, divided by torso length)
  channel 2: per-frame speed of each joint
The 0..255 encoding matches the reference project; the deployed model
divides by 127.5 and subtracts 1 inside the graph.
"""
import numpy as np

T_WIN = 16           # frames per window
N_JOINTS = 17
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_AN, R_AN = 11, 12, 15, 16
# index of the mirrored joint (left <-> right) for flip augmentation
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def torso_length(k):
    """Median shoulder-centre to hip-centre distance over frames (pixels)."""
    k = np.asarray(k, np.float32).reshape(-1, N_JOINTS, 2)
    hip = (k[:, L_HIP] + k[:, R_HIP]) / 2
    sh = (k[:, L_SH] + k[:, R_SH]) / 2
    t = np.linalg.norm(sh - hip, axis=1)
    t = t[t > 1]
    return float(np.median(t)) if len(t) else 0.0


def resample(k, n=T_WIN):
    k = np.asarray(k, np.float32)
    idx = np.linspace(0, len(k) - 1, n).round().astype(int)
    return k[idx]


def encode_window(k, n=T_WIN, scale=None):
    """k: (frames>=2, 17, 2) pixel coordinates -> (3, n, 17) float32 in 0..255.

    Returns None when the torso cannot be measured (person not visible).
    """
    w = resample(k, n)
    s = scale if scale else torso_length(w)
    if s <= 1:
        return None
    hip = ((w[:, L_HIP] + w[:, R_HIP]) / 2).mean(0)
    w = (w - hip) / s
    v = np.zeros((n, N_JOINTS), np.float32)
    v[1:] = np.linalg.norm(np.diff(w, axis=0), axis=2)
    img = np.stack([
        np.clip(w[..., 0] / 3, -1, 1) * 127.5 + 127.5,
        np.clip(w[..., 1] / 3, -1, 1) * 127.5 + 127.5,
        np.clip(v, 0, 1) * 255,
    ], 0)
    return img.astype(np.float32)


def flip_keypoints(k, width):
    """Mirror horizontally and swap left/right joints. k: (..., 17, 2)."""
    k = np.asarray(k, np.float32).copy()
    k[..., 0] = width - k[..., 0]
    return k[..., FLIP_IDX, :]
