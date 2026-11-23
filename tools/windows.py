"""Keypoint files -> windows for training / calibration.

Two ways to label a video:
  1. file name starts with a label key (one action per video), e.g. arms_up_minseok_01.npz
  2. a segment file next to the raw video: <stem>.segments.csv with start_s,end_s,label
     (several actions in one video). A window is labelled only when it lies fully
     inside one segment.
"""
import csv
import glob
import os

import numpy as np

from dancecam.keypoints import encode_window, flip_keypoints
from dancecam.labels import label_from_filename


def read_segments(path):
    with open(path, encoding="utf-8") as f:
        return [(float(r["start_s"]), float(r["end_s"]), r["label"]) for r in csv.DictReader(f)]


def find_segments(npz_path, raw_dirs):
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    for d in raw_dirs:
        p = os.path.join(d, stem + ".segments.csv")
        if os.path.exists(p):
            return read_segments(p)
    return None


def iter_windows(d, window_s=0.5, stride_s=0.1, flip=False):
    """Yield (start_s, x) for every window with >= 4 detected frames."""
    k, t = d["kpts"], d["t"]
    if len(t) == 0:
        return
    start = float(t[0])
    while start + window_s <= t[-1] + 1e-6:
        m = (t >= start) & (t < start + window_s)
        if m.sum() >= 4:
            kk = flip_keypoints(k[m], float(d["width"])) if flip else k[m]
            x = encode_window(kk)
            if x is not None:
                yield start, x
        start += stride_s


def load_dir(kpt_dir, window_s=0.5, stride_s=0.1, flip_aug=False, require_label=True,
             raw_dirs=("data/raw",), exclude=("intro",)):
    """Returns X (N,3,16,17), labels (N,) str, meta list of dicts (video, seg, start, seg_start, seg_end)."""
    X, labels, meta = [], [], []
    for p in sorted(glob.glob(os.path.join(kpt_dir, "*.npz"))):
        d = np.load(p)
        name = os.path.basename(p)
        segs = find_segments(p, raw_dirs)
        file_label = label_from_filename(name)
        if segs is None and file_label is None and require_label:
            print(f"skip (no label): {name}")
            continue
        for flip in ([False, True] if flip_aug else [False]):
            for start, x in iter_windows(d, window_s, stride_s, flip):
                end = start + window_s
                if segs is not None:
                    hit = [(i, s) for i, s in enumerate(segs) if s[0] <= start and end <= s[1] + 1e-6]
                    if not hit:
                        continue                      # window crosses a boundary
                    si, (s0, s1, lab) = hit[0]
                else:
                    si, s0, s1, lab = 0, 0.0, float(d["t"][-1]), file_label
                if lab in exclude:
                    continue
                X.append(x), labels.append(lab)
                meta.append(dict(video=name, seg=si, start=start, seg_start=s0, seg_end=s1, flip=flip))
    if not X:
        return np.zeros((0, 3, 16, 17), np.float32), np.array([]), []
    return np.stack(X), np.array(labels), meta
