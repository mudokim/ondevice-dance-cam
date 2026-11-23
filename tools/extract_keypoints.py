"""Videos -> keypoint files (one .npz per video) for training and calibration.

  python tools/extract_keypoints.py data/raw data/kpts
Each .npz holds: kpts (N,17,2) pixels, conf (N,17), t (N,) seconds, fps, width, height.
Frames with nobody detected are skipped (their time is simply missing).
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dancecam.pose import UltralyticsPose  # noqa: E402

EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv")


def extract(pose, path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ks, cs, ts, i = [], [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        k, c = pose(f)
        if k is not None:
            ks.append(k), cs.append(c), ts.append(i / fps)
        i += 1
    cap.release()
    return dict(kpts=np.array(ks, np.float32).reshape(-1, 17, 2), conf=np.array(cs, np.float32).reshape(-1, 17),
                t=np.array(ts, np.float32), fps=fps, width=w, height=h, n_frames=i)


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    pose = UltralyticsPose()
    files = [os.path.join(src, f) for f in sorted(os.listdir(src)) if f.lower().endswith(EXTS)] \
        if os.path.isdir(src) else [src]
    for p in files:
        d = extract(pose, p)
        out = os.path.join(dst, os.path.splitext(os.path.basename(p))[0] + ".npz")
        np.savez_compressed(out, **d)
        print(f"{os.path.basename(p)}: {len(d['kpts'])}/{d['n_frames']} frames with a person -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
