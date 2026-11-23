"""PC preview: one video -> pose (Ultralytics, FP32) -> ONNX classifier -> camera moves.

Use it to check a new model and effect preset on the PC before copying to the Pi.

  python pc_preview.py --video data/raw/trend_L.mp4 --classifier models/cnn2d.onnx \
      --labels models/cnn2d_labels.json --out runs/preview.mp4 --compare runs/compare.mp4

--gt-segments uses the true labels instead of the classifier (to tune effects alone).
Original audio is copied into --out when ffmpeg is available.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from dancecam.classifier import OnnxClassifier  # noqa: E402
from dancecam.effects import VirtualCamera  # noqa: E402
from dancecam.pose import UltralyticsPose  # noqa: E402


class LabelFilter:
    """Change the active move only after a new label is held for hold_s with prob >= min_prob."""

    def __init__(self, hold_s, min_prob):
        self.hold_s, self.min_prob = hold_s, min_prob
        self.active, self.cand, self.since = None, None, 0.0

    def update(self, t, label, prob):
        if prob < self.min_prob or label == self.active:
            self.cand = None
            return self.active
        if label != self.cand:
            self.cand, self.since = label, t
        if t - self.since >= self.hold_s:
            self.active, self.cand = label, None
        return self.active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--effects", default="presets/effects.json")
    ap.add_argument("--gt-segments", default=None, help="use true segment labels instead of the classifier")
    ap.add_argument("--window", type=float, default=0.33, help="classifier window (must match training)")
    ap.add_argument("--out", default="runs_fx/out.mp4")
    ap.add_argument("--compare", default=None, help="side-by-side check video")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    labels = json.load(open(args.labels, encoding="utf-8"))
    fx_cfg = json.load(open(args.effects, encoding="utf-8"))
    clf = OnnxClassifier(args.classifier, window_s=args.window, every=2, labels=labels)
    filt = LabelFilter(fx_cfg.get("label_hold_s", 0.15), fx_cfg.get("label_min_prob", 0.5))
    gt = None
    if args.gt_segments:
        from windows import read_segments
        gt = read_segments(args.gt_segments)
    pose = UltralyticsPose()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam = VirtualCamera(W, H, args.effects)
    tmp = args.out + ".noaudio.mp4"
    four = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(tmp, four, fps, (W, H))
    cw = cv2.VideoWriter(args.compare, four, fps, (W, H // 2)) if args.compare else None
    log, stage, n, t_all = [], defaultdict(float), 0, time.perf_counter()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = n / fps
        a = time.perf_counter()
        k, c = pose(f)
        b = time.perf_counter()
        if gt is not None:
            active = next((lab for s0, s1, lab in gt if s0 <= t <= s1), None)
            prob = 1.0
        else:
            lab, prob = clf.update(t, k, c)
            active = filt.update(t, lab, prob)
        d = time.perf_counter()
        z, th, cx, cy, eff = cam.update(t, k, active)
        out = cam.render(f, z, th, cx, cy)
        e = time.perf_counter()
        wr.write(out)
        if cw:
            half = (W // 2, H // 2)
            left = cv2.resize(f, half)
            right = cv2.resize(out, half)
            cv2.putText(left, f"{active or '-'}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(right, f"{eff} z{z:.2f} r{th:+.0f}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cw.write(np.hstack([left, right]))
        stage["pose"] += b - a
        stage["classify"] += d - b
        stage["effect_render"] += e - d
        log.append((round(t, 3), active, eff, round(float(z), 3), round(float(th), 2),
                    round(float(cx), 1), round(float(cy), 1)))
        n += 1
    cap.release(), wr.release()
    if cw:
        cw.release()

    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", args.video, "-map", "0:v", "-map", "1:a?",
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", args.out],
                       check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, args.out)
    json.dump({"frames": n, "fps_overall": round(n / (time.perf_counter() - t_all), 1),
               "ms_per_frame": {k_: round(v / n * 1000, 2) for k_, v in stage.items()}, "log": log},
              open(args.out.rsplit(".", 1)[0] + "_log.json", "w"))
    print(f"{n} frames -> {args.out}  ms/frame: " + ", ".join(f"{k_} {v / n * 1000:.1f}" for k_, v in stage.items()))


if __name__ == "__main__":
    main()
