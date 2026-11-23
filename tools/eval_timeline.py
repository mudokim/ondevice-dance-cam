"""Run a trained ONNX classifier over a whole video, frame by frame, like the live app.

  python tools/eval_timeline.py --npz data/kpts/trend_L.npz --segments data/raw/trend_L.segments.csv \
      --onnx build_trend/cnn2d.onnx --labels runs_trend/cnn2d_labels.json --fps 30

Keypoints are subsampled to --fps (the board camera rate). At every frame the last
--window seconds go into the model. Reports per-frame agreement with the segment label
and, for every segment change, how long until the prediction shows the new label
(held for --stable frames).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from dancecam.keypoints import encode_window  # noqa: E402
from windows import read_segments  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--window", type=float, default=0.33)
    ap.add_argument("--stable", type=int, default=3)
    ap.add_argument("--out", default=None, help="csv of per-frame predictions")
    args = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    labels = json.load(open(args.labels, encoding="utf-8"))
    segs = read_segments(args.segments)
    d = np.load(args.npz)
    k, t = d["kpts"], d["t"]
    src_fps = float(d["fps"])
    step = max(1, int(round(src_fps / args.fps)))
    k, t = k[::step], t[::step]

    def truth(tt):
        for s0, s1, lab in segs:
            if s0 <= tt <= s1:
                return lab
        return None

    rows = []
    for i in range(len(t)):
        m = (t > t[i] - args.window) & (t <= t[i])
        if m.sum() < 4:
            continue
        x = encode_window(k[m])
        if x is None:
            continue
        p = sess.run(None, {"input": x[None]})[0][0]
        p = np.exp(p - p.max())
        p /= p.sum()
        rows.append((float(t[i]), labels[int(p.argmax())], float(p.max()), truth(float(t[i]))))

    valid = [r for r in rows if r[3] in labels]
    acc = np.mean([r[1] == r[3] for r in valid])
    print(f"frames evaluated: {len(valid)} at {args.fps:g} fps   agreement with segment label: {acc:.3f}")

    print("segment change -> prediction catches up after:")
    delays = []
    for s0, s1, lab in segs:
        if lab not in labels:
            continue
        after = [r for r in rows if s0 <= r[0] <= s1]
        hit = None
        for j in range(len(after) - args.stable + 1):
            if all(a[1] == lab for a in after[j:j + args.stable]):
                hit = after[j][0] - s0
                break
        delays.append(hit)
        print(f"  {s0:6.2f}s {lab:20s} " + (f"{hit:.2f}s" if hit is not None else "never"))
    got = [x for x in delays if x is not None]
    if got:
        print(f"median catch-up {np.median(got):.2f}s, max {max(got):.2f}s (incl. {args.stable} stable frames)")
    if args.out:
        import csv
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "pred", "prob", "truth"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
