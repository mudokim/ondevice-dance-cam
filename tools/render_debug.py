"""Render the debug view AFTER a live run, from what the board recorded per frame
(<out>_frames.npz from pi_live.py): the skeleton, probabilities, move and camera
values are the board's own numbers; this only draws them. Drawing during the live
run costs CPU and lowered the Pi to ~25 FPS, so it is done afterwards.

  python tools/render_debug.py --src data/trend_L.mp4 --cam-fps 30 --frames runs_pi/out_frames.npz --out runs_pi/debug.mp4
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dancecam.effects import VirtualCamera  # noqa: E402
from dancecam.overlay import draw_debug  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cam-fps", type=float, default=30)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--effects", default=os.path.join(os.path.dirname(__file__), "..", "presets", "effects.json"))
    ap.add_argument("--title", default="INPUT + YOLOv8n-Pose (M1 NPU)")
    a = ap.parse_args()
    r = np.load(a.frames, allow_pickle=False)
    labels = [str(x) for x in r["labels"]]
    cap = cv2.VideoCapture(a.src)
    skip = max(1, int(round((cap.get(cv2.CAP_PROP_FPS) or 30) / a.cam_fps)))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam = VirtualCamera(W, H, a.effects)
    by_seq = {int(s): i for i, s in enumerate(r["seq"])}
    wr, i_src, cam_seq, last = None, 0, 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i_src % skip:
            i_src += 1
            continue
        i_src += 1
        j = by_seq.get(cam_seq)
        cam_seq += 1
        if j is None:              # frame the board dropped: keep the last view (camera time)
            if wr is not None and last is not None:
                wr.write(last)
            continue
        k = r["kpts"][j]
        has = not np.isnan(k).any()
        p = r["probs"][j]
        top = [(labels[q], float(p[q])) for q in np.argsort(-p)[:3]] if p.sum() > 0 else []
        z, th, cx, cy = float(r["zoom"][j]), float(r["theta"][j]), float(r["cx"][j]), float(r["cy"][j])
        out = cam.render(f, z, th, cx, cy)
        img = draw_debug(f, out, {"kpts": k if has else None, "conf": r["conf"][j], "top3": top,
                                  "active": str(r["active"][j]), "effect": str(r["effect"][j]),
                                  "theta": th, "zoom": z, "cx": cx, "cy": cy, "fps": float(r["fps"][j]),
                                  "pose_ms": float(r["pose_ms"][j]), "t": float(r["t"][j]),
                                  "signal": float(r["signal"][j]),
                                  "infer_ms": float(r["infer_ms"][j]) if "infer_ms" in r.files else None,
                                  "title_left": a.title}, a.scale)
        if wr is None:
            wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.cam_fps, (img.shape[1], img.shape[0]))
        wr.write(img)
        last = img
    wr.release()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
