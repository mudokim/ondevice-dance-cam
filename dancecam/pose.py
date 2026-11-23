"""Pose backends. Both return (kpts (17,2) pixels, conf (17,)) for the
largest person, or (None, None) when nobody is found.

- UltralyticsPose : PC (PyTorch, FP32). Used for development and data prep.
- DxPose          : Pi 5 + M1 (.dxnn via dx_engine), reusing dx_app's
                    letterbox pre-processing and YOLOv8 pose post-processing.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "..", "weights", "yolov8n-pose.pt")


class UltralyticsPose:
    def __init__(self, weights=DEFAULT_WEIGHTS, imgsz=640, device=None):
        from ultralytics import YOLO
        weights = os.path.abspath(weights)
        if not os.path.exists(weights):
            os.makedirs(os.path.dirname(weights), exist_ok=True)
            cwd = os.getcwd()
            os.chdir(os.path.dirname(weights))   # ultralytics downloads into cwd
            try:
                YOLO(os.path.basename(weights))
            finally:
                os.chdir(cwd)
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.device = device

    def __call__(self, frame):
        r = self.model(frame, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return None, None
        i = int(r.boxes.xywh[:, 2:].prod(1).argmax())
        xy = r.keypoints.xy[i].cpu().numpy().astype(np.float32)
        conf = r.keypoints.conf[i].cpu().numpy().astype(np.float32) if r.keypoints.conf is not None else np.ones(17, np.float32)
        return xy, conf


DX_APP_PY = os.path.expanduser("~/dx-all-suite/dx-runtime/dx_app/src/python_example")


class DxPose:
    """Pi 5 + M1: YOLOv8n-Pose .dxnn on the NPU, with dx_app's own letterbox
    preprocessing and YOLOv8 pose postprocessing (NMS + coordinate restore)."""

    def __init__(self, dxnn_path, dx_app_py=DX_APP_PY):
        import sys
        if dx_app_py not in sys.path:
            sys.path.insert(0, dx_app_py)
        from dx_engine import InferenceEngine
        from common.processors import LetterboxPreprocessor, YOLOv8PosePostprocessor
        self.ie = InferenceEngine(dxnn_path)
        shape = self.ie.get_input_tensors_info()[0]["shape"]       # [1, 640, 640, 3]
        h, w = int(shape[1]), int(shape[2])
        self.pre = LetterboxPreprocessor(w, h)
        self.post = YOLOv8PosePostprocessor(w, h, {})
        self.t_pre = self.t_npu = self.t_post = 0.0

    # split steps, used by the Async pipeline (NPU works on one frame while the
    # CPU post-processes the previous one)
    def prepare(self, frame):
        return self.pre.process(frame)

    def finish(self, out, ctx):
        res = self.post.process(out, ctx)
        if not res:
            return None, None
        best = max(res, key=lambda r: (r.box[2] - r.box[0]) * (r.box[3] - r.box[1]))
        xy = np.array([[p.x, p.y] for p in best.keypoints], np.float32)
        conf = np.array([p.confidence for p in best.keypoints], np.float32)
        return xy, conf

    def __call__(self, frame):
        import time
        a = time.perf_counter()
        x, ctx = self.prepare(frame)
        b = time.perf_counter()
        out = self.ie.run([x])
        c = time.perf_counter()
        k, conf = self.finish(out, ctx)
        d = time.perf_counter()
        self.t_pre += b - a
        self.t_npu += c - b
        self.t_post += d - c
        return k, conf
