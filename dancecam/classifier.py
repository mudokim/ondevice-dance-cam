"""Per-frame keypoints -> (move label, prob).

OnnxClassifier keeps the last ``window_s`` seconds of keypoints and runs the trained
model every ``every`` frames (FP32 ONNX on the PC, INT8 .dxnn on the M1).
"""
from collections import deque

import numpy as np

from .keypoints import encode_window
from .labels import KEYS


class OnnxClassifier:
    """Trained classifier. path *.onnx -> ONNX Runtime (PC, FP32);
    path *.dxnn -> dx_engine on the M1 NPU (INT8, input uint8 HWC [16,17,3])."""
    name = "onnx"

    def __init__(self, path, window_s=0.5, every=3, labels=KEYS):
        self.dx = path.endswith(".dxnn")
        if self.dx:
            from dx_engine import InferenceEngine
            self.ie = InferenceEngine(path)
        else:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
        self.t_infer = 0.0
        self.buf = deque()
        self.window_s, self.every, self.labels = window_s, every, labels
        self.n = 0
        self.last = ("unknown", 0.0)
        self.probs = None              # full softmax of the last call
        self.ran = False               # True if the model ran on this update

    def update(self, t, kpts, conf):
        if kpts is not None:
            self.buf.append((t, kpts))
        while self.buf and self.buf[0][0] < t - self.window_s:
            self.buf.popleft()
        self.n += 1
        self.ran = False
        if self.n % self.every or len(self.buf) < 4:
            return self.last
        x = encode_window(np.stack([k for _, k in self.buf]))
        if x is None:
            return self.last
        import time
        a = time.perf_counter()
        if self.dx:
            xin = np.ascontiguousarray(np.clip(np.round(x), 0, 255).astype(np.uint8).transpose(1, 2, 0))
            logits = np.asarray(self.ie.run([xin])[0], np.float32).reshape(-1)
        else:
            logits = self.sess.run(None, {self.inp: x[None]})[0][0]
        self.t_infer += time.perf_counter() - a
        p = np.exp(logits - logits.max())
        p /= p.sum()
        self.probs, self.ran = p, True
        i = int(p.argmax())
        self.last = (self.labels[i], float(p[i]))
        return self.last
