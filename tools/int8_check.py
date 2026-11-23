"""FP32 (PC, ONNX) vs INT8 (Pi, .dxnn) on the same input windows.

  PC : python tools/int8_check.py make --onnx X.onnx --labels L.json --out int8_in.npz
  Pi : python tools/int8_check.py run  --dxnn X.dxnn --inp int8_in.npz --out int8_logits.npy
  PC : python tools/int8_check.py cmp  --inp int8_in.npz --logits int8_logits.npy
Windows: every 1/30 s over each labelled video, last --window seconds, true label = segment.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))


def make(a):
    import onnxruntime as ort
    from dancecam.keypoints import encode_window
    from windows import find_segments, read_segments
    labels = json.load(open(a.labels, encoding="utf-8"))
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    X, Y, L = [], [], []
    for f in sorted(os.listdir(a.kpts)):
        d = np.load(os.path.join(a.kpts, f))
        segs = read_segments(os.path.join(a.raw, f.replace(".npz", ".segments.csv")))
        t, k, c = d["t"], d["kpts"], d["conf"]
        for tt in np.arange(a.window, t[-1], 1 / 30):
            lab = [s[2] for s in segs if s[0] <= tt < s[1]]
            if not lab or lab[0] not in labels:
                continue
            m = (t > tt - a.window) & (t <= tt) & ~np.isnan(k).any((1, 2))
            if m.sum() < 4:
                continue
            x = encode_window(k[m])
            if x is None:
                continue
            xq = np.clip(np.round(x), 0, 255).astype(np.uint8)
            X.append(xq.transpose(1, 2, 0))
            Y.append(labels.index(lab[0]))
            L.append(sess.run(None, {name: xq.astype(np.float32)[None]})[0][0])
    np.savez(a.out, X=np.array(X), y=np.array(Y), fp32=np.array(L), labels=np.array(labels))
    print(len(X), "windows ->", a.out)


def run(a):
    from dx_engine import InferenceEngine
    d = np.load(a.inp)
    ie = InferenceEngine(a.dxnn)
    out = [np.asarray(ie.run([np.ascontiguousarray(x)])[0], np.float32).reshape(-1) for x in d["X"]]
    np.save(a.out, np.array(out))
    print(len(out), "->", a.out)


def cmp(a):
    d = np.load(a.inp)
    q = np.load(a.logits)
    pf, pq, y = d["fp32"].argmax(1), q.argmax(1), d["y"]
    print(json.dumps({"windows": int(len(y)), "agree": round(float((pf == pq).mean()), 4),
                      "fp32_acc": round(float((pf == y).mean()), 4), "int8_acc": round(float((pq == y).mean()), 4),
                      "max_abs_logit_diff": round(float(np.abs(d["fp32"] - q).max()), 3)}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "run", "cmp"])
    ap.add_argument("--onnx"), ap.add_argument("--dxnn"), ap.add_argument("--labels")
    ap.add_argument("--kpts", default="data/kpts"), ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--window", type=float, default=0.33)
    ap.add_argument("--inp"), ap.add_argument("--logits"), ap.add_argument("--out")
    a = ap.parse_args()
    {"make": make, "run": run, "cmp": cmp}[a.mode](a)
