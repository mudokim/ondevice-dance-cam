"""Plain-python tests (no pytest needed):  python tests/run_tests.py"""
import json
import os
import sys
import tempfile
import traceback

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from dancecam.keypoints import encode_window, flip_keypoints  # noqa: E402
from dancecam.labels import label_from_filename  # noqa: E402


def _skeleton(arms="down"):
    """Synthetic front-facing person, image coordinates (y down)."""
    k = np.zeros((17, 2), np.float32)
    k[0] = (100, 40)                                   # nose
    k[5], k[6] = (120, 80), (80, 80)                   # shoulders
    k[11], k[12] = (115, 180), (85, 180)               # hips
    k[13], k[14], k[15], k[16] = (115, 240), (85, 240), (115, 300), (85, 300)
    if arms == "up":
        k[7], k[8], k[9], k[10] = (125, 40), (75, 40), (125, 0), (75, 0)
    else:
        k[7], k[8], k[9], k[10] = (125, 130), (75, 130), (125, 175), (75, 175)
    return k


def test_encode_window():
    k = np.stack([_skeleton() + i for i in range(20)])
    x = encode_window(k)
    assert x.shape == (3, 16, 17) and x.dtype == np.float32
    assert 0 <= x.min() and x.max() <= 255


def test_encode_ignores_position_and_size():
    k = np.stack([_skeleton("up") + i for i in range(20)])
    moved = k * 2 + 300                                # twice as big, elsewhere in the frame
    assert np.allclose(encode_window(k), encode_window(moved), atol=1e-3)


def test_flip():
    k = np.stack([_skeleton()] * 4)
    f = flip_keypoints(k, 200)
    assert np.allclose(f[0, 5, 0], 200 - k[0, 6, 0])   # left shoulder <- mirrored right


def test_label_from_filename():
    assert label_from_filename("selca_minseok_01.mp4") == "selca"
    assert label_from_filename("random.mp4") is None


def test_segments_to_windows():
    """A window gets a label only when it lies fully inside one segment."""
    from windows import load_dir
    with tempfile.TemporaryDirectory() as d:
        kd, rd = os.path.join(d, "kpts"), os.path.join(d, "raw")
        os.makedirs(kd), os.makedirs(rd)
        t = np.arange(0, 2, 1 / 30, dtype=np.float32)
        k = np.stack([_skeleton("up" if ti >= 1 else "down") + ti for ti in t])
        np.savez(os.path.join(kd, "v.npz"), kpts=k, conf=np.ones((len(t), 17)), t=t, fps=30, width=200, height=320)
        with open(os.path.join(rd, "v.segments.csv"), "w") as f:
            f.write("start_s,end_s,label\n0.00,1.00,idle\n1.00,2.00,selca\n")
        X, lab, meta = load_dir(kd, window_s=0.33, stride_s=0.1, raw_dirs=(rd,))
        assert X.shape[1:] == (3, 16, 17) and set(lab) == {"idle", "selca"}
        for m, l in zip(meta, lab):
            s0, s1 = (0, 1) if l == "idle" else (1, 2)
            assert s0 <= m["start"] and m["start"] + 0.33 <= s1 + 1e-6


def test_model_size_and_shape():
    import torch
    from dancecam.models import build, n_params
    m = build("cnn2d", 6).eval()
    assert m(torch.zeros(2, 3, 16, 17)).shape == (2, 6)
    assert 50e3 < n_params(m) < 150e3


def test_shipped_onnx_matches_labels():
    import onnxruntime as ort
    labels = json.load(open(os.path.join(ROOT, "models", "cnn2d_labels.json")))
    s = ort.InferenceSession(os.path.join(ROOT, "models", "cnn2d.onnx"), providers=["CPUExecutionProvider"])
    x = encode_window(np.stack([_skeleton() + i for i in range(20)]))[None]
    assert s.run(None, {s.get_inputs()[0].name: x})[0].shape == (1, len(labels))


def test_every_label_has_an_effect():
    labels = json.load(open(os.path.join(ROOT, "models", "cnn2d_labels.json")))
    moves = json.load(open(os.path.join(ROOT, "presets", "effects.json")))["moves"]
    missing = [l for l in labels if l != "idle" and l not in moves]
    assert not missing, missing


def test_virtual_camera_keeps_frame_size():
    from dancecam.effects import VirtualCamera
    W, H = 360, 640
    cam = VirtualCamera(W, H, os.path.join(ROOT, "presets", "effects.json"))
    frame = np.zeros((H, W, 3), np.uint8)
    k = _skeleton() * 1.5 + 100
    for i in range(30):
        z, th, cx, cy, _ = cam.update(i / 30, k, "L_gajyeoga")
    out = cam.render(frame, z, th, cx, cy)
    assert out.shape == frame.shape
    assert z >= cam._min_zoom(th) - 1e-6               # rotated, but no black corners


if __name__ == "__main__":
    names = [n for n in dir() if n.startswith("test_")]
    fails = 0
    for n in names:
        try:
            globals()[n]()
            print(f"PASS {n}")
        except Exception:
            fails += 1
            print(f"FAIL {n}")
            traceback.print_exc()
    print(f"\n{len(names) - fails}/{len(names)} passed")
    sys.exit(1 if fails else 0)
