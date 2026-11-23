"""Export classifier candidates to ONNX and compile them with DX-COM.

  python tools/export_compile.py --models cnn2d tcn resnet18 --calib data/kpts --out build
  (add --weights cnn2d=runs/cnn2d.pt to export a trained model)

Per model: parameter count, ONNX check, ONNX Runtime vs PyTorch difference,
DX-COM compile result and .dxnn size. Calibration uses real keypoint windows.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from dancecam.models import Deployable, build, n_params  # noqa: E402
from windows import load_dir  # noqa: E402

SHAPE = (1, 3, 16, 17)


def export(net, path):
    m = Deployable(net).eval()
    torch.onnx.export(m, torch.zeros(SHAPE), path, input_names=["input"], output_names=["logits"],
                      opset_version=12, dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(path))
    return m


def ort_diff(m, path, X):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = X[:8] if len(X) else np.random.uniform(0, 255, (8, *SHAPE[1:])).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    got = np.concatenate([s.run(None, {"input": xi[None]})[0] for xi in x])
    return float(np.abs(ref - got).max())


class Calib(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cnn2d", "tcn", "resnet18"])
    ap.add_argument("--weights", nargs="*", default=[], help="name=path.pt")
    ap.add_argument("--calib", default="data/kpts", help="folder of .npz keypoint files")
    ap.add_argument("--calib-num", type=int, default=100)
    ap.add_argument("--window", type=float, default=0.33, help="calibration window, must match training")
    ap.add_argument("--out", default="build")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    weights = dict(w.split("=", 1) for w in args.weights)

    X, _, _ = load_dir(args.calib, window_s=args.window, require_label=False)
    rng = np.random.default_rng(0)
    if len(X) > args.calib_num:
        X = X[rng.choice(len(X), args.calib_num, replace=False)]
    print(f"calibration windows: {len(X)} ({args.window} s) from {args.calib}")

    results = {}
    for name in args.models:
        n_cls = None
        if name in weights:
            lab_path = weights[name].replace(".pt", "_labels.json")
            if os.path.exists(lab_path):
                n_cls = len(json.load(open(lab_path, encoding="utf-8")))
        net = build(name, n_cls) if n_cls else build(name)
        if name in weights:
            net.load_state_dict(torch.load(weights[name], map_location="cpu"))
        net.eval()
        onnx_path = os.path.join(args.out, f"{name}.onnx")
        m = export(net, onnx_path)
        r = {"params": n_params(net), "onnx_mb": round(os.path.getsize(onnx_path) / 1e6, 3),
             "trained": name in weights, "ort_max_abs_diff": ort_diff(m, onnx_path, X)}
        if not args.no_compile:
            import dx_com
            odir = os.path.join(args.out, f"{name}_dxnn")
            t0 = time.time()
            try:
                dx_com.compile(model=onnx_path, output_dir=odir,
                               dataloader=torch.utils.data.DataLoader(Calib(X), batch_size=1, shuffle=False),
                               calibration_num=len(X), opt_level=1, gen_log=True, export_html=True)
                dx = [f for f in os.listdir(odir) if f.endswith(".dxnn")]
                r.update(compile="ok", compile_s=round(time.time() - t0, 1), dxnn=dx,
                         dxnn_mb=round(os.path.getsize(os.path.join(odir, dx[0])) / 1e6, 3) if dx else None)
            except Exception as e:  # keep going with the other models
                r.update(compile="failed", compile_s=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}"[:500])
        results[name] = r
        print(name, json.dumps(r, ensure_ascii=False))
    json.dump(results, open(os.path.join(args.out, "export_results.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
