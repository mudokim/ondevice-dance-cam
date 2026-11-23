"""Debug view: what the board sees and decides, frame by frame.

Left : camera frame + YOLO pose skeleton (from the NPU) + the virtual-camera window
Right: the output after the camera move
Text : FPS, pose NPU ms, classifier top-3 probabilities, active move, effect, angle/zoom
"""
import cv2
import numpy as np

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}


def _txt(img, s, org, scale=0.6, color=(255, 255, 255), th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def view_window(W, H, z, theta, cx, cy):
    """Corners of the output frame mapped back into the source frame."""
    M = cv2.getRotationMatrix2D((cx, cy), theta, z)
    M[0, 2] += W / 2 - cx
    M[1, 2] += H / 2 - cy
    Mi = cv2.invertAffineTransform(M)
    pts = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], np.float32)
    return (pts @ Mi.T).astype(np.int32)


def draw_debug(frame, out, info, scale=0.75):
    """info: dict with kpts, conf, top3 [(label, p)], active, effect, theta, zoom, cx, cy,
    fps, pose_ms, t, raw_label, signal."""
    H, W = frame.shape[:2]
    left = frame.copy()
    k, c = info.get("kpts"), info.get("conf")
    if k is not None:
        for a, b in SKELETON:
            if c[a] > 0.3 and c[b] > 0.3:
                col = (255, 160, 0) if a in LEFT and b in LEFT else (0, 160, 255) if a not in LEFT and b not in LEFT else (0, 255, 0)
                cv2.line(left, tuple(map(int, k[a])), tuple(map(int, k[b])), col, 4, cv2.LINE_AA)
        for i, (x, y) in enumerate(k):
            if c[i] > 0.3:
                cv2.circle(left, (int(x), int(y)), 7, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(left, (int(x), int(y)), 7, (0, 0, 0), 1, cv2.LINE_AA)
    poly = view_window(W, H, info["zoom"], info["theta"], info["cx"], info["cy"])
    cv2.polylines(left, [poly], True, (0, 255, 255), 4, cv2.LINE_AA)

    sw, sh = int(W * scale), int(H * scale)
    L = cv2.resize(left, (sw, sh), interpolation=cv2.INTER_AREA)
    R = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)
    bar_h = 190
    canvas = np.zeros((sh + bar_h, sw * 2, 3), np.uint8)
    canvas[:sh, :sw], canvas[:sh, sw:] = L, R
    cv2.line(canvas, (sw, 0), (sw, sh), (80, 80, 80), 2)
    _txt(canvas, info.get("title_left", "INPUT + YOLOv8n-Pose (M1 NPU)"), (12, 30), 0.7, (0, 255, 255), 2)
    _txt(canvas, "OUTPUT (virtual camera)", (sw + 12, 30), 0.7, (0, 255, 255), 2)

    y0 = sh + 32
    if info.get("infer_ms") is not None:
        _txt(canvas, f"t {info['t']:5.2f}s  {info['fps']:4.1f} FPS  pose infer {info['infer_ms']:4.1f} ms (NPU {info['pose_ms']:3.1f})", (12, y0), 0.62, (255, 255, 255), 2)
    else:
        _txt(canvas, f"t {info['t']:5.2f}s   {info['fps']:4.1f} FPS   pose NPU {info['pose_ms']:4.1f} ms", (12, y0), 0.7, (255, 255, 255), 2)
    act = info.get("active") or "-"
    _txt(canvas, f"MOVE: {act}", (12, y0 + 40), 0.95, (0, 255, 0), 2)
    _txt(canvas, f"effect {info['effect']}  roll {info['theta']:+5.1f} deg  zoom {info['zoom']:.2f}x",
         (12, y0 + 78), 0.65)
    if info.get("signal") is not None and info["effect"] == "roll":
        s = info["signal"]
        _txt(canvas, f"leg signal {s:+.2f} ({'right leg out' if s > 0 else 'left leg out'} on screen)", (12, y0 + 112), 0.6, (200, 200, 200))
    # classifier top-3 bars
    x0 = sw + 12
    _txt(canvas, "classifier (cnn2d INT8, last 0.33 s)", (x0, y0), 0.6, (200, 200, 200))
    for j, (lab, p) in enumerate(info.get("top3", [])):
        yy = y0 + 22 + j * 40
        cv2.rectangle(canvas, (x0, yy), (x0 + int(260 * p), yy + 26), (0, 200, 0) if j == 0 else (90, 90, 90), -1)
        _txt(canvas, f"{lab} {p * 100:4.1f}%", (x0 + 6, yy + 20), 0.6)
    return canvas
