"""Live run on Raspberry Pi 5 + DEEPX M1: camera -> pose (NPU) -> move classifier (NPU)
-> per-move camera effects (CPU) -> FPS overlay -> display/save.

  python pi_live.py --src 0 --mode async ...               # USB camera
  python pi_live.py --src dance.mp4 --cam-fps 30 ...       # video file played as a live camera

--mode sync  : one loop does  pre -> NPU (wait) -> post -> classify -> effect, frame by frame.
               The CPU sits idle while the NPU runs, and the NPU sits idle during CPU work.
--mode async : two stages overlap. Stage 1 pre-processes frame N+1 and submits it to the
               NPU (run_async) while stage 2 waits for frame N, then post-processes,
               classifies and renders it. Up to --depth frames are in flight.

A capture thread keeps only the newest camera frame: when processing is slower than the
camera, old frames are dropped (not queued), so the output never falls behind.
The FPS shown on screen is measured live: output frames in the last 1 s.
"""
import argparse
import json
import os
import queue
import threading
import time
from collections import defaultdict, deque

import cv2
import numpy as np

from dancecam.classifier import OnnxClassifier
from dancecam.effects import VirtualCamera
from dancecam.pose import DxPose


class LiveSource(threading.Thread):
    def __init__(self, src, cam_fps):
        super().__init__(daemon=True)
        self.is_file = not src.isdigit()
        self.cap = cv2.VideoCapture(int(src) if not self.is_file else src)
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.cam_fps = cam_fps or self.src_fps
        self.skip = max(1, int(round(self.src_fps / self.cam_fps))) if self.is_file else 1
        self.lock = threading.Condition()
        self.frame, self.seq, self.t_cap, self.done, self.produced = None, -1, 0.0, False, 0

    def run(self):
        t0 = time.perf_counter()
        i = 0
        while True:
            ok, f = self.cap.read()
            if not ok:
                break
            if i % self.skip:
                i += 1
                continue
            if self.is_file:                       # release frames at camera speed
                dt = t0 + self.produced / self.cam_fps - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
            with self.lock:
                self.frame, self.seq, self.t_cap = f, self.produced, time.perf_counter()
                self.lock.notify_all()
            self.produced += 1
            i += 1
        with self.lock:
            self.done = True
            self.lock.notify_all()

    def latest(self, last_seq):
        with self.lock:
            while self.seq == last_seq and not self.done:
                self.lock.wait(0.5)
            if self.seq == last_seq:
                return None
            return self.frame, self.seq, self.t_cap


class Writer(threading.Thread):
    """Saves frames off the processing loop."""

    def __init__(self, path, fps, size):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=64)
        self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.dropped = 0
        self.carry = 0                     # repeats owed by frames that did not fit

    def put(self, f, times=1):
        """times > 1 when camera frames were skipped: the saved video keeps camera time
        (the last output stays on screen, as it would on a live display)."""
        try:
            self.q.put_nowait((f, times + self.carry))
            self.carry = 0
        except queue.Full:
            self.dropped += 1
            self.carry += times

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            f, times = item
            for _ in range(times):
                self.w.write(f)
        self.w.release()


class Display(threading.Thread):
    """Shows the newest output frame (for the VNC demo) without blocking processing."""

    def __init__(self, scale=0.5):
        super().__init__(daemon=True)
        self.scale, self.frame, self.stop = scale, None, False
        self.cv = threading.Condition()

    def put(self, f):
        with self.cv:
            self.frame = f
            self.cv.notify()

    def run(self):
        cv2.namedWindow("dance cam (Pi5 + M1)", cv2.WINDOW_AUTOSIZE)
        while not self.stop:
            with self.cv:
                self.cv.wait(0.1)
                f, self.frame = self.frame, None
            if f is not None:
                cv2.imshow("dance cam (Pi5 + M1)", cv2.resize(f, None, fx=self.scale, fy=self.scale))
            if cv2.waitKey(1) == 27:
                self.stop = True
        cv2.destroyAllWindows()


def draw_hud(img, fps, mode, infer_ms, npu_ms, lat_ms, move, prob, theta, zoom):
    """Small, cheap overlay: one filled box + 4 text lines."""
    cv2.rectangle(img, (0, 0), (470, 150), (0, 0, 0), -1)
    cv2.putText(img, f"FPS {fps:4.1f}", (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, f"Pi5 + DEEPX M1 | {mode}", (14, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"pose infer {infer_ms:4.1f} ms (NPU {npu_ms:3.1f}) | lat {lat_ms:3.0f} ms", (14, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{move or '-'} {prob * 100:3.0f}% | roll {theta:+5.1f} zoom {zoom:.2f}", (14, 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)


class Back:
    """Everything after the pose keypoints: classify -> label filter -> camera move ->
    FPS overlay -> save/show, plus per-frame logging. Shared by both modes."""

    def __init__(self, args, labels, fx, W, H, mode, writer, display):
        self.clf = OnnxClassifier(args.clf, window_s=args.window, every=2, labels=labels)
        self.cam = VirtualCamera(W, H, args.effects)
        self.hold, self.minp = fx.get("label_hold_s", 0.1), fx.get("label_min_prob", 0.5)
        self.labels, self.mode, self.wr, self.disp, self.hud = labels, mode, writer, display, not args.no_hud
        self.active, self.cand, self.since, self.aprob = None, None, 0.0, 0.0
        self.t0, self.n, self.last_seq = None, 0, -1
        self.out_times = deque()
        self.stage = defaultdict(float)
        self.lat = []
        self.rec = defaultdict(list)

    def step(self, frame, seq, t_cap, k, c, npu_ms, pre_ms, post_ms, npu_wall_ms, infer_ms):
        if self.t0 is None:
            self.t0 = t_cap
        t = t_cap - self.t0
        a = time.perf_counter()
        lab, prob = self.clf.update(t, k, c)
        if prob >= self.minp and lab != self.active:
            if lab != self.cand:
                self.cand, self.since = lab, t
            if t - self.since >= self.hold:
                self.active, self.cand, self.aprob = lab, None, prob
        elif lab == self.active:
            self.cand = None
            self.aprob = prob
        b = time.perf_counter()
        z, th, cx, cy, eff = self.cam.update(t, k, self.active)
        out = self.cam.render(frame, z, th, cx, cy)
        d = time.perf_counter()
        now = d
        self.out_times.append(now)
        while self.out_times and self.out_times[0] < now - 1.0:
            self.out_times.popleft()
        fps = (len(self.out_times) - 1) / max(1e-3, self.out_times[-1] - self.out_times[0]) if len(self.out_times) > 1 else 0.0
        lat_ms = (now - t_cap) * 1000
        if self.hud:
            draw_hud(out, fps, self.mode, infer_ms, npu_ms, lat_ms, self.active, self.aprob if self.active else 0.0, th, z)
        e = time.perf_counter()
        self.wr.put(out, max(1, seq - self.last_seq) if self.last_seq >= 0 else 1)
        self.last_seq = seq
        if self.disp is not None:
            self.disp.put(out)
        f_ = time.perf_counter()
        s = self.stage
        s["pre"] += pre_ms / 1000
        s["npu_wait"] += npu_wall_ms / 1000
        s["post"] += post_ms / 1000
        s["classify"] += b - a
        s["effect_render"] += d - b
        s["hud"] += e - d
        s["queue_out"] += f_ - e
        self.lat.append(f_ - t_cap)
        r = self.rec
        probs = self.clf.probs
        r["t"].append(t); r["seq"].append(seq); r["fps"].append(fps); r["pose_ms"].append(npu_ms)
        r["kpts"].append(k if k is not None else np.full((17, 2), np.nan, np.float32))
        r["conf"].append(c if c is not None else np.zeros(17, np.float32))
        r["probs"].append(probs if probs is not None else np.zeros(len(self.labels), np.float32))
        r["active"].append(self.active or ""); r["effect"].append(eff)
        r["theta"].append(th); r["zoom"].append(z); r["cx"].append(cx); r["cy"].append(cy)
        r["signal"].append(self.cam.last_signal); r["lat_ms"].append(lat_ms); r["npu_ms"].append(npu_ms); r["infer_ms"].append(infer_ms)
        self.n += 1


def run_sync(src, pose, back):
    last = -1
    while True:
        got = src.latest(last)
        if got is None:
            break
        frame, seq, t_cap = got
        last = seq
        a = time.perf_counter()
        x, ctx = pose.prepare(frame)
        b = time.perf_counter()
        out = pose.ie.run([x])
        c = time.perf_counter()
        k, conf = pose.finish(out, ctx)
        d = time.perf_counter()
        npu, inf = pose.ie.get_npu_inference_time() / 1000, pose.ie.get_latency() / 1000
        back.step(frame, seq, t_cap, k, conf, npu, (b - a) * 1000, (d - c) * 1000, (c - b) * 1000, inf)


def run_async(src, pose, back, depth):
    """Stage 1 (thread): newest frame -> pre -> run_async.  Stage 2 (this thread):
    wait -> post -> classify -> effect -> overlay -> save. The bounded queue keeps at
    most `depth` frames in flight, so latency stays bounded."""
    inflight = queue.Queue(maxsize=depth)

    def submit():
        last = -1
        while True:
            got = src.latest(last)
            if got is None:
                break
            frame, seq, t_cap = got
            last = seq
            a = time.perf_counter()
            x, ctx = pose.prepare(frame)
            b = time.perf_counter()
            jid = pose.ie.run_async([x])
            inflight.put((jid, x, ctx, frame, seq, t_cap, (b - a) * 1000))   # blocks when full
        inflight.put(None)

    th = threading.Thread(target=submit, daemon=True)
    th.start()
    while True:
        item = inflight.get()
        if item is None:
            break
        jid, x, ctx, frame, seq, t_cap, pre_ms = item
        c0 = time.perf_counter()
        out = pose.ie.wait(jid)
        c = time.perf_counter()
        k, conf = pose.finish(out, ctx)
        d = time.perf_counter()
        npu, inf = pose.ie.get_npu_inference_time() / 1000, pose.ie.get_latency() / 1000
        back.step(frame, seq, t_cap, k, conf, npu, pre_ms, (d - c) * 1000, (c - c0) * 1000, inf)
    th.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="camera index or video file")
    ap.add_argument("--cam-fps", type=float, default=30)
    ap.add_argument("--pose", required=True, help="yolov8n-pose .dxnn")
    ap.add_argument("--clf", required=True, help="classifier .dxnn (or .onnx)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--window", type=float, default=0.33)
    ap.add_argument("--effects", default="presets/effects.json")
    ap.add_argument("--mode", choices=["sync", "async"], default="sync")
    ap.add_argument("--depth", type=int, default=2, help="async: frames in flight")
    ap.add_argument("--out", default="runs_pi/out.mp4")
    ap.add_argument("--show", action="store_true", help="show the output window (VNC demo)")
    ap.add_argument("--no-hud", action="store_true", help="no FPS overlay on the output")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    labels = json.load(open(args.labels, encoding="utf-8"))
    fx = json.load(open(args.effects, encoding="utf-8"))
    pose = DxPose(args.pose)
    src = LiveSource(args.src, args.cam_fps)
    W = int(src.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(src.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = Writer(args.out, src.cam_fps, (W, H))
    wr.start()
    disp = None
    if args.show:
        disp = Display()
        disp.start()
    mode = "Sync" if args.mode == "sync" else f"Async (depth {args.depth})"
    back = Back(args, labels, fx, W, H, mode, wr, disp)

    src.start()
    t_start = time.perf_counter()
    if args.mode == "sync":
        run_sync(src, pose, back)
    else:
        run_async(src, pose, back, args.depth)
    wall = time.perf_counter() - t_start

    wr.q.put(None)
    wr.join()
    if disp is not None:
        disp.stop = True
        disp.join(1)
    n = back.n
    fps_live = np.array(back.rec["fps"][int(src.cam_fps):] or [0])     # after the first second
    rep = {
        "mode": mode,
        "camera_fps": src.cam_fps, "camera_frames": src.produced, "processed": n, "dropped": src.produced - n,
        "processed_fps": round(n / wall, 1),
        "fps_on_screen": {"mean": round(float(fps_live.mean()), 1), "min": round(float(fps_live.min()), 1)},
        "ms_per_frame": {k_: round(v / n * 1000, 2) for k_, v in back.stage.items()},
        "pose_npu_core_ms": round(float(np.mean(back.rec["npu_ms"])), 2),
        "pose_infer_latency_ms": round(float(np.mean(back.rec["infer_ms"])), 2),
        "classifier_npu_ms_per_call": round(back.clf.t_infer / max(1, n // 2) * 1000, 2),
        "latency_ms": {"median": round(float(np.median(back.lat)) * 1000, 1),
                       "p95": round(float(np.percentile(back.lat, 95)) * 1000, 1)},
        "writer_dropped": wr.dropped,
    }
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    stem = args.out.rsplit(".", 1)[0]
    json.dump(rep, open(stem + "_report.json", "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(stem + "_frames.npz", labels=np.array(labels),
                        **{k_: np.array(v) for k_, v in back.rec.items()})


if __name__ == "__main__":
    main()
