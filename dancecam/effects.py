"""Digital camera moves on a fixed phone shot: zoom, roll, pan/tilt by cropping.

The classifier decides WHICH move (label -> effect in a preset JSON);
joint motion decides WHEN and HOW MUCH (hop height, wrist height, elbow bend...).
Every move keeps the output window inside the source frame, so no black corners:
the zoom is raised automatically when the frame is rotated.

Effects (preset "effect" field):
  roll       : tilt left/right up to max_deg, following which leg is stretched out
               (zoom held at the level that hides corners at max_deg)
  bob        : snappy vertical follow of the wrists (pulling motion)
  pan_zoom   : slight zoom, shifted toward the side of the bent arm
  fist_zoom  : zoom in as the highest fist rises, out as it drops
  face_zoom  : slow zoom to frame the face and both hands (V sign)
  none       : back to the full frame
"""
import json
import math

import cv2
import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12
L_KNEE, R_KNEE, L_ANK, R_ANK = 13, 14, 15, 16


def _ema(cur, target, dt, tau):
    if tau <= 0:
        return target
    a = 1 - math.exp(-dt / tau)
    return cur + a * (target - cur)


def _elbow_angle(k, s, e, w):
    v1, v2 = k[s] - k[e], k[w] - k[e]
    c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class VirtualCamera:
    def __init__(self, width, height, preset_path):
        self.W, self.H = width, height
        self.cfg = json.load(open(preset_path, encoding="utf-8"))
        self.moves = self.cfg["moves"]
        self.default = self.cfg.get("default", {"effect": "none", "tau": 0.25})
        self.z, self.theta = 1.0, 0.0
        self.cx, self.cy = width / 2, height / 2
        self.k = None                 # smoothed keypoints
        self.hip_base = None
        self.wy_base = None
        self.t_prev = None
        self.last_signal = 0.0

    # --- geometry -------------------------------------------------------
    def _min_zoom(self, theta_deg):
        th = math.radians(abs(theta_deg))
        W, H = self.W, self.H
        return max(math.cos(th) + (H / W) * math.sin(th), math.cos(th) + (W / H) * math.sin(th), 1.0)

    def _clamp_center(self, cx, cy, z, theta_deg):
        th = math.radians(abs(theta_deg))
        ex = (self.W / 2 * math.cos(th) + self.H / 2 * math.sin(th)) / z
        ey = (self.W / 2 * math.sin(th) + self.H / 2 * math.cos(th)) / z
        cx = min(max(cx, ex), self.W - ex) if ex < self.W / 2 else self.W / 2
        cy = min(max(cy, ey), self.H - ey) if ey < self.H / 2 else self.H / 2
        return cx, cy

    # --- per-frame update ----------------------------------------------
    def update(self, t, kpts, label):
        dt = 1 / 30 if self.t_prev is None else max(1e-3, t - self.t_prev)
        self.t_prev = t
        mv = self.moves.get(label, self.default)
        eff = mv.get("effect", "none")
        tz, tth, tcx, tcy = 1.0, 0.0, self.W / 2, self.H / 2

        if kpts is not None:
            kp = np.asarray(kpts, np.float32)
            self.k = kp if self.k is None else self.k + (1 - math.exp(-dt / 0.04)) * (kp - self.k)
            k = self.k
            hip = (k[L_HIP] + k[R_HIP]) / 2
            sh = (k[L_SH] + k[R_SH]) / 2
            s = float(np.linalg.norm(sh - hip)) or 1.0
            body_c = (sh + hip) / 2
            self.hip_base = hip[1] if self.hip_base is None else _ema(self.hip_base, hip[1], dt, 0.6)
            wy = (k[L_WR, 1] + k[R_WR, 1]) / 2
            self.wy_base = wy if self.wy_base is None else _ema(self.wy_base, wy, dt, 0.5)

            if eff == "roll":
                # leg stretched to image-right -> ext > 0, to image-left -> ext < 0
                ext = ((k[L_ANK, 0] - hip[0]) + (k[R_ANK, 0] - hip[0])) / s
                u = max(-1.0, min(1.0, ext / mv.get("ext_full", 0.8)))
                tth = mv.get("sign", 1.0) * mv.get("max_deg", 15) * u
                tz = self._min_zoom(mv.get("max_deg", 15)) if mv.get("hold_zoom", True) else 1.0
                tcx, tcy = body_c
                self.last_signal = ext
            elif eff == "bob":
                tz = mv.get("zoom", 1.15)
                tcx = body_c[0]
                tcy = self.H / 2 + mv.get("gain", 0.6) * (wy - self.wy_base)
            elif eff == "pan_zoom":
                tz = mv.get("zoom", 1.18)
                aL = _elbow_angle(k, L_SH, L_EL, L_WR)
                aR = _elbow_angle(k, R_SH, R_EL, R_WR)
                bent = L_EL if aL < aR else R_EL
                side = 1.0 if k[bent, 0] > body_c[0] else -1.0
                tcx = body_c[0] + side * mv.get("shift", 0.12) * self.W
                tcy = body_c[1]
            elif eff == "fist_zoom":
                top = min(k[L_WR, 1], k[R_WR, 1])
                h = (sh[1] - top) / s                                # fist height above shoulders
                lo, hi = mv.get("low", -0.45), mv.get("high", 0.25)
                u = min(1.0, max(0.0, (h - lo) / (hi - lo)))
                tz = 1.0 + mv.get("max_zoom_add", 0.35) * u
                tcx = body_c[0]
                tcy = (1 - u) * self.H / 2 + u * (k[NOSE, 1] + sh[1]) / 2
            elif eff == "face_zoom":
                pts = k[[NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_WR, R_WR]]
                x0, y0 = pts.min(0)
                x1, y1 = pts.max(0)
                pad = mv.get("pad", 1.6)
                bw, bh = (x1 - x0) * pad + 1, (y1 - y0) * pad + 1
                tz = min(mv.get("max_zoom", 2.2), max(1.0, min(self.W / bw, self.H / bh)))
                tcx, tcy = (x0 + x1) / 2, (y0 + y1) / 2

        tau = mv.get("tau", 0.25)
        self.theta = _ema(self.theta, tth, dt, mv.get("tau_roll", tau))
        self.z = _ema(self.z, tz, dt, tau)
        self.cx = _ema(self.cx, tcx, dt, tau)
        self.cy = _ema(self.cy, tcy, dt, tau)
        z = max(self.z, self._min_zoom(self.theta))       # hide corners when rotated
        cx, cy = self._clamp_center(self.cx, self.cy, z, self.theta)
        return z, self.theta, cx, cy, eff

    def render(self, frame, z, theta, cx, cy):
        M = cv2.getRotationMatrix2D((cx, cy), theta, z)
        M[0, 2] += self.W / 2 - cx
        M[1, 2] += self.H / 2 - cy
        return cv2.warpAffine(frame, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
