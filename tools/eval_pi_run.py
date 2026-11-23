"""Score a board run (<out>_frames.npz from pi_live.py) against true segment labels.

  python tools/eval_pi_run.py --frames runs_pi/mirror_async30_frames.npz \
      --segments data/raw_gen/mirror_L.segments.csv --cam-fps 30

Video time of a processed frame = camera sequence number / camera fps.
Frames outside every labelled segment are not scored.
- raw accuracy   : classifier argmax on that frame (last 0.33 s window)
- active accuracy: label after the 0.1 s filter, i.e. the move whose camera effect is on
- switch delay   : per segment, time from its start until the active label first equals it
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from windows import read_segments  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--cam-fps", type=float, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = np.load(a.frames)
    labels = [str(x) for x in r["labels"]]
    segs = read_segments(a.segments)
    tv = r["seq"] / a.cam_fps
    raw = np.array([labels[int(p.argmax())] if p.sum() > 0 else "" for p in r["probs"]])
    act = np.array([str(x) for x in r["active"]])
    gt = np.array([next((s[2] for s in segs if s[0] <= t < s[1]), "") for t in tv])
    m = gt != ""
    per = {}
    for lab in sorted(set(gt[m]), key=lambda x: labels.index(x) if x in labels else 99):
        k = gt == lab
        per[lab] = {"frames": int(k.sum()), "raw_acc": round(float((raw[k] == lab).mean()), 3),
                    "active_acc": round(float((act[k] == lab).mean()), 3)}
    delays = []
    for s0, s1, lab in segs[1:]:
        idx = np.where((tv >= s0) & (tv < s1) & (act == lab))[0]
        delays.append({"segment": f"{lab} {s0:.2f}-{s1:.2f}",
                       "switch_delay_s": round(float(tv[idx[0]] - s0), 2) if len(idx) else None})
    # confusion of active label (rows true, cols predicted)
    names = [l for l in labels if l in set(gt[m]) or l in set(act[m])]
    cm = {t_: {p_: int(((gt == t_) & (act == p_)).sum()) for p_ in names + [""] if ((gt == t_) & (act == p_)).sum()}
          for t_ in sorted(set(gt[m]))}
    dv = [d["switch_delay_s"] for d in delays if d["switch_delay_s"] is not None]
    res = {"frames_scored": int(m.sum()),
           "raw_acc": round(float((raw[m] == gt[m]).mean()), 3),
           "active_acc": round(float((act[m] == gt[m]).mean()), 3),
           "switch_delay_s": {"median": round(float(np.median(dv)), 2) if dv else None,
                              "max": round(float(np.max(dv)), 2) if dv else None,
                              "missed_segments": sum(d["switch_delay_s"] is None for d in delays)},
           "per_class": per, "segments": delays, "active_confusion": cm}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
