"""Train a classifier on keypoint windows.

  python tools/train.py --kpts data/kpts --raw data/raw --model cnn2d --split time --out runs
  python tools/train.py --kpts data/kpts --model cnn2d --split video --val-match lee

--split video : whole videos go to validation (file name contains --val-match,
                otherwise every 5th video). Use this once there are many videos.
--split time  : inside every labelled segment, the last --val-tail seconds go to
                validation and only windows that end before that point are used for
                training, so no training window overlaps a validation window.
                For a single video with many segments.
Classes are the labels found in the data (saved to <model>_labels.json).
Uses CUDA when the installed PyTorch has it (the dxcom-env build is CPU-only).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from dancecam.models import build, n_params  # noqa: E402
from windows import load_dir  # noqa: E402


def make_split(meta, mode, window_s, val_tail, val_match, holdout_seg=-1):
    tr, va = [], []
    videos = sorted({m["video"] for m in meta})
    for m in meta:
        if mode == "holdout":
            tr.append(m["seg"] != holdout_seg)
            va.append(m["seg"] == holdout_seg and not m["flip"])
            continue
        if mode == "time":
            cut = m["seg_end"] - val_tail
            end = m["start"] + window_s
            tr.append(end <= cut + 1e-6)
            va.append(m["start"] >= cut - 1e-6 and not m["flip"])
        else:
            is_val = (val_match in m["video"]) if val_match else (videos.index(m["video"]) % 5 == 0)
            tr.append(not is_val)
            va.append(is_val and not m["flip"])
    return np.array(tr), np.array(va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpts", default="data/kpts")
    ap.add_argument("--raw", default="data/raw", help="folder with <stem>.segments.csv files")
    ap.add_argument("--model", default="cnn2d", choices=["cnn2d", "tcn", "resnet18"])
    ap.add_argument("--split", default="video", choices=["video", "time", "holdout"])
    ap.add_argument("--val-match", default="")
    ap.add_argument("--val-tail", type=float, default=0.6)
    ap.add_argument("--holdout-seg", type=int, default=-1, help="--split holdout: segment index used only for validation")
    ap.add_argument("--window", type=float, default=0.33)
    ap.add_argument("--stride", type=float, default=1 / 60)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed), np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, lab, meta = load_dir(args.kpts, args.window, args.stride, flip_aug=True, raw_dirs=(args.raw,))
    lab = np.array([str(l) for l in lab])
    # "idle" first, the rest alphabetical; the order is saved to <model>_labels.json
    classes = (["idle"] if "idle" in set(lab) else []) + sorted(set(lab) - {"idle"})
    y = np.array([classes.index(l) for l in lab])
    tr, va = make_split(meta, args.split, args.window, args.val_tail, args.val_match, args.holdout_seg)
    print(f"classes {classes}")
    for i, c in enumerate(classes):
        print(f"  {c:20s} train {int(((y == i) & tr).sum()):4d}  val {int(((y == i) & va).sum()):4d}")
    if tr.sum() == 0 or va.sum() == 0:
        sys.exit("need windows in both train and val")

    net = build(args.model, len(classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    counts = np.bincount(y[tr], minlength=len(classes)).astype(np.float32)
    w = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xtr, ytr = torch.tensor(X[tr] / 127.5 - 1), torch.tensor(y[tr])
    Xva, yva = torch.tensor(X[va] / 127.5 - 1).to(dev), torch.tensor(y[va]).to(dev)

    best, hist, t0 = -1, [], time.time()
    ckpt = os.path.join(args.out, f"{args.model}.pt")
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = lossf(net(Xtr[b].to(dev)), ytr[b].to(dev))
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            acc = (net(Xva).argmax(1) == yva).float().mean().item()
        hist.append(round(acc, 4))
        if acc > best:
            best = acc
            torch.save(net.state_dict(), ckpt)
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep + 1:3d}  loss {loss.item():.3f}  val acc {acc:.3f}")

    net.load_state_dict(torch.load(ckpt))
    net.eval()
    with torch.no_grad():
        pred = net(Xva).argmax(1).cpu().numpy()
    cm = np.zeros((len(classes), len(classes)), int)
    for t_, p_ in zip(y[va], pred):
        cm[t_, p_] += 1
    res = {"model": args.model, "params": n_params(net), "split": args.split, "classes": classes,
           "n_train": int(tr.sum()), "n_val": int(va.sum()), "best_val_acc": round(best, 4),
           "train_s": round(time.time() - t0, 1), "confusion_rows_true_cols_pred": cm.tolist(),
           "val_acc_per_epoch": hist}
    json.dump(res, open(os.path.join(args.out, f"{args.model}_result.json"), "w"), indent=2, ensure_ascii=False)
    json.dump(classes, open(os.path.join(args.out, f"{args.model}_labels.json"), "w"), ensure_ascii=False)
    print(f"best val acc {best:.3f} ({int(va.sum())} windows) -> {ckpt}")


if __name__ == "__main__":
    main()
