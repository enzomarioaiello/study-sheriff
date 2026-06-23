#!/usr/bin/env python3
"""
StudySheriff -- build the activity-classifier dataset from labelled photos/videos.

Runs the DEPLOYED yolov8s_pose.hef on the Pi over each labelled file and, for the
main subject in each frame, saves the two classifier inputs + label:
    crop      : 224x224x3 uint8 RGB   (person box, padded)
    keypoints : 51 float32            (17 COCO kpts (x,y,vis), normalized to the box)
    label     : int                   (class index from the filename)
    subject   : str                   (person id from the filename -> person-disjoint split)

Extracting with the deployed INT8 HEF (not a float ultralytics model) makes the
classifier train on exactly the keypoints/crops it will see live.

Filename convention (tokens split by '_'):
    classN_<label>_<subject>_<idx>.<ext>   e.g.  class2_phone_alice_001.jpg
    odd_<subject>_<idx>.<ext>              ->    label -1  (for the Unknown threshold)
Handles images (.jpg/.png/...) and videos (.mp4/.mov/...).

Run on the Pi (venv active):
    python src/data/extract_features.py --model models/yolov8s_pose.hef \
        --input-dir data/raw --output data/dataset.npz --stride 3
"""
import argparse
import glob
import os

import cv2
import numpy as np
from hailo_platform import HEF, VDevice, FormatType

REG_MAX = 16
PROJ = np.arange(REG_MAX, dtype=np.float32)
NUM_KPTS = 17
INPUT = 640
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".mov", ".avi", ".mkv", ".h264")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def decode_scale(box, cls, kpt, stride):
    h, w, _ = box.shape
    cols, rows = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
    ax, ay = cols + 0.5, rows + 0.5
    dist = (softmax(box.reshape(h, w, 4, REG_MAX), axis=-1) * PROJ).sum(-1)
    x1 = (ax - dist[..., 0]) * stride
    y1 = (ay - dist[..., 1]) * stride
    x2 = (ax + dist[..., 2]) * stride
    y2 = (ay + dist[..., 3]) * stride
    conf = cls[..., 0]                       # class head is already sigmoid-activated
    k = kpt.reshape(h, w, NUM_KPTS, 3)
    kx = (k[..., 0] * 2.0 + cols[..., None]) * stride
    ky = (k[..., 1] * 2.0 + rows[..., None]) * stride
    kv = sigmoid(k[..., 2])
    n = h * w
    boxes = np.stack([x1, y1, x2, y2], -1).reshape(n, 4)
    scores = conf.reshape(n)
    kpts = np.stack([kx, ky, kv], -1).reshape(n, NUM_KPTS, 3)
    return boxes, scores, kpts


def nms(boxes, scores, iou_thr):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def best_person(outs, conf_thr, iou_thr):
    """Return (box, score, kpts) for the most confident person, in 640-input space."""
    by = {}
    for a in outs:
        h, _, c = a.shape
        by.setdefault(h, {})[c] = a
    b_all, s_all, k_all = [], [], []
    for h, g in by.items():
        b, s, k = decode_scale(g[64], g[1], g[51], INPUT // h)
        b_all.append(b); s_all.append(s); k_all.append(k)
    boxes = np.concatenate(b_all); scores = np.concatenate(s_all); kpts = np.concatenate(k_all)
    m = scores >= conf_thr
    boxes, scores, kpts = boxes[m], scores[m], kpts[m]
    if len(boxes) == 0:
        return None
    keep = nms(boxes, scores, iou_thr)
    boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]
    i = int(scores.argmax())                 # main subject = most confident person
    return boxes[i], float(scores[i]), kpts[i]


def parse_name(fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    toks = stem.split("_")
    if toks[0] == "odd":
        return -1, (toks[1] if len(toks) > 1 else "odd")
    if toks[0].startswith("class"):
        try:
            label = int(toks[0][len("class"):])
        except ValueError:
            return None
        subject = toks[2] if len(toks) > 2 else toks[1]
        return label, subject
    return None


def frames_of(path, stride):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMG_EXT:
        img = cv2.imread(path)
        if img is not None:
            yield img
    elif ext in VID_EXT:
        cap = cv2.VideoCapture(path)
        idx = 0
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                yield fr
            idx += 1
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", default="data/dataset.npz")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--stride", type=int, default=3, help="sample every Nth video frame")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--pad", type=float, default=0.15, help="box padding fraction")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.input_dir, "*"))
                   if os.path.splitext(f)[1].lower() in IMG_EXT + VID_EXT)
    print(f"[INFO] {len(files)} files in {args.input_dir}")

    in_h, in_w, _ = HEF(args.model).get_input_vstream_infos()[0].shape
    crops, kpts_all, labels, subjects, sources = [], [], [], [], []

    with VDevice() as target:
        im = target.create_infer_model(args.model)
        im.input().set_format_type(FormatType.UINT8)
        onames = list(im.output_names)
        for n in onames:
            im.output(n).set_format_type(FormatType.FLOAT32)
        with im.configure() as cfg:
            ish = tuple(im.input().shape)
            ibuf = np.empty(ish, np.uint8)
            obuf = {n: np.empty(tuple(im.output(n).shape), np.float32) for n in onames}
            bind = cfg.create_bindings()
            bind.input(im.input().name).set_buffer(ibuf)
            for n, b in obuf.items():
                bind.output(n).set_buffer(b)

            for f in files:
                parsed = parse_name(f)
                if parsed is None:
                    print(f"[skip] bad filename: {os.path.basename(f)}")
                    continue
                label, subject = parsed
                kept = 0
                for frame_bgr in frames_of(f, args.stride):
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    oh, ow = frame_rgb.shape[:2]
                    np.copyto(ibuf, cv2.resize(frame_rgb, (in_w, in_h))[None]
                              if len(ish) == 4 else cv2.resize(frame_rgb, (in_w, in_h)))
                    cfg.run([bind], 10_000)
                    res = best_person([obuf[n] for n in onames], args.conf, args.iou)
                    if res is None:
                        continue
                    box, _, kp = res
                    sx, sy = ow / in_w, oh / in_h
                    fx1, fy1, fx2, fy2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
                    bw, bh = fx2 - fx1, fy2 - fy1
                    x1 = max(0, int(fx1 - args.pad * bw)); y1 = max(0, int(fy1 - args.pad * bh))
                    x2 = min(ow, int(fx2 + args.pad * bw)); y2 = min(oh, int(fy2 + args.pad * bh))
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue
                    crop = cv2.resize(frame_rgb[y1:y2, x1:x2], (args.imgsz, args.imgsz))
                    kx = (kp[:, 0] * sx - x1) / max(1, x2 - x1)   # normalize kpts to the box
                    ky = (kp[:, 1] * sy - y1) / max(1, y2 - y1)
                    kn = np.stack([kx, ky, kp[:, 2]], -1).reshape(-1).astype(np.float32)
                    crops.append(crop.astype(np.uint8))
                    kpts_all.append(kn)
                    labels.append(label)
                    subjects.append(subject)
                    sources.append(os.path.basename(f))
                    kept += 1
                print(f"[ok] {os.path.basename(f):42s} label={label:2d} subj={subject:10s} kept={kept}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        crops=np.array(crops, dtype=np.uint8),
        keypoints=np.array(kpts_all, dtype=np.float32),
        labels=np.array(labels, dtype=np.int64),
        subjects=np.array(subjects),
        sources=np.array(sources),
    )
    labs = np.array(labels)
    print(f"\n[DONE] {len(crops)} samples -> {args.output}")
    for c in sorted(set(labs.tolist())):
        print(f"   label {c:2d}: {(labs == c).sum()} samples")


if __name__ == "__main__":
    main()
