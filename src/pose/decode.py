#!/usr/bin/env python3
"""
StudySheriff -- shared yolov8s_pose decode (boxes + 17 COCO keypoints).

The Model-Zoo HEF is cut *before* NMS (raw conv heads over 3 scales), so the box
(DFL, 64-ch) and keypoint (51-ch) heads are decoded here on the CPU. The class
head (1-ch) is ALREADY sigmoid-activated in this HEF -- do NOT sigmoid it again
(§17.6 decode gotcha).

Imported by both src/pose/pose_live.py (live skeleton tool) and
src/dashboard/inference.py (dashboard loop) so the decode lives in one place.
"""
import numpy as np

# ---- YOLOv8-pose head constants ----
REG_MAX = 16                                  # DFL bins per box side (4*16 = 64 ch)
PROJ = np.arange(REG_MAX, dtype=np.float32)
NUM_KPTS = 17                                 # COCO-17 (17*3 = 51 ch)
INPUT = 640                                   # model input is 640x640

# COCO-17 skeleton: pairs of keypoint indices to connect
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def decode_scale(box, cls, kpt, stride):
    """One feature scale -> (boxes_xyxy, scores, kpts) in 640-input pixel space."""
    h, w, _ = box.shape
    cols, rows = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))      # (h,w)
    ax, ay = cols + 0.5, rows + 0.5                               # box anchor centers

    # boxes via DFL: (h,w,64) -> (h,w,4,16) -> softmax over bins -> expected distance
    dist = (softmax(box.reshape(h, w, 4, REG_MAX), axis=-1) * PROJ).sum(-1)  # (h,w,4)
    x1 = (ax - dist[..., 0]) * stride
    y1 = (ay - dist[..., 1]) * stride
    x2 = (ax + dist[..., 2]) * stride
    y2 = (ay + dist[..., 3]) * stride

    conf = cls[..., 0]                                           # already sigmoid-activated

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
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def postprocess(outs, conf_thr, iou_thr):
    """outs: list of (h,w,c) float32 arrays. Group by scale (by grid h), decode, NMS.

    Returns (boxes_xyxy, scores, kpts) in 640-input pixel space.
    """
    by_scale = {}                              # h -> {channels: array}
    for a in outs:
        h, _, c = a.shape
        by_scale.setdefault(h, {})[c] = a
    b_all, s_all, k_all = [], [], []
    for h, group in by_scale.items():
        stride = INPUT // h                    # 80->8, 40->16, 20->32
        b, s, k = decode_scale(group[64], group[1], group[51], stride)
        b_all.append(b); s_all.append(s); k_all.append(k)
    boxes = np.concatenate(b_all)
    scores = np.concatenate(s_all)
    kpts = np.concatenate(k_all)

    m = scores >= conf_thr
    boxes, scores, kpts = boxes[m], scores[m], kpts[m]
    if len(boxes) == 0:
        return boxes, scores, kpts
    keep = nms(boxes, scores, iou_thr)
    return boxes[keep], scores[keep], kpts[keep]
