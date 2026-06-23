#!/usr/bin/env python3
"""
StudySheriff -- fold external datasets into the classifier dataset format.

Converts Stanford-40 Actions and SCB (Student Classroom Behavior, YOLO format) into
the SAME {crop, keypoints, label, subject} npz as src/data/extract_features.py, so
they train alongside your own Pi-captured data.

Keypoints come from the FLOAT ultralytics yolov8s-pose (the HEF runs only on the Pi;
the INT8-vs-float gap is negligible next to the cross-dataset domain gap). External
samples are tagged subject='ext_*' so the trainer keeps them in TRAIN ONLY -- val/test
stay on your own held-out people (the deployment domain).

Run on Mac/Colab:
    pip install ultralytics opencv-python numpy pyyaml
    python src/data/extract_external.py --format stanford40 \
        --root Stanford40 --output data/ext_stanford40.npz
    python src/data/extract_external.py --format yolo \
        --root SCB-dataset/SCB5 --output data/ext_scb.npz

Adjust the label maps below to match the exact class names of the version you download.
"""
import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np

# our classes: 0 deskwork | 1 talking | 2 phone | 3 resting | 4 absent
# (talking & absent have no clean external match -> come from your own data)
STANFORD40_MAP = {
    "phoning": 2, "texting_message": 2,
    "reading": 0, "using_a_computer": 0, "writing_on_a_book": 0,
    "looking_through_a_microscope": 0, "writing_on_a_board": 0,
    "watching_tv": 3,
}
SCB_MAP = {
    "using_phone": 2, "playing_phone": 2, "phone": 2,
    "reading": 0, "writing": 0,
    "bowing_the_head": 3, "bowing_head": 3,
    "leaning_over_the_table": 3, "leaning_on_the_desk": 3, "leaning": 3,
    # "hand_raising" / "raising_hand" -> intentionally skipped
}


def norm(s):
    return re.sub(r"[\s\-]+", "_", str(s).strip().lower())


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def stanford40_items(root):
    """Yield (image_path, [(box_xyxy, label), ...]). Action = filename prefix."""
    img_dir = os.path.join(root, "JPEGImages")
    xml_dir = os.path.join(root, "XMLAnnotations")
    for xmlf in sorted(glob.glob(os.path.join(xml_dir, "*.xml"))):
        stem = os.path.splitext(os.path.basename(xmlf))[0]
        label = STANFORD40_MAP.get(norm(re.sub(r"_\d+$", "", stem)))
        if label is None:
            continue
        img = os.path.join(img_dir, stem + ".jpg")
        if not os.path.exists(img):
            continue
        boxes = []
        for obj in ET.parse(xmlf).getroot().findall("object"):
            bb = obj.find("bndbox")
            if bb is not None:
                boxes.append(([float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")], label))
        if boxes:
            yield img, boxes


def yolo_items(root):
    """YOLO-format (SCB): data.yaml names + per-image .txt labels."""
    import yaml
    cand = glob.glob(os.path.join(root, "**", "data.yaml"), recursive=True)
    names = {}
    if cand:
        nm = yaml.safe_load(open(cand[0])).get("names")
        names = dict(enumerate(nm)) if isinstance(nm, list) else dict(nm or {})
    print(f"[yolo] class names: {names}")
    imgs = (glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True) +
            glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
    for imgf in sorted(imgs):
        lblf = os.path.splitext(re.sub(r"[\\/]images[\\/]", "/labels/", imgf))[0] + ".txt"
        if not os.path.exists(lblf):
            continue
        im = cv2.imread(imgf)
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = []
        for line in open(lblf):
            p = line.split()
            if len(p) < 5:
                continue
            label = SCB_MAP.get(norm(names.get(int(float(p[0])), p[0])))
            if label is None:
                continue
            cx, cy, w, h = map(float, p[1:5])
            boxes.append(([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], label))
        if boxes:
            yield imgf, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True, choices=["stanford40", "yolo"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pose-model", default="yolov8s-pose.pt")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--match-iou", type=float, default=0.3)
    args = ap.parse_args()

    from ultralytics import YOLO
    pose = YOLO(args.pose_model)
    items = stanford40_items if args.format == "stanford40" else yolo_items
    tag = "ext_" + args.format

    crops, kpts_all, labels, subjects, sources = [], [], [], [], []
    n_img = 0
    for img_path, anns in items(args.root):
        img = cv2.imread(img_path)
        if img is None:
            continue
        n_img += 1
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        res = pose(img, verbose=False)[0]                 # ultralytics: BGR numpy in
        dboxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        dkpts = res.keypoints.data.cpu().numpy() if res.keypoints is not None else np.zeros((0, 17, 3))
        for box, label in anns:
            bi, best = -1, args.match_iou                 # match annotation box to a person
            for j, db in enumerate(dboxes):
                v = iou(box, db)
                if v >= best:
                    bi, best = j, v
            if bi < 0:
                continue
            kp = dkpts[bi]                                 # (17,3) in image pixels
            bw, bh = box[2] - box[0], box[3] - box[1]
            x1 = max(0, int(box[0] - args.pad * bw)); y1 = max(0, int(box[1] - args.pad * bh))
            x2 = min(W, int(box[2] + args.pad * bw)); y2 = min(H, int(box[3] + args.pad * bh))
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            crop = cv2.resize(rgb[y1:y2, x1:x2], (args.imgsz, args.imgsz))
            kx = (kp[:, 0] - x1) / max(1, x2 - x1)
            ky = (kp[:, 1] - y1) / max(1, y2 - y1)
            kn = np.stack([kx, ky, kp[:, 2]], -1).reshape(-1).astype(np.float32)
            crops.append(crop.astype(np.uint8)); kpts_all.append(kn)
            labels.append(label); subjects.append(f"{tag}_{n_img}")
            sources.append(os.path.basename(img_path))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        crops=np.array(crops, np.uint8), keypoints=np.array(kpts_all, np.float32),
        labels=np.array(labels, np.int64), subjects=np.array(subjects), sources=np.array(sources))
    labs = np.array(labels)
    print(f"\n[DONE] {len(crops)} samples from {n_img} images -> {args.output}")
    for c in sorted(set(labs.tolist())):
        print(f"   label {c}: {(labs == c).sum()}")


if __name__ == "__main__":
    main()
