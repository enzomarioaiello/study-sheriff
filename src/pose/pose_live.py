#!/usr/bin/env python3
"""
StudySheriff -- live detection + pose + activity classification (the runtime pipeline).

yolov8s_pose (NPU) -> person boxes + 17 keypoints
  per person: 224 crop + 51 box-normalized keypoints  (SAME preprocessing as
              src/data/extract_features.py -- they MUST match)
              -> MobileNetV3 classifier (CPU, onnxruntime) -> activity + confidence
Draws box + skeleton + the ACTIVITY LABEL AT THE TOP OF THE BOX, and reports each
frame via on_result(...) -- consumed by src/pipeline_runner.py -> the dashboard.

Standalone (VNC desktop):
    python src/pose/pose_live.py --model models/yolov8s_pose.hef --clf models/classifier.onnx --display
Dashboard: src/dashboard/app.py calls run_pose_pipeline(model, on_result) in a thread.

Env overrides (used when the dashboard calls run_pose_pipeline without args):
    STUDY_SHERIFF_CLASSIFIER (default models/classifier.onnx)
    STUDY_SHERIFF_PAD        (default 0.15 -- MUST match extract_features --pad)
    STUDY_SHERIFF_TAU        (default 0.0  -- Unknown threshold; try 0.77)
"""
import argparse
import os
import time

import cv2
import numpy as np
from hailo_platform import HEF, VDevice, FormatType

try:                                                     # works both as a script and as a package
    from decode import postprocess, SKELETON
except ImportError:                                      # imported by the dashboard as src.pose.pose_live
    from src.pose.decode import postprocess, SKELETON

CLASS_NAMES = ["deskwork", "talking", "phone", "resting", "absent"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# per-activity colour (BGR) -- matches frontend/js/config.js COLORS
ACT_COLORS = {
    "deskwork": (117, 158, 29), "talking": (221, 138, 55), "phone": (48, 90, 216),
    "resting": (23, 117, 186), "absent": (128, 135, 136),
    "unknown": (126, 83, 212), "person": (180, 180, 180),
}


# ---------------------------- classifier ----------------------------
def _load_classifier(clf_path):
    if clf_path is None:
        clf_path = os.environ.get("STUDY_SHERIFF_CLASSIFIER", os.path.join("models", "classifier.onnx"))
    if not os.path.exists(clf_path):
        print(f"[pose] no classifier at {clf_path} -> pose-only (label='person')")
        return None
    import onnxruntime as ort
    print(f"[pose] classifier: {clf_path}")
    return ort.InferenceSession(clf_path, providers=["CPUExecutionProvider"])


def _crop_and_kpts(frame_rgb, box, kp, sx, sy, pad, imgsz=224):
    """Padded 224 RGB crop + box-normalized 51 keypoints -- identical to extract_features.py."""
    oh, ow = frame_rgb.shape[:2]
    fx1, fy1, fx2, fy2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    bw, bh = fx2 - fx1, fy2 - fy1
    x1 = max(0, int(fx1 - pad * bw)); y1 = max(0, int(fy1 - pad * bh))
    x2 = min(ow, int(fx2 + pad * bw)); y2 = min(oh, int(fy2 + pad * bh))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    crop = cv2.resize(frame_rgb[y1:y2, x1:x2], (imgsz, imgsz)).astype(np.float32) / 255.0
    crop = ((crop - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    kx = (kp[:, 0] * sx - x1) / max(1, x2 - x1)
    ky = (kp[:, 1] * sy - y1) / max(1, y2 - y1)
    kin = np.stack([kx, ky, kp[:, 2]], -1).reshape(-1)[None].astype(np.float32)
    return crop, kin


def _classify(sess, frame_rgb, box, kp, sx, sy, pad, tau):
    out = _crop_and_kpts(frame_rgb, box, kp, sx, sy, pad)
    if out is None:
        return "unknown", 0.0
    logits = sess.run(None, {"crop": out[0], "keypoints": out[1]})[0][0]
    e = np.exp(logits - logits.max()); p = e / e.sum()
    i = int(p.argmax()); c = float(p[i])
    return ("unknown" if c < tau else CLASS_NAMES[i]), c


# ---------------------------- drawing ----------------------------
def _annotate(disp, boxes, kpts, acts, sx, sy, kpt_thr=0.5):
    for box, kp, (name, conf) in zip(boxes, kpts, acts):
        p1 = (int(box[0] * sx), int(box[1] * sy)); p2 = (int(box[2] * sx), int(box[3] * sy))
        color = ACT_COLORS.get(name, ACT_COLORS["unknown"])
        cv2.rectangle(disp, p1, p2, color, 2)
        pts = [(int(x * sx), int(y * sy), v) for x, y, v in kp]
        for a, b in SKELETON:
            if pts[a][2] > kpt_thr and pts[b][2] > kpt_thr:
                cv2.line(disp, pts[a][:2], pts[b][:2], (255, 128, 0), 2)
        for x, y, v in pts:
            if v > kpt_thr:
                cv2.circle(disp, (x, y), 3, (0, 0, 255), -1)
        # ACTIVITY LABEL AT THE TOP OF THE BOX (filled bar so it's readable on video)
        text = ("Unknown" if name == "unknown" else name) + f" {conf * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(th + 9, p1[1])
        cv2.rectangle(disp, (p1[0], top - th - 9), (p1[0] + tw + 8, top), color, -1)
        cv2.putText(disp, text, (p1[0] + 4, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return disp


def open_camera(w, h):
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"size": (w, h), "format": "RGB888"}))
    cam.start(); time.sleep(1.0)
    return cam


# ---------------------------- the pipeline ----------------------------
def run_pose_pipeline(model, on_result, clf_path=None, pad=None, conf=0.4, iou=0.5,
                      tau=None, width=1280, height=720):
    """Run the live pipeline forever, calling on_result(result_dict) per frame.

    result_dict keys: annotated_frame (BGR), person_count, persons [{activity,conf}],
    per_person_classes, current_class, fps, latency_ms.
    """
    sess = _load_classifier(clf_path)
    if pad is None:
        pad = float(os.environ.get("STUDY_SHERIFF_PAD", "0.15"))
    if tau is None:
        tau = float(os.environ.get("STUDY_SHERIFF_TAU", "0.0"))
    in_h, in_w, _ = HEF(model).get_input_vstream_infos()[0].shape
    print(f"[pose] model {in_w}x{in_h} | pad={pad} | tau={tau}")
    cam = open_camera(width, height)

    with VDevice() as target:
        infer = target.create_infer_model(model)
        infer.input().set_format_type(FormatType.UINT8)
        out_names = list(infer.output_names)
        for n in out_names:
            infer.output(n).set_format_type(FormatType.FLOAT32)
        with infer.configure() as cfg:
            in_name, in_shape = infer.input().name, tuple(infer.input().shape)
            in_buf = np.empty(in_shape, np.uint8)
            out_bufs = {n: np.empty(tuple(infer.output(n).shape), np.float32) for n in out_names}
            bindings = cfg.create_bindings()
            bindings.input(in_name).set_buffer(in_buf)
            for n, b in out_bufs.items():
                bindings.output(n).set_buffer(b)

            frames, t0, fps = 0, time.time(), 0.0
            try:
                while True:
                    t_frame = time.time()
                    frame = cam.capture_array()                  # picamera2 'RGB888' -> BGR array
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    oh, ow = frame_rgb.shape[:2]
                    np.copyto(in_buf, (cv2.resize(frame_rgb, (in_w, in_h))[None]
                                       if len(in_shape) == 4 else cv2.resize(frame_rgb, (in_w, in_h))))
                    cfg.run([bindings], 10_000)
                    boxes, scores, kpts = postprocess([out_bufs[n] for n in out_names], conf, iou)

                    sx, sy = ow / in_w, oh / in_h
                    if sess is not None:
                        acts = [_classify(sess, frame_rgb, box, kp, sx, sy, pad, tau)
                                for box, kp in zip(boxes, kpts)]
                    else:
                        acts = [("person", 1.0) for _ in boxes]

                    disp = frame.copy()                          # BGR -> ok for cv2 / MJPEG
                    _annotate(disp, boxes, kpts, acts, sx, sy)

                    frames += 1
                    if frames % 5 == 0:
                        fps = frames / (time.time() - t0)
                    names = [a for a, _ in acts]
                    on_result({
                        "annotated_frame": disp,
                        "person_count": len(boxes),
                        "persons": [{"activity": a, "conf": c} for a, c in acts],
                        "per_person_classes": names,
                        "current_class": max(set(names), key=names.count) if names else "no_person",
                        "fps": round(fps, 1),
                        "latency_ms": round((time.time() - t_frame) * 1000.0, 1),
                    })
            finally:
                cam.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--clf", default=None, help="classifier.onnx (default models/classifier.onnx)")
    ap.add_argument("--pad", type=float, default=None, help="MUST match extract_features --pad")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=None, help="Unknown threshold (0=off; try 0.77)")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    st = {"n": 0}

    def on_result(r):
        if args.display:
            cv2.imshow("study-sheriff", r["annotated_frame"])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
        elif args.save_dir:
            st["n"] += 1
            if st["n"] % 15 == 0:
                cv2.imwrite(os.path.join(args.save_dir, f"frame_{st['n']:05d}.jpg"), r["annotated_frame"])
        else:
            print(f"\r{r['fps']:5.1f} FPS  {r['person_count']} ppl  {r['current_class']}        ", end="")

    try:
        run_pose_pipeline(args.model, on_result, clf_path=args.clf, pad=args.pad,
                          conf=args.conf, iou=args.iou, tau=args.tau)
    except KeyboardInterrupt:
        pass
    finally:
        if args.display:
            cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()
