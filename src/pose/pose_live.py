#!/usr/bin/env python3
"""
StudySheriff -- live person detection + pose (yolov8s_pose) on the Hailo-10H.

One NPU pass of YOLOv8s-pose gives, per person: a bounding box AND 17 COCO
keypoints. The Model-Zoo HEF is cut *before* NMS (raw conv heads over 3 scales),
so boxes (DFL) and keypoints are decoded here on the CPU.

Hardware : Raspberry Pi 5 + Hailo-10H (HailoRT 5.3.0) + Camera Module 3
Runtime  : HailoRT 5.x async InferModel API (pyhailort)
Display  : run from a VNC desktop terminal (cv2.imshow needs a window); or use
           --save-dir for headless frame dumps you scp to the Mac.

Usage:
    python src/pose/pose_live.py --model models/yolov8s_pose.hef --display
    python src/pose/pose_live.py --model models/yolov8s_pose.hef --save-dir out
"""
import argparse
import os
import time

import cv2
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

    conf = cls[..., 0]                                           # class head is already
    #                                                            # sigmoid-activated in this HEF

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
    """outs: list of (h,w,c) float32 arrays. Group by scale (by grid h), decode, NMS."""
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


def draw(frame, boxes, scores, kpts, sx, sy, kpt_thr=0.5):
    for (x1, y1, x2, y2), sc, kp in zip(boxes, scores, kpts):
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(frame, f"person {sc:.2f}", (p1[0], p1[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        pts = [(int(x * sx), int(y * sy), v) for x, y, v in kp]
        for a, b in SKELETON:
            if pts[a][2] > kpt_thr and pts[b][2] > kpt_thr:
                cv2.line(frame, pts[a][:2], pts[b][:2], (255, 128, 0), 2)
        for x, y, v in pts:
            if v > kpt_thr:
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
    return frame


def open_camera(w, h):
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (w, h), "format": "RGB888"}))
    cam.start()
    time.sleep(1.0)                            # let auto-exposure settle
    return cam


def load_hailo_runtime():
    from hailo_platform import HEF, FormatType, VDevice
    return HEF, FormatType, VDevice


def run_pose_pipeline(
    model,
    conf=0.4,
    iou=0.5,
    width=1280,
    height=720,
    display=False,
    save_dir=None,
    on_result=None,
):
    HEF, FormatType, VDevice = load_hailo_runtime()

    in_h, in_w, _ = HEF(model).get_input_vstream_infos()[0].shape
    print(f"[INFO] model input {in_w}x{in_h}")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    cam = open_camera(width, height)

    with VDevice() as target:
        infer = target.create_infer_model(model)
        infer.input().set_format_type(FormatType.UINT8)          # feed raw uint8
        out_names = list(infer.output_names)
        for n in out_names:
            infer.output(n).set_format_type(FormatType.FLOAT32)  # HailoRT dequantizes

        with infer.configure() as cfg:
            in_name = infer.input().name
            in_shape = tuple(infer.input().shape)
            in_buf = np.empty(in_shape, dtype=np.uint8)
            out_bufs = {n: np.empty(tuple(infer.output(n).shape), np.float32)
                        for n in out_names}

            bindings = cfg.create_bindings()
            bindings.input(in_name).set_buffer(in_buf)
            for n, b in out_bufs.items():
                bindings.output(n).set_buffer(b)

            frames, t0, fps = 0, time.time(), 0.0
            try:
                while True:
                    loop_start = time.time()
                    # picamera2 'RGB888' returns a BGR-ordered array (Pi/libcamera quirk)
                    frame = cam.capture_array()
                    oh, ow = frame.shape[:2]
                    resized = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
                    model_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)   # model wants RGB
                    np.copyto(in_buf, model_in[None] if len(in_shape) == 4 else model_in)

                    cfg.run([bindings], timeout=10_000)
                    boxes, scores, kpts = postprocess(
                        [out_bufs[n] for n in out_names], conf, iou)

                    disp = frame.copy()                          # already BGR -> ok for cv2
                    draw(disp, boxes, scores, kpts, ow / in_w, oh / in_h)

                    frames += 1
                    if frames % 10 == 0:
                        fps = frames / (time.time() - t0)
                    cv2.putText(disp, f"{fps:4.1f} FPS  {len(boxes)} ppl", (10, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if on_result:
                        person_count = len(boxes)
                        on_result({
                            "frame": frame,
                            "annotated_frame": disp,
                            "current_class": "person" if person_count else "no_person",
                            "person_count": person_count,
                            "fps": fps,
                            "latency_ms": (time.time() - loop_start) * 1000.0,
                            "per_person_classes": ["person"] * person_count,
                        })

                    if display:
                        cv2.imshow("study-sheriff", disp)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    elif save_dir and frames % 15 == 0:
                        cv2.imwrite(os.path.join(save_dir, f"frame_{frames:05d}.jpg"), disp)
            except KeyboardInterrupt:
                pass
            finally:
                cam.stop()
                if display:
                    cv2.destroyAllWindows()
                print(f"[INFO] stopped after {frames} frames (~{fps:.1f} FPS)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--display", action="store_true", help="show a live window (VNC/HDMI)")
    ap.add_argument("--save-dir", default=None, help="headless: save annotated frames here")
    args = ap.parse_args()

    run_pose_pipeline(
        model=args.model,
        conf=args.conf,
        iou=args.iou,
        width=args.width,
        height=args.height,
        display=args.display,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
