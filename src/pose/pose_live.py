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
from hailo_platform import HEF, VDevice, FormatType

# Shared decode (run as a script: src/pose is on sys.path[0], so `from decode`).
from decode import SKELETON, postprocess


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

    in_h, in_w, _ = HEF(args.model).get_input_vstream_infos()[0].shape
    print(f"[INFO] model input {in_w}x{in_h}")
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    cam = open_camera(args.width, args.height)

    with VDevice() as target:
        infer = target.create_infer_model(args.model)
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
                    # picamera2 'RGB888' returns a BGR-ordered array (Pi/libcamera quirk)
                    frame = cam.capture_array()
                    oh, ow = frame.shape[:2]
                    resized = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
                    model_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)   # model wants RGB
                    np.copyto(in_buf, model_in[None] if len(in_shape) == 4 else model_in)

                    cfg.run([bindings], timeout=10_000)
                    boxes, scores, kpts = postprocess(
                        [out_bufs[n] for n in out_names], args.conf, args.iou)

                    disp = frame.copy()                          # already BGR -> ok for cv2
                    draw(disp, boxes, scores, kpts, ow / in_w, oh / in_h)

                    frames += 1
                    if frames % 10 == 0:
                        fps = frames / (time.time() - t0)
                    cv2.putText(disp, f"{fps:4.1f} FPS  {len(boxes)} ppl", (10, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if args.display:
                        cv2.imshow("study-sheriff", disp)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    elif args.save_dir and frames % 15 == 0:
                        cv2.imwrite(os.path.join(args.save_dir, f"frame_{frames:05d}.jpg"), disp)
            except KeyboardInterrupt:
                pass
            finally:
                cam.stop()
                if args.display:
                    cv2.destroyAllWindows()
                print(f"[INFO] stopped after {frames} frames (~{fps:.1f} FPS)")


if __name__ == "__main__":
    main()
