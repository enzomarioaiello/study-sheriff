"""
Compile the compressed StudySheriff classifier ONNX to an INT8 Hailo HEF.

Run this on x86-64 Linux with Hailo Dataflow Compiler installed. It is expected
to fail gracefully on the Mac/Pi/dev machine where the compiler is unavailable.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_calibration(path):
    data = np.load(path)
    required = {"crop", "keypoints"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"calibration file missing arrays: {sorted(missing)}")
    return {name: data[name].astype(np.float32) for name in sorted(required)}


def write_metrics(hef_path, metrics_path):
    if metrics_path is None:
        return
    hef_path = Path(hef_path)
    payload = {
        "step": "int8_hailo_compile",
        "hef": str(hef_path),
        "hef_mb": hef_path.stat().st_size / (1024 * 1024) if hef_path.exists() else None,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def print_dry_run(args):
    print("Hailo DFC compile plan:")
    print(f"  ONNX : {args.onnx}")
    print(f"  calib: {args.calib}  (npz arrays named crop, keypoints)")
    print(f"  HEF  : {args.hef}")
    print(f"  arch : {args.hw_arch}")
    print("")
    print("This script uses hailo_sdk_client.ClientRunner when available.")
    print("Run it on x86-64 Linux with Hailo DFC 5.3.0 for the Hailo-10H target.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True,
                    help="classifier_calib.npz from compress_classifier.py")
    ap.add_argument("--hef", required=True)
    ap.add_argument("--hw-arch", default="hailo10h")
    ap.add_argument("--model-name", default="study_sheriff_classifier")
    ap.add_argument("--end-node", default="logits")
    ap.add_argument("--metrics-json", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print_dry_run(args)
        return

    try:
        from hailo_sdk_client import ClientRunner
    except ImportError as exc:
        print_dry_run(args)
        raise SystemExit(
            "hailo_sdk_client is not installed here. Re-run on the x86 Linux "
            "Hailo DFC machine/conda environment."
        ) from exc

    calib = load_calibration(args.calib)
    runner = ClientRunner(hw_arch=args.hw_arch)
    net_input_shapes = {
        "crop": list(calib["crop"].shape[1:]),
        "keypoints": list(calib["keypoints"].shape[1:]),
    }

    # The ONNX export names are fixed in train_classifier.export_onnx:
    # inputs = crop, keypoints; output = logits.
    runner.translate_onnx_model(
        args.onnx,
        args.model_name,
        start_node_names=["crop", "keypoints"],
        end_node_names=[args.end_node],
        net_input_shapes=net_input_shapes,
    )
    runner.optimize(calib)
    hef = runner.compile()

    hef_path = Path(args.hef)
    hef_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hef_path, "wb") as f:
        f.write(hef)
    write_metrics(hef_path, args.metrics_json)
    print(f"[HEF] wrote {hef_path}")


if __name__ == "__main__":
    main()
