#!/usr/bin/env python3
"""
StudySheriff -- model size / FLOPs / accuracy benchmark (Requirement 1).

Appends one row per compression stage to results/benchmarks.csv so the before/after
table (baseline -> pruned -> INT8) builds up automatically. FLOPs/params via
torch_pruning.utils.count_ops_and_params (the same counter used for pruning, §16.7),
with a thop fallback.

This is the OFF-HARDWARE size/FLOPs/accuracy tool (runs on Mac/Colab). It is NOT the
lab's on-Pi *latency* benchmark.py (Requirement 6 / §17.4) -- that one needs the NPU.

Examples
--------
# baseline fp32 classifier (params/FLOPs + on-disk .onnx size + test accuracy):
python scripts/benchmark_model.py --weights classifier.pt --file classifier.onnx \
    --tag baseline --data data/dataset_all.npz

# after structured pruning (save the slimmed model with torch.save(model, 'classifier_pruned.pt')):
python scripts/benchmark_model.py --weights classifier_pruned.pt --tag pruned-0.3 --data data/dataset_all.npz

# the compiled INT8 .hef (on-disk size only -- params/FLOPs aren't recoverable from a .hef):
python scripts/benchmark_model.py --file models/classifier.hef --tag int8-hef

Importable:  from scripts.benchmark_model import benchmark_model
"""
import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

# project root on path so we can reuse the model + eval definitions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classifier.train_classifier import (                    # noqa: E402
    FusionClassifier, PoseCropDataset, person_split, evaluate, CLASS_NAMES)

COLUMNS = ["timestamp", "tag", "params_M", "MACs_G", "FLOPs_G",
           "param_size_MB", "param+buf_MB", "file_MB", "accuracy"]


def count_ops_params(model, example):
    """Return (macs, params). macs is None if no counter is installed."""
    try:
        import torch_pruning as tp
        macs, params = tp.utils.count_ops_and_params(model, example)
        return float(macs), int(params)
    except Exception:
        pass
    try:
        from thop import profile
        macs, params = profile(model, inputs=example, verbose=False)
        return float(macs), int(params)
    except Exception:
        params = sum(p.numel() for p in model.parameters())
        print("[warn] install torch-pruning (or thop) for FLOPs -- reporting params only")
        return None, int(params)


def benchmark_model(model, tag, csv_path="results/benchmarks.csv",
                    file_path=None, accuracy=None, dtype_bytes=4):
    """Count params/MACs + sizes for a torch model, append a CSV row, print it."""
    model.eval().cpu()
    example = (torch.randn(1, 3, 224, 224), torch.randn(1, 51))
    macs, params = count_ops_params(model, example)
    buffers = sum(b.numel() for b in model.buffers())
    file_mb = os.path.getsize(file_path) / 1e6 if file_path and os.path.exists(file_path) else None

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "params_M": round(params / 1e6, 4),
        "MACs_G": round(macs / 1e9, 4) if macs is not None else "",
        "FLOPs_G": round(2 * macs / 1e9, 4) if macs is not None else "",      # FLOPs = 2 * MACs
        "param_size_MB": round(params * dtype_bytes / 1e6, 3),               # parameters only
        "param+buf_MB": round((params + buffers) * dtype_bytes / 1e6, 3),    # + buffers (BN stats etc.)
        "file_MB": round(file_mb, 3) if file_mb is not None else "",
        "accuracy": round(accuracy, 4) if accuracy is not None else "",
    }
    _append(csv_path, row)
    _print(row)
    return row


def file_only_row(tag, file_path, csv_path):
    """For artifacts with no torch graph (e.g. a .hef): record on-disk size only."""
    row = {c: "" for c in COLUMNS}
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    row["tag"] = tag
    row["file_MB"] = round(os.path.getsize(file_path) / 1e6, 3)
    _append(csv_path, row)
    _print(row)
    return row


def _append(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def _print(row):
    print(f"[bench] {row['tag']:14s} "
          f"params={row['params_M']}M  MACs={row['MACs_G']}G  FLOPs={row['FLOPs_G']}G  "
          f"size={row['param_size_MB']}MB(params)  file={row['file_MB']}MB  acc={row['accuracy']}")


def load_model(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):                         # full model (e.g. pruned, slimmed arch)
        return obj
    model = FusionClassifier(len(CLASS_NAMES))                   # a state_dict -> baseline arch
    model.load_state_dict(obj)
    return model


def compute_accuracy(model, data_path, batch=32):
    d = np.load(data_path, allow_pickle=True)
    crops, kpts, labels, subjects = d["crops"], d["keypoints"], d["labels"], d["subjects"]
    split = person_split(subjects, labels)
    name = "test" if (split == "test").sum() else "val"
    m = split == name
    dl = DataLoader(PoseCropDataset(crops[m], kpts[m], labels[m]), batch_size=batch)
    acc, _ = evaluate(model, dl, "cpu")
    return acc, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="stage label, e.g. baseline / pruned-0.3 / int8-hef")
    ap.add_argument("--weights", default=None, help="classifier .pt (state_dict or full model)")
    ap.add_argument("--file", default=None, help="artifact to measure on disk (.onnx/.pt/.hef)")
    ap.add_argument("--data", default=None, help="dataset.npz to compute test accuracy")
    ap.add_argument("--csv", default="results/benchmarks.csv")
    ap.add_argument("--dtype-bytes", type=int, default=4, help="4=fp32, 1=int8 (for size calc)")
    args = ap.parse_args()

    if args.weights:
        model = load_model(args.weights)
        acc = None
        if args.data:
            acc, name = compute_accuracy(model, args.data)
            print(f"[bench] accuracy ({name} split) = {acc:.4f}")
        benchmark_model(model, args.tag, args.csv,
                        file_path=args.file or args.weights, accuracy=acc, dtype_bytes=args.dtype_bytes)
    elif args.file:
        file_only_row(args.tag, args.file, args.csv)
    else:
        ap.error("give --weights (torch model) and/or --file (artifact to size)")

    print(f"[bench] appended to {args.csv}")


if __name__ == "__main__":
    main()
