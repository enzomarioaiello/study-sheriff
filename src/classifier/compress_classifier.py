"""
Model-compression pipeline.

Loads a trained FusionClassifier checkpoint, records baseline metrics, applies
structured pruning, optionally fine-tunes on the train split, exports a pruned
ONNX model, and writes Hailo INT8 calibration tensors for the later x86/Linux
Dataflow Compiler step.
"""
import argparse
import csv
import io
import json
import shutil
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classifier.train_classifier import (  # noqa: E402
    CLASS_NAMES,
    MEAN,
    STD,
    FusionClassifier,
    PoseCropDataset,
    export_onnx,
    person_split,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model, checkpoint, device):
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def make_loader(crops, kpts, labels, split, name, batch, shuffle=False):
    mask = split == name
    return DataLoader(
        PoseCropDataset(crops[mask], kpts[mask], labels[mask]),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=2,
    )


def evaluate(model, loader, device):
    model.eval()
    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    with torch.no_grad():
        for img, kp, y in loader:
            y_np = y.numpy()
            valid = (y_np >= 0) & (y_np < len(CLASS_NAMES))
            if not valid.any():
                continue
            pred = model(img.to(device), kp.to(device)).argmax(1).cpu().numpy()
            for target, guessed in zip(y_np[valid], pred[valid]):
                cm[int(target), int(guessed)] += 1
    total = int(cm.sum())
    acc = float(cm.trace() / total) if total else None
    return acc, cm


def state_dict_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / (1024 * 1024)


def file_size_mb(path):
    path = Path(path)
    if not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


def parameter_counts(model):
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += int(torch.count_nonzero(p.detach()).item())
    return total, nonzero


def active_out_channels(module):
    weight = module.weight.detach()
    flat = weight.reshape(weight.shape[0], -1)
    return int((flat.abs().sum(dim=1) > 0).sum().item())


def estimate_flops(model, device):
    """Estimate Conv2d/Linear FLOPs for batch=1.

    dense_flops counts the exported graph. active_flops estimates operations
    left if zeroed output channels/rows are physically removed by deployment.
    """
    totals = {"dense": 0, "active": 0}
    handles = []

    def conv_hook(module, inputs, output):
        batch, out_channels, out_h, out_w = output.shape
        kernel_h, kernel_w = module.kernel_size
        kernel_ops = kernel_h * kernel_w * (module.in_channels // module.groups)
        dense_macs = batch * out_channels * out_h * out_w * kernel_ops
        active_macs = batch * active_out_channels(module) * out_h * out_w * kernel_ops
        totals["dense"] += 2 * dense_macs
        totals["active"] += 2 * active_macs

    def linear_hook(module, inputs, output):
        batch = output.shape[0] if output.ndim > 1 else 1
        dense_macs = batch * module.in_features * module.out_features
        active_macs = batch * module.in_features * active_out_channels(module)
        totals["dense"] += 2 * dense_macs
        totals["active"] += 2 * active_macs

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(
            torch.randn(1, 3, 224, 224, device=device),
            torch.randn(1, 51, device=device),
        )
    if was_training:
        model.train()
    for handle in handles:
        handle.remove()

    return totals["dense"] / 1e6, totals["active"] / 1e6


def collect_metrics(step, model, device, accuracy=None, onnx_path=None):
    params, nonzero = parameter_counts(model)
    dense_mflops, active_mflops = estimate_flops(model, device)
    return {
        "step": step,
        "accuracy": accuracy,
        "params": params,
        "nonzero_params": nonzero,
        "sparsity": 1.0 - (nonzero / max(1, params)),
        "checkpoint_mb": state_dict_size_mb(model),
        "onnx_mb": file_size_mb(onnx_path) if onnx_path else None,
        "dense_mflops": dense_mflops,
        "active_mflops_est": active_mflops,
    }


def try_export_onnx(model, path):
    try:
        export_onnx(model, path)
        return str(path)
    except Exception as exc:
        print(f"[WARN] ONNX export skipped for {path}: {exc}")
        return None


def apply_structured_pruning(model, amount, min_channels):
    pruned = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels >= min_channels:
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
            pruned.append(name)
        elif isinstance(module, nn.Linear):
            if module.out_features == len(CLASS_NAMES):
                continue
            if module.out_features >= min_channels:
                prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
                pruned.append(name)

    for module in model.modules():
        if hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
    return pruned


def fine_tune(model, train_loader, val_loader, device, epochs, lr):
    if epochs <= 0:
        return None

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        steps = 0
        for img, kp, y in train_loader:
            valid = (y >= 0) & (y < len(CLASS_NAMES))
            if not bool(valid.any()):
                continue
            opt.zero_grad()
            logits = model(img[valid].to(device), kp[valid].to(device))
            loss = crit(logits, y[valid].to(device))
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            steps += 1

        val_acc, _ = evaluate(model, val_loader, device)
        score = -1.0 if val_acc is None else val_acc
        print(
            f"fine-tune {ep + 1:2d}/{epochs} "
            f"loss={loss_sum / max(1, steps):.3f} "
            f"val_acc={score:.3f}"
        )
        if score >= best_acc:
            best_acc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc if best_acc >= 0 else None


def preprocess_crops(crops):
    x = crops.astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(0, 3, 1, 2)
    return x.astype(np.float32)


def export_calibration(crops, kpts, split, out_path, split_name, max_samples, seed):
    mask = split == split_name
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError(f"no samples found for calibration split '{split_name}'")

    rng = np.random.default_rng(seed)
    if max_samples > 0 and len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    idx = np.sort(idx)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        crop=preprocess_crops(crops[idx]),
        keypoints=kpts[idx].astype(np.float32),
    )
    return len(idx)


def write_reports(rows, out_dir):
    json_path = out_dir / "compression_report.json"
    csv_path = out_dir / "compression_report.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def print_metrics(row):
    acc = "n/a" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
    print(
        f"{row['step']:>16s} | acc={acc} | "
        f"params={row['params']:,} | nonzero={row['nonzero_params']:,} | "
        f"sparsity={row['sparsity'] * 100:.1f}% | "
        f"active MFLOPs={row['active_mflops_est']:.1f} | "
        f"ckpt={row['checkpoint_mb']:.2f} MB"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset_all.npz",
                    help="optional dataset for accuracy, fine-tuning, and calibration")
    ap.add_argument("--checkpoint", default="models/classifier.pt",
                    help="trained baseline checkpoint from train_classifier.py")
    ap.add_argument("--baseline-onnx", default="models/classifier.onnx",
                    help="already-exported baseline ONNX to copy into the output folder")
    ap.add_argument("--out-dir", default="results/compression")
    ap.add_argument("--prune-amount", type=float, default=0.25,
                    help="fraction of output channels/units to prune in eligible layers")
    ap.add_argument("--min-prune-channels", type=int, default=16)
    ap.add_argument("--fine-tune-epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--calib-samples", type=int, default=512)
    ap.add_argument("--max-acc-drop", type=float, default=0.05,
                    help="fail if pruned test accuracy drops by more than this")
    args = ap.parse_args()

    if not (0.0 <= args.prune_amount < 1.0):
        raise SystemExit("--prune-amount must be in [0, 1)")

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path(args.data)
    has_data = data_path.exists()
    if has_data:
        data = np.load(data_path, allow_pickle=True)
        crops = data["crops"]
        kpts = data["keypoints"]
        labels = data["labels"]
        subjects = data["subjects"]
        split = person_split(subjects, labels, seed=args.seed)
        train_loader = make_loader(crops, kpts, labels, split, "train", args.batch, shuffle=True)
        val_loader = make_loader(crops, kpts, labels, split, "val", args.batch)
        test_loader = make_loader(crops, kpts, labels, split, "test", args.batch)
    else:
        print(f"[WARN] dataset not found at {data_path}; skipping accuracy, fine-tuning, and calibration")
        crops = kpts = labels = subjects = split = None
        train_loader = val_loader = test_loader = None

    model = FusionClassifier(len(CLASS_NAMES), pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device)

    baseline_onnx = Path(args.baseline_onnx)
    out_baseline_onnx = out_dir / "classifier_baseline.onnx"
    if baseline_onnx.exists():
        shutil.copy2(baseline_onnx, out_baseline_onnx)
    else:
        try_export_onnx(model, out_baseline_onnx)

    baseline_acc, baseline_cm = evaluate(model, test_loader, device) if has_data else (None, None)
    rows = [collect_metrics("baseline", model, device, baseline_acc, baseline_onnx)]
    print_metrics(rows[-1])

    pruned_layers = apply_structured_pruning(model, args.prune_amount, args.min_prune_channels)
    print(f"[PRUNE] {len(pruned_layers)} layers pruned @ amount={args.prune_amount}")
    if pruned_layers:
        suffix = " ..." if len(pruned_layers) > 12 else ""
        print("[PRUNE] " + ", ".join(pruned_layers[:12]) + suffix)

    if has_data:
        fine_tune(model, train_loader, val_loader, device, args.fine_tune_epochs, args.lr)
        pruned_acc, pruned_cm = evaluate(model, test_loader, device)
    else:
        print("[INFO] pruning masks applied without fine-tuning because no dataset is available")
        pruned_acc, pruned_cm = None, None

    pruned_ckpt = out_dir / "classifier_pruned.pt"
    torch.save(model.state_dict(), pruned_ckpt)
    pruned_onnx = out_dir / "classifier_pruned.onnx"
    exported_pruned_onnx = try_export_onnx(model, pruned_onnx)
    rows.append(collect_metrics("structured_prune", model, device, pruned_acc, pruned_onnx))
    print_metrics(rows[-1])

    if has_data:
        calib_path = out_dir / "classifier_calib.npz"
        calib_n = export_calibration(
            crops, kpts, split, calib_path, args.calib_split, args.calib_samples, args.seed
        )
        print(f"[CALIB] wrote {calib_n} samples -> {calib_path}")
    else:
        print("[WARN] no calibration file written; rerun with --data before Hailo INT8 compilation")

    if has_data:
        np.save(out_dir / "baseline_confusion_matrix.npy", baseline_cm)
        np.save(out_dir / "pruned_confusion_matrix.npy", pruned_cm)
    json_path, csv_path = write_reports(rows, out_dir)
    print(f"[REPORT] {json_path}")
    print(f"[REPORT] {csv_path}")

    if exported_pruned_onnx is None:
        print("[WARN] install the ONNX package, then rerun to produce classifier_pruned.onnx")

    if baseline_acc is not None and pruned_acc is not None:
        drop = baseline_acc - pruned_acc
        print(f"[GUARDRAIL] accuracy_drop={drop:.3f} max_allowed={args.max_acc_drop:.3f}")
        if drop > args.max_acc_drop:
            raise SystemExit(
                "pruned model failed accuracy guardrail; reduce --prune-amount "
                "or increase --fine-tune-epochs"
            )


if __name__ == "__main__":
    main()
