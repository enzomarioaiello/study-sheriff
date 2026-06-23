#!/usr/bin/env python3
"""
StudySheriff -- train the MobileNetV3 fusion activity classifier (Requirement 1).

Two branches fused into 5 classes:
    crop (224x224) -> MobileNetV3-Small backbone (576-d)
    51 pose values -> small MLP (64-d)
    concat -> linear head -> {deskwork, talking, phone, resting, absent}

Trains on data/dataset.npz from src/data/extract_features.py with a PERSON-DISJOINT
split (whole people held out for test -> honest accuracy), reports baseline accuracy
+ confusion matrix, suggests the Unknown (OOD) threshold, and exports ONNX for the
Hailo DFC (Phase 3). Run in Colab (GPU) or locally:

    python src/classifier/train_classifier.py --data data/dataset.npz --epochs 20
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

CLASS_NAMES = ["deskwork", "talking", "phone", "resting", "absent"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------- model ----------------------------
class FusionClassifier(nn.Module):
    def __init__(self, n_classes=5, n_kp=17, in_per_kp=3, pose_out=64):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        feat_dim = self.backbone.classifier[0].in_features          # 576 (read, don't hardcode)
        self.backbone.classifier = nn.Identity()
        self.pose = nn.Sequential(
            nn.Linear(n_kp * in_per_kp, 128), nn.ReLU(True),
            nn.Dropout(0.2), nn.Linear(128, pose_out))
        self.head = nn.Linear(feat_dim + pose_out, n_classes)

    def forward(self, crop, kpts):
        return self.head(torch.cat([self.backbone(crop), self.pose(kpts)], dim=1))


# ---------------------------- data ----------------------------
class PoseCropDataset(Dataset):
    def __init__(self, crops, kpts, labels):
        self.crops, self.kpts, self.labels = crops, kpts, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.crops[i].astype(np.float32) / 255.0              # HWC RGB [0,1]
        img = ((img - MEAN) / STD).transpose(2, 0, 1).copy()        # CHW, ImageNet norm
        return (torch.from_numpy(img).float(),
                torch.from_numpy(self.kpts[i]).float(),
                int(self.labels[i]))


def person_split(subjects, labels, seed=0, val=0.15, test=0.15):
    """Split by SUBJECT (person-disjoint). External 'ext_*' data -> TRAIN only;
    val/test stay on your own held-out people. label -1 -> 'odd' (threshold tuning)."""
    rng = np.random.default_rng(seed)
    is_ext = np.array([str(s).startswith("ext") for s in subjects])
    own = np.array(sorted(set(subjects[(labels >= 0) & ~is_ext].tolist())))
    rng.shuffle(own)
    n = len(own)
    n_test = max(1, int(round(test * n)))
    n_val = max(1, int(round(val * n)))
    test_s = set(own[:n_test].tolist())
    val_s = set(own[n_test:n_test + n_val].tolist())
    split = np.empty(len(labels), dtype=object)
    for i, (s, l) in enumerate(zip(subjects, labels)):
        if l < 0:
            split[i] = "odd"
        elif str(s).startswith("ext"):
            split[i] = "train"                       # external augments training only
        elif s in test_s:
            split[i] = "test"
        elif s in val_s:
            split[i] = "val"
        else:
            split[i] = "train"
    return split


# ---------------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="classifier.pt")
    ap.add_argument("--onnx", default="classifier.onnx")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.data, allow_pickle=True)
    crops, kpts, labels, subjects = d["crops"], d["keypoints"], d["labels"], d["subjects"]
    print(f"[INFO] {len(labels)} samples | device={dev}")

    split = person_split(subjects, labels)
    for name in ("train", "val", "test", "odd"):
        m = split == name
        print(f"   {name:5s}: {m.sum():4d} samples  subjects={sorted(set(subjects[m].tolist()))}")
    if (split == "test").sum() == 0:
        print("[WARN] empty test set -> record clips/photos of at least 3-4 DIFFERENT people")

    def loader(name, shuffle=False):
        m = split == name
        return DataLoader(PoseCropDataset(crops[m], kpts[m], labels[m]),
                          batch_size=args.batch, shuffle=shuffle, num_workers=2)

    tr, va, te = loader("train", True), loader("val"), loader("test")
    model = FusionClassifier(len(CLASS_NAMES)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    def evaluate(dl):
        model.eval()
        cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), int)
        with torch.no_grad():
            for img, kp, y in dl:
                p = model(img.to(dev), kp.to(dev)).argmax(1).cpu().numpy()
                for t, pr in zip(y.numpy(), p):
                    cm[t, pr] += 1
        acc = cm.trace() / max(1, cm.sum())
        return acc, cm

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for img, kp, y in tr:
            opt.zero_grad()
            loss = crit(model(img.to(dev), kp.to(dev)), y.to(dev))
            loss.backward(); opt.step()
            loss_sum += loss.item()
        acc, _ = evaluate(va)
        print(f"epoch {ep + 1:2d}/{args.epochs}  loss={loss_sum / max(1, len(tr)):.3f}  val_acc={acc:.3f}")
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), args.out)

    # ---- baseline test report (Requirement 1: before-compression numbers) ----
    model.load_state_dict(torch.load(args.out, map_location=dev))
    acc, cm = evaluate(te)
    print(f"\n[TEST] overall accuracy = {acc:.3f}")
    print("confusion matrix (rows=true, cols=pred):")
    print("           " + " ".join(f"{c[:6]:>6s}" for c in CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:9s} " + " ".join(f"{v:6d}" for v in cm[i]))

    suggest_threshold(model, dev, crops, kpts, split, args.batch)
    export_onnx(model, args.onnx)


def suggest_threshold(model, dev, crops, kpts, split, batch):
    """Pick the Unknown (OOD) softmax threshold from known(val) vs odd confidences."""
    def confs(name):
        m = split == name
        if m.sum() == 0:
            return np.array([])
        dl = DataLoader(PoseCropDataset(crops[m], kpts[m], np.zeros(m.sum())), batch_size=batch)
        model.eval(); out = []
        with torch.no_grad():
            for img, kp, _ in dl:
                out.append(torch.softmax(model(img.to(dev), kp.to(dev)), 1).max(1).values.cpu().numpy())
        return np.concatenate(out)
    known, odd = confs("val"), confs("odd")
    if len(known):
        tau = float(np.percentile(known, 5))           # keep ~95% of known above tau
        print(f"\n[OOD] suggested unknown threshold tau = {tau:.3f}")
        print(f"      known(val) conf: mean={known.mean():.3f}")
        if len(odd):
            print(f"      odd        conf: mean={odd.mean():.3f} | flagged unknown @tau: "
                  f"{(odd < tau).mean() * 100:.0f}%")


def export_onnx(model, path):
    """Hailo-DFC-ready: opset pinned, static shapes, batch 1, legacy tracer."""
    model.eval().cpu()
    img, kpts = torch.randn(1, 3, 224, 224), torch.randn(1, 51)
    torch.onnx.export(model, (img, kpts), path,
                      input_names=["crop", "keypoints"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    print(f"[ONNX] exported -> {path} (opset 17, static, batch 1)")


if __name__ == "__main__":
    main()
