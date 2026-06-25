#!/usr/bin/env python3
"""
StudySheriff -- train the MobileNetV3 fusion activity classifier (Requirement 1).

Two branches fused into 5 classes:
    crop (224x224) -> MobileNetV3-Small backbone (576-d)
    51 pose values -> small MLP (64-d)
    concat -> linear head -> {deskwork, talking, phone, resting, absent}

Anti-overfit features (matter a LOT with few subjects):
  * paired augmentation: horizontal flip (crop + L/R-swapped keypoints), photometric
    jitter, random erasing -> forces the model off appearance/background shortcuts.
  * frozen / low-LR backbone (AdamW + weight decay) -> stops the backbone memorising
    one person; label smoothing + early stopping.
  * class weighting for post-merge imbalance.
  * leave-one-person-out cross-val (--cv) -> a stable accuracy estimate with few people.

External 'ext_*' subjects are always TRAIN-only; val/test stay on held-out own people.

    python src/classifier/train_classifier.py --data data/dataset_all.npz --epochs 30 --cv
"""
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

CLASS_NAMES = ["deskwork", "talking", "phone", "resting", "absent"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# COCO-17 left<->right swap (for a correct horizontal flip of the keypoints)
FLIP_PERM = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])


# ---------------------------- model ----------------------------
class FusionClassifier(nn.Module):
    """Crop backbone + pose MLP, fused. An auxiliary pose-only head + per-sample modality
    dropout force the (person-invariant) pose branch to stay discriminative on its own —
    directly targeting talking->deskwork and phone=0 (appearance-shortcut failures)."""
    def __init__(self, n_classes=5, n_kp=17, in_per_kp=3, pose_out=128, mod_drop=0.3):
        super().__init__()
        self.mod_drop = mod_drop
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        feat_dim = self.backbone.classifier[0].in_features          # 576
        self.backbone.classifier = nn.Identity()
        self.pose = nn.Sequential(
            nn.Linear(n_kp * in_per_kp, 128), nn.ReLU(True),
            nn.Dropout(0.3), nn.Linear(128, pose_out), nn.ReLU(True))
        self.pose_head = nn.Linear(pose_out, n_classes)            # aux: pose-only classifier
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat_dim + pose_out, n_classes))

    def forward(self, crop, kpts, return_aux=False):
        crop_feat = self.backbone(crop)
        pose_feat = self.pose(kpts)
        if self.training and self.mod_drop > 0:                    # per-sample modality dropout
            keep = (torch.rand(crop_feat.size(0), 1, device=crop_feat.device) > self.mod_drop).float()
            crop_feat = crop_feat * keep
        fused = self.head(torch.cat([crop_feat, pose_feat], dim=1))
        if return_aux:
            return fused, self.pose_head(pose_feat)
        return fused


# ---------------------------- data ----------------------------
class PoseCropDataset(Dataset):
    def __init__(self, crops, kpts, labels, train=False):
        self.crops, self.kpts, self.labels, self.train = crops, kpts, labels, train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.crops[i].astype(np.float32) / 255.0              # HWC RGB [0,1]
        kp = self.kpts[i].reshape(-1, 3).copy()                     # (17,3) box-normalized
        if self.train:
            if np.random.rand() < 0.5:                              # paired horizontal flip
                img = img[:, ::-1, :].copy()
                kp = kp[FLIP_PERM]
                kp[:, 0] = 1.0 - kp[:, 0]
            img, kp = self._affine(img, kp)                         # joint rotate/scale/translate
            img = img * np.random.uniform(0.75, 1.25)               # brightness
            m = img.mean()
            img = (img - m) * np.random.uniform(0.75, 1.25) + m     # contrast
            img = np.clip(img + np.random.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
            if np.random.rand() < 0.5:                              # random erasing (cutout)
                h, w, _ = img.shape
                eh, ew = np.random.randint(h // 6, h // 2), np.random.randint(w // 6, w // 2)
                y0, x0 = np.random.randint(0, h - eh), np.random.randint(0, w - ew)
                img[y0:y0 + eh, x0:x0 + ew, :] = np.random.rand()
        img = ((img - MEAN) / STD).transpose(2, 0, 1).copy()        # CHW, ImageNet norm
        return (torch.from_numpy(img).float(),
                torch.from_numpy(kp.reshape(-1)).float(),
                int(self.labels[i]))

    @staticmethod
    def _affine(img, kp):
        """Random rotate/scale/translate applied JOINTLY to the crop and the keypoints —
        manufactures the within-class variety that one clip per class lacks, and hardens
        against the deploy-time crop framing differing from extract_features' pad."""
        h, w, _ = img.shape
        ang, scale = np.random.uniform(-15, 15), np.random.uniform(0.85, 1.15)
        tx, ty = np.random.uniform(-0.08, 0.08, 2) * np.array([w, h])
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
        M[0, 2] += tx; M[1, 2] += ty
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        pts = (M[:, :2] @ (kp[:, :2] * np.array([w, h])).T).T + M[:, 2]
        kp[:, 0], kp[:, 1] = pts[:, 0] / w, pts[:, 1] / h
        return img, kp


def person_split(subjects, labels, seed=0, val=0.2, test=0.2):
    """Person-disjoint. ext_* -> train only. <=2 own people -> train/val (no test)."""
    rng = np.random.default_rng(seed)
    is_ext = np.array([str(s).startswith("ext") for s in subjects])
    own = np.array(sorted(set(subjects[(labels >= 0) & ~is_ext].tolist())))
    rng.shuffle(own)
    n = len(own)
    split = np.empty(len(labels), dtype=object)

    if n <= 2:
        train_p, val_p = set(own[:1].tolist()), set(own[1:].tolist())
        for i, (s, l) in enumerate(zip(subjects, labels)):
            split[i] = "odd" if l < 0 else "train" if (str(s).startswith("ext") or s in train_p) else "val"
        return split

    n_test, n_val = max(1, int(test * n)), max(1, int(val * n))
    test_p = set(own[:n_test].tolist())
    val_p = set(own[n_test:n_test + n_val].tolist())
    for i, (s, l) in enumerate(zip(subjects, labels)):
        if l < 0:
            split[i] = "odd"
        elif str(s).startswith("ext"):
            split[i] = "train"
        elif s in test_p:
            split[i] = "test"
        elif s in val_p:
            split[i] = "val"
        else:
            split[i] = "train"
    return split


# ---------------------------- train / eval ----------------------------
def make_optimizer(model, lr, wd, freeze_backbone):
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    # discriminative LR: backbone fine-tunes gently, head/pose learn fast
    return torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr * 0.1},
        {"params": list(model.pose.parameters()) + list(model.head.parameters()), "lr": lr},
    ], weight_decay=wd)


def evaluate(model, dl, dev):
    model.eval()
    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), int)
    with torch.no_grad():
        for img, kp, y in dl:
            p = model(img.to(dev), kp.to(dev)).argmax(1).cpu().numpy()
            for t, pr in zip(y.numpy(), p):
                cm[t, pr] += 1
    return cm.trace() / max(1, cm.sum()), cm


def class_weights(labels, dev):
    lab = labels[labels >= 0]
    counts = np.bincount(lab, minlength=len(CLASS_NAMES)).astype(np.float32)
    w = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=dev)


def train_fold(crops, kpts, labels, split, args, dev, save=None):
    """Train on split=='train', early-stop on 'val', return best model + (val_acc)."""
    def loader(name, shuffle=False, train=False):
        m = split == name
        return DataLoader(PoseCropDataset(crops[m], kpts[m], labels[m], train=train),
                          batch_size=args.batch, shuffle=shuffle, num_workers=2)

    tr, va = loader("train", True, True), loader("val")
    model = FusionClassifier(len(CLASS_NAMES)).to(dev)
    opt = make_optimizer(model, args.lr, args.wd, args.freeze_backbone)
    w = class_weights(labels[split == "train"], dev) if args.class_weight else None
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)

    best, best_state, patience = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for img, kp, y in tr:
            opt.zero_grad()
            yb = y.to(dev)
            fused, aux = model(img.to(dev), kp.to(dev), return_aux=True)
            loss = crit(fused, yb) + 0.5 * crit(aux, yb)           # + auxiliary pose-only loss
            loss.backward(); opt.step()
            loss_sum += loss.item()
        acc, _ = evaluate(model, va, dev)
        print(f"  epoch {ep + 1:2d}/{args.epochs}  loss={loss_sum / max(1, len(tr)):.3f}  val_acc={acc:.3f}")
        if acc > best:
            best, best_state, patience = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop @ epoch {ep + 1} (best val_acc={best:.3f})")
                break
    if best_state:
        model.load_state_dict(best_state)
    if save:
        torch.save(model.state_dict(), save)
    return model, best


def print_cm(acc, cm, title):
    print(f"\n[{title}] overall accuracy = {acc:.3f}")
    print("confusion matrix (rows=true, cols=pred):")
    print("           " + " ".join(f"{c[:6]:>6s}" for c in CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        row = cm[i]
        recall = row[i] / max(1, row.sum())
        print(f"  {c:9s} " + " ".join(f"{v:6d}" for v in row) + f"   recall={recall:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4, help="weight decay (anti-overfit)")
    ap.add_argument("--patience", type=int, default=6, help="early-stopping patience")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="freeze MobileNetV3 (use ImageNet features) — strongest anti-overfit for tiny data")
    ap.add_argument("--class-weight", action="store_true", help="weight loss by inverse class frequency")
    ap.add_argument("--cv", action="store_true", help="leave-one-person-out cross-validation (stable small-data estimate)")
    ap.add_argument("--out", default="classifier.pt")
    ap.add_argument("--onnx", default="classifier.onnx")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.data, allow_pickle=True)
    crops, kpts, labels, subjects = d["crops"], d["keypoints"], d["labels"], d["subjects"]
    print(f"[INFO] {len(labels)} samples | device={dev}")

    # ---- leave-one-person-out CV: honest accuracy when you have few people ----
    if args.cv:
        is_ext = np.array([str(s).startswith("ext") for s in subjects])
        own = sorted(set(subjects[(labels >= 0) & ~is_ext].tolist()))
        print(f"[CV] leave-one-person-out over {own}")
        agg = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), int)
        accs = []
        for held in own:
            split = np.array(["test" if (s == held and l >= 0) else
                              "odd" if l < 0 else "train" for s, l in zip(subjects, labels)], dtype=object)
            # carve a small val out of train for early stopping (one of the remaining own people)
            rest = [p for p in own if p != held]
            if rest:
                vp = rest[0]
                split[(subjects == vp) & (labels >= 0)] = "val"
            print(f"\n[CV] hold out '{held}'  (train {len([p for p in rest if p!=({vp} if rest else '')])}, val={rest[0] if rest else '-'})")
            model, _ = train_fold(crops, kpts, labels, split, args, dev)
            acc, cm = evaluate(model, DataLoader(
                PoseCropDataset(crops[split == "test"], kpts[split == "test"], labels[split == "test"]),
                batch_size=args.batch), dev)
            print_cm(acc, cm, f"fold: held-out {held}")
            agg += cm; accs.append(acc)
        print_cm(float(np.mean([agg.trace() / max(1, agg.sum())])), agg, "CV aggregate")
        print(f"\n[CV] mean held-out accuracy = {np.mean(accs):.3f}  (folds: {[round(a,3) for a in accs]})")
        return

    # ---- single split + ONNX export ----
    split = person_split(subjects, labels)
    for name in ("train", "val", "test", "odd"):
        m = split == name
        print(f"   {name:5s}: {m.sum():4d}  subjects={sorted(set(subjects[m].tolist()))}")
    model, _ = train_fold(crops, kpts, labels, split, args, dev, save=args.out)

    test_name = "test" if (split == "test").sum() else "val"
    m = split == test_name
    acc, cm = evaluate(model, DataLoader(
        PoseCropDataset(crops[m], kpts[m], labels[m]), batch_size=args.batch), dev)
    print_cm(acc, cm, f"TEST ({test_name})")
    suggest_threshold(model, dev, crops, kpts, split, args.batch)
    export_onnx(model, args.onnx)


def suggest_threshold(model, dev, crops, kpts, split, batch):
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
        tau = float(np.percentile(known, 5))
        print(f"\n[OOD] suggested unknown threshold tau = {tau:.3f}  (known mean={known.mean():.3f})")
        if len(odd):
            print(f"      odd conf mean={odd.mean():.3f} | flagged unknown @tau: {(odd < tau).mean() * 100:.0f}%")


def export_onnx(model, path):
    model.eval().cpu()
    img, kpts = torch.randn(1, 3, 224, 224), torch.randn(1, 51)
    torch.onnx.export(model, (img, kpts), path,
                      input_names=["crop", "keypoints"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    print(f"[ONNX] exported -> {path} (opset 17, static, batch 1)")


if __name__ == "__main__":
    main()
