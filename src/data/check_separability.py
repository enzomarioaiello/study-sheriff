#!/usr/bin/env python3
"""
StudySheriff -- are the 5 classes actually SEPARABLE by pose?

If the seated classes have near-identical mean keypoints, NO model can tell them apart
(you'll get the deskwork-collapse). Run this after (re-)recording to confirm the
activities are pose-distinct BEFORE spending a training run.

    python src/data/check_separability.py data/dataset.npz
"""
import sys

import numpy as np

NAMES = ["deskwork", "talking", "phone", "resting", "absent"]

d = np.load(sys.argv[1] if len(sys.argv) > 1 else "data/dataset.npz", allow_pickle=True)
labels, kpts = d["labels"], d["keypoints"]
classes = [c for c in sorted(set(labels.tolist())) if 0 <= c < 5]
means = {c: kpts[labels == c].mean(0) for c in classes}        # (51,) mean keypoint vector

print("Pairwise mean-keypoint distance  (SMALL = hard to separate by pose):")
print("          " + " ".join(f"{NAMES[c][:7]:>7s}" for c in classes))
for a in classes:
    row = " ".join(f"{np.linalg.norm(means[a] - means[b]):7.3f}" for b in classes)
    print(f"  {NAMES[a]:8s} {row}")

print("\nMean wrist height (y: 0=top, 1=bottom) — 'phone' should be clearly HIGHER (smaller y) than 'deskwork':")
for c in classes:
    k = means[c].reshape(17, 3)
    wy, wv = k[[9, 10], 1].mean(), k[[9, 10], 2].mean()        # COCO 9/10 = L/R wrist
    print(f"  {NAMES[c]:8s}: wrist_y={wy:.2f}  wrist_vis={wv:.2f}")

print("\nFlagging seated-class pairs that are too close to separate:")
seated = [c for c in classes if c in (0, 1, 2, 3)]
bad = False
for i, a in enumerate(seated):
    for b in seated[i + 1:]:
        dist = np.linalg.norm(means[a] - means[b])
        if dist < 0.30:
            print(f"  ⚠ {NAMES[a]} ~ {NAMES[b]}: dist={dist:.3f}  -> act these MORE distinctly")
            bad = True
if not bad:
    print("  ✓ all seated-class pairs are reasonably distinct in pose")
