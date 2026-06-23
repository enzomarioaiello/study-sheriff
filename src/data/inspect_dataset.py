#!/usr/bin/env python3
"""
StudySheriff -- sanity-check an extracted dataset before training.

Prints per-class counts AND a per-subject breakdown (so you can see every person
actually made it into the data), then writes example crops spread across each class
(filenames include the subject) so you can eyeball that labels match the activity.

    python src/data/inspect_dataset.py data/dataset.npz
"""
import os
import sys
from collections import Counter

import cv2
import numpy as np

NAMES = ["deskwork", "talking", "phone", "resting", "absent"]

path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.npz"
d = np.load(path, allow_pickle=True)
labels, crops, subjects = d["labels"], d["crops"], d["subjects"]

print(f"{path}: {len(labels)} samples, crop shape {crops.shape[1:]}")
print(f"all subjects: {sorted(set(subjects.tolist()))}\n")

for c in sorted(set(labels.tolist())):
    m = labels == c
    nm = NAMES[c] if 0 <= c < len(NAMES) else "odd/unknown"
    by_subj = dict(Counter(str(s) for s in subjects[m]))
    print(f"  label {c:2d} {nm:11s}: {m.sum():4d}   by subject: {by_subj}")

os.makedirs("check", exist_ok=True)
for c in sorted(set(labels.tolist())):
    idxs = np.where(labels == c)[0]
    if len(idxs) == 0:
        continue
    pick = idxs[np.linspace(0, len(idxs) - 1, min(8, len(idxs))).astype(int)]  # spread, not first-N
    for i in pick:
        cv2.imwrite(f"check/lbl{c}_{subjects[i]}_{i}.jpg",
                    cv2.cvtColor(crops[i], cv2.COLOR_RGB2BGR))
print("\nwrote sample crops to check/  (filenames now include the subject)")
