#!/usr/bin/env python3
"""
StudySheriff -- merge per-source feature .npz files into one training dataset.

Concatenates any number of {crops, keypoints, labels, subjects, sources} files
(your own data.npz + ext_stanford40.npz + ext_scb.npz) into one. All must share the
same crop size (224x224x3) and the 51-value keypoint layout.

    python src/data/merge_datasets.py data/dataset.npz data/ext_stanford40.npz \
        data/ext_scb.npz --output data/dataset_all.npz
"""
import argparse
import os

import numpy as np

KEYS = ["crops", "keypoints", "labels", "subjects", "sources"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--output", default="data/dataset_all.npz")
    args = ap.parse_args()

    acc = {k: [] for k in KEYS}
    for f in args.inputs:
        d = np.load(f, allow_pickle=True)
        print(f"  {f}: {len(d['labels'])} samples")
        for k in KEYS:
            acc[k].append(d[k])
    out = {k: np.concatenate(acc[k]) for k in KEYS}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, **out)
    labs = out["labels"]
    print(f"\n[DONE] {len(labs)} total -> {args.output}")
    n_ext = sum(str(s).startswith("ext") for s in out["subjects"])
    print(f"   own: {len(labs) - n_ext} | external: {n_ext}")
    for c in sorted(set(labs.tolist())):
        print(f"   label {c:2d}: {(labs == c).sum()}")


if __name__ == "__main__":
    main()
