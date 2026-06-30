# StudySheriff Phase 3a: Model Compression

This is the plug-and-play path for Phase 3a once the final activity classifier
checkpoint exists.

## What This Covers

Requirement 1 asks for:

- an efficient model architecture: the existing MobileNetV3 crop branch plus
  pose MLP fusion classifier
- at least two compression techniques: structured pruning and INT8 quantization
- an unseen-data technique: the trainer already reports a softmax confidence
  threshold for `Unknown`
- a before/after table: size, FLOPs, and accuracy per compression step

The code here prepares the baseline and structured-pruned rows locally or in
Colab, then prepares calibration tensors for the Hailo compiler. The final INT8
`.hef` size, latency, FPS, and accuracy must still be measured on the x86 Hailo
compiler machine and on the Raspberry Pi 5 + Hailo-10H.

## 1. Train Or Export The Baseline

After the dataset is ready:

```bash
python src/classifier/train_classifier.py \
  --data data/dataset_all.npz \
  --epochs 20 \
  --out models/classifier_baseline.pt \
  --onnx models/classifier_baseline.onnx
```

The checkpoint path is the input to compression.

## 2. Structured Pruning

Run this in Colab or another machine with PyTorch:

```bash
python src/classifier/compress_classifier.py \
  --data data/dataset_all.npz \
  --checkpoint models/classifier_baseline.pt \
  --out-dir results/compression \
  --prune-amount 0.25 \
  --fine-tune-epochs 5
```

Outputs:

- `results/compression/classifier_baseline.onnx`
- `results/compression/classifier_pruned.pt`
- `results/compression/classifier_pruned.onnx`
- `results/compression/classifier_calib.npz`
- `results/compression/compression_report.csv`
- `results/compression/compression_report.json`
- baseline and pruned confusion matrices as `.npy`

The report contains test accuracy, parameter count, nonzero parameter count,
sparsity, checkpoint size, ONNX size, dense MFLOPs, and estimated active MFLOPs
after removing pruned output channels/units.

If accuracy drops too much, lower `--prune-amount` or increase
`--fine-tune-epochs`. The default guardrail fails if test accuracy drops by more
than 5 percentage points.

## 3. INT8 Quantization With Hailo DFC

Copy these files to the x86-64 Linux machine that has Hailo Dataflow Compiler
5.3.0 installed:

- `results/compression/classifier_pruned.onnx`
- `results/compression/classifier_calib.npz`
- `scripts/compile_classifier_hailo.py`

Dry-run the compile plan:

```bash
python scripts/compile_classifier_hailo.py \
  --onnx results/compression/classifier_pruned.onnx \
  --calib results/compression/classifier_calib.npz \
  --hef models/classifier_pruned_int8.hef \
  --hw-arch hailo10h \
  --dry-run
```

Then run without `--dry-run` in the Hailo DFC environment:

```bash
python scripts/compile_classifier_hailo.py \
  --onnx results/compression/classifier_pruned.onnx \
  --calib results/compression/classifier_calib.npz \
  --hef models/classifier_pruned_int8.hef \
  --hw-arch hailo10h \
  --metrics-json results/compression/int8_hailo_metrics.json
```

The calibration file stores preprocessed inputs with names matching the ONNX
graph:

- `crop`: float32, shape `[N, 3, 224, 224]`, ImageNet-normalized
- `keypoints`: float32, shape `[N, 51]`

Use real StudySheriff crops/keypoints for calibration. Random data can make INT8
accuracy collapse.

## 4. Report Table Template

Use `compression_report.csv` for the baseline and structured-pruned rows. Add
the Hailo INT8 row after compiling and benchmarking the `.hef`.

| Step | Accuracy | Checkpoint/HEF size | Dense MFLOPs | Active MFLOPs est. | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline FP32 | from CSV | from CSV | from CSV | from CSV | trained model |
| Structured pruned FP32 | from CSV | from CSV | from CSV | from CSV | after fine-tuning |
| INT8 HEF | measured | `.hef` size | Hailo/estimate | Hailo/estimate | Hailo DFC compile |

For the final demo/report, also record Pi-side latency/FPS with the integrated
pipeline and compare compressed vs uncompressed behavior.
