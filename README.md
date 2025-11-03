# YOLO11 — TTQ (Trained Ternary Quantization) + BitLinear Integration

This repository implements progressive Trained Ternary Quantization (TTQ) and a BitLinear_TTQ module for YOLO11 models (C2PSA attention). The goal is to compress YOLO11 (n/x sizes) and measure accuracy vs. compression vs. latency trade-offs on COCO.

---

## Table of Contents
- [Quick summary of what’s implemented](#quick-summary-of-whats-implemented)  
- [Prerequisites](#prerequisites)  
- [Directory / important files](#directory--important-files)  
- [How to run: full pipeline (both `yolo11n` and `yolo11x`)](#how-to-run---full-pipeline-both-yolo11n-and-yolo11x)  
- [Evaluation & benchmarking (validation, compression, latency)](#evaluation--benchmarking-validation-compression-latency)  
- [Reported results](#reported-results-from-your-logsconfigs)  
- [References](#references)

---

## Quick summary of what's implemented

- **ShadowWeightManager (TTQ)**: master (FP32) + shadow (ternary) model training, per-layer learnable scales `Wp` & `Wn`, progressive stage support.
- **BitLinear_TTQ**: Conv2d-compatible ternary BitLinear with LayerNorm + absmax activation quantization (8-bit), learnable Ap/An scales for C2PSA attention/FFN.
- **Standard BitLinear**: fixed-scale variant for ablation/comparison.
- **Scripts & tools**:
  - `train.py` — progressive TTQ stage runner (stages 1/2/3).
  - `train_c2psa.py` — train C2PSA with BitLinear_TTQ integration.
  - `train_c2psa_standard.py` — train standard BitLinear (fixed scales).
  - `compression_calculation.py` — compute theoretical compression & coverage.
  - `compare_bitlinear.py` — compare BitLinear_TTQ vs Standard BitLinear.
  - `latency_benchmark.py` — export to TensorRT and measure latency.

---

## Prerequisites

- Python 3.8+ (recommended)
- PyTorch (CUDA-enabled) matching your GPU
- Ultralytics YOLO (the `ultralytics` package)
- TensorRT (for latency benchmarking) — optional but recommended
- `pip install -r requirements.txt`
---

## Directory / important files

```

configs/
stage1_progressive.yaml       # progressive TTQ stages
stage2_c2psa.yaml             # C2PSA BitLinear_TTQ config

src/quantization/
shadow_weight_manager.py
c2psa_bitlinear_ttq.py
bitlinear_standard_manager.py

src/training/
c2psa_trainer.py
shadow_trainer.py

train.py                        # run progressive TTQ stages
train_c2psa.py                  # run C2PSA BitLinear_TTQ training
train_c2psa_standard.py         # run standard BitLinear training
compression_calculation.py
compare_bitlinear.py
latency_benchmark.py
ttq_checkpoints/                 # checkpoint outputs (created by training)
checkpoints/                     # created when running c2psa quanitzation

````

## How to run Full Pipeline

Before starting, ensure you have the correct YOLO model weights and all dependencies installed.

---

### 1. Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
````

Additional packages required for latency benchmarking:

```bash
pip install onnx onnxslim onnxruntime tensorrt
```

---

### 2. Download Pretrained YOLO Models

You need to download both pretrained YOLO11 models before training:

```bash
# YOLO11-n (nano)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

# YOLO11-x (extra-large)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
```

Keep these `.pt` files in your project root or adjust config paths accordingly.

---

### 3. Run Full Training Pipeline (YOLO11n)

A shell script `run.sh` is provided to automate all stages for YOLO11n.

```bash
bash run.sh
```

Contents of `run.sh`:

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

python train.py --config configs/stage1_progressive.yaml --stage 1
python train.py --config configs/stage1_progressive.yaml --stage 2
python train.py --config configs/stage1_progressive.yaml --stage 3

# Train C2PSA BitLinear_TTQ after TTQ stages
python train_c2psa.py \
  --config configs/stage2_c2psa.yaml \
  --model ttq_checkpoints/yolo11n/stage1_progressive_final/best.pt
```

This script downloads the base model, executes all three TTQ stages sequentially, and finally runs the C2PSA BitLinear training.

---

### 4. Validate Trained Models

To validate any saved checkpoint on the COCO dataset:

```bash
python validate.py saved_models/yolo11n/stage1.pt --data coco.yaml
```

You can replace `stage1.pt` with any other checkpoint (e.g., `stage1-3+bitlinear_ttq.pt`).

---

### 5. Run TensorRT Latency Benchmark

To benchmark baseline and quantized models (requires `onnx`, `onnxruntime`, and `tensorrt`):

```bash
python latency_benchmark.py \
  --baseline yolo11n.pt \
  --quantized saved_models/yolo11n/stage1-3+bitlinear_ttq.pt \
  --export-fp16 \
  --export-int8 \
  --num-runs 30
```

This script exports the models to ONNX and TensorRT, runs multiple inference passes, and reports:

* Mean latency
* Throughput (FPS)
* Memory usage

---

### 6. Compute Model Compression

To evaluate compression ratio between baseline and quantized models:

```bash
python compression_calculation.py \
  --baseline yolo11x.pt \
  --quantized saved_models/yolo11x/stage1-3+bitlinear_ttq.pt
```

The script prints model sizes, theoretical compression ratio, and percentage of quantized layers.

---

### 7. Inspect Quantized Layers (Ternary Check)

To verify which layers have been converted to ternary:

```bash
python test.py saved_models/yolo11n/stage1-3+bitlinear_ttq.pt
```

This script outputs layer names and indicates whether they are quantized (ternary) or remain in FP32.

---

 **Summary of Available Scripts**

| Script                       | Purpose                                               |
| ---------------------------- | ----------------------------------------------------- |
| `run.sh`                     | Downloads YOLO11n and runs all TTQ + BitLinear stages |
| `validate.py`                | Runs COCO validation and prints mAP metrics           |
| `latency_benchmark.py`       | Exports and benchmarks models using TensorRT          |
| `compression_calculation.py` | Computes compression ratios and quantization coverage |
| `test.py`                    | Checks which layers are ternary quantized             |

---

**Note:**
For YOLO11x, the same workflow applies — download `yolo11x.pt` using the command above and substitute it in place of `yolo11n.pt` wherever needed.



## Evaluation & benchmarking (what to run, where outputs go)

* Training outputs: `ttq_checkpoints/yolo11{n|x}/.../best.pt`
* Validation metrics (printed during training) — final `mAP50` and `mAP50-95` are saved in `results` printed to console by trainers.
* `analyze_compression.py` prints:

  * estimated baseline & quantized size (MB)
  * theoretical compression ratio
  * percent of parameters quantized
* `compare_bitlinear.py` prints COCO128 validation metrics (mAP50, mAP50-95, precision, recall) for Baseline / Our Impl / Standard.
* `latency_benchmark.py` prints baseline and quantized model latency (ms).

---

## Reported results (extracted from your logs / files)

> These are the numbers present in your files and logs. Use them as the canonical results table in the short report.

### YOLO11n (COCO fine-tune / validation)

| Stage                     |  mAP50 | mAP50-95 | Theoretical compression |
| ------------------------- | -----: | -------: | ----------------------: |
| Baseline                  | 0.5485 |   0.3924 |                   1.00× |
| Stage 1                   | 0.5088 |   0.3410 |                   1.30× |
| Stage 1 + 2               | 0.4702 |   0.3088 |                   1.69× |
| Stage 1 + 2 + 3           | 0.4110 |   0.2639 |                   1.97× |
| Stage 1–3 + BitLinear_TTQ | 0.2955 |   0.1819 |                   2.16× |

**BitLinear comparison (YOLO11n; COCO validation):**

* Baseline: mAP50 0.5487, mAP50-95 0.3925, Precision 0.4652, Recall 0.2807
* Our Impl (BitLinear_TTQ): mAP50 0.4421, mAP50-95 0.3098, Precision 0.3702, Recall 0.0702
* Standard BitLinear: mAP50 0.4427, mAP50-95 0.3098, Precision 0.3354, Recall 0.0709

### YOLO11x (COCO)

| Stage                     |                           mAP50 | mAP50-95 | Theoretical compression |
| ------------------------- | ------------------------------: | -------: | ----------------------: |
| Baseline                  |                          0.7135 |   0.5485 |                   1.00× |
| Stage 1                   |                          0.6891 |   0.5041 |                   1.41× |
| Stage 1 + 2               |                          0.6500 |   0.4614 |                   1.91× |
| Stage 1 + 2 + 3           |                          0.4814 |   0.3231 |                   2.79× |
| Stage 1–3 + BitLinear_TTQ |                          0.3901 |   0.2567 |                   2.94x |



## References

1. Zhu et al., “Trained Ternary Quantization,” *arXiv:1612.01064*, 2016.
2. Dettmers et al., “1-bit LLMs: 1.58 Bits is All You Need,” *arXiv:2402.17764*, 2024.
3. Ultralytics YOLO11 documentation — [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)
4. Project code (local): path to repo containing `train.py`, `train_c2psa.py`, `src/quantization/*`, etc.

---

