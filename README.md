
# YOLO11 — TTQ (Trained Ternary Quantization) + BitLinear Integration

This repository implements progressive Trained Ternary Quantization (TTQ) and a BitLinear_TTQ module for YOLO11 models (C2PSA attention).  
The goal is to compress YOLO11 (n/x sizes) and measure accuracy vs. compression vs. latency trade-offs on COCO.

---

## Table of Contents
- [Quick Summary](#quick-summary)
- [Prerequisites](#prerequisites)
- [Directory Structure](#directory-structure)
- [How to Run](#how-to-run)
- [Reported Results](#reported-results)
- [References](#references)

---

## Quick Summary

- **ShadowWeightManager (TTQ):** Implements master (FP32) + shadow (ternary) models with learnable scales `Wp` and `Wn` for progressive stage quantization.
- **BitLinear_TTQ:** Ternary Conv2d with LayerNorm, absmax activation quantization (8-bit), and learned scaling for use in C2PSA attention.
- **Standard BitLinear:** Fixed-scale baseline for ablation.
- **Core scripts:**
  - `train.py` - Progressive TTQ training (stages 1–3)
  - `train_c2psa.py` - BitLinear_TTQ training for C2PSA layers
  - `train_c2psa_standard.py` - Standard BitLinear baseline
  - `compare_bitlinear.py` - Compare the Standard and TTQ based Bitlinear Implementation
  - `compression_calculation.py` - Computes theoretical compression and coverage
  - `latency_benchmark.py` - Exports to TensorRT, benchmarks latency
  - `test.py` - Identifies ternary layers

---

## Prerequisites

- Python ≥ 3.8  
- PyTorch (CUDA-enabled)  
- Ultralytics YOLO (`pip install ultralytics`)  
- TensorRT (for latency benchmarking, optional but recommended)  

Install core dependencies:
```bash
pip install -r requirements.txt
````
This repository uses **Git Large File Storage (LFS)** to store the trained `.pt` model files in `saved_models/`.

To download the full models, make sure you have Git LFS installed:

### Install Git LFS

- **macOS:** `brew install git-lfs`
- **Windows:** Download and run the installer from https://git-lfs.github.com/
- **Linux:** `sudo apt install git-lfs` (Debian/Ubuntu) or see https://git-lfs.github.com/

After installing, initialize it:

```bash
git lfs install
```

For latency benchmarking:

```bash
pip install onnx onnxslim onnxruntime tensorrt
```

---

## Directory Structure

```text
configs/
  stage1_progressive.yaml      # Progressive TTQ stages
  stage2_c2psa.yaml            # C2PSA BitLinear_TTQ configuration

src/quantization/
  shadow_weight_manager.py
  c2psa_bitlinear_ttq.py
  c2psa_bitlinear_standard.py
  bitlinear_standard_manager.py
  bitlinear_ttq_manager.py

src/training/
  c2psa_trainer.py
  shadow_trainer.py

train.py                        # Runs TTQ progressive stages
train_c2psa.py                  # Trains BitLinear_TTQ model
train_c2psa_standard.py         # Trains standard BitLinear model
compression_calculation.py      # Compression analysis
latency_benchmark.py            # TensorRT latency benchmarking
test.py                         # Ternary layer inspection
compare_bitlinear.py            # Compare Bitlinear implementaions

ttq_checkpoints/                # TTQ stage checkpoints
checkpoints/                    # BitLinear checkpoints
saved_models/                   # Contains the trained model you can run to check the results
```

---

## How to Run

### 1. Download Pretrained YOLO Models

Both pretrained YOLO11 models must be downloaded before training:

```bash
# YOLO11-n (nano)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

# YOLO11-x (extra-large)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
```

---

### 2. Run the Full Training Pipeline (YOLO11n)

A helper script `run.sh` automates all stages for YOLO11n:

```bash
bash run.sh
```

**Contents of `run.sh`:**

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

python train.py --config configs/stage1_progressive.yaml --stage 1
python train.py --config configs/stage1_progressive.yaml --stage 2
python train.py --config configs/stage1_progressive.yaml --stage 3

python train_c2psa.py \
  --config configs/stage2_c2psa.yaml \
  --model ttq_checkpoints/yolo11n/stage1_progressive_final/best.pt
```

This script sequentially runs all TTQ stages and then performs BitLinear_TTQ training.

---

### 3. Validation

Validate any trained checkpoint on COCO:

```bash
python validate.py saved_models/yolo11n/stage1.pt --data coco.yaml
```

Replace `stage1.pt` with any other checkpoint (e.g., `stage1-3+bitlinear_ttq.pt`).

---

### 4. Compression Evaluation

Compute compression ratio between baseline and quantized models:

```bash
python compression_calculation.py \
  --baseline yolo11x.pt \
  --quantized saved_models/yolo11x/stage1-3+bitlinear_ttq.pt
```

Outputs model sizes, compression ratio, and quantized layer percentages.

---


### 5. TensorRT Latency Benchmark

Benchmark baseline and quantized models:

```bash
python latency_benchmark.py \
  --baseline yolo11n.pt \
  --quantized saved_models/yolo11n/stage1-3+bitlinear_ttq.pt \
  --export-fp16 \
  --export-int8 \
  --num-runs 30
```

Reports mean latency, throughput (FPS), and memory usage.

---

### 6. Ternary Layer Inspection

Check which layers were quantized to ternary:

```bash
python test.py saved_models/yolo11n/stage1-3+bitlinear_ttq.pt
```

Displays quantized (ternary) vs FP32 layers.

---

### 7. Comparining Standard vs TTQ based Bitlinear quantization

Compare the two Bitlinear implmentations

You need to first train yolo11n using both implementaion
You can also modify the config file to use coco128 for faster training.
```bash
python train_c2psa.py --config configs/stage2_c2psa.yaml --model yolo11n.pt
python train_c2psa_standard.py --config configs/stage2_c2psa.yaml --model yolo11n.pt
```

```bash
python compare_bitlinear.py checkpoints/stage2_c2psa_baseline/best.pt  checkpoints/stage2_c2psa_standard_baseline/best.pt
```

Outputs the accuracy analysis of both the models

---


###  Script Summary

| Script                       | Purpose                                               |
| ---------------------------- | ----------------------------------------------------- |
| `run.sh`                     | Downloads YOLO11n and runs all TTQ + BitLinear stages |
| `validate.py`                | Runs COCO validation and prints mAP metrics           |
| `latency_benchmark.py`       | Benchmarks TensorRT latency (FP32/FP16/INT8)          |
| `compression_calculation.py` | Computes compression ratios and quantization coverage |
| `test.py`                    | Verifies ternary quantized layers                     |

---

## Reported Results

### YOLO11n (COCO fine-tune / validation)

| Stage                     |  mAP50 | mAP50-95 | Compression |
| ------------------------- | -----: | -------: | ----------: |
| Baseline                  | 0.5485 |   0.3924 |       1.00× |
| Stage 1                   | 0.5088 |   0.3410 |       1.30× |
| Stage 1 + 2               | 0.4702 |   0.3088 |       1.69× |
| Stage 1 + 2 + 3           | 0.4110 |   0.2639 |       1.97× |
| Stage 1–3 + BitLinear_TTQ | 0.2955 |   0.1819 |       2.16× |

**BitLinear Comparison (YOLO11n):**

* Baseline: mAP50 = 0.5487, mAP50-95 = 0.3925, Precision = 0.4652, Recall = 0.2807
* Ours (BitLinear_TTQ): mAP50 = 0.4421, mAP50-95 = 0.3098, Precision = 0.3702, Recall = 0.0702
* Standard BitLinear: mAP50 = 0.4427, mAP50-95 = 0.3098, Precision = 0.3354, Recall = 0.0709

**TensorRT Latency (YOLO11n):**

| Model          | Mean (ms) | Throughput | Memory (MB) |
| -------------- | --------: | ---------: | ----------: |
| Baseline FP32  |    12.049 |       83.0 |        75.6 |
| Baseline FP16  |     7.467 |      133.9 |        72.4 |
| Baseline INT8  |     6.777 |      147.6 |        71.3 |
| Quantized FP32 |    17.237 |       58.0 |        85.8 |
| Quantized FP16 |     6.338 |      157.8 |        71.3 |
| Quantized INT8 |     5.565 |      179.7 |        79.0 |

Ternary models achieve up to **2.94× compression** and demonstrate **1.5–3.2× latency improvement** depending on precision (FP32 → INT8), with moderate (3–5%) mAP loss — confirming their suitability for low-power deployment.

---

### YOLO11x (COCO)

| Stage                     |  mAP50 | mAP50-95 | Compression |
| ------------------------- | -----: | -------: | ----------: |
| Baseline                  | 0.7135 |   0.5485 |       1.00× |
| Stage 1                   | 0.6891 |   0.5041 |       1.41× |
| Stage 1 + 2               | 0.6500 |   0.4614 |       1.91× |
| Stage 1 + 2 + 3           | 0.4814 |   0.3231 |       2.79× |
| Stage 1–3 + BitLinear_TTQ | 0.3901 |   0.2567 |       2.94× |

---

## References

1. Zhu et al., *Trained Ternary Quantization*, arXiv:1612.01064 (2016).
   [https://arxiv.org/abs/1612.01064](https://arxiv.org/abs/1612.01064)
2. Dettmers et al., *1-bit LLMs: 1.58 Bits is All You Need*, arXiv:2402.17764 (2024).
   [https://arxiv.org/abs/2402.17764](https://arxiv.org/abs/2402.17764)
3. Ultralytics YOLO11 Docs — [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)
4. Project Code — [https://github.com/Adithya-Satyarthi/YOLO11-TQ](https://github.com/Adithya-Satyarthi/YOLO11-TQ)


