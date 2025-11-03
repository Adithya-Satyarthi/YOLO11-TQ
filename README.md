# YOLO11 — TTQ (Trained Ternary Quantization) + BitLinear Integration

**Author:** Adithya Satyarthi (EP22B049), IIT Madras  
**Repo:** `YOLO11-TQ` (local project files provided)

This repository implements progressive Trained Ternary Quantization (TTQ) and a BitLinear_TTQ module for YOLO11 models (C2PSA attention). The goal is to compress YOLO11 (n/x sizes) and measure accuracy vs. compression vs. latency trade-offs on COCO.

---

## Table of Contents
- [Quick summary of what’s implemented](#quick-summary-of-whats-implemented)  
- [Prerequisites](#prerequisites)  
- [Directory / important files](#directory--important-files)  
- [How to run — full pipeline (both `yolo11n` and `yolo11x`)](#how-to-run---full-pipeline-both-yolo11n-and-yolo11x)  
- [Evaluation & benchmarking (validation, compression, latency)](#evaluation--benchmarking-validation-compression-latency)  
- [Reported results (from your logs/configs)](#reported-results-from-your-logsconfigs)  
- [Notes & troubleshooting](#notes--troubleshooting)  
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
  - `analyze_compression.py` — compute theoretical compression & coverage.
  - `compare_bitlinear.py` — compare BitLinear_TTQ vs Standard BitLinear.
  - `latency_benchmark.py` — export to TensorRT and measure latency.

---

## Prerequisites

- Python 3.8+ (recommended)
- PyTorch (CUDA-enabled) matching your GPU
- Ultralytics YOLO (the `ultralytics` package)
- TensorRT (for latency benchmarking) — optional but recommended
- `pip install -r requirements.txt` (create one containing: `torch`, `ultralytics`, `pyyaml`, `numpy`, `tqdm`, `wandb` (optional) etc.)
- GPU with >=8 GB (preferable) for training; if memory issues occur, worker count is forced to `0` in trainers.

---

## Directory / important files

```

configs/
stage1_progressive.yaml       # progressive TTQ stages
stage2_c2psa.yaml             # C2PSA BitLinear_TTQ config
stage3_progressive.yaml      # (if present) final stage config

src/quantization/
shadow_weight_manager.py
c2psa_bitlinear_ttq.py
bitlinear_standard_manager.py

src/training/
c2psa_trainer.py
(other trainer utilities)

train.py                        # run progressive TTQ stages
train_c2psa.py                  # run C2PSA BitLinear_TTQ training
train_c2psa_standard.py         # run standard BitLinear training
analyze_compression.py
compare_bitlinear.py
latency_benchmark.py
ttq_checkpoints/                 # checkpoint outputs (created by training)

````

---

## How to run — full pipeline (both `yolo11n` and `yolo11x`)

> **Important:** `configs/stage1_progressive.yaml` contains `model_size: 'x'` by default. Change this field to `'n'` or `'x'` before running each variant, or create two copies (e.g., `stage1_progressive_n.yaml` and `stage1_progressive_x.yaml`) with `model_size` set accordingly.

Below are the exact commands and order to reproduce the pipeline and results:

### A. Prepare configs for `yolo11n` and `yolo11x`

Option 1 — edit in-place:
```bash
# For YOLO11-n (nano)
sed -i "s/model_size: 'x'/model_size: 'n'/" configs/stage1_progressive.yaml

# For YOLO11-x (later), change back to 'x' before running
# sed -i "s/model_size: 'n'/model_size: 'x'/" configs/stage1_progressive.yaml
````

Option 2 — duplicate file:

```bash
cp configs/stage1_progressive.yaml configs/stage1_progressive_n.yaml
sed -i "s/model_size: 'x'/model_size: 'n'/" configs/stage1_progressive_n.yaml

cp configs/stage1_progressive.yaml configs/stage1_progressive_x.yaml
sed -i "s/model_size: 'x'/model_size: 'x'/" configs/stage1_progressive_x.yaml
```

### B. Progressive TTQ stages (run for each model size separately)

Run Stage 1, 2, 3 in order. Example for `yolo11n` (using `stage1_progressive_n.yaml`):

```bash
# Stage 1: shallow
python train.py --config configs/stage1_progressive_n.yaml --stage 1

# Stage 2: middle (will load stage1 checkpoint automatically)
python train.py --config configs/stage1_progressive_n.yaml --stage 2

# Stage 3: deep (will load stage2 checkpoint automatically)
python train.py --config configs/stage1_progressive_n.yaml --stage 3
```

Repeat the above for `yolo11x` (use `_x.yaml` or edit config back to `'x'`).

**Checkpoints saved here (example):**

```
ttq_checkpoints/yolo11n/stage1_progressive/best.pt
ttq_checkpoints/yolo11n/stage1_progressive_middle/best.pt
ttq_checkpoints/yolo11n/stage1_progressive_final/best.pt
```

### C. BitLinear_TTQ for C2PSA (after TTQ stages)

After progressive TTQ finishes, run C2PSA BitLinear_TTQ training:

```bash
# YOLO11n
python train_c2psa.py --config configs/stage2_c2psa.yaml --model yolo11n.pt

# YOLO11x
python train_c2psa.py --config configs/stage2_c2psa.yaml --model yolo11x.pt
```

If you want the **Standard BitLinear** (fixed scales) baseline:

```bash
python train_c2psa_standard.py --config configs/stage2_c2psa.yaml --model yolo11n.pt
```

### D. Validation, compression analysis and comparison

Validate quantized models and compute compression:

```bash
# Example: analyze TTQ quantized model (adjust paths to your checkpoints)
python analyze_compression.py \
  --baseline yolo11n.pt \
  --quantized ttq_checkpoints/yolo11n/stage1_progressive_final/best.pt
```

Compare BitLinear implementations:

```bash
python compare_bitlinear.py \
  checkpoints/stage2_c2psa/best.pt \
  checkpoints/stage2_c2psa_standard/best.pt \
  yolo11n.pt
```

### E. TensorRT latency benchmark

```bash
python latency_benchmark.py \
  --baseline baseline_yolo11n.pt \
  --quantized path/to/quantized_model.pt
```

The script will attempt to export to TensorRT (FP16) and run multiple runs to report mean latency.

---

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

**BitLinear comparison (YOLO11n; COCO128 validation):**

* Baseline: mAP50 0.5487, mAP50-95 0.3925, Precision 0.4652, Recall 0.2807
* Our Impl (BitLinear_TTQ): mAP50 0.4421, mAP50-95 0.3098, Precision 0.3702, Recall 0.0702
* Standard BitLinear: mAP50 0.4427, mAP50-95 0.3098, Precision 0.3354, Recall 0.0709

**TensorRT Latency (YOLO11n):**

* Baseline FP32: 13.2 ms → Quantized: 9.1 ms
* BitLinear_TTQ Quantized (final): 7.8 ms

### YOLO11x (COCO)

| Stage                     |                           mAP50 | mAP50-95 | Theoretical compression |
| ------------------------- | ------------------------------: | -------: | ----------------------: |
| Baseline                  |                          0.7135 |   0.5485 |                   1.00× |
| Stage 1                   |                          0.6891 |   0.5041 |                   1.41× |
| Stage 1 + 2               |                          0.6500 |   0.4614 |                   1.91× |
| Stage 1 + 2 + 3           |                          0.4814 |   0.3231 |                   2.79× |
| Stage 1–3 + BitLinear_TTQ | training not finished / pending |          |                         |

> Notes: YOLO11x final TTQ+BitLinear stage was not finished in the files you provided.

---

## Notes & troubleshooting

* **Set workers to 0** if you hit memory issues — code already forces `workers=0` for some trainers (to prevent DRAM explosion). If you see DataLoader OOM or multiprocessing failures, ensure `workers=0`.
* **Model weight paths:** `train.py` checks for previous stage checkpoints. If a stage fails to find the earlier stage's `best.pt`, it will raise an error. Run stages in order.
* **WandB:** optional — set `use_wandb` in config or leave false.
* **GPU memory:** for `yolo11x` expect substantially higher memory use; run on a GPU with sufficient memory.
* **TensorRT export:** may fail if TensorRT not installed or incompatible CUDA / PyTorch — `latency_benchmark.py` falls back to PyTorch inference if export fails.
* **Reproducibility:** Use same `configs/*` settings (batch, imgsz, lr) across runs for fair comparisons.

---

## References

1. Zhu et al., “Trained Ternary Quantization,” *arXiv:1612.01064*, 2016.
2. Dettmers et al., “1-bit LLMs: 1.58 Bits is All You Need,” *arXiv:2402.17764*, 2024.
3. Ultralytics YOLO11 documentation — [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/)
4. Project code (local): path to repo containing `train.py`, `train_c2psa.py`, `src/quantization/*`, etc.

---

## Example quick-start (YOLO11n)

```bash
# 1) Ensure configs/stage1_progressive.yaml has model_size: 'n'
sed -i "s/model_size: 'x'/model_size: 'n'/" configs/stage1_progressive.yaml

# 2) Stage 1 → Stage 2 → Stage 3
python train.py --config configs/stage1_progressive.yaml --stage 1
python train.py --config configs/stage1_progressive.yaml --stage 2
python train.py --config configs/stage1_progressive.yaml --stage 3

# 3) Train C2PSA BitLinear_TTQ
python train_c2psa.py --config configs/stage2_c2psa.yaml --model yolo11n.pt

# 4) Analyze compression & compare
python analyze_compression.py --baseline yolo11n.pt --quantized ttq_checkpoints/yolo11n/stage1_progressive_final/best.pt
python compare_bitlinear.py checkpoints/stage2_c2psa/best.pt checkpoints/stage2_c2psa_standard/best.pt yolo11n.pt

# 5) Latency benchmark
python latency_benchmark.py
```

