#!/usr/bin/env python3
"""
TensorRT-Optimized Latency Benchmarking Script
Exports models to TensorRT and measures inference time
Maximizes performance of quantized models on GPU
"""

import argparse
import torch
import numpy as np
import time
from pathlib import Path
import sys
import shutil

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO


def export_to_tensorrt(model_path: str, output_dir: str = 'tensorrt_engines', 
                      imgsz: int = 640, half: bool = True) -> str:
    """Export model to TensorRT engine"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_name = Path(model_path).stem
    engine_path = output_dir / f"{model_name}.engine"
    
    # Skip if already exported
    if engine_path.exists():
        print(f"  ✓ Using cached engine: {engine_path}")
        return str(engine_path)
    
    print(f"  Exporting to TensorRT (this may take 1-2 minutes)...", end=' ', flush=True)
    
    try:
        model = YOLO(model_path)
        
        # Export with FP16 for maximum speedup
        exported_path = model.export(
            format='engine',
            imgsz=imgsz,
            half=half,  # FP16 quantization for 2x speedup
            device=0,
            verbose=False
        )
        
        print("✓")
        return str(exported_path)
    
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Falling back to PyTorch inference")
        return None


def benchmark_tensorrt(baseline_path: str, quantized_path: str,
                      device: str = 'cuda', num_warmup: int = 5, num_runs: int = 50,
                      imgsz: int = 640, batch_size: int = 1, use_tensorrt: bool = True):
    """Benchmark with TensorRT optimization"""
    
    # Device setup
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    device_name = device_obj.type.upper()
    
    print("\n" + "="*80)
    print("TensorRT-Optimized Latency Benchmark")
    print("="*80)
    print(f"Baseline: {baseline_path}")
    print(f"Quantized: {quantized_path}")
    print(f"Device: {device_name}")
    print(f"Input: {batch_size}x3x{imgsz}x{imgsz}")
    print(f"Warmup: {num_warmup}, Runs: {num_runs}")
    print(f"TensorRT: {'ENABLED (FP16)' if use_tensorrt and device == 'cuda' else 'DISABLED'}\n")
    
    # Export to TensorRT if GPU
    baseline_engine_path = None
    quantized_engine_path = None
    
    if use_tensorrt and device == 'cuda':
        print("Exporting models to TensorRT...")
        baseline_engine_path = export_to_tensorrt(baseline_path, imgsz=imgsz, half=True)
        quantized_engine_path = export_to_tensorrt(quantized_path, imgsz=imgsz, half=True)
        print()
    
    # Load models
    print("Loading models...")
    
    # Use TensorRT engines if available, otherwise fall back to PyTorch
    if baseline_engine_path:
        print(f"✓ Baseline: TensorRT engine")
        baseline_model = YOLO(baseline_engine_path).model.to(device_obj).eval()
        baseline_type = "TensorRT FP16"
    else:
        print(f"✓ Baseline: PyTorch FP32")
        baseline_yolo = YOLO(baseline_path)
        baseline_model = baseline_yolo.model.to(device_obj).eval()
        baseline_type = "PyTorch FP32"
    
    if quantized_engine_path:
        print(f"✓ Quantized: TensorRT FP16")
        quantized_model = YOLO(quantized_engine_path).model.to(device_obj).eval()
        quantized_type = "TensorRT FP16"
    else:
        print(f"✓ Quantized: PyTorch TTQ")
        quantized_checkpoint = torch.load(quantized_path, map_location=device_obj, weights_only=False)
        if isinstance(quantized_checkpoint, dict) and 'model' in quantized_checkpoint:
            quantized_model = quantized_checkpoint['model']
        else:
            quantized_model = quantized_checkpoint
        quantized_model = quantized_model.to(device_obj).eval()
        quantized_type = "PyTorch TTQ"
    
    print()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, imgsz, imgsz).to(device_obj)
    
    # Benchmark baseline
    print(f"Benchmarking Baseline ({baseline_type})...")
    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = baseline_model(dummy_input)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        times_baseline = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = baseline_model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times_baseline.append((end - start) * 1000)
    
    mean_baseline = np.mean(times_baseline)
    std_baseline = np.std(times_baseline)
    min_baseline = np.min(times_baseline)
    max_baseline = np.max(times_baseline)
    
    print(f"  Mean: {mean_baseline:.2f}ms ± {std_baseline:.2f}ms")
    print(f"  Range: {min_baseline:.2f}ms - {max_baseline:.2f}ms\n")
    
    # Benchmark quantized
    print(f"Benchmarking Quantized ({quantized_type})...")
    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = quantized_model(dummy_input)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        times_quantized = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = quantized_model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times_quantized.append((end - start) * 1000)
    
    mean_quantized = np.mean(times_quantized)
    std_quantized = np.std(times_quantized)
    min_quantized = np.min(times_quantized)
    max_quantized = np.max(times_quantized)
    
    print(f"  Mean: {mean_quantized:.2f}ms ± {std_quantized:.2f}ms")
    print(f"  Range: {min_quantized:.2f}ms - {max_quantized:.2f}ms\n")
    
    # Calculate metrics
    speedup = mean_baseline / mean_quantized
    latency_saved = mean_baseline - mean_quantized
    throughput_baseline = 1000 / mean_baseline  # images per second
    throughput_quantized = 1000 / mean_quantized
    
    # Print summary
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"\nLatency:")
    print(f"  Baseline:  {mean_baseline:.2f}ms ({throughput_baseline:.1f} img/s)")
    print(f"  Quantized: {mean_quantized:.2f}ms ({throughput_quantized:.1f} img/s)")
    print(f"\nImprovement:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time saved: {latency_saved:.2f}ms per inference")
    print(f"  Throughput gain: {(throughput_quantized - throughput_baseline):.1f} img/s")
    
    if speedup > 1.0:
        print(f"\n✅ Quantized model is {speedup:.2f}x FASTER")
    elif speedup < 1.0:
        print(f"\n⚠️  Quantized model is {1/speedup:.2f}x slower (expected for PyTorch)")
        print(f"   Use TensorRT engine to get actual speedup!")
    else:
        print(f"\n ℹ️  Same speed (within measurement noise)")
    
    print("="*80 + "\n")
    
    return {
        'baseline_mean': mean_baseline,
        'quantized_mean': mean_quantized,
        'speedup': speedup,
        'baseline_type': baseline_type,
        'quantized_type': quantized_type,
    }


def main():
    parser = argparse.ArgumentParser(
        description='TensorRT-Optimized Latency Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GPU with TensorRT (recommended - maximum speedup)
  python latency_benchmark.py \\
    --baseline yolo11n.pt \\
    --quantized ttq_checkpoints/yolo11n/stage1_progressive_final/best.pt \\
    --device gpu

  # GPU without TensorRT (use PyTorch)
  python latency_benchmark.py \\
    --baseline yolo11n.pt \\
    --quantized ttq_checkpoints/yolo11n/stage1_progressive_final/best.pt \\
    --device gpu \\
    --no-tensorrt

  # CPU (TensorRT not supported)
  python latency_benchmark.py \\
    --baseline yolo11n.pt \\
    --quantized checkpoints/best.pt \\
    --device cpu \\
    --num-runs 20

  # Batch inference benchmarking
  python latency_benchmark.py \\
    --baseline yolo11n.pt \\
    --quantized checkpoints/best.pt \\
    --device gpu \\
    --batch-size 4 \\
    --num-runs 100
        """)
    
    parser.add_argument('--baseline', type=str, required=True, help='Baseline model path')
    parser.add_argument('--quantized', type=str, required=True, help='Quantized model path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'gpu'],
                       help='Device (default: cuda)')
    parser.add_argument('--num-runs', type=int, default=50, help='Benchmark runs (default: 50)')
    parser.add_argument('--num-warmup', type=int, default=5, help='Warmup runs (default: 5)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size (default: 640)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--no-tensorrt', action='store_true', help='Disable TensorRT (use PyTorch only)')
    
    args = parser.parse_args()
    
    device = 'cuda' if args.device in ['cuda', 'gpu'] else 'cpu'
    use_tensorrt = not args.no_tensorrt and device == 'cuda'
    
    benchmark_tensorrt(
        args.baseline,
        args.quantized,
        device=device,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        use_tensorrt=use_tensorrt
    )


if __name__ == '__main__':
    main()
