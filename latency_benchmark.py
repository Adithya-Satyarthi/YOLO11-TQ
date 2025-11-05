#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO


class ComprehensiveBenchmark:
    def __init__(self, device: str = 'cuda', num_warmup: int = 10, num_runs: int = 50):
        self.device = device
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Compute Capability: {torch.cuda.get_device_capability(0)}\n")
    
    def strip_quantized_checkpoint(self, checkpoint: dict) -> dict:
        """Strip shadow parameters from quantized checkpoint"""
        state_dict = checkpoint.get('state_dict') or checkpoint
        
        keys_to_remove = []
        for key in list(state_dict.keys()):
            if any(x in key for x in ['ap', 'an', 'num_batches_tracked', 'running_mean', 'running_var']):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del state_dict[key]
        
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = state_dict
        else:
            checkpoint = state_dict
        
        return checkpoint, len(keys_to_remove)
    
    def load_model(self, model_path: str, strip: bool = False) -> tuple:
        """Load model with optional stripping, ensure GPU placement"""
        model = YOLO(model_path)
        stripped_count = 0
        
        if strip:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            checkpoint, stripped_count = self.strip_quantized_checkpoint(checkpoint)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif isinstance(checkpoint, dict):
                model.model.load_state_dict(checkpoint, strict=False)
        
        # Ensure model is on GPU and in eval mode
        model.model = model.model.to(self.device).eval()
        
        # Force all parameters to GPU
        for param in model.model.parameters():
            param.data = param.data.to(self.device)
        
        return model, stripped_count
    
    def export_to_tensorrt(self, model_path: str, imgsz: int = 640, 
                          batch_size: int = 1, output_dir: str = 'tensorrt_engines',
                          precision: str = 'fp16') -> str:
        """Export model to TensorRT"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        model_name = Path(model_path).stem
        engine_path = output_dir / f"{model_name}_{precision}.engine"
        
        # Check if engine already exists
        if engine_path.exists():
            print(f"   Using cached engine")
            return str(engine_path)
        
        try:
            model = YOLO(model_path)
            
            export_params = {
                'format': 'engine',
                'imgsz': imgsz,
                'device': 0,
                'verbose': False,
                'batch': batch_size,
            }
            
            if precision == 'fp16':
                export_params['half'] = True
            elif precision == 'int8':
                export_params['half'] = False
                export_params['int8'] = True
                export_params['data'] = 'coco128.yaml'  # Use coco128.yaml for calibration can be changed to coco but will be slower
            
            print(f"   Exporting to TensorRT {precision.upper()}...")
            model.export(**export_params)
            
            # Find and move engine
            exported_engine = Path(model_path).parent / f"{model_name}.engine"
            if not exported_engine.exists():
                cwd_engine = Path.cwd() / f"{model_name}.engine"
                if cwd_engine.exists():
                    exported_engine = cwd_engine
            
            if exported_engine.exists() and exported_engine != engine_path:
                import shutil
                shutil.move(str(exported_engine), str(engine_path))
            
            return str(engine_path)
        
        except Exception as e:
            print(f"   Export failed: {e}")
            return None
    
    def benchmark_model(self, model_path: str, model_name: str, 
                       imgsz: int = 640, batch_size: int = 1,
                       strip: bool = False) -> dict:
        """Benchmark a single model with GPU-only inference"""
        is_tensorrt = str(model_path).endswith('.engine')
        
        # Load model
        if is_tensorrt:
            model = YOLO(model_path)
        else:
            model, stripped_count = self.load_model(model_path, strip=strip)
        
        # Create dummy input on GPU
        dummy_input = torch.randn(batch_size, 3, imgsz, imgsz, 
                                 device=self.device, dtype=torch.float32)
        
        times = []
        
        with torch.no_grad():
            # Warmup runs
            for _ in range(self.num_warmup):
                if is_tensorrt:
                    _ = model(dummy_input, verbose=False)
                else:
                    _ = model.model(dummy_input)
                torch.cuda.synchronize()
            
            # Clear cache and reset stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Benchmark runs
            for _ in range(self.num_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                if is_tensorrt:
                    _ = model(dummy_input, verbose=False)
                else:
                    _ = model.model(dummy_input)
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
        
        result = {
            'model': model_name,
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'p95': np.percentile(times, 95),
            'throughput': 1000 / np.mean(times),
            'memory': peak_memory,
        }
        
        if not is_tensorrt and strip and stripped_count > 0:
            result['stripped'] = stripped_count
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Latency Benchmark (FP32, FP16, INT8)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple benchmark with auto FP16 and INT8 export
  python latency_benchmark.py --baseline yolo11n.pt --quantized saved_models/yolo11n/stage1+2+3.pt
        """)
    
    parser.add_argument('--baseline', type=str, required=True, help='Baseline model path')
    parser.add_argument('--quantized', type=str, required=True, help='Quantized model path (auto-stripped)')
    parser.add_argument('--num-runs', type=int, default=50, help='Number of benchmark runs')
    parser.add_argument('--num-warmup', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--output-dir', type=str, default='tensorrt_engines', help='TensorRT engine output directory')
    
    args = parser.parse_args()
    
    try:
        bench = ComprehensiveBenchmark(
            device='cuda',
            num_warmup=args.num_warmup,
            num_runs=args.num_runs
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("="*110)
    print("COMPREHENSIVE LATENCY BENCHMARK: FP32, FP16, INT8")
    print("="*110)
    print(f"Baseline:   {args.baseline}")
    print(f"Quantized:  {args.quantized} (auto-stripped)")
    print(f"Dataset:    COCO (for INT8 calibration)")
    print(f"Runs:       {args.num_runs} (warmup: {args.num_warmup})")
    print(f"Image size: {args.imgsz}x{args.imgsz}\n")
    
    results = {}
    
    # ==================== BASELINE MODEL ====================
    print(" BASELINE Model:")
    
    # 1. PyTorch FP32
    print(f"  1/3 PyTorch FP32...", end=' ', flush=True)
    results['Baseline FP32'] = bench.benchmark_model(
        args.baseline, "Baseline FP32",
        imgsz=args.imgsz, batch_size=args.batch_size, strip=False
    )
    print(f" {results['Baseline FP32']['mean']:.3f} ms")
    
    # 2. TensorRT FP16
    print(f"  2/3 TensorRT FP16...", end=' ', flush=True)
    engine_baseline_fp16 = bench.export_to_tensorrt(
        args.baseline, imgsz=args.imgsz, batch_size=args.batch_size,
        output_dir=args.output_dir, precision='fp16'
    )
    if engine_baseline_fp16:
        results['Baseline FP16'] = bench.benchmark_model(
            engine_baseline_fp16, "Baseline FP16",
            imgsz=args.imgsz, batch_size=args.batch_size, strip=False
        )
        print(f" {results['Baseline FP16']['mean']:.3f} ms")
    
    # 3. TensorRT INT8
    print(f"  3/3 TensorRT INT8...", end=' ', flush=True)
    engine_baseline_int8 = bench.export_to_tensorrt(
        args.baseline, imgsz=args.imgsz, batch_size=args.batch_size,
        output_dir=args.output_dir, precision='int8'
    )
    if engine_baseline_int8:
        results['Baseline INT8'] = bench.benchmark_model(
            engine_baseline_int8, "Baseline INT8",
            imgsz=args.imgsz, batch_size=args.batch_size, strip=False
        )
        print(f" {results['Baseline INT8']['mean']:.3f} ms")
    
    # ==================== QUANTIZED MODEL ====================
    print("\n QUANTIZED Model (Auto-Stripped):")
    
    # 1. PyTorch FP32
    print(f"  1/3 PyTorch FP32...", end=' ', flush=True)
    results['Quantized FP32'] = bench.benchmark_model(
        args.quantized, "Quantized FP32",
        imgsz=args.imgsz, batch_size=args.batch_size, strip=True
    )
    stripped_info = f" [Stripped {results['Quantized FP32'].get('stripped', 0)} params]" if 'stripped' in results['Quantized FP32'] else ""
    print(f" {results['Quantized FP32']['mean']:.3f} ms{stripped_info}")
    
    # 2. TensorRT FP16
    print(f"  2/3 TensorRT FP16...", end=' ', flush=True)
    engine_quantized_fp16 = bench.export_to_tensorrt(
        args.quantized, imgsz=args.imgsz, batch_size=args.batch_size,
        output_dir=args.output_dir, precision='fp16'
    )
    if engine_quantized_fp16:
        results['Quantized FP16'] = bench.benchmark_model(
            engine_quantized_fp16, "Quantized FP16",
            imgsz=args.imgsz, batch_size=args.batch_size, strip=False
        )
        print(f" {results['Quantized FP16']['mean']:.3f} ms")
    
    # 3. TensorRT INT8
    print(f"  3/3 TensorRT INT8...", end=' ', flush=True)
    engine_quantized_int8 = bench.export_to_tensorrt(
        args.quantized, imgsz=args.imgsz, batch_size=args.batch_size,
        output_dir=args.output_dir, precision='int8'
    )
    if engine_quantized_int8:
        results['Quantized INT8'] = bench.benchmark_model(
            engine_quantized_int8, "Quantized INT8",
            imgsz=args.imgsz, batch_size=args.batch_size, strip=False
        )
        print(f" {results['Quantized INT8']['mean']:.3f} ms")
    
    # ==================== RESULTS TABLE ====================
    print("\n" + "="*110)
    print("LATENCY RESULTS")
    print("="*110)
    
    print(f"\n{'Model':<25} {'Mean (ms)':<12} {'Throughput (FPS)':<18} {'Memory (MB)':<15}")
    print("-" * 110)
    
    for name, result in results.items():
        print(f"{name:<23} {result['mean']:<12.3f} {result['throughput']:<18.1f} {result['memory']:<15.1f}")
    
    # ==================== SPEEDUP ANALYSIS ====================
    print("\n" + "="*110)
    print("SPEEDUP vs BASELINE FP32")
    print("="*110)
    
    baseline_fp32_latency = results['Baseline FP32']['mean']
    
    for name, result in results.items():
        if name == 'Baseline FP32':
            print(f"  {name:<28} 1.00x (reference)")
        else:
            speedup = baseline_fp32_latency / result['mean']
            print(f"{name:<26} {speedup:>6.2f}x")
    
    # ==================== PRECISION COMPARISON ====================
    print("\n" + "="*110)
    print("QUANTIZED vs BASELINE (Same Precision)")
    print("="*110)
    
    comparisons = [
        ('PyTorch FP32', 'Baseline FP32', 'Quantized FP32'),
        ('TensorRT FP16', 'Baseline FP16', 'Quantized FP16'),
        ('TensorRT INT8', 'Baseline INT8', 'Quantized INT8'),
    ]
    
    for precision_name, baseline_key, quantized_key in comparisons:
        if baseline_key in results and quantized_key in results:
            speedup = results[baseline_key]['mean'] / results[quantized_key]['mean']
            print(f"{precision_name:<20} {speedup:.2f}x")
    
    print("="*110 + "\n")
    
    # ==================== SAVE RESULTS ====================
    output_file = Path('comprehensive_results.txt')
    with open(output_file, 'w') as f:
        f.write("COMPREHENSIVE LATENCY BENCHMARK RESULTS\n")
        f.write("="*110 + "\n\n")
        f.write(f"Baseline:  {args.baseline}\n")
        f.write(f"Quantized: {args.quantized}\n")
        f.write(f"Runs:      {args.num_runs} (warmup: {args.num_warmup})\n\n")
        
        f.write(f"{'Model':<30} {'Mean (ms)':<12} {'Throughput (FPS)':<18}\n")
        f.write("-" * 80 + "\n")
        
        for name, result in results.items():
            f.write(f"{name:<30} {result['mean']:<12.3f} {result['throughput']:<18.1f}\n")
    
    print(f" Results saved to {output_file}")


if __name__ == '__main__':
    main()
