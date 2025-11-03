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
    def __init__(self, device: str = 'cuda', num_warmup: int = 5, num_runs: int = 30):
        self.device = device
        self.num_warmup = num_warmup
        self.num_runs = num_runs
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Compute Capability: {torch.cuda.get_device_capability(0)}\n")
    
    def strip_quantized_checkpoint(self, checkpoint: dict) -> dict:
        """
        Strip shadow parameters from quantized checkpoint on-the-fly
        Removes: ap, an, num_batches_tracked, running_mean, running_var
        """
        
        state_dict = checkpoint.get('state_dict') or checkpoint
        
        # Count removals
        keys_to_remove = []
        
        for key in list(state_dict.keys()):
            if 'ap' in key or 'an' in key or 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
                keys_to_remove.append(key)
        
        # Remove
        for key in keys_to_remove:
            del state_dict[key]
        
        # Update checkpoint
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = state_dict
        else:
            checkpoint = state_dict
        
        return checkpoint, len(keys_to_remove)
    
    def load_model(self, model_path: str, strip: bool = False) -> tuple:
        """
        Load model with optional on-the-fly stripping
        Returns: (model, stripped_params_count)
        """
        
        model = YOLO(model_path)
        stripped_count = 0
        
        if strip:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Strip shadow parameters
            checkpoint, stripped_count = self.strip_quantized_checkpoint(checkpoint)
            
            # Reload stripped state dict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif isinstance(checkpoint, dict):
                model.model.load_state_dict(checkpoint, strict=False)
        
        return model, stripped_count
    
    def export_to_tensorrt(self, model_path: str, imgsz: int = 640, 
                          batch_size: int = 1, output_dir: str = 'tensorrt_engines',
                          precision: str = 'fp16') -> str:
        """Export model to TensorRT"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        model_name = Path(model_path).stem
        engine_path = output_dir / f"{model_name}_{precision}.engine"
        
        if engine_path.exists():
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
                export_params['data'] = 'coco128.yaml'
            
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
            print(f"       Export failed: {e}")
            return None
    
    def benchmark_model(self, model_path: str, model_name: str, 
                       imgsz: int = 640, batch_size: int = 1,
                       strip: bool = False) -> dict:
        """Benchmark a single model"""
        
        is_tensorrt = str(model_path).endswith('.engine')
        
        # Load model with optional stripping
        model, stripped_count = self.load_model(model_path, strip=strip)
        
        if not is_tensorrt:
            model.model = model.model.to(self.device).eval()
        
        dummy_input = torch.randn(batch_size, 3, imgsz, imgsz, 
                                 device=self.device, dtype=torch.float32)
        
        times = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(self.num_warmup):
                if is_tensorrt:
                    _ = model(dummy_input, verbose=False)
                else:
                    _ = model.model(dummy_input)
                torch.cuda.synchronize()
            
            # Measure
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            for _ in range(self.num_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                if is_tensorrt:
                    _ = model(dummy_input, verbose=False)
                else:
                    _ = model.model(dummy_input)
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        times = np.array(times)
        peak_memory = torch.cuda.max_memory_allocated() / 1e6
        
        result = {
            'model': model_name,
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'p95': np.percentile(times, 95),
            'throughput': 1000 / np.mean(times),
            'memory': peak_memory,
        }
        
        if strip and stripped_count > 0:
            result['stripped'] = stripped_count
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Latency Benchmark with On-the-Fly Stripping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare baseline vs quantized (quantized will be auto-stripped)
  python latency_benchmark.py \\
    --baseline yolo11n.pt \\
    --quantized saved_models/yolo11n/stage1-3+bilinear_ttq.pt \\
    --export-fp16 \\
    --export-int8 \\
    --num-runs 30
        """)
    
    parser.add_argument('--baseline', type=str, required=True, help='Baseline model')
    parser.add_argument('--quantized', type=str, required=True, help='Quantized model (will auto-strip)')
    parser.add_argument('--export-fp16', action='store_true', help='Export to FP16')
    parser.add_argument('--export-int8', action='store_true', help='Export to INT8')
    parser.add_argument('--num-runs', type=int, default=30)
    parser.add_argument('--num-warmup', type=int, default=5)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='tensorrt_engines')
    
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
    print("COMPREHENSIVE LATENCY BENCHMARK: Baseline vs Quantized (Auto-Stripped)")
    print("="*110)
    print(f"Baseline:   {args.baseline}")
    print(f"Quantized:  {args.quantized} (auto-stripped)")
    print(f"Runs: {args.num_runs} (warmup: {args.num_warmup})\n")
    
    results = {}
    
    # Benchmark Baseline PyTorch FP32
    print(" BASELINE Model:")
    print(f"  PyTorch FP32...", end=' ', flush=True)
    results['Baseline FP32'] = bench.benchmark_model(
        args.baseline,
        "Baseline FP32",
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        strip=False
    )
    print(f"  {results['Baseline FP32']['mean']:.3f}ms")
    
    # Benchmark Baseline TensorRT FP16
    if args.export_fp16:
        print(f"  TensorRT FP16...", end=' ', flush=True)
        engine_baseline_fp16 = bench.export_to_tensorrt(
            args.baseline,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            precision='fp16'
        )
        if engine_baseline_fp16:
            results['Baseline FP16'] = bench.benchmark_model(
                engine_baseline_fp16,
                "Baseline FP16",
                imgsz=args.imgsz,
                batch_size=args.batch_size,
                strip=False
            )
            print(f"  {results['Baseline FP16']['mean']:.3f}ms")
    
    # Benchmark Baseline TensorRT INT8
    if args.export_int8:
        print(f"  TensorRT INT8...", end=' ', flush=True)
        engine_baseline_int8 = bench.export_to_tensorrt(
            args.baseline,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            precision='int8'
        )
        if engine_baseline_int8:
            results['Baseline INT8'] = bench.benchmark_model(
                engine_baseline_int8,
                "Baseline INT8",
                imgsz=args.imgsz,
                batch_size=args.batch_size,
                strip=False
            )
            print(f"  {results['Baseline INT8']['mean']:.3f}ms")
    
    # Benchmark Quantized PyTorch FP32 (auto-stripped)
    print("\n QUANTIZED Model (Auto-Stripped):")
    print(f"  PyTorch FP32...", end=' ', flush=True)
    results['Quantized FP32'] = bench.benchmark_model(
        args.quantized,
        "Quantized FP32",
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        strip=True
    )
    stripped_info = f" [Stripped {results['Quantized FP32'].get('stripped', 0)} params]" if 'stripped' in results['Quantized FP32'] else ""
    print(f"  {results['Quantized FP32']['mean']:.3f}ms{stripped_info}")
    
    # Benchmark Quantized TensorRT FP16
    if args.export_fp16:
        print(f"  TensorRT FP16...", end=' ', flush=True)
        engine_quantized_fp16 = bench.export_to_tensorrt(
            args.quantized,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            precision='fp16'
        )
        if engine_quantized_fp16:
            results['Quantized FP16'] = bench.benchmark_model(
                engine_quantized_fp16,
                "Quantized FP16",
                imgsz=args.imgsz,
                batch_size=args.batch_size,
                strip=False
            )
            print(f"  {results['Quantized FP16']['mean']:.3f}ms")
    
    # Benchmark Quantized TensorRT INT8
    if args.export_int8:
        print(f"  TensorRT INT8...", end=' ', flush=True)
        engine_quantized_int8 = bench.export_to_tensorrt(
            args.quantized,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            precision='int8'
        )
        if engine_quantized_int8:
            results['Quantized INT8'] = bench.benchmark_model(
                engine_quantized_int8,
                "Quantized INT8",
                imgsz=args.imgsz,
                batch_size=args.batch_size,
                strip=False
            )
            print(f"  {results['Quantized INT8']['mean']:.3f}ms")
    
    # Print comprehensive results
    print("\n" + "="*110)
    print("LATENCY RESULTS")
    print("="*110)
    
    print(f"\n{'Model':<30} {'Mean (ms)':<12} {'Std':<10} {'P95':<10} {'Throughput':<12} {'Memory (MB)':<12}")
    print("-" * 110)
    
    for name, result in results.items():
        symbol = "" if "Baseline" in name else ""
        print(f"{symbol} {name:<28} {result['mean']:<12.3f} {result['std']:<10.3f} {result['p95']:<10.3f} {result['throughput']:<12.1f} {result['memory']:<12.1f}")
    
    # Speedup analysis
    print("\n" + "="*110)
    print("SPEEDUP vs BASELINE FP32")
    print("="*110)
    
    baseline_fp32_latency = results['Baseline FP32']['mean']
    
    for name, result in results.items():
        if name == 'Baseline FP32':
            print(f"  {name:<28} 1.00x (reference)")
        else:
            speedup = baseline_fp32_latency / result['mean']
            diff = baseline_fp32_latency - result['mean']
            symbol = "" if speedup > 1.0 else " "
            print(f"  {symbol} {name:<26} {speedup:>6.2f}x ({diff:+7.3f}ms)")
    
    # Quantized vs Baseline comparison (same precision)
    print("\n" + "="*110)
    print("QUANTIZED vs BASELINE (Same Precision)")
    print("="*110)
    
    if 'Baseline FP32' in results and 'Quantized FP32' in results:
        speedup = results['Baseline FP32']['mean'] / results['Quantized FP32']['mean']
        diff = results['Baseline FP32']['mean'] - results['Quantized FP32']['mean']
        symbol = "" if speedup > 1.0 else " "
        print(f"  {symbol} PyTorch FP32: {speedup:.2f}x ({diff:+.3f}ms)")
    
    if 'Baseline FP16' in results and 'Quantized FP16' in results:
        speedup = results['Baseline FP16']['mean'] / results['Quantized FP16']['mean']
        diff = results['Baseline FP16']['mean'] - results['Quantized FP16']['mean']
        symbol = "" if speedup > 1.0 else " "
        print(f"  {symbol} TensorRT FP16: {speedup:.2f}x ({diff:+.3f}ms)")
    
    if 'Baseline INT8' in results and 'Quantized INT8' in results:
        speedup = results['Baseline INT8']['mean'] / results['Quantized INT8']['mean']
        diff = results['Baseline INT8']['mean'] - results['Quantized INT8']['mean']
        symbol = "" if speedup > 1.0 else " "
        print(f"  {symbol} TensorRT INT8: {speedup:.2f}x ({diff:+.3f}ms)")
    
    print("="*110 + "\n")
    
    # Save results
    output_file = Path('comprehensive_results.txt')
    with open(output_file, 'w') as f:
        f.write("COMPREHENSIVE LATENCY BENCHMARK RESULTS\n")
        f.write("="*110 + "\n\n")
        
        f.write(f"Baseline: {args.baseline}\n")
        f.write(f"Quantized: {args.quantized} (auto-stripped)\n")
        f.write(f"Runs: {args.num_runs} (warmup: {args.num_warmup})\n\n")
        
        f.write(f"{'Model':<30} {'Mean (ms)':<12} {'Std':<10} {'P95':<10} {'Throughput':<12}\n")
        f.write("-" * 80 + "\n")
        
        for name, result in results.items():
            f.write(f"{name:<30} {result['mean']:<12.3f} {result['std']:<10.3f} {result['p95']:<10.3f} {result['throughput']:<12.1f}\n")
    
    print(f"  Results saved to {output_file}")


if __name__ == '__main__':
    main()
