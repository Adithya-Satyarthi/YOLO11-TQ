#!/usr/bin/env python3
"""
Compression Analysis Tool (Accurate TTQ Calculation)
Analyzes quantization coverage and calculates theoretical compression ratios
for TTQ and BitLinear quantization methods

TTQ Calculation:
- Each weight: 2 bits (ternary: {-1, 0, +1})
- Per layer scales: 2 × 32-bit floats (alpha_positive, alpha_negative)
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO


class CompressionAnalyzer:
    """Analyze compression ratios of quantized models"""
    
    def __init__(self, baseline_path: str, quantized_path: str):
        self.baseline_path = baseline_path
        self.quantized_path = quantized_path
        
        print("\n" + "="*80)
        print("Compression Analysis Tool (TTQ-Aware)")
        print("="*80)
        print(f"Baseline model: {baseline_path}")
        print(f"Quantized model: {quantized_path}\n")
        
        # Load baseline
        print("Loading baseline model...")
        self.baseline_yolo = YOLO(baseline_path)
        self.baseline_model = self.baseline_yolo.model
        print("✓ Baseline loaded")
        
        # Load quantized
        print("Loading quantized model...")
        quantized_ckpt = torch.load(quantized_path, map_location='cpu', weights_only=False)
        if isinstance(quantized_ckpt, dict) and 'model' in quantized_ckpt:
            self.quantized_model = quantized_ckpt['model']
        else:
            self.quantized_model = quantized_ckpt
        print("✓ Quantized model loaded\n")
    
    def analyze_layer_types(self) -> Dict:
        """Analyze which layers are quantized (Conv, Linear, etc)"""
        
        print("="*80)
        print("Layer Analysis")
        print("="*80 + "\n")
        
        layer_stats = {
            'conv_layers': [],
            'linear_layers': [],
            'attention_layers': [],
        }
        
        for name, module in self.quantized_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                params = module.weight.numel()
                layer_stats['conv_layers'].append({
                    'name': name,
                    'params': params,
                    'weight_shape': tuple(module.weight.shape)
                })
            
            elif isinstance(module, torch.nn.Linear):
                params = module.weight.numel()
                layer_stats['linear_layers'].append({
                    'name': name,
                    'params': params,
                    'weight_shape': tuple(module.weight.shape)
                })
            
            elif 'attn' in name.lower() or 'attention' in name.lower():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    params = module.weight.numel()
                    layer_stats['attention_layers'].append({
                        'name': name,
                        'params': params,
                    })
        
        print(f"Conv2D layers: {len(layer_stats['conv_layers'])}")
        print(f"Linear layers: {len(layer_stats['linear_layers'])}")
        print(f"Attention layers: {len(layer_stats['attention_layers'])}\n")
        
        return layer_stats
    
    def detect_quantization_type(self) -> Tuple[str, Dict]:
        """Detect if model uses TTQ or BitLinear"""
        
        print("="*80)
        print("Quantization Type Detection")
        print("="*80 + "\n")
        
        # Check for ternary weights (TTQ)
        ternary_count = 0
        total_conv = 0
        ternary_details = {}
        
        for name, module in self.quantized_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                total_conv += 1
                w = module.weight.data
                unique_vals = torch.unique(w)
                
                # TTQ uses ~3 unique values per layer
                if len(unique_vals) <= 5:
                    ternary_count += 1
                    ternary_details[name] = {
                        'unique_vals': len(unique_vals),
                        'min': float(unique_vals.min()),
                        'max': float(unique_vals.max()),
                        'params': w.numel()
                    }
                    if ternary_count <= 5:  # Show first 5
                        print(f"  ✓ Ternary: {name}")
                        print(f"    Params: {w.numel():,}")
                        print(f"    Unique values: {len(unique_vals)}")
                        print(f"    Range: [{unique_vals.min():.4f}, {unique_vals.max():.4f}]\n")
        
        if ternary_count > 5:
            print(f"  ... and {ternary_count - 5} more ternary layers\n")
        
        # Check for BitLinear (checks for specific patterns)
        bitlinear_count = 0
        for name, module in self.quantized_model.named_modules():
            if 'bitlinear' in name.lower() or 'bits' in name.lower():
                bitlinear_count += 1
        
        print(f"Ternary Conv layers: {ternary_count}/{total_conv}")
        print(f"BitLinear modules: {bitlinear_count}\n")
        
        if bitlinear_count > 0:
            quant_type = "BitLinear + TTQ"
        elif ternary_count > total_conv * 0.8:
            quant_type = "TTQ (Full)"
        else:
            quant_type = "TTQ (Partial)"
        
        return quant_type, ternary_details
    
    def calculate_theoretical_compression(self) -> Dict:
        """Calculate theoretical compression ratio with accurate TTQ calculation"""
        
        print("="*80)
        print("Theoretical Compression Analysis (Accurate TTQ)")
        print("="*80 + "\n")
        
        results = {
            'baseline_size': 0,
            'quantized_size': 0,
            'compression_details': {}
        }
        
        # ========== BASELINE CALCULATION ==========
        baseline_bits = 0
        baseline_params = 0
        
        for name, module in self.baseline_model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                params = module.weight.numel()
                baseline_params += params
                baseline_bits += params * 32  # FP32 = 32 bits
        
        results['baseline_params'] = baseline_params
        results['baseline_bits'] = baseline_bits
        results['baseline_size_mb'] = baseline_bits / (8 * 1024 * 1024)
        
        print(f"Baseline Model (FP32):")
        print(f"  Total parameters: {baseline_params:,}")
        print(f"  Bits per weight: 32")
        print(f"  Total bits: {baseline_bits / 1e9:.2f} Gb")
        print(f"  Estimated size: {results['baseline_size_mb']:.2f} MB\n")
        
        # ========== QUANTIZED CALCULATION (TTQ-AWARE) ==========
        quantized_bits = 0
        quantized_params = 0
        
        ternary_layers = []
        fp32_layers = []
        
        for name, module in self.quantized_model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                params = module.weight.numel()
                w = module.weight.data
                unique_vals = torch.unique(w)
                
                if len(unique_vals) <= 5:  # Ternary (TTQ)
                    # TTQ Storage:
                    # - 2 bits per weight
                    # - 2 × 32-bit scales per layer (alpha_pos, alpha_neg)
                    
                    weight_bits = params * 2  # 2 bits per ternary weight
                    scale_bits = 2 * 32  # 2 float32 scales
                    layer_bits = weight_bits + scale_bits
                    
                    ternary_layers.append({
                        'name': name,
                        'params': params,
                        'unique_vals': len(unique_vals),
                        'weight_bits': weight_bits,
                        'scale_bits': scale_bits,
                        'total_bits': layer_bits,
                    })
                    
                    quantized_bits += layer_bits
                    quantized_params += params
                    
                else:  # Not quantized - keep as FP32
                    fp32_bits = params * 32
                    fp32_layers.append({
                        'name': name,
                        'params': params,
                        'bits': fp32_bits,
                    })
                    quantized_bits += fp32_bits
                    quantized_params += params
        
        results['quantized_params'] = quantized_params
        results['quantized_bits'] = quantized_bits
        results['quantized_size_mb'] = quantized_bits / (8 * 1024 * 1024)
        
        print(f"Quantized Model (TTQ):")
        print(f"  Total parameters: {quantized_params:,}")
        print(f"  Ternary layers: {len(ternary_layers)}")
        print(f"  FP32 layers: {len(fp32_layers)}")
        print(f"  Ternary weights: 2 bits + 64 bits scale per layer")
        print(f"  Total bits: {quantized_bits / 1e9:.2f} Gb")
        print(f"  Estimated size: {results['quantized_size_mb']:.2f} MB\n")
        
        # ========== COMPRESSION METRICS ==========
        compression_ratio = results['baseline_size_mb'] / results['quantized_size_mb']
        bit_reduction = (baseline_bits - quantized_bits) / baseline_bits * 100
        
        print(f"Compression Results:")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Bit reduction: {bit_reduction:.1f}%")
        print(f"  Size reduction: {results['baseline_size_mb'] - results['quantized_size_mb']:.2f} MB")
        print(f"  Effective bits per weight: {quantized_bits / quantized_params:.4f} bits\n")
        
        # ========== DETAILED BREAKDOWN ==========
        if ternary_layers:
            total_ternary_weight_bits = sum(l['weight_bits'] for l in ternary_layers)
            total_ternary_scale_bits = sum(l['scale_bits'] for l in ternary_layers)
            
            print(f"TTQ Breakdown:")
            print(f"  Weight bits (2b each): {total_ternary_weight_bits / 1e9:.3f} Gb")
            print(f"  Scale bits (64b/layer): {total_ternary_scale_bits / 1e9:.3f} Gb")
            print(f"  Scale overhead percentage: {total_ternary_scale_bits / (total_ternary_weight_bits + total_ternary_scale_bits) * 100:.2f}%\n")
        
        results['ternary_layers'] = ternary_layers
        results['fp32_layers'] = fp32_layers
        results['compression_ratio'] = compression_ratio
        results['bit_reduction'] = bit_reduction
        
        return results
    
    def analyze_layer_coverage(self) -> Dict:
        """Analyze what percentage of model is quantized"""
        
        print("="*80)
        print("Quantization Coverage")
        print("="*80 + "\n")
        
        coverage = {
            'total_params': 0,
            'quantized_params': 0,
            'fp32_params': 0,
            'by_type': {}
        }
        
        layer_types = {
            'Conv2D': 0,
            'Linear': 0,
        }
        
        quantized_by_type = {
            'Conv2D': 0,
            'Linear': 0,
        }
        
        for name, module in self.quantized_model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                params = module.weight.numel()
                coverage['total_params'] += params
                
                # Determine type
                if isinstance(module, torch.nn.Conv2d):
                    layer_types['Conv2D'] += params
                    layer_key = 'Conv2D'
                else:
                    layer_types['Linear'] += params
                    layer_key = 'Linear'
                
                # Check if quantized
                w = module.weight.data
                unique_vals = torch.unique(w)
                
                if len(unique_vals) <= 5:  # Ternary
                    coverage['quantized_params'] += params
                    quantized_by_type[layer_key] += params
                else:
                    coverage['fp32_params'] += params
        
        total = coverage['total_params']
        
        print(f"Overall Coverage:")
        print(f"  Total parameters: {total:,}")
        print(f"  Quantized (ternary): {coverage['quantized_params']:,}")
        print(f"    Percentage: {coverage['quantized_params']/total*100:.1f}%")
        print(f"  FP32: {coverage['fp32_params']:,}")
        print(f"    Percentage: {coverage['fp32_params']/total*100:.1f}%\n")
        
        print(f"Coverage by Layer Type:")
        for ltype in ['Conv2D', 'Linear']:
            if layer_types[ltype] > 0:
                pct = quantized_by_type[ltype] / layer_types[ltype] * 100
                print(f"  {ltype}: {pct:.1f}% quantized ({quantized_by_type[ltype]:,} / {layer_types[ltype]:,} params)")
        
        print()
        
        coverage['by_type'] = quantized_by_type
        return coverage
    
    def compare_file_sizes(self) -> Dict:
        """Compare actual file sizes"""
        
        print("="*80)
        print("Actual File Sizes")
        print("="*80 + "\n")
        
        baseline_size = Path(self.baseline_path).stat().st_size / (1024 * 1024)
        quantized_size = Path(self.quantized_path).stat().st_size / (1024 * 1024)
        
        if quantized_size > 0:
            actual_ratio = baseline_size / quantized_size
        else:
            actual_ratio = 0
        
        print(f"Baseline file size: {baseline_size:.2f} MB")
        print(f"Quantized file size: {quantized_size:.2f} MB")
        print(f"Actual compression ratio: {actual_ratio:.2f}x")
        print(f"Size saved: {baseline_size - quantized_size:.2f} MB\n")
        
        return {
            'baseline_mb': baseline_size,
            'quantized_mb': quantized_size,
            'actual_ratio': actual_ratio,
        }
    
    def generate_report(self):
        """Generate comprehensive compression report"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPRESSION REPORT")
        print("="*80 + "\n")
        
        # Get all analysis
        quant_type, ternary_details = self.detect_quantization_type()
        theoretical = self.calculate_theoretical_compression()
        coverage = self.analyze_layer_coverage()
        actual = self.compare_file_sizes()
        
        # Summary
        print("="*80)
        print("FINAL SUMMARY")
        print("="*80 + "\n")
        
        print(f"Quantization Method: {quant_type}\n")
        
        print(f"Theoretical Compression:")
        print(f"  Ratio: {theoretical['compression_ratio']:.2f}x")
        print(f"  Bit reduction: {theoretical['bit_reduction']:.1f}%")
        print(f"  Effective bits/weight: {theoretical['quantized_bits'] / theoretical['quantized_params']:.4f}\n")
        
        print(f"Actual File Compression:")
        print(f"  Ratio: {actual['actual_ratio']:.2f}x")
        print(f"  Size saved: {actual['baseline_mb'] - actual['quantized_mb']:.2f} MB\n")
        
        print(f"Quantization Coverage:")
        total = coverage['total_params']
        quant_pct = coverage['quantized_params'] / total * 100
        print(f"  {quant_pct:.1f}% of parameters quantized")
        print(f"  {coverage['quantized_params']:,} / {total:,} parameters\n")
        
        print(f"Theoretical vs Actual Compression:")
        ratio_diff = theoretical['compression_ratio'] - actual['actual_ratio']
        print(f"  Theoretical: {theoretical['compression_ratio']:.2f}x")
        print(f"  Actual: {actual['actual_ratio']:.2f}x")
        print(f"  Difference: {ratio_diff:+.2f}x (overhead from checkpoint format, optimizer state, etc.)\n")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Model Compression Analysis (TTQ-Aware)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze TTQ model
  python analyze_compression.py \\
    --baseline yolo11n.pt \\
    --quantized ttq_checkpoints/yolo11n/stage1_progressive_final/best.pt

  # Analyze BitLinear model
  python analyze_compression.py \\
    --baseline yolo11n.pt \\
    --quantized checkpoints/stage2_c2psa_standard/best.pt
        """)
    
    parser.add_argument('--baseline', type=str, required=True, help='Baseline model path')
    parser.add_argument('--quantized', type=str, required=True, help='Quantized model path')
    
    args = parser.parse_args()
    
    analyzer = CompressionAnalyzer(args.baseline, args.quantized)
    analyzer.generate_report()


if __name__ == '__main__':
    main()
