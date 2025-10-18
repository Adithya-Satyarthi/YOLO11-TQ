#!/usr/bin/env python3
"""
Evaluate TTQ-quantized YOLO11 model
"""

import argparse
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantization.quantize_model import print_quantization_stats
from quantization.ttq_layer import TTQConv2dWithGrad
import torch


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TTQ-YOLO11')
    
    # Config or direct arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained TTQ model weights')
    parser.add_argument('--data', type=str, default=None,
                       help='Dataset YAML file (overrides config)')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='Inference image size')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='CUDA device')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("TTQ-YOLO11 Evaluation")
    print("="*70)
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Set defaults
    data = args.data or config.get('data', {}).get('val', 'coco.yaml')
    imgsz = args.imgsz or config.get('val', {}).get('imgsz', 640)
    batch = args.batch or config.get('val', {}).get('batch', 32)
    device = args.device or config.get('train', {}).get('device', '0')
    
    # Load quantized model
    print(f"\nLoading model from: {args.weights}")
    from ultralytics import YOLO
    model = YOLO(args.weights)
    
    if args.verbose:
        print_quantization_stats(model.model)
    
    # Evaluate
    print("\nRunning evaluation...")
    print(f"  Dataset: {data}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    
    results = model.val(
        data=data,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=args.verbose
    )
    
    # Print results
    print("\n" + "="*70)
    print("Evaluation Results:")
    print("="*70)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print("="*70)
    
    return results


if __name__ == '__main__':
    main()
