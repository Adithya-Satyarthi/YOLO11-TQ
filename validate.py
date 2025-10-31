#!/usr/bin/env python3
"""
Validate and Compare YOLO Model Accuracy
Compare mAP metrics between FP32 and quantized models
"""

import torch
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml


def load_model_safely(model_path):
    """
    Load YOLO model from checkpoint.
    
    Args:
        model_path: Path to model file
    
    Returns:
        YOLO model object
    """
    try:
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}")
        raise


def validate_model(model, data_yaml, imgsz=640, batch=16, device=0, verbose=True):
    """
    Validate model and return metrics.
    
    Args:
        model: YOLO model object
        data_yaml: Path to dataset YAML file
        imgsz: Image size for validation
        batch: Batch size
        device: Device to use (0 for GPU, 'cpu' for CPU)
        verbose: Print detailed results
    
    Returns:
        dict: Validation metrics
    """
    print(f"\nRunning validation on {data_yaml}...")
    print(f"  Image size: {imgsz}, Batch size: {batch}, Device: {device}")
    
    try:
        results = model.val(
            data=data_yaml,
            imgsz=imgsz,
            batch=batch,
            device=device,
            verbose=verbose,
            plots=False  # Disable plot generation for speed
        )
        
        # Extract metrics
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.p[-1] if hasattr(results.box, 'p') and len(results.box.p) else 0),
            'recall': float(results.box.r[-1] if hasattr(results.box, 'r') and len(results.box.r) else 0),
            'map75': float(results.box.map75) if hasattr(results.box, 'map75') else 0.0,
        }
        
        # Per-class mAP if available
        if hasattr(results.box, 'maps'):
            metrics['per_class_map'] = results.box.maps.tolist()
        
        return metrics
    
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return None


def print_metrics_table(fp32_metrics, quant_metrics, model_name="Model"):
    """
    Print formatted comparison table.
    
    Args:
        fp32_metrics: Metrics from FP32 model
        quant_metrics: Metrics from quantized model
        model_name: Name to display in table
    """
    print("\n" + "=" * 90)
    print(f"ACCURACY COMPARISON: {model_name}")
    print("=" * 90)
    print(f"{'Metric':<20} {'FP32':>15} {'Quantized':>15} {'Difference':>15} {'Degradation':>15}")
    print("-" * 90)
    
    metrics_to_compare = ['mAP50-95', 'mAP50', 'map75', 'precision', 'recall']
    
    for metric in metrics_to_compare:
        if metric in fp32_metrics and metric in quant_metrics:
            fp32_val = fp32_metrics[metric]
            quant_val = quant_metrics[metric]
            diff = quant_val - fp32_val
            degradation_pct = (diff / fp32_val * 100) if fp32_val > 0 else 0
            
            # Color coding for terminal (optional)
            if abs(degradation_pct) < 2:
                status = "✓"  # Excellent
            elif abs(degradation_pct) < 5:
                status = "○"  # Good
            else:
                status = "⚠"  # Warning
            
            print(f"{metric:<20} {fp32_val:>15.4f} {quant_val:>15.4f} {diff:>+15.4f} {degradation_pct:>+14.2f}% {status}")
    
    print("=" * 90)
    
    # Summary
    main_metric_diff = quant_metrics['mAP50-95'] - fp32_metrics['mAP50-95']
    main_metric_pct = (main_metric_diff / fp32_metrics['mAP50-95'] * 100) if fp32_metrics['mAP50-95'] > 0 else 0
    
    print(f"\nSummary:")
    print(f"  Primary metric (mAP50-95): {fp32_metrics['mAP50-95']:.4f} → {quant_metrics['mAP50-95']:.4f}")
    print(f"  Accuracy degradation: {main_metric_pct:+.2f}%")
    
    if abs(main_metric_pct) < 1:
        print(f"  Status: ✓ EXCELLENT - Less than 1% degradation")
    elif abs(main_metric_pct) < 3:
        print(f"  Status: ✓ GOOD - Less than 3% degradation")
    elif abs(main_metric_pct) < 5:
        print(f"  Status: ○ ACCEPTABLE - Less than 5% degradation")
    else:
        print(f"  Status: ⚠ WARNING - More than 5% degradation")
    
    print("=" * 90)


def compare_models(fp32_path, quant_path, data_yaml, imgsz=640, batch=16, device=0):
    """
    Compare FP32 and quantized models.
    
    Args:
        fp32_path: Path to FP32 model
        quant_path: Path to quantized model
        data_yaml: Dataset YAML file
        imgsz: Image size
        batch: Batch size
        device: Device to use
    """
    print("\n" + "=" * 90)
    print("MODEL ACCURACY VALIDATION AND COMPARISON")
    print("=" * 90)
    
    # Validate FP32 model
    print("\n[1/2] Validating FP32 Model")
    print("-" * 90)
    fp32_model = load_model_safely(fp32_path)
    fp32_metrics = validate_model(fp32_model, data_yaml, imgsz, batch, device, verbose=False)
    
    if fp32_metrics is None:
        print("❌ Failed to validate FP32 model")
        return
    
    print(f"\n✓ FP32 Validation Complete")
    print(f"  mAP50-95: {fp32_metrics['mAP50-95']:.4f}")
    print(f"  mAP50:    {fp32_metrics['mAP50']:.4f}")
    
    # Validate quantized model
    print("\n[2/2] Validating Quantized Model")
    print("-" * 90)
    quant_model = load_model_safely(quant_path)
    quant_metrics = validate_model(quant_model, data_yaml, imgsz, batch, device, verbose=False)
    
    if quant_metrics is None:
        print("❌ Failed to validate quantized model")
        return
    
    print(f"\n✓ Quantized Validation Complete")
    print(f"  mAP50-95: {quant_metrics['mAP50-95']:.4f}")
    print(f"  mAP50:    {quant_metrics['mAP50']:.4f}")
    
    # Extract model name for display
    model_name = Path(quant_path).parent.name
    
    # Print comparison
    print_metrics_table(fp32_metrics, quant_metrics, model_name)
    
    # Save results to file
    results_dir = Path(quant_path).parent
    results_file = results_dir / 'validation_results.yaml'
    
    results_dict = {
        'fp32_model': str(fp32_path),
        'quantized_model': str(quant_path),
        'dataset': str(data_yaml),
        'fp32_metrics': fp32_metrics,
        'quantized_metrics': quant_metrics,
        'degradation_percent': float((quant_metrics['mAP50-95'] - fp32_metrics['mAP50-95']) / fp32_metrics['mAP50-95'] * 100)
    }
    
    with open(results_file, 'w') as f:
        yaml.dump(results_dict, f, default_flow_style=False)
    
    print(f"\n📄 Results saved to: {results_file}")
    
    return fp32_metrics, quant_metrics


def validate_single_model(model_path, data_yaml, imgsz=640, batch=16, device=0):
    """
    Validate a single model.
    
    Args:
        model_path: Path to model checkpoint
        data_yaml: Dataset YAML file
        imgsz: Image size
        batch: Batch size
        device: Device to use
    """
    print("\n" + "=" * 90)
    print("MODEL ACCURACY VALIDATION")
    print("=" * 90)
    
    model = load_model_safely(model_path)
    metrics = validate_model(model, data_yaml, imgsz, batch, device, verbose=True)
    
    if metrics is None:
        print("❌ Validation failed")
        return
    
    print("\n" + "=" * 90)
    print("VALIDATION RESULTS")
    print("=" * 90)
    print(f"{'Metric':<20} {'Value':>15}")
    print("-" * 90)
    
    for key, value in metrics.items():
        if key != 'per_class_map':
            print(f"{key:<20} {value:>15.4f}")
    
    print("=" * 90)
    
    # Save results
    results_dir = Path(model_path).parent
    results_file = results_dir / 'validation_results.yaml'
    
    results_dict = {
        'model': str(model_path),
        'dataset': str(data_yaml),
        'metrics': metrics
    }
    
    with open(results_file, 'w') as f:
        yaml.dump(results_dict, f, default_flow_style=False)
    
    print(f"\n📄 Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate YOLO model accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single model
  python test.py model.pt --data coco.yaml
  
  # Compare FP32 vs quantized
  python test.py quantized.pt --fp32 yolo11n.pt --data coco.yaml
  
  # Use custom settings
  python test.py model.pt --data coco.yaml --imgsz 640 --batch 32 --device 0
        """
    )
    
    parser.add_argument('model_path', type=str, 
                       help='Path to model checkpoint (quantized model if comparing)')
    parser.add_argument('--fp32', type=str, default=None,
                       help='Path to FP32 model for comparison')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML file (e.g., coco.yaml, coco128.yaml)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for validation (default: 640)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use: 0, 1, 2, etc. or "cpu" (default: 0)')
    
    args = parser.parse_args()
    
    # Convert device to int if numeric
    try:
        device = int(args.device)
    except ValueError:
        device = args.device
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"❌ Error: Model file not found: {args.model_path}")
        return
    
    # Compare mode
    if args.fp32:
        if not Path(args.fp32).exists():
            print(f"❌ Error: FP32 model file not found: {args.fp32}")
            return
        
        compare_models(
            fp32_path=args.fp32,
            quant_path=args.model_path,
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device
        )
    
    # Single model validation
    else:
        validate_single_model(
            model_path=args.model_path,
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device
        )


if __name__ == '__main__':
    main()
