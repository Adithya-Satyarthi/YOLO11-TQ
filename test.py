#!/usr/bin/env python3
"""
Verify Quantized YOLO Model Weights
Check if saved model has properly quantized ternary weights (-Wn, 0, +Wp)
"""

import torch
import numpy as np
import argparse
from pathlib import Path


def load_model_safely(model_path):
    """
    Load model checkpoint with proper settings for Ultralytics models.

    Args:
        model_path: Path to model file

    Returns:
        checkpoint: Loaded checkpoint
    """
    try:
        # Try with weights_only=False for Ultralytics models
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        return checkpoint
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        raise


def analyze_weight_distribution(weight_tensor, layer_name):
    """
    Analyze weight distribution to verify ternary quantization.

    Args:
        weight_tensor: PyTorch weight tensor
        layer_name: Name of the layer

    Returns:
        dict: Statistics about the weight distribution
    """
    w = weight_tensor.cpu().numpy().flatten()

    # Get unique values
    unique_vals = np.unique(w)

    # Count near-zero values (should be exactly 0 for ternary)
    zero_count = np.sum(np.abs(w) < 1e-7)

    # Get positive and negative values
    pos_vals = w[w > 1e-7]
    neg_vals = w[w < -1e-7]

    # Check if positive values are all the same (should be Wp)
    pos_unique = np.unique(pos_vals) if len(pos_vals) > 0 else np.array([])
    neg_unique = np.unique(np.abs(neg_vals)) if len(neg_vals) > 0 else np.array([])

    # Format unique values for display
    if len(unique_vals) <= 10:
        unique_display = [f"{v:.6f}" for v in unique_vals]
    else:
        unique_display = [f"{v:.6f}" for v in unique_vals[:3]] + ['...'] + [f"{v:.6f}" for v in unique_vals[-3:]]

    stats = {
        'layer': layer_name,
        'shape': weight_tensor.shape,
        'total_weights': w.size,
        'unique_values_count': len(unique_vals),
        'unique_values': unique_display,
        'zero_count': zero_count,
        'zero_percent': 100 * zero_count / w.size,
        'positive_count': len(pos_vals),
        'positive_percent': 100 * len(pos_vals) / w.size,
        'negative_count': len(neg_vals),
        'negative_percent': 100 * len(neg_vals) / w.size,
        'is_ternary': len(unique_vals) == 3,
        'positive_unique_count': len(pos_unique),
        'negative_unique_count': len(neg_unique),
        'wp_value': pos_unique[0] if len(pos_unique) == 1 else None,
        'wn_value': neg_unique[0] if len(neg_unique) == 1 else None,
    }

    return stats


def get_state_dict_from_checkpoint(checkpoint):
    """
    Extract state_dict from various checkpoint formats.

    Args:
        checkpoint: Loaded checkpoint

    Returns:
        state_dict: Model state dictionary
        metadata: Additional metadata
    """
    metadata = {
        'quantized_layers': [],
        'threshold': None
    }

    # Handle dictionary checkpoint
    if isinstance(checkpoint, dict):
        # TTQ checkpoint format
        if 'model' in checkpoint:
            model = checkpoint['model']
            metadata['quantized_layers'] = checkpoint.get('quantized_layers', [])
            metadata['threshold'] = checkpoint.get('threshold', None)

            # Extract state dict
            if hasattr(model, 'model'):
                state_dict = model.model.state_dict()
            elif hasattr(model, 'state_dict'):
                if callable(model.state_dict):
                    state_dict = model.state_dict()
                else:
                    state_dict = model.state_dict
            else:
                state_dict = model

        # Ultralytics checkpoint format
        elif 'model' not in checkpoint and any(k.startswith('model.') for k in checkpoint.keys()):
            state_dict = checkpoint

        # Direct state_dict
        else:
            state_dict = checkpoint

    # Handle YOLO model object
    elif hasattr(checkpoint, 'model'):
        if hasattr(checkpoint.model, 'state_dict'):
            state_dict = checkpoint.model.state_dict()
        else:
            state_dict = checkpoint.model

    # Handle nn.Module
    elif hasattr(checkpoint, 'state_dict'):
        if callable(checkpoint.state_dict):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint.state_dict

    else:
        state_dict = checkpoint

    return state_dict, metadata


def verify_quantized_model(model_path, verbose=True):
    """
    Verify if a model has properly quantized weights.

    Args:
        model_path: Path to the saved model checkpoint
        verbose: Print detailed statistics

    Returns:
        dict: Summary of quantization verification
    """
    print("=" * 80)
    print(f"Verifying Quantized Model: {model_path}")
    print("=" * 80)

    # Load checkpoint
    checkpoint = load_model_safely(model_path)

    # Extract state dict and metadata
    state_dict, metadata = get_state_dict_from_checkpoint(checkpoint)

    if metadata['quantized_layers']:
        print(f"\nCheckpoint Info:")
        print(f"  Quantized layers: {len(metadata['quantized_layers'])}")
        print(f"  Threshold: {metadata['threshold']}")

    print(f"\nTotal parameters in state_dict: {len(state_dict)}")

    # Analyze Conv2d weights
    conv_stats = []
    ternary_count = 0
    fp_count = 0

    print("\n" + "=" * 80)
    print("WEIGHT ANALYSIS")
    print("=" * 80)

    for name, param in state_dict.items():
        # Only analyze Conv2d weights (not biases)
        if 'weight' in name and len(param.shape) == 4:  # Conv2d weights are 4D
            stats = analyze_weight_distribution(param, name)
            conv_stats.append(stats)

            if stats['is_ternary'] and stats['positive_unique_count'] == 1 and stats['negative_unique_count'] == 1:
                ternary_count += 1
                status = "✓ TERNARY"
            else:
                fp_count += 1
                status = "✗ FP32   "

            if verbose:
                # Shorten layer name for display
                display_name = name
                if len(name) > 45:
                    parts = name.split('.')
                    if len(parts) > 3:
                        display_name = '...' + '.'.join(parts[-3:])

                print(f"\n{status} | {display_name}")
                print(f"  Shape: {list(stats['shape'])}")
                print(f"  Unique values: {stats['unique_values_count']}")

                if stats['unique_values_count'] <= 10:
                    print(f"  Values: [{', '.join(stats['unique_values'])}]")

                print(f"  Distribution: {stats['zero_percent']:.1f}% zero, " + 
                      f"{stats['positive_percent']:.1f}% pos, {stats['negative_percent']:.1f}% neg")

                if stats['is_ternary'] and stats['wp_value'] is not None:
                    print(f"  Wp = {stats['wp_value']:.6f}, Wn = {stats['wn_value']:.6f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Conv2d layers analyzed: {len(conv_stats)}")
    print(f"Ternary quantized layers: {ternary_count}")
    print(f"FP32 layers: {fp_count}")

    if len(conv_stats) > 0:
        print(f"Quantization rate: {100 * ternary_count / len(conv_stats):.1f}%")

    if ternary_count == 0:
        print("\n⚠️  WARNING: No ternary quantized layers found!")
        print("   The model appears to be in FP32 format.")
    elif fp_count == 0:
        print("\n✓ SUCCESS: All layers are ternary quantized!")
    else:
        print(f"\n✓ Partial quantization: {ternary_count}/{len(conv_stats)} layers quantized")

    # Detailed ternary layer report
    if ternary_count > 0:
        print("\n" + "=" * 80)
        print("TERNARY LAYER DETAILS")
        print("=" * 80)
        print(f"{'Layer':<50} {'Wp':>10} {'Wn':>10} {'Zeros %':>10}")
        print("-" * 80)

        for stats in conv_stats:
            if stats['is_ternary'] and stats['positive_unique_count'] == 1 and stats['wp_value'] is not None:
                layer_short = stats['layer']
                if len(layer_short) > 50:
                    parts = layer_short.split('.')
                    layer_short = '.'.join(parts[-3:]) if len(parts) > 3 else layer_short

                print(f"{layer_short:<50} {stats['wp_value']:>10.6f} {stats['wn_value']:>10.6f} {stats['zero_percent']:>9.1f}%")

    print("=" * 80)

    return {
        'total_layers': len(conv_stats),
        'ternary_layers': ternary_count,
        'fp32_layers': fp_count,
        'quantization_rate': ternary_count / len(conv_stats) if len(conv_stats) > 0 else 0,
        'all_stats': conv_stats
    }


def compare_models(original_path, quantized_path):
    """
    Compare original FP32 model with quantized model.

    Args:
        original_path: Path to original FP32 model
        quantized_path: Path to quantized model
    """
    print("\n" + "=" * 80)
    print("COMPARING MODELS")
    print("=" * 80)

    print("\n[1] Original FP32 Model:")
    orig_results = verify_quantized_model(original_path, verbose=False)

    print("\n[2] Quantized Model:")
    quant_results = verify_quantized_model(quantized_path, verbose=True)

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Original model - Ternary layers: {orig_results['ternary_layers']}/{orig_results['total_layers']}")
    print(f"Quantized model - Ternary layers: {quant_results['ternary_layers']}/{quant_results['total_layers']}")
    print(f"Quantization increase: +{quant_results['ternary_layers'] - orig_results['ternary_layers']} layers")

    if quant_results['ternary_layers'] > orig_results['ternary_layers']:
        print("\n✓ Quantization successful!")
    else:
        print("\n⚠️  WARNING: No new layers were quantized!")


def main():
    parser = argparse.ArgumentParser(description='Verify YOLO model quantization')
    parser.add_argument('model_path', type=str, help='Path to quantized model checkpoint')
    parser.add_argument('--original', type=str, default=None, help='Path to original FP32 model for comparison')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return

    if args.original:
        if not Path(args.original).exists():
            print(f"Error: Original model file not found: {args.original}")
            return
        compare_models(args.original, args.model_path)
    else:
        verify_quantized_model(args.model_path, verbose=not args.quiet)


if __name__ == '__main__':
    main()
