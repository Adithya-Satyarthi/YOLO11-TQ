#!/usr/bin/env python3
"""
Standalone script to inspect model layers and weights
Checks if quantization actually happened by examining weight distributions
"""



import argparse
import torch
import numpy as np
from pathlib import Path
from collections import Counter


def inspect_model_file(weights_path):
    """
    Directly inspect .pt file contents
    """
    print("\n" + "="*70)
    print("MODEL FILE INSPECTION")
    print("="*70)
    print(f"File: {weights_path}\n")
    
    # Load raw checkpoint
    # FIX: PyTorch 2.6 requires weights_only=False for Ultralytics models
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(weights_path, map_location='cpu')
    
    print(f"Checkpoint type: {type(checkpoint)}")
    print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}\n")
    
    # Extract model
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
            print("Found 'model' key in checkpoint")
        elif 'ema' in checkpoint:
            model = checkpoint['ema']
            print("Found 'ema' key in checkpoint")
        else:
            print("Checkpoint keys:", list(checkpoint.keys()))
            model = checkpoint
    else:
        model = checkpoint
    
    return model



def inspect_state_dict(model):
    """
    Inspect model state_dict for quantization signatures
    """
    print("\n" + "="*70)
    print("STATE_DICT INSPECTION")
    print("="*70)
    
    if hasattr(model, 'state_dict'):
        state_dict = model.state_dict()
    elif isinstance(model, dict):
        state_dict = model
    else:
        print("Cannot extract state_dict")
        return None
    
    # Look for TTQ signatures
    wp_params = [k for k in state_dict.keys() if 'Wp' in k]
    wn_params = [k for k in state_dict.keys() if 'Wn' in k]
    conv_weights = [k for k in state_dict.keys() if 'weight' in k and 'conv' in k.lower()]
    
    print(f"\nTotal parameters: {len(state_dict)}")
    print(f"Conv weight parameters: {len(conv_weights)}")
    print(f"Wp parameters (TTQ signature): {len(wp_params)}")
    print(f"Wn parameters (TTQ signature): {len(wn_params)}")
    
    if wp_params:
        print("\n✓ TTQ PARAMETERS FOUND!")
        print("\nSample Wp/Wn parameters:")
        for i, (wp, wn) in enumerate(zip(wp_params[:5], wn_params[:5])):
            print(f"  {i+1}. {wp}: {state_dict[wp].item():.6f}")
            print(f"     {wn}: {state_dict[wn].item():.6f}")
        if len(wp_params) > 5:
            print(f"  ... and {len(wp_params)-5} more")
    else:
        print("\n✗ NO TTQ PARAMETERS FOUND")
        print("  This suggests the model was saved without TTQ layer info")
    
    return state_dict, conv_weights


def analyze_weight_distribution(state_dict, conv_weights, num_layers=5):
    """
    Analyze actual weight values to detect quantization
    """
    print("\n" + "="*70)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("="*70)
    
    print(f"\nAnalyzing {min(num_layers, len(conv_weights))} conv layers...\n")
    
    for i, weight_key in enumerate(conv_weights[:num_layers]):
        weight = state_dict[weight_key].cpu().numpy().flatten()
        unique_values = np.unique(weight)
        
        print(f"{'─'*70}")
        print(f"Layer {i+1}: {weight_key}")
        print(f"  Shape: {state_dict[weight_key].shape}")
        print(f"  Total params: {len(weight):,}")
        print(f"  Unique values: {len(unique_values)}")
        print(f"  Value range: [{weight.min():.6f}, {weight.max():.6f}]")
        print(f"  Mean: {weight.mean():.6f}, Std: {weight.std():.6f}")
        
        # Check if it looks quantized
        if len(unique_values) <= 10:
            print(f"\n  ⚠ SUSPICIOUS: Only {len(unique_values)} unique values")
            print(f"  Unique values: {unique_values}")
            
            # Count value distribution
            value_counts = Counter(weight)
            most_common = value_counts.most_common(5)
            print(f"\n  Most common values:")
            for val, count in most_common:
                pct = 100 * count / len(weight)
                print(f"    {val:+.6f}: {count:7,} ({pct:5.2f}%)")
            
            # Check for ternary pattern
            if len(unique_values) == 3:
                print(f"\n  ✓ TERNARY PATTERN DETECTED!")
                zeros = np.sum(np.abs(weight) < 1e-6)
                print(f"    Sparsity: {100*zeros/len(weight):.2f}%")
            
        else:
            print(f"  → Full precision (FP32)")
            
            # Show histogram of values
            hist, bins = np.histogram(weight, bins=10)
            print(f"\n  Value distribution:")
            for j, (count, bin_start, bin_end) in enumerate(zip(hist, bins[:-1], bins[1:])):
                bar = '█' * int(50 * count / hist.max())
                print(f"    [{bin_start:+.4f}, {bin_end:+.4f}]: {bar} {count}")


def check_layer_types(model):
    """
    Check actual layer types in the model
    """
    print("\n" + "="*70)
    print("LAYER TYPE INSPECTION")
    print("="*70)
    
    if not hasattr(model, 'named_modules'):
        print("Model doesn't have named_modules() - cannot inspect layer types")
        return
    
    layer_types = {}
    ttq_layers = []
    conv_layers = []
    
    for name, module in model.named_modules():
        layer_type = type(module).__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        if 'TTQ' in layer_type:
            ttq_layers.append(name)
        elif 'Conv2d' in layer_type:
            conv_layers.append(name)
    
    print("\nLayer type summary:")
    for layer_type, count in sorted(layer_types.items(), key=lambda x: -x[1]):
        print(f"  {layer_type}: {count}")
    
    if ttq_layers:
        print(f"\n✓ Found {len(ttq_layers)} TTQ layers:")
        for layer in ttq_layers[:5]:
            print(f"  - {layer}")
        if len(ttq_layers) > 5:
            print(f"  ... and {len(ttq_layers)-5} more")
    else:
        print(f"\n✗ No TTQ layers found")
        print(f"  Found {len(conv_layers)} regular Conv2d layers")
        if conv_layers:
            print(f"\n  Sample Conv2d layers:")
            for layer in conv_layers[:5]:
                print(f"  - {layer}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect model weights to verify quantization'
    )
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model .pt file')
    parser.add_argument('--num-layers', type=int, default=5,
                       help='Number of layers to analyze in detail')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed weight statistics')
    
    args = parser.parse_args()
    
    if not Path(args.weights).exists():
        print(f"Error: File not found: {args.weights}")
        return 1
    
    print("\n" + "="*70)
    print("MODEL WEIGHT INSPECTOR")
    print("="*70)
    print(f"File: {args.weights}")
    
    # Step 1: Inspect raw file
    model = inspect_model_file(args.weights)
    
    # Step 2: Check layer types if possible
    try:
        check_layer_types(model)
    except Exception as e:
        print(f"\nCouldn't inspect layer types: {e}")
    
    # Step 3: Inspect state dict
    result = inspect_state_dict(model)
    if result:
        state_dict, conv_weights = result
        
        # Step 4: Analyze weight distributions
        analyze_weight_distribution(state_dict, conv_weights, args.num_layers)
    
    # Final verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    has_ttq_params = False
    has_ternary_weights = False
    
    if result:
        state_dict, conv_weights = result
        has_ttq_params = any('Wp' in k for k in state_dict.keys())
        
        # Quick check for ternary weights
        for weight_key in conv_weights[:10]:
            weight = state_dict[weight_key].cpu().numpy().flatten()
            unique_values = np.unique(weight)
            if len(unique_values) <= 3:
                has_ternary_weights = True
                break
    
    print(f"\nTTQ parameters (Wp/Wn) in state_dict: {'✓ YES' if has_ttq_params else '✗ NO'}")
    print(f"Ternary weight patterns detected: {'✓ YES' if has_ternary_weights else '✗ NO'}")
    
    if has_ttq_params and has_ternary_weights:
        print("\n✓ Model appears to be TTQ-quantized")
        print("  Problem: Layer types (TTQConv2d) may not be preserved when loading")
        print("  Solution: Weights are quantized, but need to re-apply TTQ layer wrappers")
    elif has_ttq_params and not has_ternary_weights:
        print("\n⚠ Model has TTQ parameters but weights are not ternary")
        print("  This might be before quantization was applied")
    elif not has_ttq_params and has_ternary_weights:
        print("\n⚠ Weights are ternary but no TTQ parameters found")
        print("  Unusual state - check your training code")
    else:
        print("\n✗ Model does NOT appear to be quantized")
        print("  This is a standard full-precision model")
    
    print("="*70)


if __name__ == '__main__':
    import sys
    sys.exit(main())
