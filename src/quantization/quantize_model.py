"""
Utilities to convert YOLO11 model to TTQ version
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from .ttq_layer import TTQConv2dWithGrad


def should_quantize_module(module_name, module, quantize_first_layer=False):
    """
    Determine if a module should be quantized based on its name and type.
    
    Excludes:
    - Detect head
    - C2PSA attention blocks
    - First conv layer
    
    Args:
        module_name: Full name of the module
        module: The module itself
        quantize_first_layer: Whether to quantize first conv layer
    
    Returns:
        bool: True if module should be quantized
    """
    # Skip if not a Conv2d layer
    if not isinstance(module, nn.Conv2d):
        return False
    
    # Parse module path
    parts = module_name.split('.')
    
    # Need at least model.X format
    if len(parts) < 3:
        return False
    
    # Get the layer index (e.g., '23' from 'model.model.23.cv2.0.0.conv')
    if parts[0] == 'model' and parts[1] == 'model':
        layer_idx = parts[2]
    else:
        return False
    
    # Exclude Detect head (model.23.*)
    if layer_idx == '23':
        return False
    
    # Exclude C2PSA blocks (model.10.*)
    if layer_idx == '10':
        return False
    
    # Optionally keep first conv layer in full precision
    if layer_idx == '0' and not quantize_first_layer:
        return False
    
    return True


def replace_conv_with_ttq(module, name, threshold=0.05, quantize_first_layer=False):
    """
    Recursively replace Conv2d layers with TTQ quantized versions.
    
    Args:
        module: PyTorch module
        name: Module name
        threshold: Threshold factor for quantization (default: 0.05 as in paper)
        quantize_first_layer: Whether to quantize first layer
    
    Returns:
        Number of quantized layers
    """
    quantized_count = 0
    
    for child_name, child_module in list(module.named_children()):
        full_name = f"{name}.{child_name}" if name else child_name
        
        if isinstance(child_module, nn.Conv2d):
            if should_quantize_module(full_name, child_module, quantize_first_layer):
                # Create TTQ layer with same parameters
                ttq_layer = TTQConv2dWithGrad(
                    in_channels=child_module.in_channels,
                    out_channels=child_module.out_channels,
                    kernel_size=child_module.kernel_size,
                    stride=child_module.stride,
                    padding=child_module.padding,
                    dilation=child_module.dilation,
                    groups=child_module.groups,
                    bias=child_module.bias is not None,
                    threshold=threshold
                )
                
                # Copy weights from original layer
                ttq_layer.weight.data.copy_(child_module.weight.data)
                if child_module.bias is not None:
                    ttq_layer.bias.data.copy_(child_module.bias.data)
                
                # Replace the layer
                setattr(module, child_name, ttq_layer)
                print(f"  ✓ Quantized: {full_name}")
                quantized_count += 1
            else:
                # Determine reason for skipping
                parts = full_name.split('.')
                if len(parts) >= 3 and parts[1] == 'model':
                    layer_idx = parts[2]
                    if layer_idx == '23':
                        print(f"  Skipping (Detect head): {full_name}")
                    elif layer_idx == '10':
                        print(f"  Skipping (C2PSA attention): {full_name}")
                    elif layer_idx == '0':
                        print(f"  Skipping (First layer): {full_name}")
        else:
            # Recursively process child modules
            quantized_count += replace_conv_with_ttq(
                child_module, full_name, threshold, quantize_first_layer
            )
    
    return quantized_count


def quantize_yolo_model(model_path, threshold=0.05, quantize_first_layer=False, verbose=True):
    """
    Load a YOLO11 model and convert Conv2d layers to TTQ.
    
    Args:
        model_path: Path to YOLO11 model (.pt file or 'yolo11n.pt', etc.)
        threshold: Threshold factor for quantization
        quantize_first_layer: Whether to quantize first conv layer
        verbose: Print quantization information
    
    Returns:
        Quantized YOLO model
    """
    # Load YOLO model
    model = YOLO(model_path)
    
    if verbose:
        print(f"\nLoading YOLO11 model from: {model_path}")
        print("="*60)
    
    # Access the actual PyTorch model
    pytorch_model = model.model
    
    # Replace Conv2d with TTQ layers
    if verbose:
        print("\nQuantizing Conv2d layers to TTQ...")
        print("="*60)
        print("Skipping: Detect head (model.23.*) and C2PSA attention (model.10.*)")
        print("="*60)
    
    quantized_count = replace_conv_with_ttq(
        pytorch_model, "model", threshold, quantize_first_layer
    )
    
    if verbose:
        print("\n" + "="*60)
        print("Quantization complete!")
        print_quantization_stats(pytorch_model)
    
    return model


def print_quantization_stats(model):
    """Print statistics about quantized model"""
    total_conv = 0
    quantized_conv = 0
    total_params = 0
    quantized_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, TTQConv2dWithGrad)):
            if isinstance(module, nn.Conv2d) and not isinstance(module, TTQConv2dWithGrad):
                total_conv += 1
                params = module.weight.numel()
                total_params += params
            elif isinstance(module, TTQConv2dWithGrad):
                total_conv += 1
                quantized_conv += 1
                params = module.weight.numel()
                total_params += params
                quantized_params += params
    
    print(f"\nQuantization Statistics:")
    print(f"  Total Conv2d layers: {total_conv}")
    print(f"  Quantized layers: {quantized_conv}")
    print(f"  Full precision layers: {total_conv - quantized_conv}")
    
    if total_conv > 0:
        print(f"  Quantization ratio: {quantized_conv}/{total_conv} ({100*quantized_conv/total_conv:.1f}%)")
    
    print(f"  Total conv parameters: {total_params:,}")
    print(f"  Quantized parameters: {quantized_params:,}")
    
    if total_params > 0:
        print(f"  Quantized parameter ratio: {100*quantized_params/total_params:.1f}%")
    
    if quantized_params > 0:
        # Calculate compression
        # Full precision: 32 bits per weight
        # Ternary: ~2 bits per weight (3 values: -Wn, 0, +Wp)
        # Plus scaling factors (Wp, Wn) per layer: negligible
        compression_ratio = 32 / 2
        print(f"  Theoretical compression: ~{compression_ratio:.0f}x for quantized layers")
        
        # Actual model size estimate
        fp_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        ttq_size_mb = ((total_params - quantized_params) * 4 + quantized_params * 0.25) / (1024 * 1024)
        print(f"  Estimated model size: {fp_size_mb:.2f} MB (FP32) → {ttq_size_mb:.2f} MB (TTQ)")
        print(f"  Overall compression: {fp_size_mb/ttq_size_mb:.2f}x")
    
    print(f"\nExcluded layers:")
    print(f"  ✗ Detect head (model.23.*)")
    print(f"  ✗ C2PSA attention (model.10.*)")
    print(f"  ✗ First conv layer (model.0.*)  [optional]")


def get_quantized_layers(model):
    """Get list of all quantized layer names"""
    quantized = []
    for name, module in model.named_modules():
        if isinstance(module, TTQConv2dWithGrad):
            quantized.append(name)
    return quantized
