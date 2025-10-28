"""
Utilities to convert YOLO11 model to TTQ version
UPDATED: Supports progressive quantization + skips 1x1 convolutions
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from .ttq_layer import TTQConv2dWithGrad


def should_quantize_module(module_name, module, quantize_first_layer=False, target_layers=None):
    """
    Determine if a module should be quantized.
    
    CRITICAL: Skip 1x1 convolutions - they're too sensitive to quantization!
    
    Args:
        module_name: Full name of the module
        module: The module itself
        quantize_first_layer: Whether to quantize first conv layer
        target_layers: List of layer indices to quantize (e.g., [2, 3, 4])
                      If None, quantize all eligible layers (full quantization)
    
    Returns:
        bool: True if module should be quantized
    """
    # Skip if not a Conv2d layer
    if not isinstance(module, nn.Conv2d):
        return False
    
    # CRITICAL FIX: Skip 1x1 convolutions (they cause 79% mAP drop!)
    # 1x1 convs are used for channel projection and are very sensitive
    kernel_size = module.kernel_size
    if isinstance(kernel_size, tuple):
        if kernel_size[0] == 1 and kernel_size[1] == 1:
            return False
    elif kernel_size == 1:
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
    
    # PROGRESSIVE QUANTIZATION: Only quantize target layers if specified
    if target_layers is not None:
        try:
            layer_num = int(layer_idx)
            if layer_num not in target_layers:
                return False
        except ValueError:
            # If layer_idx is not a number, skip it
            return False
    
    return True


def replace_conv_with_ttq(module, name, threshold=0.7, quantize_first_layer=False, target_layers=None):
    """
    Recursively replace Conv2d layers with TTQ quantized versions.
    
    UPDATED: Supports progressive quantization + detailed skip reasons
    
    Args:
        module: PyTorch module
        name: Module name
        threshold: Threshold factor for quantization (default: 0.7 for mean-based)
        quantize_first_layer: Whether to quantize first layer
        target_layers: List of layer indices to quantize (progressive mode)
    
    Returns:
        Number of quantized layers
    """
    quantized_count = 0
    
    for child_name, child_module in list(module.named_children()):
        full_name = f"{name}.{child_name}" if name else child_name
        
        if isinstance(child_module, nn.Conv2d):
            # Check kernel size for skip reason
            k = child_module.kernel_size
            is_1x1 = (isinstance(k, tuple) and k[0] == 1 and k[1] == 1) or (k == 1)
            
            if should_quantize_module(full_name, child_module, quantize_first_layer, target_layers):
                # Extract pretrained weights BEFORE creating TTQ layer
                pretrained_weight = child_module.weight.data.clone()
                pretrained_bias = child_module.bias.data.clone() if child_module.bias is not None else None
                
                # Create TTQ layer with pretrained weights for proper initialization
                ttq_layer = TTQConv2dWithGrad(
                    in_channels=child_module.in_channels,
                    out_channels=child_module.out_channels,
                    kernel_size=child_module.kernel_size,
                    stride=child_module.stride,
                    padding=child_module.padding,
                    dilation=child_module.dilation,
                    groups=child_module.groups,
                    bias=child_module.bias is not None,
                    threshold=threshold,
                    pretrained_weight=pretrained_weight
                )
                
                # Copy bias separately
                if pretrained_bias is not None:
                    ttq_layer.bias.data.copy_(pretrained_bias)
                
                # Replace the layer
                setattr(module, child_name, ttq_layer)
                
                # Show kernel size in quantization message
                k_str = f"{k[0]}x{k[1]}" if isinstance(k, tuple) else f"{k}x{k}"
                print(f"  ✓ Quantized ({k_str}): {full_name} (Wp={ttq_layer.Wp:.4f}, Wn={ttq_layer.Wn:.4f})")
                quantized_count += 1
            else:
                # Determine reason for skipping
                if is_1x1:
                    print(f"  Skipping (1x1 conv - too sensitive): {full_name}")
                else:
                    parts = full_name.split('.')
                    if len(parts) >= 3 and parts[1] == 'model':
                        layer_idx = parts[2]
                        if layer_idx == '23':
                            print(f"  Skipping (Detect head): {full_name}")
                        elif layer_idx == '10':
                            print(f"  Skipping (C2PSA attention): {full_name}")
                        elif layer_idx == '0':
                            print(f"  Skipping (First layer): {full_name}")
                        elif target_layers is not None:
                            try:
                                if int(layer_idx) not in target_layers:
                                    k_str = f"{k[0]}x{k[1]}" if isinstance(k, tuple) else f"{k}x{k}"
                                    print(f"  Skipping ({k_str}, Progressive - not in target): {full_name}")
                            except ValueError:
                                pass
        else:
            # Recursively process child modules
            quantized_count += replace_conv_with_ttq(
                child_module, full_name, threshold, quantize_first_layer, target_layers
            )
    
    return quantized_count


def quantize_yolo_model(model_path, threshold=0.7, quantize_first_layer=False, 
                        target_layers=None, verbose=True):
    """
    Load a YOLO11 model and convert Conv2d layers to TTQ.
    
    UPDATED: Supports progressive quantization
    
    Args:
        model_path: Path to YOLO11 model (.pt file or 'yolo11n.pt', etc.)
        threshold: Threshold factor for quantization (default: 0.7 for mean-based)
        quantize_first_layer: Whether to quantize first conv layer
        target_layers: List of layer indices to quantize (e.g., [2, 3, 4])
                      If None, quantize all eligible layers (full quantization)
        verbose: Print quantization information
    
    Returns:
        Quantized YOLO model
    """
    # Load YOLO model
    model = YOLO(model_path)
    
    if verbose:
        print(f"\nLoading YOLO11 model from: {model_path}")
        print("="*60)
        
        if target_layers is not None:
            print(f"\n⚡ PROGRESSIVE QUANTIZATION MODE")
            print(f"Target layers: {target_layers}")
            print(f"All other layers remain in FP32")
        else:
            print(f"\n⚡ FULL QUANTIZATION MODE")
            print(f"Quantizing all eligible layers")
        print("="*60)
    
    # Access the actual PyTorch model
    pytorch_model = model.model
    
    # Replace Conv2d with TTQ layers
    if verbose:
        print("\nQuantizing Conv2d layers to TTQ...")
        print("="*60)
        print("Exclusions:")
        print("  ✗ 1x1 convolutions (too sensitive)")
        print("  ✗ Detect head (model.23.*)")
        print("  ✗ C2PSA attention (model.10.*)")
        print("="*60)
    
    quantized_count = replace_conv_with_ttq(
        pytorch_model, "model", threshold, quantize_first_layer, target_layers
    )
    
    if verbose:
        print("\n" + "="*60)
        print("Quantization complete!")
        print_quantization_stats(pytorch_model)
        print_wp_wn_initialization_stats(pytorch_model)
    
    return model


def print_wp_wn_initialization_stats(model):
    """Print statistics about initialized Wp/Wn values"""
    wp_vals = []
    wn_vals = []
    
    for module in model.modules():
        if isinstance(module, TTQConv2dWithGrad):
            wp_vals.append(module.Wp)
            wn_vals.append(module.Wn)
    
    if wp_vals:
        import numpy as np
        print(f"\nInitialized Wp/Wn Statistics (from pretrained weights):")
        print(f"  Wp: min={min(wp_vals):.4f}, max={max(wp_vals):.4f}, mean={np.mean(wp_vals):.4f}")
        print(f"  Wn: min={min(wn_vals):.4f}, max={max(wn_vals):.4f}, mean={np.mean(wn_vals):.4f}")
        print(f"  ✓ Scales initialized based on pretrained weight statistics")


def print_quantization_stats(model):
    """Print statistics about quantized model"""
    total_conv = 0
    quantized_conv = 0
    conv_1x1 = 0
    total_params = 0
    quantized_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, TTQConv2dWithGrad)):
            # Check if 1x1
            k = module.kernel_size
            is_1x1 = (isinstance(k, tuple) and k[0] == 1 and k[1] == 1) or (k == 1)
            
            if isinstance(module, nn.Conv2d) and not isinstance(module, TTQConv2dWithGrad):
                total_conv += 1
                if is_1x1:
                    conv_1x1 += 1
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
    print(f"    ├─ 1x1 convolutions (skipped): {conv_1x1}")
    print(f"    └─ Other exclusions: {total_conv - quantized_conv - conv_1x1}")
    
    if total_conv > 0:
        print(f"  Quantization ratio: {quantized_conv}/{total_conv} ({100*quantized_conv/total_conv:.1f}%)")
    
    print(f"  Total conv parameters: {total_params:,}")
    print(f"  Quantized parameters: {quantized_params:,}")
    
    if total_params > 0:
        print(f"  Quantized parameter ratio: {100*quantized_params/total_params:.1f}%")
    
    if quantized_params > 0:
        compression_ratio = 32 / 2
        print(f"  Theoretical compression: ~{compression_ratio:.0f}x for quantized layers")
        
        fp_size_mb = (total_params * 4) / (1024 * 1024)
        ttq_size_mb = ((total_params - quantized_params) * 4 + quantized_params * 0.25) / (1024 * 1024)
        print(f"  Estimated model size: {fp_size_mb:.2f} MB (FP32) → {ttq_size_mb:.2f} MB (TTQ)")
        print(f"  Overall compression: {fp_size_mb/ttq_size_mb:.2f}x")
    
    print(f"\nExcluded layers:")
    print(f"  ✗ 1x1 convolutions (channel projection - too sensitive)")
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
