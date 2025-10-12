import torch
import torch.nn as nn


def quantize_layer_ternary(model, layer_name, threshold_ratio=0.7):
    """
    Apply heuristic ternary quantization to a specific layer for sensitivity analysis.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer to quantize
        threshold_ratio: Ratio for computing threshold (default: 0.7)
                        Higher value = more weights pushed to zero
    
    This function quantizes weights to {-α, 0, +α} where:
    - Δ (threshold) = threshold_ratio * mean(|weights|)
    - α (scaling factor) minimizes reconstruction error
    """
    # Find the target layer
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
    
    if target_module is None:
        return  # Layer not found, skip
    
    # Only quantize layers with weights (Conv, Linear, etc.)
    if not hasattr(target_module, 'weight') or target_module.weight is None:
        return
    
    with torch.no_grad():
        original_weight = target_module.weight.data.clone()
        
        # Compute threshold using mean absolute value heuristic (TWN-style)
        # Δ = threshold_ratio * mean(|W|)
        threshold = threshold_ratio * torch.mean(torch.abs(original_weight))
        
        # Ternarize: {-1, 0, +1}
        ternary_weight = torch.zeros_like(original_weight)
        ternary_weight[original_weight > threshold] = 1.0
        ternary_weight[original_weight < -threshold] = -1.0
        # Weights in [-threshold, threshold] become 0
        
        # Compute optimal scaling factor α
        # α* = argmin ||W - α * W_ternary||^2
        # Closed form: α = W^T * W_ternary / (W_ternary^T * W_ternary)
        
        non_zero_mask = (ternary_weight != 0)
        if non_zero_mask.sum() > 0:
            # Compute scaling factor
            numerator = (original_weight * ternary_weight).sum()
            denominator = (ternary_weight * ternary_weight).sum()
            alpha = numerator / (denominator + 1e-8)
        else:
            alpha = 1.0  # If all zeros, use default
        
        # Apply quantized weights: W_quant = α * W_ternary
        quantized_weight = alpha * ternary_weight
        
        # Replace the layer's weights
        target_module.weight.data = quantized_weight


def quantize_layer_simple(model, layer_name):
    """
    Simpler sensitivity test: set layer weights to zero.
    This gives maximum sensitivity (worst case scenario).
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer to zero out
    """
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, 'weight'):
            with torch.no_grad():
                module.weight.data.zero_()
            break


def add_noise_to_layer(model, layer_name, noise_std=0.1):
    """
    Add Gaussian noise to layer weights for sensitivity testing.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer
        noise_std: Standard deviation of Gaussian noise
    """
    for name, module in model.named_modules():
        if name == layer_name and hasattr(module, 'weight'):
            with torch.no_grad():
                noise = torch.randn_like(module.weight) * noise_std
                module.weight.data += noise
            break
