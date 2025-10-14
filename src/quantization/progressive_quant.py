import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizationFunction(torch.autograd.Function):
    """
    Straight-Through Estimator for quantization.
    Forward: quantize weights
    Backward: pass gradients through unchanged
    """
    @staticmethod
    def forward(ctx, input, scale, bitwidth):
        if bitwidth == 'ternary':
            # Ternary: {-1, 0, +1}
            threshold = 0.7 * input.abs().mean()
            output = torch.zeros_like(input)
            output[input > threshold] = 1.0
            output[input < -threshold] = -1.0
        else:
            # Uniform quantization for 2, 4, 8 bits
            n_levels = 2 ** bitwidth
            qmin = -(n_levels // 2)
            qmax = (n_levels // 2) - 1
            # Quantize
            output = torch.clamp(torch.round(input / scale), qmin, qmax)
            # Dequantize
            output = output * scale
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradient passes unchanged
        return grad_output, None, None


class ProgressiveQuantizer(nn.Module):
    """
    Learnable quantizer that supports multiple bitwidths.
    Can transition from high precision (8-bit) to ternary.
    """
    def __init__(self, weight_shape, init_bitwidth=8):
        super().__init__()
        self.bitwidth = init_bitwidth
        
        # Learnable scaling factor (per-channel for better accuracy)
        if len(weight_shape) == 4:  # Conv2d: [out_ch, in_ch, h, w]
            scale_shape = (weight_shape[0], 1, 1, 1)
        else:  # Linear: [out_features, in_features]
            scale_shape = (weight_shape[0], 1)
        
        self.scale = nn.Parameter(torch.ones(scale_shape))
        
        # For ternary: learnable positive and negative weights
        self.W_p = nn.Parameter(torch.ones(1))
        self.W_n = nn.Parameter(torch.ones(1))
        
    def forward(self, weight):
        if self.bitwidth == 'ternary':
            # Ternary quantization with learnable scaling
            threshold = 0.7 * weight.abs().mean()
            ternary_mask = torch.zeros_like(weight)
            ternary_mask[weight > threshold] = 1.0
            ternary_mask[weight < -threshold] = -1.0
            
            # Apply learned scaling factors
            quantized = torch.where(
                ternary_mask > 0,
                self.W_p.abs(),
                torch.where(
                    ternary_mask < 0,
                    -self.W_n.abs(),
                    torch.zeros_like(weight)
                )
            )
        else:
            # Standard uniform quantization
            quantized = QuantizationFunction.apply(weight, self.scale.abs(), self.bitwidth)
        
        # Straight-through estimator
        return weight + (quantized - weight).detach()
    
    def set_bitwidth(self, bitwidth):
        """Change bitwidth on-the-fly"""
        self.bitwidth = bitwidth


class QuantizedConv2d(nn.Conv2d):
    """
    Conv2d with progressive quantization support.
    Weights can be quantized at different bitwidths.
    """
    def __init__(self, *args, init_bitwidth=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = ProgressiveQuantizer(self.weight.shape, init_bitwidth)
        self.current_bitwidth = init_bitwidth
        
    def forward(self, x):
        # Quantize weights during forward pass
        quantized_weight = self.quantizer(self.weight)
        return F.conv2d(x, quantized_weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
    
    def set_bitwidth(self, bitwidth):
        """Update quantization bitwidth"""
        self.current_bitwidth = bitwidth
        self.quantizer.set_bitwidth(bitwidth)


class QuantizedLinear(nn.Linear):
    """
    Linear layer with progressive quantization support.
    """
    def __init__(self, *args, init_bitwidth=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = ProgressiveQuantizer(self.weight.shape, init_bitwidth)
        self.current_bitwidth = init_bitwidth
        
    def forward(self, x):
        quantized_weight = self.quantizer(self.weight)
        return F.linear(x, quantized_weight, self.bias)
    
    def set_bitwidth(self, bitwidth):
        self.current_bitwidth = bitwidth
        self.quantizer.set_bitwidth(bitwidth)


def get_parent_modules(model, target_path):
    """
    Get all parent modules along a given path.
    Returns list of (name, module) tuples from root to target.
    
    Example: For path 'model.23.cv2.0.conv', returns:
    [('model', Sequential), ('model.23', Detect), ('model.23.cv2', ModuleList), ...]
    """
    parts = target_path.split('.')
    parents = []
    current_module = model
    current_path = ''
    
    for part in parts:
        if current_path:
            current_path += '.' + part
        else:
            current_path = part
        
        # Navigate to child
        if hasattr(current_module, part):
            current_module = getattr(current_module, part)
        elif hasattr(current_module, '_modules') and part in current_module._modules:
            current_module = current_module._modules[part]
        elif part.isdigit() and hasattr(current_module, '__getitem__'):
            current_module = current_module[int(part)]
        else:
            break
        
        parents.append((current_path, current_module))
    
    return parents

def is_in_detection_head(model, module_path):
    """
    Check if a module is part of the YOLO detection head.
    Detection head is identified by the 'Detect' class, not by layer number.
    """
    parents = get_parent_modules(model, module_path)
    
    for path, parent_module in parents:
        class_name = parent_module.__class__.__name__
        if class_name == 'Detect':
            return True
    
    return False


def is_in_attention_block(model, module_path):
    """
    Check if a module is inside a C2PSA (attention) block.
    C2PSA is identified by class name, not layer number.
    Skip only attention components (attn.*), allow FFN by default.
    """
    parents = get_parent_modules(model, module_path)
    
    for path, parent_module in parents:
        class_name = parent_module.__class__.__name__
        
        # Found C2PSA parent
        if class_name == 'C2PSA':
            # Check if this is in the attention submodule
            if '.attn.' in module_path or module_path.endswith('.attn'):
                return True
            # Uncomment to skip all C2PSA layers (including FFN):
            # return True
    
    return False


def should_skip_layer(model, module_path, conv_idx, total_conv, skip_first_last):
    """
    Determine if a layer should be skipped from quantization.
    
    Args:
        model: Full YOLO model (needed to traverse hierarchy)
        module_path: Full path to the module (e.g., 'model.0.conv')
        conv_idx: Index of this conv layer (1-indexed)
        total_conv: Total number of conv layers
        skip_first_last: Whether to skip first and last conv
    
    Returns:
        (should_skip, reason) tuple
    """
    # 1. Check first/last layer
    if skip_first_last and (conv_idx == 1 or conv_idx == total_conv):
        return True, "first/last layer"
    
    # 2. Check if in detection head (Detect module)
    if is_in_detection_head(model, module_path):
        return True, "detection head (Detect)"
    
    # 3. Check if in attention block (C2PSA.attn)
    if is_in_attention_block(model, module_path):
        return True, "C2PSA attention block"
    
    return False, None


def convert_to_progressive_quant(model, init_bitwidth=8, skip_first_last=True):
    """
    Convert model layers to progressive quantized versions.
    Skips Detect module and C2PSA attention blocks based on class names.
    
    Args:
        model: YOLO11 model
        init_bitwidth: Starting bitwidth (8 for progressive training)
        skip_first_last: Skip first and last conv layers (more sensitive)
    
    Returns:
        Model with quantized layers
    """
    conv_count = 0
    converted_count = 0
    skipped_layers = {
        'first/last layer': 0,
        'detection head (Detect)': 0,
        'C2PSA attention block': 0
    }
    
    # Count total conv layers
    total_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    
    print(f"\nTotal Conv2d layers in model: {total_conv}")
    print(f"Starting conversion with bitwidth: {init_bitwidth}")
    print(f"Skipping: Detect module, C2PSA.attn blocks, first/last layers\n")
    
    def replace_layer(module, parent_path=''):
        nonlocal conv_count, converted_count
        
        for child_name, child in list(module.named_children()):
            current_path = f"{parent_path}.{child_name}" if parent_path else child_name
            
            if isinstance(child, nn.Conv2d):
                conv_count += 1
                
                # Determine if should skip this layer
                should_skip, skip_reason = should_skip_layer(
                    model,  # Pass full model for hierarchy traversal
                    current_path,
                    conv_count,
                    total_conv,
                    skip_first_last
                )
                
                if should_skip:
                    skipped_layers[skip_reason] += 1
                    print(f"⊗ Skipped {current_path:50s} [{skip_reason}]")
                    continue
                
                # Create quantized replacement
                quant_conv = QuantizedConv2d(
                    child.in_channels, child.out_channels,
                    child.kernel_size, child.stride, child.padding,
                    child.dilation, child.groups,
                    child.bias is not None,
                    init_bitwidth=init_bitwidth
                )
                
                # Copy pre-trained weights (detach from inference mode)
                with torch.no_grad():
                    quant_conv.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        quant_conv.bias.data.copy_(child.bias.data)

                # Explicitly enable gradients
                quant_conv.weight.requires_grad_(True)
                if quant_conv.bias is not None:
                    quant_conv.bias.requires_grad_(True)
                
                setattr(module, child_name, quant_conv)
                converted_count += 1
                print(f"✓ Converted {current_path:50s} to {init_bitwidth}-bit")
                
            elif isinstance(child, nn.Linear):
                # Also check if Linear is in head or attention
                should_skip, skip_reason = should_skip_layer(
                    model,
                    current_path,
                    conv_count,
                    total_conv,
                    False  # Don't apply first/last logic to Linear
                )
                
                if should_skip:
                    skipped_layers[skip_reason] += 1
                    print(f"⊗ Skipped {current_path:50s} [{skip_reason}]")
                    continue
                
                quant_linear = QuantizedLinear(
                    child.in_features, child.out_features,
                    child.bias is not None,
                    init_bitwidth=init_bitwidth
                )
                # Copy pre-trained weights (detach from inference mode)
                with torch.no_grad():
                    quant_linear.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        quant_linear.bias.data.copy_(child.bias.data)

                # Explicitly enable gradients
                quant_linear.weight.requires_grad_(True)
                if quant_linear.bias is not None:
                    quant_linear.bias.requires_grad_(True)
                
                setattr(module, child_name, quant_linear)
                converted_count += 1
                print(f"✓ Converted {current_path:50s} to {init_bitwidth}-bit")
            else:
                # Recursively process child modules
                replace_layer(child, current_path)
    
    replace_layer(model)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Conversion Summary:")
    print(f"  Total Conv2d layers:          {total_conv}")
    print(f"  Converted to quantized:       {converted_count}")
    print(f"  Skipped (first/last):         {skipped_layers['first/last layer']}")
    print(f"  Skipped (Detect head):        {skipped_layers['detection head (Detect)']}")
    print(f"  Skipped (C2PSA attention):    {skipped_layers['C2PSA attention block']}")
    print(f"{'='*80}\n")
    
    return model


def set_model_bitwidth(model, bitwidth):
    """
    Change bitwidth for all quantized layers in the model.
    
    Args:
        model: Model with quantized layers
        bitwidth: Target bitwidth (8, 4, 2, or 'ternary')
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            module.set_bitwidth(bitwidth)
            count += 1
    
    print(f"\n{'='*60}")
    print(f"Updated {count} layers to {bitwidth}-bit quantization")
    print(f"{'='*60}\n")


def enable_quantized_training(model):
    """
    Enable training mode for the model.
    Assumes parameters already have requires_grad=True from conversion.
    """
    model.train()  # Set to training mode
    
    # Count layers
    quant_count = sum(1 for m in model.modules() if isinstance(m, (QuantizedConv2d, QuantizedLinear)))
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model in training mode: {quant_count} quantized layers, {total_params} trainable parameters")
    return model