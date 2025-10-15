
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

# ==============================================================================
# 1. Core Quantization Functions & Modules
# ==============================================================================

class TernaryQuantizationFunction(torch.autograd.Function):
    """
    Straight-Through Estimator for ternary quantization.
    Forward: ternarize weights to {-W_n, 0, W_p}
    Backward: pass gradients through unchanged
    """
    @staticmethod
    def forward(ctx, weight, threshold, w_p, w_n):
        """
        Ternarizes the input weight tensor.
        
        Args:
            weight (torch.Tensor): The input weight tensor.
            threshold (torch.Tensor): The learnable threshold for quantization.
            w_p (torch.Tensor): The learnable positive scaling factor.
            w_n (torch.Tensor): The learnable negative scaling factor.
            
        Returns:
            torch.Tensor: The ternarized weight tensor.
        """
        ternary_mask = torch.zeros_like(weight)
        ternary_mask[weight > threshold] = 1.0
        ternary_mask[weight < -threshold] = -1.0
        
        quantized_weight = torch.where(
            ternary_mask > 0,
            w_p,
            torch.where(
                ternary_mask < 0,
                -w_n,
                torch.zeros_like(weight)
            )
        )
        return quantized_weight

    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight-through estimator: gradient passes unchanged.
        """
        return grad_output, None, None, None


class UniformQuantizationFunction(torch.autograd.Function):
    """
    Straight-Through Estimator for uniform quantization.
    Forward: quantize weights to a specified bitwidth
    Backward: pass gradients through unchanged
    """
    @staticmethod
    def forward(ctx, weight, scale, bitwidth):
        """
        Performs uniform quantization on the input weight tensor.
        
        Args:
            weight (torch.Tensor): The input weight tensor.
            scale (torch.Tensor): The learnable scaling factor.
            bitwidth (int): The target bitwidth for quantization.
            
        Returns:
            torch.Tensor: The quantized and de-quantized weight tensor.
        """
        n_levels = 2 ** bitwidth
        qmin = -(n_levels // 2)
        qmax = (n_levels // 2) - 1
        
        # Quantize
        quantized = torch.clamp(torch.round(weight / scale), qmin, qmax)
        # De-quantize
        dequantized = quantized * scale
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight-through estimator: gradient passes unchanged.
        """
        return grad_output, None, None


class ProgressiveQuantizer(nn.Module):
    """
    A learnable quantizer that supports progressive quantization from high-bitwidth
    down to ternary. It uses learnable scaling factors for uniform quantization
    and learnable thresholds and weights for ternary quantization.
    """
    def __init__(self, weight_shape, init_bitwidth=8):
        super().__init__()
        self.bitwidth = init_bitwidth
        
        # --- Learnable parameters for UNIFORM quantization ---
        # Per-channel scaling factor for better accuracy in Conv layers
        if len(weight_shape) == 4:  # Conv2d: [out_ch, in_ch, h, w]
            scale_shape = (weight_shape[0], 1, 1, 1)
        else:  # Linear: [out_features, in_features]
            scale_shape = (weight_shape[0], 1)
        self.uniform_scale = nn.Parameter(torch.ones(scale_shape))
        
        # --- Learnable parameters for TERNARY quantization ---
        # A single learnable threshold for the entire layer
        self.ternary_threshold = nn.Parameter(torch.tensor(0.1)) 
        # Learnable positive and negative weights
        self.w_p = nn.Parameter(torch.tensor(1.0))
        self.w_n = nn.Parameter(torch.tensor(1.0))

    def forward(self, weight):
        """
        Applies quantization using the Straight-Through Estimator (STE).
        The STE allows gradients to flow back to the original weights.
        """
        if self.bitwidth == 'ternary':
            # Use the ternary quantization function
            quantized_weight = TernaryQuantizationFunction.apply(
                weight, self.ternary_threshold.abs(), self.w_p.abs(), self.w_n.abs()
            )
        else:
            # Use the uniform quantization function
            quantized_weight = UniformQuantizationFunction.apply(
                weight, self.uniform_scale.abs(), self.bitwidth
            )
        
        # STE: return original weight in backward pass, quantized in forward
        return weight + (quantized_weight - weight).detach()

    def set_bitwidth(self, bitwidth):
        """Updates the bitwidth for the next quantization stage."""
        self.bitwidth = bitwidth

# ==============================================================================
# 2. Quantized Layer Wrappers
# ==============================================================================

class QuantizedLayer(nn.Module):
    """
    Base class for quantized layers (Conv, Linear).
    Handles the creation and management of the quantizer.
    """
    def __init__(self, *args, init_bitwidth=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = ProgressiveQuantizer(self.weight.shape, init_bitwidth)
        self.current_bitwidth = init_bitwidth
        
    def set_bitwidth(self, bitwidth):
        """Propagates the bitwidth change to the quantizer."""
        self.current_bitwidth = bitwidth
        self.quantizer.set_bitwidth(bitwidth)


class QuantizedConv2d(QuantizedLayer, nn.Conv2d):
    """
    A Conv2d layer with built-in progressive quantization support.
    """
    def __init__(self, *args, init_bitwidth=8, **kwargs):
        # The order of super().__init__ calls matters here.
        # Call nn.Conv2d first to initialize weights, etc.
        nn.Conv2d.__init__(self, *args, **kwargs)
        # Then call QuantizedLayer to create the quantizer
        QuantizedLayer.__init__(self, init_bitwidth=init_bitwidth)
        
    def forward(self, x):
        """Quantizes weights before the convolution operation."""
        quantized_weight = self.quantizer(self.weight)
        return F.conv2d(x, quantized_weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)


class QuantizedLinear(QuantizedLayer, nn.Linear):
    """
    A Linear layer with built-in progressive quantization support.
    """
    def __init__(self, *args, init_bitwidth=8, **kwargs):
        # Call nn.Linear first
        nn.Linear.__init__(self, *args, **kwargs)
        # Then call QuantizedLayer
        QuantizedLayer.__init__(self, init_bitwidth=init_bitwidth)
        
    def forward(self, x):
        """Quantizes weights before the linear operation."""
        quantized_weight = self.quantizer(self.weight)
        return F.linear(x, quantized_weight, self.bias)

# ==============================================================================
# 3. Quantization Manager & Model Wrapper
# ==============================================================================

class QuantizationManager:
    """
    Handles the logic of converting a standard YOLO model into a quantized
    version. It identifies target layers and replaces them with their
    quantized counterparts based on a defined configuration.
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.quantized_layers = []

    def convert_to_quantized(self):
        """
        Replaces standard layers with quantized versions based on config rules.
        """
        init_bitwidth = self.config['progressive']['bitwidth_schedule'][0]
        
        # Recursively replace layers
        self._replace_layer(self.model.model, init_bitwidth)
        
        self._log_conversion_summary()

    def _replace_layer(self, module, init_bitwidth, parent_path=''):
        for child_name, child in list(module.named_children()):
            current_path = f"{parent_path}.{child_name}" if parent_path else child_name

            if self._should_quantize(child, current_path):
                quant_layer = None
                if isinstance(child, nn.Conv2d):
                    quant_layer = QuantizedConv2d(
                        child.in_channels, child.out_channels,
                        child.kernel_size, child.stride, child.padding,
                        child.dilation, child.groups, child.bias is not None,
                        init_bitwidth=init_bitwidth
                    )
                    quant_layer.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        quant_layer.bias.data.copy_(child.bias.data)
                
                elif isinstance(child, nn.Linear):
                    quant_layer = QuantizedLinear(
                        child.in_features, child.out_features,
                        child.bias is not None,
                        init_bitwidth=init_bitwidth
                    )
                    quant_layer.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        quant_layer.bias.data.copy_(child.bias.data)

                if quant_layer:
                    setattr(module, child_name, quant_layer)
                    self.quantized_layers.append(quant_layer)
            else:
                # Recurse into child module
                self._replace_layer(child, init_bitwidth, current_path)

    def _should_quantize(self, layer, layer_path):
        """
        Determines if a layer should be quantized based on rules in the config.
        """
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            return False
            
        # Check against exclusion rules from the config
        # (This part can be expanded with more sophisticated rules)
        if 'attention' in layer_path or 'detect' in layer_path:
            return False
            
        return True

    def _log_conversion_summary(self):
        total_quantized = len(self.quantized_layers)
        print("\n--- Quantization Conversion Summary ---")
        print(f"Successfully converted {total_quantized} layers to quantized versions.")
        # Add more details here if needed, e.g., skipped layers
        print("-------------------------------------\\n")

    def set_bitwidth(self, bitwidth):
        """Sets the bitwidth for all managed quantized layers."""
        print(f"\n{'='*60}\nUpdating all quantized layers to {bitwidth}-bit\n{'='*60}\n")
        for layer in self.quantized_layers:
            layer.set_bitwidth(bitwidth)

    def set_trainable_parameters(self, stage='quantizer'):
        """
        Freezes or unfreezes model parameters based on the training stage.
        - 'quantizer': Only train quantizer-specific parameters (scales, thresholds).
        - 'full': Train all weights and quantizer parameters.
        """
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze based on the stage
        if stage == 'quantizer':
            for layer in self.quantized_layers:
                for param in layer.quantizer.parameters():
                    param.requires_grad = True
        elif stage == 'full':
            for layer in self.quantized_layers:
                for param in layer.parameters(): # Includes weights, bias, and quantizer params
                    param.requires_grad = True
        else:
            raise ValueError(f"Unknown training stage: {stage}")


class QuantizedYOLOModel:
    """
    A wrapper for a YOLO model that integrates the QuantizationManager to
    facilitate progressive quantization training.
    """
    def __init__(self, model_path, config):
        self.yolo = YOLO(model_path)
        self.config = config
        self.manager = QuantizationManager(self.yolo, config)
        
        # Convert the model to its quantized version upon initialization
        self.manager.convert_to_quantized()

    def __getattr__(self, name):
        """Delegates calls to the underlying YOLO object."""
        return getattr(self.yolo, name)

    def set_bitwidth(self, bitwidth):
        """Proxy to the manager's set_bitwidth method."""
        self.manager.set_bitwidth(bitwidth)

    def set_trainable_parameters(self, stage):
        """Proxy to the manager's set_trainable_parameters method."""
        self.manager.set_trainable_parameters(stage)
