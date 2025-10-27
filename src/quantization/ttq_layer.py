"""
Trained Ternary Quantization (TTQ) Layer - Exact Paper Implementation
Based on "Trained Ternary Quantization" (Zhu et al., ICLR 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTQConv2d(nn.Module):
    """
    Ternary quantized Conv2d layer with learnable scaling factors.
    Maintains full-precision weights, quantizes during forward pass.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, threshold=0.05):
        super(TTQConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.threshold = threshold
        
        # Full precision latent weights (never quantized in storage)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Learnable scaling factors
        self.Wp = nn.Parameter(torch.Tensor(1))
        self.Wn = nn.Parameter(torch.Tensor(1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        # Initialize scaling factors to 1.0
        nn.init.constant_(self.Wp, 1.0)
        nn.init.constant_(self.Wn, 1.0)
    
    def quantize_weight(self, weight):
        """
        Quantize full precision weights to ternary values {-Wn, 0, +Wp}
        Used for inference/deployment only (not during training)
        """
        delta = self.threshold * weight.abs().max()
        
        weight_quantized = torch.zeros_like(weight)
        weight_quantized[weight > delta] = self.Wp
        weight_quantized[weight < -delta] = -self.Wn
        
        return weight_quantized, delta
    
    def forward(self, x):
        """
        Forward pass: Use quantized weights via custom autograd function
        """
        # Quantize weights on-the-fly (TTQWeightFunction handles gradients)
        weight_q = TTQWeightFunction.apply(
            self.weight, self.Wp, self.Wn, self.threshold
        )
        
        # Use quantized weights for convolution
        return F.conv2d(x, weight_q, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
    
    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'threshold={self.threshold}, Wp={self.Wp.item():.4f}, Wn={self.Wn.item():.4f}')


# Alias for compatibility
TTQConv2dWithGrad = TTQConv2d


class TTQWeightFunction(torch.autograd.Function):
    """
    Custom autograd function implementing TTQ gradient computation
    
    Forward: Quantize W -> W^t = {-Wn, 0, +Wp}
    Backward: Compute gradients for W, Wp, and Wn
    """
    
    @staticmethod
    def forward(ctx, weight, Wp, Wn, threshold):
        """Forward: Quantize full-precision weights to ternary values"""
        # Calculate threshold delta
        delta = threshold * weight.abs().max()
        
        # Create masks for three regions
        pos_mask = weight > delta
        neg_mask = weight < -delta
        
        # Quantize to ternary values
        weight_quantized = torch.zeros_like(weight)
        weight_quantized[pos_mask] = Wp
        weight_quantized[neg_mask] = -Wn
        
        # Save for backward
        ctx.save_for_backward(weight, Wp, Wn)
        ctx.pos_mask = pos_mask
        ctx.neg_mask = neg_mask
        
        return weight_quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: Compute gradients for W, Wp, and Wn
        
        From TTQ paper Equations (2) and (3):
        - ∂L/∂W_p = Σ_{i∈I_p} ∂L/∂w^t(i)
        - ∂L/∂W_n = Σ_{i∈I_n} ∂L/∂w^t(i) 
        - ∂L/∂w̃ scaled by W_p, W_n, or 1 depending on region
        """
        weight, Wp, Wn = ctx.saved_tensors
        pos_mask = ctx.pos_mask
        neg_mask = ctx.neg_mask
        
        # Gradient w.r.t. full-precision weights (Equation 8 from paper)
        grad_weight = grad_output.clone()
        
        # Positive region: scale by Wp
        if pos_mask.any():
            grad_weight = torch.where(pos_mask, grad_output * Wp, grad_weight)
        
        # Negative region: scale by Wn
        if neg_mask.any():
            grad_weight = torch.where(neg_mask, grad_output * Wn, grad_weight)
        
        # Zero region: gradient passes through (multiplied by 1)
        
        # Gradient w.r.t. Wp (Equation 7 from paper)
        if pos_mask.any():
            grad_Wp = grad_output[pos_mask].sum().reshape(1)
        else:
            grad_Wp = torch.zeros(1, dtype=grad_output.dtype, device=grad_output.device)
        
        # Gradient w.r.t. Wn (Equation 7 from paper)
        # Note: In the paper, gradient is same sign because weight is -W_n
        # So ∂L/∂W_n = -Σ ∂L/∂w^t where w^t = -W_n
        if neg_mask.any():
            grad_Wn = -grad_output[neg_mask].sum().reshape(1)
        else:
            grad_Wn = torch.zeros(1, dtype=grad_output.dtype, device=grad_output.device)
        
        return grad_weight, grad_Wp, grad_Wn, None

