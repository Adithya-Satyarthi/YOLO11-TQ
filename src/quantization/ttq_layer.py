"""
Trained Ternary Quantization (TTQ) Layer
Based on "Trained Ternary Quantization" (Zhu et al., ICLR 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTQConv2d(nn.Module):
    """
    Ternary quantized Conv2d layer with learnable scaling factors.
    Quantizes weights to {-Wn, 0, +Wp} where Wp and Wn are learnable parameters.
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
        
        # Full precision latent weights (used during training)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Learnable scaling factors for positive and negative weights
        self.Wp = nn.Parameter(torch.Tensor(1))
        self.Wn = nn.Parameter(torch.Tensor(1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        # Initialize scaling factors
        nn.init.constant_(self.Wp, 1.0)
        nn.init.constant_(self.Wn, 1.0)
    
    def quantize_weight(self, weight):
        """
        Quantize full precision weights to ternary values {-Wn, 0, +Wp}
        
        Formula from paper:
        wt = { +Wp  if w > Δ
             {  0   if |w| ≤ Δ
             { -Wn  if w < -Δ
        
        where Δ = threshold × max(|w|)
        """
        # Normalize weights to [-1, 1]
        max_val = weight.abs().max()
        if max_val > 0:
            weight_norm = weight / max_val
        else:
            weight_norm = weight
        
        # Calculate threshold
        delta = self.threshold * weight_norm.abs().max()
        
        # Create ternary weight tensor
        weight_ternary = torch.zeros_like(weight_norm)
        weight_ternary[weight_norm > delta] = 1.0
        weight_ternary[weight_norm < -delta] = -1.0
        
        # Scale by learnable factors
        weight_quantized = torch.where(
            weight_ternary > 0, 
            self.Wp * weight_ternary,
            torch.where(
                weight_ternary < 0,
                self.Wn * weight_ternary,
                weight_ternary
            )
        )
        
        return weight_quantized, delta, max_val
    
    def forward(self, x):
        """Forward pass with quantized weights"""
        if self.training:
            # During training, quantize weights
            weight_q, _, _ = self.quantize_weight(self.weight)
        else:
            # During inference, use quantized weights
            with torch.no_grad():
                weight_q, _, _ = self.quantize_weight(self.weight)
        
        return F.conv2d(x, weight_q, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
    
    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'threshold={self.threshold}, Wp={self.Wp.item():.4f}, Wn={self.Wn.item():.4f}')


class TTQConv2dWithGrad(TTQConv2d):
    """
    TTQ Conv2d with custom gradient computation for latent weights.
    Implements Equation 8 from the paper for scaled gradients.
    """
    
    def forward(self, x):
        """Forward with custom backward for gradient scaling"""
        if self.training:
            weight_q = TTQWeightFunction.apply(
                self.weight, self.Wp, self.Wn, self.threshold
            )
        else:
            with torch.no_grad():
                weight_q, _, _ = self.quantize_weight(self.weight)
        
        return F.conv2d(x, weight_q, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)


class TTQWeightFunction(torch.autograd.Function):
    """
    Custom autograd function for TTQ weights with scaled gradients.
    Implements gradient computation from Equation 7 and 8 in the paper.
    """
    
    @staticmethod
    def forward(ctx, weight, Wp, Wn, threshold):
        """
        Forward pass: quantize weights to ternary values
        """
        # Normalize weights
        max_val = weight.abs().max()
        if max_val > 0:
            weight_norm = weight / max_val
        else:
            weight_norm = weight
        
        # Calculate threshold
        delta = threshold * weight_norm.abs().max()
        
        # Create masks for positive, negative, and zero weights
        pos_mask = weight_norm > delta
        neg_mask = weight_norm < -delta
        zero_mask = weight_norm.abs() <= delta
        
        # Quantize weights
        weight_q = torch.zeros_like(weight_norm)
        weight_q[pos_mask] = Wp
        weight_q[neg_mask] = -Wn
        
        # Save for backward
        ctx.save_for_backward(weight_norm, Wp, Wn, pos_mask, neg_mask, zero_mask)
        ctx.delta = delta
        
        return weight_q
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute scaled gradients for latent weights and scaling factors
        
        Equation 7 (scaling factor gradients):
        ∂L/∂Wp = Σ(i∈Ip) ∂L/∂wt(i)
        ∂L/∂Wn = Σ(i∈In) ∂L/∂wt(i)
        
        Equation 8 (latent weight gradients):
        ∂L/∂w = { Wp × ∂L/∂wt  if w > Δ
                {  1 × ∂L/∂wt  if |w| ≤ Δ
                { Wn × ∂L/∂wt  if w < -Δ
        """
        weight_norm, Wp, Wn, pos_mask, neg_mask, zero_mask = ctx.saved_tensors
        
        # Gradient for latent weights (Equation 8)
        grad_weight = grad_output.clone()
        grad_weight[pos_mask] = grad_output[pos_mask] * Wp
        grad_weight[neg_mask] = grad_output[neg_mask] * Wn
        grad_weight[zero_mask] = grad_output[zero_mask] * 1.0  # Factor of 1 for zeros
        
        # Gradient for Wp (Equation 7)
        grad_Wp = (grad_output[pos_mask]).sum()
        
        # Gradient for Wn (Equation 7)
        grad_Wn = (grad_output[neg_mask]).sum()
        
        return grad_weight, grad_Wp, grad_Wn, None
