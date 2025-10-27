"""
Trained Ternary Quantization (TTQ) Layer - Final Fixed Implementation
Uses Softplus to guarantee Wp/Wn > 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTQConv2d(nn.Module):
    """
    Ternary quantized Conv2d with GUARANTEED positive scaling factors
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, threshold=0.05,
                 pretrained_weight=None):
        super(TTQConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.threshold = threshold
        
        # Full precision weights
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Unconstrained parameters (can be any value)
        # Wp = softplus(Wp_param) and Wn = softplus(Wn_param) will be positive
        self.Wp_param = nn.Parameter(torch.Tensor(1))
        self.Wn_param = nn.Parameter(torch.Tensor(1))
        
        self.reset_parameters(pretrained_weight)
    
    def reset_parameters(self, pretrained_weight=None):
        """Initialize parameters"""
        if pretrained_weight is not None:
            self.weight.data.copy_(pretrained_weight)
            
            with torch.no_grad():
                w_abs = torch.abs(pretrained_weight)
                delta = self.threshold * torch.max(w_abs)
                
                pos_mask = pretrained_weight > delta
                neg_mask = pretrained_weight < -delta
                
                if pos_mask.any():
                    Wp_init = torch.mean(w_abs[pos_mask])
                else:
                    Wp_init = torch.mean(w_abs)
                
                if neg_mask.any():
                    Wn_init = torch.mean(w_abs[neg_mask])
                else:
                    Wn_init = torch.mean(w_abs)
                
                # Clamp to reasonable range
                Wp_init = torch.clamp(Wp_init, min=0.01, max=1.0)
                Wn_init = torch.clamp(Wn_init, min=0.01, max=1.0)
                
                # Initialize params such that softplus(param) ≈ init_value
                # softplus(x) = log(1 + exp(x))
                # Inverse: x = log(exp(init) - 1) ≈ init for init > 1
                # For init < 1, use: x = log(exp(init) - 1) with clipping
                self.Wp_param.data.fill_(self._inverse_softplus(Wp_init).item())
                self.Wn_param.data.fill_(self._inverse_softplus(Wn_init).item())
        else:
            nn.init.kaiming_uniform_(self.weight, a=0)
            # Initialize to give softplus(param) ≈ 0.3
            self.Wp_param.data.fill_(self._inverse_softplus(torch.tensor(0.3)).item())
            self.Wn_param.data.fill_(self._inverse_softplus(torch.tensor(0.3)).item())
        
        if self.bias is not None and pretrained_weight is None:
            nn.init.zeros_(self.bias)
    
    @staticmethod
    def _inverse_softplus(y):
        """Compute inverse of softplus: x such that softplus(x) = y"""
        # softplus(x) = log(1 + exp(x))
        # Inverse: x = log(exp(y) - 1)
        # For numerical stability, use: x = y + log(1 - exp(-y)) for y > 0
        return torch.log(torch.exp(y) - 1 + 1e-8)
    
    @property
    def Wp(self):
        """Get Wp (guaranteed positive via softplus)"""
        return F.softplus(self.Wp_param)
    
    @property
    def Wn(self):
        """Get Wn (guaranteed positive via softplus)"""
        return F.softplus(self.Wn_param)
    
    def forward(self, x):
        """Forward pass using softplus-constrained Wp/Wn"""
        # Get guaranteed-positive scaling factors
        Wp_pos = F.softplus(self.Wp_param)
        Wn_pos = F.softplus(self.Wn_param)
        
        # Quantize weights
        weight_q = TTQWeightFunction.apply(
            self.weight, Wp_pos, Wn_pos, self.threshold
        )
        
        return F.conv2d(x, weight_q, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
    
    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'threshold={self.threshold}, Wp={self.Wp.item():.4f}, Wn={self.Wn.item():.4f}')


TTQConv2dWithGrad = TTQConv2d


class TTQWeightFunction(torch.autograd.Function):
    """TTQ autograd function with gradient averaging"""
    
    @staticmethod
    def forward(ctx, weight, Wp, Wn, threshold):
        delta = threshold * weight.abs().max()
        
        pos_mask = weight > delta
        neg_mask = weight < -delta
        
        weight_quantized = torch.zeros_like(weight)
        weight_quantized[pos_mask] = Wp
        weight_quantized[neg_mask] = -Wn
        
        ctx.save_for_backward(weight, Wp, Wn)
        ctx.pos_mask = pos_mask
        ctx.neg_mask = neg_mask
        
        return weight_quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        weight, Wp, Wn = ctx.saved_tensors
        pos_mask = ctx.pos_mask
        neg_mask = ctx.neg_mask
        
        # Gradient w.r.t. weights
        grad_weight = grad_output.clone()
        
        if pos_mask.any():
            grad_weight = torch.where(pos_mask, grad_output * Wp, grad_weight)
        
        if neg_mask.any():
            grad_weight = torch.where(neg_mask, grad_output * Wn, grad_weight)
        
        # Gradient w.r.t. Wp/Wn (averaged to prevent explosion)
        if pos_mask.any():
            n_pos = pos_mask.sum().float()
            grad_Wp = grad_output[pos_mask].sum() / n_pos
            grad_Wp = grad_Wp.reshape(1)
        else:
            grad_Wp = torch.zeros(1, dtype=grad_output.dtype, device=grad_output.device)
        
        if neg_mask.any():
            n_neg = neg_mask.sum().float()
            grad_Wn = -grad_output[neg_mask].sum() / n_neg
            grad_Wn = grad_Wn.reshape(1)
        else:
            grad_Wn = torch.zeros(1, dtype=grad_output.dtype, device=grad_output.device)
        
        return grad_weight, grad_Wp, grad_Wn, None
