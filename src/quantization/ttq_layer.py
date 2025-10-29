"""
Trained Ternary Quantization (TTQ) Layer - FINAL VERSION
Critical fixes:
1. Paper uses SUM for Wp/Wn gradients, not AVERAGE
2. Removed softplus - use direct parameters (diagnostic proved positivity)
3. Simple offset for numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTQConv2d(nn.Module):
    """
    TTQ Conv2d with direct parameter learning
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, threshold=0.7,
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
        
        # Direct scale parameters (no transformation)
        self.Wp_param = nn.Parameter(torch.Tensor(1))
        self.Wn_param = nn.Parameter(torch.Tensor(1))
        
        self.reset_parameters(pretrained_weight)
    
    def reset_parameters(self, pretrained_weight=None):
        """
        Initialize with mean-based threshold
        """
        if pretrained_weight is not None:
            self.weight.data.copy_(pretrained_weight)
            
            with torch.no_grad():
                w_abs = torch.abs(pretrained_weight)
                
                # Mean-based threshold
                delta = self.threshold * torch.mean(w_abs)
                
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
                
                # Direct initialization (no transformation)
                self.Wp_param.data.fill_(Wp_init.item())
                self.Wn_param.data.fill_(Wn_init.item())
        else:
            nn.init.kaiming_uniform_(self.weight, a=0)
            # Initialize to reasonable default
            self.Wp_param.data.fill_(0.5)
            self.Wn_param.data.fill_(0.5)
        
        if self.bias is not None and pretrained_weight is None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        """
        Forward - use raw parameters with small offset
        """
        # Add small offset for numerical stability
        Wp_use = self.Wp_param + 1e-8
        Wn_use = self.Wn_param + 1e-8
        
        # Quantize weights using custom autograd function
        weight_q = TTQWeightFunction.apply(
            self.weight, Wp_use, Wn_use, self.threshold
        )
        
        return F.conv2d(x, weight_q, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
    
    @property
    def Wp(self):
        """Get current Wp value (for logging only)"""
        with torch.no_grad():
            return self.Wp_param.item()
    
    @property
    def Wn(self):
        """Get current Wn value (for logging only)"""
        with torch.no_grad():
            return self.Wn_param.item()
    
    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'threshold={self.threshold}, Wp={self.Wp:.4f}, Wn={self.Wn:.4f}')


TTQConv2dWithGrad = TTQConv2d


class TTQWeightFunction(torch.autograd.Function):
    """
    TTQ autograd function with mean-based threshold
    """
    
    @staticmethod
    def forward(ctx, weight, Wp, Wn, threshold):
        """
        Forward: Quantize to ternary using mean-based threshold
        """
        # Mean-based threshold
        w_abs = torch.abs(weight)
        delta = threshold * torch.mean(w_abs)
        
        pos_mask = weight > delta
        neg_mask = weight < -delta
        
        weight_quantized = torch.zeros_like(weight)
        weight_quantized[pos_mask] = Wp
        weight_quantized[neg_mask] = -Wn
        
        ctx.save_for_backward(weight, Wp, Wn)
        ctx.pos_mask = pos_mask
        ctx.neg_mask = neg_mask
        ctx.threshold = threshold
        
        return weight_quantized
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: Compute gradients according to paper (Equations 8 & 9)
        CRITICAL: Use SUM (not average) per paper
        """
        weight, Wp, Wn = ctx.saved_tensors
        pos_mask = ctx.pos_mask
        neg_mask = ctx.neg_mask
        
        # Gradient w.r.t. full-precision weights (Paper Equation 8)
        grad_weight = grad_output.clone()
        
        if pos_mask.any():
            grad_weight[pos_mask] = grad_output[pos_mask] * Wp
        
        if neg_mask.any():
            grad_weight[neg_mask] = grad_output[neg_mask] * Wn
        
        # Gradient w.r.t. Wp and Wn (Paper Equation 9) - Use SUM
        if pos_mask.any():
            grad_Wp = grad_output[pos_mask].sum()
            grad_Wp = grad_Wp.reshape(1)
        else:
            grad_Wp = torch.zeros(1, dtype=grad_output.dtype, device=grad_output.device)
        
        if neg_mask.any():
            grad_Wn = -grad_output[neg_mask].sum()
            grad_Wn = grad_Wn.reshape(1)
        else:
            grad_Wn = torch.zeros(1, dtype=grad_output.dtype, device=grad_output.device)
        
        return grad_weight, grad_Wp, grad_Wn, None
