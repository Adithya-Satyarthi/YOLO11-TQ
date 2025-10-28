"""
Trained Ternary Quantization (TTQ) Layer - FINAL FIXED VERSION
Critical fixes:
1. Proper gradient flow for Wp/Wn
2. Mean-based threshold (less aggressive - ~35% zeros instead of 60%)
3. Correct parameter initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTQConv2d(nn.Module):
    """
    TTQ Conv2d with GUARANTEED positive Wp/Wn and proper gradient flow
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
        self.threshold = threshold  # Now represents multiplier for mean (default 0.7)
        
        # Full precision weights
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # CRITICAL FIX: Use regular Parameters, apply softplus in forward
        self.Wp_param = nn.Parameter(torch.Tensor(1))
        self.Wn_param = nn.Parameter(torch.Tensor(1))
        
        self.reset_parameters(pretrained_weight)
    
    def reset_parameters(self, pretrained_weight=None):
        """
        Initialize with MEAN-BASED threshold (from TWN paper)
        This avoids creating too many zeros
        """
        if pretrained_weight is not None:
            self.weight.data.copy_(pretrained_weight)
            
            with torch.no_grad():
                w_abs = torch.abs(pretrained_weight)
                
                # CRITICAL FIX: Use mean-based threshold (TWN/TTQ papers)
                # OLD: delta = 0.05 * max(|W|) → 60% zeros → mAP drops 79%!
                # NEW: delta = 0.7 * mean(|W|) → 35% zeros → much better!
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
                
                # Clamp to reasonable range (increased max for larger scales)
                #Wp_init = torch.clamp(Wp_init, min=0.05, max=2.0)
                #Wn_init = torch.clamp(Wn_init, min=0.05, max=2.0)
                
                # Initialize params for softplus
                self.Wp_param.data.fill_(self._inverse_softplus(Wp_init).item())
                self.Wn_param.data.fill_(self._inverse_softplus(Wn_init).item())
        else:
            nn.init.kaiming_uniform_(self.weight, a=0)
            # Initialize to give softplus ≈ 0.5
            self.Wp_param.data.fill_(self._inverse_softplus(torch.tensor(0.5)).item())
            self.Wn_param.data.fill_(self._inverse_softplus(torch.tensor(0.5)).item())
        
        if self.bias is not None and pretrained_weight is None:
            nn.init.zeros_(self.bias)
    
    @staticmethod
    def _inverse_softplus(y):
        """Inverse softplus for initialization"""
        return torch.log(torch.exp(y) - 1 + 1e-8)
    
    def forward(self, x):
        """
        Forward with GUARANTEED gradient flow
        CRITICAL: Apply softplus HERE, not in @property
        """
        # Apply softplus to get positive Wp/Wn WITH GRADIENT TRACKING
        Wp_pos = F.softplus(self.Wp_param)
        Wn_pos = F.softplus(self.Wn_param)
        
        # Quantize weights using custom autograd function
        weight_q = TTQWeightFunction.apply(
            self.weight, Wp_pos, Wn_pos, self.threshold
        )
        
        return F.conv2d(x, weight_q, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
    
    @property
    def Wp(self):
        """Get current Wp value (for logging only)"""
        with torch.no_grad():
            return F.softplus(self.Wp_param).item()
    
    @property
    def Wn(self):
        """Get current Wn value (for logging only)"""
        with torch.no_grad():
            return F.softplus(self.Wn_param).item()
    
    def extra_repr(self):
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'threshold={self.threshold}, Wp={self.Wp:.4f}, Wn={self.Wn:.4f}')


TTQConv2dWithGrad = TTQConv2d


class TTQWeightFunction(torch.autograd.Function):
    """
    TTQ autograd function with MEAN-BASED threshold
    """
    
    @staticmethod
    def forward(ctx, weight, Wp, Wn, threshold):
        """
        Forward: Quantize to ternary using mean-based threshold
        CRITICAL: Must match initialization strategy!
        """
        # CRITICAL FIX: Use mean-based threshold (consistent with init)
        w_abs = torch.abs(weight)
        delta = threshold * torch.mean(w_abs)  # NOT max!
        
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
        Backward: Compute gradients
        """
        weight, Wp, Wn = ctx.saved_tensors
        pos_mask = ctx.pos_mask
        neg_mask = ctx.neg_mask
        
        # Gradient w.r.t. full-precision weights
        grad_weight = grad_output.clone()
        
        if pos_mask.any():
            grad_weight = torch.where(pos_mask, grad_output * Wp, grad_weight)
        
        if neg_mask.any():
            grad_weight = torch.where(neg_mask, grad_output * Wn, grad_weight)
        
        # Gradient w.r.t. Wp (averaged to prevent explosion)
        if pos_mask.any():
            n_pos = pos_mask.sum().float()
            grad_Wp = grad_output[pos_mask].sum() / n_pos
            grad_Wp = grad_Wp.reshape(1)
        else:
            grad_Wp = torch.zeros(1, dtype=grad_output.dtype, device=grad_output.device)
        
        # Gradient w.r.t. Wn (averaged)
        if neg_mask.any():
            n_neg = neg_mask.sum().float()
            grad_Wn = -grad_output[neg_mask].sum() / n_neg
            grad_Wn = grad_Wn.reshape(1)
        else:
            grad_Wn = torch.zeros(1, dtype=grad_output.dtype, device=grad_output.device)
        
        # Return: grad_weight, grad_Wp, grad_Wn, grad_threshold (None)
        return grad_weight, grad_Wp, grad_Wn, None
