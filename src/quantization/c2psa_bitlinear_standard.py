# src/quantization/c2psa_bitlinear_standard.py

"""
Standard BitLinear from "1-bit LLMs: 1.58 Bits is All You Need"
Same training as our TTQ implementation but with FIXED scales (not learned)

Key difference:
- Scales are E[|w|] and -E[|w|] (FIXED, not parameters)
- Master-Shadow training loop (same as TTQ)
- Forward on shadow (quantized), update on master (FP32)
- No learning of Ap/An parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinearStandard(nn.Module):
    """
    Standard BitLinear with FIXED scales (from paper)
    
    Ternary weights: {-E[|w|], 0, +E[|w|]}
    Scales are fixed after initialization - NOT learned
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 padding=0, groups=1, activation_bits=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.activation_bits = activation_bits
        
        # Weight (FP32) - same as TTQ
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # FIXED scales (NOT learned) - register as buffers
        self.register_buffer('scale', torch.tensor(0.01))  # Single symmetric scale
        
        # LayerNorm
        self.ln = nn.GroupNorm(1, in_channels)
        
        # BitNet quantization constants
        self.Q_b = 2 ** (activation_bits - 1)  # 128 for 8-bit
        self.epsilon = 1e-6
    
    def _compute_ttq_threshold(self, w):
        """TTQ threshold: δ = 0.7 × E[|w|]"""
        mean_abs_w = torch.mean(torch.abs(w))
        delta = 0.7 * mean_abs_w
        return delta
    
    def _quantize_weights_ternary_fixed(self, w):
        """
        Quantize to {-scale, 0, +scale} with FIXED scale
        
        Paper formula:
        - w_ternary ∈ {-α, 0, +α}
        - α = E[|w|] (mean absolute value, fixed)
        """
        delta = self._compute_ttq_threshold(w)
        
        w_q = torch.zeros_like(w)
        pos_mask = w > delta
        neg_mask = w < -delta
        
        # Use FIXED scale (not learned)
        w_q[pos_mask] = self.scale
        w_q[neg_mask] = -self.scale
        
        return w_q
    
    def _quantize_activations_bitnet(self, x):
        """BitNet 8-bit absmax quantization (same as TTQ)"""
        gamma = torch.max(torch.abs(x))
        gamma = torch.clamp(gamma, min=self.epsilon)
        
        x_scaled = x * (self.Q_b / gamma)
        x_clipped = torch.clamp(
            x_scaled,
            min=-self.Q_b + self.epsilon,
            max=self.Q_b - self.epsilon
        )
        x_quant = torch.round(x_clipped)
        
        return x_quant, gamma
    
    def forward(self, x):
        """
        Forward pass: LayerNorm → Absmax Quant → Conv (ternary) → Dequant
        (Same as BitLinear_TTQ, but with fixed scales)
        """
        # LayerNorm
        x_ln = self.ln(x)
        
        # Absmax quantization
        x_quant, gamma = self._quantize_activations_bitnet(x_ln)
        
        # Quantize weights with FIXED scale
        w_q = self._quantize_weights_ternary_fixed(self.weight)
        
        # Conv2d operation
        y = F.conv2d(
            x_quant.float(), w_q,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )
        
        # Dequantization
        y_dequant = y * (gamma / self.Q_b)
        
        # Add bias
        if self.bias is not None:
            y_dequant = y_dequant + self.bias.view(1, -1, 1, 1)
        
        return y_dequant


def replace_c2psa_with_bitlinear_standard(c2psa_module):
    """
    Replace C2PSA Conv layers with standard BitLinear (FIXED scales)
    """
    psa_block = c2psa_module.m[0]
    
    # Replace QKV
    qkv_conv = psa_block.attn.qkv.conv
    in_c = qkv_conv.in_channels
    out_c = qkv_conv.out_channels
    psa_block.attn.qkv.conv = BitLinearStandard(in_c, out_c, kernel_size=1)
    
    # Replace projection
    proj_conv = psa_block.attn.proj.conv
    in_c = proj_conv.in_channels
    out_c = proj_conv.out_channels
    psa_block.attn.proj.conv = BitLinearStandard(in_c, out_c, kernel_size=1)
    
    # Replace PE
    pe_conv = psa_block.attn.pe.conv
    in_c = pe_conv.in_channels
    out_c = pe_conv.out_channels
    kernel_size = pe_conv.kernel_size[0]
    padding = pe_conv.padding[0]
    groups = pe_conv.groups
    
    psa_block.attn.pe.conv = BitLinearStandard(
        in_c, out_c, 
        kernel_size=kernel_size, 
        padding=padding, 
        groups=groups
    )

    ffn_fc1_conv = psa_block.ffn[0].conv  # Conv2d(128, 256, kernel_size=1)
    psa_block.ffn[0].conv = BitLinearStandard(
        ffn_fc1_conv.in_channels, 
        ffn_fc1_conv.out_channels, 
        kernel_size=1
    )
    
    ffn_fc2_conv = psa_block.ffn[1].conv  # Conv2d(256, 128, kernel_size=1)
    psa_block.ffn[1].conv = BitLinearStandard(
        ffn_fc2_conv.in_channels, 
        ffn_fc2_conv.out_channels, 
        kernel_size=1
    )
    
    print("✓ Replaced C2PSA with Standard BitLinear (FIXED Scales)")
    print("  - qkv.conv: Fixed scale quantization")
    print("  - proj.conv: Fixed scale quantization")
    print("  - pe.conv: Fixed scale quantization")
    print("  - ffn[0].conv: Quantized (128→256 + SiLU)")
    print("  - ffn[1].conv: Quantized (256→128)")
    
    return c2psa_module
