import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear_TTQ(nn.Module):
    """
    BitLinear layer that wraps Conv2d for compatibility with YOLO's C2PSA.
    
    Architecture:
    - TTQ-quantized weights {-An, 0, +Ap}
    - 8-bit absmax activation quantization (BitNet Eq. 4-5)
    - LayerNorm (adapted for 4D tensors [B, C, H, W])
    - Learnable Ap/An parameters
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
        
        # Weight (FP32)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # TTQ scaling factors
        self.ap = nn.Parameter(torch.tensor(0.01))
        self.an = nn.Parameter(torch.tensor(0.01))
        
        # LayerNorm (channel-wise for [B, C, H, W])
        self.ln = nn.GroupNorm(1, in_channels)
        
        # BitNet quantization constants
        self.Q_b = 2 ** (activation_bits - 1)  # 128 for 8-bit
        self.epsilon = 1e-6
    
    def _compute_ttq_threshold(self, w):
        """TTQ threshold: delta = 0.7 * E[|w|]"""
        mean_abs_w = torch.mean(torch.abs(w))
        delta = 0.7 * mean_abs_w
        return delta
    
    def _quantize_weights_ternary(self, w):
        """Quantize to {-An, 0, +Ap}"""
        delta = self._compute_ttq_threshold(w)
        
        w_q = torch.zeros_like(w)
        pos_mask = w > delta
        neg_mask = w < -delta
        
        w_q[pos_mask] = self.ap
        w_q[neg_mask] = -self.an
        
        return w_q
    
    def _quantize_activations_bitnet(self, x):

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
        Forward pass: LayerNorm, Absmax Quant, Conv2d (ternary), Dequant
        """
        # LayerNorm
        x_ln = self.ln(x)
        
        # Absmax quantization
        x_quant, gamma = self._quantize_activations_bitnet(x_ln)
        
        # Quantize weights
        w_q = self._quantize_weights_ternary(self.weight)
        
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


def replace_c2psa_with_bitlinear(c2psa_module):
    """
    Replace Conv layers in C2PSA with BitLinear_TTQ.
    """
    # Access PSABlock (first element in Sequential)
    psa_block = c2psa_module.m[0]
    
    # Replace QKV projection
    qkv_conv = psa_block.attn.qkv.conv
    in_c = qkv_conv.in_channels
    out_c = qkv_conv.out_channels
    psa_block.attn.qkv.conv = BitLinear_TTQ(in_c, out_c, kernel_size=1)
    
    # Replace output projection
    proj_conv = psa_block.attn.proj.conv
    in_c = proj_conv.in_channels
    out_c = proj_conv.out_channels
    psa_block.attn.proj.conv = BitLinear_TTQ(in_c, out_c, kernel_size=1)
    
    # Replace positional encoding with BitLinear_TTQ
    pe_conv = psa_block.attn.pe.conv
    in_c = pe_conv.in_channels
    out_c = pe_conv.out_channels
    kernel_size = pe_conv.kernel_size[0]
    padding = pe_conv.padding[0]
    groups = pe_conv.groups
    
    psa_block.attn.pe.conv = BitLinear_TTQ(
        in_c, out_c, 
        kernel_size=kernel_size, 
        padding=padding, 
        groups=groups
    )

    ffn_fc1_conv = psa_block.ffn[0].conv  # Conv2d(128, 256, kernel_size=1)
    psa_block.ffn[0].conv = BitLinear_TTQ(
        ffn_fc1_conv.in_channels, 
        ffn_fc1_conv.out_channels, 
        kernel_size=1
    )
    
    ffn_fc2_conv = psa_block.ffn[1].conv  # Conv2d(256, 128, kernel_size=1)
    psa_block.ffn[1].conv = BitLinear_TTQ(
        ffn_fc2_conv.in_channels, 
        ffn_fc2_conv.out_channels, 
        kernel_size=1
    )
    
    print("   Replaced C2PSA attention Conv layers with BitLinear_TTQ")
    print("  - qkv.conv: Quantized (128→256)")
    print("  - proj.conv: Quantized (128→128)")
    print("  - pe.conv: Quantized (128→128, 3*3 depthwise)")
    print("  - ffn[0].conv: Quantized (128→256 + SiLU)")
    print("  - ffn[1].conv: Quantized (256→128)")
    
    return c2psa_module


def replace_c2psa_with_bitlinear_shadow(c2psa_module):
    """
    Alias for replace_c2psa_with_bitlinear for consistency with shadow model naming
    """
    return replace_c2psa_with_bitlinear(c2psa_module)
