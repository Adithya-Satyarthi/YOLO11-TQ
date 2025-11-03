# src/quantization/bitlinear_standard_manager.py

"""
Manager for Standard BitLinear (FIXED scales)
Same master-shadow training loop as TTQ, but scales are NOT learned
"""

import torch
import torch.nn as nn
import numpy as np


class BitLinearStandardManager:
    """
    Manages standard BitLinear training with FIXED scales
    
    Same as BitLinearTTQManager but:
    - Scales are FIXED (not learned parameters)
    - No gradient computation for scales
    - Only master weights are updated
    """
    
    def __init__(self, master_model, shadow_model, threshold=0.7, device='cuda'):
        self.master_model = master_model
        self.shadow_model = shadow_model
        self.threshold = threshold
        self.device = device
        
        self.quantized_layers = []
        self._initialize_fixed_scales()
        
        print(f"✓ Standard BitLinear Manager initialized")
        print(f"  Quantized layers: {len(self.quantized_layers)}")
    
    def _get_master_conv_modules(self):
        """Get Conv2d modules from master"""
        modules = {}
        for name, module in self.master_model.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'model.10' in name and ('qkv.conv' in name or 'proj.conv' in name or 'pe.conv' in name):
                    modules[name] = module
                elif 'model.10' in name and 'ffn' in name:
                    modules[name] = module
        return modules
    
    def _get_shadow_bitlinear_modules(self):
        """Get BitLinearStandard modules from shadow"""
        from src.quantization.c2psa_bitlinear_standard import BitLinearStandard
        
        modules = {}
        for name, module in self.shadow_model.named_modules():
            if isinstance(module, BitLinearStandard):
                if 'model.10' in name and ('qkv.conv' in name or 'proj.conv' in name or 'pe.conv' in name):
                    modules[name] = module
                elif 'model.10' in name and 'ffn' in name:
                    modules[name] = module
        return modules
    
    def _initialize_fixed_scales(self):
        """
        Initialize FIXED scales from master weights (Paper formula)
        scale = E[|w|] (mean absolute value of weights)
        """
        master_modules = self._get_master_conv_modules()
        
        for name, master_conv in master_modules.items():
            w = master_conv.weight.data
            w_abs = torch.abs(w)
            
            # Fixed scale (symmetric): α = E[|w|]
            scale = torch.mean(w_abs)
            
            self.quantized_layers.append(name)
            
            print(f"  ✓ Init {name}: Fixed scale={scale:.6f}")
    
    def quantize_master_to_shadow(self):
        """
        Quantize master (FP32) → shadow (ternary) with FIXED scales
        Same as TTQ but scales don't change
        """
        with torch.no_grad():
            master_modules = self._get_master_conv_modules()
            shadow_modules = self._get_shadow_bitlinear_modules()
            
            for name in self.quantized_layers:
                if name not in master_modules or name not in shadow_modules:
                    continue
                
                master_conv = master_modules[name]
                shadow_bitlinear = shadow_modules[name]
                
                # Get FP32 weights and FIXED scale
                w = master_conv.weight.data
                scale = shadow_bitlinear.scale  # FIXED, not learned
                
                # Compute threshold
                w_abs = torch.abs(w)
                delta = self.threshold * torch.mean(w_abs)
                
                # Quantize with FIXED scale
                w_q = torch.zeros_like(w)
                pos_mask = w > delta
                neg_mask = w < -delta
                
                w_q[pos_mask] = scale
                w_q[neg_mask] = -scale
                
                # Copy to shadow
                shadow_bitlinear.weight.data.copy_(w_q)
                
                # Copy bias
                if master_conv.bias is not None:
                    shadow_bitlinear.bias.data.copy_(master_conv.bias.data)
    
    def compute_standard_gradients(self):
        """
        Compute gradients for FIXED scale (standard BitLinear)
        
        Unlike TTQ, scales don't have gradients - only weights do
        Simply return master weight gradients (no scale updates)
        """
        master_grads = {}
        
        master_modules = self._get_master_conv_modules()
        shadow_modules = self._get_shadow_bitlinear_modules()
        
        for name in self.quantized_layers:
            if name not in master_modules or name not in shadow_modules:
                continue
            
            master_conv = master_modules[name]
            shadow_bitlinear = shadow_modules[name]
            
            if shadow_bitlinear.weight.grad is None:
                continue
            
            grad_wt = shadow_bitlinear.weight.grad.data
            w = master_conv.weight.data
            scale = shadow_bitlinear.scale.data
            
            # Compute masks
            w_abs = torch.abs(w)
            delta = self.threshold * torch.mean(w_abs)
            pos_mask = w > delta
            neg_mask = w < -delta
            zero_mask = torch.abs(w) <= delta
            
            # Gradient for master weights (same as TTQ)
            grad_w = torch.zeros_like(grad_wt)
            grad_w[pos_mask] = grad_wt[pos_mask] * scale
            grad_w[zero_mask] = grad_wt[zero_mask] * 1.0
            grad_w[neg_mask] = grad_wt[neg_mask] * scale
            
            master_grads[name] = grad_w
        
        return master_grads, {}, {}  # No scale gradients
    
    def apply_gradients_to_master(self, master_grads):
        """Apply gradients to master weights only"""
        master_modules = self._get_master_conv_modules()
        
        for name in self.quantized_layers:
            if name in master_grads and name in master_modules:
                master_conv = master_modules[name]
                if master_conv.weight.grad is None:
                    master_conv.weight.grad = master_grads[name].clone()
                else:
                    master_conv.weight.grad.copy_(master_grads[name])
    
    def get_scaling_parameters(self):
        """
        Get scale parameters for optimizer
        Returns empty list since scales are FIXED (not learned)
        """
        return []
    
    def export_quantized_model(self, save_path):
        """Export shadow model"""
        self.quantize_master_to_shadow()
        
        torch.save({
            'model': self.shadow_model,
            'quantized_layers': self.quantized_layers,
            'threshold': self.threshold,
            'scale_type': 'fixed'
        }, save_path)
        
        print(f"✓ Exported standard BitLinear model to {save_path}")
    
    def print_statistics(self):
        """Print fixed scales"""
        shadow_modules = self._get_shadow_bitlinear_modules()
        
        scales = [shadow_modules[n].scale.item() for n in self.quantized_layers if n in shadow_modules]
        
        if scales:
            print(f"\nStandard BitLinear Fixed Scales:")
            print(f"  Scale: min={min(scales):.6f}, max={max(scales):.6f}, mean={np.mean(scales):.6f}")
