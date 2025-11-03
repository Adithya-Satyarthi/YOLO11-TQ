# src/quantization/bitlinear_ttq_manager.py

"""
BitLinear TTQ Manager - Handles master-shadow synchronization
Master model: Standard Conv2d with FP32 weights
Shadow model: BitLinear_TTQ with ternary weights
"""

import torch
import torch.nn as nn
import numpy as np


class BitLinearTTQManager:
    """
    Manages TTQ quantization for C2PSA BitLinear layers.
    
    Master model: FP32 weights, standard Conv2d, gradient computation
    Shadow model: Ternary weights, BitLinear_TTQ, forward pass only
    """
    
    def __init__(self, master_model, shadow_model, threshold=0.7, device='cuda'):
        self.master_model = master_model
        self.shadow_model = shadow_model
        self.threshold = threshold
        self.device = device
        
        # External scaling factors
        self.ap_dict = {}  # Positive scaling
        self.an_dict = {}  # Negative scaling
        
        self.quantized_layers = []
        
        # Initialize
        self._initialize_scaling_factors()
        
        print(f"✓ BitLinear TTQ Manager initialized")
        print(f"  Quantized layers: {len(self.quantized_layers)}")
    
    def _get_master_conv_modules(self):
        """Get Conv2d modules from master model (layer.conv) - INCLUDING PE"""
        modules = {}
        for name, module in self.master_model.named_modules():
            if isinstance(module, nn.Conv2d):
                if 'model.10' in name and ('qkv.conv' in name or 'proj.conv' in name or 'pe.conv' in name):
                    modules[name] = module
                elif 'model.10' in name and 'ffn' in name:
                    modules[name] = module
        return modules

    def _get_shadow_bitlinear_modules(self):
        """Get BitLinear_TTQ modules from shadow model - INCLUDING PE"""
        from src.quantization.c2psa_bitlinear_ttq import BitLinear_TTQ
        
        modules = {}
        for name, module in self.shadow_model.named_modules():
            if isinstance(module, BitLinear_TTQ):
                if 'model.10' in name and ('qkv.conv' in name or 'proj.conv' in name or 'pe.conv' in name):
                    modules[name] = module
                elif 'model.10' in name and 'ffn' in name:
                    modules[name] = module
        return modules

    
    def _initialize_scaling_factors(self):
        """
        Initialize Ap/An from master model FP32 weights.
        """
        master_modules = self._get_master_conv_modules()
        
        for name, master_conv in master_modules.items():
            w = master_conv.weight.data
            w_abs = torch.abs(w)
            
            # TTQ threshold
            delta = self.threshold * torch.mean(w_abs)
            
            # Compute Ap/An
            pos_mask = w > delta
            if pos_mask.any():
                ap_init = torch.mean(w_abs[pos_mask])
            else:
                ap_init = torch.tensor(0.01)
            
            neg_mask = w < -delta
            if neg_mask.any():
                an_init = torch.mean(w_abs[neg_mask])
            else:
                an_init = torch.tensor(0.01)
            
            # Store as learnable parameters
            self.ap_dict[name] = nn.Parameter(ap_init.to(self.device))
            self.an_dict[name] = nn.Parameter(an_init.to(self.device))
            
            self.quantized_layers.append(name)
            
            print(f"  ✓ Init {name}: Ap={ap_init:.4f}, An={an_init:.4f}")
    
    def quantize_master_to_shadow(self):
        """
        Quantize master model FP32 weights → shadow model ternary weights.
        
        Process:
        1. Get FP32 weights from master
        2. Quantize to ternary using current Ap/An
        3. Copy to shadow model BitLinear_TTQ weights
        """
        with torch.no_grad():
            master_modules = self._get_master_conv_modules()
            shadow_modules = self._get_shadow_bitlinear_modules()
            
            for name in self.quantized_layers:
                if name not in master_modules or name not in shadow_modules:
                    continue
                
                # Get modules
                master_conv = master_modules[name]
                shadow_bitlinear = shadow_modules[name]
                
                # Get FP32 weights
                w = master_conv.weight.data
                ap = self.ap_dict[name].data
                an = self.an_dict[name].data
                
                # Compute threshold
                w_abs = torch.abs(w)
                delta = self.threshold * torch.mean(w_abs)
                
                # Quantize to ternary
                w_q = torch.zeros_like(w)
                pos_mask = w > delta
                neg_mask = w < -delta
                
                w_q[pos_mask] = ap
                w_q[neg_mask] = -an
                
                # Copy to shadow model
                shadow_bitlinear.weight.data.copy_(w_q)
                
                # Copy bias
                if master_conv.bias is not None:
                    shadow_bitlinear.bias.data.copy_(master_conv.bias.data)
    
    def compute_ttq_gradients(self):
        """
        Compute TTQ gradients using shadow model gradients.
        
        Returns:
        - master_grads: Gradients for master model weights
        - ap_grads: Gradients for Ap parameters
        - an_grads: Gradients for An parameters
        """
        master_grads = {}
        ap_grads = {}
        an_grads = {}
        
        master_modules = self._get_master_conv_modules()
        shadow_modules = self._get_shadow_bitlinear_modules()
        
        for name in self.quantized_layers:
            if name not in master_modules or name not in shadow_modules:
                continue
            
            master_conv = master_modules[name]
            shadow_bitlinear = shadow_modules[name]
            
            if shadow_bitlinear.weight.grad is None:
                continue
            
            # Get gradients from shadow
            grad_wt = shadow_bitlinear.weight.grad.data
            w = master_conv.weight.data
            ap = self.ap_dict[name].data
            an = self.an_dict[name].data
            
            # Compute masks
            w_abs = torch.abs(w)
            delta = self.threshold * torch.mean(w_abs)
            pos_mask = w > delta
            neg_mask = w < -delta
            zero_mask = torch.abs(w) <= delta
            
            # Gradient for master weights (Eq. 8)
            grad_w = torch.zeros_like(grad_wt)
            grad_w[pos_mask] = grad_wt[pos_mask] * ap
            grad_w[zero_mask] = grad_wt[zero_mask] * 1.0
            grad_w[neg_mask] = grad_wt[neg_mask] * an
            
            master_grads[name] = grad_w
            
            # Gradient for Ap (Eq. 7)
            if pos_mask.any():
                grad_ap_raw = torch.mean(grad_wt[pos_mask])
                grad_ap_normalized = grad_ap_raw / (ap.abs() + 1e-6)
                grad_ap_clipped = torch.clamp(grad_ap_normalized, -1.0, 1.0)
                ap_grads[name] = grad_ap_clipped
            else:
                ap_grads[name] = torch.tensor(0.0).to(self.device)
            
            # Gradient for An (Eq. 7)
            if neg_mask.any():
                grad_an_raw = torch.mean(-grad_wt[neg_mask])
                grad_an_normalized = grad_an_raw / (an.abs() + 1e-6)
                grad_an_clipped = torch.clamp(grad_an_normalized, -1.0, 1.0)
                an_grads[name] = grad_an_clipped
            else:
                an_grads[name] = torch.tensor(0.0).to(self.device)
        
        return master_grads, ap_grads, an_grads
    
    def apply_gradients_to_master(self, master_grads):
        """Apply gradients to master model weights"""
        master_modules = self._get_master_conv_modules()
        
        for name in self.quantized_layers:
            if name in master_grads and name in master_modules:
                master_conv = master_modules[name]
                if master_conv.weight.grad is None:
                    master_conv.weight.grad = master_grads[name].clone()
                else:
                    master_conv.weight.grad.copy_(master_grads[name])
    
    def apply_gradients_to_scales(self, ap_grads, an_grads):
        """Apply gradients to Ap/An parameters"""
        for name in self.quantized_layers:
            if name in ap_grads:
                if self.ap_dict[name].grad is None:
                    self.ap_dict[name].grad = ap_grads[name].clone().reshape(())
                else:
                    self.ap_dict[name].grad = ap_grads[name].clone().reshape(())
            
            if name in an_grads:
                if self.an_dict[name].grad is None:
                    self.an_dict[name].grad = an_grads[name].clone().reshape(())
                else:
                    self.an_dict[name].grad = an_grads[name].clone().reshape(())
    
    def get_scaling_parameters(self):
        """Get Ap/An parameters for optimizer"""
        return list(self.ap_dict.values()) + list(self.an_dict.values())
    
    def export_quantized_model(self, save_path):
        """Export shadow model (already has ternary weights)"""
        self.quantize_master_to_shadow()
        
        torch.save({
            'model': self.shadow_model,
            'quantized_layers': self.quantized_layers,
            'threshold': self.threshold
        }, save_path)
        
        print(f"✓ Exported quantized C2PSA model to {save_path}")
    
    def print_statistics(self):
        """Print Ap/An statistics"""
        ap_vals = [self.ap_dict[n].item() for n in self.quantized_layers]
        an_vals = [self.an_dict[n].item() for n in self.quantized_layers]
        
        if ap_vals and an_vals:
            print(f"\nC2PSA Scaling Factor Statistics:")
            print(f"  Ap: min={min(ap_vals):.4f}, max={max(ap_vals):.4f}, mean={np.mean(ap_vals):.4f}")
            print(f"  An: min={min(an_vals):.4f}, max={max(an_vals):.4f}, mean={np.mean(an_vals):.4f}")
