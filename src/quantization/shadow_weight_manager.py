import torch
import torch.nn as nn
from pathlib import Path
import numpy as np


class ShadowWeightManager:
    """
    Manages TTQ quantization using shadow weights approach.
    - Master model: FP32 weights (for gradient accumulation)
    - Shadow model: Quantized ternary weights (for forward pass)
    - External Wp/Wn: Learned scaling factors stored separately
    """
    
    def __init__(self, master_model, shadow_model, threshold=0.7, device='cuda', 
                 target_layers=None, quantize_1x1=False):

        self.master_model = master_model
        self.shadow_model = shadow_model
        self.threshold = threshold
        self.device = device
        self.target_layers = target_layers
        self.quantize_1x1 = quantize_1x1
        
        # External scaling factor
        self.wp_dict = {}  # {layer_name: Wp tensor}
        self.wn_dict = {}  # {layer_name: Wn tensor}
        
        # Track which layers are quantized
        self.quantized_layers = []
        
        # Initialize scaling factors
        self._initialize_scaling_factors()
        
        print(f"Shadow Weight Manager initialized")
        print(f"Quantized layers: {len(self.quantized_layers)}")
        print(f"1x1 convolution quantization: {'Enabled' if quantize_1x1 else 'Disabled'}")
    
    def _should_quantize(self, name, module):
        """Determine if a layer should be quantized"""
        if not isinstance(module, nn.Conv2d):
            return False
        
        # Check 1x1 convolutions
        k = module.kernel_size
        is_1x1 = (isinstance(k, tuple) and k[0] == 1 and k[1] == 1) or k == 1
        
        if is_1x1 and not self.quantize_1x1:
            return False
        
        parts = name.split('.')
        if len(parts) < 3 or parts[0] != 'model':
            return False
        
        try:
            layer_idx = int(parts[1])
        except ValueError:
            return False
        
        # Exclusions
        if layer_idx in [0, 10, 23]:  # First, C2PSA, Detect
            return False
        
        # Progressive quantization: only quantize target layers
        if self.target_layers is not None and layer_idx not in self.target_layers:
            return False
        
        return True
    
    def _initialize_scaling_factors(self):
        """
        Initialize Wp/Wn from pretrained weights.
        """
        for name, master_module in self.master_model.named_modules():
            if self._should_quantize(name, master_module):
                # Get pretrained weights
                w = master_module.weight.data
                w_abs = torch.abs(w)
                
                delta = self.threshold * torch.mean(w_abs)
                
                pos_mask = w > delta
                if pos_mask.any():
                    wp_init = torch.mean(w_abs[pos_mask])  # Mean of absolute weight
                else:
                    wp_init = torch.tensor(0.01)  # Fallback
                
                neg_mask = w < -delta
                if neg_mask.any():
                    wn_init = torch.mean(w_abs[neg_mask])  # Mean of absolute weight
                else:
                    wn_init = torch.tensor(0.01)  # Fallback
                
                # Store as learnable parameters
                self.wp_dict[name] = nn.Parameter(wp_init.to(self.device))
                self.wn_dict[name] = nn.Parameter(wn_init.to(self.device))
                
                self.quantized_layers.append(name)
                
                
                parts = name.split('.')
                layer_idx = parts[1] if len(parts) > 1 else '?'
                k = master_module.kernel_size
                k_str = f"{k[0]}x{k[1]}" if isinstance(k, tuple) else f"{k}x{k}"
                
                print(f"Init [{layer_idx:2s}] {k_str} {name}: Wp={wp_init:.4f}, Wn={wn_init:.4f}")
    
    def quantize_master_to_shadow(self):
        """
        Quantize master model weights into shadow model using current Wp/Wn.
        """
        with torch.no_grad():
            for name in self.quantized_layers:
                # Get modules
                master_module = dict(self.master_model.named_modules())[name]
                shadow_module = dict(self.shadow_model.named_modules())[name]
                
                # Get master weights and scaling factors
                w = master_module.weight.data
                wp = self.wp_dict[name].data
                wn = self.wn_dict[name].data
                
                # Compute threshold
                w_abs = torch.abs(w)
                delta = self.threshold * torch.mean(w_abs)
                
                # Quantize
                w_q = torch.zeros_like(w)
                pos_mask = w > delta
                neg_mask = w < -delta
                
                w_q[pos_mask] = wp
                w_q[neg_mask] = -wn
                
                # Copy to shadow model
                shadow_module.weight.data.copy_(w_q)
                
                # Copy bias (no quantization)
                if master_module.bias is not None:
                    shadow_module.bias.data.copy_(master_module.bias.data)
    
    def compute_ttq_gradients(self, shadow_grads):
        """
        Compute TTQ gradients with stabilization to prevent Wp/Wn explosion.
        """
        master_grads = {}
        wp_grads = {}
        wn_grads = {}
        
        for name in self.quantized_layers:
            master_module = dict(self.master_model.named_modules())[name]
            shadow_module = dict(self.shadow_model.named_modules())[name]
            
            if shadow_module.weight.grad is None:
                continue
            
            grad_wt = shadow_module.weight.grad.data
            w = master_module.weight.data
            wp = self.wp_dict[name].data
            wn = self.wn_dict[name].data
            
            # Compute masks
            w_abs = torch.abs(w)
            delta = self.threshold * torch.mean(w_abs)
            pos_mask = w > delta
            neg_mask = w < -delta
            zero_mask = torch.abs(w) <= delta
            
            # Count weights in each partition
            n_pos = pos_mask.sum().float().clamp(min=1.0)
            n_neg = neg_mask.sum().float().clamp(min=1.0)
            
            grad_w = torch.zeros_like(grad_wt)
            grad_w[pos_mask] = grad_wt[pos_mask] * wp
            grad_w[zero_mask] = grad_wt[zero_mask] * 1.0
            grad_w[neg_mask] = grad_wt[neg_mask] * wn
            
            master_grads[name] = grad_w
            
            if pos_mask.any():
                
                grad_wp_raw = torch.mean(grad_wt[pos_mask])
                
                grad_wp_normalized = grad_wp_raw / (wp.abs() + 1e-6)
                
                grad_wp_clipped = torch.clamp(grad_wp_normalized, -1.0, 1.0)
                
                wp_grads[name] = grad_wp_clipped
            else:
                wp_grads[name] = torch.tensor(0.0).to(self.device)
            
            if neg_mask.any():
                
                grad_wn_raw = torch.mean(-grad_wt[neg_mask])
                
                grad_wn_normalized = grad_wn_raw / (wn.abs() + 1e-6)
                
                grad_wn_clipped = torch.clamp(grad_wn_normalized, -1.0, 1.0)
                
                wn_grads[name] = grad_wn_clipped
            else:
                wn_grads[name] = torch.tensor(0.0).to(self.device)
        
        return master_grads, wp_grads, wn_grads
    
    def apply_gradients_to_master(self, master_grads):
        """Apply computed gradients to master model weights"""
        for name in self.quantized_layers:
            if name in master_grads:
                master_module = dict(self.master_model.named_modules())[name]
                # Set gradient manually
                if master_module.weight.grad is None:
                    master_module.weight.grad = master_grads[name].clone()
                else:
                    master_module.weight.grad.copy_(master_grads[name])
    
    def apply_gradients_to_scales(self, wp_grads, wn_grads):
        """Apply computed gradients to Wp/Wn parameters"""
        for name in self.quantized_layers:
            if name in wp_grads:
                if self.wp_dict[name].grad is None:
                    self.wp_dict[name].grad = wp_grads[name].clone().reshape(())
                else:
                    self.wp_dict[name].grad = wp_grads[name].clone().reshape(())
            
            if name in wn_grads:
                if self.wn_dict[name].grad is None:
                    self.wn_dict[name].grad = wn_grads[name].clone().reshape(())
                else:
                    self.wn_dict[name].grad = wn_grads[name].clone().reshape(())
    
    def get_scaling_parameters(self):
        """Get Wp/Wn parameters for optimizer"""
        return list(self.wp_dict.values()) + list(self.wn_dict.values())
    
    def export_ternary_model(self, save_path):
        """
        Export shadow model as standard YOLO checkpoint (ternary weights only).
        """
        # Final quantization
        self.quantize_master_to_shadow()
        
        # Save shadow model
        torch.save({
            'model': self.shadow_model,
            'quantized_layers': self.quantized_layers,
            'threshold': self.threshold
        }, save_path)
        
        print(f"Exported ternary model to {save_path}")
        print(f"This is a standard YOLO checkpoint with ternary weights")
        print(f"No custom layers required for inference!")
    
    def print_statistics(self):
        """Print Wp/Wn statistics"""
        wp_vals = [self.wp_dict[n].item() for n in self.quantized_layers]
        wn_vals = [self.wn_dict[n].item() for n in self.quantized_layers]
        
        print(f"\nScaling Factor Statistics:")
        print(f"  Wp: min={min(wp_vals):.4f}, max={max(wp_vals):.4f}, mean={np.mean(wp_vals):.4f}")
        print(f"  Wn: min={min(wn_vals):.4f}, max={max(wn_vals):.4f}, mean={np.mean(wn_vals):.4f}")
