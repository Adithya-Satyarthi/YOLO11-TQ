#!/usr/bin/env python3
"""
Diagnostic script to check if Wp/Wn are being trained
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.quantization.quantize_model import quantize_yolo_model
from src.quantization.ttq_layer import TTQConv2d

def check_ttq_parameters():
    print("="*70)
    print("TTQ Parameter Training Diagnostic")
    print("="*70)
    
    # Create a small quantized model
    print("\n[1] Creating quantized model...")
    model = quantize_yolo_model('yolo11n.pt', threshold=0.05, verbose=False)
    
    # Find all parameters
    print("\n[2] Checking parameter groups...")
    
    all_params = list(model.model.named_parameters())
    ttq_params = [(n, p) for n, p in all_params if 'Wp' in n or 'Wn' in n]
    regular_params = [(n, p) for n, p in all_params if 'Wp' not in n and 'Wn' not in n]
    
    print(f"  Total parameters: {len(all_params)}")
    print(f"  TTQ parameters (Wp/Wn): {len(ttq_params)}")
    print(f"  Regular parameters: {len(regular_params)}")
    
    # Check if TTQ parameters require gradients
    print("\n[3] Checking gradient requirements...")
    for name, param in ttq_params[:5]:
        print(f"  {name}: requires_grad={param.requires_grad}, value={param.item():.4f}")
    
    # Check if TTQ parameters are in optimizer
    print("\n[4] Creating optimizer...")
    from torch.optim import Adam
    
    # Standard approach (will it include Wp/Wn?)
    optimizer_standard = Adam(model.model.parameters(), lr=0.01)
    
    # Count parameters in optimizer
    std_param_count = sum(1 for group in optimizer_standard.param_groups for p in group['params'])
    print(f"  Standard optimizer has {std_param_count} parameters")
    
    # Check if Wp/Wn are included
    optimizer_param_ids = set(id(p) for group in optimizer_standard.param_groups for p in group['params'])
    ttq_in_optimizer = sum(1 for _, p in ttq_params if id(p) in optimizer_param_ids)
    
    print(f"  TTQ parameters in optimizer: {ttq_in_optimizer}/{len(ttq_params)}")
    
    if ttq_in_optimizer == len(ttq_params):
        print("\n✓ TTQ parameters ARE in optimizer - should be trained!")
    else:
        print("\n✗ TTQ parameters NOT in optimizer - will NOT be trained!")
        print("  This is why Wp/Wn stay at 1.0")
    
    # Test a forward/backward pass
    print("\n[5] Testing forward/backward pass...")
    
    model.model.train()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # Forward pass
    output = model.model(dummy_input)
    
    # Dummy loss (sum of outputs)
    loss = sum(o.sum() for o in output) if isinstance(output, (list, tuple)) else output.sum()
    
    # Backward pass
    optimizer_standard.zero_grad()
    loss.backward()
    
    # Check if gradients were computed for Wp/Wn
    print("\n[6] Checking gradients after backward pass...")
    has_grad_count = 0
    for name, param in ttq_params[:5]:
        has_grad = param.grad is not None and param.grad.abs().sum() > 0
        has_grad_count += has_grad
        grad_str = f"{param.grad.item():.6f}" if param.grad is not None else "None"
        print(f"  {name}: grad={grad_str}, has_grad={has_grad}")
    
    if has_grad_count > 0:
        print(f"\n✓ Gradients computed for {has_grad_count} TTQ parameters")
    else:
        print("\n✗ NO gradients for TTQ parameters!")
        print("  Custom autograd function may not be working")
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    check_ttq_parameters()
