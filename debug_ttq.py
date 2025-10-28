#!/usr/bin/env python3
"""
Standalone TTQ Debug Script - UPDATED for mean-based threshold
Tests TTQ quantization layer-by-layer to find where mAP drops to 0
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from src.quantization.ttq_layer import TTQConv2d


def test_baseline_fp32():
    """Test baseline FP32 model"""
    print("\n" + "="*70)
    print("TEST 1: Baseline FP32 Model")
    print("="*70)
    
    model = YOLO('yolo11n.pt')
    results = model.val(data='coco8.yaml', batch=4, verbose=False)
    
    print(f"✓ FP32 Baseline mAP50: {results.box.map50:.4f}")
    return results.box.map50


def test_single_layer_quantization():
    """Test quantizing ONLY ONE layer to isolate the problem"""
    print("\n" + "="*70)
    print("TEST 2: Single Layer Quantization (model.2.cv1.conv)")
    print("="*70)
    
    model = YOLO('yolo11n.pt')
    
    # Quantize ONLY model.2.cv1.conv
    target_module = model.model.model[2].cv1.conv
    
    print(f"Original layer: {type(target_module)}")
    print(f"  Weight shape: {target_module.weight.shape}")
    print(f"  Weight stats: min={target_module.weight.min():.4f}, max={target_module.weight.max():.4f}")
    
    # UPDATED: Use mean-based threshold (0.7 * mean)
    w_abs = torch.abs(target_module.weight.data)
    delta_old = 0.05 * torch.max(w_abs)
    delta_new = 0.7 * torch.mean(w_abs)
    
    print(f"\n  Threshold comparison:")
    print(f"    OLD (max-based): delta = 0.05 * max(|W|) = {delta_old:.4f}")
    print(f"    NEW (mean-based): delta = 0.7 * mean(|W|) = {delta_new:.4f}")
    
    # Use NEW threshold
    delta = delta_new
    pos_mask = target_module.weight.data > delta
    neg_mask = target_module.weight.data < -delta
    
    Wp_init = torch.mean(w_abs[pos_mask]) if pos_mask.any() else torch.mean(w_abs)
    Wn_init = torch.mean(w_abs[neg_mask]) if neg_mask.any() else torch.mean(w_abs)
    
    print(f"\n  Using NEW threshold:")
    print(f"  Wp init: {Wp_init:.4f}, Wn init: {Wn_init:.4f}")
    print(f"  Threshold delta: {delta:.4f}")
    print(f"  Positive weights: {pos_mask.sum()}/{target_module.weight.numel()} ({100*pos_mask.sum()/target_module.weight.numel():.1f}%)")
    print(f"  Negative weights: {neg_mask.sum()}/{target_module.weight.numel()} ({100*neg_mask.sum()/target_module.weight.numel():.1f}%)")
    print(f"  Zero weights: {(~pos_mask & ~neg_mask).sum()}/{target_module.weight.numel()} ({100*(~pos_mask & ~neg_mask).sum()/target_module.weight.numel():.1f}%)")
    
    # Replace with TTQ (threshold=0.7 means mean-based)
    ttq_layer = TTQConv2d(
        in_channels=target_module.in_channels,
        out_channels=target_module.out_channels,
        kernel_size=target_module.kernel_size,
        stride=target_module.stride,
        padding=target_module.padding,
        pretrained_weight=target_module.weight.data.clone(),
        threshold=0.7  # UPDATED: mean-based threshold
    )
    
    if target_module.bias is not None:
        ttq_layer.bias.data.copy_(target_module.bias.data)
    
    model.model.model[2].cv1.conv = ttq_layer
    
    # Test accuracy
    results = model.val(data='coco8.yaml', batch=4, verbose=False)
    
    print(f"✓ Single Layer TTQ mAP50: {results.box.map50:.4f}")
    return results.box.map50


def test_ttq_forward_pass():
    """Test TTQ forward pass to verify quantization is correct"""
    print("\n" + "="*70)
    print("TEST 3: TTQ Forward Pass Verification")
    print("="*70)
    
    # Create a simple weight tensor
    W = torch.randn(16, 16, 3, 3)
    print(f"Original weight stats:")
    print(f"  min={W.min():.4f}, max={W.max():.4f}, mean={W.mean():.4f}")
    
    # UPDATED: Compare old vs new threshold
    w_abs = torch.abs(W)
    
    # OLD method
    threshold_old = 0.05
    delta_old = threshold_old * torch.max(w_abs)
    pos_old = (W > delta_old).sum()
    neg_old = (W < -delta_old).sum()
    zero_old = W.numel() - pos_old - neg_old
    
    # NEW method
    threshold_new = 0.7
    delta_new = threshold_new * torch.mean(w_abs)
    pos_new = (W > delta_new).sum()
    neg_new = (W < -delta_new).sum()
    zero_new = W.numel() - pos_new - neg_new
    
    print(f"\nOLD (max-based):")
    print(f"  delta = 0.05 * max = {delta_old:.4f}")
    print(f"  Zeros: {zero_old}/{W.numel()} ({100*zero_old/W.numel():.1f}%)")
    
    print(f"\nNEW (mean-based):")
    print(f"  delta = 0.7 * mean = {delta_new:.4f}")
    print(f"  Zeros: {zero_new}/{W.numel()} ({100*zero_new/W.numel():.1f}%)")
    
    # Use NEW method
    pos_mask = W > delta_new
    neg_mask = W < -delta_new
    
    Wp = torch.mean(w_abs[pos_mask]) if pos_mask.any() else torch.tensor(1.0)
    Wn = torch.mean(w_abs[neg_mask]) if neg_mask.any() else torch.tensor(1.0)
    
    print(f"\nUsing NEW method:")
    print(f"  Wp={Wp:.4f}, Wn={Wn:.4f}, delta={delta_new:.4f}")
    
    # Quantize
    W_ternary = torch.zeros_like(W)
    W_ternary[pos_mask] = Wp
    W_ternary[neg_mask] = -Wn
    
    print(f"\nTernary weight stats:")
    print(f"  Unique values: {torch.unique(W_ternary).numel()}")
    print(f"  min={W_ternary.min():.4f}, max={W_ternary.max():.4f}")
    print(f"  Distribution: +Wp: {pos_mask.sum()}, 0: {(~pos_mask & ~neg_mask).sum()}, -Wn: {neg_mask.sum()}")
    
    # Test convolution output
    x = torch.randn(1, 16, 32, 32)
    
    conv_fp32 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
    conv_fp32.weight.data = W.clone()
    
    conv_ttq = nn.Conv2d(16, 16, 3, padding=1, bias=False)
    conv_ttq.weight.data = W_ternary.clone()
    
    y_fp32 = conv_fp32(x)
    y_ttq = conv_ttq(x)
    
    mse = torch.mean((y_fp32 - y_ttq) ** 2).item()
    relative_error = mse / torch.mean(y_fp32 ** 2).item()
    
    print(f"\nConvolution output comparison:")
    print(f"  FP32 output: mean={y_fp32.mean():.4f}, std={y_fp32.std():.4f}")
    print(f"  TTQ output: mean={y_ttq.mean():.4f}, std={y_ttq.std():.4f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Relative error: {relative_error:.6f} ({100*relative_error:.2f}%)")
    
    if relative_error > 0.5:
        print("  ⚠ WARNING: Relative error > 50%! Quantization may be too aggressive!")
    else:
        print("  ✓ Relative error is acceptable")
    
    return relative_error


def test_gradient_flow():
    """Test if gradients flow correctly through TTQ layer"""
    print("\n" + "="*70)
    print("TEST 4: Gradient Flow Test")
    print("="*70)
    
    # Create TTQ layer
    W = torch.randn(8, 8, 3, 3, requires_grad=True)
    
    ttq_layer = TTQConv2d(
        in_channels=8,
        out_channels=8,
        kernel_size=3,
        padding=1,
        pretrained_weight=W.data.clone(),
        threshold=0.7  # UPDATED: mean-based
    )
    
    # FIXED: Wp/Wn are now properties that return float
    print(f"Initial Wp: {ttq_layer.Wp:.4f}, Wn: {ttq_layer.Wn:.4f}")
    
    # Forward pass
    x = torch.randn(2, 8, 16, 16)
    y = ttq_layer(x)
    loss = y.sum()
    
    # Backward pass
    loss.backward()
    
    print(f"Gradients:")
    print(f"  weight.grad: exists={ttq_layer.weight.grad is not None}, norm={ttq_layer.weight.grad.norm().item():.4f}")
    
    # Check actual parameter gradients (Wp_param and Wn_param)
    wp_grad_exists = ttq_layer.Wp_param.grad is not None
    wn_grad_exists = ttq_layer.Wn_param.grad is not None
    
    print(f"  Wp_param.grad: exists={wp_grad_exists}", end="")
    if wp_grad_exists:
        print(f", value={ttq_layer.Wp_param.grad.item():.6f}")
    else:
        print()
    
    print(f"  Wn_param.grad: exists={wn_grad_exists}", end="")
    if wn_grad_exists:
        print(f", value={ttq_layer.Wn_param.grad.item():.6f}")
    else:
        print()
    
    if not wp_grad_exists or not wn_grad_exists:
        print("  ✗ ERROR: Wp/Wn parameter gradients are None!")
        return False
    
    if abs(ttq_layer.Wp_param.grad.item()) < 1e-6 and abs(ttq_layer.Wn_param.grad.item()) < 1e-6:
        print("  ⚠ WARNING: Wp/Wn gradients are very small!")
        return False
    
    print("  ✓ Gradients flow correctly through Wp_param and Wn_param")
    return True


def test_threshold_verification():
    """Verify that TTQConv2d is using mean-based threshold"""
    print("\n" + "="*70)
    print("TEST 5: Threshold Implementation Verification")
    print("="*70)
    
    W = torch.randn(32, 32, 1, 1)
    
    # Create TTQ layer
    ttq = TTQConv2d(32, 32, 1, pretrained_weight=W, threshold=0.7)
    
    # Forward pass to trigger quantization
    x = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        y = ttq(x)
        
        # Get quantized weights
        from src.quantization.ttq_layer import TTQWeightFunction
        w_q = TTQWeightFunction.apply(
            ttq.weight, 
            torch.tensor(ttq.Wp), 
            torch.tensor(ttq.Wn), 
            ttq.threshold
        )
        
        zeros = (w_q == 0).sum().item()
        total = w_q.numel()
        zero_pct = 100 * zeros / total
        
        print(f"Quantized weights:")
        print(f"  Total: {total}")
        print(f"  Zeros: {zeros} ({zero_pct:.1f}%)")
        print(f"  Non-zeros: {total - zeros} ({100 - zero_pct:.1f}%)")
        
        if zero_pct > 50:
            print(f"  ✗ FAIL: >50% zeros indicates OLD max-based threshold still in use!")
            return False
        elif zero_pct < 40:
            print(f"  ✓ SUCCESS: ~30-40% zeros indicates NEW mean-based threshold working!")
            return True
        else:
            print(f"  ⚠ BORDERLINE: Zero percentage is in transition range")
            return True


def main():
    """Run all debug tests"""
    print("\n" + "="*70)
    print("TTQ DEBUG SUITE - Finding the Root Cause (UPDATED)")
    print("="*70)
    
    results = {}
    
    # Test 1: Baseline
    try:
        results['baseline'] = test_baseline_fp32()
    except Exception as e:
        print(f"✗ Baseline test failed: {e}")
        results['baseline'] = None
    
    # Test 2: Single layer
    try:
        results['single_layer'] = test_single_layer_quantization()
    except Exception as e:
        print(f"✗ Single layer test failed: {e}")
        import traceback
        traceback.print_exc()
        results['single_layer'] = None
    
    # Test 3: Forward pass
    try:
        results['forward_pass'] = test_ttq_forward_pass()
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        results['forward_pass'] = None
    
    # Test 4: Gradients
    try:
        results['gradients'] = test_gradient_flow()
    except Exception as e:
        print(f"✗ Gradient test failed: {e}")
        import traceback
        traceback.print_exc()
        results['gradients'] = None
    
    # Test 5: Threshold verification
    try:
        results['threshold_check'] = test_threshold_verification()
    except Exception as e:
        print(f"✗ Threshold test failed: {e}")
        import traceback
        traceback.print_exc()
        results['threshold_check'] = None
    
    # Summary
    print("\n" + "="*70)
    print("DEBUG SUMMARY")
    print("="*70)
    print(f"Baseline mAP50: {results['baseline']:.4f}" if results['baseline'] else "Baseline: FAILED")
    print(f"Single layer mAP50: {results['single_layer']:.4f}" if results['single_layer'] else "Single layer: FAILED")
    print(f"Forward pass relative error: {results['forward_pass']:.4f}" if results['forward_pass'] else "Forward pass: FAILED")
    print(f"Gradient flow: {'✓ PASS' if results['gradients'] else '✗ FAIL'}")
    print(f"Threshold check: {'✓ PASS' if results['threshold_check'] else '✗ FAIL'}")
    
    # Diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)
    
    if results['threshold_check'] == False:
        print("✗ CRITICAL: Still using OLD max-based threshold!")
        print("  You must update ttq_layer.py:")
        print("    Line 69: delta = self.threshold * torch.mean(w_abs)")
        print("    Line 167: delta = threshold * torch.mean(w_abs)")
    
    if results['single_layer'] is not None:
        if results['single_layer'] < 0.5:
            print("✗ PROBLEM: Single layer quantization drops mAP significantly")
            print("  Expected with NEW threshold: mAP50 > 0.65")
            print("  Your result: mAP50 = {:.4f}".format(results['single_layer']))
        else:
            print("✓ SUCCESS: Single layer quantization preserves most accuracy!")
            print(f"  Baseline: 0.8468 → TTQ: {results['single_layer']:.4f}")
            print(f"  Drop: {100*(0.8468 - results['single_layer'])/0.8468:.1f}% (acceptable!)")
    
    if results['gradients'] == True:
        print("✓ Gradients flow correctly - training should work!")
    else:
        print("✗ Gradients not flowing - training won't work!")
    
    print("="*70)


if __name__ == '__main__':
    main()
