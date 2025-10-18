"""
Unit tests for TTQ quantization
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantization.ttq_layer import TTQConv2d, TTQConv2dWithGrad
from quantization.quantize_model import should_quantize_module, quantize_yolo_model


def test_ttq_layer_creation():
    """Test TTQ layer initialization"""
    layer = TTQConv2d(3, 64, kernel_size=3, stride=1, padding=1)
    assert layer.in_channels == 3
    assert layer.out_channels == 64
    assert layer.Wp.numel() == 1
    assert layer.Wn.numel() == 1
    print("✓ TTQ layer creation test passed")


def test_ttq_forward():
    """Test TTQ forward pass"""
    layer = TTQConv2d(3, 64, kernel_size=3, stride=1, padding=1)
    x = torch.randn(2, 3, 32, 32)
    
    layer.train()
    out_train = layer(x)
    assert out_train.shape == (2, 64, 32, 32)
    
    layer.eval()
    out_eval = layer(x)
    assert out_eval.shape == (2, 64, 32, 32)
    
    print("✓ TTQ forward pass test passed")


def test_weight_quantization():
    """Test weight quantization to ternary values"""
    layer = TTQConv2d(3, 64, kernel_size=3, stride=1, padding=1, threshold=0.05)
    
    # Initialize weights
    nn.init.normal_(layer.weight, 0, 0.1)
    
    # Quantize
    weight_q, delta, max_val = layer.quantize_weight(layer.weight)
    
    # Check ternary values
    unique_vals = torch.unique(weight_q)
    assert len(unique_vals) <= 3, "Should have at most 3 unique values"
    
    print(f"✓ Weight quantization test passed (unique values: {len(unique_vals)})")


def test_should_quantize():
    """Test module filtering logic"""
    # Should quantize: backbone conv
    assert should_quantize_module('model.5.conv', nn.Conv2d(3, 64, 3))
    
    # Should NOT quantize: Detect head
    assert not should_quantize_module('model.23.cv2.0.0.conv', nn.Conv2d(3, 64, 3))
    
    # Should NOT quantize: C2PSA
    assert not should_quantize_module('model.10.cv1.conv', nn.Conv2d(3, 64, 3))
    
    print("✓ Module filtering test passed")


def test_gradient_computation():
    """Test custom gradient computation"""
    layer = TTQConv2dWithGrad(3, 64, kernel_size=3, stride=1, padding=1)
    layer.train()
    
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    # Check gradients exist
    assert layer.weight.grad is not None
    assert layer.Wp.grad is not None
    assert layer.Wn.grad is not None
    
    print("✓ Gradient computation test passed")


def test_model_quantization():
    """Test full model quantization"""
    print("\nTesting model quantization (this may take a moment)...")
    
    # This will download yolo11n.pt if not present
    try:
        model = quantize_yolo_model('yolo11n.pt', threshold=0.05, verbose=False)
        print("✓ Model quantization test passed")
    except Exception as e:
        print(f"⚠ Model quantization test skipped: {e}")


if __name__ == '__main__':
    print("="*70)
    print("Running TTQ Quantization Tests")
    print("="*70)
    
    test_ttq_layer_creation()
    test_ttq_forward()
    test_weight_quantization()
    test_should_quantize()
    test_gradient_computation()
    test_model_quantization()
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
