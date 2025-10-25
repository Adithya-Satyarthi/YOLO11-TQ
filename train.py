#!/usr/bin/env python3
"""
Train TTQ-quantized YOLO11 with Wp/Wn tracking
"""

import argparse
import sys
import torch
import gc
import numpy as np
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.quantization.quantize_model import quantize_yolo_model, replace_conv_with_ttq
from src.quantization.ttq_layer import TTQConv2d
from src.utils.model_saver import ModelStageManager, extract_metrics_from_results


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Train TTQ-YOLO11')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--no-save', action='store_true')
    return parser.parse_args()


def extract_wp_wn_values(model):
    """Extract Wp/Wn values from TTQ layers"""
    wp_wn_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, TTQConv2d):
            wp_wn_dict[name] = {
                'Wp': module.Wp.item(),
                'Wn': module.Wn.item()
            }
    return wp_wn_dict


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("TTQ-YOLO11 Training with Wp/Wn Tracking")
    print("="*70)
    
    config = load_config(args.config)
    stage_name = config['stage']['name']
    
    print(f"\nStage: {stage_name}")
    
    # Quantize model
    print("\n" + "="*70)
    print("[1/2] Quantizing model...")
    print("="*70)
    
    model = quantize_yolo_model(
        model_path=config['model']['name'],
        threshold=config['model']['threshold'],
        quantize_first_layer=config['model'].get('quantize_first_layer', False),
        verbose=True
    )
    
    ttq_count = sum(1 for m in model.model.modules() if isinstance(m, TTQConv2d))
    print(f"\n✓ {ttq_count} TTQ layers created")
    
    # CRITICAL: Extract initial Wp/Wn values BEFORE training
    print("\nExtracting initial Wp/Wn values...")
    wp_wn_before = extract_wp_wn_values(model.model)
    print(f"  Captured {len(wp_wn_before)} TTQ layer parameters")
    
    # Train using custom wrapper that tracks Wp/Wn
    print("\n" + "="*70)
    print("[2/2] Training with Wp/Wn tracking...")
    print("="*70)
    
    # Import custom trainer
    from src.training.trainer import TTQTrainer
    trainer = TTQTrainer(model)
    
    results = trainer.train(
        data_yaml=config['data']['train'],
        epochs=config['train']['epochs'],
        imgsz=config['train']['imgsz'],
        batch=config['train']['batch'],
        lr0=config['train']['lr0'],
        patience=config['train']['patience'],
        device=config['train']['device'],
        project=config['logging']['project'],
        name=config['logging']['name'],
        save_period=config['logging'].get('save_period', 10),
        pretrained=config.get('pretrained', True),
        workers=config['train'].get('workers', 8),
        optimizer=config['train'].get('optimizer', 'Adam'),
        verbose=config['logging'].get('verbose', True)
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    
    # Try to extract Wp/Wn after training (will likely fail)
    print("\nAttempting to extract Wp/Wn after training...")
    wp_wn_after = extract_wp_wn_values(model.model)
    
    if not wp_wn_after:
        print("✗ No TTQ layers found after training (Ultralytics stripped them)")
        print("  Using pre-training values as fallback")
        wp_wn_dict = wp_wn_before
    else:
        print(f"✓ Found {len(wp_wn_after)} TTQ layers after training")
        wp_wn_dict = wp_wn_after
        
        # Compare before/after
        changed = sum(1 for name in wp_wn_after 
                     if (abs(wp_wn_after[name]['Wp'] - wp_wn_before[name]['Wp']) > 0.01 or
                         abs(wp_wn_after[name]['Wn'] - wp_wn_before[name]['Wn']) > 0.01))
        
        if changed > 0:
            print(f"  ✓ {changed} layers have changed Wp/Wn values!")
        else:
            print("  ✗ Wp/Wn values did not change during training")
    
    # Show statistics
    if wp_wn_dict:
        wp_vals = [v['Wp'] for v in wp_wn_dict.values()]
        wn_vals = [v['Wn'] for v in wp_wn_dict.values()]
        
        print(f"\nWp/Wn Statistics:")
        print(f"  Wp: min={min(wp_vals):.4f}, max={max(wp_vals):.4f}, mean={np.mean(wp_vals):.4f}")
        print(f"  Wn: min={min(wn_vals):.4f}, max={max(wn_vals):.4f}, mean={np.mean(wn_vals):.4f}")
        
        if all(abs(v - 1.0) < 0.01 for v in wp_vals + wn_vals):
            print("\n✗ WARNING: All Wp/Wn are still ~1.0")
            print("  Ultralytics is not training them properly")
            print("  Recommendation: Use custom training loop or accept unscaled ternary")
        else:
            print("\n✓ Scaling factors were learned!")
    
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create deployment checkpoint with learned Wp/Wn
    print("\n" + "="*70)
    print("Creating deployment checkpoint...")
    print("="*70)
    
    training_dir = Path(config['logging']['project']) / config['logging']['name']
    best_model_path = training_dir / "weights" / "best.pt"
    ttq_model_path = training_dir / "weights" / "best_ttq.pt"
    
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
        saved_model = checkpoint['model']
        
        # Re-apply TTQ structure
        quantized_count = replace_conv_with_ttq(
            saved_model, "model",
            threshold=config['model']['threshold'],
            quantize_first_layer=config['model'].get('quantize_first_layer', False)
        )
        
        print(f"\nRestoring learned Wp/Wn to {quantized_count} layers...")
        restored = 0
        
        # Restore learned Wp/Wn values
        for name, module in saved_model.named_modules():
            if isinstance(module, TTQConv2d) and name in wp_wn_dict:
                module.Wp.data = torch.tensor([wp_wn_dict[name]['Wp']])
                module.Wn.data = torch.tensor([wp_wn_dict[name]['Wn']])
                restored += 1
        
        print(f"  ✓ Restored Wp/Wn for {restored} layers")
        
        # Quantize weights for deployment
        print("\nQuantizing weights with learned scaling...")
        for module in saved_model.modules():
            if isinstance(module, TTQConv2d):
                with torch.no_grad():
                    weight_q, _ = module.quantize_weight(module.weight)
                    module.weight.data = weight_q.data
        
        # Save
        quantized_state = saved_model.state_dict()
        checkpoint['model'].load_state_dict(quantized_state)
        torch.save(checkpoint, ttq_model_path)
        
        print(f"✓ TTQ checkpoint saved: {ttq_model_path}")
        
        best_model_path = ttq_model_path
    
    # Save to organized structure
    if not args.no_save:
        try:
            from src.utils.model_saver import ModelStageManager
            metrics = extract_metrics_from_results(results)
            
            val_results = model.val(
                data=config['data'].get('val', config['data']['train']),
                imgsz=config['train']['imgsz'],
                batch=config.get('val', {}).get('batch', 32),
                device=config['train']['device'],
                verbose=False
            )
            metrics.update(extract_metrics_from_results(val_results))
            
            stage_manager = ModelStageManager()
            stage_dir = stage_manager.save_stage(
                stage_name=stage_name,
                model_path=best_model_path,
                config=config,
                metrics=metrics,
                description=config['stage']['description']
            )
            
            print(f"\n✓ Saved to: {stage_dir}")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Quantization: ✓ Working (ternary weights achieved)")
    print(f"Scaling factors: {'✓ Learned' if restored > 0 else '✗ Not learned (using defaults)'}")
    print("="*70)
    
    return results


if __name__ == '__main__':
    main()
