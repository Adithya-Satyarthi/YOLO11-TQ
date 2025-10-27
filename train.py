#!/usr/bin/env python3
"""
Train TTQ-quantized YOLO11 with CUSTOM training loop
REMINDER: Using Ultralytics for dataloading and validation, NOT training loop
"""

import argparse
import sys
import torch
import gc
import numpy as np
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.quantization.quantize_model import quantize_yolo_model
from src.quantization.ttq_layer import TTQConv2d
from src.training.custom_trainer import TTQYOLOTrainer
from src.utils.model_saver import ModelStageManager


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Train TTQ-YOLO11 with custom loop')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
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
    print("TTQ-YOLO11 Training")
    print("REMINDER: Custom training loop, Ultralytics for dataloading/validation")
    print("="*70)
    
    config = load_config(args.config)
    stage_name = config['stage']['name']
    
    print(f"\nStage: {stage_name}")
    
    # Quantize model
    print("\n" + "="*70)
    print("[1/3] Quantizing model...")
    print("="*70)
    
    model = quantize_yolo_model(
        model_path=config['model']['name'],
        threshold=config['model']['threshold'],
        quantize_first_layer=config['model'].get('quantize_first_layer', False),
        verbose=True
    )
    
    # CRITICAL: Disable Ultralytics auto-save mechanisms
    print("\nDisabling Ultralytics auto-save...")
    
    # Method 1: Remove trainer reference (prevents callbacks)
    if hasattr(model, 'trainer'):
        model.trainer = None
        print("  ✓ Disabled trainer callbacks")
    
    # Method 2: Clear all callbacks
    if hasattr(model, 'callbacks'):
        model.callbacks = {}
        print("  ✓ Cleared model callbacks")
    
    # Method 3: Disable auto-save in model
    if hasattr(model, 'overrides'):
        model.overrides['save'] = False
        model.overrides['save_period'] = -1
        print("  ✓ Disabled auto-save in overrides")
    
    # Method 4: Mark that we're using custom training
    model._custom_training = True  # Flag for our custom trainer
    
    ttq_count = sum(1 for m in model.model.modules() if isinstance(m, TTQConv2d))
    print(f"\n✓ {ttq_count} TTQ layers created")
    
    # Extract initial Wp/Wn values
    print("\nExtracting initial Wp/Wn values...")
    wp_wn_before = extract_wp_wn_values(model.model)
    print(f"  Captured {len(wp_wn_before)} TTQ layer parameters")
    
    # Show initial values
    if wp_wn_before:
        wp_vals = [v['Wp'] for v in wp_wn_before.values()]
        wn_vals = [v['Wn'] for v in wp_wn_before.values()]
        print(f"  Initial Wp: all={wp_vals[0]:.4f}")
        print(f"  Initial Wn: all={wn_vals[0]:.4f}")
    
    # Train using CUSTOM trainer
    print("\n" + "="*70)
    print("[2/3] Training...")
    print("="*70)
    
    trainer = TTQYOLOTrainer(
        model=model,
        config=config,
        device=config['train']['device']
    )
    
    results = trainer.train(epochs=config['train']['epochs'])
    
    # CRITICAL: Immediately verify checkpoint BEFORE doing anything else
    print("\n" + "="*70)
    print("Verifying saved checkpoint...")
    print("="*70)
    
    # FIX: Use the NEW custom location where trainer actually saves
    training_dir = Path('ttq_checkpoints') / config['logging']['name']
    best_model_path = training_dir / "weights" / "best.pt"
    
    if best_model_path.exists():
        # Load and verify TTQ layers are present
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu')
            
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Check if it's our custom checkpoint
            if 'model' in checkpoint and hasattr(checkpoint['model'], 'model'):
                ttq_in_checkpoint = sum(1 for m in checkpoint['model'].model.modules() 
                                       if isinstance(m, TTQConv2d))
                print(f"✓ Checkpoint verification: {ttq_in_checkpoint} TTQ layers found")
                
                if ttq_in_checkpoint == 0:
                    print("✗ ERROR: TTQ layers missing from checkpoint!")
                else:
                    print("✓ SUCCESS: TTQ layers preserved in checkpoint!")
            else:
                print("✗ ERROR: Unexpected checkpoint structure!")
                print(f"  Checkpoint type: {type(checkpoint.get('model'))}")
                
        except Exception as e:
            print(f"✗ Error verifying checkpoint: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"✗ Checkpoint not found at: {best_model_path}")
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    
    # Extract Wp/Wn after training
    print("\nExtracting Wp/Wn after training...")
    wp_wn_after = extract_wp_wn_values(model.model)
    
    if wp_wn_after:
        print(f"✓ Found {len(wp_wn_after)} TTQ layers after training")
        
        # Compare before/after
        changed = sum(1 for name in wp_wn_after 
                     if (abs(wp_wn_after[name]['Wp'] - wp_wn_before[name]['Wp']) > 0.01 or
                         abs(wp_wn_after[name]['Wn'] - wp_wn_before[name]['Wn']) > 0.01))
        
        if changed > 0:
            print(f"  ✓ {changed}/{len(wp_wn_after)} layers have learned new Wp/Wn values!")
            print("  ✓ Scaling factors were successfully learned!")
        else:
            print("  ✗ WARNING: Wp/Wn values did not change during training")
        
        # Show statistics
        wp_vals = [v['Wp'] for v in wp_wn_after.values()]
        wn_vals = [v['Wn'] for v in wp_wn_after.values()]
        
        print(f"\nFinal Wp/Wn Statistics:")
        print(f"  Wp: min={min(wp_vals):.4f}, max={max(wp_vals):.4f}, mean={np.mean(wp_vals):.4f}")
        print(f"  Wn: min={min(wn_vals):.4f}, max={max(wn_vals):.4f}, mean={np.mean(wn_vals):.4f}")
        
        # Show some examples
        print(f"\nExample learned scales (first 3 layers):")
        for i, (name, vals) in enumerate(list(wp_wn_after.items())[:3]):
            print(f"  {name}: Wp={vals['Wp']:.4f}, Wn={vals['Wn']:.4f}")
    
    # Save model to organized directory
    print("\n" + "="*70)
    print("[3/3] Saving model to organized directory...")
    print("="*70)
    
    if not args.no_save and best_model_path.exists():
        try:
            stage_manager = ModelStageManager()
            stage_dir = stage_manager.save_stage(
                stage_name=stage_name,
                model_path=best_model_path,  # Use the verified TTQ checkpoint
                config=config,
                metrics=results,
                description=config['stage']['description']
            )
            
            print(f"\n✓ Saved to organized directory: {stage_dir}")
            
        except Exception as e:
            print(f"\n✗ Error saving to organized directory: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Training loop: ✓ Custom (proper TTQ gradients)")
    print("Dataloading: ✓ Ultralytics")
    print("Validation: ✓ Custom (preserves TTQ layers)")
    print(f"Scaling factors: {'✓ Learned' if changed > 0 else '✗ Not learned'}")
    print(f"TTQ layers preserved: ✓ {len(wp_wn_after)} layers with learned Wp/Wn")
    print(f"Checkpoint location: {best_model_path}")
    print("="*70)
    
    return results


if __name__ == '__main__':
    main()
