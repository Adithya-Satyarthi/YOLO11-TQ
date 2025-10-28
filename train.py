#!/usr/bin/env python3
"""
Train TTQ-quantized YOLO11 with CUSTOM training loop
UPDATED: Supports multi-stage progressive quantization from single config
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


def load_config(config_path, stage=None):
    """
    Load config and extract specific stage if multi-stage config
    
    Args:
        config_path: Path to YAML config
        stage: Stage number (1, 2, 3) or None for single-stage config
    
    Returns:
        Processed config dict
    """
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Check if it's a multi-stage config
    if 'stages' in full_config:
        if stage is None:
            print("\n⚠ WARNING: Multi-stage config detected but no --stage specified!")
            print("Available stages:", list(full_config['stages'].keys()))
            print("Usage: python train.py --config <path> --stage <1|2|3>")
            sys.exit(1)
        
        stage_key = f'stage{stage}'
        if stage_key not in full_config['stages']:
            print(f"\n✗ ERROR: Stage {stage} not found in config!")
            print(f"Available stages: {list(full_config['stages'].keys())}")
            sys.exit(1)
        
        # Extract stage-specific config
        stage_config = full_config['stages'][stage_key].copy()
        
        # Merge with shared data config if exists
        if 'data' in full_config:
            stage_config['data'] = full_config['data']
        
        print(f"\n✓ Loaded multi-stage config - Stage {stage}")
        print(f"   Stage name: {stage_config['name']}")
        print(f"   Description: {stage_config['description']}")
        
        return stage_config
    else:
        # Single-stage config (backward compatibility)
        if stage is not None:
            print("\n⚠ WARNING: --stage specified but config is single-stage")
        return full_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train TTQ-YOLO11 with progressive quantization')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], default=None,
                       help='Stage number for progressive quantization (1, 2, or 3)')
    parser.add_argument('--no-save', action='store_true', help='Skip saving to organized directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    return parser.parse_args()


def extract_wp_wn_values(model):
    """Extract Wp/Wn values from TTQ layers"""
    wp_wn_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, TTQConv2d):
            wp_wn_dict[name] = {
                'Wp': module.Wp,
                'Wn': module.Wn
            }
    return wp_wn_dict


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("TTQ-YOLO11 Progressive Quantization Training")
    print("="*70)
    
    # Load config (handles multi-stage)
    config = load_config(args.config, args.stage)
    stage_name = config['logging']['name']
    
    print(f"\nStage: {stage_name}")
    if 'description' in config:
        print(f"Description: {config['description']}")
    
    # Display quantization strategy
    target_layers = config['quantization'].get('target_layers', None)
    if target_layers is not None:
        print(f"\n⚡ PROGRESSIVE QUANTIZATION")
        print(f"   Target layers: {target_layers}")
        print(f"   (All other layers remain in FP32)")
    else:
        print(f"\n⚡ FULL QUANTIZATION MODE")
    
    # Quantize model
    print("\n" + "="*70)
    print("[1/3] Quantizing model...")
    print("="*70)
    
    model = quantize_yolo_model(
        model_path=config['model']['weights'],
        threshold=config['quantization']['threshold'],
        quantize_first_layer=config['quantization'].get('quantize_first_layer', False),
        target_layers=target_layers,
        verbose=True
    )
    
    # Disable Ultralytics auto-save
    print("\nDisabling Ultralytics auto-save...")
    if hasattr(model, 'trainer'):
        model.trainer = None
        print("  ✓ Disabled trainer callbacks")
    if hasattr(model, 'callbacks'):
        model.callbacks = {}
        print("  ✓ Cleared model callbacks")
    if hasattr(model, 'overrides'):
        model.overrides['save'] = False
        model.overrides['save_period'] = -1
        print("  ✓ Disabled auto-save in overrides")
    model._custom_training = True
    
    ttq_count = sum(1 for m in model.model.modules() if isinstance(m, TTQConv2d))
    total_conv = sum(1 for m in model.model.modules() if isinstance(m, torch.nn.Conv2d))
    
    print(f"\n✓ {ttq_count} TTQ layers created ({ttq_count}/{total_conv} = {100*ttq_count/total_conv:.1f}% of Conv2d layers)")
    
    # Extract initial Wp/Wn values
    print("\nExtracting initial Wp/Wn values...")
    wp_wn_before = extract_wp_wn_values(model.model)
    
    if wp_wn_before:
        wp_vals = [v['Wp'] for v in wp_wn_before.values()]
        wn_vals = [v['Wn'] for v in wp_wn_before.values()]
        print(f"  Initial Wp: min={min(wp_vals):.4f}, max={max(wp_vals):.4f}, mean={np.mean(wp_vals):.4f}")
        print(f"  Initial Wn: min={min(wn_vals):.4f}, max={max(wn_vals):.4f}, mean={np.mean(wn_vals):.4f}")
    
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
    
    # Verify checkpoint
    print("\n" + "="*70)
    print("Verifying saved checkpoint...")
    print("="*70)
    
    training_dir = Path('ttq_checkpoints') / config['logging']['name']
    best_model_path = training_dir / "weights" / "best.pt"
    
    if best_model_path.exists():
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu')
            
            if 'model' in checkpoint and hasattr(checkpoint['model'], 'model'):
                ttq_in_checkpoint = sum(1 for m in checkpoint['model'].model.modules() 
                                       if isinstance(m, TTQConv2d))
                
                if ttq_in_checkpoint == ttq_count:
                    print(f"✓ SUCCESS: All {ttq_in_checkpoint} TTQ layers preserved in checkpoint!")
                else:
                    print(f"⚠ WARNING: Expected {ttq_count} TTQ layers, found {ttq_in_checkpoint}")
                    
        except Exception as e:
            print(f"✗ Error verifying checkpoint: {e}")
    else:
        print(f"✗ Checkpoint not found at: {best_model_path}")
    
    # Extract Wp/Wn after training
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    
    wp_wn_after = extract_wp_wn_values(model.model)
    changed = 0
    
    if wp_wn_after:
        changed = sum(1 for name in wp_wn_after 
                     if (abs(wp_wn_after[name]['Wp'] - wp_wn_before[name]['Wp']) > 0.01 or
                         abs(wp_wn_after[name]['Wn'] - wp_wn_before[name]['Wn']) > 0.01))
        
        wp_vals = [v['Wp'] for v in wp_wn_after.values()]
        wn_vals = [v['Wn'] for v in wp_wn_after.values()]
        
        print(f"\nFinal Wp/Wn Statistics:")
        print(f"  Wp: min={min(wp_vals):.4f}, max={max(wp_vals):.4f}, mean={np.mean(wp_vals):.4f}")
        print(f"  Wn: min={min(wn_vals):.4f}, max={max(wn_vals):.4f}, mean={np.mean(wn_vals):.4f}")
        print(f"  Changed: {changed}/{len(wp_wn_after)} layers")
    
    # Save to organized directory
    if not args.no_save and best_model_path.exists():
        print("\n" + "="*70)
        print("[3/3] Saving to organized directory...")
        print("="*70)
        
        try:
            stage_manager = ModelStageManager()
            stage_dir = stage_manager.save_stage(
                stage_name=stage_name,
                model_path=best_model_path,
                config=config,
                metrics=results,
                description=config.get('description', 'Progressive quantization')
            )
            print(f"✓ Saved to: {stage_dir}")
        except Exception as e:
            print(f"✗ Error saving: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Stage: {stage_name}")
    if target_layers:
        print(f"Quantization: Progressive - Layers {target_layers}")
    else:
        print(f"Quantization: Full (all eligible layers)")
    print(f"TTQ layers: {ttq_count}/{total_conv} Conv2d ({100*ttq_count/total_conv:.1f}%)")
    print(f"Scaling factors: {'✓ Learned' if changed > 0 else '⚠ Not learned'}")
    print(f"Best checkpoint: {best_model_path}")
    print("="*70)
    
    # Next stage suggestion
    if args.stage and args.stage < 3:
        next_stage = args.stage + 1
        print(f"\n💡 Next step: Run Stage {next_stage}")
        print(f"   python train.py --config {args.config} --stage {next_stage}")
    elif args.stage == 3:
        print(f"\n🎉 All stages complete! Final model ready for evaluation.")
    
    return results


if __name__ == '__main__':
    main()
