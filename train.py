#!/usr/bin/env python3
"""
Train TTQ-YOLO with Shadow Weight Approach
Supports dynamic model sizes with automatic path resolution
"""

import argparse
import torch
from pathlib import Path
import yaml
import sys
import re

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from src.training.shadow_trainer import ShadowTTQTrainer


def interpolate_config(config, context):
    """
    Recursively interpolate ${variable} references in config.
    
    Args:
        config: Dict or value to interpolate
        context: Dict containing variables for substitution
    
    Returns:
        Interpolated config
    """
    if isinstance(config, dict):
        return {k: interpolate_config(v, context) for k, v in config.items()}
    elif isinstance(config, list):
        return [interpolate_config(item, context) for item in config]
    elif isinstance(config, str):
        # Replace ${variable} or ${dict.key} with actual values
        def replacer(match):
            var_path = match.group(1)
            parts = var_path.split('.')
            
            value = context
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return match.group(0)  # Return original if not found
            
            return str(value)
        
        return re.sub(r'\$\{([^}]+)\}', replacer, config)
    else:
        return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--stage', type=int, required=True, help='Training stage (1, 2, or 3)')
    parser.add_argument('--model-size', type=str, default=None, 
                       help='Override model size (n, s, m, l, x). If not provided, uses config value.')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Override model size if provided via CLI
    if args.model_size:
        full_config['model_size'] = args.model_size
    
    # Build interpolation context
    context = {
        'model_size': full_config.get('model_size', 'n'),
        'base': full_config.get('base', {})
    }
    
    # Interpolate all variables
    full_config = interpolate_config(full_config, context)
    
    # Extract stage config
    stage_key = f'stage{args.stage}'
    if 'stages' not in full_config or stage_key not in full_config['stages']:
        raise ValueError(f"Stage {args.stage} not found in config")
    
    config = full_config['stages'][stage_key]
    
    # Merge data config from root
    if 'data' not in config:
        config['data'] = full_config['data']
    
    # Add model size to config for trainer access
    config['model_size'] = full_config['model_size']
    
    print("\n" + "="*80)
    print("TTQ-YOLO Shadow Weight Training")
    print("="*80)
    print(f"Model: YOLO11-{full_config['model_size'].upper()}")
    print(f"Stage: {args.stage}")
    print(f"Checkpoint directory: {config['logging']['save_dir']}")
    
    # Display quantization strategy
    target_layers = config['quantization'].get('target_layers', None)
    if target_layers:
        print(f"\nProgressive Quantization: Stage {args.stage}")
        print(f"Target layers: {target_layers}")
    else:
        print("\nFull Quantization Mode")
    
    print(f"Quantize 1x1 convolutions: {config['quantization'].get('quantize_1x1_conv', False)}")
    
    # Load TWO standard YOLO models
    print("\nLoading models...")
    model_weights = config['model']['weights']
    
    # Check if weights exist
    if not Path(model_weights).exists():
        # If stage 2/3 checkpoint doesn't exist, provide helpful error
        if args.stage > 1:
            raise FileNotFoundError(
                f"\n❌ Checkpoint not found: {model_weights}\n"
                f"   You need to complete Stage {args.stage - 1} first!\n"
                f"   Run: python train.py --config {args.config} --stage {args.stage - 1}"
            )
        else:
            raise FileNotFoundError(f"Model weights not found: {model_weights}")
    
    master_model = YOLO(model_weights)
    shadow_model = YOLO(model_weights)
    
    print(f"✓ Master model (FP32) loaded from {model_weights}")
    print(f"✓ Shadow model (ternary) loaded")
    print("  Both are standard YOLO models - no custom layers!")
    
    # Train
    trainer = ShadowTTQTrainer(
        master_model=master_model,
        shadow_model=shadow_model,
        config=config,
        device=config['train']['device']
    )
    
    results = trainer.train(epochs=config['train']['epochs'])
    
    print("\n" + "="*80)
    print("✓ Training complete!")
    print(f"  Final mAP50: {results['mAP50']:.4f}")
    print(f"  Final mAP50-95: {results['mAP50-95']:.4f}")
    print(f"  Checkpoint saved: {config['logging']['save_dir']}/best.pt")
    print("="*80)


if __name__ == '__main__':
    main()
