#!/usr/bin/env python3
"""
Train TTQ-YOLO with Shadow Weight Approach
"""

import argparse
import torch
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from src.training.shadow_trainer import ShadowTTQTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--stage', type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Extract stage config if multi-stage
    if 'stages' in full_config and args.stage:
        config = full_config['stages'][f'stage{args.stage}']
        # Merge data config from root
        if 'data' not in config:
            config['data'] = full_config['data']
    else:
        config = full_config
    
    print("\n" + "="*70)
    print("TTQ-YOLO Shadow Weight Training")
    print("="*70)
    
    # Display quantization strategy
    target_layers = config['quantization'].get('target_layers', None)
    if target_layers:
        print(f"\nProgressive Quantization: Stage {args.stage}")
        print(f"Target layers: {target_layers}")
    else:
        print("\nFull Quantization Mode")
    
    # Load TWO standard YOLO models
    print("\nLoading models...")
    master_model = YOLO(config['model']['weights'])
    shadow_model = YOLO(config['model']['weights'])
    
    print("✓ Master model (FP32) loaded")
    print("✓ Shadow model (ternary) loaded")
    print("  Both are standard YOLO models - no custom layers!")
    
    # Train
    trainer = ShadowTTQTrainer(
        master_model=master_model,
        shadow_model=shadow_model,
        config=config,
        device=config['train']['device']
    )
    
    results = trainer.train(epochs=config['train']['epochs'])
    
    print("\n✓ Training complete!")
    print(f"  Final mAP50: {results['mAP50']:.4f}")


if __name__ == '__main__':
    main()
