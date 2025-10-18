#!/usr/bin/env python3
"""
Train TTQ-quantized YOLO11 model
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quantization.quantize_model import quantize_yolo_model
from training.trainer import TTQTrainer


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='Train TTQ-YOLO11')
    
    # Config file (primary way to set parameters)
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file (default: configs/train_config.yaml)')
    
    # Optional overrides
    parser.add_argument('--model', type=str, default=None,
                       help='Override model from config')
    parser.add_argument('--data', type=str, default=None,
                       help='Override data from config')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override epochs from config')
    parser.add_argument('--batch', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--device', type=str, default=None,
                       help='Override device from config')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("TTQ-YOLO11 Training Script")
    print("="*70)
    
    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.model:
        config['model']['name'] = args.model
    if args.data:
        config['data']['train'] = args.data
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.batch:
        config['train']['batch'] = args.batch
    if args.device:
        config['train']['device'] = args.device
    
    print("\nConfiguration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Dataset: {config['data']['train']}")
    print(f"  Epochs: {config['train']['epochs']}")
    print(f"  Batch size: {config['train']['batch']}")
    print(f"  Threshold: {config['model']['threshold']}")
    print(f"  Device: {config['train']['device']}")
    
    # Step 1: Quantize model
    print("\n" + "="*70)
    print("[1/2] Quantizing YOLO11 model...")
    print("="*70)
    
    model = quantize_yolo_model(
        model_path=config['model']['name'],
        threshold=config['model']['threshold'],
        quantize_first_layer=config['model'].get('quantize_first_layer', False),
        verbose=True
    )
    
    # Step 2: Train model
    print("\n" + "="*70)
    print("[2/2] Starting training...")
    print("="*70)
    
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
    print(f"Results saved to: {config['logging']['project']}/{config['logging']['name']}")
    print("="*70)
    
    return results


if __name__ == '__main__':
    main()
