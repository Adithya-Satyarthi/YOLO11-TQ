#!/usr/bin/env python3


import argparse
import torch
from pathlib import Path
import yaml
import sys
import os
import warnings
warnings.filterwarnings('ignore', category=Warning)


sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from src.quantization.c2psa_bitlinear_ttq import replace_c2psa_with_bitlinear_shadow
from src.training.c2psa_trainer import C2PSABitLinearTTQTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config['train'].get('device', 'cuda')
    
    print("\n" + "="*80)
    print("C2PSA BitLinear_TTQ Training")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    
    
    config['train']['workers'] = 0
    print(f"  Forcing workers=0 to prevent memory issues")
    
    # Load models
    print("\nLoading master model (FP32)...")
    master_yolo = YOLO(args.model)
    master_model = master_yolo.model
    
    print(f"Moving master model to {device}...")
    master_model = master_model.to(device)
    print("   Master model moved to device")
    
    print("\nCreating shadow model (Quantized)...")
    shadow_yolo = YOLO(args.model)
    shadow_model = shadow_yolo.model
    
    print(f"Moving shadow model to {device}...")
    shadow_model = shadow_model.to(device)
    print("   Shadow model moved to device")
    
    # Replace C2PSA with BitLinear_TTQ in SHADOW model
    print("\nReplacing C2PSA in shadow model with BitLinear_TTQ...")
    shadow_model.model[10] = replace_c2psa_with_bitlinear_shadow(shadow_model.model[10])
    
    shadow_model.model[10] = shadow_model.model[10].to(device)
    print("   Replaced C2PSA moved to device")
    
    # Wrap back
    master_yolo.model = master_model
    shadow_yolo.model = shadow_model
    
    # Train
    print("\nInitializing trainer...")
    trainer = C2PSABitLinearTTQTrainer(
        master_model=master_yolo,
        shadow_model=shadow_yolo,
        config=config,
        device=device
    )
    
    results = trainer.train(epochs=config['train']['epochs'])
    
    print("\n" + "="*80)
    print(" Training Complete!")
    print(f"  mAP50: {results['mAP50']:.4f}")
    print(f"  mAP50-95: {results['mAP50-95']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
