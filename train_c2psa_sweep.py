#!/usr/bin/env python3
"""
WandB Sweep Script for BitLinear_TTQ Hyperparameter Tuning
Finds optimal: lr, threshold, batch size, weight decay, momentum
No modifications needed to existing files!
"""

import argparse
import torch
from pathlib import Path
import yaml
import sys
import os
import warnings

warnings.filterwarnings('ignore', category=Warning)

#os.environ['OMP_NUM_THREADS'] = '1'
#torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).parent))

import wandb
from ultralytics import YOLO
from src.quantization.c2psa_bitlinear_ttq import replace_c2psa_with_bitlinear_shadow
from src.quantization.bitlinear_ttq_manager import BitLinearTTQManager


def sweep_train():
    """Main training function for WandB sweep"""
    
    # Initialize WandB
    wandb.init()
    
    # Get hyperparameters from sweep
    sweep_config = wandb.config
    
    print("\n" + "="*80)
    print("WandB Sweep: BitLinear_TTQ Hyperparameter Tuning")
    print("="*80)
    print(f"\nSweep Hyperparameters:")
    print(f"  lr0: {sweep_config.lr0}")
    print(f"  threshold: {sweep_config.threshold}")
    print(f"  batch: {sweep_config.batch}")
    print(f"  epochs: {sweep_config.epochs}")
    print(f"  weight_decay: {sweep_config.weight_decay}")
    print(f"  momentum: {sweep_config.momentum}")
    
    # Load config
    with open('configs/stage2_c2psa.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with sweep parameters
    config['train']['lr0'] = sweep_config.lr0
    config['train']['batch'] = sweep_config.batch
    config['train']['epochs'] = sweep_config.epochs
    config['quantization']['threshold'] = sweep_config.threshold
    
    device = config['train'].get('device', 'cuda')
    
    # Load models
    print("\nLoading models...")
    master_yolo = YOLO('yolo11n.pt')
    master_model = master_yolo.model.to(device)
    
    shadow_yolo = YOLO('yolo11n.pt')
    shadow_model = shadow_yolo.model.to(device)
    shadow_model.model[10] = replace_c2psa_with_bitlinear_shadow(shadow_model.model[10])
    shadow_model.model[10] = shadow_model.model[10].to(device)
    
    master_yolo.model = master_model
    shadow_yolo.model = shadow_model
    
    # Initialize trainer
    from src.training.c2psa_trainer import C2PSABitLinearTTQTrainer
    
    trainer = C2PSABitLinearTTQTrainer(
        master_model=master_yolo,
        shadow_model=shadow_yolo,
        config=config,
        device=device
    )
    
    # Override learning rate and weight decay
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = sweep_config.lr0
        param_group['weight_decay'] = sweep_config.weight_decay
    
    # Train
    print(f"\nStarting training with sweep parameters...")
    trainer.use_wandb = True
    
    # Build dataloaders
    train_loader = trainer._build_dataloader(
        config['data']['train'],
        sweep_config.batch,
        mode='train'
    )
    
    val_loader = trainer._build_dataloader(
        config['data']['train'],
        sweep_config.batch,
        mode='val'
    )
    
    # Training loop
    best_mAP50 = 0.0
    patience = config['train'].get('patience', 5)
    patience_counter = 0
    
    for epoch in range(sweep_config.epochs):
        trainer.epoch = epoch
        
        print(f"\nEpoch {epoch + 1}/{sweep_config.epochs}")
        print("-" * 70)
        
        # Train epoch
        mloss = trainer.train_epoch(train_loader)
        print(f"Loss: {mloss[0]:.4f} (box: {mloss[1]:.4f}, cls: {mloss[2]:.4f})")
        
        # Validate
        metrics = trainer.validate()
        print(f"mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50-95']:.4f}")
        
        # Log to WandB
        wandb.log({
            'epoch': epoch,
            'train/loss': mloss[0].item(),
            'train/box_loss': mloss[1].item(),
            'train/cls_loss': mloss[2].item(),
            'val/mAP50': metrics['mAP50'],
            'val/mAP50-95': metrics['mAP50-95'],
            'val/precision': metrics['precision'],
            'val/recall': metrics['recall'],
            'hyperparameters/lr': sweep_config.lr0,
            'hyperparameters/threshold': sweep_config.threshold,
            'hyperparameters/batch': sweep_config.batch,
            'hyperparameters/weight_decay': sweep_config.weight_decay,
        })
        
        # Early stopping
        if metrics['mAP50'] > best_mAP50:
            best_mAP50 = metrics['mAP50']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    print("\n" + "="*80)
    print(f"✅ Sweep trial complete!")
    print(f"   Best mAP50: {best_mAP50:.4f}")
    print("="*80)


if __name__ == '__main__':
    sweep_train()
