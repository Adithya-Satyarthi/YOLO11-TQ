#!/usr/bin/env python3
"""
TTQ-YOLO Hyperparameter Sweep with WandB
Standalone script that uses your existing codebase - no modifications needed!
"""

import argparse
import torch
import sys
from pathlib import Path
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from your existing codebase
from ultralytics import YOLO
from src.training.shadow_trainer import ShadowTTQTrainer


def train_with_config(config=None):
    """
    Train TTQ model with given hyperparameters.
    Called by WandB sweep agent.
    """
    with wandb.init(config=config):
        config = wandb.config

        print("=" * 80)
        print("TTQ-YOLO Hyperparameter Sweep")
        print("=" * 80)
        print(f"Threshold: {config.threshold}")
        print(f"Learning Rate: {config.lr0}")
        print(f"Scaling LR Factor: {config.scale_lr_factor}")
        print("=" * 80)

        # Build training config using your existing format
        train_config = {
            'model': {
                'weights': 'yolo11n.pt'
            },
            'data': {
                'train': 'coco_dataset.yaml',  # ✓ UPDATED
                'val': 'coco_dataset.yaml'     # ✓ UPDATED
            },
            'train': {
                'epochs': 1,  # Single epoch for sweep
                'batch': 8,
                'imgsz': 640,
                'lr0': config.lr0,
                'weight_decay': 0.0005,
                'workers': 8,
                'device': 'cuda'
            },
            'val': {
                'batch': 32
            },
            'quantization': {
                'threshold': config.threshold,
                'target_layers': [17, 19, 20, 22],  # Stage 1
                'scale_lr_factor': config.scale_lr_factor
            },
            'logging': {
                'name': f'sweep_t{config.threshold:.2f}_lr{config.lr0:.6f}_s{config.scale_lr_factor:.4f}'
            }
        }

        # Load models
        print("\nLoading models...")
        master_model = YOLO(train_config['model']['weights'])
        shadow_model = YOLO(train_config['model']['weights'])

        print("✓ Models loaded")

        # Initialize your existing trainer
        trainer = ShadowTTQTrainer(
            master_model=master_model,
            shadow_model=shadow_model,
            config=train_config,
            device=train_config['train']['device']
        )

        # Train for 1 epoch
        print("\nStarting training...")
        results = trainer.train(epochs=1)

        # Get Wp/Wn statistics
        wp_vals = [trainer.shadow_manager.wp_dict[n].item() 
                   for n in trainer.shadow_manager.quantized_layers]
        wn_vals = [trainer.shadow_manager.wn_dict[n].item() 
                   for n in trainer.shadow_manager.quantized_layers]

        # Log final metrics to WandB
        wandb.log({
            'final_mAP50': results['mAP50'],
            'final_mAP50_95': results['mAP50-95'],
            'final_precision': results['precision'],
            'final_recall': results['recall'],
            'avg_Wp': sum(wp_vals) / len(wp_vals),
            'avg_Wn': sum(wn_vals) / len(wn_vals),
            'max_Wp': max(wp_vals),
            'max_Wn': max(wn_vals),
            'min_Wp': min(wp_vals),
            'min_Wn': min(wn_vals),
            'wp_stable': max(wp_vals) < 1.0,  # Check if Wp/Wn didn't explode
            'wn_stable': max(wn_vals) < 1.0,
        })

        print("\n" + "=" * 80)
        print(f"Training complete!")
        print(f"  mAP50: {results['mAP50']:.4f}")
        print(f"  mAP50-95: {results['mAP50-95']:.4f}")
        print(f"  Avg Wp: {sum(wp_vals)/len(wp_vals):.4f}")
        print(f"  Avg Wn: {sum(wn_vals)/len(wn_vals):.4f}")
        print("=" * 80)


def create_sweep_config():
    """
    Create WandB sweep configuration with Bayesian optimization.
    """
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization (smarter than random/grid)
        'metric': {
            'name': 'final_mAP50',
            'goal': 'maximize'
        },
        'parameters': {
            'threshold': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 0.9
            },
            'lr0': {
                'distribution': 'log_uniform_values',
                'min': 0.0001,
                'max': 0.01
            },
            'scale_lr_factor': {
                'distribution': 'log_uniform_values',
                'min': 0.001,
                'max': 0.5
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 1,
            's': 2,
            'eta': 3,
            'max_iter': 1
        }
    }

    return sweep_config


def main():
    parser = argparse.ArgumentParser(description='TTQ-YOLO Hyperparameter Sweep')
    parser.add_argument('--sweep-id', type=str, default=None,
                        help='Existing sweep ID to join (optional)')
    parser.add_argument('--count', type=int, default=20,
                        help='Number of sweep runs to perform')
    parser.add_argument('--project', type=str, default='ttq-yolo-sweep',
                        help='WandB project name')
    parser.add_argument('--entity', type=str, default=None,
                        help='WandB entity (username/team)')

    args = parser.parse_args()

    if args.sweep_id:
        # Join existing sweep
        print(f"Joining existing sweep: {args.sweep_id}")
        sweep_id = args.sweep_id
    else:
        # Create new sweep
        sweep_config = create_sweep_config()

        print("=" * 80)
        print("Creating WandB Sweep")
        print("=" * 80)
        print(f"Project: {args.project}")
        if args.entity:
            print(f"Entity: {args.entity}")
        print(f"Method: {sweep_config['method']}")
        print(f"Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
        print("\nParameter Ranges:")
        for param, config in sweep_config['parameters'].items():
            if config['distribution'] == 'uniform':
                print(f"  {param}: [{config['min']}, {config['max']}]")
            elif config['distribution'] == 'log_uniform_values':
                print(f"  {param}: [{config['min']}, {config['max']}] (log scale)")
        print("=" * 80)

        sweep_id = wandb.sweep(
            sweep_config, 
            project=args.project,
            entity=args.entity
        )
        print(f"\nCreated sweep: {sweep_id}")

        if args.entity:
            print(f"View at: https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}")
        else:
            print(f"View at: https://wandb.ai/YOUR_USERNAME/{args.project}/sweeps/{sweep_id}")

    # Run sweep agent
    print(f"\nStarting sweep agent (running {args.count} trials)...")
    print("Press Ctrl+C to stop gracefully")
    print("=" * 80)

    wandb.agent(
        sweep_id, 
        function=train_with_config, 
        count=args.count,
        project=args.project,
        entity=args.entity
    )

    print("\n" + "=" * 80)
    print("Sweep complete!")
    if args.entity:
        print(f"View results at: https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}")
    else:
        print(f"View results at: https://wandb.ai/YOUR_USERNAME/{args.project}/sweeps/{sweep_id}")
    print("=" * 80)


if __name__ == '__main__':
    main()
