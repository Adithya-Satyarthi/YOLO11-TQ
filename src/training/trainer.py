"""
Training script for TTQ-quantized YOLO11
"""

import torch
from ultralytics import YOLO
import yaml


class TTQTrainer:
    """Trainer for TTQ-quantized YOLO models"""
    
    def __init__(self, model, config_path=None):
        self.model = model
        self.config = self.load_config(config_path) if config_path else {}
    
    def load_config(self, config_path):
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def train(self, data_yaml, epochs=100, imgsz=640, batch=16, 
              lr0=0.01, patience=50, save_period=10, **kwargs):
        """
        Train TTQ-quantized YOLO model.
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            lr0: Initial learning rate
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            **kwargs: Additional training arguments
        """
        # Training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'lr0': lr0,
            'patience': patience,
            'save_period': save_period,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'verbose': True,
            **kwargs
        }
        
        # Merge with config
        train_args.update(self.config.get('train', {}))
        
        print("\nStarting TTQ-YOLO11 Training...")
        print("="*60)
        print(f"Configuration:")
        for k, v in train_args.items():
            print(f"  {k}: {v}")
        print("="*60)
        
        # Train using Ultralytics API
        results = self.model.train(**train_args)
        
        return results
    
    def validate(self, data_yaml=None, **kwargs):
        """Validate model"""
        val_args = {
            'data': data_yaml,
            'verbose': True,
            **kwargs
        }
        return self.model.val(**val_args)


def setup_optimizer_for_ttq(model, lr=0.01, weight_decay=0.0002):
    """
    Setup optimizer with separate parameter groups for TTQ.
    Scaling factors (Wp, Wn) might need different learning rates.
    
    Args:
        model: TTQ-quantized model
        lr: Learning rate for regular parameters
        weight_decay: Weight decay
    
    Returns:
        Optimizer with parameter groups
    """
    # Separate parameters into groups
    regular_params = []
    scaling_params = []
    
    for name, param in model.named_parameters():
        if 'Wp' in name or 'Wn' in name:
            scaling_params.append(param)
        else:
            regular_params.append(param)
    
    param_groups = [
        {'params': regular_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': scaling_params, 'lr': lr * 10, 'weight_decay': 0}  # Higher LR for scaling
    ]
    
    return torch.optim.Adam(param_groups)
