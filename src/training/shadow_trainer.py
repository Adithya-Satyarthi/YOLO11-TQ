"""
TTQ Training with Shadow Weights
Custom training loop - NO Ultralytics training, only dataloading
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.loss import v8DetectionLoss

from src.quantization.shadow_weight_manager import ShadowWeightManager


class ShadowTTQTrainer:
    """
    Custom TTQ trainer using shadow weight approach.
    Ultralytics used ONLY for dataloading.
    """
    
    def __init__(self, master_model, shadow_model, config, device='cuda'):
        self.master_model = master_model
        self.shadow_model = shadow_model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.master_model.model.to(self.device)
        self.shadow_model.model.to(self.device)
        
        # Enable gradients on master model
        self.master_model.model.train()
        for param in self.master_model.model.parameters():
            param.requires_grad = True
        
        # Shadow model needs gradients for backward pass
        self.shadow_model.model.train()
        for param in self.shadow_model.model.parameters():
            param.requires_grad = True
        
        # Initialize shadow weight manager
        self.shadow_manager = ShadowWeightManager(
            self.master_model.model,
            self.shadow_model.model,
            threshold=config['quantization']['threshold'],
            device=self.device,
            target_layers=config['quantization'].get('target_layers', None)
        )
        
        # Setup loss function
        self.compute_loss = v8DetectionLoss(self.shadow_model.model)
        
        # Set hyperparameters with VALUES
        if hasattr(self.compute_loss, 'hyp') and isinstance(self.compute_loss.hyp, dict):
            from types import SimpleNamespace
            hyp_dict = self.compute_loss.hyp
            hyp_dict.setdefault('box', 7.5)
            hyp_dict.setdefault('cls', 0.5)
            hyp_dict.setdefault('dfl', 1.5)
            self.compute_loss.hyp = SimpleNamespace(**hyp_dict)
        elif not hasattr(self.compute_loss, 'hyp'):
            from types import SimpleNamespace
            self.compute_loss.hyp = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        self.epoch = 0
        self.best_fitness = 0.0
        
        print("\n" + "="*70)
        print("Shadow Weight TTQ Trainer Initialized")
        print("="*70)
    
    def _setup_optimizer(self):
        """Setup optimizer with separate parameter groups"""
        # Master model parameters
        master_params = []
        for name, param in self.master_model.model.named_parameters():
            if param.requires_grad:
                master_params.append(param)
        
        # Scaling factors (Wp/Wn)
        scaling_params = self.shadow_manager.get_scaling_parameters()
        
        lr = self.config['train']['lr0']
        
        param_groups = [
            {
                'params': master_params,
                'lr': lr,
                'weight_decay': self.config['train'].get('weight_decay', 0.0005)
            },
            {
                'params': scaling_params,
                'lr': lr * 10,  # Higher LR for scales
                'weight_decay': 0.0
            }
        ]
        
        optimizer = torch.optim.Adam(param_groups)
        
        print(f"Optimizer: Adam")
        print(f"  Master params: {len(master_params)}, LR={lr}")
        print(f"  Scaling params: {len(scaling_params)}, LR={lr*10}")
        
        return optimizer
    
    def _build_dataloader(self, data_yaml, batch_size, mode='train'):
        """Build dataloader using Ultralytics"""
        overrides = {
            'data': data_yaml,
            'imgsz': self.config['train']['imgsz'],
            'batch': batch_size,
            'workers': self.config['train'].get('workers', 8),
        }
        
        cfg = get_cfg(overrides=overrides)
        data_dict = check_det_dataset(data_yaml)
        
        dataset = build_yolo_dataset(
            cfg,
            img_path=data_dict[mode],
            batch=batch_size,
            data=data_dict,
            mode=mode,
            stride=int(max(self.master_model.model.stride))
        )
        
        loader = build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.config['train'].get('workers', 8),
            shuffle=(mode == 'train'),
            rank=-1
        )
        
        return loader
    
    def train_epoch(self, train_loader):
        """Train one epoch with shadow weight approach"""
        self.master_model.model.train()
        self.shadow_model.model.train()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {self.epoch}")
        mloss = torch.zeros(3, device=self.device)
        
        for i, batch in pbar:
            # Prepare batch
            batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255.0
            batch['cls'] = batch['cls'].to(self.device)
            batch['bboxes'] = batch['bboxes'].to(self.device)
            batch['batch_idx'] = batch['batch_idx'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # STEP 1: Quantize master → shadow
            self.shadow_manager.quantize_master_to_shadow()
            
            # STEP 2: Forward pass on SHADOW model (NO AMP)
            pred = self.shadow_model.model(batch['img'])
            loss, loss_items = self.compute_loss(pred, batch)
            if loss.dim() > 0:
                loss = loss.sum()
            
            # STEP 3: Backward on shadow model (NO SCALER)
            loss.backward()
            
            # STEP 4: Manually compute TTQ gradients (Equations 7 & 8)
            master_grads, wp_grads, wn_grads = self.shadow_manager.compute_ttq_gradients(
                shadow_grads=None
            )
            
            # STEP 5: Apply gradients to master and Wp/Wn
            self.shadow_manager.apply_gradients_to_master(master_grads)
            self.shadow_manager.apply_gradients_to_scales(wp_grads, wn_grads)
            
            # STEP 6: Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.master_model.model.parameters(), max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(self.shadow_manager.get_scaling_parameters(), max_norm=10.0)
            
            # STEP 7: Optimizer step (NO SCALER)
            self.optimizer.step()
            
            # Update metrics
            mloss = (mloss * i + loss_items) / (i + 1)
            pbar.set_postfix({
                'loss': f'{mloss[0]:.4f}',
                'box': f'{mloss[1]:.4f}',
                'cls': f'{mloss[2]:.4f}'
            })
            
            # Debug: Print every 100 steps
            if i % 100 == 0 and i > 0:
                wp_vals = [self.shadow_manager.wp_dict[n].item() for n in self.shadow_manager.quantized_layers]
                wn_vals = [self.shadow_manager.wn_dict[n].item() for n in self.shadow_manager.quantized_layers]
                print(f"\n  Step {i}: Wp={wp_vals[0]:.4f}, Wn={wn_vals[0]:.4f}, Loss={mloss[0]:.4f}")
        
        return mloss
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validation using shadow model (quantized)"""
        self.shadow_manager.quantize_master_to_shadow()
        self.shadow_model.model.eval()
        
        # Use Ultralytics validation
        model_copy = deepcopy(self.shadow_model)
        
        try:
            results = model_copy.val(
                data=self.config['data']['train'],
                batch=self.config['val'].get('batch', 32),
                imgsz=self.config['train']['imgsz'],
                device=self.device,
                verbose=False
            )
            
            metrics = {
                'precision': float(results.box.p[-1] if hasattr(results.box, 'p') and len(results.box.p) else 0),
                'recall': float(results.box.r[-1] if hasattr(results.box, 'r') and len(results.box.r) else 0),
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
            }
        except Exception as e:
            print(f"Validation error: {e}")
            metrics = {'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0}
        
        del model_copy
        self.shadow_model.model.train()
        
        return metrics
    
    def train(self, epochs):
        """Main training loop"""
        print("\nStarting Shadow Weight TTQ Training")
        
        # Setup dataloaders
        train_loader = self._build_dataloader(
            self.config['data']['train'],
            self.config['train']['batch'],
            mode='train'
        )
        val_loader = self._build_dataloader(
            self.config['data'].get('val', self.config['data']['train']),
            self.config['val'].get('batch', 32),
            mode='val'
        )
        
        save_dir = Path('ttq_shadow_checkpoints') / self.config['logging']['name']
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 70)
            
            # Train
            mloss = self.train_epoch(train_loader)
            
            # Validate
            metrics = self.validate(val_loader)
            
            # Print stats
            print(f"  Loss: {mloss[0]:.4f} (box: {mloss[1]:.4f}, cls: {mloss[2]:.4f})")
            print(f"  mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50-95']:.4f}")
            
            self.shadow_manager.print_statistics()
            
            # Save checkpoint
            checkpoint_path = save_dir / f'epoch_{epoch+1}.pt'
            self.shadow_manager.export_ternary_model(checkpoint_path)
            
            # Save best
            if metrics['mAP50'] > self.best_fitness:
                self.best_fitness = metrics['mAP50']
                best_path = save_dir / 'best.pt'
                self.shadow_manager.export_ternary_model(best_path)
                print(f"  ✓ New best model! mAP50: {self.best_fitness:.4f}")
        
        print("\n" + "="*70)
        print(f"Training complete! Best mAP50: {self.best_fitness:.4f}")
        print("="*70)
        
        return metrics
