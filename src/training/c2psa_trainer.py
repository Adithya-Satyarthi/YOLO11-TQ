# src/training/c2psa_trainer.py

"""
C2PSA BitLinear_TTQ Trainer
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

from src.quantization.bitlinear_ttq_manager import BitLinearTTQManager

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class C2PSABitLinearTTQTrainer:
    """
    Custom trainer for C2PSA with BitLinear_TTQ quantization.
    """
    
    def __init__(self, master_model, shadow_model, config, device='cuda'):
        self.master_model = master_model
        self.shadow_model = shadow_model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.master_model.model.to(self.device)
        self.shadow_model.model.to(self.device)
        
        # Enable gradients
        self.master_model.model.train()
        for param in self.master_model.model.parameters():
            param.requires_grad = True
        
        self.shadow_model.model.train()
        for param in self.shadow_model.model.parameters():
            param.requires_grad = True
        
        # Initialize BitLinear TTQ manager
        self.bitlinear_manager = BitLinearTTQManager(
            self.master_model.model,
            self.shadow_model.model,
            threshold=config['quantization'].get('threshold', 0.7),
            device=self.device
        )
        
        # Setup loss
        self.compute_loss = v8DetectionLoss(self.shadow_model.model)
        
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
        
        self.optimizer = self._setup_optimizer()
        
        self.epoch = 0
        self.best_fitness = 0.0
        self.patience = config['train'].get('patience', None)
        self.patience_counter = 0
        
        self.use_wandb = config['logging'].get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb(config)
        
        print("\n" + "="*70)
        print("C2PSA BitLinear_TTQ Trainer Initialized")
        print("="*70)
    
    def _init_wandb(self, config):
        wandb_config = {
            'stage': config.get('name', 'unknown'),
            'batch_size': config['train']['batch'],
            'lr': config['train']['lr0'],
            'epochs': config['train']['epochs'],
            'threshold': config['quantization'].get('threshold', 0.7),
        }
        
        wandb.init(
            project=config['logging'].get('wandb_project', 'ttq-yolo-c2psa'),
            entity=config['logging'].get('wandb_entity', None),
            name=config['logging']['name'],
            config=wandb_config,
            reinit=True
        )
    
    def _setup_optimizer(self):
        master_params = []
        for name, param in self.master_model.model.named_parameters():
            if 'model.10' in name and param.requires_grad:
                master_params.append(param)
        
        scaling_params = self.bitlinear_manager.get_scaling_parameters()
        
        lr = self.config['train']['lr0']
        
        param_groups = [
            {'params': master_params, 'lr': lr, 'weight_decay': 0.0005},
            {'params': scaling_params, 'lr': 0.00002, 'weight_decay': 0.005}
        ]
        
        optimizer = torch.optim.Adam(param_groups)
        
        print(f"Optimizer: Adam")
        print(f"  C2PSA params: {len(master_params)}, LR={lr}")
        print(f"  Scaling params: {len(scaling_params)}, LR=0.00002")
        
        return optimizer
    
    def _build_dataloader(self, data_yaml, batch_size, mode='train'):
        """Build dataloader using Ultralytics - NO WORKERS"""
        
        # CRITICAL: Force workers to 0 to prevent DRAM memory explosion
        workers = 0
        
        print(f"\n  Building {mode} dataloader...")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Workers: {workers} (disabled for memory safety)")
        
        overrides = {
            'data': data_yaml,
            'imgsz': self.config['train']['imgsz'],
            'batch': batch_size,
            'workers': workers,  # CRITICAL: 0 workers
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
            workers=workers,  # CRITICAL: 0 workers
            shuffle=(mode == 'train'),
            rank=-1
        )
        
        print(f"  ✓ Dataloader created: {len(loader)} batches")
        
        return loader

    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.master_model.model.train()
        self.shadow_model.model.train()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f"Epoch {self.epoch+1}")
        mloss = torch.zeros(3, device=self.device)
        
        for i, batch in pbar:
            try:
                # Prepare batch
                batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255.0
                batch['cls'] = batch['cls'].to(self.device)
                batch['bboxes'] = batch['bboxes'].to(self.device)
                batch['batch_idx'] = batch['batch_idx'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Quantize master → shadow
                self.bitlinear_manager.quantize_master_to_shadow()
                
                # Forward on shadow
                with torch.amp.autocast('cuda', enabled=False): 
                    pred = self.shadow_model.model(batch['img'])
                    loss, loss_items = self.compute_loss(pred, batch)
                    if loss.dim() > 0:
                        loss = loss.sum()
                
                # Backward
                loss.backward()
                
                # Compute TTQ gradients - FIXED: no arguments
                master_grads, ap_grads, an_grads = self.bitlinear_manager.compute_ttq_gradients()
                
                # Apply gradients
                self.bitlinear_manager.apply_gradients_to_master(master_grads)
                self.bitlinear_manager.apply_gradients_to_scales(ap_grads, an_grads)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.master_model.model.parameters() if p.requires_grad],
                    max_norm=10.0
                )
                torch.nn.utils.clip_grad_norm_(
                    self.bitlinear_manager.get_scaling_parameters(),
                    max_norm=10.0
                )
                
                # Optimizer step
                self.optimizer.step()
                
                # Update metrics
                mloss = (mloss * i + loss_items) / (i + 1)
                pbar.set_postfix({
                    'loss': f'{mloss[0]:.4f}',
                    'box': f'{mloss[1]:.4f}',
                    'cls': f'{mloss[2]:.4f}'
                })
                
                if self.use_wandb and i % 10 == 0:
                    ap_vals = [self.bitlinear_manager.ap_dict[n].item() 
                            for n in self.bitlinear_manager.quantized_layers]
                    an_vals = [self.bitlinear_manager.an_dict[n].item() 
                            for n in self.bitlinear_manager.quantized_layers]
                    
                    wandb.log({
                        'train/loss': mloss[0].item(),
                        'train/box_loss': mloss[1].item(),
                        'train/cls_loss': mloss[2].item(),
                        'train/ap_mean': np.mean(ap_vals) if ap_vals else 0,
                        'train/an_mean': np.mean(an_vals) if an_vals else 0,
                        'epoch': self.epoch,
                        'step': self.epoch * len(train_loader) + i
                    })
                
                # Clear cache
                del batch, pred, loss
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"\n❌ OOM at step {i}, reducing batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        return mloss

    
    @torch.no_grad()
    def validate(self, val_loader):
        self.bitlinear_manager.quantize_master_to_shadow()
        self.shadow_model.model.eval()
        
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
    
    def _check_early_stopping(self, current_fitness):
        if self.patience is None:
            return False
        
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"\n🛑 Early stopping triggered!")
                return True
            return False
    
    def train(self, epochs):
        print("\nStarting C2PSA BitLinear_TTQ Training")
        
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
        
        save_dir = Path(self.config['logging'].get(
            'save_dir',
            f"c2psa_checkpoints/{self.config['logging']['name']}"
        ))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Checkpoint directory: {save_dir.absolute()}")
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 70)
            
            mloss = self.train_epoch(train_loader)
            metrics = self.validate(val_loader)
            
            print(f"  Loss: {mloss[0]:.4f} (box: {mloss[1]:.4f}, cls: {mloss[2]:.4f})")
            print(f"  mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50-95']:.4f}")
            
            self.bitlinear_manager.print_statistics()
            
            if self.use_wandb:
                wandb.log({
                    'val/mAP50': metrics['mAP50'],
                    'val/mAP50-95': metrics['mAP50-95'],
                    'epoch': epoch,
                    'best_mAP50': self.best_fitness
                })
            
            checkpoint_path = save_dir / f'epoch_{epoch+1}.pt'
            self.bitlinear_manager.export_quantized_model(checkpoint_path)
            
            if metrics['mAP50'] > self.best_fitness:
                self.best_fitness = metrics['mAP50']
                best_path = save_dir / 'best.pt'
                self.bitlinear_manager.export_quantized_model(best_path)
                print(f"  ✓ New best model! mAP50: {self.best_fitness:.4f}")
            
            if self._check_early_stopping(metrics['mAP50']):
                break
        
        print("\n" + "="*70)
        print(f"✅ Training complete! Best mAP50: {self.best_fitness:.4f}")
        print("="*70)
        
        if self.use_wandb:
            wandb.finish()
        
        return metrics
