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

# Wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("  Wandb not installed. Install with: pip install wandb")


class ShadowTTQTrainer:
    """
    Custom TTQ trainer using shadow weight approach.
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
            target_layers=config['quantization'].get('target_layers', None),
            quantize_1x1=config['quantization'].get('quantize_1x1_conv', False)  
        )
        
        # Setup loss function
        self.compute_loss = v8DetectionLoss(self.shadow_model.model)
        
        # Set hyperparameters with values
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
        
        # Early stopping
        self.patience = config['train'].get('patience', None)
        self.patience_counter = 0
        
        # Wandb setup
        self.use_wandb = config['logging'].get('use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            self._init_wandb(config)
        
        print("\n" + "="*70)
        print("Shadow Weight TTQ Trainer Initialized")
        print("="*70)
        if self.patience:
            print(f"Early stopping enabled with patience={self.patience}")
        if self.use_wandb:
            print(f"Wandb logging enabled: {wandb.run.url}")
    
    def _init_wandb(self, config):
        """Initialize Wandb logging"""
        wandb_config = {
            'stage': config.get('name', 'unknown'),
            'model': config['model']['weights'],
            'batch_size': config['train']['batch'],
            'lr': config['train']['lr0'],
            'epochs': config['train']['epochs'],
            'threshold': config['quantization']['threshold'],
            'target_layers': config['quantization'].get('target_layers', 'all'),
            'quantize_1x1': config['quantization'].get('quantize_1x1_conv', False),
            'patience': config['train'].get('patience', None),
        }
        
        wandb.init(
            project=config['logging'].get('wandb_project', 'ttq-yolo'),
            entity=config['logging'].get('wandb_entity', None),
            name=config['logging']['name'],
            config=wandb_config
        )
        
        # Log model architecture summary
        total_params = sum(p.numel() for p in self.master_model.model.parameters())
        quantized_params = sum(
            dict(self.master_model.model.named_modules())[name].weight.numel() 
            for name in self.shadow_manager.quantized_layers
        )
        wandb.config.update({
            'total_params': total_params,
            'quantized_params': quantized_params,
            'quantization_ratio': quantized_params / total_params
        })
    
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
                'lr': 0.00002, 
                'weight_decay': 0.005
            }
        ]
        
        optimizer = torch.optim.Adam(param_groups)
        
        print(f"Optimizer: Adam")
        print(f"  Master params: {len(master_params)}, LR={lr}")
        print(f"  Scaling params: {len(scaling_params)}, LR=0.00002")
        
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
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {self.epoch+1}")
        mloss = torch.zeros(3, device=self.device)
        
        for i, batch in pbar:
            # Prepare batch
            batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255.0
            batch['cls'] = batch['cls'].to(self.device)
            batch['bboxes'] = batch['bboxes'].to(self.device)
            batch['batch_idx'] = batch['batch_idx'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            
            self.shadow_manager.quantize_master_to_shadow()
            
            
            pred = self.shadow_model.model(batch['img'])
            loss, loss_items = self.compute_loss(pred, batch)
            if loss.dim() > 0:
                loss = loss.sum()
            
            loss.backward()
            
            master_grads, wp_grads, wn_grads = self.shadow_manager.compute_ttq_gradients(
                shadow_grads=None
            )
            
            self.shadow_manager.apply_gradients_to_master(master_grads)
            self.shadow_manager.apply_gradients_to_scales(wp_grads, wn_grads)
            
            torch.nn.utils.clip_grad_norm_(self.master_model.model.parameters(), max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(self.shadow_manager.get_scaling_parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            mloss = (mloss * i + loss_items) / (i + 1)
            pbar.set_postfix({
                'loss': f'{mloss[0]:.4f}',
                'box': f'{mloss[1]:.4f}',
                'cls': f'{mloss[2]:.4f}'
            })
            
            # Wandb logging (every 100 steps)
            if self.use_wandb and i % 100 == 0:
                wp_vals = [self.shadow_manager.wp_dict[n].item() for n in self.shadow_manager.quantized_layers]
                wn_vals = [self.shadow_manager.wn_dict[n].item() for n in self.shadow_manager.quantized_layers]
                
                wandb.log({
                    'train/loss': mloss[0].item(),
                    'train/box_loss': mloss[1].item(),
                    'train/cls_loss': mloss[2].item(),
                    'train/wp_mean': np.mean(wp_vals),
                    'train/wn_mean': np.mean(wn_vals),
                    'train/wp_max': max(wp_vals),
                    'train/wn_max': max(wn_vals),
                    'epoch': self.epoch,
                    'step': self.epoch * len(train_loader) + i
                })
            
            # Debug print every 100 steps
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
    
    def _check_early_stopping(self, current_fitness):
        """Check if training should stop early"""
        if self.patience is None:
            return False
        
        if current_fitness > self.best_fitness:
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"\n Early stopping triggered! No improvement for {self.patience} epochs.")
                return True
            else:
                print(f"  No improvement. Patience: {self.patience_counter}/{self.patience}")
                return False
        
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
        
        # Use save_dir from config
        save_dir = Path(self.config['logging'].get(
            'save_dir', 
            f"ttq_shadow_checkpoints/{self.config['logging']['name']}"
        ))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f" Checkpoint directory: {save_dir.absolute()}")
        
        # Get save_interval if specified in config
        save_interval = self.config['train'].get('save_interval', None)
        
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
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'val/mAP50': metrics['mAP50'],
                    'val/mAP50-95': metrics['mAP50-95'],
                    'val/precision': metrics['precision'],
                    'val/recall': metrics['recall'],
                    'epoch': epoch,
                    'best_mAP50': self.best_fitness
                })
            
            should_stop = self._check_early_stopping(metrics['mAP50'])
            
            # Save best model
            if metrics['mAP50'] > self.best_fitness:
                self.best_fitness = metrics['mAP50']  
                best_path = save_dir / 'best.pt'
                self.shadow_manager.export_ternary_model(best_path)
                print(f"New best model! mAP50: {self.best_fitness:.4f}")
                
                # Save timestamped backup of best model
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = save_dir / f'best_epoch{epoch+1}_{timestamp}.pt'
                self.shadow_manager.export_ternary_model(backup_path)
            
            
            if save_interval and (epoch + 1) % save_interval == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch{epoch+1}.pt'
                self.shadow_manager.export_ternary_model(checkpoint_path)
                print(f"   Checkpoint saved: {checkpoint_path.name}")
            
            # Stop if early stopping triggered
            if should_stop:
                break
        
        print("\n" + "="*70)
        print(f" Training complete! Best mAP50: {self.best_fitness:.4f}")
        print(f" Checkpoints saved to: {save_dir.absolute()}")
        print("="*70)
        
        if self.use_wandb:
            # Log final summary
            wandb.run.summary['best_mAP50'] = self.best_fitness
            wandb.run.summary['best_mAP50-95'] = metrics['mAP50-95']
            wandb.run.summary['total_epochs'] = epoch + 1
            wandb.finish()
        
        return metrics
