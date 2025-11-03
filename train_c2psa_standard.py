#!/usr/bin/env python3

import argparse
import torch
from pathlib import Path
import yaml
import sys
import os
import copy
import warnings
warnings.filterwarnings('ignore', category=Warning)


sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from src.quantization.c2psa_bitlinear_standard import replace_c2psa_with_bitlinear_standard
from src.quantization.bitlinear_standard_manager import BitLinearStandardManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config['train'].get('device', 'cuda')
    
    
    original_save_dir = config['logging'].get('save_dir', 'checkpoints/stage2_c2psa_baseline/')
    original_name = config['logging'].get('name', 'stage2_baseline_c2psa')
    
    new_save_dir = original_save_dir.rstrip('/') + '_standard/'
    new_name = original_name + '_standard'
    
    # Update config
    config['logging']['save_dir'] = new_save_dir
    config['logging']['name'] = new_name
    
    print("\n" + "="*80)
    print("Standard BitLinear Training (Fixed Scales)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"\nSave Location:")
    print(f"  Original: {original_save_dir}")
    print(f"  Standard: {new_save_dir}")
    print(f"\nNote: Same training loop as TTQ, but with FIXED scales (not learned)")
    
    # Load models
    print("\nLoading master model (FP32)...")
    master_yolo = YOLO(args.model)
    master_model = master_yolo.model.to(device)
    print("    Master loaded")
    
    print("Loading shadow model (Standard BitLinear)...")
    shadow_yolo = YOLO(args.model)
    shadow_model = shadow_yolo.model.to(device)
    shadow_model.model[10] = replace_c2psa_with_bitlinear_standard(shadow_model.model[10])
    shadow_model.model[10] = shadow_model.model[10].to(device)
    print("    Shadow loaded")
    
    master_yolo.model = master_model
    shadow_yolo.model = shadow_model
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = StandardBitLinearTrainer(
        master_model=master_yolo,
        shadow_model=shadow_yolo,
        config=config,
        device=device
    )
    
    # Train
    print("\nStarting training...")
    results = trainer.train(epochs=config['train']['epochs'])
    
    print("\n" + "="*80)
    print(" Training complete!")
    print(f" Models saved to: {new_save_dir}")
    print("="*80)


from src.training.c2psa_trainer import C2PSABitLinearTTQTrainer


class StandardBitLinearTrainer(C2PSABitLinearTTQTrainer):
    """
    Trainer for Standard BitLinear
    """
    
    def __init__(self, master_model, shadow_model, config, device='cuda'):

        self.master_model = master_model
        self.shadow_model = shadow_model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"\n    Using device: {self.device}")
        
        # Move models
        self.master_model.model.to(self.device)
        self.shadow_model.model.to(self.device)
        
        # Enable gradients
        self.master_model.model.train()
        for param in self.master_model.model.parameters():
            param.requires_grad = True
        
        self.shadow_model.model.train()
        for param in self.shadow_model.model.parameters():
            param.requires_grad = True
        
        # Initialize manager
        self.bitlinear_manager = BitLinearStandardManager(
            self.master_model.model,
            self.shadow_model.model,
            threshold=config['quantization'].get('threshold', 0.7),
            device=self.device
        )
        
        from ultralytics.utils.loss import v8DetectionLoss
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
        
        self.use_wandb = config['logging'].get('use_wandb', False)
        if self.use_wandb:
            self._init_wandb(config)
        
        print("="*70)
        print("Standard BitLinear Trainer Initialized (Fixed Scales)")
        print("="*70)
    
    def _init_wandb(self, config):
        """Initialize WandB logging"""
        try:
            import wandb
            wandb_config = {
                'stage': config.get('name', 'unknown'),
                'batch_size': config['train']['batch'],
                'lr': config['train']['lr0'],
                'epochs': config['train']['epochs'],
                'threshold': config['quantization'].get('threshold', 0.7),
                'scale_type': 'fixed'
            }
            
            wandb.init(
                project=config['logging'].get('wandb_project', 'ttq-yolo-c2psa'),
                entity=config['logging'].get('wandb_entity', None),
                name=config['logging']['name'],
                config=wandb_config,
                reinit=True
            )
        except:
            pass
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        master_params = []
        for name, param in self.master_model.model.named_parameters():
            if 'model.10' in name and param.requires_grad:
                master_params.append(param)
        
        scaling_params = []
        
        lr = self.config['train']['lr0']
        
        param_groups = [
            {'params': master_params, 'lr': lr, 'weight_decay': 0.0005},
        ]
        
        optimizer = torch.optim.Adam(param_groups)
        
        print(f"Optimizer: Adam")
        print(f"  C2PSA params: {len(master_params)}, LR={lr}")
        print(f"  Scaling params: 0 (Fixed scales - not optimized)")
        
        return optimizer
    
    def _build_dataloader(self, data_yaml, batch_size, mode='train'):
        """Build dataloader"""
        from ultralytics.data import build_dataloader, build_yolo_dataset
        from ultralytics.cfg import get_cfg
        from ultralytics.data.utils import check_det_dataset
        
        workers = 0
        
        print(f"\n  Building {mode} dataloader...")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Workers: {workers}")
        
        overrides = {
            'data': data_yaml,
            'imgsz': self.config['train']['imgsz'],
            'batch': batch_size,
            'workers': workers,
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
            workers=workers,
            shuffle=(mode == 'train'),
            rank=-1
        )
        
        print(f"      Dataloader created: {len(loader)} batches")
        
        return loader
    
    def train_epoch(self, train_loader):
        """Train epoch - uses standard manager"""
        from tqdm import tqdm
        
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
                
                # Quantize master to shadow
                self.bitlinear_manager.quantize_master_to_shadow()
                
                # Forward on shadow
                with torch.amp.autocast('cuda', enabled=False):
                    pred = self.shadow_model.model(batch['img'])
                    loss, loss_items = self.compute_loss(pred, batch)
                    if loss.dim() > 0:
                        loss = loss.sum()
                
                loss.backward()
                
                master_grads, _, _ = self.bitlinear_manager.compute_standard_gradients()
                
                self.bitlinear_manager.apply_gradients_to_master(master_grads)
                
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.master_model.model.parameters() if p.requires_grad],
                    max_norm=10.0
                )
                
                self.optimizer.step()
                
                # Update metrics
                mloss = (mloss * i + loss_items) / (i + 1)
                pbar.set_postfix({
                    'loss': f'{mloss[0]:.4f}',
                    'box': f'{mloss[1]:.4f}',
                    'cls': f'{mloss[2]:.4f}'
                })
                
                if self.use_wandb and i % 10 == 0:
                    try:
                        import wandb
                        wandb.log({
                            'train/loss': mloss[0].item(),
                            'train/box_loss': mloss[1].item(),
                            'train/cls_loss': mloss[2].item(),
                            'epoch': self.epoch,
                            'step': self.epoch * len(train_loader) + i
                        })
                    except:
                        pass
                
                # Clear cache
                del batch, pred, loss
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        return mloss
    
    def validate(self, val_loader):
        """Validate on temporary copy (no gradient issues)"""
        import copy
        
        # Create deep copy of shadow model for validation
        shadow_copy = copy.deepcopy(self.shadow_model)
        shadow_copy.model = shadow_copy.model.to(self.device)
        
        self.bitlinear_manager.quantize_master_to_shadow()
        
        with torch.no_grad():
            for (name_orig, param_orig), (name_copy, param_copy) in zip(
                self.shadow_model.model.named_parameters(),
                shadow_copy.model.named_parameters()
            ):
                param_copy.data.copy_(param_orig.data)
        
        # Validate on copy
        metrics = {}
        try:
            with torch.no_grad():
                results = shadow_copy.val(
                    data=self.config['data']['train'],
                    batch=self.config['train'].get('batch_val', 16),
                    imgsz=640,
                    device=self.device,
                    verbose=False,
                    plots=False
                )
            
            # Get metrics
            metrics = {
                'precision': float(results.box.p[-1]) if hasattr(results.box, 'p') and len(results.box.p) > 0 else 0.0,
                'recall': float(results.box.r[-1]) if hasattr(results.box, 'r') and len(results.box.r) > 0 else 0.0,
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
            }
        except Exception as e:
            print(f"    Validation error: {e}")
            metrics = {'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0}
        
        # Delete copy
        del shadow_copy
        torch.cuda.empty_cache()
        
        return metrics
    
    def train(self, epochs):
        """Main training loop"""
        print("\nStarting Standard BitLinear Training\n")
        
        train_loader = self._build_dataloader(
            self.config['data']['train'],
            self.config['train']['batch'],
            mode='train'
        )
        
        val_loader = self._build_dataloader(
            self.config['data']['train'],
            self.config['train'].get('batch_val', 16),
            mode='val'
        )
        
        save_dir = Path(self.config['logging'].get(
            'save_dir',
            f"c2psa_checkpoints/{self.config['logging']['name']}"
        ))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f" Checkpoint directory: {save_dir.absolute()}\n")
        
        patience = self.patience if self.patience is not None else 10
        patience_counter = 0
        best_mAP50 = 0.0
        
        print(f" Early Stopping Patience: {patience} epochs")
        print(f" Saving: Best model only (no epoch checkpoints)\n")
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 70)
            
            try:
                mloss = self.train_epoch(train_loader)
                
                print(f"  Loss: {mloss[0]:.4f} (box: {mloss[1]:.4f}, cls: {mloss[2]:.4f})")
                
                self.bitlinear_manager.print_statistics()
                
                # Validate on temporary copy
                metrics = self.validate(val_loader)
                
                print(f"  Loss: {mloss[0]:.4f} (box: {mloss[1]:.4f}, cls: {mloss[2]:.4f})")
                print(f"  mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50-95']:.4f}")
                
                if metrics['mAP50'] > best_mAP50:
                    best_mAP50 = metrics['mAP50']
                    patience_counter = 0  # Reset patience
                    self.best_fitness = metrics['mAP50']
                    
                    # Save best model
                    best_path = save_dir / 'best.pt'
                    self.bitlinear_manager.export_quantized_model(best_path)
                    print(f"      New best model! mAP50: {metrics['mAP50']:.4f} | Patience: {patience_counter}/{patience}\n")
                else:
                    patience_counter += 1
                    print(f"    No improvement | Patience: {patience_counter}/{patience}\n")
                    
                    # Early stopping triggered
                    if patience_counter >= patience:
                        print(f"\n" + "="*70)
                        print(f" EARLY STOPPING: Patience ({patience}) exceeded!")
                        print(f"   Best mAP50: {best_mAP50:.4f}")
                        print(f"="*70 + "\n")
                        break
                
                if self.use_wandb:
                    try:
                        import wandb
                        wandb.log({
                            'epoch': epoch,
                            'train/loss_epoch': mloss[0].item(),
                            'val/mAP50': metrics['mAP50'],
                            'val/mAP50-95': metrics['mAP50-95'],
                            'patience': patience_counter,
                        })
                    except:
                        pass
                
            except KeyboardInterrupt:
                print("\n Training interrupted by user")
                break
            except Exception as e:
                print(f"\n Error in epoch {epoch + 1}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print("\n" + "="*70)
        print(f" Training complete!")
        print(f"   Best mAP50: {best_mAP50:.4f}")
        print(f"   Models saved to: {save_dir.absolute()}")
        print(f"   Best model: {save_dir / 'best.pt'}")
        print("="*70)
        
        try:
            import wandb
            wandb.finish()
        except:
            pass
        
        return {'mAP50': best_mAP50, 'mAP50-95': 0.0, 'precision': 0.0, 'recall': 0.0}



if __name__ == '__main__':
    main()
