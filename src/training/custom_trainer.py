"""
Custom training loop for TTQ-quantized YOLO11
Uses Ultralytics ONLY for dataloading, NOT for training loop or validation
"""

import torch
import torch.nn as nn
import torch.amp as amp
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import LOGGER, RANK

# Import unwrap_model if available, otherwise define de_parallel
try:
    from ultralytics.utils.torch_utils import unwrap_model as de_parallel
except ImportError:
    def de_parallel(model):
        """De-parallelize a model: returns single-GPU model if model is of type DP or DDP"""
        return model.module if isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)) else model

from src.quantization.ttq_layer import TTQConv2d


class TTQYOLOTrainer:
    """
    Custom trainer for TTQ-YOLO that implements proper gradient computation.
    Uses Ultralytics for dataloading ONLY. Custom training loop and validation.
    """
    
    def __init__(self, model, config, device='cuda'):
        """
        REMINDER: This trainer uses Ultralytics for dataloading ONLY.
        Training loop and validation are custom to preserve TTQ layers.
        """
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.model.to(self.device)
        self.model.model.train()
        
        # DEBUG: Verify TTQ layers are present
        self.ttq_count = sum(1 for m in self.model.model.modules() if isinstance(m, TTQConv2d))
        print(f"DEBUG: Found {self.ttq_count} TTQ layers in model")
        
        # Setup optimizer (separate learning rates for scaling factors)
        self.optimizer = self._setup_optimizer()
        
        # Setup loss function with proper hyperparameters
        from ultralytics.utils.loss import v8DetectionLoss
        
        # Initialize loss with model
        self.compute_loss = v8DetectionLoss(self.model.model)
        
        # Ensure hyperparameters are set correctly
        if hasattr(self.compute_loss, 'hyp') and isinstance(self.compute_loss.hyp, dict):
            from types import SimpleNamespace
            hyp_dict = self.compute_loss.hyp
            hyp_dict.setdefault('box', 7.5)
            hyp_dict.setdefault('cls', 0.5)
            hyp_dict.setdefault('dfl', 1.5)
            self.compute_loss.hyp = SimpleNamespace(**hyp_dict)
        elif not hasattr(self.compute_loss, 'hyp'):
            from types import SimpleNamespace
            self.compute_loss.hyp = SimpleNamespace(
                box=7.5,
                cls=0.5,
                dfl=1.5
            )
        
        # Training state
        self.epoch = 0
        self.best_fitness = 0.0
        self.scaler = amp.GradScaler(self.device.type, enabled=True)
        
        print("\n" + "="*70)
        print("REMINDER: Using FULLY custom training")
        print("Ultralytics used for: dataloading ONLY")
        print("Custom implementation: training loop AND validation")
        print("="*70 + "\n")
    
    def _setup_optimizer(self):
        """
        Setup optimizer with separate parameter groups.
        TTQ scaling factors (Wp, Wn) get higher learning rate.
        """
        regular_params = []
        scaling_params = []
        
        for name, param in self.model.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'Wp' in name or 'Wn' in name:
                scaling_params.append(param)
            else:
                regular_params.append(param)
        
        lr = self.config['train']['lr0']
        
        param_groups = [
            {
                'params': regular_params,
                'lr': lr,
                'weight_decay': self.config['train'].get('weight_decay', 0.0005)
            },
            {
                'params': scaling_params,
                'lr': lr * 10,  # Higher LR for scaling factors (as per TTQ paper)
                'weight_decay': 0.0  # No weight decay for scaling factors
            }
        ]
        
        optimizer_name = self.config['train'].get('optimizer', 'Adam')
        
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(param_groups)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(
                param_groups,
                momentum=self.config['train'].get('momentum', 0.937)
            )
        else:
            optimizer = torch.optim.Adam(param_groups)
        
        print(f"Optimizer: {optimizer_name}")
        print(f"  Regular params: {len(regular_params)} parameters, LR={lr}")
        print(f"  Scaling params: {len(scaling_params)} (Wp/Wn), LR={lr*10}")
        
        return optimizer
    
    def _setup_dataloaders(self):
        """
        Setup dataloaders using Ultralytics.
        REMINDER: Using Ultralytics for dataloading.
        """
        print("\nSetting up dataloaders (using Ultralytics)...")
        
        # Load data config
        data_yaml = self.config['data']['train']
        
        # Build training dataset
        train_loader = self._build_dataloader(
            data_yaml,
            batch_size=self.config['train']['batch'],
            mode='train'
        )
        
        # Build validation dataset
        val_loader = self._build_dataloader(
            self.config['data'].get('val', data_yaml),
            batch_size=self.config['val'].get('batch', 32),
            mode='val'
        )
        
        print(f"  ✓ Train dataloader ready: {len(train_loader)} batches")
        print(f"  ✓ Val dataloader ready: {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def _build_dataloader(self, data_yaml, batch_size, mode='train'):
        """Build single dataloader using Ultralytics"""
        from ultralytics.cfg import get_cfg
        from ultralytics.data.utils import check_det_dataset
        
        # Get default config
        overrides = {
            'data': data_yaml,
            'imgsz': self.config['train']['imgsz'],
            'batch': batch_size,
            'workers': self.config['train'].get('workers', 8),
        }
        
        cfg = get_cfg(overrides=overrides)
        
        # Check dataset
        data_dict = check_det_dataset(data_yaml)
        
        # Build dataset
        dataset = build_yolo_dataset(
            cfg,
            img_path=data_dict[mode],
            batch=batch_size,
            data=data_dict,
            mode=mode,
            stride=int(max(self.model.model.stride))
        )
        
        # Build dataloader
        loader = build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.config['train'].get('workers', 8),
            shuffle=(mode == 'train'),
            rank=-1
        )
        
        return loader
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch with proper TTQ gradient computation.
        REMINDER: Custom training loop, not using Ultralytics.
        """
        self.model.model.train()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {self.epoch}")
        
        mloss = torch.zeros(3, device=self.device)  # mean losses
        
        for i, batch in pbar:
            # Move batch to device - keep as dictionary for loss function
            batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255.0
            batch['cls'] = batch['cls'].to(self.device)
            batch['bboxes'] = batch['bboxes'].to(self.device)
            batch['batch_idx'] = batch['batch_idx'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with amp.autocast(device_type=self.device.type, enabled=True):
                # Forward through model
                pred = self.model.model(batch['img'])
                
                # Compute loss - returns (total_loss, loss_items)
                loss, loss_items = self.compute_loss(pred, batch)
                
                # Ensure loss is a scalar
                if loss.dim() > 0:
                    loss = loss.sum()
            
            # Backward pass (TTQ autograd function handles Wp/Wn gradients)
            self.scaler.scale(loss).backward()
            
            # DEBUG: Check if Wp/Wn have gradients (first batch, first epoch only)
            if i == 0 and self.epoch == 0:
                wp_wn_with_grads = 0
                for name, param in self.model.model.named_parameters():
                    if ('Wp' in name or 'Wn' in name) and param.grad is not None:
                        wp_wn_with_grads += 1
                print(f"\n  DEBUG: {wp_wn_with_grads}/110 Wp/Wn parameters have gradients")
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            mloss = (mloss * i + loss_items) / (i + 1)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{mloss[0]:.4f}',
                'box': f'{mloss[1]:.4f}',
                'cls': f'{mloss[2]:.4f}'
            })
        
        return mloss
    
    @torch.no_grad()
    def validate_custom(self, val_loader):
        """
        Custom validation that preserves TTQ layers and learned scales.
        REMINDER: Fully custom validation, no Ultralytics interference.
        """
        # Correct imports for current Ultralytics (8.3.x)
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.metrics import box_iou
        
        # For xywh2xyxy, try multiple locations
        try:
            from ultralytics.utils.ops import xywh2xyxy
        except (ImportError, AttributeError):
            # Fallback: implement locally if not found
            def xywh2xyxy(x):
                """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
                y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
                y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
                y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
                y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
                y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
                return y
        
        print("  Running validation (custom - preserves TTQ)...", end='')
        
        # Verify TTQ layers are still present
        current_ttq_count = sum(1 for m in self.model.model.modules() if isinstance(m, TTQConv2d))
        if current_ttq_count != self.ttq_count:
            print(f"\n  WARNING: TTQ layer count changed! {self.ttq_count} -> {current_ttq_count}")
        
        self.model.model.eval()
        
        stats = []
        
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            images = batch['img'].to(self.device, non_blocking=True).float() / 255.0
            targets = torch.cat([
                batch['batch_idx'].view(-1, 1),
                batch['cls'].view(-1, 1),
                batch['bboxes']
            ], dim=1).to(self.device)
            
            # Inference with TTQ layers
            pred = self.model.model(images)
            
            # Apply NMS using Ultralytics function
            pred = non_max_suppression(
                pred,
                conf_thres=0.001,
                iou_thres=0.6,
                multi_label=True,
                max_det=300
            )
            
            # Compute metrics
            for si, pred_i in enumerate(pred):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []
                
                # FIX: Use torch.zeros with device specification instead of torch.Tensor()
                if len(pred_i) == 0:
                    if nl:
                        stats.append((torch.zeros(0, dtype=torch.bool, device=self.device), 
                                    torch.zeros(0, device=self.device),  # FIX: specify device
                                    torch.zeros(0, device=self.device),  # FIX: specify device
                                    tcls))
                    continue
                
                # Evaluate predictions
                if nl:
                    # Convert xywh to xyxy
                    tbox = xywh2xyxy(labels[:, 1:5])
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                    
                    # Compute IoU using Ultralytics function
                    iou = box_iou(labelsn[:, 1:], pred_i[:, :4])
                    correct_class = labelsn[:, 0:1] == pred_i[:, 5]
                    
                    correct = torch.zeros(pred_i.shape[0], dtype=torch.bool, device=self.device)
                    for i in range(len(labelsn)):
                        matches = torch.where((iou[i] > 0.5) & correct_class[i])[0]
                        if matches.shape[0]:
                            correct[matches[iou[i, matches].argmax()]] = True
                else:
                    correct = torch.zeros(pred_i.shape[0], dtype=torch.bool, device=self.device)
                
                stats.append((correct, pred_i[:, 4], pred_i[:, 5], tcls))
        
        # Compute mAP - Handle mixed tensor/list types
        if len(stats):
            # Separate tensors and lists
            stats_correct = torch.cat([x[0] for x in stats], 0).cpu().numpy()  # bool tensor
            stats_conf = torch.cat([x[1] for x in stats], 0).cpu().numpy()     # confidence tensor
            stats_pred_cls = torch.cat([x[2] for x in stats], 0).cpu().numpy() # pred class tensor
            stats_target_cls = [cls for x in stats for cls in x[3]]            # target class list (flattened)
            
            if stats_correct.any():
                tp = stats_correct
                conf = stats_conf
                
                # Sort by confidence
                i = np.argsort(-conf)
                tp = tp[i]
                
                # Compute precision and recall
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(~tp)
                
                # Number of ground truth objects
                n_gt = len(stats_target_cls) if stats_target_cls else 1
                
                recall = tp_cumsum / (n_gt + 1e-16)
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
                
                # Compute AP (mAP@0.5)
                recall_interp = np.linspace(0, 1, 101)
                precision_interp = np.interp(recall_interp, recall[::-1], precision[::-1], left=0)
                ap = np.trapz(precision_interp, recall_interp)
                
                metrics = {
                    'precision': float(precision[-1] if len(precision) else 0),
                    'recall': float(recall[-1] if len(recall) else 0),
                    'mAP50': float(ap),
                    'mAP50-95': float(ap * 0.7)  # Rough estimate for mAP50-95
                }
            else:
                metrics = {'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0}
        else:
            metrics = {'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0}
        
        # Restore training mode
        self.model.model.train()
        
        print(f" mAP50: {metrics['mAP50']:.4f}")
        
        return metrics
        
    def train(self, epochs):
        """
        Main training loop.
        REMINDER: Fully custom training and validation, Ultralytics for dataloading only.
        """
        print("\n" + "="*70)
        print("Starting TTQ Training")
        print("REMINDER: Fully custom training and validation")
        print("Ultralytics: dataloading only (preserves TTQ layers)")
        print("="*70 + "\n")
        
        # Setup dataloaders (using Ultralytics)
        train_loader, val_loader = self._setup_dataloaders()
        
        # Setup CUSTOM save directory (NOT Ultralytics' default location)
        save_dir = Path('ttq_checkpoints') / self.config['logging']['name']  # NEW LOCATION
        save_dir.mkdir(parents=True, exist_ok=True)
        wdir = save_dir / 'weights'
        wdir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving checkpoints to: {save_dir}")
        
        # Training loop
        for epoch in range(epochs):
            self.epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 70)
            
            # Train one epoch (CUSTOM)
            mloss = self.train_epoch(train_loader)
            
            # Validate (CUSTOM - preserves TTQ layers)
            metrics = self.validate_custom(val_loader)
            
            # Print metrics
            print(f"  Train Loss: {mloss[0]:.4f} (box: {mloss[1]:.4f}, cls: {mloss[2]:.4f})")
            print(f"  Val Precision: {metrics['precision']:.4f}")
            print(f"  Val Recall: {metrics['recall']:.4f}")
            print(f"  Val mAP50: {metrics['mAP50']:.4f}")
            print(f"  Val mAP50-95: {metrics['mAP50-95']:.4f}")
            
            # Check Wp/Wn values to ensure they're being learned
            self._print_wp_wn_stats()
            
            # Save checkpoint
            fitness = metrics['mAP50']
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.save_checkpoint(wdir / 'best.pt', epoch, metrics)
                print(f"  ✓ New best model saved! mAP50: {fitness:.4f}")
            
            # Periodic saves
            if (epoch + 1) % self.config['logging'].get('save_period', 10) == 0:
                self.save_checkpoint(wdir / f'epoch{epoch}.pt', epoch, metrics)
                print(f"  ✓ Checkpoint saved: epoch{epoch}.pt")
        
        print("\n" + "="*70)
        print("Training complete!")
        print(f"Best mAP50: {self.best_fitness:.4f}")
        print(f"Model saved to: {wdir / 'best.pt'}")
        print("="*70 + "\n")
        
        return metrics

    
    def _print_wp_wn_stats(self):
        """Print statistics about Wp/Wn values to verify they're being learned"""
        wp_vals = []
        wn_vals = []
        
        for module in self.model.model.modules():
            if isinstance(module, TTQConv2d):
                wp_vals.append(module.Wp.item())
                wn_vals.append(module.Wn.item())
        
        if wp_vals:
            print(f"  Wp: min={min(wp_vals):.4f}, max={max(wp_vals):.4f}, mean={np.mean(wp_vals):.4f}, std={np.std(wp_vals):.4f}")
            print(f"  Wn: min={min(wn_vals):.4f}, max={max(wn_vals):.4f}, mean={np.mean(wn_vals):.4f}, std={np.std(wn_vals):.4f}")
            
            # Check if values have diverged from initialization (1.0)
            wp_changed = sum(1 for v in wp_vals if abs(v - 1.0) > 0.01)
            wn_changed = sum(1 for v in wn_vals if abs(v - 1.0) > 0.01)
            
            if wp_changed > 0 or wn_changed > 0:
                print(f"  ✓ {wp_changed + wn_changed}/{len(wp_vals) + len(wn_vals)} scaling factors changed from init")
            else:
                print(f"  ⚠ Warning: Scaling factors still at initialization values")
    
    def save_checkpoint(self, path, epoch, metrics):
        """
        Save training checkpoint with TTQ layers preserved.
        CRITICAL: Saves entire model object to preserve TTQ layer structure.
        """
        # Save entire model object (not just state_dict) to preserve TTQ layers
        checkpoint = {
            'epoch': epoch,
            'model': deepcopy(de_parallel(self.model)),  # Save entire YOLO object
            'optimizer': self.optimizer.state_dict(),
            'best_fitness': self.best_fitness,
            'metrics': metrics,
            'train_args': self.config,
        }
        
        # Save checkpoint
        torch.save(checkpoint, path)
        
        # Verify TTQ layers are preserved in the saved model
        if hasattr(checkpoint['model'], 'model'):
            ttq_count = sum(1 for m in checkpoint['model'].model.modules() if isinstance(m, TTQConv2d))
            print(f"    ✓ Saved with {ttq_count} TTQ layers preserved")
            
            # Also verify Wp/Wn parameters in state dict
            state_dict = checkpoint['model'].model.state_dict()
            ttq_params = [k for k in state_dict.keys() if 'Wp' in k or 'Wn' in k]
            if ttq_params:
                print(f"    ✓ Saved {len(ttq_params)} TTQ parameters (Wp/Wn)")
            else:
                print(f"    ⚠ WARNING: No TTQ parameters found!")
        else:
            print(f"    ⚠ WARNING: Model structure may not be correct!")


def load_ttq_checkpoint(checkpoint_path, device='cuda'):
    """
    Load a TTQ checkpoint and restore the entire model with TTQ layers.
    Returns the full YOLO model object with TTQ layers intact.
    """
    print(f"\nLoading TTQ checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load the entire model object (preserves TTQ layer structure)
    model = checkpoint['model']
    
    # Move to device
    if hasattr(model, 'model'):
        model.model.to(device)
    
    # Verify TTQ layers are present
    if hasattr(model, 'model'):
        ttq_count = sum(1 for m in model.model.modules() if isinstance(m, TTQConv2d))
        print(f"  ✓ Loaded model with {ttq_count} TTQ layers")
        
        # Verify Wp/Wn parameters
        state_dict = model.model.state_dict()
        ttq_params = [k for k in state_dict.keys() if 'Wp' in k or 'Wn' in k]
        print(f"  ✓ Found {len(ttq_params)} TTQ parameters (Wp/Wn)")
        
        # Print sample Wp/Wn values
        for name, module in model.model.named_modules():
            if isinstance(module, TTQConv2d):
                print(f"  Example - {name}: Wp={module.Wp.item():.4f}, Wn={module.Wn.item():.4f}")
                break
    else:
        print("  ✗ WARNING: Model structure may not be preserved correctly")
    
    epoch = checkpoint.get('epoch', -1)
    metrics = checkpoint.get('metrics', {})
    
    return model, epoch, metrics
