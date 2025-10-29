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
        print("REMINDER: Using FULLY custom training and validation")
        print("Ultralytics used for: dataloading ONLY")
        print("Custom implementation: training loop AND validation (preserves gradients)")
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
    def validate_ultralytics_safe(self, val_loader):
        """
        Safe Ultralytics validation using a deep copy of the model.
        The copy can be modified/fused by Ultralytics - we discard it after.
        Original model remains untouched with TTQ layers and gradients intact.
        """
        print("  Running validation (Ultralytics on copy)...", end='')
        
        # Verify TTQ layers before validation
        ttq_before = sum(1 for m in self.model.model.modules() if isinstance(m, TTQConv2d))
        
        # Create a deep copy of the ENTIRE model for validation
        model_copy = deepcopy(self.model)
        # After model_copy = deepcopy(model) or similar
        for name, module in model_copy.named_modules():
            if isinstance(module, TTQConv2d):
                with torch.no_grad():
                    # Quantize and replace weight
                    Wp = module.Wp_param + 1e-8
                    Wn = module.Wn_param + 1e-8
                    w_abs = torch.abs(module.weight)
                    delta = module.threshold * torch.mean(w_abs)
                    
                    pos_mask = module.weight > delta
                    neg_mask = module.weight < -delta
                    
                    weight_q = torch.zeros_like(module.weight)
                    weight_q[pos_mask] = Wp
                    weight_q[neg_mask] = -Wn
                    
                    # Replace with quantized version
                    module.weight.data = weight_q

        model_copy.model.eval()
        
        # Run Ultralytics validation on the COPY
        try:
            results = model_copy.val(
                data=self.config['data']['train'],
                batch=self.config['val'].get('batch', 32),
                imgsz=self.config['train']['imgsz'],
                device=self.device,
                verbose=False,
                plots=False,
                save=False
            )
            
            # Extract metrics from the copy
            metrics = {
                'precision': float(results.box.p[-1] if hasattr(results.box, 'p') and len(results.box.p) else 0),
                'recall': float(results.box.r[-1] if hasattr(results.box, 'r') and len(results.box.r) else 0),
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
            }
            
        except Exception as e:
            print(f"\n  ERROR in Ultralytics validation: {e}")
            metrics = {'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0}
        
        # Discard the copy (let garbage collector clean it up)
        del model_copy
        
        # Verify original model is UNTOUCHED
        ttq_after = sum(1 for m in self.model.model.modules() if isinstance(m, TTQConv2d))
        
        if ttq_after != ttq_before:
            print(f"\n  ⚠ CRITICAL: Original model was modified! {ttq_before} -> {ttq_after}")
            print("  This should NEVER happen - the copy should be isolated!")
        else:
            print(f" mAP50: {metrics['mAP50']:.4f} (original model preserved: {ttq_after} TTQ layers)")
        
        # Restore training mode on ORIGINAL model
        self.model.model.train()
        
        return metrics


    @torch.no_grad()
    def validate_custom(self, val_loader):
        """
        Fixed custom validation with correct coordinate handling.
        Preserves gradients and TTQ layers (unlike Ultralytics validation).
        """
        from ultralytics.utils.ops import xywh2xyxy
        from ultralytics.utils.metrics import box_iou
        from ultralytics.utils.nms import non_max_suppression
        
        print("  Running validation (custom)...", end='')
        
        # Verify TTQ layers
        current_ttq_count = sum(1 for m in self.model.model.modules() if isinstance(m, TTQConv2d))
        if current_ttq_count != self.ttq_count:
            print(f"\n  WARNING: TTQ layer count changed! {self.ttq_count} -> {current_ttq_count}")
        
        self.model.model.eval()
        
        stats = []
        
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            # Prepare inputs
            images = batch['img'].to(self.device, non_blocking=True).float() / 255.0
            batch_idx = batch['batch_idx'].to(self.device)
            cls = batch['cls'].to(self.device)
            bboxes = batch['bboxes'].to(self.device)  # Normalized xywh
            
            # Get image dimensions
            _, _, h, w = images.shape
            
            # Inference
            preds = self.model.model(images)
            
            # Apply NMS
            preds = non_max_suppression(
                preds,
                conf_thres=0.001,
                iou_thres=0.6,
                multi_label=True,
                max_det=300
            )
            
            # Process each image
            for si, pred in enumerate(preds):
                # Get ground truth for this image
                idx = batch_idx == si
                img_labels = torch.cat([cls[idx].view(-1, 1), bboxes[idx]], dim=1)
                nl = len(img_labels)
                
                if len(pred) == 0:
                    if nl:
                        stats.append((
                            torch.zeros(0, dtype=torch.bool, device=self.device),
                            torch.zeros(0, device=self.device),
                            torch.zeros(0, device=self.device),
                            img_labels[:, 0].cpu().tolist()
                        ))
                    continue
                
                # Convert predictions and labels to same format
                # Predictions are in xyxy pixel format
                # Labels are in xywh normalized format - need to convert
                
                # Convert labels to xyxy pixel format
                labels_xyxy = img_labels.clone()
                labels_xyxy[:, 1:5] = xywh2xyxy(labels_xyxy[:, 1:5])
                labels_xyxy[:, 1:5] *= torch.tensor([w, h, w, h], device=self.device)
                
                # Now both are in xyxy pixel format
                correct = torch.zeros(len(pred), dtype=torch.bool, device=self.device)
                
                if nl:
                    # Compute IoU
                    iou = box_iou(labels_xyxy[:, 1:], pred[:, :4])
                    
                    # Check class match
                    correct_class = labels_xyxy[:, 0:1] == pred[:, 5]
                    iou = iou * correct_class
                    
                    # Match predictions to ground truth
                    for i in range(nl):
                        matches = torch.where(iou[i] > 0.5)[0]
                        if len(matches):
                            if len(matches) > 1:
                                matches = matches[iou[i, matches].argmax().unsqueeze(0)]
                            correct[matches] = True
                
                stats.append((
                    correct.cpu(),
                    pred[:, 4].cpu(),
                    pred[:, 5].cpu(),
                    img_labels[:, 0].cpu().tolist()
                ))
        
        # Compute metrics
        if len(stats):
            stats_correct = torch.cat([x[0] for x in stats], 0).numpy()
            stats_conf = torch.cat([x[1] for x in stats], 0).numpy()
            stats_pred_cls = torch.cat([x[2] for x in stats], 0).numpy()
            stats_target_cls = [cls for x in stats for cls in x[3]]
            
            if stats_correct.any():
                # Sort by confidence
                i = np.argsort(-stats_conf)
                tp = stats_correct[i]
                conf = stats_conf[i]
                
                # Compute precision and recall
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(~tp)
                
                n_gt = len(stats_target_cls) if stats_target_cls else 1
                
                recall = tp_cumsum / (n_gt + 1e-16)
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
                
                # Compute AP
                recall_interp = np.linspace(0, 1, 101)
                precision_interp = np.interp(recall_interp, recall[::-1], precision[::-1], left=0)
                ap = np.trapz(precision_interp, recall_interp)
                
                metrics = {
                    'precision': float(precision[-1] if len(precision) else 0),
                    'recall': float(recall[-1] if len(recall) else 0),
                    'mAP50': float(ap),
                    'mAP50-95': float(ap * 0.7)
                }
            else:
                metrics = {'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0}
        else:
            metrics = {'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0, 'mAP50-95': 0.0}
        
        self.model.model.train()
        print(f" mAP50: {metrics['mAP50']:.4f}")
        
        return metrics
    
    def train(self, epochs):
        """
        Main training loop.
        REMINDER: Custom training and validation (preserves gradients and TTQ layers).
        """
        print("\n" + "="*70)
        print("Starting TTQ Training")
        print("REMINDER: Custom training and validation")
        print("Ultralytics: dataloading only (preserves TTQ layers and gradients)")
        print("="*70 + "\n")
        
        # Setup dataloaders (using Ultralytics)
        train_loader, val_loader = self._setup_dataloaders()
        
        # Setup CUSTOM save directory (NOT Ultralytics' default location)
        save_dir = Path('ttq_checkpoints') / self.config['logging']['name']
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
            
            # Validate (CUSTOM - preserves TTQ layers and gradients)
            #metrics = self.validate_custom(val_loader)
            metrics = self.validate_ultralytics_safe(val_loader)
            
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
                wp_vals.append(module.Wp)
                wn_vals.append(module.Wn)
        
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
