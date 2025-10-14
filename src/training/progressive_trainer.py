import torch
import os
from pathlib import Path
import wandb
from datetime import datetime


class ProgressiveTrainer:
    """
    Manages multi-stage progressive quantization training.
    Handles stage transitions, checkpoints, and metric tracking.
    """
    def __init__(self, model, config, save_dir='runs/progressive'):
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_stage = 0
        self.bitwidth_schedule = config['progressive']['bitwidth_schedule']
        self.best_map_per_stage = {}
        self.epochs_without_improvement = 0
        
        # Initialize wandb if enabled
        if config.get('wandb', {}).get('enabled', False):
            wandb.init(
                project=config['wandb']['project'],
                name=f"progressive_quant_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config
            )
    
    def should_transition_stage(self, current_map, epoch, stage_config):
        """
        Determine if we should move to next bitwidth stage.
        
        Criteria:
        1. Minimum epochs elapsed
        2. Plateau detected (no improvement for N epochs)
        3. Maximum epochs reached
        """
        min_epochs = stage_config.get('min_epochs', 10)
        max_epochs = stage_config.get('max_epochs', 50)
        plateau_patience = stage_config.get('plateau_patience', 5)
        plateau_threshold = stage_config.get('plateau_threshold', 0.005)
        
        current_bitwidth = self.bitwidth_schedule[self.current_stage]
        
        # Update best mAP for this stage
        if current_bitwidth not in self.best_map_per_stage:
            self.best_map_per_stage[current_bitwidth] = current_map
            self.epochs_without_improvement = 0
        else:
            improvement = current_map - self.best_map_per_stage[current_bitwidth]
            if improvement > plateau_threshold:
                self.best_map_per_stage[current_bitwidth] = current_map
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
        
        # Check transition conditions
        if epoch < min_epochs:
            return False  # Too early
        
        if epoch >= max_epochs:
            print(f"\nMax epochs ({max_epochs}) reached for stage {self.current_stage}")
            return True
        
        if self.epochs_without_improvement >= plateau_patience:
            print(f"\nPlateau detected ({self.epochs_without_improvement} epochs without improvement)")
            return True
        
        return False
    
    def save_checkpoint(self, yolo_model, bitwidth, epoch, metrics, is_best=False):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'epoch': epoch,
            'bitwidth': bitwidth,
            'stage': self.current_stage,
            'metrics': metrics,
            'model_state_dict': self.model.state_dict(),
            'best_map_per_stage': self.best_map_per_stage
        }
        
        # Save latest checkpoint
        latest_path = self.save_dir / f'checkpoint_stage{self.current_stage}_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint for this stage
        if is_best:
            best_path = self.save_dir / f'checkpoint_stage{self.current_stage}_best.pt'
            torch.save(checkpoint, best_path)
            # Also save the full YOLO model
            yolo_model.save(self.save_dir / f'yolo11_{bitwidth}bit_best.pt')
            
            # Get mAP value - handle both dict and object access patterns
            if isinstance(metrics, dict):
                # If metrics is a dict, try different key patterns
                if 'metrics/mAP50-95(B)' in metrics:
                    map_value = metrics['metrics/mAP50-95(B)']
                elif 'box' in metrics and 'map' in metrics['box']:
                    map_value = metrics['box']['map']
                else:
                    # Try to find any mAP-related key
                    map_value = metrics.get('map', metrics.get('mAP', 'N/A'))
            else:
                # If it's an object, assume it has .box.map
                map_value = getattr(getattr(metrics, 'box', None), 'map', 'N/A')
            
            print(f"✓ Saved best checkpoint: mAP={map_value if map_value == 'N/A' else f'{map_value:.4f}'}")
    
    def load_best_checkpoint(self, stage):
        """Load best checkpoint from previous stage"""
        checkpoint_path = self.save_dir / f'checkpoint_stage{stage}_best.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_map_per_stage = checkpoint['best_map_per_stage']
            print(f"✓ Loaded checkpoint from stage {stage}")
            return checkpoint
        return None
    
    def transition_to_next_stage(self):
        """Move to next bitwidth stage"""
        self.current_stage += 1
        if self.current_stage >= len(self.bitwidth_schedule):
            return False  # Training complete
        
        # Reset improvement counter
        self.epochs_without_improvement = 0
        
        # Get new bitwidth
        new_bitwidth = self.bitwidth_schedule[self.current_stage]
        
        print(f"\n{'='*80}")
        print(f"TRANSITIONING TO STAGE {self.current_stage}: {new_bitwidth}-bit")
        print(f"{'='*80}\n")
        
        # Update model bitwidth
        from src.quantization.progressive_quant import set_model_bitwidth
        set_model_bitwidth(self.model, new_bitwidth)
        
        return True
    
    def log_metrics(self, metrics, epoch, bitwidth):
        """Log metrics to console and wandb"""
        map50_95 = metrics.box.map
        map50 = metrics.box.map50
        
        print(f"Epoch {epoch:3d} | Bitwidth: {str(bitwidth):7s} | "
              f"mAP50-95: {map50_95:.4f} | mAP50: {map50:.4f}")
        
        if wandb.run is not None:
            wandb.log({
                f'mAP50-95_{bitwidth}': map50_95,
                f'mAP50_{bitwidth}': map50,
                'stage': self.current_stage,
                'epoch': epoch
            })
    
    def get_current_bitwidth(self):
        """Get current training bitwidth"""
        return self.bitwidth_schedule[self.current_stage]
    
    def is_complete(self):
        """Check if all stages are complete"""
        return self.current_stage >= len(self.bitwidth_schedule)
