# src/train_progressive_refactored.py

import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from src.utils import set_seed
from src.quantization.progressive_quant import convert_to_progressive_quant, set_model_bitwidth
from src.training.progressive_trainer import ProgressiveTrainer

def load_config(config_path='configs/progressive.yaml'):
    """Loads the training configuration from a YAML file."""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_environment(config):
    """Sets up the environment for reproducibility."""
    print("Setting up environment...")
    set_seed(config['training']['seed'])
    torch.backends.cudnn.benchmark = False
    print("✓ Environment setup complete")

def evaluate_baseline(config):
    """Evaluates the baseline FP32 model performance."""
    print("\n" + "="*80)
    print("Evaluating Baseline (FP32) Performance")
    print("="*80)
    
    model_path = config['model']['model_path']
    print(f"Loading pre-trained model: {model_path}")
    yolo = YOLO(model_path)
    
    metrics = yolo.val(
        data=config['dataset']['data_yaml'],
        batch=config['training']['batch'],
        imgsz=config['dataset']['imgsz'],
        device=config['training']['device'],
        verbose=False,
        project='runs/progressive_refactored',
        name='baseline_eval'
    )
    
    baseline_map = metrics.box.map
    print(f"✓ Baseline mAP50-95: {baseline_map:.4f}")
    return baseline_map

def create_quantized_model(config):
    """Creates and prepares a quantized model for training."""
    print("\n" + "="*80)
    print("Creating Quantized Model")
    print("="*80)

    model_path = config['model']['model_path']
    print(f"Loading fresh pre-trained model for quantization: {model_path}")
    yolo_fresh = YOLO(model_path)

    print("\nConverting model to progressive quantization...")
    init_bitwidth = config['progressive']['bitwidth_schedule'][0]
    quantized_model = convert_to_progressive_quant(
        yolo_fresh.model,
        init_bitwidth=init_bitwidth,
        skip_first_last=True
    )

    print("\nUnfreezing all model parameters for training...")
    quantized_model.train()
    for param in quantized_model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in quantized_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in quantized_model.parameters())
    
    if total_params > 0 and trainable_params == total_params:
        print(f"✓ Model unfrozen successfully: {trainable_params:,}/{total_params:,} parameters are trainable.")
    else:
        print(f"✗ Warning: Not all parameters are trainable. {trainable_params:,}/{total_params:,} trainable.")

    yolo_quant = YOLO(model_path)
    yolo_quant.model = quantized_model
    
    return yolo_quant

def run_progressive_training(yolo_quant, config, baseline_map):
    """Runs the main progressive quantization training loop using a robust checkpointing strategy."""
    print("\n" + "="*80)
    print("Starting Progressive Training")
    print("="*80)

    save_dir = 'runs/progressive_refactored'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    trainer = ProgressiveTrainer(yolo_quant.model, config, save_dir=save_dir)

    initial_checkpoint_path = Path(save_dir) / 'stage_initial_model.pt'
    yolo_quant.save(initial_checkpoint_path)
    latest_checkpoint_path = initial_checkpoint_path

    cumulative_epoch = 0
    
    while not trainer.is_complete():
        current_bitwidth = trainer.get_current_bitwidth()
        stage_config = config['progressive']['stages'][current_bitwidth]
        
        print(f"\n--- STAGE {trainer.current_stage}: Training at {current_bitwidth}-bit ---")

        print(f"Loading clean model from checkpoint: {latest_checkpoint_path}")
        yolo_stage_model = YOLO(latest_checkpoint_path)

        print(f"Setting model bitwidth to {current_bitwidth}-bit...")
        set_model_bitwidth(yolo_stage_model.model, current_bitwidth)

        train_args = {
            'model': str(latest_checkpoint_path),
            'data': config['dataset']['data_yaml'],
            'epochs': stage_config['max_epochs'],
            'batch': config['training']['batch'],
            'imgsz': config['dataset']['imgsz'],
            'device': config['training']['device'],
            'lr0': stage_config['learning_rate'],
            'optimizer': config['training']['optimizer'],
            'weight_decay': config['training']['weight_decay'],
            'project': save_dir,
            'name': f'stage{trainer.current_stage}_{current_bitwidth}bit',
            'exist_ok': True,
            'verbose': False,
            'plots': False,
            'save': True,
            'workers': config['training']['workers'],
            'patience': stage_config.get('plateau_patience', 10),
        }
        
        print(f"Initializing trainer for {current_bitwidth}-bit quantization...")
        det_trainer = DetectionTrainer(overrides=train_args)

        det_trainer.model = yolo_stage_model.model
        
        print(f"Training for up to {stage_config['max_epochs']} epochs with LR={stage_config['learning_rate']:.1e}...")
        det_trainer.train()
        
        yolo_stage_model.model = det_trainer.model
        cumulative_epoch += det_trainer.epochs
        
        print(f"\nValidating stage {trainer.current_stage}...")
        val_metrics = yolo_stage_model.val(
            data=config['dataset']['data_yaml'],
            batch=config['training']['batch'],
            imgsz=config['dataset']['imgsz'],
            device=config['training']['device'],
            verbose=False,
            project=save_dir,
            name=f'val_stage{trainer.current_stage}_{current_bitwidth}bit'
        )
        current_map = val_metrics.box.map
        
        trainer.log_metrics(val_metrics, cumulative_epoch, current_bitwidth)
        is_best = current_map > trainer.best_map_per_stage.get(current_bitwidth, 0)
        if is_best:
            trainer.best_map_per_stage[current_bitwidth] = current_map
        
        trainer.save_checkpoint(yolo_stage_model, current_bitwidth, cumulative_epoch, val_metrics, is_best)
        
        if is_best:
            latest_checkpoint_path = Path(save_dir) / f'stage{trainer.current_stage}_{current_bitwidth}bit_best_model.pt'

        print(f"\n--- Stage {trainer.current_stage} Summary ---")
        print(f"  Bitwidth: {current_bitwidth}-bit")
        print(f"  Final mAP50-95: {current_map:.4f}")
        print(f"  Best mAP50-95 for this bitwidth: {trainer.best_map_per_stage.get(current_bitwidth, 0):.4f}")
        print(f"  Accuracy vs Baseline: {((current_map/baseline_map)*100 if baseline_map > 0 else 0):.2f}%")
        print(f"---------------------------")
        
        if trainer.transition_to_next_stage() is None:
            break
            
    print("Loading best model from final stage for summary...")
    yolo_final = YOLO(latest_checkpoint_path) 
    return yolo_final, trainer

def finalize_and_summarize(yolo_final, trainer, baseline_map, config):
    """Performs final evaluation and saves a summary of the training."""
    print("\n" + "="*80)
    print("Training Complete - Final Evaluation and Summary")
    print("="*80)

    # Run a final validation on the loaded best model to populate its metrics
    print("Running final validation on the best overall model...")
    final_metrics = yolo_final.val(
        data=config['dataset']['data_yaml'],
        batch=config['training']['batch'],
        imgsz=config['dataset']['imgsz'],
        device=config['training']['device'],
        project=trainer.save_dir,
        name='final_validation'
    )

    final_map = final_metrics.box.map # Now this will work
    accuracy_retention = (final_map / baseline_map) * 100 if baseline_map > 0 else 0
    
    print(f"\nResults:")
    print(f"  Baseline (FP32) mAP50-95: {baseline_map:.4f}")
    print(f"  Final (Quantized) mAP50-95: {final_map:.4f}")
    print(f"  Accuracy Retention:         {accuracy_retention:.2f}%")
    
    print("\nStage-by-Stage Best mAP:")
    for bitwidth, best_map in sorted(trainer.best_map_per_stage.items(), key=lambda item: item[0], reverse=True):
        retention = (best_map / baseline_map) * 100 if baseline_map > 0 else 0
        print(f"  {str(bitwidth):>2}-bit: mAP={best_map:.4f} | Retention={retention:.2f}%")
    
    final_model_path = Path(trainer.save_dir) / 'yolo11_quantized_final.pt'
    yolo_final.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    summary_path = Path(trainer.save_dir) / 'training_summary.yaml'
    summary = {
        'baseline_map': float(baseline_map),
        'final_map': float(final_map),
        'accuracy_retention_percent': float(accuracy_retention),
        'stage_best_maps': {str(k): float(v) for k, v in trainer.best_map_per_stage.items()},
        'config': config
    }
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Training summary saved to: {summary_path}")


def main():
    """Main function to run the progressive quantization training pipeline."""
    print("="*80)
    print("YOLO11 Progressive Quantization Training (Refactored)")
    print("="*80)

    try:
        config = load_config()
        setup_environment(config)
        baseline_map = evaluate_baseline(config)
        yolo_quant = create_quantized_model(config)
        yolo_final, trainer = run_progressive_training(yolo_quant, config, baseline_map)
        finalize_and_summarize(yolo_final, trainer, baseline_map, config)
        print("\n✓ Progressive training finished successfully!")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()