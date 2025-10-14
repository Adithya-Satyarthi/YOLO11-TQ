from ultralytics import YOLO
import yaml
import torch
from pathlib import Path
from src.utils import set_seed
from src.quantization.progressive_quant import convert_to_progressive_quant, set_model_bitwidth, enable_quantized_training
from src.training.progressive_trainer import ProgressiveTrainer


def main():
    # Load configuration
    config_path = 'configs/progressive.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("="*80)
    print("YOLO11 Progressive Quantization Training")
    print("="*80)
    
    # Load pre-trained YOLO11 model
    model_path = config['model']['model_path']
    print(f"\nLoading pre-trained model: {model_path}")
    yolo = YOLO(model_path)
    
    # Get baseline performance
    print("\nEvaluating baseline (FP32) performance...")
    baseline_metrics = yolo.val(
        data=config['dataset']['data_yaml'],
        batch=config['training']['batch'],
        imgsz=config['dataset']['imgsz'],
        device=config['training']['device'],
        verbose=False
    )
    baseline_map = baseline_metrics.box.map
    print(f"Baseline mAP50-95: {baseline_map:.4f}")
    
    # Convert model to quantized version (starts at 8-bit)
    print("\nConverting model to progressive quantization...")
    quantized_model = convert_to_progressive_quant(
        yolo.model,
        init_bitwidth=config['progressive']['bitwidth_schedule'][0],
        skip_first_last=True
    )
    
    # CRITICAL: Unfreeze ALL layers (both quantized and non-quantized)
    # This ensures the entire model is trainable
    print("\nUnfreezing all model parameters for training...")
    quantized_model.train()  # Set to training mode

    # Recursively enable gradients for all parameters
    def unfreeze_model(model):
        """Recursively unfreeze all parameters in the model."""
        for name, param in model.named_parameters():
            param.requires_grad = True
        return model

    quantized_model = unfreeze_model(quantized_model)

    # Verify all parameters are trainable
    trainable_params = sum(p.numel() for p in quantized_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in quantized_model.parameters())
    print(f"✓ Model unfrozen: {trainable_params:,}/{total_params:,} parameters trainable ({100*trainable_params/total_params:.1f}%)")

    # Save the quantized model to a temporary file
    Path('runs/progressive').mkdir(parents=True, exist_ok=True)
    temp_model_path = 'runs/progressive/quantized_init.pt'
    
    # Create a new YOLO object with the quantized model
    yolo_quant = YOLO(model_path)  # Start with original
    yolo_quant.model = quantized_model
    yolo_quant.save(temp_model_path)
    
    # Initialize progressive trainer
    trainer = ProgressiveTrainer(
        quantized_model,
        config,
        save_dir='runs/progressive'
    )
    
    # Progressive training loop
    print("\nStarting progressive training...\n")
    
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import DEFAULT_CFG
    import copy
    
    # Track cumulative epochs for each stage
    cumulative_epoch = 0
    
    while not trainer.is_complete():
        current_bitwidth = trainer.get_current_bitwidth()
        stage_config = config['progressive']['stages'][current_bitwidth]
        
        print(f"\n{'='*80}")
        print(f"STAGE {trainer.current_stage}: Training at {current_bitwidth}-bit")
        print(f"{'='*80}\n")
        
        # Update learning rate for this stage
        lr = stage_config['learning_rate']
        
        # Prepare training arguments as a dictionary
        train_args = {
            'data': config['dataset']['data_yaml'],
            'epochs': stage_config['max_epochs'],
            'batch': config['training']['batch'],
            'imgsz': config['dataset']['imgsz'],
            'device': config['training']['device'],
            'lr0': lr,
            'optimizer': config['training']['optimizer'],
            'weight_decay': config['training']['weight_decay'],
            'project': 'runs/progressive',
            'name': f'stage{trainer.current_stage}_{current_bitwidth}bit',
            'exist_ok': True,
            'verbose': False,
            'plots': False,
            'save': True,
            'workers': config['training']['workers'],
            'patience': stage_config.get('plateau_patience', 10),
            'model': temp_model_path,
            'task': 'detect',
        }
        
        # Enable training mode and gradients
        quantized_model = enable_quantized_training(quantized_model)

        # Create trainer with quantized model
        print(f"Initializing trainer for {current_bitwidth}-bit quantization...")
        det_trainer = DetectionTrainer(overrides=train_args)

        # Move model to device and ensure training mode
        quantized_model = quantized_model.to(config['training']['device'])
        quantized_model.train()

        # Assign to trainer
        det_trainer.model = quantized_model
        
        # Train for this stage
        print(f"Training for up to {stage_config['max_epochs']} epochs...")
        try:
            det_trainer.train()
        except Exception as e:
            print(f"Warning: Training encountered an issue: {e}")
            print("Attempting to continue...")
        
        # Update cumulative epoch count
        cumulative_epoch += stage_config['max_epochs']
        
        # Get final validation metrics for this stage
        print(f"\nValidating stage {trainer.current_stage}...")
        yolo_temp = YOLO(model_path)
        yolo_temp.model = quantized_model
        
        val_metrics = yolo_temp.val(
            data=config['dataset']['data_yaml'],
            batch=config['training']['batch'],
            imgsz=config['dataset']['imgsz'],
            device=config['training']['device'],
            verbose=False
        )
        
        current_map = val_metrics.box.map
        
        # Log metrics
        trainer.log_metrics(val_metrics, cumulative_epoch, current_bitwidth)
        
        # Save checkpoint
        metrics_dict = {
            'box': {
                'map': current_map,
                'map50': val_metrics.box.map50
            }
        }
        is_best = current_map > trainer.best_map_per_stage.get(current_bitwidth, 0)
        
        # Update best mAP
        if is_best:
            trainer.best_map_per_stage[current_bitwidth] = current_map
        
        # Create temporary yolo wrapper for saving
        yolo_save = YOLO(model_path)
        yolo_save.model = quantized_model
        trainer.save_checkpoint(yolo_save, current_bitwidth, cumulative_epoch, metrics_dict, is_best)
        
        # Display stage summary
        print(f"\n{'='*80}")
        print(f"Stage {trainer.current_stage} Summary:")
        print(f"  Bitwidth: {current_bitwidth}")
        print(f"  Final mAP50-95: {current_map:.4f}")
        print(f"  Best mAP50-95: {trainer.best_map_per_stage.get(current_bitwidth, 0):.4f}")
        print(f"  Accuracy vs Baseline: {(current_map/baseline_map)*100:.2f}%")
        print(f"{'='*80}\n")
        
        # Transition to next stage
        if not trainer.transition_to_next_stage():
            break  # All stages complete
    
    # Final evaluation
    print("\n" + "="*80)
    print("TRAINING COMPLETE - Final Evaluation")
    print("="*80 + "\n")
    
    yolo_final = YOLO(model_path)
    yolo_final.model = quantized_model
    
    final_metrics = yolo_final.val(
        data=config['dataset']['data_yaml'],
        batch=config['training']['batch'],
        imgsz=config['dataset']['imgsz'],
        device=config['training']['device']
    )
    
    final_map = final_metrics.box.map
    accuracy_retention = (final_map / baseline_map) * 100
    
    print(f"\nResults Summary:")
    print(f"-" * 80)
    print(f"Baseline (FP32) mAP50-95:     {baseline_map:.4f}")
    print(f"Final (Ternary) mAP50-95:     {final_map:.4f}")
    print(f"Accuracy Retention:           {accuracy_retention:.2f}%")
    print(f"Accuracy Drop:                {baseline_map - final_map:.4f} ({100 - accuracy_retention:.2f}%)")
    print(f"-" * 80)
    
    # Print stage-by-stage results
    print(f"\nStage-by-Stage Results:")
    print(f"-" * 80)
    for bitwidth, best_map in trainer.best_map_per_stage.items():
        retention = (best_map / baseline_map) * 100
        print(f"{str(bitwidth):8s}-bit: mAP={best_map:.4f} | Retention={retention:.2f}%")
    print(f"-" * 80)
    
    # Save final model
    final_model_path = Path('runs/progressive') / 'yolo11_ternary_final.pt'
    yolo_final.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Save training summary
    summary_path = Path('runs/progressive') / 'training_summary.yaml'
    summary = {
        'baseline_map': float(baseline_map),
        'final_map': float(final_map),
        'accuracy_retention': float(accuracy_retention),
        'stage_results': {str(k): float(v) for k, v in trainer.best_map_per_stage.items()},
        'config': config
    }
    
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f"✓ Training summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
