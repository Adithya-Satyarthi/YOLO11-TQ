from ultralytics.models.yolo.detect import DetectionValidator


def evaluate_model(model, dataloader):
    """Custom evaluation that returns mAP"""
    # Set model to eval mode
    model.eval()
    
    # Reinitialize validator with the modified model
    temp_validator = DetectionValidator(dataloader=dataloader, save_dir='runs/temp')
    temp_validator.model = model
    
    # Run validation
    metrics = temp_validator()
    return metrics.box.map