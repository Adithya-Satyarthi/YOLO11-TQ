"""
Utilities for saving quantized models in organized structure
"""

import os
import shutil
from pathlib import Path
import yaml
from datetime import datetime
import torch


class ModelStageManager:
    """
    Manages saving models after each quantization stage
    """
    
    def __init__(self, base_dir="saved_models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def save_stage(self, stage_name, model_path, config=None, metrics=None, description=""):
        """
        Save a model from a specific quantization stage
        
        Args:
            stage_name: Name of the stage (e.g., "stage1_ttq_backbone_neck")
            model_path: Path to the best model weights
            config: Configuration dict used for this stage
            metrics: Performance metrics (mAP, etc.)
            description: Text description of this stage
        
        Returns:
            Path to saved model directory
        """
        # Create stage directory
        stage_dir = self.base_dir / stage_name
        stage_dir.mkdir(exist_ok=True)
        
        # Copy best model
        best_model_dest = stage_dir / "best.pt"
        shutil.copy2(model_path, best_model_dest)
        print(f"\n{'='*70}")
        print(f"✓ Saved best model to: {best_model_dest}")
        
        # Save metadata
        metadata = {
            "stage": stage_name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "source_model": str(model_path),
            "metrics": metrics or {},
            "config": config or {}
        }
        
        metadata_path = stage_dir / "metadata.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Saved metadata to: {metadata_path}")
        
        # Create README
        readme_content = self._generate_readme(stage_name, description, metrics, config)
        readme_path = stage_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"✓ Saved README to: {readme_path}")
        
        print(f"\nStage '{stage_name}' saved successfully!")
        print(f"Location: {stage_dir.absolute()}")
        print(f"{'='*70}\n")
        
        return stage_dir
    
    def _generate_readme(self, stage_name, description, metrics, config):
        """Generate README content for the stage"""
        
        # Clean up stage name for title
        title = stage_name.replace('_', ' ').title()
        
        readme = f"# {title}\n\n"
        readme += f"## Description\n\n{description}\n\n"
        readme += f"## Model Information\n\n"
        readme += f"- **Saved**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        readme += f"- **Stage**: {stage_name}\n"
        readme += f"- **Model File**: `best.pt`\n\n"
        
        readme += "## Performance Metrics\n\n"
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    readme += f"- **{key}**: {value:.4f}\n"
                else:
                    readme += f"- **{key}**: {value}\n"
        else:
            readme += "- No metrics recorded\n"
        
        readme += "\n## Configuration\n\n"
        if config:
            readme += "```"
            readme += yaml.dump(config, default_flow_style=False, sort_keys=False)
            readme += "```\n\n"
        else:
            readme += "- No configuration recorded\n\n"
        
        readme += "## Usage\n\n"
        readme += "### Loading the Model\n\n"
        readme += "```"
        readme += "from ultralytics import YOLO\n\n"
        readme += "# Load the quantized model\n"
        readme += "model = YOLO('best.pt')\n\n"
        readme += "# Run inference\n"
        readme += "results = model('image.jpg')\n\n"
        readme += "# Run validation\n"
        readme += "results = model.val(data='coco.yaml')\n"
        readme += "```\n\n"
        readme += "### Model Structure\n\n"
        readme += "This model contains quantized layers. See `metadata.yaml` for complete configuration details.\n\n"
        readme += "## Next Steps\n\n"
        readme += "- Validate model performance on your dataset\n"
        readme += "- Run inference tests\n"
        readme += "- Proceed to next quantization stage (if applicable)\n"
        
        return readme
    
    def list_stages(self):
        """List all saved stages"""
        stages = []
        for stage_dir in sorted(self.base_dir.iterdir()):
            if stage_dir.is_dir():
                metadata_path = stage_dir / "metadata.yaml"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = yaml.safe_load(f)
                    stages.append({
                        'name': stage_dir.name,
                        'path': stage_dir,
                        'metadata': metadata
                    })
        return stages
    
    def get_stage_model(self, stage_name):
        """Get path to best model for a specific stage"""
        stage_dir = self.base_dir / stage_name
        model_path = stage_dir / "best.pt"
        if model_path.exists():
            return model_path
        return None


def find_best_model(training_dir):
    """
    Find the best model weights from a training run
    
    Args:
        training_dir: Path to training output directory (e.g., runs/train/yolo11n-ttq)
    
    Returns:
        Path to best.pt file
    """
    training_path = Path(training_dir)
    
    # Check for weights/best.pt (standard Ultralytics structure)
    best_path = training_path / "weights" / "best.pt"
    if best_path.exists():
        return best_path
    
    # Check for best.pt directly in training dir
    best_path = training_path / "best.pt"
    if best_path.exists():
        return best_path
    
    raise FileNotFoundError(f"Could not find best.pt in {training_dir}")


def extract_metrics_from_results(results):
    """
    Extract key metrics from Ultralytics training results
    
    Args:
        results: Results object from model.train() or model.val()
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    try:
        # Try to extract from results object
        if hasattr(results, 'box'):
            metrics['mAP50'] = float(results.box.map50)
            metrics['mAP50-95'] = float(results.box.map)
        elif hasattr(results, 'results_dict'):
            metrics = results.results_dict
        
        # Add any additional metrics
        if hasattr(results, 'speed'):
            metrics['inference_speed_ms'] = results.speed.get('inference', 'N/A')
    except:
        pass
    
    return metrics
