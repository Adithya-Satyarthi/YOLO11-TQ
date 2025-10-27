import shutil
from pathlib import Path
import yaml


class ModelStageManager:
    """Manages saving and loading of model training stages"""
    
    def __init__(self, base_dir='saved_models'):
        self.stages_dir = Path(base_dir)  # ADD THIS LINE
        self.stages_dir.mkdir(parents=True, exist_ok=True)
    
    def save_stage(self, stage_name, model_path, config, metrics, description):
        """
        Save a training stage by COPYING the checkpoint (preserves TTQ layers).
        DO NOT load and re-save as this strips custom layers.
        """
        stage_dir = self.stages_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # JUST COPY the checkpoint file - don't load/re-save
        dest_path = stage_dir / "best.pt"
        shutil.copy2(model_path, dest_path)  # copy2 preserves metadata
        
        # Save metadata
        metadata = {
            'stage': stage_name,
            'description': description,
            'config': config,
            'metrics': metrics,
            'model_path': str(dest_path)
        }
        
        with open(stage_dir / 'metadata.yaml', 'w') as f:
            yaml.dump(metadata, f)
        
        # Save README
        readme_content = f"""# {stage_name}

## Description
{description}

## Metrics
- Precision: {metrics.get('precision', 0):.4f}
- Recall: {metrics.get('recall', 0):.4f}
- mAP50: {metrics.get('mAP50', 0):.4f}
- mAP50-95: {metrics.get('mAP50-95', 0):.4f}

## Files
- `best.pt`: Best model checkpoint (TTQ layers preserved)
- `metadata.yaml`: Stage configuration and metrics
"""
        
        with open(stage_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        print(f"\n======================================================================")
        print(f"✓ Saved to: {stage_dir}")
        print(f"✓ Saved metadata to: {stage_dir / 'metadata.yaml'}")
        print(f"✓ Saved README to: {stage_dir / 'README.md'}")
        print(f"\nStage '{stage_name}' saved successfully!")
        print(f"Location: {stage_dir.absolute()}")
        print(f"======================================================================")
        
        return stage_dir
