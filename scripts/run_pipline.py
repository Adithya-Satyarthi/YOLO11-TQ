#!/usr/bin/env python3
"""
Run complete quantization pipeline (all stages sequentially)
"""

import argparse
import sys
from pathlib import Path
import yaml
import subprocess

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_stage(stage_config_path, dry_run=False):
    """Run a single training stage"""
    print("\n" + "="*70)
    print(f"Running stage: {stage_config_path}")
    print("="*70)
    
    if dry_run:
        print("DRY RUN: Would execute:")
        print(f"  python train.py --config {stage_config_path}")
        return True
    
    # Run training
    cmd = ["python", "train.py", "--config", stage_config_path]
    result = subprocess.run(cmd)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run full quantization pipeline')
    parser.add_argument('--config', type=str, default='configs/pipeline_all_stages.yaml',
                       help='Pipeline configuration file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print stages without running')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("YOLO11 Quantization Pipeline")
    print("="*70)
    
    # Load pipeline config
    pipeline_config = load_config(args.config)
    
    print(f"\nPipeline: {pipeline_config['pipeline']['name']}")
    print(f"Description: {pipeline_config['pipeline']['description']}")
    print(f"\nStages to run:")
    
    stages = []
    for i, stage in enumerate(pipeline_config['pipeline']['stages'], 1):
        if stage.get('enabled', True):
            stages.append(stage['config'])
            print(f"  {i}. {stage['config']}")
    
    if args.dry_run:
        print("\nDRY RUN MODE - No training will be performed")
    
    # Run each stage
    for i, stage_config in enumerate(stages, 1):
        print(f"\n{'='*70}")
        print(f"STAGE {i}/{len(stages)}")
        print(f"{'='*70}")
        
        success = run_stage(stage_config, dry_run=args.dry_run)
        
        if not success and pipeline_config['pipeline'].get('stop_on_error', True):
            print(f"\n✗ Stage {i} failed. Stopping pipeline.")
            sys.exit(1)
    
    print("\n" + "="*70)
    print("Pipeline completed successfully!")
    print("="*70)
    print("\nSaved models:")
    print("  - saved_models/stage1_ttq_backbone_neck/best.pt")
    print("  - saved_models/stage2_bitnet_c2psa/best.pt")
    print("="*70)


if __name__ == '__main__':
    main()
