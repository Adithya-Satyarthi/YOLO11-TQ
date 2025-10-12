import os
from ultralytics import YOLO
import yaml
from src.utils import set_seed
from src.evaluate import evaluate_model
from src.quantization.ternary_heuristic import quantize_layer_ternary
from tqdm import tqdm
from copy import deepcopy

# Setting seed value
SEED = 42
set_seed(SEED)

with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_version = config['model']['version']
model_path = config['model']['model_path']

yolo = YOLO(model_path)
model = yolo.model

print(f'Running Model Version: YOLO11{model_version}\n')

val_results = yolo.val(data='coco8.yaml', batch=16, imgsz=640, save=False, plots=False)
baseline_map = val_results.box.map
print(f"Baseline mAP50-95: {baseline_map}\n")


validator = yolo.validator
val_loader = validator.dataloader


layers_to_test = [name for name, module in model.named_modules() 
                  if len(list(module.children())) == 0 and hasattr(module, 'weight')]

print(f"Testing {len(layers_to_test)} layers...\n")

results = {}
for layer_name in tqdm(layers_to_test):
    # Clone model
    model_copy = deepcopy(model)
    
    # Apply ternary quantization to this layer
    quantize_layer_ternary(model_copy, layer_name, threshold_ratio=0.7)
    
    metric = evaluate_model(model_copy, val_loader)
    
    results[layer_name] = metric
    sensitivity = baseline_map - metric
    print(f"Layer: {layer_name:50s} | mAP: {metric:.4f} | Sensitivity: {sensitivity:.4f}")

# Sort and display most sensitive layers
print("\n" + "="*80)
print("Most Sensitive Layers (sorted by sensitivity):")
print("="*80)
sorted_results = sorted(results.items(), key=lambda x: baseline_map - x[1], reverse=True)
for layer_name, metric in sorted_results[:10]:
    sensitivity = baseline_map - metric
    print(f"{layer_name:50s} | Sensitivity: {sensitivity:.4f}")


