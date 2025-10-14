import torch
from ultralytics import YOLO

# Load your YOLO model (change the weight path as needed)
model_path = 'yolo11s.pt'  # or 'yolov8s.pt', etc.
yolo = YOLO(model_path)
model = yolo.model

print("YOLO Model Architecture (Named Children):\n")
for name, module in model.named_modules():
    print(f"{name:30s}: {module.__class__.__name__}")

# Optionally print just the direct children for top-level overview:
print("\nTop-Level Model Children:\n")
for i, (name, module) in enumerate(model.named_children()):
    print(f"{i:2d}. {name:20s}: {module.__class__.__name__}")
