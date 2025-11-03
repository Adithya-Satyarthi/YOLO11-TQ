#!/bin/bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
python train.py --config configs/stage1_progressive.yaml --stage 1
python train.py --config configs/stage1_progressive.yaml --stage 2
python train.py --config configs/stage1_progressive.yaml --stage 3
python train_c2psa.py --config configs/stage2_c2psa.yaml --model ttq_checkpoints/yolo11n/stage1_progressive_final/best.pt
