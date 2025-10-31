#!/bin/bash
python train.py --config configs/stage1_progressive.yaml --stage 1
python train.py --config configs/stage1_progressive.yaml --stage 2
python train.py --config configs/stage1_progressive.yaml --stage 3
