import os
from ultralytics import YOLO
import yaml
from src.utils import set_seed
from tqdm import tqdm

# Setting seed value
SEED = 42
set_seed(SEED)

# Setting custom path for dataset
os.environ['ULTRALYTICS_DATASET_DIR'] = os.path.join(os.getcwd(), 'data')

with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_version = config['model']['version']
model_path = config['model']['model_path']

yolo = YOLO(model_path)
model = yolo.model

print(f'Running Model Version: YOLO11{model_version}\n')


