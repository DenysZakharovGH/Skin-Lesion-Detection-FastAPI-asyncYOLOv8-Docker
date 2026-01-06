import os
from pathlib import Path

dir_path = os.getcwd()

PROJECT_ROOT = Path(__file__).resolve().parents[0]

YOLO_settings_file_path = f'{dir_path}/train/dataset/custom_data.yaml'
dataset_dir = f"{dir_path}/dataset"
dataset_image_dir = dataset_dir + "/images/"
dataset_labels_dir = dataset_dir + "/labels/"
image_format = ".jpg"
label_format = ".txt"

COLORS = {
    "AKIEC": (255, 0, 0),
    "BCC": (0, 255, 0),
    "BKL": (0, 0, 255),
    "DF": (255, 255, 0),
    "MEL": (255, 0, 255),  # melanoma → magenta
    "NV": (0, 255, 255),
    "VASC": (128, 128, 255),
}