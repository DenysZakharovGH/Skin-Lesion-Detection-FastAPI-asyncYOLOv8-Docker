from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
dataset_dir = f"{PROJECT_ROOT}/dataset"

dataset_image_dir = dataset_dir + "/images/train/"
dataset_labels_dir = dataset_dir + "/labels/train/"

val_image_dir = dataset_dir + "/images/val/"
val_labels_dir = dataset_dir + "/labels/val/"

image_format = ".jpg"
label_format = ".txt"
num_classes = 7 + 1