import os
import torch
from ultralytics import YOLO

from config import YOLO_settings_file_path

# Load the model.
#model = YOLO('yolov8s.pt')
model = YOLO('/home/denys/PycharmProjects/Classification-of-pigmented-skin-lesions/runs/detect/train9/weights/last.pt')

# Training.
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))




# Training.
results = model.train(
    data=YOLO_settings_file_path,
    imgsz=768,
    epochs=100,
    batch=8,
    mosaic=0.0,
    amp=False,
    box=7.5,
    cls=0.5,
    save_period=5,
    name='')