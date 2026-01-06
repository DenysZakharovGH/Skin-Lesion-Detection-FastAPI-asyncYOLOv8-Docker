
from ultralytics import YOLO


from config import PROJECT_ROOT


model = YOLO(f"{PROJECT_ROOT}/runs/detect/train2/weights/best.pt")