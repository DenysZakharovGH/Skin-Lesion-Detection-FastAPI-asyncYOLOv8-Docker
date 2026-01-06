import os
import time
from glob import glob
import cv2
from pathlib import Path
from ultralytics import YOLO

from app.utils import draw_detection
from config import image_format, PROJECT_ROOT

model = YOLO(f"{PROJECT_ROOT}/runs/detect/train9/weights/best.pt")

dataset_image_dir = f"{PROJECT_ROOT}" + "/train/dataset/images"

print(dataset_image_dir + "/val/*" + image_format)

for image_path in glob(dataset_image_dir + "/val/*" + image_format):
    print(image_path)

    start_time = time.time()

    results = model.predict(
        source=image_path,
        device="cpu",
        conf=0.25,
        save=False
    )

    print(start_time - time.time())

    # Read image with OpenCV
    img = cv2.imread(image_path)
    processed_img = img.copy()

    # Get first result
    r = results[0]

    if r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
            label = f"{model.names[cls]} {score:.2f}"
            processed_img = draw_detection(img, (x1, y1, x2, y2), label, score, model.names[cls])

    # Show image
    cv2.imshow("YOLO Detection", processed_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()


