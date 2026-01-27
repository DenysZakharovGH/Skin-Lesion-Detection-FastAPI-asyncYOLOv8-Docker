import os

import torch
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms, models
from app.core.app_config import settings

from concurrent.futures import ThreadPoolExecutor
import asyncio

from app.core.resnet_custom import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current CPU count: {os.cpu_count()} / Cuda is available: {torch.cuda.device_count()}", )


model_yolo = YOLO(settings.cnn.model_path_yolo)
model_resnet = ResNet(settings.cnn.model_path_resnet)


for model in [model_resnet, model_yolo]:
    model.to(device)
    model.eval()


executor = ThreadPoolExecutor(max_workers=os.cpu_count())

async def predict_async(image):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, model_inference_sync, image)
    return result


def model_inference_sync(img):
    # synchronous YOLO inference
    return model.predict(img, conf=settings.cnn.conf,  iou=settings.cnn.iou, verbose=False)

# ---------- Async helper for ResNet ----------
async def classify_crop_async(crop_img: Image.Image):
    """
    Run ResNet classification in a separate thread to not block FastAPI event loop.
    """
    return await asyncio.to_thread(_classify_crop, crop_img)

def _classify_crop(crop_img: Image.Image):
    """Synchronous classification helper."""
    img_t = model_resnet.resnet_preprocess(crop_img).unsqueeze(0)
    return model_resnet.predict(img_t)