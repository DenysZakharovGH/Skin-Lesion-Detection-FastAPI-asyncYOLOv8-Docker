import os

from ultralytics import YOLO

from app.core.app_config import settings

from concurrent.futures import ThreadPoolExecutor
import asyncio


model = YOLO(settings.cnn.model_path)



print("Current CPU count:", os.cpu_count())
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

async def predict_async(image):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, model_inference_sync, image)
    return result


def model_inference_sync(img):
    # synchronous YOLO inference
    return model.predict(img, device="cpu", conf=0.05, verbose=False)