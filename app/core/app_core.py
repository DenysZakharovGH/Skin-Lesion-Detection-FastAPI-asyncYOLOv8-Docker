import asyncio
import base64

from fastapi import FastAPI, UploadFile, File, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io

from starlette.responses import HTMLResponse

from app.core.app_config import settings, FRONTEND_STORAGE, FrontendConfig
from app.model import model, predict_async, classify_crop_async
from app.utils import draw_detection
from slowapi import Limiter
from slowapi.util import get_remote_address


cnn_route = APIRouter()
limiter = Limiter(key_func=get_remote_address)
description = FrontendConfig()


@cnn_route.post("/predict")
@limiter.limit(settings.cnn.rate_limits)
async def predict_image(request: Request, file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    if img.shape[0] > settings.cnn.max_image_h_w or img.shape[1] > settings.cnn.max_image_h_w :
        raise HTTPException(status_code=400, detail=f"Image too large, max size: {settings.cnn.max_image_h_w}")

    processed_img = img.copy()
    results = await predict_async(image)

    r = results[0]

    # Prepare async classification tasks for each detected object
    tasks = []
    for bbox in results[0].boxes.xyxy:  # xyxy = [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox.cpu().numpy()
        crop = image.crop((x1, y1, x2, y2))
        tasks.append(classify_crop_async(crop))

    classifications = await asyncio.gather(*tasks)

    YOLO_ResNet_results = []

    if r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), (pred_class, prob), conf, cls_id in zip(boxes, classifications, scores, classes):

            processed_img = draw_detection(img, (x1, y1, x2, y2), "", float(prob), "default")

            YOLO_ResNet_results.append({
                "class": f"{pred_class}",
                "confidence": float(prob),
                "description": description.add_description(pred_class)
            })

    _, buffer = cv2.imencode(".jpg", processed_img)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return JSONResponse({
        "filename": file.filename,
        "detections": YOLO_ResNet_results,
        "annotated_image": img_base64
    })

cnn_route.get("/health")
@limiter.limit(settings.run.rate_limits)
def health_check(request: Request,):
    return {"msg": "OK"}

@cnn_route.get("/", response_class=HTMLResponse)
@limiter.limit(settings.run.rate_limits)
async def root(request: Request):
    with open(f"{FRONTEND_STORAGE}/index.html") as f:
        return f.read()
